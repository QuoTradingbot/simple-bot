"""
QuoTrading Cloud Signal Engine
Centralized VWAP + RSI signal generation for all customers

This runs 24/7 on Azure, calculates signals, and broadcasts them via API.
Customers connect to fetch signals and execute locally on their TopStep accounts.
"""

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime, time as datetime_time, timedelta
from typing import Dict, Optional, List
from collections import deque
import pytz
import logging
import stripe
import secrets
import hashlib
import json
import time as time_module
import os

# Import database and Redis managers
from database import (
    DatabaseManager, db_manager as global_db_manager, get_db,
    User, APILog, TradeHistory, RLExperience,
    get_user_by_license_key, get_user_by_account_id,
    create_user, log_api_call, update_user_activity
)
from redis_manager import RedisManager, get_redis
from sqlalchemy.orm import Session

# Initialize FastAPI
app = FastAPI(
    title="QuoTrading Signal Engine",
    description="Real-time VWAP mean reversion signals with user management",
    version="2.1"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        return super().default(obj)

# Global instances (initialized on startup)
db_manager: Optional[DatabaseManager] = None
redis_manager: Optional[RedisManager] = None

# Global RL experience pools (shared across all users)
signal_experiences = []  # Signal RL learning pool
exit_experiences = []    # Exit RL learning pool (deprecated, now in PostgreSQL)

# ============================================================================
# RATE LIMITING (Redis-backed with in-memory fallback)
# ============================================================================

# Rate limit settings
RATE_LIMIT_REQUESTS = 100  # requests
RATE_LIMIT_WINDOW = 60  # seconds (1 minute)
RATE_LIMIT_BLOCK_TIME = 300  # 5 minutes block after exceeding

def check_rate_limit(request: Request) -> dict:
    """
    Check if request should be rate limited using Redis or fallback
    
    Returns:
        dict with 'allowed' and 'retry_after' keys
    """
    client_ip = request.client.host
    
    # Use Redis manager if available
    if redis_manager:
        return redis_manager.check_rate_limit(
            identifier=client_ip,
            max_requests=RATE_LIMIT_REQUESTS,
            window_seconds=RATE_LIMIT_WINDOW,
            block_duration=RATE_LIMIT_BLOCK_TIME
        )
    
    # Should never happen since redis_manager has in-memory fallback
    # But just in case, allow the request
    return {'allowed': True, 'retry_after': 0}

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to all requests and log API activity"""
    
    # Skip rate limit and logging for health check
    if request.url.path == "/health" or request.url.path == "/":
        return await call_next(request)
    
    # Track start time for response time measurement
    start_time = time_module.time()
    
    # Check rate limit
    limit_result = check_rate_limit(request)
    
    if not limit_result['allowed']:
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Try again in {limit_result['retry_after']} seconds.",
                "retry_after": limit_result['retry_after']
            }
        )
    
    # Process request
    response = await call_next(request)
    
    # **NEW: Log API call to database for admin dashboard tracking**
    try:
        # Calculate response time
        response_time_ms = int((time_module.time() - start_time) * 1000)
        
        # Get user info from request (if license key provided)
        license_key = request.query_params.get('license_key')
        
        if license_key:
            # Get database session using dependency injection
            db = next(get_db())
            try:
                user = db.query(User).filter(User.license_key == license_key).first()
                if user:
                    # Update last_active timestamp
                    user.last_active = datetime.utcnow()
                    db.commit()
                    
                    # Log API call
                    api_log = APILog(
                        user_id=user.id,
                        endpoint=str(request.url.path),
                        method=request.method,
                        status_code=response.status_code,
                        response_time_ms=response_time_ms,
                        ip_address=request.client.host
                    )
                    db.add(api_log)
                    db.commit()
            except Exception as log_error:
                db.rollback()
                logger.debug(f"Failed to log API call: {log_error}")
            finally:
                db.close()
    except Exception as e:
        # Don't fail the request if logging fails
        logger.debug(f"Error in API logging middleware: {e}")
        logger.debug(f"Error in API logging middleware: {e}")
    
    return response

# ============================================================================
# STRIPE CONFIGURATION
# ============================================================================

# Stripe API keys - Load from environment variables
stripe.api_key = os.getenv("STRIPE_API_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", None)

# In-memory license storage (for beta - will move to database later)
active_licenses = {}  # {license_key: {email, expires_at, stripe_customer_id, stripe_subscription_id}}

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "QuoTrading Cloud Signal Engine",
        "status": "running",
        "version": "2.0",
        "endpoints": {
            "ml": "/api/ml/get_confidence, /api/ml/save_trade, /api/ml/stats",
            "license": "/api/license/validate, /api/license/activate",
            "calendar": "/api/calendar/today, /api/calendar/events",
            "time": "/api/time, /api/time/simple",
            "admin": "/api/admin/kill_switch, /api/admin/refresh_calendar"
        }
    }


@app.get("/health")
async def health():
    """Health check for monitoring"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/rate-limit/status")
async def rate_limit_status(request: Request):
    """Check current rate limit status for this IP"""
    client_ip = request.client.host
    
    if redis_manager:
        status = redis_manager.get_rate_limit_status(
            identifier=client_ip,
            max_requests=RATE_LIMIT_REQUESTS,
            window_seconds=RATE_LIMIT_WINDOW
        )
        return status
    
    # Fallback (should never happen)
    return {
        "ip": client_ip,
        "requests_used": 0,
        "requests_remaining": RATE_LIMIT_REQUESTS,
        "limit": RATE_LIMIT_REQUESTS,
        "window_seconds": RATE_LIMIT_WINDOW,
        "blocked": False
    }

# ============================================================================
# USER MANAGEMENT & ADMIN ENDPOINTS
# ============================================================================

def verify_admin_license(license_key: str, db: Session) -> User:
    """Verify admin license and return user"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    user = get_user_by_license_key(db, license_key)
    if not user:
        raise HTTPException(status_code=403, detail="Invalid license key")
    
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privileges required")
    
    if not user.is_license_valid:
        raise HTTPException(status_code=403, detail="License expired or inactive")
    
    return user

@app.get("/api/admin/users")
async def get_all_users(
    license_key: str,
    db: Session = Depends(get_db)
):
    """Get all users with activity counts (admin only)"""
    verify_admin_license(license_key, db)
    
    from sqlalchemy import func
    
    users = db.query(User).all()
    users_with_stats = []
    
    for user in users:
        user_dict = user.to_dict()
        
        # Get API call count
        api_call_count = db.query(func.count(APILog.id)).filter(
            APILog.user_id == user.id
        ).scalar() or 0
        
        # Get trade count
        trade_count = db.query(func.count(TradeHistory.id)).filter(
            TradeHistory.user_id == user.id
        ).scalar() or 0
        
        user_dict['api_call_count'] = api_call_count
        user_dict['trade_count'] = trade_count
        
        users_with_stats.append(user_dict)
    
    return {
        "total_users": len(users),
        "users": users_with_stats
    }

@app.get("/api/admin/user/{account_id}")
async def get_user_details(
    account_id: str,
    license_key: str,
    db: Session = Depends(get_db)
):
    """Get specific user details (admin only)"""
    verify_admin_license(license_key, db)
    
    user = get_user_by_account_id(db, account_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get recent API calls
    recent_calls = db.query(APILog).filter(
        APILog.user_id == user.id
    ).order_by(APILog.timestamp.desc()).limit(20).all()
    
    # Get trade stats
    from sqlalchemy import func
    trade_stats = db.query(
        func.count(TradeHistory.id).label('total_trades'),
        func.sum(TradeHistory.pnl).label('total_pnl'),
        func.avg(TradeHistory.pnl).label('avg_pnl')
    ).filter(TradeHistory.user_id == user.id).first()
    
    return {
        "user": user.to_dict(),
        "recent_api_calls": len(recent_calls),
        "trade_stats": {
            "total_trades": trade_stats.total_trades or 0,
            "total_pnl": float(trade_stats.total_pnl or 0),
            "avg_pnl": float(trade_stats.avg_pnl or 0)
        }
    }

@app.post("/api/admin/add-user")
async def add_user(
    license_key: str,
    account_id: str,
    email: Optional[str] = None,
    license_type: str = "BETA",
    license_duration_days: Optional[int] = None,
    notes: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Create a new user (admin only)"""
    verify_admin_license(license_key, db)
    
    # Check if user already exists
    existing = get_user_by_account_id(db, account_id)
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Create user
    new_user = create_user(
        db_session=db,
        account_id=account_id,
        email=email,
        license_type=license_type,
        license_duration_days=license_duration_days,
        is_admin=False,
        notes=notes
    )
    
    return {
        "message": "User created successfully",
        "user": new_user.to_dict()
    }

@app.put("/api/admin/extend-license/{account_id}")
async def extend_license(
    account_id: str,
    license_key: str,
    additional_days: int,
    db: Session = Depends(get_db)
):
    """Extend user license (admin only)"""
    verify_admin_license(license_key, db)
    
    user = get_user_by_account_id(db, account_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Extend license
    if user.license_expiration:
        user.license_expiration = user.license_expiration + timedelta(days=additional_days)
    else:
        user.license_expiration = datetime.utcnow() + timedelta(days=additional_days)
    
    db.commit()
    db.refresh(user)
    
    return {
        "message": f"License extended by {additional_days} days",
        "user": user.to_dict()
    }

@app.put("/api/admin/suspend-user/{account_id}")
async def suspend_user(
    account_id: str,
    license_key: str,
    db: Session = Depends(get_db)
):
    """Suspend user account (admin only)"""
    verify_admin_license(license_key, db)
    
    user = get_user_by_account_id(db, account_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.license_status = "SUSPENDED"
    db.commit()
    db.refresh(user)
    
    return {
        "message": "User suspended",
        "user": user.to_dict()
    }

@app.put("/api/admin/activate-user/{account_id}")
async def activate_user(
    account_id: str,
    license_key: str,
    db: Session = Depends(get_db)
):
    """Activate suspended user (admin only)"""
    verify_admin_license(license_key, db)
    
    user = get_user_by_account_id(db, account_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.license_status = "ACTIVE"
    db.commit()
    db.refresh(user)
    
    return {
        "message": "User activated",
        "user": user.to_dict()
    }

@app.get("/api/admin/stats")
async def get_stats(
    license_key: str,
    db: Session = Depends(get_db)
):
    """Get overall system stats (admin only)"""
    verify_admin_license(license_key, db)
    
    from sqlalchemy import func
    
    # User stats
    total_users = db.query(func.count(User.id)).scalar()
    active_users = db.query(func.count(User.id)).filter(User.license_status == 'ACTIVE').scalar()
    
    # License type breakdown
    license_breakdown = db.query(
        User.license_type,
        func.count(User.id)
    ).group_by(User.license_type).all()
    
    # API call stats (last 24 hours)
    yesterday = datetime.utcnow() - timedelta(days=1)
    api_calls_24h = db.query(func.count(APILog.id)).filter(
        APILog.timestamp >= yesterday
    ).scalar()
    
    # Trade stats
    total_trades = db.query(func.count(TradeHistory.id)).scalar()
    total_pnl = db.query(func.sum(TradeHistory.pnl)).scalar()
    
    # RL Experience stats
    total_signal_experiences = db.query(func.count(RLExperience.id)).filter(
        RLExperience.experience_type == 'SIGNAL'
    ).scalar()
    total_exit_experiences = db.query(func.count(RLExperience.id)).filter(
        RLExperience.experience_type == 'EXIT'
    ).scalar()
    
    # Recent RL growth (last 24 hours)
    signal_experiences_24h = db.query(func.count(RLExperience.id)).filter(
        RLExperience.experience_type == 'SIGNAL',
        RLExperience.timestamp >= yesterday
    ).scalar()
    exit_experiences_24h = db.query(func.count(RLExperience.id)).filter(
        RLExperience.experience_type == 'EXIT',
        RLExperience.timestamp >= yesterday
    ).scalar()
    
    # Win rate from RL experiences
    total_rl = db.query(func.count(RLExperience.id)).scalar()
    winning_rl = db.query(func.count(RLExperience.id)).filter(
        RLExperience.outcome == 'WIN'
    ).scalar()
    win_rate = (winning_rl / total_rl * 100) if total_rl > 0 else 0
    
    return {
        "users": {
            "total": total_users,
            "active": active_users,
            "by_license_type": {lt: count for lt, count in license_breakdown}
        },
        "api_calls_24h": api_calls_24h,
        "trades": {
            "total": total_trades,
            "total_pnl": float(total_pnl or 0)
        },
        "rl_experiences": {
            "total_signal_experiences": total_signal_experiences or 0,
            "total_exit_experiences": total_exit_experiences or 0,
            "signal_experiences_24h": signal_experiences_24h or 0,
            "exit_experiences_24h": exit_experiences_24h or 0,
            "total_experiences": (total_signal_experiences or 0) + (total_exit_experiences or 0),
            "win_rate": round(win_rate, 1)
        }
    }

@app.get("/api/admin/online-users")
async def get_online_users(
    license_key: str,
    db: Session = Depends(get_db)
):
    """Get users who are currently online (active in last 5 minutes) - admin only"""
    verify_admin_license(license_key, db)
    
    # Consider user "online" if last_active within 5 minutes
    five_minutes_ago = datetime.utcnow() - timedelta(minutes=5)
    
    online_users = db.query(User).filter(
        User.last_active >= five_minutes_ago,
        User.license_status == 'ACTIVE'
    ).order_by(User.last_active.desc()).all()
    
    return {
        "online_count": len(online_users),
        "users": [
            {
                "account_id": user.account_id,
                "email": user.email,
                "license_type": user.license_type,
                "last_active": user.last_active.isoformat() if user.last_active else None,
                "seconds_ago": int((datetime.utcnow() - user.last_active).total_seconds()) if user.last_active else None
            }
            for user in online_users
        ]
    }

@app.get("/api/admin/recent-activity")
async def get_recent_activity(
    license_key: str,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get recent API calls and activity (admin only)"""
    verify_admin_license(license_key, db)
    
    # Recent API calls with user info
    recent_logs = db.query(APILog).join(User).order_by(
        APILog.timestamp.desc()
    ).limit(limit).all()
    
    activity = []
    for log in recent_logs:
        user = db.query(User).filter(User.id == log.user_id).first()
        activity.append({
            "timestamp": log.timestamp.isoformat(),
            "account_id": user.account_id if user else "unknown",
            "endpoint": log.endpoint,
            "method": log.method,
            "status_code": log.status_code,
            "response_time_ms": log.response_time_ms,
            "ip_address": log.ip_address
        })
    
    return {
        "activity_count": len(activity),
        "activity": activity
    }

@app.get("/api/admin/dashboard-stats")
async def get_dashboard_stats(
    license_key: str,
    db: Session = Depends(get_db)
):
    """Get comprehensive dashboard statistics (admin only)"""
    verify_admin_license(license_key, db)
    
    from sqlalchemy import func
    
    # User stats
    total_users = db.query(func.count(User.id)).scalar()
    active_users = db.query(func.count(User.id)).filter(User.license_status == 'ACTIVE').scalar()
    suspended_users = db.query(func.count(User.id)).filter(User.license_status == 'SUSPENDED').scalar()
    
    # Online users (active in last 5 min)
    five_min_ago = datetime.utcnow() - timedelta(minutes=5)
    online_now = db.query(func.count(User.id)).filter(
        User.last_active >= five_min_ago,
        User.license_status == 'ACTIVE'
    ).scalar()
    
    # License type breakdown
    license_breakdown = {}
    for license_type, count in db.query(User.license_type, func.count(User.id)).group_by(User.license_type).all():
        license_breakdown[license_type] = count
    
    # API calls
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    one_day_ago = datetime.utcnow() - timedelta(days=1)
    
    api_calls_1h = db.query(func.count(APILog.id)).filter(APILog.timestamp >= one_hour_ago).scalar()
    api_calls_24h = db.query(func.count(APILog.id)).filter(APILog.timestamp >= one_day_ago).scalar()
    
    # Trade stats
    total_trades = db.query(func.count(TradeHistory.id)).scalar()
    total_pnl = db.query(func.sum(TradeHistory.pnl)).scalar()
    trades_today = db.query(func.count(TradeHistory.id)).filter(
        func.date(TradeHistory.entry_time) == datetime.utcnow().date()
    ).scalar()
    
    # Expiring licenses (next 7 days)
    seven_days = datetime.utcnow() + timedelta(days=7)
    expiring_soon = db.query(func.count(User.id)).filter(
        User.license_expiration <= seven_days,
        User.license_expiration >= datetime.utcnow(),
        User.license_status == 'ACTIVE'
    ).scalar()
    
    # RL Experience stats
    total_signal_experiences = db.query(func.count(RLExperience.id)).filter(
        RLExperience.experience_type == 'SIGNAL'
    ).scalar()
    total_exit_experiences = db.query(func.count(RLExperience.id)).filter(
        RLExperience.experience_type == 'EXIT'
    ).scalar()
    
    # Recent RL growth (last 24 hours)
    signal_exp_24h = db.query(func.count(RLExperience.id)).filter(
        RLExperience.experience_type == 'SIGNAL',
        RLExperience.timestamp >= one_day_ago
    ).scalar()
    exit_exp_24h = db.query(func.count(RLExperience.id)).filter(
        RLExperience.experience_type == 'EXIT',
        RLExperience.timestamp >= one_day_ago
    ).scalar()
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "users": {
            "total": total_users or 0,
            "active": active_users or 0,
            "suspended": suspended_users or 0,
            "online_now": online_now or 0,
            "expiring_soon": expiring_soon or 0,
            "by_license_type": license_breakdown
        },
        "api_calls": {
            "last_hour": api_calls_1h or 0,
            "last_24h": api_calls_24h or 0
        },
        "trades": {
            "total": total_trades or 0,
            "today": trades_today or 0,
            "total_pnl": float(total_pnl or 0)
        },
        "rl_experiences": {
            "total_signal_experiences": total_signal_experiences or 0,
            "total_exit_experiences": total_exit_experiences or 0,
            "signal_experiences_24h": signal_exp_24h or 0,
            "exit_experiences_24h": exit_exp_24h or 0,
            "total_experiences": (total_signal_experiences or 0) + (total_exit_experiences or 0)
        }
    }

# ============================================================================
# ML/RL ENDPOINTS
# ============================================================================

# ============================================================================
# RL EXPERIENCE INITIALIZATION - POSTGRESQL HIVE MIND
# ============================================================================

def load_initial_experiences():
    """
    Initialize PostgreSQL database and migrate exit experiences from JSON.
    ALL USERS share signal + exit experiences from PostgreSQL!
    """
    import json
    import os
    
    try:
        # Create database tables (including new ExitExperience table)
        from database import Base, DatabaseManager
        
        # Initialize database and create tables if they don't exist
        db_manager = DatabaseManager()
        engine = db_manager.get_engine()
        Base.metadata.create_all(engine)
        logger.info("✅ Database tables initialized (including ExitExperience table)")
        
        # Import exit experiences from JSON to PostgreSQL (one-time migration)
        if os.path.exists("exit_experience.json"):
            from database import ExitExperience
            session = db_manager.get_session()
            
            try:
                # Check if table is empty
                count = session.query(ExitExperience).count()
                
                if count == 0:
                    logger.info("🔄 Migrating exit experiences from JSON to PostgreSQL...")
                    
                    # Load JSON data
                    with open("exit_experience.json", "r") as f:
                        data = json.load(f)
                        json_experiences = data.get("exit_experiences", [])
                    
                    # Import each experience
                    for exp in json_experiences:
                        exit_exp = ExitExperience(
                            regime=exp.get('regime', 'UNKNOWN'),
                            exit_params_json=json.dumps(exp.get('exit_params', {})),
                            outcome_json=json.dumps(exp.get('outcome', {})),
                            situation_json=json.dumps(exp.get('situation', {})),
                            market_state_json=json.dumps(exp.get('market_state', {})),
                            partial_exits_json=json.dumps(exp.get('partial_exits', [])),
                            quality_score=exp.get('quality_score')
                        )
                        session.add(exit_exp)
                    
                    session.commit()
                    logger.info(f"✅ Migrated {len(json_experiences):,} exit experiences to PostgreSQL")
                else:
                    logger.info(f"✅ PostgreSQL already has {count:,} exit experiences - skipping migration")
                
            except Exception as e:
                session.rollback()
                logger.error(f"❌ Migration error: {e}")
            finally:
                session.close()
        else:
            logger.warning("⚠️  exit_experience.json not found - starting fresh")
        
        # Count experiences in database
        session = db_manager.get_session()
        try:
            from database import RLExperience, ExitExperience
            signal_count = session.query(RLExperience).filter_by(experience_type='SIGNAL').count()
            exit_count = session.query(ExitExperience).count()
            logger.info(f"🧠 HIVE MIND INITIALIZED: {signal_count:,} signals + {exit_count:,} exits IN POSTGRESQL")
            logger.info(f"   🌍 ALL USERS share and contribute to this collective learning pool!")
        finally:
            session.close()
        
    except Exception as e:
        logger.error(f"❌ Could not initialize database: {e}")
        logger.info("Starting with empty experience pool")


def load_database_experiences(db: Session):
    """
    Load RL experiences from database and merge with JSON baseline
    
    This combines Kevin's baseline (6,880 signals + 2,961 exits from JSON)
    with all new experiences saved to database by live users.
    """
    global signal_experiences
    
    try:
        # Query all signal experiences from database
        db_signal_experiences = db.query(RLExperience).filter(
            RLExperience.experience_type == 'SIGNAL'
        ).all()
        
        # Convert to dictionaries for pattern matching
        for exp in db_signal_experiences:
            exp_dict = {
                'user_id': exp.account_id,
                'symbol': exp.symbol,
                'signal': exp.signal_type,
                'signal_type': exp.signal_type,
                'outcome': exp.outcome,
                'pnl': exp.pnl,
                'confidence': exp.confidence_score,
                'quality_score': exp.quality_score,
                # Handle new columns gracefully (may not exist in old schema)
                'rsi': getattr(exp, 'rsi', None),
                'vwap_distance': getattr(exp, 'vwap_distance', None),
                'vix': getattr(exp, 'vix', None),
                'day_of_week': getattr(exp, 'day_of_week', None),
                'hour_of_day': getattr(exp, 'hour_of_day', None),
                'timestamp': exp.timestamp.isoformat(),
                'experience_id': f"db_{exp.id}"
            }
            signal_experiences.append(exp_dict)
        
        logger.info(f"📊 Loaded {len(db_signal_experiences)} new RL experiences from database")
        logger.info(f"🧠 Total signal experiences: {len(signal_experiences):,} (baseline + live trades)")
        
    except Exception as e:
        logger.error(f"❌ Could not load database experiences: {e}")
        logger.info("📊 Continuing with baseline signal experiences only")

# Load experiences at startup
load_initial_experiences()

def save_experiences():
    """Save updated experiences back to disk (persist hive mind growth)"""
    import json
    try:
        with open("signal_experience.json", "w") as f:
            json.dump(signal_experiences, f)
        with open("exit_experience.json", "w") as f:
            json.dump(exit_experiences, f)
        logger.info(f"💾 Saved hive mind: {len(signal_experiences):,} signals + {len(exit_experiences):,} exits")
    except Exception as e:
        logger.error(f"Failed to save experiences: {e}")

def get_all_experiences() -> List:
    """Get all RL experiences (shared learning - same strategy for everyone)"""
    return signal_experiences

@app.post("/api/ml/get_confidence")
async def get_ml_confidence(request: Dict):
    """
    ADVANCED ML confidence with pattern matching and context-aware learning
    
    Request: {
        user_id: str,  # REQUIRED - for data isolation
        symbol: str,
        vwap: float,
        vwap_std_dev: float,
        rsi: float,
        price: float,
        volume: int,
        signal: str,  # 'LONG' or 'SHORT'
        vix: float  # Optional - current VIX level (default 15.0)
    }
    
    Returns: {
        ml_confidence: float,  # 0.0 to 0.95
        win_rate: float,  # Historical win rate for similar setups
        sample_size: int,  # Number of similar past experiences
        avg_pnl: float,  # Average P&L from similar trades
        reason: str,  # Human-readable explanation
        should_take: bool,  # True if confidence >= 65%
        action: str,  # 'LONG', 'SHORT', or 'NONE'
        model_version: str
    }
    """
    try:
        # CRITICAL: Require user_id for data isolation
        user_id = request.get('user_id', '')
        if not user_id:
            return {
                "error": "user_id required",
                "ml_confidence": 0.0,
                "action": "NONE"
            }
        
        symbol = request.get('symbol', 'ES')
        vwap = request.get('vwap', 0.0)
        rsi = request.get('rsi', 50.0)
        price = request.get('price', 0.0)
        signal = request.get('signal', 'NONE')
        vix = request.get('vix', 15.0)
        
        # Get ALL signal experiences (everyone learns from same strategy)
        all_trades = signal_experiences
        
        # ADVANCED ML confidence with pattern matching
        result = calculate_signal_confidence(
            all_experiences=all_trades,
            vwap_distance=abs(price - vwap) / vwap if vwap > 0 else 0,
            rsi=rsi,
            signal=signal,
            current_time=datetime.now(pytz.timezone('US/Eastern')),
            current_vix=vix,
            similarity_threshold=0.6  # 60% similarity required
        )
        
        logger.info(f"🧠 ML Confidence: {symbol} {signal} @ {price}, RSI={rsi:.1f} → {result['reason']}")
        
        return {
            "ml_confidence": result['confidence'],
            "win_rate": result['win_rate'],
            "sample_size": result['sample_size'],
            "avg_pnl": result['avg_pnl'],
            "reason": result['reason'],
            "should_take": result['should_take'],
            "action": signal if result['should_take'] else "NONE",
            "model_version": "v5.0-advanced-pattern-matching",
            "total_experience_count": len(all_trades),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error calculating ML confidence: {e}")
        # Return neutral confidence on error
        return {
            "ml_confidence": 0.5,
            "action": "NONE",
            "model_version": "v1.0-simple",
            "error": str(e)
        }


@app.post("/api/ml/should_take_signal")
async def should_take_signal(request: Dict):
    """
    REAL-TIME TRADE DECISION ENGINE
    
    Bot calls this before entering a trade. Returns intelligent recommendation
    based on pattern matching across ALL 6,880+ signal experiences.
    
    Request: {
        user_id: str,
        symbol: str,
        signal: str,  # 'LONG' or 'SHORT'
        entry_price: float,
        vwap: float,
        rsi: float,
        vix: float,  # Optional, default 15.0
        volume_ratio: float,  # Optional, default 1.0
        recent_pnl: float,  # Optional, default 0.0
        streak: int  # Optional, default 0
    }
    
    Returns: {
        take_trade: bool,  # True = TAKE IT, False = SKIP IT
        confidence: float,  # 0.0 to 0.95
        win_rate: float,  # Historical win rate for similar setups
        sample_size: int,  # Number of similar past trades analyzed
        avg_pnl: float,  # Expected P&L based on history
        reason: str,  # Human-readable explanation
        risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    }
    """
    try:
        user_id = request.get('user_id', '')
        if not user_id:
            return {
                "take_trade": False,
                "confidence": 0.0,
                "reason": "user_id required",
                "error": "Missing user_id"
            }
        
        symbol = request.get('symbol', 'ES')
        signal = request.get('signal', 'NONE')
        entry_price = request.get('entry_price', 0.0)
        vwap = request.get('vwap', 0.0)
        rsi = request.get('rsi', 50.0)
        vix = request.get('vix', 15.0)
        volume_ratio = request.get('volume_ratio', 1.0)
        recent_pnl = request.get('recent_pnl', 0.0)
        streak = request.get('streak', 0)
        
        # Calculate VWAP distance
        vwap_distance = abs(entry_price - vwap) / vwap if vwap > 0 else 0
        
        # Get advanced ML confidence with pattern matching
        result = calculate_signal_confidence(
            all_experiences=signal_experiences,
            vwap_distance=vwap_distance,
            rsi=rsi,
            signal=signal,
            current_time=datetime.now(pytz.timezone('US/Eastern')),
            current_vix=vix,
            similarity_threshold=0.6,
            volume_ratio=volume_ratio,
            recent_pnl=recent_pnl,
            streak=streak
        )
        
        # Determine risk level
        if result['confidence'] >= 0.80:
            risk_level = 'LOW'
        elif result['confidence'] >= 0.65:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'
        
        # Log decision
        decision = "✅ TAKE TRADE" if result['should_take'] else "⚠️ SKIP TRADE"
        logger.info(f"🎯 {decision}: {symbol} {signal} @ {entry_price}, {result['reason']}, Risk: {risk_level}")
        
        return {
            "take_trade": result['should_take'],
            "confidence": result['confidence'],
            "win_rate": result['win_rate'],
            "sample_size": result['sample_size'],
            "avg_pnl": result['avg_pnl'],
            "reason": result['reason'],
            "risk_level": risk_level,
            "model_version": "v5.0-advanced-pattern-matching",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in should_take_signal: {e}")
        return {
            "take_trade": False,
            "confidence": 0.0,
            "reason": f"Error: {str(e)}",
            "error": str(e)
        }


def calculate_similarity_score(exp: Dict, current_rsi: float, current_vwap_dist: float, current_time: datetime_time, current_vix: float, 
                               current_volume_ratio: float = 1.0, current_hour: int = 12, current_day_of_week: int = 0,
                               current_recent_pnl: float = 0.0, current_streak: int = 0) -> float:
    """
    Calculate how similar a past experience is to current market conditions
    
    Returns score from 0.0 (completely different) to 1.0 (identical conditions)
    """
    score = 0.0
    weights = []
    
    # Handle both flat and nested (state-based) experience structures
    state = exp.get('state', exp)
    
    # RSI similarity (±5 range = very similar) - 25% weight
    exp_rsi = state.get('rsi') or 50.0  # Default to neutral RSI if None
    rsi_diff = abs(exp_rsi - current_rsi)
    if rsi_diff <= 5:
        rsi_similarity = 1.0 - (rsi_diff / 5)
        score += rsi_similarity * 0.25
        weights.append(0.25)
    elif rsi_diff <= 10:
        rsi_similarity = 1.0 - ((rsi_diff - 5) / 5) * 0.5
        score += rsi_similarity * 0.25
        weights.append(0.25)
    
    # VWAP distance similarity (±0.002 range) - 20% weight
    exp_vwap_dist = state.get('vwap_distance', 0.0) or 0.0  # Handle None values
    vwap_diff = abs(exp_vwap_dist - current_vwap_dist)
    if vwap_diff <= 0.002:
        vwap_similarity = 1.0 - (vwap_diff / 0.002)
        score += vwap_similarity * 0.20
        weights.append(0.20)
    
    # Time-of-day similarity (±30 min = similar) - 15% weight
    if 'timestamp' in exp:
        try:
            exp_time = datetime.fromisoformat(exp['timestamp']).time() if isinstance(exp['timestamp'], str) else exp['timestamp'].time()
            exp_minutes = exp_time.hour * 60 + exp_time.minute
            current_minutes = current_time.hour * 60 + current_time.minute
            time_diff = abs(exp_minutes - current_minutes)
            if time_diff <= 30:
                time_similarity = 1.0 - (time_diff / 30)
                score += time_similarity * 0.15
                weights.append(0.15)
            elif time_diff <= 60:
                time_similarity = 1.0 - ((time_diff - 30) / 30) * 0.5
                score += time_similarity * 0.15
                weights.append(0.15)
        except:
            pass
    
    # VIX similarity (±5 range) - 10% weight
    exp_vix = state.get('vix', 15)
    vix_diff = abs(exp_vix - current_vix)
    if vix_diff <= 5:
        vix_similarity = 1.0 - (vix_diff / 5)
        score += vix_similarity * 0.10
        weights.append(0.10)
    
    # Volume ratio similarity (±0.5 range) - 15% weight
    exp_volume_ratio = state.get('volume_ratio', 1.0)
    volume_diff = abs(exp_volume_ratio - current_volume_ratio)
    if volume_diff <= 0.5:
        volume_similarity = 1.0 - (volume_diff / 0.5)
        score += volume_similarity * 0.15
        weights.append(0.15)
    
    # Hour similarity (±2 hours = similar) - 5% weight
    exp_hour = state.get('hour', 12)
    hour_diff = abs(exp_hour - current_hour)
    if hour_diff <= 2:
        hour_similarity = 1.0 - (hour_diff / 2)
        score += hour_similarity * 0.05
        weights.append(0.05)
    
    # Day of week similarity (same day = 1.0, adjacent = 0.5) - 5% weight
    exp_day = state.get('day_of_week', 0)
    day_diff = abs(exp_day - current_day_of_week)
    if day_diff == 0:
        score += 1.0 * 0.05
        weights.append(0.05)
    elif day_diff == 1 or day_diff == 6:  # Adjacent days (or Monday-Sunday wrap)
        score += 0.5 * 0.05
        weights.append(0.05)
    
    # Streak similarity (same direction, ±2 range) - 5% weight
    exp_streak = state.get('streak', 0)
    # Both winning streaks or both losing streaks = similar
    if (current_streak > 0 and exp_streak > 0) or (current_streak < 0 and exp_streak < 0):
        streak_diff = abs(abs(exp_streak) - abs(current_streak))
        if streak_diff <= 2:
            streak_similarity = 1.0 - (streak_diff / 2)
            score += streak_similarity * 0.05
            weights.append(0.05)
    elif current_streak == 0 and exp_streak == 0:
        score += 1.0 * 0.05
        weights.append(0.05)
    
    # Normalize score by sum of applied weights
    total_weight = sum(weights)
    if total_weight > 0:
        return score / total_weight
    return 0.0


def filter_experiences_by_context(experiences: List, signal_type: str, current_day: int, current_vix: float, min_similarity: float = 0.6) -> List:
    """
    Filter experiences by trading context
    
    Args:
        experiences: All past experiences
        signal_type: 'LONG' or 'SHORT'
        current_day: Day of week (0=Monday, 4=Friday)
        current_vix: Current VIX level
        min_similarity: Minimum similarity score to include
    
    Returns:
        Filtered list of relevant experiences
    """
    filtered = []
    total_checked = 0
    signal_mismatch = 0
    vix_mismatch = 0
    
    for exp in experiences:
        total_checked += 1
        # Handle both flat and nested (state-based) experience structures
        # Nested: exp['state']['side'], Flat: exp['side']
        state = exp.get('state', exp)  # Use 'state' dict if exists, otherwise use exp itself
        
        # Must match signal type (check all possible field names: signal_type, signal, or side)
        exp_signal = state.get('signal_type') or state.get('signal') or state.get('side', '').upper()
        if not exp_signal or exp_signal.upper() != signal_type.upper():
            signal_mismatch += 1
            continue
        
        # Filter by VIX (only consider trades in similar volatility environments)
        exp_vix = state.get('vix', 15)
        if abs(exp_vix - current_vix) > 10:  # Skip if VIX difference > 10
            vix_mismatch += 1
            continue
        
        # Filter by day of week (optional - can enable for day-specific patterns)
        # For now, include all days but could add: if exp.get('day_of_week') == current_day
        
        filtered.append(exp)
    
    # Debug logging
    if total_checked > 0:
        logger.info(f"📊 Filter Stats: {total_checked} total, {signal_mismatch} signal mismatch, {vix_mismatch} VIX mismatch, {len(filtered)} matched")
    
    return filtered


def calculate_advanced_confidence(similar_experiences: List, recency_hours: int = 168) -> Dict:
    """
    Calculate smart confidence based on similar past experiences
    
    Args:
        similar_experiences: Pre-filtered experiences matching current conditions
        recency_hours: How many hours back to weight more heavily (default 1 week)
    
    Returns:
        Dict with confidence, win_rate, sample_size, avg_pnl
    """
    if len(similar_experiences) == 0:
        return {
            'confidence': 0.5,
            'win_rate': 0.5,
            'sample_size': 0,
            'avg_pnl': 0.0,
            'reason': 'No similar past experiences found'
        }
    
    # Separate wins and losses
    # Handle both flat and nested structures: exp['reward'] (nested) or exp['pnl'] (flat)
    wins = []
    losses = []
    for exp in similar_experiences:
        # Get PnL from either 'reward' (nested RL format) or 'pnl' (flat format)
        pnl = exp.get('reward', exp.get('pnl', 0))
        outcome = exp.get('outcome', '')
        
        if outcome == 'WIN' or pnl > 0:
            wins.append(exp)
        else:
            losses.append(exp)
    
    # Calculate basic win rate
    win_rate = len(wins) / len(similar_experiences)
    
    # Calculate weighted win rate (recent trades matter more)
    now = datetime.utcnow()
    weighted_wins = 0
    weighted_total = 0
    
    for exp in similar_experiences:
        # Calculate recency weight (1.0 for very recent, 0.5 for old)
        if 'timestamp' in exp:
            try:
                exp_time = datetime.fromisoformat(exp['timestamp']) if isinstance(exp['timestamp'], str) else exp['timestamp']
                hours_ago = (now - exp_time).total_seconds() / 3600
                recency_weight = max(0.5, 1.0 - (hours_ago / recency_hours) * 0.5)
            except:
                recency_weight = 0.7  # Default weight if timestamp parsing fails
        else:
            recency_weight = 0.7
        
        # Apply quality score weight (better trades = more weight)
        quality_weight = exp.get('quality_score', 0.5)
        
        # Combined weight
        total_weight = recency_weight * quality_weight
        
        weighted_total += total_weight
        # Check both 'reward' (nested RL) and 'pnl' (flat) formats
        pnl = exp.get('reward', exp.get('pnl', 0))
        if exp.get('outcome') == 'WIN' or pnl > 0:
            weighted_wins += total_weight
    
    # Weighted win rate
    weighted_win_rate = weighted_wins / weighted_total if weighted_total > 0 else win_rate
    
    # Calculate average P&L (check both 'reward' and 'pnl' fields)
    total_pnl = sum(exp.get('reward', exp.get('pnl', 0)) for exp in similar_experiences)
    avg_pnl = total_pnl / len(similar_experiences)
    
    # Calculate confidence score
    # Start with weighted win rate, then adjust based on sample size and avg P&L
    confidence = weighted_win_rate
    
    # Sample size adjustment (more samples = more confidence in the estimate)
    if len(similar_experiences) < 10:
        confidence *= 0.7  # Low confidence with few samples
    elif len(similar_experiences) < 30:
        confidence *= 0.85  # Medium confidence
    # else: Full confidence with 30+ samples
    
    # P&L quality adjustment (reward consistently profitable setups)
    if avg_pnl > 50:
        confidence = min(confidence + 0.1, 0.95)  # Boost for very profitable setups
    elif avg_pnl < -20:
        confidence *= 0.7  # Penalize consistently losing setups
    
    # Generate reason string
    reason = f"{int(weighted_win_rate * 100)}% win rate in {len(similar_experiences)} similar setups (avg ${avg_pnl:.0f} P&L)"
    
    return {
        'confidence': min(confidence, 0.95),  # Cap at 95%
        'win_rate': weighted_win_rate,
        'sample_size': len(similar_experiences),
        'avg_pnl': avg_pnl,
        'reason': reason
    }


def calculate_signal_confidence(
    all_experiences: List, 
    vwap_distance: float, 
    rsi: float, 
    signal: str,
    current_time: Optional[datetime] = None,
    current_vix: float = 15.0,
    similarity_threshold: float = 0.6,
    volume_ratio: float = 1.0,
    recent_pnl: float = 0.0,
    streak: int = 0
) -> Dict:
    """
    ADVANCED ML confidence based on pattern matching across ALL RL experiences
    
    This is the BRAIN of the bot - it learns from 6,880+ signal experiences and 
    2,961+ exit experiences to make intelligent trading decisions.
    
    Features:
    - Pattern matching: Finds similar past setups (RSI, VWAP, time, VIX, volume, hour, day, streak)
    - Context-aware: Filters by day of week, time of day, volatility
    - Recency-weighted: Recent trades matter more than old ones
    - Quality-weighted: Better trades (higher quality_score) matter more
    - Sample-size aware: More data = higher confidence in predictions
    
    Args:
        all_experiences: All trade results from all users (shared learning)
        vwap_distance: Distance from VWAP (e.g., 0.001 = 0.1%)
        rsi: Current RSI value (0-100)
        signal: 'LONG' or 'SHORT'
        current_time: Current timestamp (for time-of-day filtering)
        current_vix: Current VIX level (for volatility filtering)
        similarity_threshold: Minimum similarity score (0.0-1.0) to consider a past trade
        volume_ratio: Current volume / 20-bar average
        recent_pnl: Sum of last 5 trades P&L
        streak: Consecutive wins (positive) or losses (negative)
    
    Returns:
        Dict with:
            - ml_confidence: 0.0 to 0.95 (never 100% certain)
            - win_rate: Historical win rate for similar setups
            - sample_size: Number of similar past experiences found
            - avg_pnl: Average P&L from similar setups
            - reason: Human-readable explanation
            - should_take: Boolean recommendation (True if confidence > 0.65)
    """
    if current_time is None:
        current_time = datetime.now(pytz.timezone('US/Eastern'))
    
    current_time_only = current_time.time()
    current_day_of_week = current_time.weekday()  # 0=Monday, 6=Sunday
    current_hour = current_time.hour  # 0-23
    
    # STEP 1: Filter experiences by context (signal type, VIX, day of week)
    logger.info(f"🧠 RL Pattern Matching: Analyzing {len(all_experiences)} total experiences for {signal} signal")
    
    contextual_experiences = filter_experiences_by_context(
        experiences=all_experiences,
        signal_type=signal,
        current_day=current_day_of_week,
        current_vix=current_vix,
        min_similarity=similarity_threshold
    )
    
    logger.info(f"   → Filtered to {len(contextual_experiences)} contextually relevant experiences")
    
    # STEP 2: Calculate similarity scores and find matching patterns
    similar_experiences = []
    
    for exp in contextual_experiences:
        similarity = calculate_similarity_score(
            exp=exp,
            current_rsi=rsi,
            current_vwap_dist=vwap_distance,
            current_time=current_time_only,
            current_vix=current_vix,
            current_volume_ratio=volume_ratio,
            current_hour=current_hour,
            current_day_of_week=current_day_of_week,
            current_recent_pnl=recent_pnl,
            current_streak=streak
        )
        
        if similarity >= similarity_threshold:
            exp['similarity_score'] = similarity
            similar_experiences.append(exp)
    
    logger.info(f"   → Found {len(similar_experiences)} highly similar patterns (similarity >= {similarity_threshold})")
    
    # STEP 3: Calculate advanced confidence from similar experiences
    result = calculate_advanced_confidence(similar_experiences)
    
    # Add recommendation
    result['should_take'] = result['confidence'] >= 0.65  # Only recommend high-confidence trades
    
    # Log decision
    if result['should_take']:
        logger.info(f"   ✅ RECOMMEND {signal}: {result['reason']}")
    else:
        logger.info(f"   ⚠️  SKIP {signal}: Confidence {result['confidence']:.1%} < 65% threshold")
    
    return result


@app.post("/api/ml/save_trade")
async def save_trade_experience(trade: Dict, db: Session = Depends(get_db)):
    """
    Save trade experience for RL model training with FULL CONTEXT for pattern matching
    
    Request: {
        user_id: str,  # REQUIRED - for data isolation (account_id)
        symbol: str,
        side: str,  # 'LONG' or 'SHORT'
        entry_price: float,
        exit_price: float,
        entry_time: str,  # ISO format
        exit_time: str,   # ISO format
        pnl: float,
        entry_vwap: float,
        entry_rsi: float,  # RSI at entry (for pattern matching)
        exit_reason: str,
        duration_minutes: float,
        volatility: float,
        streak: int,
        
        # NEW: Optional context for advanced RL pattern matching
        vwap_distance: float,  # Distance from VWAP (percentage, e.g., 0.001 = 0.1%)
        vix: float,  # VIX level at entry (default 15.0)
        confidence: float  # ML confidence at entry (0.0-1.0)
    }
    
    Returns: {
        saved: bool,
        experience_id: str,
        total_shared_trades: int,
        shared_win_rate: float
    }
    """
    try:
        # CRITICAL: Require user_id for data isolation
        user_id = trade.get('user_id', '')
        if not user_id:
            return {
                "saved": False,
                "error": "user_id required"
            }
        
        # Validate required fields (relaxed - only critical fields)
        required_fields = ['symbol', 'side', 'pnl']
        for field in required_fields:
            if field not in trade:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        symbol = trade['symbol']
        
        # Add timestamp and ID
        experience = {
            **trade,
            "saved_at": datetime.utcnow().isoformat(),
            "experience_id": f"{user_id}_{symbol}_{datetime.utcnow().timestamp()}"
        }
        
        # Store in SHARED array (everyone contributes to same strategy learning)
        signal_experiences.append(experience)
        
        # Persist the hive mind to disk every 10 trades
        if len(signal_experiences) % 10 == 0:
            save_experiences()
        
        # **NEW: Save to PostgreSQL database for admin dashboard tracking**
        try:
            # Get user from database
            user = db.query(User).filter(User.account_id == user_id).first()
            if user:
                # Create trade history record
                trade_record = TradeHistory(
                    user_id=user.id,
                    account_id=user_id,
                    symbol=symbol,
                    signal_type=trade['side'],
                    entry_price=trade.get('entry_price', 0.0),
                    exit_price=trade.get('exit_price', 0.0),
                    quantity=trade.get('quantity', 1),
                    pnl=trade['pnl'],
                    entry_time=datetime.fromisoformat(trade['entry_time']) if trade.get('entry_time') else datetime.utcnow(),
                    exit_time=datetime.fromisoformat(trade['exit_time']) if trade.get('exit_time') else datetime.utcnow(),
                    exit_reason=trade.get('exit_reason', 'unknown'),
                    confidence_score=trade.get('confidence', 0.0)
                )
                db.add(trade_record)
                
                # **NEW: Track RL experiences for signal and exit WITH CONTEXT**
                pnl = trade['pnl']
                outcome = 'WIN' if pnl > 0 else 'LOSS'
                
                # Extract context data (with defaults if not provided)
                rsi = trade.get('entry_rsi', trade.get('rsi', 50.0))
                vwap_distance = trade.get('vwap_distance', 0.0)
                vix = trade.get('vix', 15.0)
                
                # Calculate time context
                entry_time = datetime.fromisoformat(trade['entry_time']) if trade.get('entry_time') else datetime.utcnow()
                day_of_week = entry_time.weekday()  # 0=Monday, 6=Sunday
                hour_of_day = entry_time.hour  # 0-23
                
                # Signal experience (entry decision) WITH CONTEXT
                signal_exp = RLExperience(
                    user_id=user.id,
                    account_id=user_id,
                    experience_type='SIGNAL',
                    symbol=symbol,
                    signal_type=trade['side'],
                    outcome=outcome,
                    pnl=pnl,
                    confidence_score=trade.get('confidence', 0.0),
                    quality_score=min(abs(pnl) / 100, 1.0),  # Quality based on P&L magnitude
                    rsi=rsi,
                    vwap_distance=vwap_distance,
                    vix=vix,
                    day_of_week=day_of_week,
                    hour_of_day=hour_of_day
                )
                db.add(signal_exp)
                
                # Exit experience (exit decision) WITH CONTEXT
                exit_exp = RLExperience(
                    user_id=user.id,
                    account_id=user_id,
                    experience_type='EXIT',
                    symbol=symbol,
                    signal_type=trade['side'],
                    outcome=outcome,
                    pnl=pnl,
                    confidence_score=trade.get('confidence', 0.0),
                    quality_score=min(abs(pnl) / 100, 1.0),
                    rsi=rsi,
                    vwap_distance=vwap_distance,
                    vix=vix,
                    day_of_week=day_of_week,
                    hour_of_day=hour_of_day
                )
                db.add(exit_exp)
                
                db.commit()
                logger.debug(f"✅ Trade & RL experiences (with context) saved to database for {user_id}")
        except Exception as db_error:
            logger.warning(f"Failed to save trade to database: {db_error}")
            # Don't fail the entire request if database save fails
        
        # Calculate SHARED win rate (collective wisdom) - handle both formats
        if len(signal_experiences) > 0:
            wins = sum(1 for exp in signal_experiences if exp.get('pnl', exp.get('reward', 0)) > 0)
            win_rate = wins / len(signal_experiences)
        else:
            win_rate = 0.0
        
        logger.info(f"[{user_id}] Trade Saved: {symbol} {trade['side']} P&L=${trade['pnl']:.2f} | "
                   f"🧠 HIVE MIND: {len(signal_experiences):,} trades, WR: {win_rate:.1%}")
        
        return {
            "saved": True,
            "experience_id": experience["experience_id"],
            "total_shared_trades": len(signal_experiences),
            "shared_win_rate": win_rate
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving trade experience: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/save_rejected_signal")
async def save_rejected_signal(signal: Dict):
    """
    Save rejected signal for RL learning - helps model understand when to skip trades.
    
    Request: {
        user_id: str,
        symbol: str,
        side: str,
        signal: str,  # e.g., "long_bounce"
        rsi: float,
        vwap_distance: float,
        volume_ratio: float,
        streak: int,
        took_trade: bool (False),
        rejection_reason: str,
        price_move_ticks: float
    }
    
    Returns: {
        saved: bool,
        total_rejections: int
    }
    """
    try:
        user_id = signal.get('user_id', '')
        if not user_id:
            return {"saved": False, "error": "user_id required"}
        
        # Add timestamp
        experience = {
            **signal,
            "timestamp": datetime.utcnow().isoformat(),
            "took_trade": False  # Always false for rejections
        }
        
        # Store in shared experience pool with negative reward (represents opportunity cost)
        # The RL will learn: "Was skipping this signal the right decision?"
        signal_experiences.append(experience)
        
        # Persist every 25 rejected signals (less frequently than trades)
        if len(signal_experiences) % 25 == 0:
            save_experiences()
        
        rejections = sum(1 for exp in signal_experiences if not exp.get('took_trade', True))
        
        logger.debug(f"[{user_id}] Rejected signal saved: {signal.get('rejection_reason')} | "
                    f"Total rejections tracked: {rejections}")
        
        return {
            "saved": True,
            "total_rejections": rejections
        }
        
    except Exception as e:
        logger.error(f"Error saving rejected signal: {e}")
        return {"saved": False, "error": str(e)}


@app.get("/api/ml/get_exit_experiences")
async def get_exit_experiences():
    """
    Get all exit experiences for Exit RL learning from PostgreSQL.
    ALL USERS share the same collective exit learning pool!
    """
    try:
        from database import DatabaseManager, ExitExperience
        
        db_manager = DatabaseManager()
        session = db_manager.get_session()
        
        try:
            # Query all exit experiences from PostgreSQL
            experiences = session.query(ExitExperience).order_by(ExitExperience.timestamp.desc()).all()
            
            # Convert to dict format matching JSON file structure
            exit_experiences_list = [exp.to_dict() for exp in experiences]
            
            logger.info(f"📤 Serving {len(exit_experiences_list):,} exit experiences from PostgreSQL")
            
            return {
                "success": True,
                "exit_experiences": exit_experiences_list,
                "total_count": len(exit_experiences_list),
                "message": f"🌍 Loaded {len(exit_experiences_list):,} shared exit experiences from PostgreSQL"
            }
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"❌ Error getting exit experiences from PostgreSQL: {e}")
        return {
            "success": False,
            "exit_experiences": [],
            "total_count": 0,
            "error": str(e)
        }


@app.post("/api/ml/save_exit_experience")
async def save_exit_experience(experience: dict):
    """
    Save new exit experience to PostgreSQL - ALL USERS contribute!
    Every trade exit adds to the collective learning pool.
    
    NEW: Saves to rl_experiences table with 9-feature context for pattern matching
    """
    try:
        from database import DatabaseManager, RLExperience
        from datetime import datetime
        
        db_manager = DatabaseManager()
        session = db_manager.get_session()
        
        try:
            # Extract 9-feature market context (RSI, volume_ratio, hour, day_of_week, streak, recent_pnl, VIX, VWAP distance, ATR)
            market_state = experience.get('market_state', {})
            outcome = experience.get('outcome', {})
            
            # Get or create a default user for backtest data
            from database import User
            default_user = session.query(User).filter_by(account_id='backtest').first()
            if not default_user:
                # Create backtest user if doesn't exist
                default_user = User(
                    account_id='backtest',
                    email='backtest@quotrading.com',
                    license_key='BACKTEST-KEY',
                    license_type='SYSTEM',
                    license_status='ACTIVE'
                )
                session.add(default_user)
                session.flush()
            
            # Create RLExperience record for exit (experience_type='EXIT')
            exit_exp = RLExperience(
                experience_type='EXIT',
                user_id=default_user.id,  # Use backtest user ID
                account_id='backtest',
                symbol='ES',  # Default to ES
                signal_type=outcome.get('side', 'long').upper(),  # LONG or SHORT
                outcome='WIN' if outcome.get('win', False) else 'LOSS',
                pnl=float(outcome.get('pnl', 0.0)),
                confidence_score=None,
                quality_score=None,
                
                # 9-feature market context for exit pattern matching
                rsi=float(market_state.get('rsi', 50.0)),
                volume_ratio=float(market_state.get('volume_ratio', 1.0)),
                hour_of_day=int(market_state.get('hour', 12)),
                day_of_week=int(market_state.get('day_of_week', 0)),
                streak=int(market_state.get('streak', 0)),
                recent_pnl=float(market_state.get('recent_pnl', 0.0)),
                vix=float(market_state.get('vix', 15.0)),
                vwap_distance=float(market_state.get('vwap_distance', 0.0)),
                atr=float(market_state.get('atr', 0.0)),
                
                # Additional fields (not used for pattern matching but stored for reference)
                entry_price=None,
                vwap=None,
                price=None,
                side=outcome.get('side', 'long'),
                
                timestamp=datetime.fromisoformat(experience.get('timestamp', datetime.utcnow().isoformat()))
            )
            
            session.add(exit_exp)
            session.commit()
            
            # Get updated count
            total_count = session.query(RLExperience).filter(
                RLExperience.experience_type == 'EXIT'
            ).count()
            
            logger.info(f"✅ Saved EXIT experience to rl_experiences table (now {total_count:,} total exits)")
            
            return {
                "saved": True,
                "total_exit_experiences": total_count,
                "message": f"🌍 Exit experience saved to rl_experiences table ({total_count:,} exits with 9 features)"
            }
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
        
    except Exception as e:
        logger.error(f"❌ Error saving exit experience to PostgreSQL: {e}")
        return {"saved": False, "error": str(e)}


@app.get("/api/ml/stats")
async def get_ml_stats():
    """
    Get shared ML statistics (everyone learns from same strategy)
    """
    if len(signal_experiences) == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "total_pnl": 0.0,
            "message": "No trades yet - shared learning pool is empty"
        }
    
    # Handle both loaded format (reward) and new trades (pnl)
    # Debug: Check first few experiences
    sample_exp = signal_experiences[0] if signal_experiences else {}
    logger.info(f"DEBUG Stats - Sample experience keys: {list(sample_exp.keys())}")
    logger.info(f"DEBUG Stats - Sample PnL field: pnl={sample_exp.get('pnl')}, reward={sample_exp.get('reward')}")
    
    wins = sum(1 for exp in signal_experiences if exp.get('pnl', exp.get('reward', 0)) > 0)
    total_pnl = sum(exp.get('pnl', exp.get('reward', 0)) for exp in signal_experiences)
    
    logger.info(f"DEBUG Stats - Wins: {wins}, Total PnL: {total_pnl}, Total Trades: {len(signal_experiences)}")
    
    return {
        "total_trades": len(signal_experiences),
        "win_rate": wins / len(signal_experiences),
        "avg_pnl": total_pnl / len(signal_experiences),
        "total_pnl": total_pnl,
        "message": "Shared learning - all users contribute and benefit",
        "last_updated": datetime.utcnow().isoformat()
    }

# ============================================================================
# LICENSE MANAGEMENT ENDPOINTS
# ============================================================================

def generate_license_key() -> str:
    """Generate a unique license key"""
    random_bytes = secrets.token_bytes(16)
    return hashlib.sha256(random_bytes).hexdigest()[:24].upper()

@app.post("/api/license/validate")
async def validate_license(data: dict, db: Session = Depends(get_db)):
    """Validate if a license key is active"""
    license_key = data.get("license_key", "").strip()
    
    if not license_key:
        raise HTTPException(status_code=400, detail="License key required")
    
    # Check database if available
    if db_manager:
        user = get_user_by_license_key(db, license_key)
        if not user:
            raise HTTPException(status_code=403, detail="Invalid license key")
        
        if not user.is_license_valid:
            if user.license_status == 'SUSPENDED':
                raise HTTPException(status_code=403, detail="License suspended")
            elif user.license_expiration and user.license_expiration < datetime.utcnow():
                raise HTTPException(status_code=403, detail="License expired")
            else:
                raise HTTPException(status_code=403, detail="License inactive")
        
        # Update last active
        update_user_activity(db, user.id)
        
        return {
            "valid": True,
            "account_id": user.account_id,
            "email": user.email,
            "license_type": user.license_type,
            "expires_at": user.license_expiration.isoformat() if user.license_expiration else None,
            "message": "License is valid"
        }
    
    # Fallback to hardcoded licenses if database unavailable
    if license_key not in active_licenses:
        raise HTTPException(status_code=403, detail="Invalid license key")
    
    license_info = active_licenses[license_key]
    
    # Check if expired
    if license_info.get("expires_at"):
        expires_at = datetime.fromisoformat(license_info["expires_at"])
        if datetime.utcnow() > expires_at:
            raise HTTPException(status_code=403, detail="License expired")
    
    return {
        "valid": True,
        "email": license_info.get("email"),
        "expires_at": license_info.get("expires_at"),
        "subscription_status": license_info.get("status", "active"),
        "message": "License is valid (fallback mode)"
    }

@app.post("/api/license/activate")
async def activate_license(data: dict):
    """Manually activate a license (for beta testing)"""
    email = data.get("email")
    days = data.get("days", 30)  # Default 30 days
    
    if not email:
        raise HTTPException(status_code=400, detail="Email required")
    
    license_key = generate_license_key()
    expires_at = datetime.utcnow() + timedelta(days=days)
    
    active_licenses[license_key] = {
        "email": email,
        "expires_at": expires_at.isoformat(),
        "status": "active",
        "created_at": datetime.utcnow().isoformat()
    }
    
    logger.info(f"🔑 License created: {license_key} for {email} (expires: {expires_at})")
    
    return {
        "license_key": license_key,
        "email": email,
        "expires_at": expires_at.isoformat()
    }

@app.get("/api/license/list")
async def list_licenses():
    """List all active licenses (admin only - add auth later)"""
    return {
        "total": len(active_licenses),
        "licenses": [
            {
                "key": key[:8] + "..." + key[-4:],  # Partially hide key
                "email": info.get("email"),
                "status": info.get("status"),
                "expires_at": info.get("expires_at")
            }
            for key, info in active_licenses.items()
        ]
    }

# ============================================================================
# EMERGENCY KILL SWITCH
# ============================================================================

# Global kill switch state
kill_switch_state = {
    "active": False,
    "reason": "",
    "activated_at": None,
    "activated_by": "system"
}

@app.get("/api/kill_switch/status")
async def get_kill_switch_status():
    """
    Bots check this endpoint every 30 seconds.
    If kill switch is active, bots flatten positions and stop trading.
    """
    return {
        "kill_switch_active": kill_switch_state["active"],
        "trading_enabled": not kill_switch_state["active"],
        "reason": kill_switch_state["reason"] if kill_switch_state["active"] else "Trading active",
        "activated_at": kill_switch_state["activated_at"]
    }

@app.post("/api/admin/kill_switch")
async def toggle_kill_switch(data: dict):
    """
    ADMIN ONLY: Emergency stop all customer bots.
    
    Use cases:
    - Bug discovered in strategy
    - Major market event (flash crash, news)
    - Strategy needs revision
    - Emergency maintenance
    
    Request body:
    {
        "active": true/false,
        "reason": "Bug in stop loss logic - emergency halt",
        "admin_key": "your_secret_admin_key"
    }
    """
    # Simple admin authentication (in production, use proper auth)
    admin_key = data.get("admin_key")
    if admin_key != "QUOTRADING_ADMIN_2025":  # Change this to env variable!
        raise HTTPException(status_code=403, detail="Invalid admin key")
    
    active = data.get("active", False)
    reason = data.get("reason", "Emergency stop activated by admin")
    
    kill_switch_state["active"] = active
    kill_switch_state["reason"] = reason
    kill_switch_state["activated_at"] = datetime.utcnow().isoformat() if active else None
    kill_switch_state["activated_by"] = "admin"
    
    status = "ACTIVATED" if active else "DEACTIVATED"
    logger.warning(f"🚨 KILL SWITCH {status}: {reason}")
    
    return {
        "kill_switch_active": active,
        "reason": reason,
        "message": f"Kill switch {status.lower()}. All bots will {'stop' if active else 'resume'} within 30 seconds.",
        "activated_at": kill_switch_state["activated_at"]
    }

# ============================================================================
# STRIPE WEBHOOK HANDLER
# ============================================================================

@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events"""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    
    if not STRIPE_WEBHOOK_SECRET:
        logger.warning("⚠️ Stripe webhook secret not configured - skipping verification")
        event = json.loads(payload)
    else:
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, STRIPE_WEBHOOK_SECRET
            )
        except ValueError as e:
            logger.error(f"Invalid payload: {e}")
            raise HTTPException(status_code=400, detail="Invalid payload")
        except stripe.error.SignatureVerificationError as e:
            logger.error(f"Invalid signature: {e}")
            raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Handle the event
    event_type = event["type"]
    data = event["data"]["object"]
    
    logger.info(f"📬 Stripe webhook: {event_type}")
    
    if event_type == "checkout.session.completed":
        # Payment successful - create license
        customer_email = data.get("customer_email")
        subscription_id = data.get("subscription")
        customer_id = data.get("customer")
        
        if customer_email:
            license_key = generate_license_key()
            
            # Subscription licenses don't expire (auto-renew)
            active_licenses[license_key] = {
                "email": customer_email,
                "expires_at": None,  # Subscription - no expiry
                "status": "active",
                "stripe_customer_id": customer_id,
                "stripe_subscription_id": subscription_id,
                "created_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"🎉 License created from payment: {license_key} for {customer_email}")
            
            # TODO: Send email with license key (add later)
    
    elif event_type == "customer.subscription.deleted":
        # Subscription cancelled - revoke license
        subscription_id = data.get("id")
        
        for key, info in list(active_licenses.items()):
            if info.get("stripe_subscription_id") == subscription_id:
                active_licenses[key]["status"] = "cancelled"
                logger.info(f"❌ License cancelled: {key} (subscription ended)")
    
    elif event_type == "invoice.payment_failed":
        # Payment failed - suspend license
        subscription_id = data.get("subscription")
        
        for key, info in list(active_licenses.items()):
            if info.get("stripe_subscription_id") == subscription_id:
                active_licenses[key]["status"] = "suspended"
                logger.warning(f"⚠️ License suspended: {key} (payment failed)")
    
    return {"status": "success"}

# ============================================================================
# STRIPE CHECKOUT
# ============================================================================

@app.post("/api/stripe/create-checkout")
async def create_checkout_session():
    """Create a Stripe Checkout session for subscription"""
    try:
        # QuoTrading Bot - $200/month subscription
        PRICE_ID = "price_1SRMSvBcgS15fNXbyHGeG9IZ"
        
        checkout_session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{
                "price": PRICE_ID,
                "quantity": 1,
            }],
            success_url="https://quotrading.com/success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url="https://quotrading.com/cancel",
        )
        
        return {"session_id": checkout_session.id}
    except Exception as e:
        logger.error(f"Checkout error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ECONOMIC CALENDAR - FOMC AUTO-SCRAPER
# ============================================================================

import asyncio
import threading
from bs4 import BeautifulSoup
import requests as req_lib

# Global calendar state
economic_calendar = {
    "events": [],
    "last_updated": None,
    "next_update": None,
    "source": "Federal Reserve + Manual"
}

def scrape_fomc_dates() -> List[Dict]:
    """
    Scrape FOMC meeting dates from Federal Reserve website
    Returns list of FOMC events
    """
    fomc_events = []
    
    try:
        logger.info("📅 Fetching FOMC dates from federalreserve.gov...")
        
        # Fetch Federal Reserve calendar page
        url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
        response = req_lib.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find FOMC meeting dates in the page
        # The structure typically has dates in specific div/table elements
        # This is a simplified parser - may need adjustment if Fed changes format
        
        # Look for date patterns (MM/DD/YYYY or Month DD, YYYY)
        import re
        date_pattern = r'(\d{1,2}/\d{1,2}/\d{4}|\w+ \d{1,2}, \d{4})'
        
        # Find all text containing potential dates
        page_text = soup.get_text()
        potential_dates = re.findall(date_pattern, page_text)
        
        logger.info(f"Found {len(potential_dates)} potential FOMC dates on Fed website")
        
        # Parse and format dates
        from dateutil import parser as date_parser
        
        for date_str in potential_dates[:20]:  # Limit to next 20 meetings (years ahead)
            try:
                parsed_date = date_parser.parse(date_str)
                
                # Only include future dates
                if parsed_date.date() > datetime.now().date():
                    # Add FOMC Statement (2 PM ET)
                    fomc_events.append({
                        "date": parsed_date.strftime("%Y-%m-%d"),
                        "time": "2:00pm",
                        "currency": "USD",
                        "event": "FOMC Statement",
                        "impact": "high"
                    })
                    
                    # Add FOMC Press Conference (2:30 PM ET)
                    fomc_events.append({
                        "date": parsed_date.strftime("%Y-%m-%d"),
                        "time": "2:30pm",
                        "currency": "USD",
                        "event": "FOMC Press Conference",
                        "impact": "high"
                    })
            except Exception as e:
                logger.debug(f"Could not parse date: {date_str}")
                continue
        
        logger.info(f"✅ Scraped {len(fomc_events)} FOMC events from Federal Reserve")
        
    except Exception as e:
        logger.error(f"❌ Failed to scrape FOMC dates: {e}")
        logger.info("Will use manual FOMC dates as fallback")
    
    return fomc_events

def generate_predictable_events() -> List[Dict]:
    """
    Generate predictable economic events (NFP, CPI, PPI)
    These follow consistent schedules
    """
    events = []
    current_date = datetime.now().date()
    
    # Generate 12 months of events
    for month_offset in range(12):
        year = current_date.year + (current_date.month + month_offset - 1) // 12
        month = (current_date.month + month_offset - 1) % 12 + 1
        
        # NFP - First Friday of month at 8:30 AM ET
        first_day = datetime(year, month, 1).date()
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        
        if first_friday > current_date:
            events.append({
                "date": first_friday.strftime("%Y-%m-%d"),
                "time": "8:30am",
                "currency": "USD",
                "event": "Non-Farm Employment Change",
                "impact": "high"
            })
        
        # CPI - Typically around 13th of month at 8:30 AM ET
        cpi_date = datetime(year, month, 13).date()
        if cpi_date > current_date:
            events.append({
                "date": cpi_date.strftime("%Y-%m-%d"),
                "time": "8:30am",
                "currency": "USD",
                "event": "Core CPI m/m",
                "impact": "high"
            })
        
        # PPI - Typically around 14th of month at 8:30 AM ET
        ppi_date = datetime(year, month, 14).date()
        if ppi_date > current_date:
            events.append({
                "date": ppi_date.strftime("%Y-%m-%d"),
                "time": "8:30am",
                "currency": "USD",
                "event": "Core PPI m/m",
                "impact": "high"
            })
    
    return events

def update_calendar():
    """
    Update economic calendar with latest FOMC + predictable events
    Runs daily at 5 PM ET (Sunday-Friday)
    """
    try:
        logger.info("📅 Updating economic calendar...")
        
        # Scrape FOMC dates from Federal Reserve
        fomc_events = scrape_fomc_dates()
        
        # Generate predictable events
        predictable_events = generate_predictable_events()
        
        # Combine and sort by date
        all_events = fomc_events + predictable_events
        all_events.sort(key=lambda x: x["date"])
        
        # Remove duplicates (keep first occurrence)
        seen_dates = set()
        unique_events = []
        for event in all_events:
            event_key = (event["date"], event["event"])
            if event_key not in seen_dates:
                seen_dates.add(event_key)
                unique_events.append(event)
        
        # Update global calendar
        economic_calendar["events"] = unique_events
        economic_calendar["last_updated"] = datetime.utcnow().isoformat()
        economic_calendar["next_update"] = get_next_update_time().isoformat()
        
        logger.info(f"✅ Calendar updated: {len(unique_events)} events ({len(fomc_events)} FOMC + {len(predictable_events)} NFP/CPI/PPI)")
        
    except Exception as e:
        logger.error(f"❌ Calendar update failed: {e}")

def get_next_update_time() -> datetime:
    """
    Calculate next update time: 1st of every month at 5 PM ET
    """
    et_tz = pytz.timezone("America/New_York")
    now_et = datetime.now(et_tz)
    
    # Target: 1st of next month at 5 PM ET
    if now_et.day == 1 and now_et.hour < 17:
        # It's the 1st and before 5 PM - update today at 5 PM
        target_time = now_et.replace(hour=17, minute=0, second=0, microsecond=0)
    else:
        # Schedule for 1st of next month at 5 PM
        if now_et.month == 12:
            next_month = now_et.replace(year=now_et.year + 1, month=1, day=1, hour=17, minute=0, second=0, microsecond=0)
        else:
            next_month = now_et.replace(month=now_et.month + 1, day=1, hour=17, minute=0, second=0, microsecond=0)
        target_time = next_month
    
    return target_time

async def calendar_update_loop():
    """
    Background task that updates calendar daily at 5 PM ET (Sunday-Friday)
    """
    while True:
        try:
            next_update = get_next_update_time()
            now = datetime.now(pytz.timezone("America/New_York"))
            sleep_seconds = (next_update - now).total_seconds()
            
            logger.info(f"📅 Next calendar update: {next_update.strftime('%Y-%m-%d %I:%M %p ET')} ({sleep_seconds/3600:.1f} hours)")
            
            await asyncio.sleep(sleep_seconds)
            
            # Update calendar
            update_calendar()
            
        except Exception as e:
            logger.error(f"❌ Calendar update loop error: {e}")
            # Wait 1 hour and retry
            await asyncio.sleep(3600)

def start_calendar_updater():
    """Start background calendar updater in separate thread"""
    def run_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(calendar_update_loop())
    
    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    logger.info("📅 Calendar updater started (1st of month at 5 PM ET)")

@app.get("/api/calendar/events")
async def get_calendar_events(days: int = 7):
    """
    Get upcoming economic events for next N days
    
    Query params:
        days: Number of days ahead to fetch (default 7)
    """
    today = datetime.now().date()
    end_date = today + timedelta(days=days)
    
    upcoming_events = [
        event for event in economic_calendar["events"]
        if today <= datetime.strptime(event["date"], "%Y-%m-%d").date() <= end_date
    ]
    
    return {
        "events": upcoming_events,
        "count": len(upcoming_events),
        "last_updated": economic_calendar["last_updated"],
        "next_update": economic_calendar["next_update"]
    }

@app.get("/api/calendar/today")
async def get_todays_events():
    """
    Get today's high-impact economic events
    Bots check this before placing trades
    """
    today = datetime.now().date().strftime("%Y-%m-%d")
    
    todays_events = [
        event for event in economic_calendar["events"]
        if event["date"] == today and event["impact"] == "high"
    ]
    
    has_fomc = any("FOMC" in event["event"] for event in todays_events)
    has_nfp = any("Non-Farm" in event["event"] for event in todays_events)
    has_cpi = any("CPI" in event["event"] for event in todays_events)
    
    return {
        "date": today,
        "events": todays_events,
        "count": len(todays_events),
        "has_fomc": has_fomc,
        "has_nfp": has_nfp,
        "has_cpi": has_cpi,
        "trading_recommended": len(todays_events) == 0
    }

@app.post("/api/admin/refresh_calendar")
async def refresh_calendar(data: dict):
    """
    ADMIN ONLY: Manually trigger calendar refresh
    """
    admin_key = data.get("admin_key")
    if admin_key != "QUOTRADING_ADMIN_2025":
        raise HTTPException(status_code=403, detail="Invalid admin key")
    
    update_calendar()
    
    return {
        "status": "refreshed",
        "events_count": len(economic_calendar["events"]),
        "last_updated": economic_calendar["last_updated"]
    }

# ============================================================================
# TIME SERVICE - SINGLE SOURCE OF TRUTH
# ============================================================================

def get_market_hours_status(now_et: datetime) -> str:
    """
    Determine current market status
    
    Futures hours (ES/NQ):
    - Sunday 6:00 PM - Friday 5:00 PM ET (weekly)
    - Daily maintenance: Mon-Thu 5:00-6:00 PM, Fri-Sun 5:00 PM - Sun 6:00 PM
    
    Returns: pre_market, market_open, after_hours, futures_open, maintenance, weekend_closed
    """
    weekday = now_et.weekday()  # Monday=0, Sunday=6
    current_time = now_et.time()
    
    # Saturday - Weekend closed
    if weekday == 5:
        return "weekend_closed"
    
    # Sunday
    elif weekday == 6:
        if current_time >= datetime_time(18, 0):
            # Sunday 6 PM onwards - futures open
            return "futures_open"
        else:
            # Before Sunday 6 PM - weekend closed
            return "weekend_closed"
    
    # Monday-Thursday
    elif weekday <= 3:  # Monday(0) to Thursday(3)
        if datetime_time(17, 0) <= current_time < datetime_time(18, 0):
            # 5:00-6:00 PM - Daily maintenance
            return "maintenance"
        elif current_time < datetime_time(9, 30):
            # Before 9:30 AM - Pre-market/futures
            if current_time >= datetime_time(4, 0):
                return "pre_market"
            else:
                return "futures_open"
        elif datetime_time(9, 30) <= current_time < datetime_time(16, 0):
            # 9:30 AM - 4:00 PM - Market open
            return "market_open"
        elif datetime_time(16, 0) <= current_time < datetime_time(17, 0):
            # 4:00-5:00 PM - After hours
            return "after_hours"
        else:
            # After 6:00 PM - Futures open
            return "futures_open"
    
    # Friday
    elif weekday == 4:
        if current_time >= datetime_time(17, 0):
            # Friday 5 PM onwards - Weekend maintenance
            return "weekend_closed"
        elif current_time < datetime_time(9, 30):
            # Before 9:30 AM
            if current_time >= datetime_time(4, 0):
                return "pre_market"
            else:
                return "futures_open"
        elif datetime_time(9, 30) <= current_time < datetime_time(16, 0):
            # 9:30 AM - 4:00 PM - Market open
            return "market_open"
        elif datetime_time(16, 0) <= current_time < datetime_time(17, 0):
            # 4:00-5:00 PM - After hours (approaching weekend)
            return "after_hours"
    
    return "unknown"

def get_trading_session(now_et: datetime) -> str:
    """
    Determine current trading session
    
    Returns: asian, european, us, overlap
    """
    current_time = now_et.time()
    
    # Asian session: 6 PM - 3 AM ET (Tokyo/Hong Kong)
    if current_time >= datetime_time(18, 0) or current_time < datetime_time(3, 0):
        return "asian"
    # European session: 3 AM - 12 PM ET (London)
    elif datetime_time(3, 0) <= current_time < datetime_time(12, 0):
        # Overlap with US: 9:30 AM - 12 PM ET
        if datetime_time(9, 30) <= current_time < datetime_time(12, 0):
            return "overlap"
        return "european"
    # US session: 9:30 AM - 4 PM ET
    elif datetime_time(9, 30) <= current_time < datetime_time(16, 0):
        return "us"
    else:
        return "asian"

def check_if_event_active(events: List[Dict], now_et: datetime) -> tuple:
    """
    Check if any high-impact economic event is currently active
    
    Returns: (is_active, event_name, event_window)
    """
    today_str = now_et.date().strftime("%Y-%m-%d")
    current_time = now_et.time()
    
    # Filter today's events
    todays_events = [e for e in events if e["date"] == today_str and e["impact"] == "high"]
    
    for event in todays_events:
        event_time_str = event["time"]
        
        # Parse event time (e.g., "8:30am" or "2:00pm")
        event_time_str = event_time_str.lower().replace("am", "").replace("pm", "").strip()
        hour, minute = map(int, event_time_str.split(":"))
        
        # Adjust for PM
        if "pm" in event["time"].lower() and hour != 12:
            hour += 12
        elif "am" in event["time"].lower() and hour == 12:
            hour = 0
        
        event_time = datetime_time(hour, minute)
        
        # Event window: 30 minutes before to 1 hour after
        event_start = (datetime.combine(now_et.date(), event_time) - timedelta(minutes=30)).time()
        event_end = (datetime.combine(now_et.date(), event_time) + timedelta(hours=1)).time()
        
        # Check if we're in the event window
        if event_start <= current_time <= event_end:
            window = f"{event_start.strftime('%I:%M %p')} - {event_end.strftime('%I:%M %p')}"
            return (True, event["event"], window)
    
    return (False, None, None)

@app.get("/api/time")
async def get_time_service():
    """
    Centralized time service - Single source of truth for all bots
    
    Provides:
    - Current ET time
    - Market hours status
    - Trading session
    - Economic event awareness
    - Trading permission
    
    Bots should call this every 30-60 seconds to stay synchronized
    """
    # Get current ET time
    et_tz = pytz.timezone("America/New_York")
    now_et = datetime.now(et_tz)
    
    # Market status
    market_status = get_market_hours_status(now_et)
    session = get_trading_session(now_et)
    
    # Check for active economic events
    event_active, event_name, event_window = check_if_event_active(economic_calendar["events"], now_et)
    
    # Determine if trading is allowed
    trading_allowed = True
    halt_reason = None
    
    if event_active:
        trading_allowed = False
        halt_reason = f"{event_name} in progress ({event_window})"
    
    # Get today's upcoming events
    today_str = now_et.date().strftime("%Y-%m-%d")
    todays_events = [
        {
            "event": e["event"],
            "time": e["time"],
            "impact": e["impact"]
        }
        for e in economic_calendar["events"]
        if e["date"] == today_str and e["impact"] == "high"
    ]
    
    return {
        # Time information
        "current_et": now_et.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "current_timestamp": now_et.isoformat(),
        "timezone": "America/New_York",
        
        # Market information
        "market_status": market_status,
        "trading_session": session,
        "weekday": now_et.strftime("%A"),
        
        # Economic events
        "event_active": event_active,
        "active_event": event_name if event_active else None,
        "event_window": event_window if event_active else None,
        "events_today": todays_events,
        "events_count": len(todays_events),
        
        # Trading permission
        "trading_allowed": trading_allowed,
        "halt_reason": halt_reason,
        
        # Calendar info
        "calendar_last_updated": economic_calendar.get("last_updated"),
        "calendar_next_update": economic_calendar.get("next_update")
    }

@app.get("/api/time/simple")
async def get_simple_time():
    """
    Lightweight time check - Just ET time and trading permission
    For bots that need quick checks without full details
    
    Checks:
    - Economic events (FOMC/NFP/CPI)
    - Maintenance windows (Mon-Thu 5-6 PM, Fri 5 PM - Sun 6 PM)
    - Weekend closure
    """
    et_tz = pytz.timezone("America/New_York")
    now_et = datetime.now(et_tz)
    
    # Get market status
    market_status = get_market_hours_status(now_et)
    
    # Check for active events
    event_active, event_name, event_window = check_if_event_active(economic_calendar["events"], now_et)
    
    # Determine if trading is allowed
    trading_allowed = True
    halt_reason = None
    
    # Priority 1: Economic events
    if event_active:
        trading_allowed = False
        halt_reason = f"{event_name} ({event_window})"
    # Priority 2: Maintenance windows
    elif market_status == "maintenance":
        trading_allowed = False
        halt_reason = "Daily maintenance (5-6 PM ET)"
    # Priority 3: Weekend closure
    elif market_status == "weekend_closed":
        trading_allowed = False
        weekday = now_et.weekday()
        if weekday == 5:  # Saturday
            halt_reason = "Weekend - Market closed (opens Sunday 6 PM ET)"
        elif weekday == 6:  # Sunday before 6 PM
            halt_reason = "Weekend - Market opens at 6:00 PM ET"
        else:  # Friday after 5 PM
            halt_reason = "Weekend - Market closed (opens Sunday 6 PM ET)"
    
    return {
        "current_et": now_et.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "trading_allowed": trading_allowed,
        "halt_reason": halt_reason,
        "market_status": market_status  # Added for debugging
    }

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize signal engine on startup"""
    global db_manager, redis_manager
    
    logger.info("=" * 60)
    logger.info("QuoTrading Signal Engine v2.1 - STARTING")
    logger.info("=" * 60)
    logger.info("Multi-instrument support: ES, NQ, YM, RTY")
    logger.info("Features: ML/RL signals, licensing, economic calendar, user management")
    logger.info("=" * 60)
    
    # Initialize Database
    logger.info("💾 Initializing database connection...")
    try:
        import database as db_module
        db_manager = DatabaseManager()
        db_module.db_manager = db_manager  # Set global instance
        
        # Create tables if they don't exist
        db_manager.create_tables()
        logger.info(f"✅ Database connected: {db_manager.database_url}")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        logger.warning("⚠️  Running without database - license validation will be limited")
        db_manager = None
    
    # Initialize Redis
    logger.info("🔴 Initializing Redis connection...")
    try:
        redis_manager = RedisManager(fallback_to_memory=True)
        if redis_manager.using_memory_fallback:
            logger.info("📦 Using in-memory fallback for rate limiting")
        else:
            logger.info("✅ Redis connected for rate limiting")
    except Exception as e:
        logger.error(f"❌ Redis initialization failed: {e}")
        logger.info("� Using in-memory fallback for rate limiting")
        redis_manager = RedisManager(fallback_to_memory=True)
    
    # Initialize economic calendar
    logger.info("�📅 Initializing economic calendar...")
    update_calendar()  # Initial fetch
    start_calendar_updater()  # Start background updater
    
    # Load RL experiences (baseline from JSON + new from database)
    logger.info("🧠 Loading RL experiences for pattern matching...")
    load_initial_experiences()  # Load baseline from JSON (6,880 + 2,961)
    
    if db_manager:
        try:
            db = next(get_db())
            load_database_experiences(db)  # Add new experiences from database
            db.close()
        except Exception as e:
            logger.error(f"❌ Could not load database experiences: {e}")
    
    logger.info("=" * 60)
    logger.info("✅ Signal Engine Ready!")
    logger.info("=" * 60)


# ============================================================================
# STATIC FILES - Mount admin dashboard
# ============================================================================
import os
if os.path.exists("admin-dashboard"):
    app.mount("/admin-dashboard", StaticFiles(directory="admin-dashboard"), name="admin-dashboard")
    logger.info("📊 Admin dashboard mounted at /admin-dashboard")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
