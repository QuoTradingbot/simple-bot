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

# Import neural network confidence scorer
from neural_confidence import init_neural_scorer, get_neural_prediction

# Import neural network exit predictor
from neural_exit import NeuralExitPredictor

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
neural_exit_predictor: Optional[NeuralExitPredictor] = None  # Exit parameter predictor

# Global RL experience pools (shared across all users)
signal_experiences = []  # Signal RL learning pool

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
                    user.last_active = datetime.now(pytz.UTC)
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
            "ml": "/api/ml/get_confidence, /api/ml/predict_exit_params, /api/ml/save_trade, /api/ml/stats",
            "license": "/api/license/validate, /api/license/activate",
            "time": "/api/time, /api/time/simple"
        }
    }


@app.get("/health")
async def health():
    """Health check for monitoring"""
    return {"status": "healthy", "timestamp": datetime.now(pytz.UTC).isoformat()}


@app.get("/api/performance")
async def performance_stats():
    """
    PRODUCTION MONITORING: Database connection pool stats and throughput metrics.
    Use this to monitor scalability and identify bottlenecks.
    
    Returns:
        - Connection pool status (active, idle, total)
        - Database write throughput estimates
        - Experience counts per type
        - System capacity metrics
    """
    try:
        from database import DatabaseManager, MLExperience
        
        db_manager = DatabaseManager()
        
        # Get connection pool statistics
        pool_stats = db_manager.get_pool_status()
        
        # Get experience counts (fast aggregation)
        session = db_manager.get_session()
        try:
            from sqlalchemy import func
            
            # Count experiences by type (fast query with index)
            experience_counts = session.query(
                MLExperience.experience_type,
                func.count(MLExperience.id).label('count')
            ).group_by(MLExperience.experience_type).all()
            
            counts_dict = {exp_type: count for exp_type, count in experience_counts}
            total_experiences = sum(counts_dict.values())
            
            # Estimate write throughput (last hour)
            one_hour_ago = datetime.now(pytz.UTC) - timedelta(hours=1)
            recent_writes = session.query(func.count(MLExperience.id)).filter(
                MLExperience.timestamp >= one_hour_ago
            ).scalar()
            
            writes_per_second = round(recent_writes / 3600, 2)
            
        except Exception as e:
            logger.warning(f"Stats query failed: {e}")
            counts_dict = {}
            total_experiences = 0
            recent_writes = 0
            writes_per_second = 0
        finally:
            session.close()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(pytz.UTC).isoformat(),
            "connection_pool": pool_stats,
            "database": {
                "total_experiences": total_experiences,
                "experience_counts": counts_dict,
                "recent_writes_1h": recent_writes,
                "writes_per_second": writes_per_second
            },
            "capacity": {
                "estimated_max_writes_per_second": 1000,
                "current_utilization_percent": round((writes_per_second / 1000) * 100, 2),
                "headroom": f"{round((1 - writes_per_second / 1000) * 100, 1)}% available"
            }
        }
        
    except Exception as e:
        logger.error(f"Performance stats error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }


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
        user.license_expiration = datetime.now(pytz.UTC) + timedelta(days=additional_days)
    
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
    yesterday = datetime.now(pytz.UTC) - timedelta(days=1)
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
    five_minutes_ago = datetime.now(pytz.UTC) - timedelta(minutes=5)
    
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
                "seconds_ago": int((datetime.now(pytz.UTC) - user.last_active).total_seconds()) if user.last_active else None
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
    five_min_ago = datetime.now(pytz.UTC) - timedelta(minutes=5)
    online_now = db.query(func.count(User.id)).filter(
        User.last_active >= five_min_ago,
        User.license_status == 'ACTIVE'
    ).scalar()
    
    # License type breakdown
    license_breakdown = {}
    for license_type, count in db.query(User.license_type, func.count(User.id)).group_by(User.license_type).all():
        license_breakdown[license_type] = count
    
    # API calls
    one_hour_ago = datetime.now(pytz.UTC) - timedelta(hours=1)
    one_day_ago = datetime.now(pytz.UTC) - timedelta(days=1)
    
    api_calls_1h = db.query(func.count(APILog.id)).filter(APILog.timestamp >= one_hour_ago).scalar()
    api_calls_24h = db.query(func.count(APILog.id)).filter(APILog.timestamp >= one_day_ago).scalar()
    
    # Trade stats
    total_trades = db.query(func.count(TradeHistory.id)).scalar()
    total_pnl = db.query(func.sum(TradeHistory.pnl)).scalar()
    trades_today = db.query(func.count(TradeHistory.id)).filter(
        func.date(TradeHistory.entry_time) == datetime.now(pytz.UTC).date()
    ).scalar()
    
    # Expiring licenses (next 7 days)
    seven_days = datetime.now(pytz.UTC) + timedelta(days=7)
    expiring_soon = db.query(func.count(User.id)).filter(
        User.license_expiration <= seven_days,
        User.license_expiration >= datetime.now(pytz.UTC),
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
        "timestamp": datetime.now(pytz.UTC).isoformat(),
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
        logger.info("✅ Database tables initialized")
        
        logger.info("� All experiences stored in PostgreSQL database")
        
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
        
        # Prepare state for neural network (same format as backtest)
        current_time = datetime.now(pytz.UTC)
        state = {
            'rsi': rsi,
            'vwap_distance': abs(price - vwap) / vwap if vwap > 0 else 0,
            'vix': vix,
            'spread_ticks': request.get('spread_ticks', 1.0),
            'hour': current_time.hour,
            'day_of_week': current_time.weekday(),
            'volume_ratio': request.get('volume_ratio', 1.0),
            'atr': request.get('atr', 10.0),
            'recent_pnl': request.get('recent_pnl', 0.0),
            'streak': request.get('streak', 0)
        }
        
        # Get neural network prediction (SAME AS BACKTEST)
        result = get_neural_prediction(state, signal)
        
        logger.info(f"🧠 Neural ML: {symbol} {signal} @ {price}, RSI={rsi:.1f} → {result['reason']}")
        
        return {
            "ml_confidence": result['confidence'],
            "confidence": result['confidence'],
            "should_trade": result['should_trade'],
            "size_multiplier": result['size_multiplier'],
            "reason": result['reason'],
            "should_take": result['should_trade'],
            "action": signal if result['should_trade'] else "NONE",
            "model_version": "v6.0-neural-network",
            "model_used": result['model_used'],
            "threshold": result.get('threshold', 0.5),
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error calculating ML confidence: {e}")
        # Return neutral confidence on error
        return {
            "ml_confidence": 0.5,
            "confidence": 0.5,
            "should_trade": False,
            "size_multiplier": 1.0,
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
        
        # PRODUCTION OPTIMIZATION: Check Redis cache first
        # Same market conditions = same answer (60 second cache)
        # Reduces DB load for multi-user production
        cache_key = f"signal_rl:{signal}:{int(rsi)}:{int(vix)}:{int(volume_ratio*100)}"
        redis_mgr = get_redis()
        
        if redis_mgr:
            cached = redis_mgr.get(cache_key)
            if cached:
                logger.info(f"⚡ Cache HIT: {cache_key}")
                return cached
        
        # Calculate VWAP distance
        vwap_distance = abs(entry_price - vwap) / vwap if vwap > 0 else 0
        
        # Get advanced ML confidence with pattern matching
        result = calculate_signal_confidence(
            all_experiences=signal_experiences,
            vwap_distance=vwap_distance,
            rsi=rsi,
            signal=signal,
            current_time=datetime.now(pytz.UTC),
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
        
        response = {
            "take_trade": result['should_take'],
            "confidence": result['confidence'],
            "win_rate": result['win_rate'],
            "sample_size": result['sample_size'],
            "avg_pnl": result['avg_pnl'],
            "reason": result['reason'],
            "risk_level": risk_level,
            "model_version": "v5.0-advanced-pattern-matching",
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }
        
        # Cache the result for 60 seconds (multi-user optimization)
        if redis_mgr:
            redis_mgr.set(cache_key, response, ex=60)
        
        return response
        
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


def calculate_advanced_confidence(
    winner_experiences: List,
    loser_experiences: List, 
    recency_hours: int = 168, 
    streak: int = 0, 
    current_vix: float = 15.0, 
    volume_ratio: float = 1.0
) -> Dict:
    """
    DUAL PATTERN MATCHING: Calculate confidence by comparing to WINNERS and LOSERS
    
    This is the UPGRADED Feature 3 - learns from ALL experiences, not just winners.
    
    Formula: confidence = winner_similarity - loser_penalty
    
    Args:
        winner_experiences: Similar past winning trades (teaches what TO do)
        loser_experiences: Similar past losing trades (teaches what to AVOID)
        recency_hours: How many hours back to weight more heavily (default 1 week)
        streak: Current win/loss streak (positive = wins, negative = losses)
        current_vix: Current VIX level for volatility adjustment
        volume_ratio: Current volume / average volume ratio
    
    Returns:
        Dict with confidence, win_rate, sample_size, avg_pnl
    """
    total_samples = len(winner_experiences) + len(loser_experiences)
    
    if total_samples == 0:
        return {
            'confidence': 0.5,
            'win_rate': 0.5,
            'sample_size': 0,
            'avg_pnl': 0.0,
            'reason': 'No similar past experiences found'
        }
    
    # Calculate win rate
    win_rate = len(winner_experiences) / total_samples if total_samples > 0 else 0.5
    
    # Calculate win rate
    win_rate = len(winner_experiences) / total_samples if total_samples > 0 else 0.5
    
    # DUAL PATTERN MATCHING: Calculate average similarity to winners
    now = datetime.now(pytz.UTC)
    winner_similarity_sum = 0
    winner_weighted_total = 0
    
    for exp in winner_experiences:
        # Get similarity score (set during filtering)
        similarity = exp.get('similarity_score', 0.7)
        
        # Calculate recency weight
        if 'timestamp' in exp:
            try:
                exp_time = datetime.fromisoformat(exp['timestamp']) if isinstance(exp['timestamp'], str) else exp['timestamp']
                hours_ago = (now - exp_time).total_seconds() / 3600
                recency_weight = max(0.5, 1.0 - (hours_ago / recency_hours) * 0.5)
            except:
                recency_weight = 0.7
        else:
            recency_weight = 0.7
        
        # Apply quality score weight
        quality_weight = exp.get('quality_score', 0.5)
        
        # Combined weight
        total_weight = recency_weight * quality_weight
        
        winner_weighted_total += total_weight
        winner_similarity_sum += similarity * total_weight
    
    # Average winner similarity (0.0-1.0)
    avg_winner_similarity = winner_similarity_sum / winner_weighted_total if winner_weighted_total > 0 else 0.5
    
    # DUAL PATTERN MATCHING: Calculate average similarity to losers
    loser_similarity_sum = 0
    loser_weighted_total = 0
    
    for exp in loser_experiences:
        # Get similarity score
        similarity = exp.get('similarity_score', 0.7)
        
        # Calculate recency weight
        if 'timestamp' in exp:
            try:
                exp_time = datetime.fromisoformat(exp['timestamp']) if isinstance(exp['timestamp'], str) else exp['timestamp']
                hours_ago = (now - exp_time).total_seconds() / 3600
                recency_weight = max(0.5, 1.0 - (hours_ago / recency_hours) * 0.5)
            except:
                recency_weight = 0.7
        else:
            recency_weight = 0.7
        
        # Apply quality score weight
        quality_weight = exp.get('quality_score', 0.5)
        
        # Combined weight
        total_weight = recency_weight * quality_weight
        
        loser_weighted_total += total_weight
        loser_similarity_sum += similarity * total_weight
    
    # Average loser similarity (0.0-1.0)
    avg_loser_similarity = loser_similarity_sum / loser_weighted_total if loser_weighted_total > 0 else 0.0
    
    # DUAL PATTERN FORMULA: confidence = winner_similarity - loser_penalty
    # If similar to winners but NOT similar to losers → HIGH confidence
    # If similar to both winners and losers → MODERATE confidence
    # If similar to losers but NOT winners → LOW confidence (reject)
    loser_penalty = avg_loser_similarity * 0.5  # Penalize 50% of loser similarity
    base_confidence = avg_winner_similarity - loser_penalty
    
    # Clamp to reasonable range
    base_confidence = max(0.0, min(0.95, base_confidence))
    
    # Clamp to reasonable range
    base_confidence = max(0.0, min(0.95, base_confidence))
    
    # Calculate average P&L from ALL experiences (winners + losers)
    all_experiences = winner_experiences + loser_experiences
    total_pnl = sum(exp.get('reward', exp.get('pnl', 0)) for exp in all_experiences)
    avg_pnl = total_pnl / len(all_experiences) if all_experiences else 0.0
    
    # Start with base dual-pattern confidence
    confidence = base_confidence
    
    # Sample size adjustment (more samples = more confidence in the estimate)
    if total_samples < 10:
        confidence *= 0.7  # Low confidence with few samples
    elif total_samples < 30:
        confidence *= 0.85  # Medium confidence
    # else: Full confidence with 30+ samples
    
    # P&L quality adjustment (reward consistently profitable setups)
    if avg_pnl > 50:
        confidence = min(confidence + 0.1, 0.95)  # Boost for very profitable setups
    elif avg_pnl < -20:
        confidence *= 0.7  # Penalize consistently losing setups
    
    # Calculate position size multiplier (0.25x-2.0x)
    # Base multiplier from confidence
    size_mult = max(0.25, min(2.0, confidence * 1.5))  # 0.25-1.0 range for low confidence
    
    # Adjust for streak (±25%)
    if streak > 0:  # Win streak
        streak_bonus = min(0.25, streak * 0.05)  # Max +25%
        size_mult *= (1.0 + streak_bonus)
    elif streak < 0:  # Loss streak
        streak_penalty = min(0.25, abs(streak) * 0.05)  # Max -25%
        size_mult *= (1.0 - streak_penalty)
    
    # Adjust for volatility (VIX-based)
    if current_vix > 25:  # High volatility
        size_mult *= 0.8  # -20% in high vol
    elif current_vix < 12:  # Low volatility
        size_mult *= 1.15  # +15% in low vol
    
    # Adjust for volume (volume_ratio-based)
    if volume_ratio > 2.0:  # High volume
        size_mult *= 1.1  # +10%
    elif volume_ratio < 0.5:  # Low volume
        size_mult *= 0.9  # -10%
    
    # Final clamp
    size_mult = max(0.25, min(2.0, size_mult))
    
    # Generate reason string with dual-pattern details
    reason = (
        f"Dual Pattern: {len(winner_experiences)}W/{len(loser_experiences)}L "
        f"(win_sim={avg_winner_similarity:.2f}, lose_sim={avg_loser_similarity:.2f}, "
        f"penalty={loser_penalty:.2f}) → {int(win_rate * 100)}% WR, avg ${avg_pnl:.0f}"
    )
    
    return {
        'confidence': min(confidence, 0.95),  # Cap at 95%
        'win_rate': win_rate,
        'sample_size': total_samples,
        'avg_pnl': avg_pnl,
        'reason': reason,
        'size_multiplier': round(size_mult, 2),
        'winner_similarity': round(avg_winner_similarity, 3),
        'loser_similarity': round(avg_loser_similarity, 3),
        'loser_penalty': round(loser_penalty, 3)
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
        current_time = datetime.now(pytz.UTC)
    
    current_time_only = current_time.time()
    current_day_of_week = current_time.weekday()  # 0=Monday, 6=Sunday
    current_hour = current_time.hour  # 0-23
    
    # STEP 1: Filter ALL experiences by context (signal type, VIX, day of week)
    # NO arbitrary limits - let similarity scoring find what's relevant
    logger.info(f"🧠 RL Pattern Matching: Analyzing {len(all_experiences)} total experiences for {signal} signal")
    
    contextual_experiences = filter_experiences_by_context(
        experiences=all_experiences,  # Use ALL experiences
        signal_type=signal,
        current_day=current_day_of_week,
        current_vix=current_vix,
        min_similarity=similarity_threshold
    )
    
    logger.info(f"   → Found {len(contextual_experiences)} contextually relevant experiences (after context filter)")
    
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
    
    # STEP 2.5: DUAL PATTERN MATCHING - Separate winners and losers
    winner_experiences = []
    loser_experiences = []
    
    for exp in similar_experiences:
        # Get PnL from either 'reward' (nested RL format) or 'pnl' (flat format)
        pnl = exp.get('reward', exp.get('pnl', 0))
        outcome = exp.get('outcome', '')
        
        if outcome == 'WIN' or pnl > 0:
            winner_experiences.append(exp)
        else:
            loser_experiences.append(exp)
    
    logger.info(f"   → Dual Pattern: {len(winner_experiences)} winners, {len(loser_experiences)} losers")
    
    # STEP 2.75: SIGNAL-EXIT CROSS-TALK - Check exit performance for similar signals
    # If similar signals consistently hit stop loss, reduce confidence
    exit_penalty = 0.0
    try:
        from database import DatabaseManager, ExitExperience
        db_manager = DatabaseManager()
        session = db_manager.get_session()
        
        # Query exit experiences with similar market conditions
        similar_exits = []
        all_exits = session.query(ExitExperience).limit(1000).all()  # Last 1000 exits
        
        for exit_exp in all_exits:
            exit_data = exit_exp.to_dict()
            exit_market = exit_data.get('market_state', {})
            exit_outcome = exit_data.get('outcome', {})
            
            # Check if exit had similar signal conditions
            exit_rsi = exit_market.get('rsi', 50)
            exit_vwap_dist = exit_market.get('vwap_distance', 0)
            
            # Calculate similarity to current signal
            rsi_diff = abs(exit_rsi - rsi)
            vwap_diff = abs(exit_vwap_dist - vwap_distance)
            
            if rsi_diff <= 10 and vwap_diff <= 2.0:
                # Similar signal conditions
                similar_exits.append({
                    'win': exit_outcome.get('win', False),
                    'pnl': exit_outcome.get('pnl', 0.0),
                    'exit_reason': exit_outcome.get('exit_reason', 'UNKNOWN')
                })
        
        if len(similar_exits) >= 10:
            # Calculate exit performance
            exit_wins = sum(1 for e in similar_exits if e['win'])
            exit_win_rate = exit_wins / len(similar_exits)
            stop_loss_count = sum(1 for e in similar_exits if 'STOP' in e['exit_reason'].upper())
            stop_loss_rate = stop_loss_count / len(similar_exits)
            
            # Apply penalty if exits are poor
            if exit_win_rate < 0.50:
                # Poor exit performance → reduce signal confidence
                exit_penalty = (0.50 - exit_win_rate) * 0.3  # Max 15% penalty
                logger.info(f"   ⚠️  Signal-Exit Cross-Talk: {len(similar_exits)} similar exits, {exit_win_rate*100:.0f}% WR, {stop_loss_rate*100:.0f}% stops → penalty={exit_penalty:.3f}")
            else:
                logger.info(f"   ✅ Signal-Exit Cross-Talk: {len(similar_exits)} similar exits, {exit_win_rate*100:.0f}% WR (good exits)")
        
        session.close()
    except Exception as e:
        logger.warning(f"   Signal-Exit cross-talk failed: {e}")
    
    # STEP 3: Calculate advanced confidence using DUAL pattern matching
    result = calculate_advanced_confidence(
        winner_experiences=winner_experiences,
        loser_experiences=loser_experiences,
        recency_hours=168,
        streak=streak,
        current_vix=current_vix,
        volume_ratio=volume_ratio
    )
    
    # Apply exit penalty to final confidence
    if exit_penalty > 0:
        original_confidence = result['confidence']
        result['confidence'] = max(0.0, result['confidence'] - exit_penalty)
        result['exit_penalty'] = round(exit_penalty, 3)
        result['reason'] += f" | Exit penalty: -{exit_penalty:.1%} (poor exit performance on similar signals)"
        logger.info(f"   Applied exit penalty: {original_confidence:.2%} → {result['confidence']:.2%}")
    
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
            "saved_at": datetime.now(pytz.UTC).isoformat(),
            "experience_id": f"{user_id}_{symbol}_{datetime.now(pytz.UTC).timestamp()}"
        }
        
        # Store in SHARED array (everyone contributes to same strategy learning)
        signal_experiences.append(experience)
        
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
                    entry_time=datetime.fromisoformat(trade['entry_time']) if trade.get('entry_time') else datetime.now(pytz.UTC),
                    exit_time=datetime.fromisoformat(trade['exit_time']) if trade.get('exit_time') else datetime.now(pytz.UTC),
                    exit_reason=trade.get('exit_reason', 'unknown'),
                    confidence_score=trade.get('confidence', 0.0)
                )
                db.add(trade_record)
                
                # **NEW: Track RL experiences for signal and exit WITH CONTEXT**
                pnl = trade['pnl']
                outcome = 'WIN' if pnl > 0 else 'LOSS'
                
                # Extract ALL context data (with defaults if not provided)
                rsi = trade.get('entry_rsi', trade.get('rsi', 50.0))
                vwap_distance = trade.get('vwap_distance', 0.0)
                vix = trade.get('vix', 15.0)
                atr = trade.get('volatility', trade.get('atr', 0.0))
                volume_ratio = trade.get('volume_ratio', 1.0)
                recent_pnl = trade.get('recent_pnl', 0.0)
                streak = trade.get('streak', 0)
                entry_price = trade.get('entry_price', 0.0)
                vwap = trade.get('entry_vwap', trade.get('vwap', 0.0))
                price = trade.get('price', entry_price)
                
                # Calculate time context
                entry_time = datetime.fromisoformat(trade['entry_time']) if trade.get('entry_time') else datetime.now(pytz.UTC)
                day_of_week = entry_time.weekday()  # 0=Monday, 6=Sunday
                hour_of_day = entry_time.hour  # 0-23
                
                # Signal experience (entry decision) WITH FULL 13-FEATURE CONTEXT
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
                    # All 13 pattern matching features:
                    rsi=rsi,
                    vwap_distance=vwap_distance,
                    vix=vix,
                    day_of_week=day_of_week,
                    hour_of_day=hour_of_day,
                    atr=atr,
                    volume_ratio=volume_ratio,
                    recent_pnl=recent_pnl,
                    streak=streak,
                    entry_price=entry_price,
                    vwap=vwap,
                    price=price,
                    side=trade['side']
                )
                db.add(signal_exp)
                
                # EXIT experiences are saved separately via /api/ml/save_exit_experience
                # (Not saved here to avoid duplication)
                
                db.commit()
                logger.debug(f"✅ Trade & Signal RL experience saved to database for {user_id}")
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
            "timestamp": datetime.now(pytz.UTC).isoformat(),
            "took_trade": False  # Always false for rejections
        }
        
        # Store in shared experience pool with negative reward (represents opportunity cost)
        # The RL will learn: "Was skipping this signal the right decision?"
        signal_experiences.append(experience)
        
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
    
    Saves to TWO tables:
    1. RLExperience: 9-feature context for pattern matching
    2. ExitExperience: Full exit parameters for regime-based learning
    """
    try:
        from database import DatabaseManager, RLExperience, ExitExperience
        from datetime import datetime
        import json
        
        db_manager = DatabaseManager()
        session = db_manager.get_session()
        
        try:
            # Extract data
            market_state = experience.get('market_state', {})
            outcome = experience.get('outcome', {})
            exit_params = experience.get('exit_params', {})
            regime = experience.get('regime', 'UNKNOWN')
            situation = experience.get('situation', {})
            partial_exits = experience.get('partial_exits', [])
            
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
            
            # 1. Create RLExperience record for exit pattern matching (experience_type='EXIT')
            exit_rl_exp = RLExperience(
                experience_type='EXIT',
                user_id=default_user.id,
                account_id='backtest',
                symbol='ES',
                signal_type=outcome.get('side', 'long').upper(),
                outcome='WIN' if outcome.get('win', False) else 'LOSS',
                pnl=float(outcome.get('pnl', 0.0)),
                confidence_score=None,
                quality_score=min(abs(outcome.get('pnl', 0.0)) / 100.0, 1.0),
                
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
                
                entry_price=None,
                vwap=None,
                price=None,
                side=outcome.get('side', 'long'),
                
                timestamp=datetime.fromisoformat(experience.get('timestamp', datetime.now(pytz.UTC).isoformat()))
            )
            session.add(exit_rl_exp)
            
            # 2. Create ExitExperience record for full exit parameter learning
            full_exit_exp = ExitExperience(
                regime=regime,
                exit_params_json=json.dumps(exit_params),
                outcome_json=json.dumps(outcome),
                situation_json=json.dumps(situation),
                market_state_json=json.dumps(market_state),
                partial_exits_json=json.dumps(partial_exits),
                quality_score=min(abs(outcome.get('pnl', 0.0)) / 100.0, 1.0),
                timestamp=datetime.fromisoformat(experience.get('timestamp', datetime.now(pytz.UTC).isoformat()))
            )
            session.add(full_exit_exp)
            
            session.commit()
            
            # Get updated counts
            rl_exit_count = session.query(RLExperience).filter(
                RLExperience.experience_type == 'EXIT'
            ).count()
            
            full_exit_count = session.query(ExitExperience).count()
            
            logger.info(f"✅ Saved EXIT to BOTH tables: RLExperience ({rl_exit_count:,}) + ExitExperience ({full_exit_count:,})")
            
            return {
                "saved": True,
                "total_exit_experiences": rl_exit_count,
                "total_full_exit_experiences": full_exit_count,
                "message": f"🌍 Exit saved to BOTH tables (RL pattern matching + full params)"
            }
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
        
    except Exception as e:
        logger.error(f"❌ Error saving exit experience to PostgreSQL: {e}")
        return {"saved": False, "error": str(e)}


@app.post("/api/ml/save_experience")
async def save_ml_experience(experience: dict):
    """
    Save COMPLETE ML experience to PostgreSQL ml_experiences table (JSONB).
    SCALABLE: Handles high-volume inserts with connection pooling and fast commits.
    Supports signal experiences, exit experiences, AND ghost trades.
    ALL fields stored as-is in JSONB format - no schema limitations!
    
    Request: {
        user_id: str,
        symbol: str,
        experience_type: str,  # 'signal', 'exit', 'ghost_trade', 'ghost_exit'
        rl_state: dict,  # Full state (33 fields for signals, 64 for exits)
        outcome: dict,   # Full outcome data
        quality_score: float,  # 0-1
        timestamp: str  # ISO format
    }
    
    PERFORMANCE: ~1000 writes/sec capacity, optimized for multi-user scale
    """
    try:
        from database import DatabaseManager, MLExperience
        from datetime import datetime
        
        db_manager = DatabaseManager()
        session = db_manager.get_session()
        
        try:
            # Extract fields (with validation)
            user_id = experience.get('user_id', 'unknown')
            if not user_id or user_id == 'unknown':
                logger.warning("Missing user_id in experience - using 'unknown'")
            
            symbol = experience.get('symbol', 'ES')
            exp_type = experience.get('experience_type', 'signal')
            
            # Validate experience type
            valid_types = ['signal', 'exit', 'ghost_trade', 'ghost_exit']
            if exp_type not in valid_types:
                logger.warning(f"Invalid experience_type '{exp_type}', defaulting to 'signal'")
                exp_type = 'signal'
            
            rl_state = experience.get('rl_state', {})
            outcome = experience.get('outcome', {})
            quality_score = experience.get('quality_score', 1.0)
            
            # Handle 'experience' wrapper (some payloads nest data in 'experience' key)
            if 'experience' in experience and isinstance(experience['experience'], dict):
                nested = experience['experience']
                rl_state = rl_state or nested.get('rl_state', {})
                outcome = outcome or nested.get('outcome', {})
            
            timestamp_str = experience.get('timestamp', datetime.now(pytz.UTC).isoformat())
            
            # Parse timestamp
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except:
                timestamp = datetime.now(pytz.UTC)
            
            # Create ML experience record (JSONB storage - no field limits!)
            ml_exp = MLExperience(
                user_id=user_id,
                symbol=symbol,
                experience_type=exp_type,
                rl_state=rl_state,  # Stored as JSONB - 33+ signal features or 64+ exit features
                outcome=outcome,    # Stored as JSONB - complete outcome data
                quality_score=quality_score,
                timestamp=timestamp
            )
            session.add(ml_exp)
            session.commit()
            
            # Efficient count - only query if needed (skip for high volume)
            # For production scale, consider caching this count or removing it
            try:
                total_count = session.query(MLExperience).filter(
                    MLExperience.experience_type == exp_type
                ).count()
            except:
                total_count = 0  # Don't fail if count query fails
            
            logger.info(f"✅ Saved {exp_type} experience from {user_id[:8]}... (total: {total_count:,})")
            
            return {
                "saved": True,
                "experience_type": exp_type,
                "total_experiences": total_count,
                "message": f"Experience saved to cloud database"
            }
            
        except Exception as e:
            session.rollback()
            logger.error(f"❌ DB error saving experience: {e}")
            raise e
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"❌ Error saving ML experience: {e}")
        return {"saved": False, "error": str(e)}


@app.post("/api/ml/save_experiences_batch")
async def save_experiences_batch(experiences: list[dict]):
    """
    BATCH INSERT for high-volume ML experience saves.
    5-10x FASTER than individual inserts for multi-user scale.
    
    Request: [
        {user_id, symbol, experience_type, rl_state, outcome, quality_score, timestamp},
        {user_id, symbol, experience_type, rl_state, outcome, quality_score, timestamp},
        ...
    ]
    
    PERFORMANCE: Handles 50-100 experiences per batch, reduces DB round trips.
    SCALABILITY: Essential for 1000+ users with 5 ghost trades each.
    """
    try:
        from database import DatabaseManager, MLExperience
        from datetime import datetime
        
        if not experiences or len(experiences) == 0:
            return {"saved": 0, "errors": 0, "message": "Empty batch"}
        
        db_manager = DatabaseManager()
        session = db_manager.get_session()
        
        saved_count = 0
        error_count = 0
        
        try:
            # Prepare batch insert data
            batch_data = []
            for exp in experiences:
                try:
                    # Extract and validate fields
                    user_id = exp.get('user_id', 'unknown')
                    symbol = exp.get('symbol', 'ES')
                    exp_type = exp.get('experience_type', 'signal')
                    
                    # Validate experience type
                    valid_types = ['signal', 'exit', 'ghost_trade', 'ghost_exit']
                    if exp_type not in valid_types:
                        exp_type = 'signal'
                    
                    rl_state = exp.get('rl_state', {})
                    outcome = exp.get('outcome', {})
                    quality_score = exp.get('quality_score', 1.0)
                    timestamp_str = exp.get('timestamp', datetime.now(pytz.UTC).isoformat())
                    
                    # Parse timestamp
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    except:
                        timestamp = datetime.now(pytz.UTC)
                    
                    # Add to batch
                    batch_data.append({
                        'user_id': user_id,
                        'symbol': symbol,
                        'experience_type': exp_type,
                        'rl_state': rl_state,
                        'outcome': outcome,
                        'quality_score': quality_score,
                        'timestamp': timestamp
                    })
                    
                except Exception as e:
                    logger.warning(f"⚠️ Skipped invalid experience in batch: {e}")
                    error_count += 1
            
            # Bulk insert (MUCH faster than individual inserts)
            if batch_data:
                session.bulk_insert_mappings(MLExperience, batch_data)
                session.commit()
                saved_count = len(batch_data)
                
                logger.info(f"✅ Batch saved {saved_count} experiences ({error_count} errors)")
            
            return {
                "saved": saved_count,
                "errors": error_count,
                "message": f"Batch insert completed: {saved_count} saved, {error_count} failed"
            }
            
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Batch insert failed: {e}")
            return {"saved": 0, "errors": len(experiences), "error": str(e)}
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"❌ Batch save error: {e}")
        return {"saved": 0, "errors": 0, "error": str(e)}


@app.post("/api/ml/get_adaptive_exit_params")
async def get_adaptive_exit_params(request: dict):
    """
    REAL-TIME exit parameter recommendations based on current market conditions.
    Queries 3,214+ exit experiences to find similar situations and returns optimal params.
    
    Called MID-TRADE to adapt exit strategy as market conditions change.
    
    Request:
        {
            "regime": "HIGH_VOL_CHOPPY",
            "market_state": {
                "rsi": 65.5,
                "atr": 3.2,
                "vwap_distance": 1.5,
                "volume_ratio": 1.8,
                "hour": 10,
                "day_of_week": 2,
                "streak": 2,
                "recent_pnl": 150.0,
                "vix": 18.5
            },
            "position": {
                "side": "long",
                "duration_minutes": 12.5,
                "unrealized_pnl": 225.0,
                "entry_price": 5950.0,
                "r_multiple": 2.3
            },
            "entry_confidence": 0.85  # From signal RL
        }
    
    Returns:
        {
            "success": true,
            "params": {
                "breakeven_threshold_ticks": 9,
                "breakeven_offset_ticks": 2,
                "trailing_distance_ticks": 10,
                "partial_1_r": 2.0,
                "partial_1_pct": 0.50,
                "stop_mult": 3.8
            },
            "similar_exits_analyzed": 342,
            "avg_pnl_similar": 187.50,
            "win_rate_similar": 0.68,
            "recommendation": "TIGHTEN (68% win rate in 342 similar choppy exits, avg $188 P&L)"
        }
    """
    try:
        from database import DatabaseManager, ExitExperience
        import json
        import statistics
        
        db_manager = DatabaseManager()
        session = db_manager.get_session()
        
        try:
            # Extract request data
            regime = request.get('regime', 'NORMAL')
            market_state = request.get('market_state', {})
            position = request.get('position', {})
            entry_confidence = request.get('entry_confidence', 0.75)
            
            # Query ALL exit experiences from PostgreSQL
            # NO arbitrary limits - similarity scoring will find what's relevant
            all_exits = session.query(ExitExperience).order_by(ExitExperience.timestamp.desc()).all()
            
            logger.info(f"🎯 Exit RL: Analyzing {len(all_exits)} total exit experiences for {regime} regime")
            
            if len(all_exits) == 0:
                # No experiences yet - return defaults
                return {
                    "success": True,
                    "params": {
                        "breakeven_threshold_ticks": 8,
                        "breakeven_offset_ticks": 1,
                        "trailing_distance_ticks": 8,
                        "partial_1_r": 2.0,
                        "partial_1_pct": 0.50,
                        "stop_mult": 3.6
                    },
                    "similar_exits_analyzed": 0,
                    "recommendation": "Using defaults (no exit experiences yet)"
                }
            
            # Find similar exit situations (same regime, similar market conditions)
            similar_exits = []
            for exp in all_exits:
                exit_data = exp.to_dict()
                exp_regime = exit_data.get('regime', 'NORMAL')
                exp_market_state = exit_data.get('market_state', {})
                exp_outcome = exit_data.get('outcome', {})
                
                # Match regime (exact match or similar)
                regime_match = False
                if exp_regime == regime:
                    regime_match = True
                elif "HIGH_VOL" in regime and "HIGH_VOL" in exp_regime:
                    regime_match = True
                elif "LOW_VOL" in regime and "LOW_VOL" in exp_regime:
                    regime_match = True
                elif "CHOPPY" in regime and "CHOPPY" in exp_regime:
                    regime_match = True
                elif "TRENDING" in regime and "TRENDING" in exp_regime:
                    regime_match = True
                
                if not regime_match:
                    continue
                
                # Calculate similarity score (0.0 - 1.0)
                similarity = 0.0
                weights = 0.0
                
                # RSI similarity (most important for exits)
                if 'rsi' in market_state and 'rsi' in exp_market_state:
                    rsi_diff = abs(market_state['rsi'] - exp_market_state['rsi'])
                    rsi_sim = max(0, 1.0 - (rsi_diff / 50.0))  # 50 RSI points = 0 similarity
                    similarity += rsi_sim * 3.0  # Weight 3x
                    weights += 3.0
                
                # ATR similarity (volatility)
                if 'atr' in market_state and 'atr' in exp_market_state:
                    atr_diff = abs(market_state['atr'] - exp_market_state['atr'])
                    atr_sim = max(0, 1.0 - (atr_diff / 5.0))  # 5 ATR points = 0 similarity
                    similarity += atr_sim * 2.0  # Weight 2x
                    weights += 2.0
                
                # VWAP distance similarity
                if 'vwap_distance' in market_state and 'vwap_distance' in exp_market_state:
                    vwap_diff = abs(market_state['vwap_distance'] - exp_market_state['vwap_distance'])
                    vwap_sim = max(0, 1.0 - (vwap_diff / 3.0))
                    similarity += vwap_sim * 1.5
                    weights += 1.5
                
                # Hour of day (time-based patterns)
                if 'hour' in market_state and 'hour' in exp_market_state:
                    hour_diff = abs(market_state['hour'] - exp_market_state['hour'])
                    hour_sim = max(0, 1.0 - (hour_diff / 12.0))
                    similarity += hour_sim * 1.0
                    weights += 1.0
                
                # Normalize similarity
                if weights > 0:
                    similarity /= weights
                
                # Only include if similarity >= 0.60 (60% match)
                if similarity >= 0.60:
                    similar_exits.append({
                        'experience': exit_data,
                        'similarity': similarity,
                        'pnl': exp_outcome.get('pnl', 0.0),
                        'win': exp_outcome.get('win', False),
                        'exit_params': exit_data.get('exit_params', {})
                    })
            
            # Sort by similarity (most similar first)
            similar_exits.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Take top 500 most similar (or all if fewer)
            similar_exits = similar_exits[:500]
            
            # DUAL PATTERN MATCHING: Separate winners and losers
            winner_exits = [e for e in similar_exits if e['win']]
            loser_exits = [e for e in similar_exits if not e['win']]
            
            logger.info(f"📊 Exit Dual Pattern: {len(winner_exits)} winners, {len(loser_exits)} losers from {len(similar_exits)} similar exits")
            
            if len(similar_exits) == 0:
                # No similar exits found - return defaults
                return {
                    "success": True,
                    "params": {
                        "breakeven_threshold_ticks": 8,
                        "breakeven_offset_ticks": 1,
                        "trailing_distance_ticks": 8,
                        "partial_1_r": 2.0,
                        "partial_1_pct": 0.50,
                        "stop_mult": 3.6
                    },
                    "similar_exits_analyzed": 0,
                    "recommendation": f"Using defaults (no similar {regime} exits found)"
                }
            
            # Calculate statistics from similar exits
            pnls = [e['pnl'] for e in similar_exits]
            wins = [e['win'] for e in similar_exits]
            win_rate = sum(wins) / len(wins) if wins else 0.0
            avg_pnl = statistics.mean(pnls) if pnls else 0.0
            
            # DUAL PATTERN MATCHING: Learn from winners, avoid loser patterns
            # Winners teach what params WORK, losers teach what params to AVOID
            
            # Extract parameters from WINNERS (what to do)
            winner_params = {
                'breakeven_mult': [],
                'trailing_mult': [],
                'stop_mult': [],
                'partial_1_r': [],
                'partial_1_pct': []
            }
            
            for exit in winner_exits:
                params = exit['exit_params']
                if 'breakeven_mult' in params:
                    winner_params['breakeven_mult'].append(params['breakeven_mult'])
                if 'trailing_mult' in params:
                    winner_params['trailing_mult'].append(params['trailing_mult'])
                if 'stop_mult' in params:
                    winner_params['stop_mult'].append(params['stop_mult'])
                if 'partial_1_r' in params:
                    winner_params['partial_1_r'].append(params['partial_1_r'])
                if 'partial_1_pct' in params:
                    winner_params['partial_1_pct'].append(params['partial_1_pct'])
            
            # Extract parameters from LOSERS (what to avoid)
            loser_params = {
                'breakeven_mult': [],
                'trailing_mult': [],
                'stop_mult': [],
                'partial_1_r': [],
                'partial_1_pct': []
            }
            
            for exit in loser_exits:
                params = exit['exit_params']
                if 'breakeven_mult' in params:
                    loser_params['breakeven_mult'].append(params['breakeven_mult'])
                if 'trailing_mult' in params:
                    loser_params['trailing_mult'].append(params['trailing_mult'])
                if 'stop_mult' in params:
                    loser_params['stop_mult'].append(params['stop_mult'])
                if 'partial_1_r' in params:
                    loser_params['partial_1_r'].append(params['partial_1_r'])
                if 'partial_1_pct' in params:
                    loser_params['partial_1_pct'].append(params['partial_1_pct'])
            
            # Calculate optimal parameters using DUAL PATTERN MATCHING
            # Strategy: Use winner medians, but AVOID loser medians
            optimal_params = {}
            
            for key in winner_params.keys():
                winner_values = winner_params[key]
                loser_values = loser_params[key]
                
                if winner_values and loser_values:
                    # Both winners and losers have data
                    winner_median = statistics.median(winner_values)
                    loser_median = statistics.median(loser_values)
                    
                    # If winner params and loser params are similar → unclear which is better
                    # If winner params differ from loser params → use winner params
                    param_diff = abs(winner_median - loser_median)
                    
                    if param_diff < 0.2:
                        # Too similar - can't tell which is better, use winner median
                        optimal_params[key] = winner_median
                        logger.info(f"   {key}: winners={winner_median:.2f}, losers={loser_median:.2f} (similar, using winner)")
                    else:
                        # Clear difference - strongly prefer winner params
                        optimal_params[key] = winner_median
                        logger.info(f"   {key}: winners={winner_median:.2f}, losers={loser_median:.2f} (diff={param_diff:.2f}, using winner)")
                        
                elif winner_values:
                    # Only winners have data
                    optimal_params[key] = statistics.median(winner_values)
                    logger.info(f"   {key}: only winners={optimal_params[key]:.2f}")
                elif loser_values:
                    # Only losers have data - avoid their params by using opposite/defaults
                    loser_median = statistics.median(loser_values)
                    if key == 'stop_mult':
                        # Losers had tight stops → use wider
                        optimal_params[key] = max(3.6, loser_median + 0.5)
                    elif key in ['breakeven_mult', 'trailing_mult']:
                        # Losers had loose params → use tighter
                        optimal_params[key] = max(0.8, loser_median - 0.2)
                    else:
                        optimal_params[key] = loser_median
                    logger.info(f"   {key}: only losers={loser_median:.2f}, adjusted to {optimal_params[key]:.2f}")
                else:
                    # No data - use defaults
                    if key == 'breakeven_mult':
                        optimal_params[key] = 1.0
                    elif key == 'trailing_mult':
                        optimal_params[key] = 1.0
                    elif key == 'stop_mult':
                        optimal_params[key] = 3.6
                    elif key == 'partial_1_r':
                        optimal_params[key] = 2.0
                    elif key == 'partial_1_pct':
                        optimal_params[key] = 0.50
            
            # Convert multipliers to actual tick values (assuming base values)
            base_breakeven = 8
            base_trailing = 8
            
            recommended_params = {
                "breakeven_threshold_ticks": int(base_breakeven * optimal_params.get('breakeven_mult', 1.0)),
                "breakeven_offset_ticks": 2,  # Fixed at 2 ticks
                "trailing_distance_ticks": int(base_trailing * optimal_params.get('trailing_mult', 1.0)),
                "partial_1_r": optimal_params.get('partial_1_r', 2.0),
                "partial_1_pct": optimal_params.get('partial_1_pct', 0.50),
                "stop_mult": optimal_params.get('stop_mult', 3.6)
            }
            
            # Generate recommendation message with dual pattern info
            if win_rate >= 0.65:
                action = "HOLD"
            elif win_rate >= 0.55:
                action = "NORMAL"
            else:
                action = "TIGHTEN"
            
            recommendation = f"{action} - Dual Pattern: {len(winner_exits)}W/{len(loser_exits)}L ({win_rate*100:.0f}% WR, avg ${avg_pnl:.0f})"
            
            logger.info(f"🎯 Exit RL Dual Pattern: {len(winner_exits)}W/{len(loser_exits)}L → {action} ({win_rate*100:.0f}% WR, avg ${avg_pnl:.0f})")
            
            return {
                "success": True,
                "params": recommended_params,
                "similar_exits_analyzed": len(similar_exits),
                "winner_exits": len(winner_exits),
                "loser_exits": len(loser_exits),
                "avg_pnl_similar": round(avg_pnl, 2),
                "win_rate_similar": round(win_rate, 3),
                "recommendation": recommendation,
                "dual_pattern": True,  # Flag to indicate dual pattern matching is active
                "regime": regime
            }
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"❌ Error getting adaptive exit params: {e}")
        return {
            "success": False,
            "params": {
                "breakeven_threshold_ticks": 8,
                "breakeven_offset_ticks": 1,
                "trailing_distance_ticks": 8,
                "partial_1_r": 2.0,
                "partial_1_pct": 0.50,
                "stop_mult": 3.6
            },
            "similar_exits_analyzed": 0,
            "recommendation": f"Using defaults (error: {str(e)})"
        }


@app.post("/api/ml/predict_exit_params")
async def predict_exit_params(request: dict):
    """
    NEURAL NETWORK exit parameter prediction (CLOUD-BASED).
    Uses trained exit neural network to predict optimal exit parameters in real-time.
    
    Called EVERY TICK during trades for dynamic exit management.
    
    Request:
        {
            "market_regime": "HIGH_VOL_TRENDING",
            "rsi": 65.5,
            "volume_ratio": 1.8,
            "atr": 3.2,
            "vix": 18.5,
            "volatility_regime_change": false,
            "entry_confidence": 0.85,
            "side": "long",
            "session": 0,
            "bid_ask_spread_ticks": 1.0,
            "hour": 10,
            "day_of_week": 2,
            "duration_bars": 12,
            "time_in_breakeven_bars": 0,
            "bars_until_breakeven": 5,
            "mae": -25.0,
            "mfe": 87.50,
            "max_r_achieved": 1.5,
            "min_r_achieved": -0.2,
            "r_multiple": 1.2,
            "breakeven_activated": false,
            "trailing_activated": false,
            "exit_param_update_count": 3,
            "stop_adjustment_count": 1,
            "bars_until_trailing": 8,
            "current_pnl": 50.0,
            "entry_atr": 2.8,
            "avg_atr_during_trade": 3.0,
            "profit_drawdown_from_peak": 12.50,
            "high_volatility_bars": 2,
            "wins_in_last_5_trades": 3,
            "losses_in_last_5_trades": 2,
            "minutes_to_close": 240
        }
    
    Returns:
        {
            "success": true,
            "exit_params": {
                "breakeven_threshold_ticks": 7.2,
                "trailing_distance_ticks": 9.5,
                "stop_mult": 3.8,
                "partial_1_r": 2.1,
                "partial_2_r": 3.2,
                "partial_3_r": 5.5
            },
            "model_version": "v2_45features",
            "prediction_time_ms": 12.3
        }
    """
    try:
        import time
        start_time = time.time()
        
        # Check if exit predictor is initialized
        if neural_exit_predictor is None:
            logger.error("❌ Exit neural network not initialized")
            return {
                "success": False,
                "error": "Exit predictor not available",
                "exit_params": {
                    "breakeven_threshold_ticks": 8.0,
                    "trailing_distance_ticks": 10.0,
                    "stop_mult": 3.5,
                    "partial_1_r": 2.0,
                    "partial_2_r": 3.0,
                    "partial_3_r": 5.0
                }
            }
        
        # Get neural network prediction
        exit_params = neural_exit_predictor.predict(request)
        
        prediction_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"🧠 Exit NN: BE={exit_params['breakeven_threshold_ticks']:.1f}t, "
                   f"Trail={exit_params['trailing_distance_ticks']:.1f}t, "
                   f"Partials={exit_params['partial_1_r']:.2f}R/{exit_params['partial_2_r']:.2f}R/{exit_params['partial_3_r']:.2f}R "
                   f"({prediction_time_ms:.1f}ms)")
        
        return {
            "success": True,
            "exit_params": exit_params,
            "model_version": "v2_45features",
            "prediction_time_ms": round(prediction_time_ms, 2)
        }
        
    except Exception as e:
        logger.error(f"❌ Exit prediction error: {e}")
        return {
            "success": False,
            "error": str(e),
            "exit_params": {
                "breakeven_threshold_ticks": 8.0,
                "trailing_distance_ticks": 10.0,
                "stop_mult": 3.5,
                "partial_1_r": 2.0,
                "partial_2_r": 3.0,
                "partial_3_r": 5.0
            }
        }


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
        "last_updated": datetime.now(pytz.UTC).isoformat()
    }

@app.get("/api/ml/db_stats")
async def get_database_stats():
    """
    Get actual PostgreSQL database statistics (not in-memory)
    Shows what's really saved in the cloud database
    """
    try:
        from database import DatabaseManager, MLExperience
        
        db_manager = DatabaseManager()
        session = db_manager.get_session()
        
        try:
            # Count by experience type
            signal_count = session.query(MLExperience).filter(
                MLExperience.experience_type == 'signal'
            ).count()
            
            exit_count = session.query(MLExperience).filter(
                MLExperience.experience_type == 'exit'
            ).count()
            
            ghost_count = session.query(MLExperience).filter(
                MLExperience.experience_type == 'ghost_trade'
            ).count()
            
            total_count = session.query(MLExperience).count()
            
            # Get sample record
            sample = session.query(MLExperience).first()
            sample_data = None
            if sample:
                import json
                # Handle both dict (PostgreSQL) and string (SQLite) formats
                rl_state = sample.rl_state
                outcome = sample.outcome
                if isinstance(rl_state, str):
                    rl_state = json.loads(rl_state)
                if isinstance(outcome, str):
                    outcome = json.loads(outcome)
                    
                sample_data = {
                    "id": sample.id,
                    "user_id": sample.user_id,
                    "symbol": sample.symbol,
                    "experience_type": sample.experience_type,
                    "quality_score": sample.quality_score,
                    "timestamp": sample.timestamp.isoformat() if sample.timestamp else None,
                    "rl_state_keys": list(rl_state.keys()) if rl_state else [],
                    "outcome_keys": list(outcome.keys()) if outcome else []
                }
            
            return {
                "database": "PostgreSQL",
                "total_experiences": total_count,
                "by_type": {
                    "signal": signal_count,
                    "exit": exit_count,
                    "ghost_trade": ghost_count
                },
                "sample_record": sample_data,
                "message": "Database query successful"
            }
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"❌ Error querying database: {e}")
        import traceback
        return {
            "database": "PostgreSQL",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "message": "Database query failed"
        }

@app.get("/api/admin/db_sample")
async def get_database_sample(limit: int = 5):
    """
    Get sample records from PostgreSQL database to verify data integrity
    Returns actual experience data (not just counts)
    """
    try:
        from database import DatabaseManager, MLExperience
        
        db_manager = DatabaseManager()
        session = db_manager.get_session()
        
        try:
            # Get sample records
            samples = session.query(MLExperience).limit(limit).all()
            
            sample_list = []
            for sample in samples:
                import json
                # Handle both dict (PostgreSQL) and string (SQLite) formats
                rl_state = sample.rl_state
                outcome = sample.outcome
                
                if isinstance(rl_state, str):
                    try:
                        rl_state = json.loads(rl_state)
                    except:
                        rl_state = {"error": "failed to parse", "raw": str(rl_state)[:100]}
                
                if isinstance(outcome, str):
                    try:
                        outcome = json.loads(outcome)
                    except:
                        outcome = {"error": "failed to parse", "raw": str(outcome)[:100]}
                
                sample_list.append({
                    "id": sample.id,
                    "user_id": sample.user_id,
                    "symbol": sample.symbol,
                    "experience_type": sample.experience_type,
                    "quality_score": sample.quality_score,
                    "timestamp": sample.timestamp.isoformat() if sample.timestamp else None,
                    "rl_state": rl_state,
                    "outcome": outcome
                })
            
            total_count = session.query(MLExperience).count()
            
            return {
                "database": "PostgreSQL",
                "total_experiences": total_count,
                "sample_count": len(sample_list),
                "samples": sample_list,
                "message": "Sample retrieved successfully"
            }
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"❌ Error sampling database: {e}")
        import traceback
        return {
            "database": "PostgreSQL",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "message": "Database sample failed"
        }

@app.get("/api/admin/db_all")
async def get_all_experiences(offset: int = 0, limit: int = 10000):
    """
    Get ALL experiences from PostgreSQL database with pagination
    Returns complete dataset for admin monitoring
    """
    try:
        from database import DatabaseManager, MLExperience
        
        db_manager = DatabaseManager()
        session = db_manager.get_session()
        
        try:
            # Get all records with pagination
            query = session.query(MLExperience).order_by(MLExperience.timestamp.desc())
            
            total_count = session.query(MLExperience).count()
            experiences = query.offset(offset).limit(limit).all()
            
            exp_list = []
            for exp in experiences:
                import json
                # Handle both dict (PostgreSQL) and string (SQLite) formats
                rl_state = exp.rl_state
                outcome = exp.outcome
                
                if isinstance(rl_state, str):
                    try:
                        rl_state = json.loads(rl_state)
                    except:
                        rl_state = {"error": "failed to parse", "raw": str(rl_state)[:100]}
                
                if isinstance(outcome, str):
                    try:
                        outcome = json.loads(outcome)
                    except:
                        outcome = {"error": "failed to parse", "raw": str(outcome)[:100]}
                
                exp_list.append({
                    "id": exp.id,
                    "user_id": exp.user_id,
                    "symbol": exp.symbol,
                    "experience_type": exp.experience_type,
                    "quality_score": exp.quality_score,
                    "timestamp": exp.timestamp.isoformat() if exp.timestamp else None,
                    "rl_state": rl_state,
                    "outcome": outcome
                })
            
            return {
                "database": "PostgreSQL",
                "total_experiences": total_count,
                "returned_count": len(exp_list),
                "offset": offset,
                "limit": limit,
                "experiences": exp_list,
                "message": "All experiences retrieved successfully"
            }
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"❌ Error fetching all experiences: {e}")
        import traceback
        return {
            "database": "PostgreSQL",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "message": "Failed to fetch all experiences"
        }

@app.delete("/api/admin/clear_database")
async def clear_database(confirm: str = ""):
    """
    DANGER: Clear ALL experiences from PostgreSQL database
    Requires confirm="DELETE_ALL_DATA" to execute
    """
    if confirm != "DELETE_ALL_DATA":
        return {
            "error": "Missing confirmation",
            "message": "Add ?confirm=DELETE_ALL_DATA to URL to confirm deletion",
            "warning": "This will permanently delete all experiences from cloud database"
        }
    
    try:
        from database import DatabaseManager, MLExperience
        
        db_manager = DatabaseManager()
        session = db_manager.get_session()
        
        try:
            # Count before deletion
            total_before = session.query(MLExperience).count()
            signal_before = session.query(MLExperience).filter(MLExperience.experience_type == 'signal').count()
            exit_before = session.query(MLExperience).filter(MLExperience.experience_type == 'exit').count()
            
            # Delete all
            deleted = session.query(MLExperience).delete()
            session.commit()
            
            # Verify empty
            total_after = session.query(MLExperience).count()
            
            logger.warning(f"🗑️ DELETED {deleted} experiences from cloud database")
            
            return {
                "deleted": deleted,
                "before": {
                    "total": total_before,
                    "signal": signal_before,
                    "exit": exit_before
                },
                "after": {
                    "total": total_after
                },
                "message": f"Successfully deleted {deleted} experiences from cloud database"
            }
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"❌ Error clearing database: {e}")
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "message": "Database clear failed"
        }

# ============================================================================
# EXPERIENCE EXPORT ENDPOINTS (for local dev backtesting)
# ============================================================================

@app.get("/api/experiences/export/signal")
async def export_signal_experiences():
    """
    Export all signal experiences for local dev backtesting.
    This allows developers to run fast local backtests without API calls.
    """
    db = next(get_db())
    try:
        # Query all signal experiences from database
        db_experiences = db.query(RLExperience).filter(
            RLExperience.experience_type == 'SIGNAL'
        ).all()
        
        # Convert to dict format
        experiences = []
        for exp in db_experiences:
            experiences.append({
                'rsi': exp.rsi,
                'vwap_distance': exp.vwap_distance,
                'vix': exp.vix,
                'hour': exp.hour,
                'day_of_week': exp.day_of_week,
                'volume_ratio': exp.volume_ratio,
                'atr': exp.atr,
                'recent_pnl': exp.recent_pnl,
                'streak': exp.streak,
                'signal': exp.signal,
                'took_trade': exp.took_trade,
                'pnl': exp.pnl,
                'created_at': exp.created_at.isoformat() if exp.created_at else None
            })
        
        logger.info(f"📤 Exported {len(experiences)} signal experiences for local dev")
        
        return {
            "experiences": experiences,
            "count": len(experiences),
            "exported_at": datetime.now(pytz.UTC).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error exporting signal experiences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/experiences/export/exit")
async def export_exit_experiences():
    """
    Export all exit experiences for local dev backtesting.
    """
    db = next(get_db())
    try:
        # Query all exit experiences from database  
        db_experiences = db.query(RLExperience).filter(
            RLExperience.experience_type == 'EXIT'
        ).all()
        
        # Convert to dict format
        experiences = []
        for exp in db_experiences:
            experiences.append({
                'vix': exp.vix,
                'hour': exp.hour,
                'atr': exp.atr,
                'regime': exp.regime,
                'pnl': exp.pnl,
                'duration_min': exp.duration_min,
                'exit_reason': exp.exit_reason,
                'partial_exits': exp.partial_exits,
                'created_at': exp.created_at.isoformat() if exp.created_at else None
            })
        
        logger.info(f"📤 Exported {len(experiences)} exit experiences for local dev")
        
        return {
            "experiences": experiences,
            "count": len(experiences),
            "exported_at": datetime.now(pytz.UTC).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error exporting exit experiences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
            elif user.license_expiration and user.license_expiration < datetime.now(pytz.UTC):
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
        if datetime.now(pytz.UTC) > expires_at:
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
    expires_at = datetime.now(pytz.UTC) + timedelta(days=days)
    
    active_licenses[license_key] = {
        "email": email,
        "expires_at": expires_at.isoformat(),
        "status": "active",
        "created_at": datetime.now(pytz.UTC).isoformat()
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
    kill_switch_state["activated_at"] = datetime.now(pytz.UTC).isoformat() if active else None
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
                "created_at": datetime.now(pytz.UTC).isoformat()
            }
            
            logger.info(f"🎉 License created from payment: {license_key} for {customer_email}")
    
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

@app.get("/api/time")
async def get_time_service():
    """
    Centralized time service - Single source of truth for all bots
    
    Provides:
    - Current ET time
    - Market hours status
    - Trading session
    - Trading permission
    
    Bots should call this every 30-60 seconds to stay synchronized
    """
    # Get current ET time
    et_tz = pytz.timezone("America/New_York")
    now_et = datetime.now(et_tz)
    
    # Market status
    market_status = get_market_hours_status(now_et)
    session = get_trading_session(now_et)
    
    # Determine if trading is allowed (based on market hours only)
    trading_allowed = True
    halt_reason = None
    
    return {
        # Time information
        "current_et": now_et.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "current_timestamp": now_et.isoformat(),
        "timezone": "America/New_York",
        
        # Market information
        "market_status": market_status,
        "trading_session": session,
        "weekday": now_et.strftime("%A"),
        
        # Trading permission
        "trading_allowed": trading_allowed,
        "halt_reason": halt_reason
    }

@app.get("/api/time/simple")
async def get_simple_time():
    """
    Lightweight time check - Just ET time and trading permission
    For bots that need quick checks without full details
    
    Checks:
    - Maintenance windows (Mon-Thu 5-6 PM, Fri 5 PM - Sun 6 PM)
    - Weekend closure
    """
    et_tz = pytz.timezone("America/New_York")
    now_et = datetime.now(et_tz)
    
    # Get market status
    market_status = get_market_hours_status(now_et)
    
    # Determine if trading is allowed
    trading_allowed = True
    halt_reason = None
    
    # Priority 1: Maintenance windows
    if market_status == "maintenance":
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
    logger.info("Features: ML/RL signals, licensing, user management")
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


# ============================================================================
# STARTUP: Initialize Neural Networks
# ============================================================================
@app.on_event("startup")
async def startup_event():
    """Initialize neural network models on API startup"""
    global neural_exit_predictor
    
    logger.info("🚀 Initializing neural network confidence scorer...")
    init_neural_scorer(model_path="neural_model.pth")
    logger.info("✅ Neural network ready for predictions")
    
    logger.info("🚀 Initializing neural network exit predictor...")
    neural_exit_predictor = NeuralExitPredictor(model_path="exit_model.pth")
    logger.info("✅ Exit neural network ready for predictions")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
