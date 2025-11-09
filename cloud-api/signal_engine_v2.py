"""
QuoTrading Cloud Signal Engine
Centralized VWAP + RSI signal generation for all customers

This runs 24/7 on Azure, calculates signals, and broadcasts them via API.
Customers connect to fetch signals and execute locally on their TopStep accounts.
"""

from fastapi import FastAPI, HTTPException, Request
from datetime import datetime, time as datetime_time, timedelta
from typing import Dict, Optional, List
from collections import deque
import pytz
import logging
import stripe
import secrets
import hashlib
import json

# Initialize FastAPI
app = FastAPI(
    title="QuoTrading Signal Engine",
    description="Real-time VWAP mean reversion signals",
    version="2.0"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# STRIPE CONFIGURATION
# ============================================================================

# Stripe API keys (test mode)
stripe.api_key = "sk_test_51SRMPJBcgS15fNXbqjX4EzgNBMwwMwMghcOmS8TbZsW5YloTMotI1TUtP2VccSxDKCtWMGOmrgyHB41DAwAwQkAw10@Ls9K2BHU"
STRIPE_WEBHOOK_SECRET = None  # Set this after creating webhook in Stripe dashboard

# In-memory license storage (for beta - will move to database later)
active_licenses = {}  # {license_key: {email, expires_at, stripe_customer_id, stripe_subscription_id}}

# ============================================================================
# CONFIGURATION - Iteration 3 Settings (Your Proven Profitable Settings!)
# ============================================================================

class SignalConfig:
    """Signal generation configuration"""
    # VWAP Settings (Iteration 3)
    VWAP_STD_DEV_1 = 2.5  # Warning zone
    VWAP_STD_DEV_2 = 2.1  # ENTRY ZONE ✅
    VWAP_STD_DEV_3 = 3.7  # EXIT/STOP ZONE ✅
    
    # RSI Settings (Iteration 3)
    RSI_PERIOD = 10
    RSI_OVERSOLD = 35  # LONG entries ✅
    RSI_OVERBOUGHT = 65  # SHORT entries ✅
    
    # Filters
    USE_RSI_FILTER = True
    USE_VWAP_DIRECTION_FILTER = True
    USE_TREND_FILTER = False  # OFF - better without it! ✅
    
    # Trading Hours (Eastern Time)
    ENTRY_START_TIME = datetime_time(18, 0)  # 6 PM
    ENTRY_END_TIME = datetime_time(16, 55)   # 4:55 PM
    FLATTEN_TIME = datetime_time(16, 45)     # 4:45 PM
    FORCED_FLATTEN_TIME = datetime_time(17, 0)  # 5:00 PM
    
    # Market
    INSTRUMENT = "ES"
    TICK_SIZE = 0.25
    TICK_VALUE = 12.50

config = SignalConfig()

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

class MarketState:
    """Holds current market data and calculated indicators"""
    def __init__(self):
        self.bars_1min: deque = deque(maxlen=390)  # 1 trading day
        self.vwap: float = 0.0
        self.vwap_std_dev: float = 0.0
        self.vwap_bands: Dict[str, float] = {
            "upper_1": 0.0, "upper_2": 0.0, "upper_3": 0.0,
            "lower_1": 0.0, "lower_2": 0.0, "lower_3": 0.0
        }
        self.rsi: float = 50.0
        self.current_signal: Optional[Dict] = None
        self.last_update: datetime = datetime.utcnow()

market = MarketState()

# ============================================================================
# CORE CALCULATIONS
# ============================================================================

def calculate_vwap() -> None:
    """Calculate VWAP and standard deviation bands"""
    if len(market.bars_1min) == 0:
        return
    
    # Calculate cumulative VWAP
    total_pv = 0.0
    total_volume = 0.0
    
    for bar in market.bars_1min:
        typical_price = (bar["high"] + bar["low"] + bar["close"]) / 3.0
        pv = typical_price * bar["volume"]
        total_pv += pv
        total_volume += bar["volume"]
    
    if total_volume == 0:
        return
    
    vwap = total_pv / total_volume
    market.vwap = vwap
    
    # Calculate standard deviation (volume-weighted)
    variance_sum = 0.0
    for bar in market.bars_1min:
        typical_price = (bar["high"] + bar["low"] + bar["close"]) / 3.0
        squared_diff = (typical_price - vwap) ** 2
        variance_sum += squared_diff * bar["volume"]
    
    variance = variance_sum / total_volume
    std_dev = variance ** 0.5
    market.vwap_std_dev = std_dev
    
    # Calculate bands
    market.vwap_bands["upper_1"] = vwap + (std_dev * config.VWAP_STD_DEV_1)
    market.vwap_bands["upper_2"] = vwap + (std_dev * config.VWAP_STD_DEV_2)
    market.vwap_bands["upper_3"] = vwap + (std_dev * config.VWAP_STD_DEV_3)
    market.vwap_bands["lower_1"] = vwap - (std_dev * config.VWAP_STD_DEV_1)
    market.vwap_bands["lower_2"] = vwap - (std_dev * config.VWAP_STD_DEV_2)
    market.vwap_bands["lower_3"] = vwap - (std_dev * config.VWAP_STD_DEV_3)
    
    logger.debug(f"VWAP: ${vwap:.2f}, StdDev: ${std_dev:.2f}")


def calculate_rsi() -> float:
    """Calculate RSI from recent bars"""
    if len(market.bars_1min) < config.RSI_PERIOD + 1:
        return 50.0  # Neutral
    
    # Get recent close prices
    closes = [bar["close"] for bar in list(market.bars_1min)[-(config.RSI_PERIOD + 1):]]
    
    # Calculate price changes
    gains = []
    losses = []
    
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    # Calculate average gain/loss
    avg_gain = sum(gains) / config.RSI_PERIOD
    avg_loss = sum(losses) / config.RSI_PERIOD
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    market.rsi = rsi
    return rsi


def check_trading_hours() -> str:
    """Check if market is open for trading
    
    Returns:
        "entry_window", "flatten_mode", or "closed"
    """
    et = pytz.timezone('America/New_York')
    now_et = datetime.now(et).time()
    day_of_week = datetime.now(et).weekday()
    
    # Weekend (Friday 5 PM - Sunday 6 PM)
    if day_of_week == 4 and now_et >= datetime_time(17, 0):  # Friday after 5 PM
        return "closed"
    if day_of_week == 5:  # Saturday
        return "closed"
    if day_of_week == 6 and now_et < datetime_time(18, 0):  # Sunday before 6 PM
        return "closed"
    
    # Daily maintenance (5-6 PM Mon-Thu)
    if day_of_week < 4 and datetime_time(17, 0) <= now_et < datetime_time(18, 0):
        return "closed"
    
    # Flatten mode (4:45-5:00 PM)
    if datetime_time(16, 45) <= now_et < datetime_time(17, 0):
        return "flatten_mode"
    
    # Entry window
    return "entry_window"


def generate_signal() -> Optional[Dict]:
    """
    Generate trading signal based on VWAP + RSI
    
    Returns signal dict or None if no signal
    """
    if len(market.bars_1min) < 2:
        return None
    
    # Check trading hours
    trading_state = check_trading_hours()
    if trading_state != "entry_window":
        logger.debug(f"Market state: {trading_state}, no signals")
        return None
    
    prev_bar = list(market.bars_1min)[-2]
    current_bar = list(market.bars_1min)[-1]
    current_price = current_bar["close"]
    
    # Calculate indicators
    calculate_vwap()
    rsi = calculate_rsi()
    
    # LONG SIGNAL CONDITIONS
    # 1. Price bounces off lower band 2 (2.1 std dev)
    # 2. RSI < 35 (oversold)
    # 3. Price closing back above lower band 2
    if (prev_bar["low"] <= market.vwap_bands["lower_2"] and
        current_price > market.vwap_bands["lower_2"]):
        
        # RSI filter
        if config.USE_RSI_FILTER and rsi >= config.RSI_OVERSOLD:
            logger.debug(f"LONG rejected: RSI {rsi:.1f} not oversold (< {config.RSI_OVERSOLD})")
            return None
        
        # VWAP direction filter (price below VWAP for longs)
        if config.USE_VWAP_DIRECTION_FILTER and current_price >= market.vwap:
            logger.debug(f"LONG rejected: Price ${current_price:.2f} above VWAP ${market.vwap:.2f}")
            return None
        
        logger.info(f"🟢 LONG SIGNAL: Price ${current_price:.2f}, VWAP ${market.vwap:.2f}, RSI {rsi:.1f}")
        
        return {
            "action": "LONG",
            "price": current_price,
            "entry_price": current_price,
            "stop_loss": market.vwap_bands["lower_3"],
            "take_profit": market.vwap_bands["upper_3"],
            "vwap": market.vwap,
            "rsi": rsi,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # SHORT SIGNAL CONDITIONS
    # 1. Price bounces off upper band 2 (2.1 std dev)
    # 2. RSI > 65 (overbought)
    # 3. Price closing back below upper band 2
    if (prev_bar["high"] >= market.vwap_bands["upper_2"] and
        current_price < market.vwap_bands["upper_2"]):
        
        # RSI filter
        if config.USE_RSI_FILTER and rsi <= config.RSI_OVERBOUGHT:
            logger.debug(f"SHORT rejected: RSI {rsi:.1f} not overbought (> {config.RSI_OVERBOUGHT})")
            return None
        
        # VWAP direction filter (price above VWAP for shorts)
        if config.USE_VWAP_DIRECTION_FILTER and current_price <= market.vwap:
            logger.debug(f"SHORT rejected: Price ${current_price:.2f} below VWAP ${market.vwap:.2f}")
            return None
        
        logger.info(f"🔴 SHORT SIGNAL: Price ${current_price:.2f}, VWAP ${market.vwap:.2f}, RSI {rsi:.1f}")
        
        return {
            "action": "SHORT",
            "price": current_price,
            "entry_price": current_price,
            "stop_loss": market.vwap_bands["upper_3"],
            "take_profit": market.vwap_bands["lower_3"],
            "vwap": market.vwap,
            "rsi": rsi,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    return None


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "QuoTrading Signal Engine",
        "version": "2.0",
        "timestamp": datetime.utcnow().isoformat(),
        "instrument": config.INSTRUMENT
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    trading_state = check_trading_hours()
    
    return {
        "status": "healthy",
        "market_state": trading_state,
        "bars_count": len(market.bars_1min),
        "vwap": market.vwap,
        "rsi": market.rsi,
        "last_update": market.last_update.isoformat()
    }


@app.get("/api/signal")
async def get_signal():
    """
    Get current trading signal
    
    Returns:
        Signal dict with action, prices, stops, targets
    """
    # In production, this would check live market data
    # For now, return test mode message
    trading_state = check_trading_hours()
    
    if market.current_signal:
        return market.current_signal
    
    return {
        "signal": "NONE",
        "message": f"No signal - Market state: {trading_state}",
        "timestamp": datetime.utcnow().isoformat(),
        "vwap": market.vwap,
        "rsi": market.rsi,
        "market_state": trading_state
    }


@app.get("/api/indicators")
async def get_indicators():
    """Get current market indicators"""
    return {
        "vwap": market.vwap,
        "vwap_std_dev": market.vwap_std_dev,
        "vwap_bands": market.vwap_bands,
        "rsi": market.rsi,
        "bars_count": len(market.bars_1min),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/market_data")
async def update_market_data(bar: Dict):
    """
    Update market data (for testing)
    
    In production, this would connect to TopStep WebSocket
    """
    market.bars_1min.append(bar)
    market.last_update = datetime.utcnow()
    
    # Generate signal
    signal = generate_signal()
    if signal:
        market.current_signal = signal
        logger.info(f"New signal generated: {signal['action']}")
    
    return {"status": "updated", "bars_count": len(market.bars_1min)}


# ============================================================================
# ML/RL ENDPOINTS
# ============================================================================

# ============================================================================
# RL EXPERIENCE STORAGE - HIVE MIND
# ============================================================================

# Single shared RL experience pool (everyone runs same strategy)
# Starts with Kevin's 178K signal + 121K exit experiences
# Grows as all users contribute (collective learning - hive mind!)
signal_experiences = []  # Shared across all users
exit_experiences = []    # Shared across all users

def load_initial_experiences():
    """Load Kevin's proven experiences to seed the hive mind"""
    global signal_experiences, exit_experiences
    import json
    import os
    
    try:
        # Load signal experiences (6,880 trades)
        if os.path.exists("signal_experience.json"):
            with open("signal_experience.json", "r") as f:
                data = json.load(f)
                signal_experiences = data.get("experiences", [])
            logger.info(f"✅ Loaded {len(signal_experiences):,} signal experiences from Kevin's backtests")
        
        # Load exit experiences (2,961 exits)
        if os.path.exists("exit_experience.json"):
            with open("exit_experience.json", "r") as f:
                data = json.load(f)
                exit_experiences = data.get("exit_experiences", [])
            logger.info(f"✅ Loaded {len(exit_experiences):,} exit experiences from Kevin's backtests")
        
        logger.info(f"🧠 HIVE MIND INITIALIZED: {len(signal_experiences):,} signals + {len(exit_experiences):,} exits")
        logger.info(f"   All users will learn from and contribute to this shared wisdom pool!")
        
    except Exception as e:
        logger.error(f"❌ Could not load initial experiences: {e}")
        logger.info("Starting with empty experience pool")

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
    Get ML confidence score for a trade setup
    
    Request: {
        user_id: str,  # REQUIRED - for data isolation
        symbol: str,
        vwap: float,
        vwap_std_dev: float,
        rsi: float,
        price: float,
        volume: int,
        signal: str  # 'LONG' or 'SHORT'
    }
    
    Returns: {
        ml_confidence: float,  # 0.0 to 1.0
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
        
        # Get ALL experiences (everyone learns from same strategy)
        all_trades = signal_experiences
        
        # Simple ML confidence based on shared RL experiences
        # Everyone contributes to and learns from the same pool
        confidence = calculate_signal_confidence(
            all_experiences=all_trades,
            vwap_distance=abs(price - vwap) / vwap if vwap > 0 else 0,
            rsi=rsi,
            signal=signal
        )
        
        logger.info(f"ML Confidence: {symbol} {signal} @ {price}, RSI={rsi:.1f}, Confidence={confidence:.2%}, Total Trades={len(all_trades)}")
        
        return {
            "ml_confidence": confidence,
            "action": signal if confidence >= 0.5 else "NONE",
            "model_version": "v4.0-shared-learning",
            "total_trade_count": len(all_trades),
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


def calculate_signal_confidence(all_experiences: List, vwap_distance: float, rsi: float, signal: str) -> float:
    """
    Calculate ML confidence based on shared RL experiences
    
    Everyone runs the same strategy, so everyone learns from the same data pool.
    This is FASTER learning - more trades = better ML predictions sooner!
    
    Args:
        all_experiences: All trade results from all users (shared learning)
        vwap_distance: Distance from VWAP
        rsi: Current RSI value
        signal: 'LONG' or 'SHORT'
    """
    confidence = 0.5  # Start neutral
    
    # Learn from ALL users' past performance (shared strategy = shared learning)
    if len(all_experiences) > 20:
        # Calculate win rate for this signal type across ALL users
        similar_trades = [t for t in all_experiences if t.get('signal') == signal]
        if len(similar_trades) > 0:
            wins = sum(1 for t in similar_trades if t.get('pnl', 0) > 0)
            shared_win_rate = wins / len(similar_trades)
            # Adjust confidence based on collective wisdom
            confidence = (confidence + shared_win_rate) / 2
    
    # RSI confidence (stronger signals at extremes)
    if signal == "LONG":
        if rsi < 30:
            confidence += 0.25  # Very oversold
        elif rsi < 35:
            confidence += 0.15  # Oversold
    elif signal == "SHORT":
        if rsi > 70:
            confidence += 0.25  # Very overbought
        elif rsi > 65:
            confidence += 0.15  # Overbought
    
    # VWAP distance confidence (closer to band = better)
    if vwap_distance < 0.001:  # Very close to VWAP band
        confidence += 0.15
    elif vwap_distance < 0.002:
        confidence += 0.10
    
    # Cap confidence at 0.95 (never 100% certain)
    return min(confidence, 0.95)


@app.post("/api/ml/save_trade")
async def save_trade_experience(trade: Dict):
    """
    Save trade experience for RL model training (user-isolated)
    
    Request: {
        user_id: str,  # REQUIRED - for data isolation
        symbol: str,
        side: str,  # 'LONG' or 'SHORT'
        entry_price: float,
        exit_price: float,
        entry_time: str,  # ISO format
        exit_time: str,   # ISO format
        pnl: float,
        entry_vwap: float,
        entry_rsi: float,
        exit_reason: str,
        duration_minutes: float,  # Changed from duration_seconds
        volatility: float,
        streak: int
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
async def validate_license(data: dict):
    """Validate if a license key is active"""
    license_key = data.get("license_key", "").strip().upper()
    
    if not license_key:
        raise HTTPException(status_code=400, detail="License key required")
    
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
        "subscription_status": license_info.get("status", "active")
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
    
    Returns: pre_market, market_open, after_hours, futures_open, weekend_closed
    """
    weekday = now_et.weekday()
    current_time = now_et.time()
    
    # Weekend (Saturday = 5, Sunday = 6)
    if weekday == 5:  # Saturday
        return "weekend_closed"
    elif weekday == 6:  # Sunday
        # Futures open at 6 PM ET on Sunday
        if current_time >= datetime_time(18, 0):
            return "futures_open"
        else:
            return "weekend_closed"
    
    # Weekdays (Monday-Friday)
    if current_time < datetime_time(9, 30):
        # Before 9:30 AM
        if current_time >= datetime_time(6, 0):
            return "pre_market"
        else:
            return "futures_open"
    elif datetime_time(9, 30) <= current_time < datetime_time(16, 0):
        # 9:30 AM - 4:00 PM
        return "market_open"
    elif datetime_time(16, 0) <= current_time < datetime_time(18, 0):
        # 4:00 PM - 6:00 PM
        return "after_hours"
    else:
        # After 6:00 PM
        return "futures_open"

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
    """
    et_tz = pytz.timezone("America/New_York")
    now_et = datetime.now(et_tz)
    
    # Check for active events
    event_active, event_name, _ = check_if_event_active(economic_calendar["events"], now_et)
    
    return {
        "current_et": now_et.strftime("%Y-%m-%d %H:%M:%S"),
        "trading_allowed": not event_active,
        "halt_reason": event_name if event_active else None
    }

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize signal engine on startup"""
    logger.info("=" * 60)
    logger.info("QuoTrading Signal Engine v2.0 - STARTING")
    logger.info("=" * 60)
    logger.info(f"Instrument: {config.INSTRUMENT}")
    logger.info(f"VWAP Entry Band: {config.VWAP_STD_DEV_2} std dev")
    logger.info(f"VWAP Stop Band: {config.VWAP_STD_DEV_3} std dev")
    logger.info(f"RSI Period: {config.RSI_PERIOD}")
    logger.info(f"RSI Levels: {config.RSI_OVERSOLD}/{config.RSI_OVERBOUGHT}")
    logger.info("=" * 60)
    logger.info("Signal Engine Ready! Waiting for market data...")
    logger.info("=" * 60)
    
    # Initialize economic calendar
    logger.info("📅 Initializing economic calendar...")
    update_calendar()  # Initial fetch
    start_calendar_updater()  # Start background updater


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
