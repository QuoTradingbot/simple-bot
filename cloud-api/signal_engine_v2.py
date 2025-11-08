"""
QuoTrading Cloud Signal Engine
Centralized VWAP + RSI signal generation for all customers

This runs 24/7 on Azure, calculates signals, and broadcasts them via API.
Customers connect to fetch signals and execute locally on their TopStep accounts.
"""

from fastapi import FastAPI, HTTPException
from datetime import datetime, time as datetime_time, timedelta
from typing import Dict, Optional, List
from collections import deque
import pytz
import logging

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

# In-memory storage for trade experiences (will move to database later)
# CRITICAL: Separated by user_id + symbol to prevent data contamination
# Format: {user_id: {symbol: [experiences]}}
user_experiences = {}

def get_user_experiences(user_id: str, symbol: str) -> List:
    """Get experiences for specific user + symbol (isolated data)"""
    if user_id not in user_experiences:
        user_experiences[user_id] = {}
    if symbol not in user_experiences[user_id]:
        user_experiences[user_id][symbol] = []
    return user_experiences[user_id][symbol]

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
        
        # Get user-specific experiences for ML calculation
        user_trades = get_user_experiences(user_id, symbol)
        
        # Simple ML confidence based on user's historical performance
        # TODO: Replace with actual RL model inference per user
        confidence = calculate_signal_confidence(
            user_experiences=user_trades,
            vwap_distance=abs(price - vwap) / vwap if vwap > 0 else 0,
            rsi=rsi,
            signal=signal
        )
        
        logger.info(f"[{user_id}] ML Confidence: {symbol} {signal} @ {price}, RSI={rsi:.1f}, Confidence={confidence:.2%}, Trades={len(user_trades)}")
        
        return {
            "ml_confidence": confidence,
            "action": signal if confidence >= 0.5 else "NONE",
            "model_version": "v2.0-user-isolated",
            "user_trade_count": len(user_trades),
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


def calculate_signal_confidence(user_experiences: List, vwap_distance: float, rsi: float, signal: str) -> float:
    """
    Calculate ML confidence based on user's historical patterns
    
    This is a simple heuristic model that learns from each user's trades.
    Will be replaced with trained RL model.
    
    Args:
        user_experiences: List of this user's past trades (user-specific learning)
        vwap_distance: Distance from VWAP
        rsi: Current RSI value
        signal: 'LONG' or 'SHORT'
    """
    confidence = 0.5  # Start neutral
    
    # Learn from user's past performance (user-specific RL)
    if len(user_experiences) > 5:
        # Calculate user's win rate for this signal type
        similar_trades = [t for t in user_experiences if t.get('signal') == signal]
        if len(similar_trades) > 0:
            wins = sum(1 for t in similar_trades if t.get('pnl', 0) > 0)
            user_win_rate = wins / len(similar_trades)
            # Adjust confidence based on user's history
            confidence = (confidence + user_win_rate) / 2
    
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
        duration_seconds: int,
        ml_confidence_used: float
    }
    
    Returns: {
        saved: bool,
        experience_id: str,
        user_total_trades: int,
        user_win_rate: float
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
        
        # Validate required fields
        required_fields = ['symbol', 'side', 'entry_price', 'exit_price', 'pnl']
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
        
        # Store in user-specific array (data isolation!)
        user_trades = get_user_experiences(user_id, symbol)
        user_trades.append(experience)
        
        # Calculate THIS USER's win rate (not global)
        if len(user_trades) > 0:
            wins = sum(1 for exp in user_trades if exp.get('pnl', 0) > 0)
            win_rate = wins / len(user_trades)
        else:
            win_rate = 0.0
        
        logger.info(f"[{user_id}] Trade Saved: {symbol} {trade['side']} P&L=${trade['pnl']:.2f} | "
                   f"User Total: {len(user_trades)} trades, Win Rate: {win_rate:.1%}")
        
        return {
            "saved": True,
            "experience_id": experience["experience_id"],
            "user_total_trades": len(user_trades),
            "user_win_rate": win_rate
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving trade experience: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/stats")
async def get_ml_stats(user_id: str = None, symbol: str = None):
    """
    Get ML model statistics
    
    Query params:
        user_id: Optional - get stats for specific user
        symbol: Optional - get stats for specific symbol
    """
    if user_id and symbol:
        # User-specific stats
        user_trades = get_user_experiences(user_id, symbol)
        if len(user_trades) == 0:
            return {
                "user_id": user_id,
                "symbol": symbol,
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "total_pnl": 0.0
            }
        
        wins = sum(1 for exp in user_trades if exp.get('pnl', 0) > 0)
        total_pnl = sum(exp.get('pnl', 0) for exp in user_trades)
        
        return {
            "user_id": user_id,
            "symbol": symbol,
            "total_trades": len(user_trades),
            "win_rate": wins / len(user_trades),
            "avg_pnl": total_pnl / len(user_trades),
            "total_pnl": total_pnl,
            "last_updated": datetime.utcnow().isoformat()
        }
    else:
        # Global stats (all users)
        total_users = len(user_experiences)
        total_trades = sum(len(symbols.get(sym, [])) 
                          for symbols in user_experiences.values() 
                          for sym in symbols)
        
        return {
            "total_users": total_users,
            "total_trades": total_trades,
            "message": "Use ?user_id=XXX&symbol=XXX for user-specific stats",
            "last_updated": datetime.utcnow().isoformat()
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
