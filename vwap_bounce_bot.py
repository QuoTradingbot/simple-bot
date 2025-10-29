"""
VWAP Bounce Bot - Mean Reversion Trading Strategy
Event-driven bot that trades bounces off VWAP standard deviation bands
"""

import os
import logging
from datetime import datetime, time, timedelta
from collections import deque
from typing import Dict, List, Optional, Tuple, Callable
import pytz

# Configuration Dictionary
CONFIG = {
    # Trading Parameters
    "instrument": "MES",
    "timezone": "America/New_York",
    
    # Enhanced Time Parameters (Phase One)
    "entry_window_start": time(9, 0),      # 9:00 AM ET - signals enabled
    "entry_window_end": time(14, 30),      # 2:30 PM ET - signals disabled
    "warning_time": time(16, 30),          # 4:30 PM ET - flatten mode begins
    "forced_flatten_time": time(16, 45),   # 4:45 PM ET - force close positions
    "shutdown_time": time(17, 0),          # 5:00 PM ET - bot shutdown
    "vwap_reset_time": time(9, 30),        # 9:30 AM ET - VWAP daily reset
    # Note: VWAP resets at 9:30 AM (stock market open) while entry window starts at 9:00 AM.
    # This allows overnight VWAP to carry into early trading (9:00-9:30 AM), then resets
    # at market open for proper equity index alignment.
    
    # Phase 9-14: Advanced Safety Parameters
    "proactive_stop_buffer_ticks": 2,      # Close proactively when within N ticks of stop
    "friday_entry_cutoff": time(13, 0),    # 1:00 PM ET - no new trades on Friday
    "friday_close_target": time(15, 0),    # 3:00 PM ET - target close time on Friday
    
    # Risk Management
    "risk_per_trade": 0.001,  # 0.1% of account equity
    "max_contracts": 1,
    "max_trades_per_day": 5,
    "daily_loss_limit": 400.0,  # Conservative limit before TopStep's $1000
    
    # Instrument Specifications (MES)
    "tick_size": 0.25,
    "tick_value": 1.25,
    
    # Strategy Parameters
    "trend_filter_period": 50,  # bars
    "trend_timeframe": 15,  # minutes
    "vwap_timeframe": 1,  # minutes
    "vwap_sd_multipliers": {
        "band_1": 1.0,
        "band_2": 2.0
    },
    "risk_reward_ratio": 1.5,
    "stop_buffer_ticks": 2,  # Buffer beyond band for stop placement
    "flatten_buffer_ticks": 2,  # Ticks worse than bid/offer for flatten orders
    
    # Safety Mechanisms (Phase 12)
    "max_drawdown_percent": 2.0,  # Maximum total drawdown percentage
    "tick_timeout_seconds": 60,  # Max seconds without tick during market hours
    
    # System Settings
    "dry_run": True,
    "log_file": "vwap_bounce_bot.log",
    "max_tick_storage": 10000,
    "max_bars_storage": 200
}

# Global SDK client instance
sdk_client = None

# State management dictionary
state = {}

# Global tracking for safety mechanisms (Phase 12)
bot_status = {
    "trading_enabled": True,
    "starting_equity": None,
    "last_tick_time": None,
    "emergency_stop": False,
    "stop_reason": None,
    "flatten_mode": False,  # Phase One: Aggressive exit mode flag
    # Phase 10: Track target vs early close decisions
    "target_wait_wins": 0,  # Times waiting for target paid off
    "target_wait_losses": 0,  # Times waiting for target caused reversal
    "early_close_saves": 0,  # Times early close prevented loss
}


def setup_logging():
    """Configure logging for the bot"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(CONFIG["log_file"]),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================================
# PHASE TWO: SDK Integration
# ============================================================================

def initialize_sdk():
    """
    Initialize the TopStep SDK client using environment variable token.
    Returns the initialized client or exits if token is missing.
    """
    global sdk_client
    
    token = os.getenv("TOPSTEP_API_TOKEN")
    if not token:
        logger.error("TOPSTEP_API_TOKEN environment variable not found!")
        logger.error("Please set your API token: export TOPSTEP_API_TOKEN='your_token_here'")
        exit(1)
    
    try:
        # Placeholder for actual SDK initialization
        # sdk_client = TopStepSDK(api_token=token)
        logger.info("SDK client initialized successfully")
        logger.warning("Running in simulation mode - actual SDK not imported")
        sdk_client = {"mock": True, "token": token}  # Mock client for now
        return sdk_client
    except Exception as e:
        logger.error(f"Failed to initialize SDK: {e}")
        exit(1)


def get_account_equity() -> float:
    """
    Fetch current account equity from SDK.
    Returns account equity/balance.
    """
    if sdk_client is None:
        logger.error("SDK client not initialized")
        return 0.0
    
    try:
        # Placeholder for actual SDK call
        # account_info = sdk_client.get_account_info()
        # equity = account_info.get('equity') or account_info.get('balance')
        
        # Mock equity for development
        equity = 50000.0
        logger.info(f"Account equity: ${equity:.2f}")
        return equity
    except Exception as e:
        logger.error(f"Error fetching account equity: {e}")
        return 0.0


def place_market_order(symbol: str, side: str, quantity: int) -> Optional[Dict]:
    """
    Place a market order through the SDK.
    
    Args:
        symbol: Instrument symbol (e.g., 'MES')
        side: 'BUY' or 'SELL'
        quantity: Number of contracts
    
    Returns:
        Order object or None if failed
    """
    logger.info(f"{'[DRY RUN] ' if CONFIG['dry_run'] else ''}Market Order: {side} {quantity} {symbol}")
    
    if CONFIG["dry_run"]:
        return {
            "order_id": f"MOCK_{datetime.now().timestamp()}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": "MARKET",
            "status": "FILLED",
            "dry_run": True
        }
    
    try:
        # Placeholder for actual SDK call
        # order = sdk_client.create_market_order(
        #     symbol=symbol,
        #     side=side,
        #     quantity=quantity
        # )
        # return order
        
        logger.warning("Live trading not implemented - SDK integration required")
        return None
    except Exception as e:
        logger.error(f"Error placing market order: {e}")
        return None


def place_stop_order(symbol: str, side: str, quantity: int, stop_price: float) -> Optional[Dict]:
    """
    Place a stop order through the SDK.
    
    Args:
        symbol: Instrument symbol
        side: 'BUY' or 'SELL'
        quantity: Number of contracts
        stop_price: Stop trigger price
    
    Returns:
        Order object or None if failed
    """
    logger.info(f"{'[DRY RUN] ' if CONFIG['dry_run'] else ''}Stop Order: {side} {quantity} {symbol} @ {stop_price}")
    
    if CONFIG["dry_run"]:
        return {
            "order_id": f"MOCK_STOP_{datetime.now().timestamp()}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": "STOP",
            "stop_price": stop_price,
            "status": "PENDING",
            "dry_run": True
        }
    
    try:
        # Placeholder for actual SDK call
        # order = sdk_client.create_stop_order(
        #     symbol=symbol,
        #     side=side,
        #     quantity=quantity,
        #     stop_price=stop_price
        # )
        # return order
        
        logger.warning("Live trading not implemented - SDK integration required")
        return None
    except Exception as e:
        logger.error(f"Error placing stop order: {e}")
        return None


def place_limit_order(symbol: str, side: str, quantity: int, limit_price: float) -> Optional[Dict]:
    """
    Place a limit order through the SDK.
    Phase Seven: Used for aggressive flatten orders to avoid market order slippage.
    
    Args:
        symbol: Instrument symbol
        side: 'BUY' or 'SELL'
        quantity: Number of contracts
        limit_price: Limit price
    
    Returns:
        Order object or None if failed
    """
    logger.info(f"{'[DRY RUN] ' if CONFIG['dry_run'] else ''}Limit Order: {side} {quantity} {symbol} @ {limit_price}")
    
    if CONFIG["dry_run"]:
        return {
            "order_id": f"MOCK_LIMIT_{datetime.now().timestamp()}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": "LIMIT",
            "limit_price": limit_price,
            "status": "PENDING",
            "dry_run": True
        }
    
    try:
        # Placeholder for actual SDK call
        # order = sdk_client.create_limit_order(
        #     symbol=symbol,
        #     side=side,
        #     quantity=quantity,
        #     limit_price=limit_price
        # )
        # return order
        
        logger.warning("Live trading not implemented - SDK integration required")
        return None
    except Exception as e:
        logger.error(f"Error placing limit order: {e}")
        return None


def get_position_quantity(symbol: str) -> int:
    """
    Query SDK for current position quantity.
    Phase Eight: Used to check for partial fills.
    
    Args:
        symbol: Instrument symbol
    
    Returns:
        Current position quantity (positive for long, negative for short, 0 for flat)
    """
    if CONFIG["dry_run"]:
        # In dry run mode, use our tracked position
        if state.get(symbol) and state[symbol]["position"]["active"]:
            qty = state[symbol]["position"]["quantity"]
            side = state[symbol]["position"]["side"]
            return qty if side == "long" else -qty
        return 0
    
    try:
        # Placeholder for actual SDK call
        # position = sdk_client.get_position(symbol=symbol)
        # return position.get('quantity', 0)
        
        logger.warning("Live position query not implemented - SDK integration required")
        return 0
    except Exception as e:
        logger.error(f"Error querying position: {e}")
        return 0


def subscribe_market_data(symbol: str, callback: Callable):
    """
    Subscribe to real-time market data for a symbol.
    
    Args:
        symbol: Instrument symbol
        callback: Function to call with tick data (symbol, price, volume, timestamp)
    """
    logger.info(f"{'[DRY RUN] ' if CONFIG['dry_run'] else ''}Subscribing to market data: {symbol}")
    
    if CONFIG["dry_run"]:
        logger.info(f"Mock subscription to {symbol} - callback registered")
        return
    
    try:
        # Placeholder for actual SDK call
        # sdk_client.subscribe_ticks(symbol=symbol, callback=callback)
        logger.warning("Live data subscription not implemented - SDK integration required")
    except Exception as e:
        logger.error(f"Error subscribing to market data: {e}")


def fetch_historical_bars(symbol: str, timeframe: int, count: int) -> List[Dict]:
    """
    Fetch historical bars for initial trend calculation.
    
    Args:
        symbol: Instrument symbol
        timeframe: Bar timeframe in minutes
        count: Number of bars to fetch
    
    Returns:
        List of bar dictionaries with OHLCV data
    """
    logger.info(f"Fetching {count} historical {timeframe}min bars for {symbol}")
    
    try:
        # Placeholder for actual SDK call
        # bars = sdk_client.get_historical_bars(
        #     symbol=symbol,
        #     interval=f"{timeframe}m",
        #     limit=count
        # )
        # return bars
        
        # Mock data for development
        logger.warning("Returning mock historical data - SDK integration required")
        return []
    except Exception as e:
        logger.error(f"Error fetching historical bars: {e}")
        return []


# ============================================================================
# PHASE THREE: State Management
# ============================================================================

def initialize_state(symbol: str):
    """
    Initialize state tracking for an instrument.
    
    Args:
        symbol: Instrument symbol
    """
    state[symbol] = {
        # Tick data storage
        "ticks": deque(maxlen=CONFIG["max_tick_storage"]),
        
        # Bar storage
        "bars_1min": deque(maxlen=CONFIG["max_bars_storage"]),
        "bars_15min": deque(maxlen=100),
        
        # Current incomplete bars
        "current_1min_bar": None,
        "current_15min_bar": None,
        
        # VWAP calculation data
        "vwap": None,
        "vwap_bands": {
            "upper_1": None,
            "upper_2": None,
            "lower_1": None,
            "lower_2": None
        },
        "vwap_std_dev": None,
        "vwap_day": None,  # Phase Three: Track VWAP day separately
        
        # Trend filter
        "trend_ema": None,
        "trend_direction": None,  # 'up', 'down', or 'neutral'
        
        # Signal tracking
        "last_signal": None,
        "signal_bar_price": None,  # Track price of bar that generated signal
        
        # Daily tracking
        "trading_day": None,
        "daily_trade_count": 0,
        "daily_pnl": 0.0,
        
        # Session tracking (Phase 13)
        "session_stats": {
            "trades": [],
            "win_count": 0,
            "loss_count": 0,
            "total_pnl": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "pnl_variance": 0.0,
            # Phase 20: Position duration statistics
            "trade_durations": [],  # List of durations in minutes
            "force_flattened_count": 0,  # Trades closed due to time limit
            "after_noon_entries": 0,  # Entries after 12 PM
            "after_noon_force_flattened": 0  # After-noon entries force-closed
        },
        
        # Position tracking
        "position": {
            "active": False,
            "side": None,
            "quantity": 0,
            "entry_price": None,
            "stop_price": None,
            "target_price": None,
            "entry_time": None
        },
        
        # Volume history
        "volume_history": deque(maxlen=CONFIG["max_bars_storage"])
    }
    
    logger.info(f"State initialized for {symbol}")


# ============================================================================
# PHASE FOUR: Data Processing Pipeline
# ============================================================================

def on_tick(symbol: str, price: float, volume: int, timestamp_ms: int):
    """
    Handle incoming tick data.
    
    Args:
        symbol: Instrument symbol
        price: Tick price
        volume: Tick volume
        timestamp_ms: Timestamp in milliseconds
    """
    if symbol not in state:
        initialize_state(symbol)
    
    # Phase 12: Update last tick time for connection health check
    tz = pytz.timezone(CONFIG["timezone"])
    dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=tz)
    bot_status["last_tick_time"] = dt
    
    # Create tick object
    tick = {
        "price": price,
        "volume": volume,
        "timestamp": timestamp_ms
    }
    
    # Append to tick storage
    state[symbol]["ticks"].append(tick)
    
    # Phase Three: Check for VWAP reset at 9:30 AM ET
    check_vwap_reset(symbol, dt)
    
    # Phase 11: Check for daily reset
    check_daily_reset(symbol, dt)
    
    # Phase Eleven: Critical safety check - NO positions past 5 PM
    check_no_overnight_positions(symbol)
    
    # Update 1-minute bars
    update_1min_bar(symbol, price, volume, dt)
    
    # Update 15-minute bars
    update_15min_bar(symbol, price, volume, dt)


def update_1min_bar(symbol: str, price: float, volume: int, dt: datetime):
    """
    Update or create 1-minute bars for VWAP calculation.
    
    Args:
        symbol: Instrument symbol
        price: Current price
        volume: Current volume
        dt: Current datetime
    """
    # Get current minute boundary
    minute_boundary = dt.replace(second=0, microsecond=0)
    
    current_bar = state[symbol]["current_1min_bar"]
    
    if current_bar is None or current_bar["timestamp"] != minute_boundary:
        # Finalize previous bar if exists
        if current_bar is not None:
            state[symbol]["bars_1min"].append(current_bar)
            # Calculate VWAP after new bar is added
            calculate_vwap(symbol)
            # Check for exit conditions if position is active
            check_exit_conditions(symbol)
            # Check for entry signals if no position
            check_for_signals(symbol)
        
        # Start new bar
        state[symbol]["current_1min_bar"] = {
            "timestamp": minute_boundary,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": volume
        }
    else:
        # Update current bar
        current_bar["high"] = max(current_bar["high"], price)
        current_bar["low"] = min(current_bar["low"], price)
        current_bar["close"] = price
        current_bar["volume"] += volume


def update_15min_bar(symbol: str, price: float, volume: int, dt: datetime):
    """
    Update or create 15-minute bars for trend filter.
    
    Args:
        symbol: Instrument symbol
        price: Current price
        volume: Current volume
        dt: Current datetime
    """
    # Get 15-minute boundary
    minute = (dt.minute // 15) * 15
    boundary_15min = dt.replace(minute=minute, second=0, microsecond=0)
    
    current_bar = state[symbol]["current_15min_bar"]
    
    if current_bar is None or current_bar["timestamp"] != boundary_15min:
        # Finalize previous bar if exists
        if current_bar is not None:
            state[symbol]["bars_15min"].append(current_bar)
            # Update trend filter after new bar is added
            update_trend_filter(symbol)
        
        # Start new bar
        state[symbol]["current_15min_bar"] = {
            "timestamp": boundary_15min,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": volume
        }
    else:
        # Update current bar
        current_bar["high"] = max(current_bar["high"], price)
        current_bar["low"] = min(current_bar["low"], price)
        current_bar["close"] = price
        current_bar["volume"] += volume


def update_trend_filter(symbol: str):
    """
    Update the trend filter using EMA of 15-minute bars.
    
    Args:
        symbol: Instrument symbol
    """
    bars = state[symbol]["bars_15min"]
    period = CONFIG["trend_filter_period"]
    
    if len(bars) < period:
        logger.debug(f"Not enough bars for trend filter: {len(bars)}/{period}")
        return
    
    # Calculate EMA
    closes = [bar["close"] for bar in bars]
    ema = calculate_ema(closes, period)
    
    if ema is not None:
        state[symbol]["trend_ema"] = ema
        
        # Determine trend direction with neutral zone (half tick)
        current_price = closes[-1]
        half_tick = CONFIG["tick_size"] / 2.0
        
        if current_price > ema + half_tick:
            state[symbol]["trend_direction"] = "up"
        elif current_price < ema - half_tick:
            state[symbol]["trend_direction"] = "down"
        else:
            state[symbol]["trend_direction"] = "neutral"
        
        logger.debug(f"Trend EMA: {ema:.2f}, Direction: {state[symbol]['trend_direction']}")


def calculate_ema(values: List[float], period: int) -> Optional[float]:
    """
    Calculate Exponential Moving Average.
    
    Args:
        values: List of values
        period: EMA period
    
    Returns:
        EMA value or None
    """
    if len(values) < period:
        return None
    
    multiplier = 2.0 / (period + 1)
    
    # Start with SMA
    ema = sum(values[:period]) / period
    
    # Calculate EMA for remaining values
    for value in values[period:]:
        ema = (value - ema) * multiplier + ema
    
    return ema


# ============================================================================
# PHASE FIVE: VWAP Calculation
# ============================================================================

def calculate_vwap(symbol: str):
    """
    Calculate VWAP and standard deviation bands from 1-minute bars.
    VWAP is volume-weighted average price, reset daily.
    
    Args:
        symbol: Instrument symbol
    """
    bars = state[symbol]["bars_1min"]
    
    if len(bars) == 0:
        return
    
    # Calculate cumulative VWAP
    total_pv = 0.0  # price * volume
    total_volume = 0.0
    
    for bar in bars:
        typical_price = (bar["high"] + bar["low"] + bar["close"]) / 3.0
        pv = typical_price * bar["volume"]
        total_pv += pv
        total_volume += bar["volume"]
    
    if total_volume == 0:
        return
    
    # VWAP = sum(price * volume) / sum(volume)
    vwap = total_pv / total_volume
    state[symbol]["vwap"] = vwap
    
    # Calculate standard deviation (volume-weighted)
    variance_sum = 0.0
    for bar in bars:
        typical_price = (bar["high"] + bar["low"] + bar["close"]) / 3.0
        squared_diff = (typical_price - vwap) ** 2
        variance_sum += squared_diff * bar["volume"]
    
    variance = variance_sum / total_volume
    std_dev = variance ** 0.5
    state[symbol]["vwap_std_dev"] = std_dev
    
    # Calculate bands
    multipliers = CONFIG["vwap_sd_multipliers"]
    state[symbol]["vwap_bands"]["upper_1"] = vwap + (std_dev * multipliers["band_1"])
    state[symbol]["vwap_bands"]["upper_2"] = vwap + (std_dev * multipliers["band_2"])
    state[symbol]["vwap_bands"]["lower_1"] = vwap - (std_dev * multipliers["band_1"])
    state[symbol]["vwap_bands"]["lower_2"] = vwap - (std_dev * multipliers["band_2"])
    
    logger.debug(f"VWAP: {vwap:.2f}, StdDev: {std_dev:.2f}")
    logger.debug(f"Bands - U2: {state[symbol]['vwap_bands']['upper_2']:.2f}, "
                f"U1: {state[symbol]['vwap_bands']['upper_1']:.2f}, "
                f"L1: {state[symbol]['vwap_bands']['lower_1']:.2f}, "
                f"L2: {state[symbol]['vwap_bands']['lower_2']:.2f}")


# ============================================================================
# PHASE SEVEN: Signal Generation Logic
# ============================================================================

def check_for_signals(symbol: str):
    """
    Check for trading signals on each completed 1-minute bar.
    Called after VWAP calculation is complete.
    
    Args:
        symbol: Instrument symbol
    """
    # Phase 12: Check safety conditions first
    is_safe, reason = check_safety_conditions(symbol)
    if not is_safe:
        logger.debug(f"Safety check failed: {reason}")
        return
    
    # Get the latest bar to check its timestamp
    if len(state[symbol]["bars_1min"]) == 0:
        return
    
    latest_bar = state[symbol]["bars_1min"][-1]
    bar_time = latest_bar["timestamp"]
    
    # Phase Two: Check trading state using centralized time check
    trading_state = get_trading_state(bar_time)
    
    # Only generate signals during entry_window
    if trading_state != "entry_window":
        # Phase Sixteen: Log time-based blocking
        if trading_state == "exit_only":
            log_time_based_action(
                "entry_blocked",
                "Entry window closed at 2:30 PM, no new trades until tomorrow 9:00 AM ET",
                {"current_state": trading_state, "time": bar_time.strftime('%H:%M:%S')}
            )
        logger.debug(f"Not in entry window (state: {trading_state}), skipping signal check")
        return
    
    # Phase Fourteen: Friday position management - no new trades after 1 PM
    if bar_time.weekday() == 4:  # Friday
        if bar_time.time() >= CONFIG["friday_entry_cutoff"]:
            # Phase Sixteen: Log Friday restriction
            log_time_based_action(
                "friday_entry_blocked",
                f"Friday after {CONFIG['friday_entry_cutoff']}, no new trades to avoid weekend gap risk",
                {"day": "Friday", "time": bar_time.strftime('%H:%M:%S')}
            )
            logger.info(f"Friday after {CONFIG['friday_entry_cutoff']} - no new trades (weekend gap risk)")
            return
    
    # Check if already have a position
    if state[symbol]["position"]["active"]:
        logger.debug("Position already active, skipping signal generation")
        return
    
    # Check daily trade limit
    if state[symbol]["daily_trade_count"] >= CONFIG["max_trades_per_day"]:
        logger.warning(f"Daily trade limit reached ({CONFIG['max_trades_per_day']}), stopping for the day")
        return
    
    # Check daily loss limit (redundant with safety check but keep for clarity)
    if state[symbol]["daily_pnl"] <= -CONFIG["daily_loss_limit"]:
        logger.warning(f"Daily loss limit hit (${state[symbol]['daily_pnl']:.2f}), stopping for the day")
        return
    
    # Get required data
    if len(state[symbol]["bars_1min"]) < 2:
        logger.info(f"Not enough bars for signal: {len(state[symbol]['bars_1min'])}/2")
        return  # Need at least 2 bars to check for bounce
    
    vwap_bands = state[symbol]["vwap_bands"]
    trend = state[symbol]["trend_direction"]
    
    # Check if VWAP bands are calculated
    if any(v is None for v in vwap_bands.values()):
        logger.info("VWAP bands not yet calculated")
        return
    
    if trend is None or trend == "neutral":
        logger.info(f"Trend not established or neutral: {trend}")
        return
    
    # Get latest bars
    prev_bar = state[symbol]["bars_1min"][-2]
    current_bar = state[symbol]["bars_1min"][-1]
    
    logger.debug(f"Signal check: trend={trend}, prev_low={prev_bar['low']:.2f}, "
                f"current_close={current_bar['close']:.2f}, lower_band_2={vwap_bands['lower_2']:.2f}")
    
    # Long signal: trend is up, price touched/crossed below lower band 2, then closed back above it
    if trend == "up":
        # Check if previous bar touched lower band 2
        touched_lower = prev_bar["low"] <= vwap_bands["lower_2"]
        # Check if current bar closed back above lower band 2
        bounced_back = current_bar["close"] > vwap_bands["lower_2"]
        
        logger.debug(f"Long check: touched_lower={touched_lower}, bounced_back={bounced_back}")
        
        if touched_lower and bounced_back:
            logger.info(f"LONG SIGNAL: Bounce off lower band 2 with uptrend")
            logger.info(f"  Price: {current_bar['close']:.2f}, Lower Band 2: {vwap_bands['lower_2']:.2f}")
            logger.info(f"  Trend: {trend}, EMA: {state[symbol]['trend_ema']:.2f}")
            execute_entry(symbol, "long", current_bar["close"])
            return
    
    # Short signal: trend is down, price touched/crossed above upper band 2, then closed back below it
    if trend == "down":
        # Check if previous bar touched upper band 2
        touched_upper = prev_bar["high"] >= vwap_bands["upper_2"]
        # Check if current bar closed back below upper band 2
        bounced_back = current_bar["close"] < vwap_bands["upper_2"]
        
        if touched_upper and bounced_back:
            logger.info(f"SHORT SIGNAL: Bounce off upper band 2 with downtrend")
            logger.info(f"  Price: {current_bar['close']:.2f}, Upper Band 2: {vwap_bands['upper_2']:.2f}")
            logger.info(f"  Trend: {trend}, EMA: {state[symbol]['trend_ema']:.2f}")
            execute_entry(symbol, "short", current_bar["close"])
            return


# ============================================================================
# PHASE EIGHT: Position Sizing
# ============================================================================

def calculate_position_size(symbol: str, side: str, entry_price: float) -> Tuple[int, float, float]:
    """
    Calculate position size based on risk management rules.
    
    Args:
        symbol: Instrument symbol
        side: 'long' or 'short'
        entry_price: Expected entry price
    
    Returns:
        Tuple of (contracts, stop_price, target_price)
    """
    # Get account equity
    equity = get_account_equity()
    
    # Calculate risk allowance (0.1% of equity)
    risk_dollars = equity * CONFIG["risk_per_trade"]
    logger.info(f"Account equity: ${equity:.2f}, Risk allowance: ${risk_dollars:.2f}")
    
    # Determine stop price
    vwap_bands = state[symbol]["vwap_bands"]
    tick_size = CONFIG["tick_size"]
    buffer_ticks = 2  # 2 tick buffer beyond band
    
    if side == "long":
        # Stop below lower band 2
        stop_price = vwap_bands["lower_2"] - (buffer_ticks * tick_size)
    else:  # short
        # Stop above upper band 2
        stop_price = vwap_bands["upper_2"] + (buffer_ticks * tick_size)
    
    stop_price = round_to_tick(stop_price)
    
    # Calculate stop distance in ticks
    stop_distance = abs(entry_price - stop_price)
    ticks_at_risk = stop_distance / tick_size
    
    # Calculate risk per contract
    tick_value = CONFIG["tick_value"]
    risk_per_contract = ticks_at_risk * tick_value
    
    # Calculate number of contracts
    if risk_per_contract > 0:
        contracts = int(risk_dollars / risk_per_contract)
    else:
        contracts = 0
    
    # Cap at max contracts
    contracts = min(contracts, CONFIG["max_contracts"])
    
    if contracts == 0:
        logger.warning(f"Position size too small: risk=${risk_per_contract:.2f}, allowance=${risk_dollars:.2f}")
        return 0, stop_price, None
    
    # Calculate target price (1.5:1 risk/reward)
    target_distance = stop_distance * CONFIG["risk_reward_ratio"]
    if side == "long":
        target_price = entry_price + target_distance
    else:
        target_price = entry_price - target_distance
    
    target_price = round_to_tick(target_price)
    
    logger.info(f"Position sizing: {contracts} contract(s)")
    logger.info(f"  Entry: ${entry_price:.2f}, Stop: ${stop_price:.2f}, Target: ${target_price:.2f}")
    logger.info(f"  Risk: {ticks_at_risk:.1f} ticks (${risk_per_contract:.2f})")
    logger.info(f"  Reward: {target_distance/tick_size:.1f} ticks ({CONFIG['risk_reward_ratio']}:1 R/R)")
    
    return contracts, stop_price, target_price


# ============================================================================
# PHASE NINE: Entry Execution
# ============================================================================

def execute_entry(symbol: str, side: str, entry_price: float):
    """
    Execute entry order with stop loss and target.
    Phase Four: Double time check before placing entry order.
    
    Args:
        symbol: Instrument symbol
        side: 'long' or 'short'
        entry_price: Approximate entry price
    """
    # Calculate position size
    contracts, stop_price, target_price = calculate_position_size(symbol, side, entry_price)
    
    if contracts == 0:
        logger.warning("Cannot enter trade - position size is zero")
        return
    
    # Phase 12: Validate order before placing
    is_valid, error_msg = validate_order(symbol, side, contracts, entry_price, stop_price)
    if not is_valid:
        logger.error(f"Order validation failed: {error_msg}")
        return
    
    # Phase Four: Final time check before placing order
    # Check if we're still in entry window after calculations
    tz = pytz.timezone(CONFIG["timezone"])
    entry_time = datetime.now(tz)
    trading_state = get_trading_state(entry_time)
    
    if trading_state != "entry_window":
        logger.warning("=" * 60)
        logger.warning("ENTRY ABORTED - No longer in entry window")
        logger.warning(f"  Current state: {trading_state}")
        logger.warning(f"  Time: {entry_time.strftime('%H:%M:%S %Z')}")
        logger.warning(f"  Signal triggered but calculations took too long")
        logger.warning("=" * 60)
        return
    
    # Place market order
    order_side = "BUY" if side == "long" else "SELL"
    
    logger.info(f"=" * 60)
    logger.info(f"ENTERING {side.upper()} POSITION")
    logger.info(f"  Time: {entry_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"  Symbol: {symbol}")
    logger.info(f"  Contracts: {contracts}")
    logger.info(f"  Entry Price: ${entry_price:.2f}")
    logger.info(f"  Stop Loss: ${stop_price:.2f}")
    logger.info(f"  Target: ${target_price:.2f}")
    
    # Place market order
    order = place_market_order(symbol, order_side, contracts)
    
    if order is None:
        logger.error("Failed to place entry order")
        return
    
    # Update position tracking
    state[symbol]["position"] = {
        "active": True,
        "side": side,
        "quantity": contracts,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "entry_time": entry_time,
        "order_id": order.get("order_id")
    }
    
    # Place stop loss order
    stop_side = "SELL" if side == "long" else "BUY"
    stop_order = place_stop_order(symbol, stop_side, contracts, stop_price)
    
    if stop_order:
        state[symbol]["position"]["stop_order_id"] = stop_order.get("order_id")
        logger.info(f"Stop loss order placed: {stop_order.get('order_id')}")
    
    # Increment daily trade counter
    state[symbol]["daily_trade_count"] += 1
    
    logger.info(f"Position opened successfully (Trade {state[symbol]['daily_trade_count']}/{CONFIG['max_trades_per_day']})")
    logger.info("=" * 60)


# ============================================================================
# PHASE TEN: Exit Management
# ============================================================================

def check_exit_conditions(symbol: str):
    """
    Check exit conditions for open position on each bar.
    Implements flatten mode for aggressive position closing.
    Phase Five: Time-based exit tightening after 3 PM.
    Phase Six: Enhanced flatten mode with minute-by-minute monitoring.
    
    Args:
        symbol: Instrument symbol
    """
    if not state[symbol]["position"]["active"]:
        return
    
    position = state[symbol]["position"]
    
    if len(state[symbol]["bars_1min"]) == 0:
        return
    
    current_bar = state[symbol]["bars_1min"][-1]
    bar_time = current_bar["timestamp"]
    side = position["side"]
    entry_price = position["entry_price"]
    stop_price = position["stop_price"]
    target_price = position["target_price"]
    
    # Phase Two: Check trading state
    trading_state = get_trading_state(bar_time)
    
    # Phase Six: Enhanced flatten mode with critical warnings
    if trading_state == "flatten_mode" and not bot_status["flatten_mode"]:
        bot_status["flatten_mode"] = True
        
        # Phase Sixteen: Log flatten mode activation with position details
        position_details = {
            "side": position["side"],
            "quantity": position["quantity"],
            "entry_price": f"${position['entry_price']:.2f}",
            "current_time": bar_time.strftime('%H:%M:%S %Z')
        }
        
        # Calculate unrealized P&L for logging
        tick_size = CONFIG["tick_size"]
        tick_value = CONFIG["tick_value"]
        if side == "long":
            price_change = current_bar["close"] - entry_price
        else:
            price_change = entry_price - current_bar["close"]
        ticks = price_change / tick_size
        unrealized_pnl = ticks * tick_value * position["quantity"]
        position_details["unrealized_pnl"] = f"${unrealized_pnl:+.2f}"
        
        log_time_based_action(
            "flatten_mode_activated",
            "All positions must be closed by 4:45 PM ET to avoid settlement risk",
            position_details
        )
        
        logger.critical("=" * 60)
        logger.critical("FLATTEN MODE ACTIVATED - POSITION MUST CLOSE IN 15 MINUTES")
        logger.critical("=" * 60)
    
    # Phase Six: Minute-by-minute status logging during flatten mode
    if bot_status["flatten_mode"]:
        tz = pytz.timezone(CONFIG["timezone"])
        current_time = datetime.now(tz)
        forced_flatten_time = datetime.combine(current_time.date(), CONFIG["forced_flatten_time"])
        forced_flatten_time = tz.localize(forced_flatten_time)
        minutes_remaining = (forced_flatten_time - current_time).total_seconds() / 60.0
        
        # Calculate unrealized P&L
        tick_size = CONFIG["tick_size"]
        tick_value = CONFIG["tick_value"]
        if side == "long":
            price_change = current_bar["close"] - entry_price
        else:
            price_change = entry_price - current_bar["close"]
        ticks = price_change / tick_size
        unrealized_pnl = ticks * tick_value * position["quantity"]
        
        logger.warning(f"Flatten Mode Status: {minutes_remaining:.1f} min remaining, "
                      f"P&L: ${unrealized_pnl:+.2f}, Side: {side}, Qty: {position['quantity']}")
        
        # Phase Six: Time-based forced exits at specific times
        if current_time.time() >= time(16, 40) and unrealized_pnl > 0:
            # 4:40 PM - close profitable positions immediately
            logger.critical("4:40 PM - Closing profitable position immediately")
            flatten_price = get_flatten_price(symbol, side, current_bar["close"])
            execute_exit(symbol, flatten_price, "time_based_profit_take")
            return
        
        if current_time.time() >= time(16, 42):
            # 4:42 PM - close small losses
            stop_distance = abs(entry_price - stop_price)
            if unrealized_pnl < 0 and abs(unrealized_pnl) < (stop_distance * tick_value * position["quantity"] / 2):
                logger.critical("4:42 PM - Cutting small loss before settlement")
                flatten_price = get_flatten_price(symbol, side, current_bar["close"])
                execute_exit(symbol, flatten_price, "time_based_loss_cut")
                return
    
    # Force close at forced_flatten_time (4:45 PM)
    if trading_state == "closed":
        logger.critical("=" * 60)
        logger.critical("EMERGENCY FORCED FLATTEN - 4:45 PM DEADLINE REACHED")
        logger.critical("=" * 60)
        # Phase Seven: Use aggressive limit order for forced flatten
        flatten_price = get_flatten_price(symbol, side, current_bar["close"])
        execute_exit(symbol, flatten_price, "emergency_forced_flatten")
        return
    
    # Check for stop hit
    if side == "long":
        if current_bar["low"] <= stop_price:
            execute_exit(symbol, stop_price, "stop_loss")
            return
    else:  # short
        if current_bar["high"] >= stop_price:
            execute_exit(symbol, stop_price, "stop_loss")
            return
    
    # Phase Nine: Proactive stop handling during flatten mode
    if bot_status["flatten_mode"]:
        tick_size = CONFIG["tick_size"]
        proactive_buffer = CONFIG["proactive_stop_buffer_ticks"] * tick_size
        
        if side == "long":
            # Check if within 2 ticks of stop price
            if current_bar["close"] <= stop_price + proactive_buffer:
                logger.warning(f"Proactive stop close: within {CONFIG['proactive_stop_buffer_ticks']} ticks of stop")
                flatten_price = get_flatten_price(symbol, side, current_bar["close"])
                execute_exit(symbol, flatten_price, "proactive_stop")
                return
        else:  # short
            # Check if within 2 ticks of stop price
            if current_bar["close"] >= stop_price - proactive_buffer:
                logger.warning(f"Proactive stop close: within {CONFIG['proactive_stop_buffer_ticks']} ticks of stop")
                flatten_price = get_flatten_price(symbol, side, current_bar["close"])
                execute_exit(symbol, flatten_price, "proactive_stop")
                return
    
    # Check for target reached
    if side == "long":
        if current_bar["high"] >= target_price:
            execute_exit(symbol, target_price, "target_reached")
            # Phase 10: Track successful target wait
            if bot_status["flatten_mode"]:
                bot_status["target_wait_wins"] += 1
            return
    else:  # short
        if current_bar["low"] <= target_price:
            execute_exit(symbol, target_price, "target_reached")
            # Phase 10: Track successful target wait
            if bot_status["flatten_mode"]:
                bot_status["target_wait_wins"] += 1
            return
    
    # Phase Ten: Profit target handling during flatten mode after 4:40 PM
    if bot_status["flatten_mode"]:
        tz = pytz.timezone(CONFIG["timezone"])
        current_time = datetime.now(tz)
        
        if current_time.time() >= time(16, 40):
            # Calculate current P&L
            tick_size = CONFIG["tick_size"]
            tick_value = CONFIG["tick_value"]
            if side == "long":
                price_change = current_bar["close"] - entry_price
            else:
                price_change = entry_price - current_bar["close"]
            ticks = price_change / tick_size
            unrealized_pnl = ticks * tick_value * position["quantity"]
            
            # If showing any profit past 4:40 PM, close immediately
            if unrealized_pnl > 0:
                logger.warning(f"Phase 10: Closing early at ${unrealized_pnl:+.2f} profit instead of waiting for target")
                flatten_price = get_flatten_price(symbol, side, current_bar["close"])
                bot_status["early_close_saves"] += 1
                execute_exit(symbol, flatten_price, "early_profit_lock")
                return
    
    # Phase Five: Time-based exit tightening after 3 PM
    if bar_time.time() >= time(15, 0) and not bot_status["flatten_mode"]:
        # After 3 PM - tighten profit taking to 1:1 R/R
        stop_distance = abs(entry_price - stop_price)
        tightened_target_distance = stop_distance  # 1:1 instead of 1.5:1
        
        if side == "long":
            tightened_target = entry_price + tightened_target_distance
            if current_bar["high"] >= tightened_target:
                logger.info("Time-based tightened profit target reached (1:1 R/R after 3 PM)")
                execute_exit(symbol, tightened_target, "tightened_target")
                return
        else:  # short
            tightened_target = entry_price - tightened_target_distance
            if current_bar["low"] <= tightened_target:
                logger.info("Time-based tightened profit target reached (1:1 R/R after 3 PM)")
                execute_exit(symbol, tightened_target, "tightened_target")
                return
    
    # Phase Fourteen: Friday aggressive position management
    if bar_time.weekday() == 4:  # Friday
        # Target close by 3 PM to avoid weekend gap risk
        if bar_time.time() >= CONFIG["friday_close_target"]:
            logger.critical("=" * 60)
            logger.critical("FRIDAY 3 PM - CLOSING POSITION TO AVOID WEEKEND GAP RISK")
            logger.critical("=" * 60)
            flatten_price = get_flatten_price(symbol, side, current_bar["close"])
            execute_exit(symbol, flatten_price, "friday_weekend_protection")
            return
        
        # After 2 PM on Friday, be more aggressive about taking profits
        if bar_time.time() >= time(14, 0):
            tick_size = CONFIG["tick_size"]
            tick_value = CONFIG["tick_value"]
            if side == "long":
                price_change = current_bar["close"] - entry_price
            else:
                price_change = entry_price - current_bar["close"]
            ticks = price_change / tick_size
            unrealized_pnl = ticks * tick_value * position["quantity"]
            
            # Take any profit after 2 PM Friday
            if unrealized_pnl > 0:
                logger.warning(f"Friday 2 PM+ - Taking ${unrealized_pnl:+.2f} profit to avoid weekend risk")
                flatten_price = get_flatten_price(symbol, side, current_bar["close"])
                execute_exit(symbol, flatten_price, "friday_profit_protection")
                return
    
    # Phase Five: After 3:30 PM - cut losses early if less than 75% of stop distance
    if bar_time.time() >= time(15, 30) and not bot_status["flatten_mode"]:
        tick_size = CONFIG["tick_size"]
        tick_value = CONFIG["tick_value"]
        
        if side == "long":
            current_loss_distance = entry_price - current_bar["close"]
            stop_distance = entry_price - stop_price
            
            if current_loss_distance > 0:  # In a loss
                loss_percent = current_loss_distance / stop_distance
                if loss_percent < 0.75:  # Less than 75% of stop distance
                    logger.warning(f"Time-based early loss cut (3:30 PM+): {loss_percent*100:.1f}% of stop distance")
                    execute_exit(symbol, current_bar["close"], "early_loss_cut")
                    return
        else:  # short
            current_loss_distance = current_bar["close"] - entry_price
            stop_distance = stop_price - entry_price
            
            if current_loss_distance > 0:  # In a loss
                loss_percent = current_loss_distance / stop_distance
                if loss_percent < 0.75:  # Less than 75% of stop distance
                    logger.warning(f"Time-based early loss cut (3:30 PM+): {loss_percent*100:.1f}% of stop distance")
                    execute_exit(symbol, current_bar["close"], "early_loss_cut")
                    return
    
    # Flatten mode: More aggressive profit taking and loss cutting
    if bot_status["flatten_mode"]:
        # In flatten mode, take any profit or cut losses quickly
        if side == "long":
            profit_ticks = (current_bar["close"] - entry_price) / CONFIG["tick_size"]
            # Take profit if we're up even 1 tick, or cut loss if down more than half the stop distance
            midpoint = entry_price - (entry_price - stop_price) / 2
            if profit_ticks > 1 or current_bar["close"] < midpoint:
                logger.info(f"Flatten mode exit: profit_ticks={profit_ticks:.1f}")
                flatten_price = get_flatten_price(symbol, side, current_bar["close"])
                execute_exit(symbol, flatten_price, "flatten_mode_exit")
                return
        else:  # short
            profit_ticks = (entry_price - current_bar["close"]) / CONFIG["tick_size"]
            # Take profit if we're up even 1 tick, or cut loss if down more than half the stop distance
            midpoint = entry_price + (stop_price - entry_price) / 2
            if profit_ticks > 1 or current_bar["close"] > midpoint:
                logger.info(f"Flatten mode exit: profit_ticks={profit_ticks:.1f}")
                flatten_price = get_flatten_price(symbol, side, current_bar["close"])
                execute_exit(symbol, flatten_price, "flatten_mode_exit")
                return
    
    # Check for signal reversal (only when not in flatten mode)
    if not bot_status["flatten_mode"]:
        vwap_bands = state[symbol]["vwap_bands"]
        trend = state[symbol]["trend_direction"]
        
        if side == "long" and trend == "up":
            # If price crosses back above upper band 2, bounce is complete
            if current_bar["close"] > vwap_bands["upper_2"]:
                execute_exit(symbol, current_bar["close"], "signal_reversal")
                return
        
        if side == "short" and trend == "down":
            # If price crosses back below lower band 1, bounce is complete
            if current_bar["close"] < vwap_bands["lower_1"]:
                execute_exit(symbol, current_bar["close"], "signal_reversal")
                return


def get_flatten_price(symbol: str, side: str, current_price: float) -> float:
    """
    Calculate flatten price with buffer to avoid worst price.
    Places limit order N ticks worse than current bid/offer.
    
    Args:
        symbol: Instrument symbol
        side: Position side ('long' or 'short')
        current_price: Current market price
    
    Returns:
        Adjusted price for flatten order
    """
    tick_size = CONFIG["tick_size"]
    buffer_ticks = CONFIG["flatten_buffer_ticks"]
    
    if side == "long":
        # Selling, so go buffer ticks below current price (worse than bid)
        flatten_price = current_price - (buffer_ticks * tick_size)
    else:  # short
        # Buying, so go buffer ticks above current price (worse than offer)
        flatten_price = current_price + (buffer_ticks * tick_size)
    
    return round_to_tick(flatten_price)


def execute_exit(symbol: str, exit_price: float, reason: str):
    """
    Execute exit order and update P&L.
    Phase Seven: Use aggressive limit orders during flatten mode.
    Phase Eight: Handle partial fills with retries.
    
    Args:
        symbol: Instrument symbol
        exit_price: Exit price
        reason: Reason for exit (stop_loss, target_reached, signal_reversal, etc.)
    """
    position = state[symbol]["position"]
    
    if not position["active"]:
        return
    
    order_side = "SELL" if position["side"] == "long" else "BUY"
    exit_time = datetime.now(pytz.timezone(CONFIG["timezone"]))
    
    logger.info("=" * 60)
    logger.info(f"EXITING {position['side'].upper()} POSITION")
    logger.info(f"  Reason: {reason.replace('_', ' ').title()}")
    logger.info(f"  Time: {exit_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Calculate P&L
    entry_price = position["entry_price"]
    contracts = position["quantity"]
    tick_size = CONFIG["tick_size"]
    tick_value = CONFIG["tick_value"]
    
    if position["side"] == "long":
        price_change = exit_price - entry_price
    else:
        price_change = entry_price - exit_price
    
    ticks = price_change / tick_size
    pnl = ticks * tick_value * contracts
    
    logger.info(f"  Entry: ${entry_price:.2f}, Exit: ${exit_price:.2f}")
    logger.info(f"  Ticks: {ticks:+.1f}, P&L: ${pnl:+.2f}")
    
    # Phase Sixteen: Log time-based exits with detailed audit trail
    time_based_reasons = [
        "flatten_mode_exit", "time_based_profit_take", "time_based_loss_cut",
        "emergency_forced_flatten", "tightened_target", "early_loss_cut",
        "proactive_stop", "early_profit_lock", "friday_weekend_protection",
        "friday_profit_protection"
    ]
    
    if reason in time_based_reasons:
        exit_details = {
            "exit_price": f"${exit_price:.2f}",
            "pnl": f"${pnl:+.2f}",
            "side": position["side"],
            "quantity": contracts,
            "entry_price": f"${entry_price:.2f}"
        }
        
        reason_descriptions = {
            "flatten_mode_exit": "Flatten mode aggressive exit",
            "time_based_profit_take": "4:40 PM profit lock before settlement",
            "time_based_loss_cut": "4:42 PM small loss cut before settlement",
            "emergency_forced_flatten": "4:45 PM emergency deadline flatten",
            "tightened_target": "3 PM tightened target (1:1 R/R)",
            "early_loss_cut": "3:30 PM early loss cut (<75% stop)",
            "proactive_stop": "Proactive stop (within 2 ticks)",
            "early_profit_lock": "Early profit lock in flatten mode",
            "friday_weekend_protection": "Friday 3 PM weekend protection",
            "friday_profit_protection": "Friday 2 PM profit protection"
        }
        
        log_time_based_action(
            "position_closed",
            reason_descriptions.get(reason, reason),
            exit_details
        )
    
    # Phase Seven: Use aggressive limit orders during flatten mode
    is_flatten_mode = bot_status["flatten_mode"] or reason in ["flatten_mode_exit", "time_based_profit_take", 
                                                                 "time_based_loss_cut", "emergency_forced_flatten"]
    
    if is_flatten_mode:
        logger.info("Using aggressive limit order strategy for flatten")
        execute_flatten_with_limit_orders(symbol, order_side, contracts, exit_price, reason)
    else:
        # Normal exit - use market order
        order = place_market_order(symbol, order_side, contracts)
        
        if order:
            logger.info(f"Exit order placed: {order.get('order_id')}")
    
    # Update daily P&L
    state[symbol]["daily_pnl"] += pnl
    
    # Phase 20: Track position duration
    if position["entry_time"] is not None:
        duration_seconds = (exit_time - position["entry_time"]).total_seconds()
        duration_minutes = duration_seconds / 60.0
        state[symbol]["session_stats"]["trade_durations"].append(duration_minutes)
        
        # Track if this was a forced flatten due to time
        if reason in time_based_reasons:
            state[symbol]["session_stats"]["force_flattened_count"] += 1
        
        # Track after-noon entries
        entry_hour = position["entry_time"].hour
        if entry_hour >= 12:
            state[symbol]["session_stats"]["after_noon_entries"] += 1
            if reason in time_based_reasons:
                state[symbol]["session_stats"]["after_noon_force_flattened"] += 1
        
        logger.info(f"  Position Duration: {duration_minutes:.1f} minutes")
    
    # Phase 13: Update session statistics
    update_session_stats(symbol, pnl)
    
    logger.info(f"Daily P&L: ${state[symbol]['daily_pnl']:+.2f}")
    logger.info(f"Trades today: {state[symbol]['daily_trade_count']}/{CONFIG['max_trades_per_day']}")
    logger.info("=" * 60)
    
    # Reset position tracking
    state[symbol]["position"] = {
        "active": False,
        "side": None,
        "quantity": 0,
        "entry_price": None,
        "stop_price": None,
        "target_price": None,
        "entry_time": None
    }


def execute_flatten_with_limit_orders(symbol: str, order_side: str, contracts: int, 
                                       base_price: float, reason: str):
    """
    Phase Seven & Eight: Execute flatten using aggressive limit orders with partial fill handling.
    
    Args:
        symbol: Instrument symbol
        order_side: 'BUY' or 'SELL'
        contracts: Number of contracts to close
        base_price: Base price for limit orders
        reason: Exit reason
    """
    import time as time_module
    
    tick_size = CONFIG["tick_size"]
    remaining_contracts = contracts
    attempt = 0
    max_attempts = 10
    
    while remaining_contracts > 0 and attempt < max_attempts:
        attempt += 1
        
        # Calculate aggressive limit price
        # Start 1 tick aggressive, increase by 1 tick each attempt
        ticks_aggressive = attempt
        
        if order_side == "SELL":
            # Selling - go below bid
            limit_price = base_price - (ticks_aggressive * tick_size)
        else:  # BUY
            # Buying - go above offer
            limit_price = base_price + (ticks_aggressive * tick_size)
        
        limit_price = round_to_tick(limit_price)
        
        logger.info(f"Flatten attempt {attempt}/{max_attempts}: {order_side} {remaining_contracts} @ {limit_price:.2f}")
        
        # Place aggressive limit order
        order = place_limit_order(symbol, order_side, remaining_contracts, limit_price)
        
        if order:
            logger.info(f"Flatten limit order placed: {order.get('order_id')}")
        
        # Phase Eight: Wait and check for fills
        if attempt < max_attempts:
            wait_seconds = 5 if attempt < 5 else 2  # Shorter waits as we get more urgent
            logger.debug(f"Waiting {wait_seconds} seconds for fill...")
            time_module.sleep(wait_seconds)
            
            # Check current position
            current_qty = get_position_quantity(symbol)
            
            if current_qty == 0:
                logger.info("Position fully closed")
                break
            else:
                filled_contracts = contracts - abs(current_qty)
                if filled_contracts > 0:
                    logger.warning(f"Partial fill: {filled_contracts} of {contracts} filled, {abs(current_qty)} remaining")
                    remaining_contracts = abs(current_qty)
                else:
                    logger.warning(f"No fill on attempt {attempt}, retrying with more aggressive price")
        else:
            # Final attempt - at market price
            logger.critical(f"Final attempt - placing market order for remaining {remaining_contracts}")
            order = place_market_order(symbol, order_side, remaining_contracts)
            if order:
                logger.info(f"Emergency market order placed: {order.get('order_id')}")
            break
    
    if remaining_contracts > 0:
        logger.error(f"Failed to fully flatten position - {remaining_contracts} contracts may remain")
    else:
        logger.info(f"Successfully flattened {contracts} contracts using aggressive limit orders")


# ============================================================================
# PHASE ELEVEN: Daily Reset Logic
# ============================================================================

def check_vwap_reset(symbol: str, current_time: datetime):
    """
    Check if VWAP should reset at 9:30 AM ET (stock market open).
    VWAP resets daily at market open since MES/MNQ track equity indexes.
    
    Args:
        symbol: Instrument symbol
        current_time: Current datetime in Eastern Time
    """
    current_date = current_time.date()
    vwap_reset_time = CONFIG["vwap_reset_time"]
    
    # Check if we've crossed 9:30 AM on a new day
    if state[symbol]["vwap_day"] is None:
        # First run - initialize VWAP day
        state[symbol]["vwap_day"] = current_date
        logger.info(f"VWAP day initialized: {current_date}")
        return
    
    # If it's a new day and we're past 9:30 AM, reset VWAP
    if state[symbol]["vwap_day"] != current_date and current_time.time() >= vwap_reset_time:
        perform_vwap_reset(symbol, current_date, current_time)


def perform_vwap_reset(symbol: str, new_date, reset_time: datetime):
    """
    Perform VWAP reset at 9:30 AM ET daily.
    
    Args:
        symbol: Instrument symbol
        new_date: The new VWAP date
        reset_time: Time of the reset
    """
    logger.info("=" * 60)
    logger.info(f"VWAP RESET at {reset_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"Stock market open alignment - New VWAP day: {new_date}")
    logger.info("=" * 60)
    
    # Clear accumulated 1-minute bars for VWAP calculation
    state[symbol]["bars_1min"].clear()
    
    # Reset cumulative VWAP data
    state[symbol]["vwap"] = None
    state[symbol]["vwap_bands"] = {
        "upper_1": None,
        "upper_2": None,
        "lower_1": None,
        "lower_2": None
    }
    state[symbol]["vwap_std_dev"] = None
    
    # Update VWAP day
    state[symbol]["vwap_day"] = new_date
    
    # Note: 15-minute trend bars continue running - trend carries from overnight
    logger.info("VWAP data cleared - 15-minute trend bars continue running")
    logger.info(f"Current 15-min bars: {len(state[symbol]['bars_15min'])}")
    logger.info("=" * 60)


def check_daily_reset(symbol: str, current_time: datetime):
    """
    Check if we've crossed into a new trading day and reset daily counters.
    This happens at 9:30 AM ET along with VWAP reset, but tracks date changes.
    
    Args:
        symbol: Instrument symbol
        current_time: Current datetime in Eastern Time
    """
    current_date = current_time.date()
    vwap_reset_time = CONFIG["vwap_reset_time"]
    
    # If we have a trading day stored and it's different from current date
    if state[symbol]["trading_day"] is not None:
        if state[symbol]["trading_day"] != current_date:
            # Reset daily counters at 9:30 AM (same as VWAP reset)
            if current_time.time() >= vwap_reset_time:
                perform_daily_reset(symbol, current_date)
    else:
        # First run - initialize trading day
        state[symbol]["trading_day"] = current_date
        logger.info(f"Trading day initialized: {current_date}")


def perform_daily_reset(symbol: str, new_date):
    """
    Perform the actual daily reset operations.
    Resets daily counters and session stats.
    VWAP reset is handled separately by perform_vwap_reset.
    
    Args:
        symbol: Instrument symbol
        new_date: The new trading date
    """
    logger.info("="*60)
    logger.info(f"DAILY RESET - New Trading Day: {new_date}")
    logger.info("="*60)
    
    # Log session summary before reset
    log_session_summary(symbol)
    
    # Reset daily counters
    state[symbol]["daily_trade_count"] = 0
    state[symbol]["daily_pnl"] = 0.0
    state[symbol]["trading_day"] = new_date
    
    # Reset session stats
    state[symbol]["session_stats"] = {
        "trades": [],
        "win_count": 0,
        "loss_count": 0,
        "total_pnl": 0.0,
        "largest_win": 0.0,
        "largest_loss": 0.0,
        "pnl_variance": 0.0
    }
    
    # Re-enable trading if it was stopped for daily limits
    if bot_status["stop_reason"] == "daily_loss_limit":
        bot_status["trading_enabled"] = True
        bot_status["stop_reason"] = None
        logger.info("Trading re-enabled for new day")
    
    # Reset flatten mode flag for new day
    bot_status["flatten_mode"] = False
    
    logger.info("Daily reset complete - Ready for trading")
    logger.info("(VWAP reset handled separately at 9:30 AM)")
    logger.info("="*60)


# ============================================================================
# PHASE TWELVE: Safety Mechanisms
# ============================================================================

def check_safety_conditions(symbol: str) -> Tuple[bool, Optional[str]]:
    """
    Check all safety conditions before allowing trading.
    
    Args:
        symbol: Instrument symbol
    
    Returns:
        Tuple of (is_safe, reason) where is_safe is True if safe to trade
    """
    tz = pytz.timezone(CONFIG["timezone"])
    current_time = datetime.now(tz)
    
    # Check if emergency stop is active
    if bot_status["emergency_stop"]:
        return False, f"Emergency stop active: {bot_status['stop_reason']}"
    
    # Check if trading is disabled
    if not bot_status["trading_enabled"]:
        return False, f"Trading disabled: {bot_status['stop_reason']}"
    
    # Check daily loss limit
    if state[symbol]["daily_pnl"] <= -CONFIG["daily_loss_limit"]:
        if bot_status["trading_enabled"]:
            logger.critical(f"DAILY LOSS LIMIT BREACHED: ${state[symbol]['daily_pnl']:.2f}")
            logger.critical(f"Trading STOPPED for the day")
            bot_status["trading_enabled"] = False
            bot_status["stop_reason"] = "daily_loss_limit"
        return False, "Daily loss limit exceeded"
    
    # Check maximum drawdown
    if bot_status["starting_equity"] is not None:
        current_equity = get_account_equity()
        drawdown_percent = ((bot_status["starting_equity"] - current_equity) / 
                           bot_status["starting_equity"] * 100)
        
        if drawdown_percent >= CONFIG["max_drawdown_percent"]:
            if not bot_status["emergency_stop"]:
                logger.critical(f"MAXIMUM DRAWDOWN EXCEEDED: {drawdown_percent:.2f}%")
                logger.critical(f"Starting: ${bot_status['starting_equity']:.2f}, "
                              f"Current: ${current_equity:.2f}")
                logger.critical("EMERGENCY STOP ACTIVATED")
                bot_status["emergency_stop"] = True
                bot_status["trading_enabled"] = False
                bot_status["stop_reason"] = "max_drawdown_exceeded"
            return False, f"Max drawdown exceeded: {drawdown_percent:.2f}%"
    
    # Check time-based kill switch - use shutdown_time from new time management
    if current_time.time() >= CONFIG["shutdown_time"]:
        if bot_status["trading_enabled"]:
            logger.warning(f"Market closed - Shutting down bot at {current_time.time()}")
            bot_status["trading_enabled"] = False
            bot_status["stop_reason"] = "market_closed"
        return False, "Market closed"
    
    # Check connection health (no ticks in 60 seconds during trading hours)
    # Use get_trading_state instead of legacy is_trading_hours
    if bot_status["last_tick_time"] is not None:
        trading_state = get_trading_state(current_time)
        # Check for tick timeout during any active trading state (not before_open or closed)
        if trading_state not in ["before_open", "closed"]:
            time_since_tick = (current_time - bot_status["last_tick_time"]).total_seconds()
            if time_since_tick > CONFIG["tick_timeout_seconds"]:
                logger.error(f"DATA FEED ISSUE: No tick in {time_since_tick:.0f} seconds")
                logger.error("Trading paused - connection health check failed")
                bot_status["trading_enabled"] = False
                bot_status["stop_reason"] = "data_feed_timeout"
                return False, f"No tick data for {time_since_tick:.0f} seconds"
    
    return True, None


def check_no_overnight_positions(symbol: str):
    """
    Phase Eleven: Critical safety check - ensure NO positions past 5 PM.
    This prevents gap risk and TopStep evaluation issues.
    
    Args:
        symbol: Instrument symbol
    """
    if not state[symbol]["position"]["active"]:
        return  # No position, all good
    
    tz = pytz.timezone(CONFIG["timezone"])
    current_time = datetime.now(tz)
    
    # Critical: If it's past 5 PM and we still have a position, this is a SERIOUS ERROR
    if current_time.time() >= CONFIG["shutdown_time"]:
        logger.critical("=" * 70)
        logger.critical("CRITICAL ERROR: POSITION DETECTED PAST 5 PM ET")
        logger.critical("OVERNIGHT POSITION RISK - IMMEDIATE EMERGENCY CLOSE REQUIRED")
        logger.critical("=" * 70)
        logger.critical(f"Position: {state[symbol]['position']['side']} "
                       f"{state[symbol]['position']['quantity']} contracts")
        logger.critical(f"Entry: ${state[symbol]['position']['entry_price']:.2f}")
        logger.critical("This should NEVER happen - flatten logic failed")
        logger.critical("Manual intervention required")
        logger.critical("=" * 70)
        
        # Emergency flatten at market
        position = state[symbol]["position"]
        order_side = "SELL" if position["side"] == "long" else "BUY"
        place_market_order(symbol, order_side, position["quantity"])
        
        # Force close the position in tracking
        state[symbol]["position"]["active"] = False
        bot_status["emergency_stop"] = True
        bot_status["stop_reason"] = "overnight_position_detected"


def validate_order(symbol: str, side: str, quantity: int, entry_price: float, 
                   stop_price: float) -> Tuple[bool, Optional[str]]:
    """
    Validate order parameters before placing.
    
    Args:
        symbol: Instrument symbol
        side: 'long' or 'short'
        quantity: Number of contracts
        entry_price: Entry price
        stop_price: Stop loss price
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check quantity is positive
    if quantity <= 0:
        return False, f"Invalid quantity: {quantity}"
    
    # Check stop price is on correct side of entry
    if side == "long":
        if stop_price >= entry_price:
            return False, f"Stop price ${stop_price:.2f} must be below entry ${entry_price:.2f} for long"
    else:  # short
        if stop_price <= entry_price:
            return False, f"Stop price ${stop_price:.2f} must be above entry ${entry_price:.2f} for short"
    
    # Check we have sufficient account equity for margin
    equity = get_account_equity()
    if equity <= 0:
        return False, "Invalid account equity"
    
    # Basic margin check (simplified - actual margin requirements vary)
    # For MES, approximate initial margin is ~$1,200 per contract
    estimated_margin = quantity * 1200
    if estimated_margin > equity * 0.5:  # Don't use more than 50% of equity for margin
        return False, f"Insufficient margin: need ~${estimated_margin:.0f}, have ${equity:.2f}"
    
    return True, None


# ============================================================================
# PHASE THIRTEEN: Logging and Monitoring
# ============================================================================

def log_session_summary(symbol: str):
    """
    Log comprehensive session summary at end of trading day.
    
    Args:
        symbol: Instrument symbol
    """
    stats = state[symbol]["session_stats"]
    
    logger.info("="*60)
    logger.info("SESSION SUMMARY")
    logger.info("="*60)
    logger.info(f"Trading Day: {state[symbol]['trading_day']}")
    logger.info(f"Total Trades: {len(stats['trades'])}")
    logger.info(f"Wins: {stats['win_count']}")
    logger.info(f"Losses: {stats['loss_count']}")
    
    if len(stats['trades']) > 0:
        win_rate = stats['win_count'] / len(stats['trades']) * 100
        logger.info(f"Win Rate: {win_rate:.1f}%")
    else:
        logger.info("Win Rate: N/A (no trades)")
    
    logger.info(f"Total P&L: ${stats['total_pnl']:+.2f}")
    logger.info(f"Largest Win: ${stats['largest_win']:+.2f}")
    logger.info(f"Largest Loss: ${stats['largest_loss']:+.2f}")
    
    # Calculate Sharpe ratio if we have variance data
    if stats['pnl_variance'] > 0 and len(stats['trades']) > 1:
        avg_pnl = stats['total_pnl'] / len(stats['trades'])
        std_dev = (stats['pnl_variance'] / (len(stats['trades']) - 1)) ** 0.5
        if std_dev > 0:
            sharpe_ratio = avg_pnl / std_dev
            logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Phase Ten: Log target vs early close statistics
    total_decisions = (bot_status["target_wait_wins"] + bot_status["target_wait_losses"] + 
                       bot_status["early_close_saves"])
    if total_decisions > 0:
        logger.info("="*60)
        logger.info("FLATTEN MODE EXIT ANALYSIS (Phase 10)")
        logger.info(f"Target Wait Wins: {bot_status['target_wait_wins']}")
        logger.info(f"Target Wait Losses: {bot_status['target_wait_losses']}")
        logger.info(f"Early Close Saves: {bot_status['early_close_saves']}")
        if bot_status["target_wait_wins"] > 0:
            target_success_rate = (bot_status["target_wait_wins"] / 
                                   (bot_status["target_wait_wins"] + bot_status["target_wait_losses"]) * 100)
            logger.info(f"Target Wait Success Rate: {target_success_rate:.1f}%")
        logger.info("="*60)
    
    # Phase Twenty: Position duration statistics
    if len(stats['trade_durations']) > 0:
        logger.info("="*60)
        logger.info("POSITION DURATION ANALYSIS (Phase 20)")
        
        avg_duration = sum(stats['trade_durations']) / len(stats['trade_durations'])
        min_duration = min(stats['trade_durations'])
        max_duration = max(stats['trade_durations'])
        
        logger.info(f"Average Position Duration: {avg_duration:.1f} minutes")
        logger.info(f"Shortest Trade: {min_duration:.1f} minutes")
        logger.info(f"Longest Trade: {max_duration:.1f} minutes")
        
        # Calculate force flatten statistics
        total_trades = len(stats['trades'])
        force_flatten_pct = (stats['force_flattened_count'] / total_trades * 100) if total_trades > 0 else 0
        logger.info(f"Force Flattened: {stats['force_flattened_count']}/{total_trades} ({force_flatten_pct:.1f}%)")
        
        if force_flatten_pct > 30:
            logger.warning("⚠️  >30% force-flattened - trade duration too long for time window")
            logger.warning("   Consider: earlier entry cutoff or faster profit targets")
        else:
            logger.info("✅ <30% force-flattened - acceptable duration")
        
        # After-noon entry analysis
        if stats['after_noon_entries'] > 0:
            after_noon_flatten_pct = (stats['after_noon_force_flattened'] / 
                                      stats['after_noon_entries'] * 100)
            logger.info(f"After-Noon Entries: {stats['after_noon_entries']}")
            logger.info(f"After-Noon Force Flattened: {stats['after_noon_force_flattened']} "
                       f"({after_noon_flatten_pct:.1f}%)")
            
            if after_noon_flatten_pct > 50:
                logger.warning("⚠️  >50% of after-noon entries force-flattened")
                logger.warning("   Entry window may be too late - avg duration {:.1f} min vs time remaining"
                              .format(avg_duration))
        
        # Time compatibility analysis
        # If entering at 2 PM, we have 2.75 hours (165 min) until 4:45 PM flatten
        time_to_flatten_at_2pm = 165  # minutes from 2 PM to 4:45 PM
        if avg_duration > time_to_flatten_at_2pm * 0.8:
            logger.warning("⚠️  Average duration uses >80% of available time window")
            logger.warning(f"   Avg duration {avg_duration:.1f} min vs {time_to_flatten_at_2pm} min available at 2 PM")
        
        logger.info("="*60)
    
    logger.info("="*60)


def update_session_stats(symbol: str, pnl: float):
    """
    Update session statistics after a trade.
    
    Args:
        symbol: Instrument symbol
        pnl: Profit/Loss from the trade
    """
    stats = state[symbol]["session_stats"]
    
    # Add trade to history
    stats["trades"].append(pnl)
    
    # Update win/loss counts
    if pnl > 0:
        stats["win_count"] += 1
        stats["largest_win"] = max(stats["largest_win"], pnl)
    elif pnl < 0:
        stats["loss_count"] += 1
        stats["largest_loss"] = min(stats["largest_loss"], pnl)
    
    # Update total P&L
    stats["total_pnl"] += pnl
    
    # Update variance for Sharpe ratio calculation
    # Using Welford's online algorithm for variance
    n = len(stats["trades"])
    if n == 1:
        stats["mean_pnl"] = pnl
        stats["pnl_variance"] = 0.0
    else:
        old_mean = stats.get("mean_pnl", 0.0)
        new_mean = old_mean + (pnl - old_mean) / n
        stats["pnl_variance"] += (pnl - old_mean) * (pnl - new_mean)
        stats["mean_pnl"] = new_mean


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================




def round_to_tick(price: float) -> float:
    """
    Round price to nearest tick size.
    
    Args:
        price: Price to round
    
    Returns:
        Rounded price
    """
    tick_size = CONFIG["tick_size"]
    return round(price / tick_size) * tick_size


# ============================================================================
# PHASE TWO: Time Check Function
# ============================================================================

def get_trading_state(dt: datetime = None) -> str:
    """
    Centralized time checking function that returns current trading state.
    Converts to Eastern Time and determines state based on time of day.
    
    Args:
        dt: Datetime to check (defaults to now in Eastern Time)
    
    Returns:
        Trading state: 'before_open', 'entry_window', 'exit_only', 'flatten_mode', or 'closed'
    """
    if dt is None:
        tz = pytz.timezone(CONFIG["timezone"])
        dt = datetime.now(tz)
    elif dt.tzinfo is None:
        # If naive datetime provided, assume it's Eastern Time
        tz = pytz.timezone(CONFIG["timezone"])
        dt = tz.localize(dt)
    else:
        # Convert to Eastern Time
        tz = pytz.timezone(CONFIG["timezone"])
        dt = dt.astimezone(tz)
    
    current_time = dt.time()
    
    # Before 9:00 AM - before open
    if current_time < CONFIG["entry_window_start"]:
        return "before_open"
    
    # 9:00 AM to 2:30 PM - entry window (signals enabled)
    if CONFIG["entry_window_start"] <= current_time < CONFIG["entry_window_end"]:
        return "entry_window"
    
    # 2:30 PM to 4:30 PM - exit only (signals disabled, position management continues)
    if CONFIG["entry_window_end"] <= current_time < CONFIG["warning_time"]:
        return "exit_only"
    
    # 4:30 PM to 4:45 PM - flatten mode (aggressive position closing)
    if CONFIG["warning_time"] <= current_time < CONFIG["forced_flatten_time"]:
        return "flatten_mode"
    
    # After 4:45 PM - closed (force flatten complete, wait for next day)
    if current_time >= CONFIG["forced_flatten_time"]:
        return "closed"
    
    # Should not reach here, but default to closed for safety
    return "closed"


# ============================================================================
# PHASE FIFTEEN & SIXTEEN: Timezone Handling and Time-Based Logging
# ============================================================================

def validate_timezone_configuration():
    """
    Phase Fifteen: Validate timezone configuration on bot startup.
    Ensures pytz is working correctly and DST is handled properly.
    """
    tz = pytz.timezone(CONFIG["timezone"])
    current_time = datetime.now(tz)
    
    logger.info("=" * 60)
    logger.info("TIMEZONE CONFIGURATION VALIDATION")
    logger.info("=" * 60)
    logger.info(f"Configured Timezone: {CONFIG['timezone']}")
    logger.info(f"Current Time (ET): {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"UTC Offset: {current_time.strftime('%z')}")
    logger.info(f"DST Active: {bool(current_time.dst())}")
    
    # Check if DST transition is near
    tomorrow = current_time + timedelta(days=1)
    if current_time.dst() != tomorrow.dst():
        logger.warning("DST TRANSITION DETECTED - Clock changes within 24 hours")
        logger.warning("Bot has been tested for DST transitions")
    
    # Warn if system local time differs significantly from ET
    system_time = datetime.now()
    if abs((current_time.replace(tzinfo=None) - system_time).total_seconds()) > 3600:
        logger.warning(f"System local time differs from ET by >1 hour")
        logger.warning(f"System: {system_time.strftime('%H:%M:%S')}, ET: {current_time.strftime('%H:%M:%S')}")
        logger.warning("All trading decisions use ET - system time is informational only")
    
    logger.info("=" * 60)


def log_time_based_action(action: str, reason: str, details: dict = None):
    """
    Phase Sixteen: Log all time-based actions with timestamp and reason.
    Creates audit trail for reviewing time-based rule performance.
    
    Args:
        action: Type of action (e.g., 'entry_blocked', 'flatten_activated', 'position_closed')
        reason: Human-readable reason for the action
        details: Optional dictionary of additional details
    """
    tz = pytz.timezone(CONFIG["timezone"])
    timestamp = datetime.now(tz)
    
    log_msg = f"TIME-BASED ACTION: {action}"
    log_msg += f" | Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    log_msg += f" | Reason: {reason}"
    
    if details:
        for key, value in details.items():
            log_msg += f" | {key}: {value}"
    
    logger.info(log_msg)


# ============================================================================
# PHASE SEVENTEEN & EIGHTEEN: Backtesting and Monitoring Guidelines
# ============================================================================

"""
Phase Seventeen: Backtesting Time Logic

When backtesting this strategy on historical data, you must simulate time-based
flatten rules accurately:

1. For every historical trade, check entry time
   - If entered after 2 PM, calculate time remaining until 4:45 PM flatten
   - Simulate forced flatten at whatever price existed at 4:45 PM if position
     hadn't hit target or stop

2. Track forced flatten statistics:
   - Count how many trades were force-flattened before hitting target/stop
   - Calculate how many times forced flatten saved from overnight gap losses
   - Calculate how many times it cost profit by closing a winner early

3. Analyze trade duration:
   - If 30%+ of trades get force-flattened, average trade duration is too long
   - Either extend trading hours or accept lower targets to close positions faster

4. Friday-specific backtesting:
   - Simulate no new trades after 1 PM Friday
   - Simulate forced close at 3 PM Friday
   - Measure weekend gap impact on positions that would have been held

5. DST transition testing:
   - Test bot behavior on DST change days (March and November)
   - Ensure time checks still work correctly during "spring forward" and "fall back"

Phase Eighteen: Monitoring During Flatten Window (4:30 PM - 4:45 PM)

This is the highest-risk 15-minute window requiring active monitoring if possible:

1. Manual intervention scenarios:
   - Bot tries to close but gets no fills even with aggressive limits
   - Manual close at market through broker platform may be needed
   
   - Technical glitch: bot thinks it's flat but broker shows open position
   - Immediate manual intervention required
   
   - Order system failure during critical flatten window
   - Manual close through broker as backup

2. Monitoring checklist:
   - Verify position quantity matches between bot and broker
   - Check that flatten orders are actually being placed
   - Monitor fill confirmations
   - Verify position is actually closed before 5 PM

3. Contingency plan:
   - Have broker platform open and ready
   - Know how to manually close position
   - Have broker support number available
   - Test manual close procedure in paper trading

4. Post-flatten validation:
   - At 5 PM, verify zero positions in both bot and broker
   - Check overnight position safety function logged correctly
   - Review flatten window logs for any issues

This 15-minute window is when most bot failures happen due to:
- Racing against hard deadline
- Deteriorating market conditions
- Widening spreads
- Lower liquidity

Active monitoring provides safety net for automation failures.
"""


# ============================================================================
# PHASE TWENTY: Position Duration Statistics & Complete Summary
# ============================================================================

"""
Phase Twenty: Position Duration Statistics & Time-Window Compatibility

Track how long positions stay open on average to ensure compatibility with
time-based flatten requirements:

1. Position Duration Tracking:
   - Record duration (minutes) for every closed position
   - Calculate average, min, max duration
   - Compare against available time window

2. Force Flatten Analysis:
   - Count trades force-flattened due to time limits
   - Calculate percentage: force_flattened / total_trades
   - RED FLAG if >30% are force-flattened
   
3. After-Noon Entry Analysis:
   - Track entries after 12 PM (noon)
   - Calculate force-flatten rate for after-noon entries
   - If entering at 2 PM with 3-hour avg duration, you'll be force-flattened
   
4. Time Window Compatibility:
   - Entry at 2 PM → 165 minutes until 4:45 PM deadline
   - If avg duration >132 min (80% of 165), trades run out of time
   - Recommend: move entry cutoff earlier OR use faster targets

5. Strategic Adjustments Based on Data:
   - If most trades close in 30 min → plenty of buffer time
   - If most trades take 2-3 hours → cutting it close, risk force-flatten
   - Solution A: Earlier entry cutoff (12 PM instead of 2:30 PM)
   - Solution B: Faster targets (1:1 R/R instead of 1.5:1)
   - The data tells you which adjustment fits your strategy

Complete Time-Based Logic Summary
==================================

Your bot operates in distinct time-based modes controlling all actions:

TIME WINDOWS (All times Eastern Time):
- Before 9:00 AM: SLEEP - Bot inactive, waiting for market open
- 9:00 AM - 9:30 AM: PRE-OPEN - Entry allowed, overnight VWAP active
- 9:30 AM: VWAP RESET - Clears 1-min bars, aligns with stock market open
- 9:00 AM - 2:30 PM: ENTRY WINDOW - Full signal evaluation and entry allowed
- 2:30 PM - 4:30 PM: EXIT ONLY - Manage positions, NO new entries
- 3:00 PM: EXIT TIGHTENING - 1:1 R/R targets instead of 1.5:1
- 3:30 PM: EARLY LOSS CUTS - Cut losses <75% of stop distance
- 4:30 PM - 4:45 PM: FLATTEN MODE - Aggressive forced closing, escalating urgency
- 4:40 PM: FORCE CLOSE PROFITS - Lock in any gain immediately
- 4:42 PM: FORCE CUT SMALL LOSSES - Accept small loss <50% of stop
- 4:45 PM: EMERGENCY DEADLINE - Force close ANY remaining position
- After 4:45 PM: VERIFY FLAT - Check no overnight positions, shut down
- 5:00 PM: ABSOLUTE DEADLINE - Emergency flatten if position still exists

FRIDAY-SPECIFIC RULES:
- 1:00 PM: NO new trades (weekend gap protection begins)
- 2:00 PM: Take ANY profit on existing positions
- 3:00 PM: FORCE CLOSE all positions (61-hour weekend gap avoidance)

DAILY RESETS:
- 9:30 AM: VWAP reset (stock market alignment for equity indexes)
- 9:30 AM: Daily counters reset on date change (trade count, P&L, loss limits)
- 15-minute trend bars: NO reset (overnight trend carries forward)

CRITICAL SAFETY RULES:
1. NO OVERNIGHT POSITIONS - Ever. Zero exceptions. Account-destroying risk.
2. NO WEEKEND POSITIONS - Force close by 3 PM Friday, 66-hour gap risk.
3. SETTLEMENT AVOIDANCE - Flatten by 4:45 PM to avoid 4:45-5:00 PM manipulation.
4. TIMEZONE ENFORCEMENT - All decisions use America/New_York, not system time.
5. DST AWARENESS - pytz handles spring forward / fall back automatically.
6. AUDIT TRAIL - Every time-based action logged with timestamp and reason.

WHY THIS MATTERS FOR PROP FIRMS:
TopStep's rules are designed to fail traders who don't respect:
- Daily settlement (5 PM ET reset)
- Overnight gap exposure
- Weekend event risk
- Daily loss limits (restart at 5 PM, not midnight)

By building time constraints into core logic, you protect against:
- Gap risk from overnight news (Asia/Europe markets, economic data)
- Weekend geopolitical events (can't control, can't trade out)
- Settlement skew manipulation (institutional games in final 30 seconds)
- Starting day already halfway to loss limit (overnight position losses carry forward)

This time-based framework is NOT OPTIONAL for prop firm trading.
It's the difference between controlled risk and catastrophic account blowups.
Being in a position when you shouldn't be is the #1 futures trading killer.

VALIDATION:
- Phase 20 statistics tell you if your strategy fits the time windows
- If >30% force-flattened: strategy incompatible with time constraints
- Adjust entry cutoff earlier OR use faster profit targets
- After-noon entries especially risky - limited time to work
"""


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main bot execution"""
    logger.info("="*60)
    logger.info("VWAP Bounce Bot Starting")
    logger.info("="*60)
    logger.info(f"Mode: {'DRY RUN' if CONFIG['dry_run'] else 'LIVE TRADING'}")
    logger.info(f"Instrument: {CONFIG['instrument']}")
    logger.info(f"Entry Window: {CONFIG['entry_window_start']} - {CONFIG['entry_window_end']} ET")
    logger.info(f"Flatten Mode: {CONFIG['warning_time']} ET")
    logger.info(f"Force Close: {CONFIG['forced_flatten_time']} ET")
    logger.info(f"Shutdown: {CONFIG['shutdown_time']} ET")
    logger.info(f"Max Trades/Day: {CONFIG['max_trades_per_day']}")
    logger.info(f"Daily Loss Limit: ${CONFIG['daily_loss_limit']}")
    logger.info(f"Max Drawdown: {CONFIG['max_drawdown_percent']}%")
    logger.info("="*60)
    
    # Phase Fifteen: Validate timezone configuration
    validate_timezone_configuration()
    
    # Initialize SDK
    initialize_sdk()
    
    # Phase 12: Record starting equity for drawdown monitoring
    bot_status["starting_equity"] = get_account_equity()
    logger.info(f"Starting Equity: ${bot_status['starting_equity']:.2f}")
    
    # Initialize state for instrument
    symbol = CONFIG["instrument"]
    initialize_state(symbol)
    
    # Fetch historical bars for trend filter initialization
    historical_bars = fetch_historical_bars(
        symbol=symbol,
        timeframe=CONFIG["trend_timeframe"],
        count=CONFIG["trend_filter_period"]
    )
    
    if historical_bars:
        state[symbol]["bars_15min"].extend(historical_bars)
        update_trend_filter(symbol)
    
    # Subscribe to market data
    subscribe_market_data(symbol, on_tick)
    
    logger.info("Bot initialization complete")
    logger.info("Waiting for market data...")
    
    # In a real implementation, this would run indefinitely
    # For now, we just show the structure is ready
    logger.info("Bot is ready to process ticks through on_tick() callback")


if __name__ == "__main__":
    main()
