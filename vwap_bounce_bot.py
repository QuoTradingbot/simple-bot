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
    "trading_window": {
        "start": time(10, 0),  # 10:00 AM ET
        "end": time(15, 30)    # 3:30 PM ET
    },
    "timezone": "America/New_York",
    
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
    
    # Safety Mechanisms (Phase 12)
    "max_drawdown_percent": 2.0,  # Maximum total drawdown percentage
    "market_close_time": time(16, 0),  # 4:00 PM ET - hard stop
    "daily_reset_time": time(8, 0),  # 8:00 AM ET - daily reset
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
    "stop_reason": None
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
            "pnl_variance": 0.0
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
    
    # Phase 11: Check for daily reset
    check_daily_reset(symbol)
    
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
    
    # Check if within trading hours using the bar timestamp
    if not is_trading_hours(bar_time):
        logger.debug(f"Outside trading hours ({bar_time.time()}), skipping signal check")
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
    
    # Place market order
    order_side = "BUY" if side == "long" else "SELL"
    entry_time = datetime.now(pytz.timezone(CONFIG["timezone"]))
    
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
    
    Args:
        symbol: Instrument symbol
    """
    if not state[symbol]["position"]["active"]:
        return
    
    position = state[symbol]["position"]
    
    if len(state[symbol]["bars_1min"]) == 0:
        return
    
    current_bar = state[symbol]["bars_1min"][-1]
    side = position["side"]
    entry_price = position["entry_price"]
    stop_price = position["stop_price"]
    target_price = position["target_price"]
    
    # Check for stop hit
    if side == "long":
        if current_bar["low"] <= stop_price:
            execute_exit(symbol, stop_price, "stop_loss")
            return
    else:  # short
        if current_bar["high"] >= stop_price:
            execute_exit(symbol, stop_price, "stop_loss")
            return
    
    # Check for target reached
    if side == "long":
        if current_bar["high"] >= target_price:
            execute_exit(symbol, target_price, "target_reached")
            return
    else:  # short
        if current_bar["low"] <= target_price:
            execute_exit(symbol, target_price, "target_reached")
            return
    
    # Check for signal reversal
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


def execute_exit(symbol: str, exit_price: float, reason: str):
    """
    Execute exit order and update P&L.
    
    Args:
        symbol: Instrument symbol
        exit_price: Exit price
        reason: Reason for exit (stop_loss, target_reached, signal_reversal)
    """
    position = state[symbol]["position"]
    
    if not position["active"]:
        return
    
    # Place closing market order
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
    
    # Place exit order
    order = place_market_order(symbol, order_side, contracts)
    
    if order:
        logger.info(f"Exit order placed: {order.get('order_id')}")
    
    # Update daily P&L
    state[symbol]["daily_pnl"] += pnl
    
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


# ============================================================================
# PHASE ELEVEN: Daily Reset Logic
# ============================================================================

def check_daily_reset(symbol: str):
    """
    Check if we've crossed into a new trading day and reset counters.
    Should be called around 8 AM ET before market opens.
    
    Args:
        symbol: Instrument symbol
    """
    tz = pytz.timezone(CONFIG["timezone"])
    current_time = datetime.now(tz)
    current_date = current_time.date()
    
    # Check if it's time for daily reset (8 AM ET)
    reset_time = CONFIG["daily_reset_time"]
    
    # If we have a trading day stored and it's different from current date
    if state[symbol]["trading_day"] is not None:
        if state[symbol]["trading_day"] != current_date:
            # Check if we're past reset time
            if current_time.time() >= reset_time:
                perform_daily_reset(symbol, current_date)
    else:
        # First run - initialize trading day
        state[symbol]["trading_day"] = current_date
        logger.info(f"Trading day initialized: {current_date}")


def perform_daily_reset(symbol: str, new_date):
    """
    Perform the actual daily reset operations.
    
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
    
    # Clear VWAP data (new day = new VWAP)
    state[symbol]["bars_1min"].clear()
    state[symbol]["vwap"] = None
    state[symbol]["vwap_bands"] = {
        "upper_1": None,
        "upper_2": None,
        "lower_1": None,
        "lower_2": None
    }
    state[symbol]["vwap_std_dev"] = None
    
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
    
    logger.info("Daily reset complete - Ready for trading")
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
    
    # Check time-based kill switch (after 4 PM ET)
    market_close = CONFIG["market_close_time"]
    if current_time.time() >= market_close:
        if bot_status["trading_enabled"]:
            logger.warning(f"Market closed - Shutting down bot at {current_time.time()}")
            bot_status["trading_enabled"] = False
            bot_status["stop_reason"] = "market_closed"
        return False, "Market closed"
    
    # Check connection health (no ticks in 60 seconds during market hours)
    if bot_status["last_tick_time"] is not None:
        if is_trading_hours(current_time):
            time_since_tick = (current_time - bot_status["last_tick_time"]).total_seconds()
            if time_since_tick > CONFIG["tick_timeout_seconds"]:
                logger.error(f"DATA FEED ISSUE: No tick in {time_since_tick:.0f} seconds")
                logger.error("Trading paused - connection health check failed")
                bot_status["trading_enabled"] = False
                bot_status["stop_reason"] = "data_feed_timeout"
                return False, f"No tick data for {time_since_tick:.0f} seconds"
    
    return True, None


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

def is_trading_hours(dt: datetime = None) -> bool:
    """
    Check if current time is within trading window.
    
    Args:
        dt: Datetime to check (defaults to now)
    
    Returns:
        True if within trading hours
    """
    if dt is None:
        dt = datetime.now(pytz.timezone(CONFIG["timezone"]))
    
    current_time = dt.time()
    return CONFIG["trading_window"]["start"] <= current_time <= CONFIG["trading_window"]["end"]


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
# MAIN EXECUTION
# ============================================================================

def main():
    """Main bot execution"""
    logger.info("="*60)
    logger.info("VWAP Bounce Bot Starting")
    logger.info("="*60)
    logger.info(f"Mode: {'DRY RUN' if CONFIG['dry_run'] else 'LIVE TRADING'}")
    logger.info(f"Instrument: {CONFIG['instrument']}")
    logger.info(f"Trading Window: {CONFIG['trading_window']['start']} - {CONFIG['trading_window']['end']} ET")
    logger.info(f"Max Trades/Day: {CONFIG['max_trades_per_day']}")
    logger.info(f"Daily Loss Limit: ${CONFIG['daily_loss_limit']}")
    logger.info(f"Max Drawdown: {CONFIG['max_drawdown_percent']}%")
    logger.info("="*60)
    
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
