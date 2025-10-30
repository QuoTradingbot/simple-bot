"""
VWAP Bounce Bot - Mean Reversion Trading Strategy
Event-driven bot that trades bounces off VWAP standard deviation bands
"""

import os
import logging
from datetime import datetime, time, timedelta
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Callable
import pytz

# Import new production modules
from config import load_config, BotConfiguration
from event_loop import EventLoop, EventType, EventPriority, TimerManager
from error_recovery import ErrorRecoveryManager, ErrorType as RecoveryErrorType
from bid_ask_manager import BidAskManager, BidAskQuote

# Conditionally import broker (only needed for live trading, not backtesting)
try:
    from broker_interface import create_broker, BrokerInterface
except ImportError:
    # Broker interface not available (e.g., in backtest-only mode)
    create_broker = None
    BrokerInterface = None

# Load configuration from environment and config module
# Check if in backtest mode via environment variable
_backtest_mode = os.getenv("BOT_BACKTEST_MODE", "false").lower() == "true"
_bot_config = load_config(backtest_mode=_backtest_mode)
_bot_config.validate()  # Validate configuration at startup

# Convert BotConfiguration to dictionary for backward compatibility with existing code
CONFIG: Dict[str, Any] = _bot_config.to_dict()

# String constants
MSG_LIVE_TRADING_NOT_IMPLEMENTED = "Live trading not implemented - SDK integration required"
SEPARATOR_LINE = "=" * 60

# Global broker instance (replaces sdk_client)
broker: Optional[BrokerInterface] = None

# Global event loop instance
event_loop: Optional[EventLoop] = None

# Global error recovery manager
recovery_manager: Optional[ErrorRecoveryManager] = None

# Global timer manager
timer_manager: Optional[TimerManager] = None

# Global bid/ask manager
bid_ask_manager: Optional[BidAskManager] = None

# State management dictionary
state: Dict[str, Any] = {}

# Backtest mode: Track current simulation time (for backtesting)
# When None, uses real datetime.now(). When set, uses this timestamp.
backtest_current_time: Optional[datetime] = None

# Global tracking for safety mechanisms (Phase 12)
bot_status: Dict[str, Any] = {
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
    # PRODUCTION: Track trading costs
    "total_slippage_cost": 0.0,  # Total slippage costs across all trades
    "total_commission": 0.0,  # Total commissions across all trades
}


def setup_logging() -> logging.Logger:
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

def initialize_broker() -> None:
    """
    Initialize the broker interface using configuration.
    Uses TopStep broker with error recovery and circuit breaker.
    """
    global broker, recovery_manager
    
    logger.info("Initializing broker interface...")
    
    # Create error recovery manager
    recovery_manager = ErrorRecoveryManager(CONFIG)
    
    # Create broker using configuration
    broker = create_broker(_bot_config.api_token)
    
    # Connect with error recovery
    breaker = recovery_manager.get_circuit_breaker("broker_connection")
    success, result = breaker.call(broker.connect)
    
    if not success:
        logger.error("Failed to connect to broker")
        raise RuntimeError("Broker connection failed")
    
    logger.info("Broker connected successfully")


def get_account_equity() -> float:
    """
    Fetch current account equity from broker.
    Returns account equity/balance with error handling.
    In backtest mode, returns initial capital from backtest engine.
    """
    # In backtest mode, broker is None - return default starting capital
    if _bot_config.backtest_mode or broker is None:
        # Backtest mode - use initial_capital from bot_status if available
        if bot_status.get("starting_equity") is not None:
            return bot_status["starting_equity"]
        # Default starting capital for backtesting
        return 25000.0
    
    try:
        # Use circuit breaker for account query
        breaker = recovery_manager.get_circuit_breaker("account_query")
        success, equity = breaker.call(broker.get_account_equity)
        
        if success:
            logger.info(f"Account equity: ${equity:.2f}")
            return equity
        else:
            logger.error("Failed to get account equity")
            return 0.0
    except Exception as e:
        logger.error(f"Error fetching account equity: {e}")
        action = recovery_manager.handle_error(
            RecoveryErrorType.SDK_CRASH,
            {"error": str(e), "function": "get_account_equity"}
        )
        return 0.0


def place_market_order(symbol: str, side: str, quantity: int) -> Optional[Dict[str, Any]]:
    """
    Place a market order through the broker interface.
    
    Args:
        symbol: Instrument symbol (e.g., 'MES')
        side: 'BUY' or 'SELL'
        quantity: Number of contracts
    
    Returns:
        Order object or None if failed
    """
    logger.info(f"{'[DRY RUN] ' if CONFIG['dry_run'] else ''}Market Order: {side} {quantity} {symbol}")
    
    # In backtest or dry-run mode, return simulated order
    if CONFIG["dry_run"] or _bot_config.backtest_mode:
        return {
            "order_id": f"BACKTEST_{datetime.now().timestamp()}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": "MARKET",
            "status": "FILLED",
            "dry_run": True
        }
    
    if broker is None:
        logger.error("Broker not initialized")
        return None
    
    try:
        # Use circuit breaker for order placement
        breaker = recovery_manager.get_circuit_breaker("order_placement")
        success, order = breaker.call(broker.place_market_order, symbol, side, quantity)
        
        if success and order:
            # Post order fill event to event loop
            if event_loop:
                event_loop.post_event(
                    EventType.ORDER_FILL,
                    EventPriority.HIGH,
                    {"order": order, "symbol": symbol}
                )
            return order
        else:
            logger.error("Market order placement failed")
            action = recovery_manager.handle_error(
                RecoveryErrorType.ORDER_REJECTION,
                {"symbol": symbol, "side": side, "quantity": quantity}
            )
            return None
    except Exception as e:
        logger.error(f"Error placing market order: {e}")
        action = recovery_manager.handle_error(
            RecoveryErrorType.SDK_CRASH,
            {"error": str(e), "function": "place_market_order"}
        )
        return None


def place_stop_order(symbol: str, side: str, quantity: int, stop_price: float) -> Optional[Dict[str, Any]]:
    """
    Place a stop order through the broker interface.
    
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
            "order_id": f"DRY_RUN_STOP_{datetime.now().timestamp()}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": "STOP",
            "stop_price": stop_price,
            "status": "PENDING",
            "dry_run": True
        }
    
    if broker is None:
        logger.error("Broker not initialized")
        return None
    
    try:
        # Use circuit breaker for order placement
        breaker = recovery_manager.get_circuit_breaker("order_placement")
        success, order = breaker.call(broker.place_stop_order, symbol, side, quantity, stop_price)
        
        if success and order:
            return order
        else:
            logger.error("Stop order placement failed")
            action = recovery_manager.handle_error(
                RecoveryErrorType.ORDER_REJECTION,
                {"symbol": symbol, "side": side, "quantity": quantity, "stop_price": stop_price}
            )
            return None
    except Exception as e:
        logger.error(f"Error placing stop order: {e}")
        action = recovery_manager.handle_error(
            RecoveryErrorType.SDK_CRASH,
            {"error": str(e), "function": "place_stop_order"}
        )
        return None


def place_limit_order(symbol: str, side: str, quantity: int, limit_price: float) -> Optional[Dict[str, Any]]:
    """
    Place a limit order through the broker interface.
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
            "order_id": f"DRY_RUN_LIMIT_{datetime.now().timestamp()}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": "LIMIT",
            "limit_price": limit_price,
            "status": "PENDING",
            "dry_run": True
        }
    
    if broker is None:
        logger.error("Broker not initialized")
        return None
    
    try:
        # Use circuit breaker for order placement
        breaker = recovery_manager.get_circuit_breaker("order_placement")
        success, order = breaker.call(broker.place_limit_order, symbol, side, quantity, limit_price)
        
        if success and order:
            return order
        else:
            logger.error("Limit order placement failed")
            action = recovery_manager.handle_error(
                RecoveryErrorType.ORDER_REJECTION,
                {"symbol": symbol, "side": side, "quantity": quantity, "limit_price": limit_price}
            )
            return None
    except Exception as e:
        logger.error(f"Error placing limit order: {e}")
        action = recovery_manager.handle_error(
            RecoveryErrorType.SDK_CRASH,
            {"error": str(e), "function": "place_limit_order"}
        )
        return None


def get_position_quantity(symbol: str) -> int:
    """
    Query broker for current position quantity.
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
    
    if broker is None:
        logger.error("Broker not initialized")
        return 0
    
    try:
        # Use circuit breaker for position query
        breaker = recovery_manager.get_circuit_breaker("account_query")
        success, quantity = breaker.call(broker.get_position_quantity, symbol)
        
        if success:
            # Check for position discrepancy
            if state.get(symbol) and state[symbol]["position"]["active"]:
                expected_qty = state[symbol]["position"]["quantity"]
                expected_side = state[symbol]["position"]["side"]
                expected = expected_qty if expected_side == "long" else -expected_qty
                
                if quantity != expected:
                    logger.warning(f"Position discrepancy: Expected {expected}, got {quantity}")
                    action = recovery_manager.handle_error(
                        RecoveryErrorType.POSITION_DISCREPANCY,
                        {"symbol": symbol, "expected": expected, "actual": quantity}
                    )
            
            return quantity
        else:
            logger.error("Failed to get position quantity")
            return 0
    except Exception as e:
        logger.error(f"Error querying position: {e}")
        action = recovery_manager.handle_error(
            RecoveryErrorType.SDK_CRASH,
            {"error": str(e), "function": "get_position_quantity"}
        )
        return 0


def subscribe_market_data(symbol: str, callback: Callable[[str, float, int, int], None]) -> None:
    """
    Subscribe to real-time market data for a symbol through broker interface.
    
    Args:
        symbol: Instrument symbol
        callback: Function to call with tick data (symbol, price, volume, timestamp)
    """
    logger.info(f"{'[DRY RUN] ' if CONFIG['dry_run'] else ''}Subscribing to market data: {symbol}")
    
    if CONFIG["dry_run"]:
        logger.info(f"Dry run mode - skipping real broker subscription for {symbol}")
        return
    
    if broker is None:
        logger.error("Broker not initialized")
        return
    
    try:
        # Subscribe through broker interface
        broker.subscribe_market_data(symbol, callback)
        logger.info(f"Subscribed to market data for {symbol}")
    except Exception as e:
        logger.error(f"Error subscribing to market data: {e}")
        action = recovery_manager.handle_error(
            RecoveryErrorType.DATA_FEED_INTERRUPTION,
            {"symbol": symbol, "error": str(e)}
        )


def fetch_historical_bars(symbol: str, timeframe: int, count: int) -> List[Dict[str, Any]]:
    """
    Fetch historical bars for initial trend calculation through broker interface.
    
    Args:
        symbol: Instrument symbol
        timeframe: Bar timeframe in minutes
        count: Number of bars to fetch
    
    Returns:
        List of bar dictionaries with OHLCV data
    """
    logger.info(f"Fetching {count} historical {timeframe}min bars for {symbol}")
    
    if CONFIG["dry_run"]:
        logger.info("Dry run mode - returning empty bars")
        return []
    
    if broker is None:
        logger.error("Broker not initialized")
        return []
    
    try:
        # Fetch through broker interface
        breaker = recovery_manager.get_circuit_breaker("market_data")
        success, bars = breaker.call(broker.fetch_historical_bars, symbol, f"{timeframe}m", count)
        
        if success and bars:
            logger.info(f"Fetched {len(bars)} bars")
            return bars
        else:
            logger.warning("Failed to fetch historical bars")
            return []
    except Exception as e:
        logger.error(f"Error fetching historical bars: {e}")
        action = recovery_manager.handle_error(
            RecoveryErrorType.SDK_CRASH,
            {"error": str(e), "function": "fetch_historical_bars"}
        )
        return []


# ============================================================================
# PHASE THREE: State Management
# ============================================================================

def initialize_state(symbol: str) -> None:
    """
    Initialize state tracking for an instrument.
    
    Args:
        symbol: Instrument symbol
    """
    state[symbol] = {
        # Tick data storage
        "ticks": deque(maxlen=CONFIG.get("max_tick_storage", 10000)),
        
        # Bar storage
        "bars_1min": deque(maxlen=CONFIG.get("max_bars_storage", 200)),
        "bars_15min": deque(maxlen=100),
        
        # Current incomplete bars
        "current_1min_bar": None,
        "current_15min_bar": None,
        
        # VWAP calculation data
        "vwap": None,
        "vwap_bands": {
            "upper_1": None,
            "upper_2": None,
            "upper_3": None,
            "lower_1": None,
            "lower_2": None,
            "lower_3": None
        },
        "vwap_std_dev": None,
        "vwap_day": None,  # Phase Three: Track VWAP day separately
        
        # Trend filter
        "trend_ema": None,
        "trend_direction": None,  # 'up', 'down', or 'neutral'
        
        # Technical indicators
        "rsi": None,  # RSI value (0-100)
        "macd": None,  # MACD data dict with 'macd', 'signal', 'histogram'
        "avg_volume": None,  # Average volume for spike detection
        
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

def on_quote(symbol: str, bid_price: float, ask_price: float, bid_size: int, 
             ask_size: int, last_price: float, timestamp_ms: int) -> None:
    """
    Handle incoming bid/ask quote data.
    Updates bid/ask manager with real-time quote information.
    
    Args:
        symbol: Instrument symbol
        bid_price: Current bid price
        ask_price: Current ask price
        bid_size: Bid size (contracts)
        ask_size: Ask size (contracts)
        last_price: Last trade price
        timestamp_ms: Quote timestamp in milliseconds
    """
    if bid_ask_manager is not None:
        bid_ask_manager.update_quote(
            symbol=symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            last_price=last_price,
            timestamp=timestamp_ms
        )


def on_tick(symbol: str, price: float, volume: int, timestamp_ms: int) -> None:
    """
    Handle incoming tick data by posting to event loop.
    
    Args:
        symbol: Instrument symbol
        price: Tick price
        volume: Tick volume
        timestamp_ms: Timestamp in milliseconds
    """
    # Update backtest current time from tick timestamp
    global backtest_current_time
    if _bot_config.backtest_mode:
        backtest_current_time = datetime.fromtimestamp(timestamp_ms / 1000, tz=pytz.timezone(CONFIG["timezone"]))
    
    # Post tick data to event loop for processing
    if event_loop:
        event_loop.post_event(
            EventType.TICK_DATA,
            EventPriority.MEDIUM,
            {
                "symbol": symbol,
                "price": price,
                "volume": volume,
                "timestamp": timestamp_ms
            }
        )
    else:
        # Fallback if event loop not initialized (shouldn't happen in production)
        logger.warning("Event loop not initialized, processing tick directly")
        handle_tick_event({
            "symbol": symbol,
            "price": price,
            "volume": volume,
            "timestamp": timestamp_ms
        })


def update_1min_bar(symbol: str, price: float, volume: int, dt: datetime) -> None:
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


def update_15min_bar(symbol: str, price: float, volume: int, dt: datetime) -> None:
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
            # Update all indicators after new bar is added
            update_trend_filter(symbol)
            update_rsi(symbol)
            update_macd(symbol)
            update_volume_average(symbol)
        
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


def update_trend_filter(symbol: str) -> None:
    """
    Update the trend filter using EMA of 15-minute bars.
    
    Args:
        symbol: Instrument symbol
    """
    bars = state[symbol]["bars_15min"]
    period = CONFIG.get("trend_ema_period", 20)
    
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


def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """
    Calculate Relative Strength Index.
    
    Args:
        prices: List of closing prices
        period: RSI period (default 14)
    
    Returns:
        RSI value (0-100) or None if insufficient data
    """
    if len(prices) < period + 1:
        return None
    
    # Calculate price changes
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    
    # Separate gains and losses
    gains = [change if change > 0 else 0 for change in changes]
    losses = [-change if change < 0 else 0 for change in changes]
    
    # Calculate initial average gain and loss (SMA)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    # Calculate smoothed averages (EMA style)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    # Calculate RSI
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def calculate_macd(prices: List[float], fast_period: int = 12, 
                   slow_period: int = 26, signal_period: int = 9) -> Optional[Dict[str, float]]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: List of closing prices
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)
    
    Returns:
        Dictionary with 'macd', 'signal', 'histogram' or None if insufficient data
    """
    if len(prices) < slow_period + signal_period:
        return None
    
    # Calculate fast and slow EMAs
    fast_ema = calculate_ema(prices, fast_period)
    slow_ema = calculate_ema(prices, slow_period)
    
    if fast_ema is None or slow_ema is None:
        return None
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate MACD values for signal line
    # We need to calculate MACD for each point to get signal line
    macd_values = []
    for i in range(slow_period, len(prices) + 1):
        fast = calculate_ema(prices[:i], fast_period)
        slow = calculate_ema(prices[:i], slow_period)
        if fast is not None and slow is not None:
            macd_values.append(fast - slow)
    
    if len(macd_values) < signal_period:
        return None
    
    # Calculate signal line (EMA of MACD)
    signal_line = calculate_ema(macd_values, signal_period)
    
    if signal_line is None:
        return None
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram
    }


def update_rsi(symbol: str) -> None:
    """
    Update RSI indicator for the symbol.
    
    Args:
        symbol: Instrument symbol
    """
    bars = state[symbol]["bars_15min"]
    rsi_period = CONFIG.get("rsi_period", 14)
    
    if len(bars) < rsi_period + 1:
        logger.debug(f"Not enough bars for RSI: {len(bars)}/{rsi_period + 1}")
        return
    
    closes = [bar["close"] for bar in bars]
    rsi = calculate_rsi(closes, rsi_period)
    
    if rsi is not None:
        state[symbol]["rsi"] = rsi
        logger.debug(f"RSI: {rsi:.2f}")


def update_macd(symbol: str) -> None:
    """
    Update MACD indicator for the symbol.
    
    Args:
        symbol: Instrument symbol
    """
    bars = state[symbol]["bars_15min"]
    
    # Get MACD parameters from config
    fast_period = CONFIG.get("macd_fast", 12)
    slow_period = CONFIG.get("macd_slow", 26)
    signal_period = CONFIG.get("macd_signal", 9)
    
    if len(bars) < slow_period + signal_period:
        logger.debug(f"Not enough bars for MACD: {len(bars)}/{slow_period + signal_period}")
        return
    
    closes = [bar["close"] for bar in bars]
    macd_data = calculate_macd(closes, fast_period, slow_period, signal_period)
    
    if macd_data is not None:
        state[symbol]["macd"] = macd_data
        logger.debug(f"MACD: {macd_data['macd']:.2f}, Signal: {macd_data['signal']:.2f}, "
                    f"Histogram: {macd_data['histogram']:.2f}")


def update_volume_average(symbol: str) -> None:
    """
    Update average volume for spike detection.
    
    Args:
        symbol: Instrument symbol
    """
    bars = state[symbol]["bars_15min"]
    lookback = CONFIG.get("volume_lookback", 20)
    
    if len(bars) < lookback:
        logger.debug(f"Not enough bars for volume average: {len(bars)}/{lookback}")
        return
    
    # Calculate average volume over lookback period
    recent_bars = list(bars)[-lookback:]
    volumes = [bar["volume"] for bar in recent_bars]
    avg_volume = sum(volumes) / len(volumes)
    
    state[symbol]["avg_volume"] = avg_volume
    logger.debug(f"Average volume (last {lookback} bars): {avg_volume:.0f}")


# ============================================================================
# PHASE FIVE: VWAP Calculation
# ============================================================================

def calculate_vwap(symbol: str) -> None:
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
    
    # Calculate bands using configured standard deviation multipliers
    band_1_mult = CONFIG.get("vwap_std_dev_1", 1.5)
    band_2_mult = CONFIG.get("vwap_std_dev_2", 2.0)
    band_3_mult = CONFIG.get("vwap_std_dev_3", 3.0)
    state[symbol]["vwap_bands"]["upper_1"] = vwap + (std_dev * band_1_mult)
    state[symbol]["vwap_bands"]["upper_2"] = vwap + (std_dev * band_2_mult)
    state[symbol]["vwap_bands"]["upper_3"] = vwap + (std_dev * band_3_mult)
    state[symbol]["vwap_bands"]["lower_1"] = vwap - (std_dev * band_1_mult)
    state[symbol]["vwap_bands"]["lower_2"] = vwap - (std_dev * band_2_mult)
    state[symbol]["vwap_bands"]["lower_3"] = vwap - (std_dev * band_3_mult)
    
    logger.debug(f"VWAP: {vwap:.2f}, StdDev: {std_dev:.2f}")
    logger.debug(f"Bands - U3: {state[symbol]['vwap_bands']['upper_3']:.2f}, "
                f"U2: {state[symbol]['vwap_bands']['upper_2']:.2f}, "
                f"U1: {state[symbol]['vwap_bands']['upper_1']:.2f}, "
                f"L1: {state[symbol]['vwap_bands']['lower_1']:.2f}, "
                f"L2: {state[symbol]['vwap_bands']['lower_2']:.2f}, "
                f"L3: {state[symbol]['vwap_bands']['lower_3']:.2f}")


# ============================================================================
# PHASE SEVEN: Signal Generation Logic
# ============================================================================

def validate_signal_requirements(symbol: str, bar_time: datetime) -> Tuple[bool, Optional[str]]:
    """
    Validate that all requirements are met for signal generation.
    24/5 trading - signals allowed anytime except maintenance/weekend.
    
    Args:
        symbol: Instrument symbol
        bar_time: Current bar timestamp
    
    Returns:
        Tuple of (is_valid, reason)
    """
    # Check trading state
    trading_state = get_trading_state(bar_time)
    if trading_state == "maintenance":
        logger.debug(f"Maintenance window (5-6 PM), skipping signal check")
        return False, f"Maintenance window"
    elif trading_state == "weekend":
        logger.debug(f"Weekend, skipping signal check")
        return False, f"Weekend"
    
    # Friday restriction - close before weekend
    if bar_time.weekday() == 4 and bar_time.time() >= CONFIG["friday_entry_cutoff"]:
        log_time_based_action(
            "friday_entry_blocked",
            f"Friday after {CONFIG['friday_entry_cutoff']}, no new trades to avoid weekend gap risk",
            {"day": "Friday", "time": bar_time.strftime('%H:%M:%S')}
        )
        logger.info(f"Friday after {CONFIG['friday_entry_cutoff']} - no new trades (weekend gap risk)")
        return False, "Friday entry cutoff"
    
    # Check if already have position
    if state[symbol]["position"]["active"]:
        logger.debug("Position already active, skipping signal generation")
        return False, "Position active"
    
    # Check daily trade limit
    if state[symbol]["daily_trade_count"] >= CONFIG["max_trades_per_day"]:
        logger.warning(f"Daily trade limit reached ({CONFIG['max_trades_per_day']}), stopping for the day")
        return False, "Daily trade limit"
    
    # Check daily loss limit
    if state[symbol]["daily_pnl"] <= -CONFIG["daily_loss_limit"]:
        logger.warning(f"Daily loss limit hit (${state[symbol]['daily_pnl']:.2f}), stopping for the day")
        return False, "Daily loss limit"
    
    # Check data availability
    if len(state[symbol]["bars_1min"]) < 2:
        logger.info(f"Not enough bars for signal: {len(state[symbol]['bars_1min'])}/2")
        return False, "Insufficient bars"
    
    # Check VWAP bands
    vwap_bands = state[symbol]["vwap_bands"]
    if any(v is None for v in vwap_bands.values()):
        logger.info("VWAP bands not yet calculated")
        return False, "VWAP not ready"
    
    # Check trend (optional - can be disabled)
    use_trend_filter = CONFIG.get("use_trend_filter", False)
    if use_trend_filter:
        trend = state[symbol]["trend_direction"]
        if trend is None or trend == "neutral":
            logger.info(f"Trend not established or neutral: {trend}")
            return False, "Trend not established"
    
    # Check RSI (optional - for extreme overbought/oversold confirmation)
    use_rsi_filter = CONFIG.get("use_rsi_filter", True)
    rsi_oversold = CONFIG.get("rsi_oversold", 20.0)
    rsi_overbought = CONFIG.get("rsi_overbought", 80.0)
    
    if use_rsi_filter:
        rsi = state[symbol]["rsi"]
        if rsi is None:
            logger.debug("RSI not yet calculated")
            # Allow trading without RSI if not available yet
        # Note: RSI check moved to signal-specific functions for long/short
    
    # Check volume spike (optional - for confirmation)
    use_volume_filter = CONFIG.get("use_volume_filter", True)
    if use_volume_filter:
        avg_volume = state[symbol]["avg_volume"]
        if avg_volume is None:
            logger.debug("Average volume not yet calculated")
            # Allow trading without volume filter if not available yet
    
    # VWAP direction filter (optional - price vs VWAP for bias)
    use_vwap_direction_filter = CONFIG.get("use_vwap_direction_filter", True)
    if use_vwap_direction_filter:
        vwap = state[symbol]["vwap"]
        if vwap is None:
            logger.debug("VWAP not yet calculated")
            return False, "VWAP not ready"
        # Note: VWAP direction check moved to signal-specific functions
    
    # Check bid/ask spread and market condition (Phase: Bid/Ask Strategy)
    if bid_ask_manager is not None:
        # Validate spread (Requirement 8)
        is_acceptable, spread_reason = bid_ask_manager.validate_entry_spread(symbol)
        if not is_acceptable:
            logger.info(f"Spread check failed: {spread_reason}")
            return False, spread_reason
        
        # Classify market condition (Requirement 11)
        try:
            condition, condition_reason = bid_ask_manager.classify_market_condition(symbol)
            logger.info(f"Market Condition: {condition.upper()} - {condition_reason}")
            
            # Skip trading in stressed markets
            if condition == "stressed":
                logger.warning("Market is stressed - skipping trade")
                return False, "Stressed market conditions"
            
            # Warn about illiquid markets (position size already adjusted in execute_entry)
            if condition == "illiquid":
                logger.warning("Illiquid market detected - position size will be adjusted")
        except Exception as e:
            logger.warning(f"Could not classify market: {e}")
    
    return True, None


def check_long_signal_conditions(symbol: str, prev_bar: Dict[str, Any], 
                                 current_bar: Dict[str, Any]) -> bool:
    """
    Check if long signal conditions are met - PROFESSIONAL OPTIMIZED VERSION.
    
    Strategy: Mean reversion from VWAP lower band with multi-factor confirmation
    
    Args:
        symbol: Instrument symbol
        prev_bar: Previous 1-minute bar
        current_bar: Current 1-minute bar
    
    Returns:
        True if long signal detected
    """
    vwap_bands = state[symbol]["vwap_bands"]
    vwap = state[symbol]["vwap"]
    
    # PRIMARY: VWAP bounce condition (2.5 std dev = high quality setup)
    touched_lower = prev_bar["low"] <= vwap_bands["lower_2"]
    bounced_back = current_bar["close"] > vwap_bands["lower_2"]
    
    if not (touched_lower and bounced_back):
        return False
    
    # FILTER 1: VWAP Direction - price should be BELOW VWAP (discount/oversold)
    use_vwap_direction = CONFIG.get("use_vwap_direction_filter", True)
    if use_vwap_direction and vwap is not None:
        if current_bar["close"] >= vwap:
            logger.debug(f"Long rejected - price above VWAP: {current_bar['close']:.2f} >= {vwap:.2f}")
            return False
        logger.debug(f"Price below VWAP: {current_bar['close']:.2f} < {vwap:.2f} ✓")
    
    # FILTER 2: RSI - extreme oversold (< 20 instead of < 40)
    use_rsi = CONFIG.get("use_rsi_filter", True)
    rsi_oversold = CONFIG.get("rsi_oversold", 20.0)
    if use_rsi:
        rsi = state[symbol]["rsi"]
        if rsi is not None:
            if rsi >= rsi_oversold:
                logger.debug(f"Long rejected - RSI not extreme: {rsi:.2f} >= {rsi_oversold}")
                return False
            logger.debug(f"RSI extreme oversold: {rsi:.2f} < {rsi_oversold} ✓")
    
    # FILTER 3: Volume spike - confirmation of interest
    use_volume = CONFIG.get("use_volume_filter", True)
    volume_mult = CONFIG.get("volume_spike_multiplier", 1.5)
    if use_volume:
        avg_volume = state[symbol]["avg_volume"]
        if avg_volume is not None and avg_volume > 0:
            current_volume = current_bar["volume"]
            if current_volume < avg_volume * volume_mult:
                logger.debug(f"Long rejected - no volume spike: {current_volume} < {avg_volume * volume_mult:.0f}")
                return False
            logger.debug(f"Volume spike: {current_volume} >= {avg_volume * volume_mult:.0f} ✓")
    
    logger.info(f"✅ LONG SIGNAL: VWAP bounce at {current_bar['close']:.2f} (band: {vwap_bands['lower_2']:.2f})")
    return True


def check_short_signal_conditions(symbol: str, prev_bar: Dict[str, Any], 
                                  current_bar: Dict[str, Any]) -> bool:
    """
    Check if short signal conditions are met - PROFESSIONAL OPTIMIZED VERSION.
    
    Strategy: Mean reversion from VWAP upper band with multi-factor confirmation
    
    Args:
        symbol: Instrument symbol
        prev_bar: Previous 1-minute bar
        current_bar: Current 1-minute bar
    
    Returns:
        True if short signal detected
    """
    vwap_bands = state[symbol]["vwap_bands"]
    vwap = state[symbol]["vwap"]
    
    # PRIMARY: VWAP bounce condition (2.5 std dev = high quality setup)
    touched_upper = prev_bar["high"] >= vwap_bands["upper_2"]
    bounced_back = current_bar["close"] < vwap_bands["upper_2"]
    
    if not (touched_upper and bounced_back):
        return False
    
    # FILTER 1: VWAP Direction - price should be ABOVE VWAP (premium/overbought)
    use_vwap_direction = CONFIG.get("use_vwap_direction_filter", True)
    if use_vwap_direction and vwap is not None:
        if current_bar["close"] <= vwap:
            logger.debug(f"Short rejected - price below VWAP: {current_bar['close']:.2f} <= {vwap:.2f}")
            return False
        logger.debug(f"Price above VWAP: {current_bar['close']:.2f} > {vwap:.2f} ✓")
    
    # FILTER 2: RSI - extreme overbought (> 80 instead of > 60)
    use_rsi = CONFIG.get("use_rsi_filter", True)
    rsi_overbought = CONFIG.get("rsi_overbought", 80.0)
    if use_rsi:
        rsi = state[symbol]["rsi"]
        if rsi is not None:
            if rsi <= rsi_overbought:
                logger.debug(f"Short rejected - RSI not extreme: {rsi:.2f} <= {rsi_overbought}")
                return False
            logger.debug(f"RSI extreme overbought: {rsi:.2f} > {rsi_overbought} ✓")
    
    # FILTER 3: Volume spike - confirmation of interest
    use_volume = CONFIG.get("use_volume_filter", True)
    volume_mult = CONFIG.get("volume_spike_multiplier", 1.5)
    if use_volume:
        avg_volume = state[symbol]["avg_volume"]
        if avg_volume is not None and avg_volume > 0:
            current_volume = current_bar["volume"]
            if current_volume < avg_volume * volume_mult:
                logger.debug(f"Short rejected - no volume spike: {current_volume} < {avg_volume * volume_mult:.0f}")
                return False
            logger.debug(f"Volume spike: {current_volume} >= {avg_volume * volume_mult:.0f} ✓")
    
    logger.info(f"✅ SHORT SIGNAL: VWAP bounce at {current_bar['close']:.2f} (band: {vwap_bands['upper_2']:.2f})")
    return True


def check_for_signals(symbol: str) -> None:
    """
    Check for trading signals on each completed 1-minute bar.
    Coordinates signal detection through helper functions.
    
    Args:
        symbol: Instrument symbol
    """
    print(f"[DEBUG] check_for_signals called for {symbol}, bars: {len(state.get(symbol, {}).get('bars_1min', []))}")
    
    # Check safety conditions first
    is_safe, reason = check_safety_conditions(symbol)
    if not is_safe:
        print(f"[DEBUG] Safety check failed: {reason}")
        logger.debug(f"[BACKTEST] Safety check failed: {reason}")
        return
    
    # Get the latest bar
    if len(state[symbol]["bars_1min"]) == 0:
        logger.debug(f"[BACKTEST] No 1-min bars yet")
        return
    
    latest_bar = state[symbol]["bars_1min"][-1]
    bar_time = latest_bar["timestamp"]
    
    # Validate signal requirements
    is_valid, reason = validate_signal_requirements(symbol, bar_time)
    if not is_valid:
        if reason not in ["Position active", "Insufficient bars"]:
            print(f"[DEBUG] Signal validation failed: {reason} at {bar_time.strftime('%Y-%m-%d %H:%M:%S %Z')} (weekday={bar_time.weekday()})")
        logger.debug(f"[BACKTEST] Signal validation failed: {reason} at {bar_time}")
        return
    
    # Get bars for signal check
    prev_bar = state[symbol]["bars_1min"][-2]
    current_bar = state[symbol]["bars_1min"][-1]
    vwap_bands = state[symbol]["vwap_bands"]
    trend = state[symbol]["trend_direction"]
    
    logger.debug(f"Signal check: trend={trend}, prev_low={prev_bar['low']:.2f}, "
                f"current_close={current_bar['close']:.2f}, lower_band_2={vwap_bands['lower_2']:.2f}")
    
    # Check for long signal
    if check_long_signal_conditions(symbol, prev_bar, current_bar):
        execute_entry(symbol, "long", current_bar["close"])
        return
    
    # Check for short signal
    if check_short_signal_conditions(symbol, prev_bar, current_bar):
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
    
    # Determine stop price - TIGHTER STOPS FOR MEAN REVERSION
    vwap_bands = state[symbol]["vwap_bands"]
    vwap = state[symbol]["vwap"]
    tick_size = CONFIG["tick_size"]
    
    # IMPROVED: Use optimal stops (11 ticks) - sweet spot between tight and too tight
    # Mean reversion = expect quick bounce, not slow grind
    max_stop_ticks = 11  # Optimized to 11 ticks ($13.75 max risk per contract)
    
    if side == "long":
        # Stop 12 ticks below entry (or at lower band 3, whichever is tighter)
        band_stop = vwap_bands["lower_3"] - (2 * tick_size)  # 2 tick buffer
        tight_stop = entry_price - (max_stop_ticks * tick_size)
        stop_price = max(tight_stop, band_stop)  # Use tighter of the two
    else:  # short
        # Stop 12 ticks above entry (or at upper band 3, whichever is tighter)
        band_stop = vwap_bands["upper_3"] + (2 * tick_size)  # 2 tick buffer
        tight_stop = entry_price + (max_stop_ticks * tick_size)
        stop_price = min(tight_stop, band_stop)  # Use tighter of the two
    
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
    
    # Calculate target price - IMPROVED: Mean reversion to VWAP + risk/reward
    # Option 1: Traditional R/R ratio target
    traditional_target_distance = stop_distance * CONFIG["risk_reward_ratio"]
    
    # Option 2: VWAP center (mean reversion target)
    if side == "long":
        vwap_reversion_distance = vwap - entry_price
        traditional_target = entry_price + traditional_target_distance
    else:
        vwap_reversion_distance = entry_price - vwap
        traditional_target = entry_price - traditional_target_distance
    
    # Use the CLOSER of the two targets (more conservative, higher win rate)
    # This ensures we're taking profits when price reverts to mean
    if side == "long":
        target_price = min(traditional_target, entry_price + vwap_reversion_distance)
    else:
        target_price = max(traditional_target, entry_price - vwap_reversion_distance)
    
    target_price = round_to_tick(target_price)
    target_distance = abs(target_price - entry_price)
    
    logger.info(f"Position sizing: {contracts} contract(s)")
    logger.info(f"  Entry: ${entry_price:.2f}, Stop: ${stop_price:.2f}, Target: ${target_price:.2f}")
    logger.info(f"  Risk: {ticks_at_risk:.1f} ticks (${risk_per_contract:.2f})")
    logger.info(f"  Reward: {target_distance/tick_size:.1f} ticks ({target_distance/stop_distance:.1f}:1 R/R)")
    logger.info(f"  VWAP: ${vwap:.2f} (mean reversion target)")
    
    return contracts, stop_price, target_price


# ============================================================================
# PHASE NINE: Entry Execution
# ============================================================================

def execute_entry(symbol: str, side: str, entry_price: float) -> None:
    """
    Execute entry order with stop loss and target.
    Uses intelligent bid/ask order placement strategy.
    
    Args:
        symbol: Instrument symbol
        side: 'long' or 'short'
        entry_price: Approximate entry price (mid or last)
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
    entry_time = get_current_time()
    trading_state = get_trading_state(entry_time)
    
    if trading_state not in ["entry_window"]:
        logger.warning(SEPARATOR_LINE)
        logger.warning("ENTRY ABORTED - No longer in entry window")
        logger.warning(f"  Current state: {trading_state}")
        logger.warning(f"  Time: {entry_time.strftime('%H:%M:%S %Z')}")
        logger.warning(SEPARATOR_LINE)
        return
    
    logger.info(SEPARATOR_LINE)
    logger.info(f"ENTERING {side.upper()} POSITION")
    logger.info(f"  Time: {entry_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"  Symbol: {symbol}")
    
    # Spread-aware position sizing (Requirement 10)
    original_contracts = contracts
    if bid_ask_manager is not None:
        try:
            expected_profit_ticks = abs(target_price - entry_price) / CONFIG["tick_size"]
            adjusted_contracts, cost_breakdown = bid_ask_manager.calculate_spread_aware_position_size(
                symbol, contracts, expected_profit_ticks
            )
            if adjusted_contracts != original_contracts:
                logger.warning(f"  Position size adjusted: {original_contracts} -> {adjusted_contracts} contracts")
                logger.info(f"  Spread cost: {cost_breakdown['cost_percentage']:.1f}% of expected profit")
                contracts = adjusted_contracts
        except Exception as e:
            logger.warning(f"  Spread-aware sizing unavailable: {e}")
    
    logger.info(f"  Contracts: {contracts}")
    logger.info(f"  Stop Loss: ${stop_price:.2f}")
    logger.info(f"  Target: ${target_price:.2f}")
    
    # Track order execution details for post-trade analysis
    fill_start_time = datetime.now()
    order_type_used = "market"  # Default
    
    # Get intelligent order placement strategy from bid/ask manager
    order_side = "BUY" if side == "long" else "SELL"
    actual_fill_price = entry_price
    order = None
    
    if bid_ask_manager is not None:
        try:
            # Get order parameters from bid/ask manager
            order_params = bid_ask_manager.get_entry_order_params(symbol, side, contracts)
            
            logger.info(f"  Order Strategy: {order_params['strategy']}")
            logger.info(f"  Reason: {order_params['reason']}")
            
            if order_params['strategy'] == 'passive':
                # Try passive entry first (at bid for long, at ask for short)
                limit_price = order_params['limit_price']
                logger.info(f"  Passive Entry: ${limit_price:.2f} (saving spread)")
                logger.info(f"  Timeout: {order_params['timeout']}s")
                
                order = place_limit_order(symbol, order_side, contracts, limit_price)
                if order:
                    # Wait for fill or timeout
                    import time
                    time.sleep(order_params['timeout'])
                    
                    # Check if filled
                    current_position = get_position_quantity(symbol)
                    expected_qty = contracts if side == "long" else -contracts
                    
                    if abs(current_position) == abs(expected_qty):
                        # Successfully filled at passive price
                        actual_fill_price = limit_price
                        order_type_used = "passive"
                        logger.info(f"  ✓ Passive fill at ${limit_price:.2f}")
                    else:
                        # Not filled, fallback to aggressive
                        logger.warning("  ✗ Passive order not filled, using aggressive fallback")
                        aggressive_price = order_params['fallback_price']
                        order = place_limit_order(symbol, order_side, contracts, aggressive_price)
                        actual_fill_price = aggressive_price
                        order_type_used = "aggressive"
                        logger.info(f"  Aggressive Entry: ${aggressive_price:.2f}")
                else:
                    # Passive order failed, use aggressive
                    logger.warning("  Passive order placement failed, using aggressive")
                    aggressive_price = order_params['fallback_price']
                    order = place_limit_order(symbol, order_side, contracts, aggressive_price)
                    actual_fill_price = aggressive_price
                    order_type_used = "aggressive"
                    
            elif order_params['strategy'] == 'aggressive':
                # Use aggressive entry (cross the spread immediately)
                limit_price = order_params['limit_price']
                logger.info(f"  Aggressive Entry: ${limit_price:.2f} (guaranteed fill)")
                order = place_limit_order(symbol, order_side, contracts, limit_price)
                actual_fill_price = limit_price
                order_type_used = "aggressive"
                
            elif order_params['strategy'] == 'mixed':
                # Split order between passive and aggressive
                passive_qty = order_params['passive_contracts']
                aggressive_qty = order_params['aggressive_contracts']
                passive_price = order_params['passive_price']
                aggressive_price = order_params['aggressive_price']
                
                logger.info(f"  Mixed Strategy: {passive_qty} @ ${passive_price:.2f} (passive) + {aggressive_qty} @ ${aggressive_price:.2f} (aggressive)")
                
                # Place passive portion
                passive_order = place_limit_order(symbol, order_side, passive_qty, passive_price)
                # Place aggressive portion
                aggressive_order = place_limit_order(symbol, order_side, aggressive_qty, aggressive_price)
                
                # Use weighted average fill price
                actual_fill_price = (passive_price * passive_qty + aggressive_price * aggressive_qty) / contracts
                order = aggressive_order  # Use aggressive order for tracking
            
        except Exception as e:
            logger.error(f"Error using bid/ask manager for entry: {e}")
            logger.info("Falling back to market order")
            order = place_market_order(symbol, order_side, contracts)
            actual_fill_price = entry_price
    else:
        # No bid/ask manager, use traditional market order
        logger.info("  Using market order (no bid/ask manager)")
        
        # PRODUCTION: Apply slippage in backtest mode
        tick_size = CONFIG["tick_size"]
        slippage_ticks = CONFIG.get("slippage_ticks", 0.0)
        
        if _bot_config.backtest_mode and slippage_ticks > 0:
            if side == "long":
                actual_fill_price = entry_price + (slippage_ticks * tick_size)
            else:
                actual_fill_price = entry_price - (slippage_ticks * tick_size)
            actual_fill_price = round_to_tick(actual_fill_price)
            logger.info(f"  Slippage: {slippage_ticks} ticks")
        
        order = place_market_order(symbol, order_side, contracts)
    
    if order is None:
        logger.error("Failed to place entry order")
        return
    
    logger.info(f"  Actual Entry Price: ${actual_fill_price:.2f}")
    
    # Record trade execution for cost tracking (Requirement 5)
    if bid_ask_manager is not None:
        try:
            fill_time_seconds = (datetime.now() - fill_start_time).total_seconds()
            bid_ask_manager.record_trade_execution(
                symbol=symbol,
                side=side,
                signal_price=entry_price,
                fill_price=actual_fill_price,
                quantity=contracts,
                order_type=order_type_used
            )
            
            # Record for post-trade analysis (Requirement 13)
            quote = bid_ask_manager.get_current_quote(symbol)
            if quote:
                estimated_costs = {"total": quote.spread}
                actual_costs = {"total": abs(actual_fill_price - entry_price)}
                bid_ask_manager.record_post_trade_analysis(
                    signal_price=entry_price,
                    fill_price=actual_fill_price,
                    side=side,
                    order_type=order_type_used,
                    spread_at_order=quote.spread,
                    fill_time_seconds=fill_time_seconds,
                    estimated_costs=estimated_costs,
                    actual_costs=actual_costs
                )
        except Exception as e:
            logger.warning(f"  Could not record trade execution: {e}")
    
    # Update position tracking
    state[symbol]["position"] = {
        "active": True,
        "side": side,
        "quantity": contracts,
        "entry_price": actual_fill_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "entry_time": entry_time,
        "order_id": order.get("order_id"),
        "order_type_used": order_type_used  # Track for exit optimization
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
    logger.info(SEPARATOR_LINE)


# ============================================================================
# PHASE TEN: Exit Management
# ============================================================================

def check_stop_hit(symbol: str, current_bar: Dict[str, Any], position: Dict[str, Any]) -> Tuple[bool, Optional[float]]:
    """
    Check if stop loss has been hit.
    
    Args:
        symbol: Instrument symbol
        current_bar: Current 1-minute bar
        position: Position dictionary
    
    Returns:
        Tuple of (stop_hit, stop_price)
    """
    side = position["side"]
    stop_price = position["stop_price"]
    
    if side == "long":
        if current_bar["low"] <= stop_price:
            return True, stop_price
    else:  # short
        if current_bar["high"] >= stop_price:
            return True, stop_price
    
    return False, None


def check_target_reached(symbol: str, current_bar: Dict[str, Any], position: Dict[str, Any], 
                        bar_time: datetime) -> Tuple[bool, Optional[float]]:
    """
    Check if profit target has been reached, including time-based adjustments.
    
    Args:
        symbol: Instrument symbol
        current_bar: Current 1-minute bar
        position: Position dictionary
        bar_time: Current bar timestamp
    
    Returns:
        Tuple of (target_reached, target_price)
    """
    side = position["side"]
    target_price = position["target_price"]
    entry_price = position["entry_price"]
    stop_price = position["stop_price"]
    
    # Check regular target
    if side == "long":
        if current_bar["high"] >= target_price:
            return True, target_price
    else:  # short
        if current_bar["low"] <= target_price:
            return True, target_price
    
    # Phase Five: Time-based exit tightening after 3 PM
    if bar_time.time() >= time(15, 0) and not bot_status["flatten_mode"]:
        # After 3 PM - tighten profit taking to 1:1 R/R
        stop_distance = abs(entry_price - stop_price)
        tightened_target_distance = stop_distance  # 1:1 instead of 1.5:1
        
        if side == "long":
            tightened_target = entry_price + tightened_target_distance
            if current_bar["high"] >= tightened_target:
                return True, tightened_target
        else:  # short
            tightened_target = entry_price - tightened_target_distance
            if current_bar["low"] <= tightened_target:
                return True, tightened_target
    
    return False, None


def check_reversal_signal(symbol: str, current_bar: Dict[str, Any], position: Dict[str, Any]) -> Tuple[bool, Optional[float]]:
    """
    Check for signal reversal (price crossing back to opposite band).
    
    Args:
        symbol: Instrument symbol
        current_bar: Current 1-minute bar
        position: Position dictionary
    
    Returns:
        Tuple of (reversal_detected, exit_price)
    """
    if bot_status["flatten_mode"]:
        return False, None
    
    vwap_bands = state[symbol]["vwap_bands"]
    trend = state[symbol]["trend_direction"]
    side = position["side"]
    
    if side == "long" and trend == "up":
        # If price crosses back above upper band 2, bounce is complete
        if current_bar["close"] > vwap_bands["upper_2"]:
            return True, current_bar["close"]
    
    if side == "short" and trend == "down":
        # If price crosses back below lower band 1, bounce is complete
        if current_bar["close"] < vwap_bands["lower_1"]:
            return True, current_bar["close"]
    
    return False, None


def check_proactive_stop(symbol: str, current_bar: Dict[str, Any], position: Dict[str, Any]) -> Tuple[bool, Optional[float]]:
    """
    Check if position is within 2 ticks of stop during flatten mode.
    Prevents last-moment stop hunting.
    
    Args:
        symbol: Instrument symbol
        current_bar: Current 1-minute bar
        position: Position dictionary
    
    Returns:
        Tuple of (should_close, flatten_price)
    """
    if not bot_status["flatten_mode"]:
        return False, None
    
    side = position["side"]
    stop_price = position["stop_price"]
    tick_size = CONFIG["tick_size"]
    proactive_buffer = CONFIG["proactive_stop_buffer_ticks"] * tick_size
    
    if side == "long":
        # Check if within 2 ticks of stop price
        if current_bar["close"] <= stop_price + proactive_buffer:
            return True, get_flatten_price(symbol, side, current_bar["close"])
    else:  # short
        # Check if within 2 ticks of stop price
        if current_bar["close"] >= stop_price - proactive_buffer:
            return True, get_flatten_price(symbol, side, current_bar["close"])
    
    return False, None


def check_time_based_exits(symbol: str, current_bar: Dict[str, Any], position: Dict[str, Any], 
                           bar_time: datetime) -> Tuple[Optional[str], Optional[float]]:
    """
    Check all time-based exit conditions.
    
    Args:
        symbol: Instrument symbol
        current_bar: Current 1-minute bar
        position: Position dictionary
        bar_time: Current bar timestamp
    
    Returns:
        Tuple of (exit_reason, exit_price) or (None, None)
    """
    side = position["side"]
    entry_price = position["entry_price"]
    stop_price = position["stop_price"]
    tick_size = CONFIG["tick_size"]
    tick_value = CONFIG["tick_value"]
    
    # Calculate unrealized P&L (used in multiple checks)
    if side == "long":
        price_change = current_bar["close"] - entry_price
    else:
        price_change = entry_price - current_bar["close"]
    ticks = price_change / tick_size
    unrealized_pnl = ticks * tick_value * position["quantity"]
    
    # Force close at forced_flatten_time (4:45 PM)
    trading_state = get_trading_state(bar_time)
    if trading_state == "closed":
        return "emergency_forced_flatten", get_flatten_price(symbol, side, current_bar["close"])
    
    # Flatten mode: specific time-based exits
    if bot_status["flatten_mode"]:
        tz = pytz.timezone(CONFIG["timezone"])
        current_time = datetime.now(tz)
        
        # 4:40 PM - close profitable positions immediately
        if current_time.time() >= time(16, 40) and unrealized_pnl > 0:
            return "time_based_profit_take", get_flatten_price(symbol, side, current_bar["close"])
        
        # 4:42 PM - close small losses
        if current_time.time() >= time(16, 42):
            stop_distance = abs(entry_price - stop_price)
            if unrealized_pnl < 0 and abs(unrealized_pnl) < (stop_distance * tick_value * position["quantity"] / 2):
                return "time_based_loss_cut", get_flatten_price(symbol, side, current_bar["close"])
        
        # Phase 10: Early profit lock after 4:40 PM
        if current_time.time() >= time(16, 40) and unrealized_pnl > 0:
            bot_status["early_close_saves"] += 1
            return "early_profit_lock", get_flatten_price(symbol, side, current_bar["close"])
        
        # Flatten mode: aggressive profit/loss management
        if side == "long":
            profit_ticks = (current_bar["close"] - entry_price) / tick_size
            midpoint = entry_price - (entry_price - stop_price) / 2
            if profit_ticks > 1 or current_bar["close"] < midpoint:
                return "flatten_mode_exit", get_flatten_price(symbol, side, current_bar["close"])
        else:  # short
            profit_ticks = (entry_price - current_bar["close"]) / tick_size
            midpoint = entry_price + (stop_price - entry_price) / 2
            if profit_ticks > 1 or current_bar["close"] > midpoint:
                return "flatten_mode_exit", get_flatten_price(symbol, side, current_bar["close"])
    
    # Friday-specific exits
    if bar_time.weekday() == 4:  # Friday
        # Target close by 3 PM to avoid weekend gap risk
        if bar_time.time() >= CONFIG["friday_close_target"]:
            return "friday_weekend_protection", get_flatten_price(symbol, side, current_bar["close"])
        
        # After 2 PM on Friday, take any profit
        if bar_time.time() >= time(14, 0) and unrealized_pnl > 0:
            return "friday_profit_protection", get_flatten_price(symbol, side, current_bar["close"])
    
    # After 3:30 PM - cut losses early if less than 75% of stop distance
    if bar_time.time() >= time(15, 30) and not bot_status["flatten_mode"]:
        if side == "long":
            current_loss_distance = entry_price - current_bar["close"]
            stop_distance = entry_price - stop_price
            if current_loss_distance > 0:  # In a loss
                loss_percent = current_loss_distance / stop_distance
                if loss_percent < 0.75:  # Less than 75% of stop distance
                    return "early_loss_cut", current_bar["close"]
        else:  # short
            current_loss_distance = current_bar["close"] - entry_price
            stop_distance = stop_price - entry_price
            if current_loss_distance > 0:  # In a loss
                loss_percent = current_loss_distance / stop_distance
                if loss_percent < 0.75:  # Less than 75% of stop distance
                    return "early_loss_cut", current_bar["close"]
    
    return None, None


def check_exit_conditions(symbol: str) -> None:
    """
    Check exit conditions for open position on each bar.
    Coordinates various exit checks through helper functions.
    
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
    
    # Phase Two: Check trading state
    trading_state = get_trading_state(bar_time)
    
    # Phase Six: Enhanced flatten mode activation
    if trading_state == "flatten_mode" and not bot_status["flatten_mode"]:
        bot_status["flatten_mode"] = True
        
        # Log flatten mode activation with position details
        tick_size = CONFIG["tick_size"]
        tick_value = CONFIG["tick_value"]
        if side == "long":
            price_change = current_bar["close"] - entry_price
        else:
            price_change = entry_price - current_bar["close"]
        ticks = price_change / tick_size
        unrealized_pnl = ticks * tick_value * position["quantity"]
        
        position_details = {
            "side": position["side"],
            "quantity": position["quantity"],
            "entry_price": f"${position['entry_price']:.2f}",
            "current_time": bar_time.strftime('%H:%M:%S %Z'),
            "unrealized_pnl": f"${unrealized_pnl:+.2f}"
        }
        
        log_time_based_action(
            "flatten_mode_activated",
            "All positions must be closed by 4:45 PM ET to avoid settlement risk",
            position_details
        )
        
        logger.critical(SEPARATOR_LINE)
        logger.critical("FLATTEN MODE ACTIVATED - POSITION MUST CLOSE IN 15 MINUTES")
        logger.critical(SEPARATOR_LINE)
    
    # Minute-by-minute status logging during flatten mode
    if bot_status["flatten_mode"]:
        tz = pytz.timezone(CONFIG["timezone"])
        current_time = datetime.now(tz)
        forced_flatten_time = datetime.combine(current_time.date(), CONFIG["forced_flatten_time"])
        forced_flatten_time = tz.localize(forced_flatten_time)
        minutes_remaining = (forced_flatten_time - current_time).total_seconds() / 60.0
        
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
    
    # Check stop loss
    stop_hit, price = check_stop_hit(symbol, current_bar, position)
    if stop_hit:
        execute_exit(symbol, price, "stop_loss")
        return
    
    # Check proactive stop (during flatten mode)
    should_close, price = check_proactive_stop(symbol, current_bar, position)
    if should_close:
        logger.warning(f"Proactive stop close: within {CONFIG['proactive_stop_buffer_ticks']} ticks of stop")
        execute_exit(symbol, price, "proactive_stop")
        return
    
    # Check target reached
    target_hit, price = check_target_reached(symbol, current_bar, position, bar_time)
    if target_hit:
        if price == position["target_price"]:
            execute_exit(symbol, price, "target_reached")
            # Track successful target wait
            if bot_status["flatten_mode"]:
                bot_status["target_wait_wins"] += 1
        else:
            # Tightened target
            logger.info("Time-based tightened profit target reached (1:1 R/R after 3 PM)")
            execute_exit(symbol, price, "tightened_target")
        return
    
    # Check time-based exits
    reason, price = check_time_based_exits(symbol, current_bar, position, bar_time)
    if reason:
        # Log specific messages for certain exit types
        if reason == "emergency_forced_flatten":
            logger.critical(SEPARATOR_LINE)
            logger.critical("EMERGENCY FORCED FLATTEN - 4:45 PM DEADLINE REACHED")
            logger.critical(SEPARATOR_LINE)
        elif reason == "time_based_profit_take":
            logger.critical("4:40 PM - Closing profitable position immediately")
        elif reason == "time_based_loss_cut":
            logger.critical("4:42 PM - Cutting small loss before settlement")
        elif reason == "early_profit_lock":
            logger.warning(f"Phase 10: Closing early profit instead of waiting for target")
        elif reason == "friday_weekend_protection":
            logger.critical(SEPARATOR_LINE)
            logger.critical("FRIDAY 3 PM - CLOSING POSITION TO AVOID WEEKEND GAP RISK")
            logger.critical(SEPARATOR_LINE)
        elif reason == "friday_profit_protection":
            tick_size = CONFIG["tick_size"]
            tick_value = CONFIG["tick_value"]
            if side == "long":
                price_change = current_bar["close"] - entry_price
            else:
                price_change = entry_price - current_bar["close"]
            ticks = price_change / tick_size
            unrealized_pnl = ticks * tick_value * position["quantity"]
            logger.warning(f"Friday 2 PM+ - Taking ${unrealized_pnl:+.2f} profit to avoid weekend risk")
        elif reason == "early_loss_cut":
            tick_size = CONFIG["tick_size"]
            if side == "long":
                current_loss_distance = entry_price - current_bar["close"]
                stop_distance = entry_price - stop_price
                loss_percent = current_loss_distance / stop_distance
            else:
                current_loss_distance = current_bar["close"] - entry_price
                stop_distance = stop_price - entry_price
                loss_percent = current_loss_distance / stop_distance
            logger.warning(f"Time-based early loss cut (3:30 PM+): {loss_percent*100:.1f}% of stop distance")
        elif reason == "flatten_mode_exit":
            profit_ticks = abs((current_bar["close"] - entry_price) / CONFIG["tick_size"]) if side == "long" else abs((entry_price - current_bar["close"]) / CONFIG["tick_size"])
            logger.info(f"Flatten mode exit: profit_ticks={profit_ticks:.1f}")
        
        execute_exit(symbol, price, reason)
        return
    
    # Check for signal reversal
    reversal, price = check_reversal_signal(symbol, current_bar, position)
    if reversal:
        execute_exit(symbol, price, "signal_reversal")
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


def calculate_pnl(position: Dict[str, Any], exit_price: float) -> Tuple[float, float]:
    """
    Calculate profit/loss for the exit - PRODUCTION READY with slippage and commissions.
    
    Args:
        position: Position dictionary
        exit_price: Exit price (before slippage)
    
    Returns:
        Tuple of (ticks, pnl_dollars after all costs)
    """
    entry_price = position["entry_price"]
    contracts = position["quantity"]
    tick_size = CONFIG["tick_size"]
    tick_value = CONFIG["tick_value"]
    
    # Apply exit slippage in backtest mode
    slippage_ticks = CONFIG.get("slippage_ticks", 0.0)
    actual_exit_price = exit_price
    
    if _bot_config.backtest_mode and slippage_ticks > 0:
        # Exit slippage works AGAINST you
        if position["side"] == "long":
            # Selling - you get filled lower
            actual_exit_price = exit_price - (slippage_ticks * tick_size)
        else:  # short
            # Buying to cover - you get filled higher
            actual_exit_price = exit_price + (slippage_ticks * tick_size)
        
        actual_exit_price = round_to_tick(actual_exit_price)
    
    # Calculate gross P&L
    if position["side"] == "long":
        price_change = actual_exit_price - entry_price
    else:
        price_change = entry_price - actual_exit_price
    
    ticks = price_change / tick_size
    gross_pnl = ticks * tick_value * contracts
    
    # Deduct commissions
    commission = CONFIG.get("commission_per_contract", 0.0) * contracts
    net_pnl = gross_pnl - commission
    
    # Track costs globally in backtest mode
    if _bot_config.backtest_mode:
        slippage_cost = slippage_ticks * tick_value * contracts * 2  # Entry + Exit
        bot_status["total_slippage_cost"] += slippage_cost
        bot_status["total_commission"] += commission
        
        # Log costs breakdown
        if slippage_ticks > 0 or commission > 0:
            total_costs = slippage_cost + commission
            logger.debug(f"  Trading costs: Slippage ${slippage_cost:.2f} + Commission ${commission:.2f} = ${total_costs:.2f}")
            logger.debug(f"  Gross P&L: ${gross_pnl:.2f} → Net P&L: ${net_pnl:.2f}")
    
    return ticks, net_pnl


def update_position_statistics(symbol: str, position: Dict[str, Any], exit_time: datetime, 
                               reason: str, time_based_reasons: List[str]) -> None:
    """
    Update position duration and statistics.
    
    Args:
        symbol: Instrument symbol
        position: Position dictionary
        exit_time: Exit timestamp
        reason: Exit reason
        time_based_reasons: List of time-based exit reasons
    """
    if position["entry_time"] is None:
        return
    
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


def handle_exit_orders(symbol: str, position: Dict[str, Any], exit_price: float, reason: str) -> None:
    """
    Handle exit order placement using intelligent bid/ask optimization.
    Requirement 9: Exit Order Optimization
    
    Args:
        symbol: Instrument symbol
        position: Position dictionary
        exit_price: Exit price
        reason: Exit reason
    """
    order_side = "SELL" if position["side"] == "long" else "BUY"
    contracts = position["quantity"]
    
    # Determine exit type based on reason
    exit_type_map = {
        "target_reached": "target",
        "stop_loss": "stop",
        "flatten_mode_exit": "time_flatten",
        "time_based_profit_take": "time_flatten",
        "time_based_loss_cut": "time_flatten",
        "emergency_forced_flatten": "time_flatten",
        "signal_reversal": "partial",
        "early_profit_lock": "partial"
    }
    exit_type = exit_type_map.get(reason, "stop")  # Default to stop for safety
    
    # Use bid/ask manager for intelligent exit routing
    if bid_ask_manager is not None:
        try:
            strategy = bid_ask_manager.get_exit_order_strategy(
                exit_type=exit_type,
                symbol=symbol,
                side=position["side"],
                urgency="normal"
            )
            
            logger.info(f"Exit Strategy: {strategy['order_type']} - {strategy['reason']}")
            
            if strategy['order_type'] == 'passive':
                # Try passive exit to collect spread
                limit_price = strategy['limit_price']
                logger.info(f"Passive exit at ${limit_price:.2f} (collecting spread)")
                order = place_limit_order(symbol, order_side, contracts, limit_price)
                
                if order and strategy.get('timeout', 0) > 0:
                    # Wait for fill with timeout
                    import time
                    time.sleep(strategy['timeout'])
                    
                    # Check if filled
                    current_position = get_position_quantity(symbol)
                    if current_position == 0:
                        logger.info("✓ Passive exit filled")
                        return
                    else:
                        # Not filled, use fallback
                        logger.warning("✗ Passive exit not filled, using aggressive")
                        if 'fallback_price' in strategy:
                            order = place_limit_order(symbol, order_side, contracts, strategy['fallback_price'])
                        else:
                            order = place_market_order(symbol, order_side, contracts)
                else:
                    # No timeout or order failed, go aggressive
                    order = place_market_order(symbol, order_side, contracts)
            else:
                # Aggressive exit
                if 'limit_price' in strategy:
                    logger.info(f"Aggressive exit at ${strategy['limit_price']:.2f}")
                    order = place_limit_order(symbol, order_side, contracts, strategy['limit_price'])
                else:
                    order = place_market_order(symbol, order_side, contracts)
            
            if order:
                logger.info(f"Exit order placed: {order.get('order_id')}")
                return
                
        except Exception as e:
            logger.error(f"Error using bid/ask exit optimization: {e}")
            logger.info("Falling back to traditional exit")
    
    # Fallback to traditional exit logic
    is_flatten_mode = bot_status["flatten_mode"] or reason in [
        "flatten_mode_exit", "time_based_profit_take", 
        "time_based_loss_cut", "emergency_forced_flatten"
    ]
    
    if is_flatten_mode:
        logger.info("Using aggressive limit order strategy for flatten")
        execute_flatten_with_limit_orders(symbol, order_side, contracts, exit_price, reason)
    else:
        # Normal exit - use market order
        order = place_market_order(symbol, order_side, contracts)
        if order:
            logger.info(f"Exit order placed: {order.get('order_id')}")


def execute_exit(symbol: str, exit_price: float, reason: str) -> None:
    """
    Execute exit order and update P&L.
    Coordinates exit handling through helper functions.
    
    Args:
        symbol: Instrument symbol
        exit_price: Exit price
        reason: Reason for exit (stop_loss, target_reached, signal_reversal, etc.)
    """
    position = state[symbol]["position"]
    
    if not position["active"]:
        return
    
    exit_time = datetime.now(pytz.timezone(CONFIG["timezone"]))
    
    logger.info(SEPARATOR_LINE)
    logger.info(f"EXITING {position['side'].upper()} POSITION")
    logger.info(f"  Reason: {reason.replace('_', ' ').title()}")
    logger.info(f"  Time: {exit_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Calculate P&L
    ticks, pnl = calculate_pnl(position, exit_price)
    
    logger.info(f"  Entry: ${position['entry_price']:.2f}, Exit: ${exit_price:.2f}")
    logger.info(f"  Ticks: {ticks:+.1f}, P&L: ${pnl:+.2f}")
    
    # Log time-based exits with detailed audit trail
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
            "quantity": position["quantity"],
            "entry_price": f"${position['entry_price']:.2f}"
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
    
    # Handle exit orders
    handle_exit_orders(symbol, position, exit_price, reason)
    
    # Update daily P&L
    state[symbol]["daily_pnl"] += pnl
    
    # Update position statistics
    update_position_statistics(symbol, position, exit_time, reason, time_based_reasons)
    
    # Update session statistics
    update_session_stats(symbol, pnl)
    
    logger.info(f"Daily P&L: ${state[symbol]['daily_pnl']:+.2f}")
    logger.info(f"Trades today: {state[symbol]['daily_trade_count']}/{CONFIG['max_trades_per_day']}")
    logger.info(SEPARATOR_LINE)
    
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


def calculate_aggressive_price(base_price: float, order_side: str, attempt: int) -> float:
    """
    Calculate increasingly aggressive limit price based on attempt number.
    
    Args:
        base_price: Base price for limit orders
        order_side: 'BUY' or 'SELL'
        attempt: Current attempt number (1-indexed)
    
    Returns:
        Aggressive limit price
    """
    tick_size = CONFIG["tick_size"]
    ticks_aggressive = attempt
    
    if order_side == "SELL":
        # Selling - go below bid
        limit_price = base_price - (ticks_aggressive * tick_size)
    else:  # BUY
        # Buying - go above offer
        limit_price = base_price + (ticks_aggressive * tick_size)
    
    return round_to_tick(limit_price)


def wait_for_fill(symbol: str, attempt: int, max_attempts: int) -> int:
    """
    Wait for order to fill and return current position quantity.
    
    Args:
        symbol: Instrument symbol
        attempt: Current attempt number
        max_attempts: Maximum number of attempts
    
    Returns:
        Current position quantity (0 if fully closed)
    """
    import time as time_module
    
    if attempt < max_attempts:
        wait_seconds = 5 if attempt < 5 else 2  # Shorter waits as we get more urgent
        logger.debug(f"Waiting {wait_seconds} seconds for fill...")
        time_module.sleep(wait_seconds)
    
    return get_position_quantity(symbol)


def handle_partial_fill(current_qty: int, contracts: int, attempt: int) -> int:
    """
    Handle partial fill and return remaining contracts.
    
    Args:
        current_qty: Current position quantity from broker
        contracts: Original number of contracts
        attempt: Current attempt number
    
    Returns:
        Number of contracts still remaining
    """
    if current_qty == 0:
        logger.info("Position fully closed")
        return 0
    else:
        filled_contracts = contracts - abs(current_qty)
        if filled_contracts > 0:
            logger.warning(f"Partial fill: {filled_contracts} of {contracts} filled, {abs(current_qty)} remaining")
            return abs(current_qty)
        else:
            logger.warning(f"No fill on attempt {attempt}, retrying with more aggressive price")
            return abs(current_qty)


def execute_flatten_with_limit_orders(symbol: str, order_side: str, contracts: int, 
                                       base_price: float, reason: str) -> None:
    """
    Execute flatten using aggressive limit orders with partial fill handling.
    Main orchestration function that coordinates helpers.
    
    Args:
        symbol: Instrument symbol
        order_side: 'BUY' or 'SELL'
        contracts: Number of contracts to close
        base_price: Base price for limit orders
        reason: Exit reason
    """
    remaining_contracts = contracts
    attempt = 0
    max_attempts = 10
    
    while remaining_contracts > 0 and attempt < max_attempts:
        attempt += 1
        
        # Calculate aggressive limit price
        limit_price = calculate_aggressive_price(base_price, order_side, attempt)
        
        logger.info(f"Flatten attempt {attempt}/{max_attempts}: {order_side} {remaining_contracts} @ {limit_price:.2f}")
        
        # Place aggressive limit order
        order = place_limit_order(symbol, order_side, remaining_contracts, limit_price)
        
        if order:
            logger.info(f"Flatten limit order placed: {order.get('order_id')}")
        
        # Wait and check for fills
        if attempt < max_attempts:
            current_qty = wait_for_fill(symbol, attempt, max_attempts)
            remaining_contracts = handle_partial_fill(current_qty, contracts, attempt)
            
            if remaining_contracts == 0:
                break
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

def check_vwap_reset(symbol: str, current_time: datetime) -> None:
    """
    Check if VWAP should reset at 6 PM ET (futures market day start).
    For 24/5 trading: VWAP resets at 6 PM when futures trading day begins.
    
    Args:
        symbol: Instrument symbol
        current_time: Current datetime in Eastern Time
    """
    current_date = current_time.date()
    vwap_reset_time = time(18, 0)  # 6 PM ET - futures trading day starts
    
    # Check if we've crossed 6 PM on a new day
    if state[symbol]["vwap_day"] is None:
        # First run - initialize VWAP day
        state[symbol]["vwap_day"] = current_date
        logger.info(f"VWAP day initialized: {current_date}")
        return
    
    # If it's a new day and we're past 6 PM, reset VWAP
    # OR if it's the same calendar day but we just crossed 6 PM
    last_reset_date = state[symbol]["vwap_day"]
    crossed_reset_time = current_time.time() >= vwap_reset_time
    
    # New trading day starts at 6 PM, so check if we've moved to a new VWAP session
    if crossed_reset_time and last_reset_date != current_date:
        perform_vwap_reset(symbol, current_date, current_time)


def perform_vwap_reset(symbol: str, new_date: Any, reset_time: datetime) -> None:
    """
    Perform VWAP reset at 6 PM ET daily (futures trading day start).
    
    Args:
        symbol: Instrument symbol
        new_date: The new VWAP date
        reset_time: Time of the reset
    """
    logger.info(SEPARATOR_LINE)
    logger.info(f"VWAP RESET at {reset_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"Futures trading day start (6 PM ET) - New VWAP day: {new_date}")
    logger.info(SEPARATOR_LINE)
    
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
    logger.info(SEPARATOR_LINE)


def check_daily_reset(symbol: str, current_time: datetime) -> None:
    """
    Check if we've crossed into a new trading day and reset daily counters.
    For 24/5 trading: Resets at 6 PM ET (futures trading day start).
    
    Args:
        symbol: Instrument symbol
        current_time: Current datetime in Eastern Time
    """
    current_date = current_time.date()
    vwap_reset_time = time(18, 0)  # 6 PM ET - futures trading day starts
    
    # If we have a trading day stored and it's different from current date
    if state[symbol]["trading_day"] is not None:
        if state[symbol]["trading_day"] != current_date:
            # Reset daily counters at 6 PM (same as VWAP reset)
            if current_time.time() >= vwap_reset_time:
                perform_daily_reset(symbol, current_date)
    else:
        # First run - initialize trading day
        state[symbol]["trading_day"] = current_date
        logger.info(f"Trading day initialized: {current_date}")


def perform_daily_reset(symbol: str, new_date: Any) -> None:
    """
    Perform the actual daily reset operations.
    Resets daily counters and session stats.
    VWAP reset is handled separately by perform_vwap_reset.
    
    Args:
        symbol: Instrument symbol
        new_date: The new trading date
    """
    logger.info(SEPARATOR_LINE)
    logger.info(f"DAILY RESET - New Trading Day: {new_date}")
    logger.info(SEPARATOR_LINE)
    
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
        "pnl_variance": 0.0,
        # Phase 20: Position duration statistics
        "trade_durations": [],  # List of durations in minutes
        "force_flattened_count": 0,  # Trades closed due to time limit
        "after_noon_entries": 0,  # Entries after 12 PM
        "after_noon_force_flattened": 0  # After-noon entries force-closed
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
    logger.info(SEPARATOR_LINE)


# ============================================================================
# PHASE TWELVE: Safety Mechanisms
# ============================================================================

def check_daily_loss_limit(symbol: str) -> Tuple[bool, Optional[str]]:
    """
    Check if daily loss limit has been exceeded.
    
    Args:
        symbol: Instrument symbol
    
    Returns:
        Tuple of (is_safe, reason)
    """
    if state[symbol]["daily_pnl"] <= -CONFIG["daily_loss_limit"]:
        if bot_status["trading_enabled"]:
            logger.critical(f"DAILY LOSS LIMIT BREACHED: ${state[symbol]['daily_pnl']:.2f}")
            logger.critical("Trading STOPPED for the day")
            bot_status["trading_enabled"] = False
            bot_status["stop_reason"] = "daily_loss_limit"
        return False, "Daily loss limit exceeded"
    return True, None


def check_max_drawdown() -> Tuple[bool, Optional[str]]:
    """
    Check if maximum drawdown has been exceeded.
    
    Returns:
        Tuple of (is_safe, reason)
    """
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
    return True, None


def check_tick_timeout(current_time: datetime) -> Tuple[bool, Optional[str]]:
    """
    Check if data feed has timed out.
    
    Args:
        current_time: Current datetime in Eastern Time
    
    Returns:
        Tuple of (is_safe, reason)
    """
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


def check_trade_limits(current_time: datetime) -> Tuple[bool, Optional[str]]:
    """
    Check emergency stop and trading enabled status.
    24/5 trading - only stop for maintenance window and weekends.
    
    Args:
        current_time: Current datetime in Eastern Time
    
    Returns:
        Tuple of (is_safe, reason)
    """
    # Check if emergency stop is active
    if bot_status["emergency_stop"]:
        return False, f"Emergency stop active: {bot_status['stop_reason']}"
    
    # Check for weekend (Saturday/Sunday)
    if current_time.weekday() >= 5:  # 5=Saturday, 6=Sunday
        if bot_status["trading_enabled"]:
            logger.debug(f"Weekend detected - disabling trading until Monday")
            bot_status["trading_enabled"] = False
            bot_status["stop_reason"] = "weekend"
        return False, "Weekend - market closed"
    
    # Check for futures maintenance window (5:00 PM - 6:00 PM ET daily)
    maintenance_start = time(17, 0)  # 5 PM
    maintenance_end = time(18, 0)    # 6 PM
    if maintenance_start <= current_time.time() < maintenance_end:
        if bot_status["trading_enabled"]:
            logger.debug(f"Maintenance window - disabling trading")
            bot_status["trading_enabled"] = False
            bot_status["stop_reason"] = "maintenance"
        return False, "Maintenance window"
    
    # Re-enable trading after maintenance/weekend
    if not bot_status["trading_enabled"]:
        logger.debug(f"Re-enabling trading - market open at {current_time}")
        bot_status["trading_enabled"] = True
        bot_status["stop_reason"] = None
    
    return True, None


def check_safety_conditions(symbol: str) -> Tuple[bool, Optional[str]]:
    """
    Check all safety conditions before allowing trading.
    Coordinates various safety checks through helper functions.
    
    Args:
        symbol: Instrument symbol
    
    Returns:
        Tuple of (is_safe, reason) where is_safe is True if safe to trade
    """
    current_time = get_current_time()
    
    # Check trade limits and emergency stops
    is_safe, reason = check_trade_limits(current_time)
    if not is_safe:
        return False, reason
    
    # Check daily loss limit
    is_safe, reason = check_daily_loss_limit(symbol)
    if not is_safe:
        return False, reason
    
    # Check maximum drawdown
    is_safe, reason = check_max_drawdown()
    if not is_safe:
        return False, reason
    
    # Check tick timeout
    is_safe, reason = check_tick_timeout(current_time)
    if not is_safe:
        return False, reason
    
    return True, None


def check_no_overnight_positions(symbol: str) -> None:
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

def format_trade_statistics(stats: Dict[str, Any]) -> None:
    """
    Format and log basic trade statistics.
    
    Args:
        stats: Session statistics dictionary
    """
    logger.info(f"Total Trades: {len(stats['trades'])}")
    logger.info(f"Wins: {stats['win_count']}")
    logger.info(f"Losses: {stats['loss_count']}")
    
    if len(stats['trades']) > 0:
        win_rate = stats['win_count'] / len(stats['trades']) * 100
        logger.info(f"Win Rate: {win_rate:.1f}%")
    else:
        logger.info("Win Rate: N/A (no trades)")


def format_pnl_summary(stats: Dict[str, Any]) -> None:
    """
    Format and log P&L summary including Sharpe ratio.
    
    Args:
        stats: Session statistics dictionary
    """
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


def format_time_statistics(stats: Dict[str, Any]) -> None:
    """
    Format and log position duration statistics.
    
    Args:
        stats: Session statistics dictionary
    """
    if len(stats['trade_durations']) == 0:
        return
    
    logger.info(SEPARATOR_LINE)
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
    time_to_flatten_at_2pm = 165  # minutes from 2 PM to 4:45 PM
    if avg_duration > time_to_flatten_at_2pm * 0.8:
        logger.warning("⚠️  Average duration uses >80% of available time window")
        logger.warning(f"   Avg duration {avg_duration:.1f} min vs {time_to_flatten_at_2pm} min available at 2 PM")


def format_risk_metrics() -> None:
    """
    Format and log flatten mode exit analysis.
    """
    total_decisions = (bot_status["target_wait_wins"] + bot_status["target_wait_losses"] + 
                       bot_status["early_close_saves"])
    if total_decisions == 0:
        return
    
    logger.info(SEPARATOR_LINE)
    logger.info("FLATTEN MODE EXIT ANALYSIS (Phase 10)")
    logger.info(f"Target Wait Wins: {bot_status['target_wait_wins']}")
    logger.info(f"Target Wait Losses: {bot_status['target_wait_losses']}")
    logger.info(f"Early Close Saves: {bot_status['early_close_saves']}")
    if bot_status["target_wait_wins"] > 0:
        target_success_rate = (bot_status["target_wait_wins"] / 
                               (bot_status["target_wait_wins"] + bot_status["target_wait_losses"]) * 100)
        logger.info(f"Target Wait Success Rate: {target_success_rate:.1f}%")


def log_session_summary(symbol: str) -> None:
    """
    Log comprehensive session summary at end of trading day.
    Coordinates summary formatting through helper functions.
    
    Args:
        symbol: Instrument symbol
    """
    stats = state[symbol]["session_stats"]
    
    logger.info(SEPARATOR_LINE)
    logger.info("SESSION SUMMARY")
    logger.info(SEPARATOR_LINE)
    logger.info(f"Trading Day: {state[symbol]['trading_day']}")
    
    # Format trade statistics
    format_trade_statistics(stats)
    
    # Format P&L summary
    format_pnl_summary(stats)
    
    # PRODUCTION: Show trading costs breakdown
    if _bot_config.backtest_mode:
        total_slippage = bot_status["total_slippage_cost"]
        total_commission = bot_status["total_commission"]
        total_costs = total_slippage + total_commission
        
        if total_costs > 0:
            logger.info(SEPARATOR_LINE)
            logger.info("TRADING COSTS (Backtest)")
            logger.info(f"Total Slippage: ${total_slippage:.2f}")
            logger.info(f"Total Commission: ${total_commission:.2f}")
            logger.info(f"Total Costs: ${total_costs:.2f}")
            logger.info(f"Cost per Trade: ${total_costs / max(1, len(stats['trades'])):.2f}")
            
            if stats["total_pnl"] > 0:
                cost_percentage = (total_costs / stats["total_pnl"]) * 100
                logger.info(f"Costs as % of Gross P&L: {cost_percentage:.1f}%")
    
    # Format risk metrics (flatten mode analysis)
    format_risk_metrics()
    
    # Format time statistics (position duration)
    format_time_statistics(stats)
    
    logger.info(SEPARATOR_LINE)


def update_session_stats(symbol: str, pnl: float) -> None:
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
# PHASE TWO: Time Management - Support Both Live and Backtest
# ============================================================================

def get_current_time() -> datetime:
    """
    Get current time - either real time (live) or backtest simulation time.
    
    Returns:
        Current datetime with timezone
    """
    global backtest_current_time
    
    if backtest_current_time is not None:
        # Backtest mode: use simulated time
        return backtest_current_time
    else:
        # Live mode: use real time
        tz = pytz.timezone(CONFIG["timezone"])
        return datetime.now(tz)


def get_trading_state(dt: datetime = None) -> str:
    """
    Centralized time checking function that returns current trading state.
    24/5 trading - always in entry window except maintenance/weekend.
    
    Args:
        dt: Datetime to check (defaults to current time - live or backtest)
    
    Returns:
        Trading state: 'entry_window', 'maintenance', or 'weekend'
    """
    if dt is None:
        dt = get_current_time()
    elif dt.tzinfo is None:
        # If naive datetime provided, assume it's Eastern Time
        tz = pytz.timezone(CONFIG["timezone"])
        dt = tz.localize(dt)
    else:
        # Convert to Eastern Time
        tz = pytz.timezone(CONFIG["timezone"])
        dt = dt.astimezone(tz)
    
    # Check for weekend
    if dt.weekday() >= 5:  # Saturday or Sunday
        return 'weekend'
    
    # Check for maintenance window (5-6 PM ET daily)
    maintenance_start = time(17, 0)
    maintenance_end = time(18, 0)
    current_time = dt.time()
    if maintenance_start <= current_time < maintenance_end:
        return 'maintenance'
    
    # Otherwise always in entry window for 24/5 trading
    return 'entry_window'


# ============================================================================
# PHASE FIFTEEN & SIXTEEN: Timezone Handling and Time-Based Logging
# ============================================================================

def validate_timezone_configuration() -> None:
    """
    Phase Fifteen: Validate timezone configuration on bot startup.
    Ensures pytz is working correctly and DST is handled properly.
    """
    tz = pytz.timezone(CONFIG["timezone"])
    current_time = datetime.now(tz)
    
    logger.info(SEPARATOR_LINE)
    logger.info("TIMEZONE CONFIGURATION VALIDATION")
    logger.info(SEPARATOR_LINE)
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
        logger.warning("System local time differs from ET by >1 hour")
        logger.warning(f"System: {system_time.strftime('%H:%M:%S')}, ET: {current_time.strftime('%H:%M:%S')}")
        logger.warning("All trading decisions use ET - system time is informational only")
    
    logger.info(SEPARATOR_LINE)


def log_time_based_action(action: str, reason: str, details: Optional[Dict[str, Any]] = None) -> None:
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

When backtesting this strategy on historical data, you must implement time-based
flatten rules accurately:

1. For every historical trade, check entry time
   - If entered after 2 PM, calculate time remaining until 4:45 PM flatten
   - Execute forced flatten at whatever price existed at 4:45 PM if position
     hadn't hit target or stop

2. Track forced flatten statistics:
   - Count how many trades were force-flattened before hitting target/stop
   - Calculate how many times forced flatten saved from overnight gap losses
   - Calculate how many times it cost profit by closing a winner early

3. Analyze trade duration:
   - If 30%+ of trades get force-flattened, average trade duration is too long
   - Either extend trading hours or accept lower targets to close positions faster

4. Friday-specific backtesting:
   - Enforce no new trades after 1 PM Friday
   - Execute forced close at 3 PM Friday
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

def main() -> None:
    """Main bot execution with event loop integration"""
    global event_loop, timer_manager, bid_ask_manager
    
    logger.info(SEPARATOR_LINE)
    logger.info("VWAP Bounce Bot Starting")
    logger.info(SEPARATOR_LINE)
    logger.info(f"Mode: {'DRY RUN' if CONFIG['dry_run'] else 'LIVE TRADING'}")
    logger.info(f"Instrument: {CONFIG['instrument']}")
    logger.info(f"Entry Window: {CONFIG['entry_window_start']} - {CONFIG['entry_window_end']} ET")
    logger.info(f"Flatten Mode: {CONFIG['warning_time']} ET")
    logger.info(f"Force Close: {CONFIG['forced_flatten_time']} ET")
    logger.info(f"Shutdown: {CONFIG['shutdown_time']} ET")
    logger.info(f"Max Trades/Day: {CONFIG['max_trades_per_day']}")
    logger.info(f"Daily Loss Limit: ${CONFIG['daily_loss_limit']}")
    logger.info(f"Max Drawdown: {CONFIG['max_drawdown_percent']}%")
    logger.info(SEPARATOR_LINE)
    
    # Phase Fifteen: Validate timezone configuration
    validate_timezone_configuration()
    
    # Initialize bid/ask manager
    logger.info("Initializing bid/ask manager...")
    bid_ask_manager = BidAskManager(CONFIG)
    
    # Initialize broker (replaces initialize_sdk)
    initialize_broker()
    
    # Phase 12: Record starting equity for drawdown monitoring
    bot_status["starting_equity"] = get_account_equity()
    logger.info(f"Starting Equity: ${bot_status['starting_equity']:.2f}")
    
    # Initialize state for instrument
    symbol = CONFIG["instrument"]
    initialize_state(symbol)
    
    # Fetch historical bars for trend filter initialization
    historical_bars = fetch_historical_bars(
        symbol=symbol,
        timeframe=CONFIG.get("trend_timeframe", "15min"),
        count=CONFIG.get("trend_filter_period", 20)
    )
    
    if historical_bars:
        state[symbol]["bars_15min"].extend(historical_bars)
        update_trend_filter(symbol)
    
    # Initialize event loop
    logger.info("Initializing event loop...")
    event_loop = EventLoop(bot_status, CONFIG)
    
    # Register event handlers
    event_loop.register_handler(EventType.TICK_DATA, handle_tick_event)
    event_loop.register_handler(EventType.TIME_CHECK, handle_time_check_event)
    event_loop.register_handler(EventType.VWAP_RESET, handle_vwap_reset_event)
    event_loop.register_handler(EventType.FLATTEN_MODE, handle_flatten_mode_event)
    event_loop.register_handler(EventType.SHUTDOWN, handle_shutdown_event)
    
    # Register shutdown handlers for cleanup
    event_loop.register_shutdown_handler(cleanup_on_shutdown)
    
    # Initialize timer manager for periodic events
    tz = pytz.timezone(CONFIG["timezone"])
    timer_manager = TimerManager(event_loop, CONFIG, tz)
    timer_manager.start()
    
    # Subscribe to market data (trades)
    subscribe_market_data(symbol, on_tick)
    
    # Subscribe to bid/ask quotes if broker supports it
    if broker is not None and hasattr(broker, 'subscribe_quotes'):
        logger.info("Subscribing to bid/ask quotes...")
        try:
            broker.subscribe_quotes(symbol, on_quote)
        except Exception as e:
            logger.warning(f"Failed to subscribe to quotes: {e}")
            logger.warning("Continuing without bid/ask quote data")
    
    logger.info("Bot initialization complete")
    logger.info("Starting event loop...")
    logger.info("Press Ctrl+C for graceful shutdown")
    logger.info(SEPARATOR_LINE)
    
    # Run event loop (blocks until shutdown signal)
    try:
        event_loop.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        logger.info("Event loop stopped")
        
        # Print final metrics
        metrics = event_loop.get_metrics()
        logger.info("Event Loop Metrics:")
        logger.info(f"  Total iterations: {metrics['total_iterations']}")
        logger.info(f"  Events processed: {metrics['events_processed']}")
        logger.info(f"  Max queue depth: {metrics['max_queue_depth']}")
        logger.info(f"  Stall count: {metrics['stall_count']}")


# ============================================================================
# EVENT HANDLERS
# ============================================================================

def handle_tick_event(data: Dict[str, Any]) -> None:
    """Handle tick data event from event loop"""
    symbol = data["symbol"]
    price = data["price"]
    volume = data["volume"]
    timestamp_ms = data["timestamp"]
    
    if symbol not in state:
        initialize_state(symbol)
    
    # Phase 12: Update last tick time for connection health check
    tz = pytz.timezone(CONFIG["timezone"])
    dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=tz)
    bot_status["last_tick_time"] = dt
    
    # Monitor data feed health
    if recovery_manager:
        recovery_manager.data_feed_monitor.record_tick(symbol)
    
    # Create tick object
    tick = {
        "price": price,
        "volume": volume,
        "timestamp": timestamp_ms
    }
    
    # Append to tick storage
    state[symbol]["ticks"].append(tick)
    
    # Update 1-minute bars
    update_1min_bar(symbol, price, volume, dt)
    
    # Update 15-minute bars
    update_15min_bar(symbol, price, volume, dt)


def handle_time_check_event(data: Dict[str, Any]) -> None:
    """Handle time-based checks event"""
    symbol = CONFIG["instrument"]
    if symbol in state:
        tz = pytz.timezone(CONFIG["timezone"])
        current_time = datetime.now(tz).time()
        
        # Check for daily reset
        check_daily_reset(symbol, datetime.now(tz))
        
        # Critical safety check - NO positions past 5 PM
        check_no_overnight_positions(symbol)


def handle_vwap_reset_event(data: Dict[str, Any]) -> None:
    """Handle VWAP reset event"""
    symbol = CONFIG["instrument"]
    if symbol in state:
        tz = pytz.timezone(CONFIG["timezone"])
        check_vwap_reset(symbol, datetime.now(tz))


def handle_flatten_mode_event(data: Dict[str, Any]) -> None:
    """Handle flatten mode activation event"""
    logger.warning("Flatten mode activated - initiating position closure")
    bot_status["flatten_mode"] = True
    
    # If position is active, start flatten process
    symbol = CONFIG["instrument"]
    if symbol in state and state[symbol]["position"]["active"]:
        logger.warning(f"Active position detected - executing flatten")
        # The exit conditions check will handle the flatten


def handle_shutdown_event(data: Dict[str, Any]) -> None:
    """Handle shutdown event"""
    logger.info("Shutdown event received")
    bot_status["trading_enabled"] = False


def cleanup_on_shutdown() -> None:
    """Cleanup tasks on shutdown"""
    logger.info("Running cleanup tasks...")
    
    # Save state to disk
    if recovery_manager:
        try:
            recovery_manager.save_state(state)
            logger.info("State saved successfully")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    # Disconnect broker
    if broker and broker.is_connected():
        try:
            broker.disconnect()
            logger.info("Broker disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting broker: {e}")
    
    # Stop timer manager
    if timer_manager:
        try:
            timer_manager.stop()
            logger.info("Timer manager stopped")
        except Exception as e:
            logger.error(f"Error stopping timer manager: {e}")
    
    # Log session summary
    symbol = CONFIG["instrument"]
    if symbol in state:
        log_session_summary(symbol)
    
    logger.info("Cleanup complete")


if __name__ == "__main__":
    main()
