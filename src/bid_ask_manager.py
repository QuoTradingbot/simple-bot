"""
Bid/Ask Manager - Complete Trading Strategy with No Cutting Corners
Manages real-time bid/ask quotes, spread analysis, and intelligent order placement.
"""

import logging
from typing import Dict, Any, Optional, Tuple, Deque, List
from collections import deque
from datetime import datetime, time
from dataclasses import dataclass, field
import statistics

logger = logging.getLogger(__name__)


@dataclass
class BidAskQuote:
    """Real-time bid/ask market data."""
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    last_trade_price: float
    timestamp: int  # milliseconds
    
    @property
    def spread(self) -> float:
        """Calculate bid/ask spread."""
        return self.ask_price - self.bid_price
    
    @property
    def mid_price(self) -> float:
        """Calculate mid-point between bid and ask."""
        return (self.bid_price + self.ask_price) / 2.0
    
    @property
    def imbalance_ratio(self) -> float:
        """
        Calculate bid/ask size imbalance ratio.
        Gap #3: Bid/ask size imbalance detection
        
        Returns:
            Ratio of bid_size to ask_size. 
            >1.0 means more bids (bullish pressure)
            <1.0 means more asks (bearish pressure)
            Large imbalances (>3:1 or <1:3) indicate one-sided markets
        """
        if self.ask_size == 0:
            return float('inf')  # All bid, no ask - extreme bullish
        return self.bid_size / self.ask_size
    
    def get_imbalance_signal(self, threshold: float = 3.0) -> str:
        """
        Get market imbalance signal for urgency decisions.
        Gap #3: Bid/ask size imbalance detection
        
        Args:
            threshold: Ratio threshold for strong imbalance (default 3.0)
        
        Returns:
            "strong_bid" - More aggressive on long entries (>3:1 bid/ask)
            "strong_ask" - More aggressive on short entries (<1:3 bid/ask)
            "balanced" - Normal routing
        """
        ratio = self.imbalance_ratio
        
        if ratio > threshold:
            return "strong_bid"  # Heavy buying pressure
        elif ratio < (1 / threshold):
            return "strong_ask"  # Heavy selling pressure
        else:
            return "balanced"
    
    def is_valid(self) -> Tuple[bool, str]:
        """
        Validate quote for data integrity.
        
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check for inverted or crossed spread (data error)
        if self.bid_price > self.ask_price:
            return False, f"Inverted spread: bid {self.bid_price} > ask {self.ask_price}"
        
        # Check for zero or negative prices
        if self.bid_price <= 0 or self.ask_price <= 0:
            return False, f"Invalid prices: bid={self.bid_price}, ask={self.ask_price}"
        
        # Check for zero sizes (no liquidity)
        if self.bid_size <= 0 or self.ask_size <= 0:
            return False, f"No liquidity: bid_size={self.bid_size}, ask_size={self.ask_size}"
        
        return True, "Valid quote"


@dataclass
class TradeExecution:
    """Record of a trade execution with spread cost tracking."""
    symbol: str
    side: str  # 'long' or 'short'
    signal_price: float
    spread_at_order: float
    fill_price: float
    quantity: int
    order_type: str  # 'passive' or 'aggressive'
    timestamp: datetime
    spread_saved: float = 0.0  # Positive if saved, negative if paid
    
    def __post_init__(self):
        """Calculate spread saved/paid."""
        if self.order_type == 'passive':
            # Passive orders save the spread
            self.spread_saved = self.spread_at_order
        else:
            # Aggressive orders pay the spread
            self.spread_saved = -self.spread_at_order


class SpreadCostTracker:
    """
    Tracks spread costs and fill quality across all trades.
    Requirement 5: Spread Cost Tracking
    """
    
    def __init__(self):
        """Initialize spread cost tracker."""
        self.executions: List[TradeExecution] = []
        self.total_spread_saved: float = 0.0
        self.total_spread_paid: float = 0.0
        self.passive_fill_count: int = 0
        self.aggressive_fill_count: int = 0
    
    def record_execution(self, execution: TradeExecution) -> None:
        """
        Record a trade execution.
        
        Args:
            execution: TradeExecution record
        """
        self.executions.append(execution)
        
        if execution.spread_saved > 0:
            self.total_spread_saved += execution.spread_saved * execution.quantity
            self.passive_fill_count += 1
        else:
            self.total_spread_paid += abs(execution.spread_saved) * execution.quantity
            self.aggressive_fill_count += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get spread cost statistics."""
        total_trades = len(self.executions)
        
        if total_trades == 0:
            return {
                "total_trades": 0,
                "passive_fill_rate": 0.0,
                "total_spread_saved": 0.0,
                "total_spread_paid": 0.0,
                "net_spread_savings": 0.0,
                "average_spread_per_trade": 0.0
            }
        
        passive_rate = self.passive_fill_count / total_trades
        net_savings = self.total_spread_saved - self.total_spread_paid
        avg_spread = net_savings / total_trades
        
        return {
            "total_trades": total_trades,
            "passive_fills": self.passive_fill_count,
            "aggressive_fills": self.aggressive_fill_count,
            "passive_fill_rate": passive_rate,
            "total_spread_saved": self.total_spread_saved,
            "total_spread_paid": self.total_spread_paid,
            "net_spread_savings": net_savings,
            "average_spread_per_trade": avg_spread
        }


class SpreadAnalyzer:
    """
    Analyzes bid/ask spreads to determine market conditions.
    Tracks spread history and identifies abnormal spread conditions.
    """
    
    def __init__(self, lookback_periods: int = 100, abnormal_multiplier: float = 2.0):
        """
        Initialize spread analyzer.
        
        Args:
            lookback_periods: Number of spread samples to track
            abnormal_multiplier: Multiplier for abnormal spread detection
        """
        self.lookback_periods = lookback_periods
        self.abnormal_multiplier = abnormal_multiplier
        self.spread_history: Deque[float] = deque(maxlen=lookback_periods)
        self.average_spread: Optional[float] = None
        self.std_dev_spread: Optional[float] = None
        
        # Requirement 7: Time-of-day spread patterns
        self.time_of_day_spreads: Dict[int, List[float]] = {}  # hour -> spreads
        
        # Requirement 8: Spread widening detection
        self.recent_spreads: Deque[float] = deque(maxlen=5)  # Last 5 spreads for widening detection
    
    def update(self, spread: float, timestamp: Optional[datetime] = None) -> None:
        """
        Update spread history with new spread value.
        
        Args:
            spread: Current bid/ask spread
            timestamp: Optional timestamp for time-of-day tracking
        """
        self.spread_history.append(spread)
        self.recent_spreads.append(spread)
        
        # Track time-of-day patterns
        if timestamp:
            hour = timestamp.hour
            if hour not in self.time_of_day_spreads:
                self.time_of_day_spreads[hour] = []
            self.time_of_day_spreads[hour].append(spread)
            
            # Keep only last 100 spreads per hour
            if len(self.time_of_day_spreads[hour]) > 100:
                self.time_of_day_spreads[hour].pop(0)
        
        # Recalculate statistics if we have enough data
        if len(self.spread_history) >= 20:  # Minimum 20 samples for stats
            self.average_spread = statistics.mean(self.spread_history)
            if len(self.spread_history) >= 2:
                self.std_dev_spread = statistics.stdev(self.spread_history)
    
    def is_spread_widening(self) -> Tuple[bool, str]:
        """
        Detect if spread is rapidly widening (market stress).
        Requirement 8: Spread widening detection
        
        Returns:
            Tuple of (is_widening, reason)
        """
        if len(self.recent_spreads) < 3:
            return False, "Not enough data"
        
        # Check if each spread is wider than previous
        spreads_list = list(self.recent_spreads)
        is_widening = all(spreads_list[i] > spreads_list[i-1] for i in range(1, len(spreads_list)))
        
        if is_widening:
            widening_rate = (spreads_list[-1] - spreads_list[0]) / spreads_list[0] * 100
            return True, f"Spread widening {widening_rate:.1f}% (market stress)"
        
        return False, "Spread stable"
    
    def get_expected_spread_for_time(self, timestamp: datetime) -> Optional[float]:
        """
        Get expected spread for a specific time of day.
        Requirement 7: Time-of-day spread patterns
        
        Args:
            timestamp: Time to check
        
        Returns:
            Expected spread or None if no data
        """
        hour = timestamp.hour
        
        if hour not in self.time_of_day_spreads or not self.time_of_day_spreads[hour]:
            return None
        
        return statistics.mean(self.time_of_day_spreads[hour])
    
    def is_spread_acceptable(self, current_spread: float) -> Tuple[bool, str]:
        """
        Determine if current spread is acceptable for trading.
        
        Args:
            current_spread: Current bid/ask spread
        
        Returns:
            Tuple of (is_acceptable, reason)
        """
        # Always acceptable until we have baseline
        if self.average_spread is None:
            return True, "Building spread baseline"
        
        # Check if spread is abnormally wide
        threshold = self.average_spread * self.abnormal_multiplier
        if current_spread > threshold:
            return False, f"Spread too wide: {current_spread:.4f} > {threshold:.4f} (avg: {self.average_spread:.4f})"
        
        return True, "Spread acceptable"
    
    def get_spread_stats(self) -> Dict[str, Any]:
        """Get current spread statistics."""
        return {
            "average_spread": self.average_spread,
            "std_dev_spread": self.std_dev_spread,
            "current_samples": len(self.spread_history),
            "min_spread": min(self.spread_history) if self.spread_history else None,
            "max_spread": max(self.spread_history) if self.spread_history else None
        }


class QueuePositionMonitor:
    """
    Monitors order queue position and adjusts strategy.
    Requirement 6: Queue Position Awareness
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize queue position monitor.
        
        Args:
            config: Bot configuration dictionary
        """
        self.config = config
        self.tick_size = config.get("tick_size", 0.25)
        self.max_queue_size = config.get("max_queue_size", 100)  # Cancel if queue too large
        self.queue_jump_threshold = config.get("queue_jump_threshold", 50)  # Jump ahead threshold
    
    def should_jump_queue(self, quote: BidAskQuote, side: str, current_position: int) -> Tuple[bool, float, str]:
        """
        Determine if order should jump queue by adjusting price.
        
        Args:
            quote: Current bid/ask quote
            side: Trade side ('long' or 'short')
            current_position: Current position in queue (0 = front)
        
        Returns:
            Tuple of (should_jump, new_price, reason)
        """
        # If we're at front of queue, no need to jump
        if current_position == 0:
            return False, 0.0, "Already at front of queue"
        
        # If queue is too large, consider jumping
        if current_position > self.queue_jump_threshold:
            if side == "long":
                # Jump ahead by 1 tick (pay slightly more)
                new_price = quote.bid_price + self.tick_size
                return True, new_price, f"Queue too large ({current_position}), jumping to {new_price}"
            else:  # short
                # Jump ahead by 1 tick (receive slightly less)
                new_price = quote.ask_price - self.tick_size
                return True, new_price, f"Queue too large ({current_position}), jumping to {new_price}"
        
        return False, 0.0, f"Queue position acceptable ({current_position})"
    
    def monitor_limit_order_queue(self, symbol: str, order_id: Any, limit_price: float, 
                                   side: str, get_quote_func, is_filled_func, 
                                   cancel_order_func) -> Tuple[bool, str]:
        """
        Monitor passive limit order queue position and timeout.
        Gap #2: Limit order queue monitoring
        
        This monitors a live limit order and:
        - Waits for fill up to passive_order_timeout
        - Cancels if price moves away (queue jumping not worth it)
        - Checks if filled every 500ms
        - Returns status for decision to go aggressive
        
        CRITICAL FIX: When cancellation fails due to asyncio errors, we return
        "cancel_failed" to prevent the caller from placing a duplicate order.
        The original order may still be pending at the broker.
        
        Args:
            symbol: Instrument symbol
            order_id: Broker order ID
            limit_price: Limit price of order
            side: "long" or "short"
            get_quote_func: Function to get current quote
            is_filled_func: Function to check if order filled (returns bool)
            cancel_order_func: Function to cancel order (returns bool - True if cancelled)
        
        Returns:
            Tuple of (was_filled, reason)
            - (True, "filled") - Order filled successfully
            - (False, "price_moved_away") - Price moved, cancelled successfully
            - (False, "timeout") - Timeout, cancelled successfully, safe to go aggressive
            - (False, "cancel_failed") - Cancel failed, DO NOT place another order
        """
        import time
        
        max_wait = self.config.get("passive_order_timeout", 10)
        queue_check_interval = 0.5  # Check every 500ms
        start_time = time.time()
        tick_size = self.config.get("tick_size", 0.25)
        price_move_threshold = 2  # Cancel if price moves 2+ ticks away
        
        logger.info(f"📊 Queue Monitor: Watching {side} limit @ ${limit_price:.2f}")
        logger.info(f"  Max wait: {max_wait}s, check interval: {queue_check_interval}s")
        
        check_count = 0
        while time.time() - start_time < max_wait:
            check_count += 1
            
            # Check if filled
            try:
                if is_filled_func(order_id):
                    elapsed = time.time() - start_time
                    logger.info(f"  [SUCCESS] Passive fill after {elapsed:.1f}s ({check_count} checks)")
                    return True, "filled"
            except Exception as e:
                logger.debug(f"  Error checking fill status: {e}")
            
            # Check if price moving away
            try:
                current_quote = get_quote_func(symbol)
                if current_quote:
                    if side == "long":
                        # For long, we're bidding. If ask price rises too much, cancel
                        price_distance = (current_quote.ask_price - limit_price) / tick_size
                    else:
                        # For short, we're asking. If bid price falls too much, cancel
                        price_distance = (limit_price - current_quote.bid_price) / tick_size
                    
                    if price_distance > price_move_threshold:
                        logger.warning(f"  ⚠️ Price moved away {price_distance:.1f} ticks - cancelling")
                        cancel_succeeded = False
                        try:
                            cancel_result = cancel_order_func(order_id)
                            # cancel_order_func should return True on success, False on failure.
                            # We treat None as success for backward compatibility with callers
                            # that don't return a value (implicitly return None on success).
                            cancel_succeeded = cancel_result is True or cancel_result is None
                        except Exception as e:
                            logger.error(f"  Failed to cancel order: {e}")
                            cancel_succeeded = False
                        
                        if cancel_succeeded:
                            return False, "price_moved_away"
                        else:
                            # CRITICAL: Cancel failed - do not place another order!
                            logger.error(f"  [ERROR] Cancel failed - original order may still be pending!")
                            return False, "cancel_failed"
            except Exception as e:
                logger.debug(f"  Error checking price movement: {e}")
            
            # Wait before next check
            time.sleep(queue_check_interval)
        
        # Timeout - cancel and go aggressive
        elapsed = time.time() - start_time
        logger.warning(f"  ⏱️ Queue timeout after {elapsed:.1f}s ({check_count} checks)")
        
        cancel_succeeded = False
        try:
            cancel_result = cancel_order_func(order_id)
            # cancel_order_func returns True if cancelled successfully
            cancel_succeeded = cancel_result is True or cancel_result is None
        except Exception as e:
            logger.error(f"  Failed to cancel order: {e}")
            cancel_succeeded = False
        
        if cancel_succeeded:
            return False, "timeout"
        else:
            # CRITICAL: Cancel failed - do not place another order!
            # The original passive order may still be pending and could fill
            logger.error(f"  [ERROR] Cancel failed - original order may still be pending!")
            logger.error(f"  [WARN] Not switching to aggressive to avoid duplicate orders")
            return False, "cancel_failed"
    
    def should_cancel_and_reroute(self, quote: BidAskQuote, side: str, 
                                   queue_size: int, time_in_queue: float) -> Tuple[bool, str]:
        """
        Determine if passive order should be cancelled and rerouted.
        
        Args:
            quote: Current bid/ask quote
            side: Trade side
            queue_size: Total size of queue ahead
            time_in_queue: Seconds order has been in queue
        
        Returns:
            Tuple of (should_cancel, reason)
        """
        # Cancel if queue is extremely large
        if queue_size > self.max_queue_size:
            return True, f"Queue too large ({queue_size} > {self.max_queue_size})"
        
        # Cancel if been waiting too long and market moved away
        timeout = self.config.get("passive_order_timeout", 10)
        if time_in_queue > timeout:
            return True, f"Timeout ({time_in_queue:.1f}s > {timeout}s)"
        
        return False, "Continue waiting"


class OrderRejectionValidator:
    """
    Enhanced order rejection logic.
    Requirement 8: Order Rejection Logic
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize order rejection validator.
        
        Args:
            config: Bot configuration dictionary
        """
        self.config = config
        self.max_acceptable_spread = config.get("max_acceptable_spread", None)
        self.min_bid_ask_size = config.get("min_bid_ask_size", 1)
    
    def validate_order_entry(self, quote: BidAskQuote, spread_analyzer: SpreadAnalyzer) -> Tuple[bool, str]:
        """
        Comprehensive order entry validation.
        
        Args:
            quote: Current bid/ask quote
            spread_analyzer: Spread analyzer instance
        
        Returns:
            Tuple of (is_valid, reason)
        """
        # 1. Check quote validity (inverted/crossed spread, data errors)
        is_valid, reason = quote.is_valid()
        if not is_valid:
            return False, f"Invalid quote: {reason}"
        
        # 2. Check if spread exceeds maximum acceptable threshold
        if self.max_acceptable_spread and quote.spread > self.max_acceptable_spread:
            return False, f"Spread exceeds maximum: {quote.spread:.4f} > {self.max_acceptable_spread:.4f}"
        
        # 3. Check bid/ask size (liquidity depth)
        if quote.bid_size < self.min_bid_ask_size or quote.ask_size < self.min_bid_ask_size:
            return False, f"Insufficient liquidity: bid_size={quote.bid_size}, ask_size={quote.ask_size}"
        
        # 4. Check for rapidly widening spread (market stress)
        is_widening, widening_reason = spread_analyzer.is_spread_widening()
        if is_widening:
            return False, f"Market stress: {widening_reason}"
        
        # 5. Check spread acceptability
        is_acceptable, accept_reason = spread_analyzer.is_spread_acceptable(quote.spread)
        if not is_acceptable:
            return False, accept_reason
        
        return True, "Order entry validated"


class OrderPlacementStrategy:
    """
    Intelligent order placement strategy that decides between passive and aggressive approaches.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize order placement strategy.
        
        Args:
            config: Bot configuration dictionary
        """
        self.config = config
        self.tick_size = config.get("tick_size", 0.25)
        
        # Passive order timeout settings
        self.passive_timeout_seconds = config.get("passive_order_timeout", 10)
        
        # Volatility thresholds for strategy selection
        self.high_volatility_spread_mult = config.get("high_volatility_spread_mult", 3.0)
        self.calm_market_spread_mult = config.get("calm_market_spread_mult", 1.5)
    
    def should_use_passive_entry(self, quote: BidAskQuote, spread_analyzer: SpreadAnalyzer,
                                  signal_strength: str = "normal") -> Tuple[bool, str]:
        """
        Determine if passive entry should be used.
        
        Args:
            quote: Current bid/ask quote
            spread_analyzer: Spread analyzer instance
            signal_strength: Signal strength ("strong", "normal", "weak")
        
        Returns:
            Tuple of (use_passive, reason)
        """
        spread_stats = spread_analyzer.get_spread_stats()
        avg_spread = spread_stats.get("average_spread")
        
        if avg_spread is None:
            # No baseline yet, use aggressive
            return False, "No spread baseline - use aggressive entry"
        
        current_spread = quote.spread
        
        # Use aggressive when spread is already wide (low liquidity)
        if current_spread > avg_spread * self.high_volatility_spread_mult:
            return False, f"Wide spread ({current_spread:.4f} > {avg_spread * self.high_volatility_spread_mult:.4f}) - use aggressive"
        
        # Use passive when market is calm and spread is tight
        if current_spread <= avg_spread * self.calm_market_spread_mult:
            return True, f"Tight spread ({current_spread:.4f} <= {avg_spread * self.calm_market_spread_mult:.4f}) - use passive"
        
        # Use aggressive for strong signals (time-critical)
        if signal_strength == "strong":
            return False, "Strong signal - use aggressive for guaranteed fill"
        
        # Default to passive for normal conditions
        return True, "Normal conditions - try passive first"
    
    def calculate_passive_entry_price(self, side: str, quote: BidAskQuote) -> float:
        """
        Calculate passive entry price (join the opposite side).
        
        Args:
            side: Trade side ("long" or "short")
            quote: Current bid/ask quote
        
        Returns:
            Passive limit price
        """
        if side == "long":
            # For long entry: join sellers at bid price (save the spread)
            return quote.bid_price
        else:  # short
            # For short entry: join buyers at ask price (save the spread)
            return quote.ask_price
    
    def calculate_aggressive_entry_price(self, side: str, quote: BidAskQuote) -> float:
        """
        Calculate aggressive entry price (cross the spread).
        
        Args:
            side: Trade side ("long" or "short")
            quote: Current bid/ask quote
        
        Returns:
            Aggressive limit price
        """
        if side == "long":
            # For long entry: pay the ask (guaranteed fill)
            return quote.ask_price
        else:  # short
            # For short entry: hit the bid (guaranteed fill)
            return quote.bid_price


class AdaptiveSlippageModel:
    """
    Dynamic slippage model based on market conditions.
    Requirement 7: Adaptive Slippage Model
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize adaptive slippage model.
        
        Args:
            config: Bot configuration dictionary
        """
        self.config = config
        self.tick_size = config.get("tick_size", 0.25)
        
        # Base slippage ticks for different conditions
        self.normal_hours_slippage = config.get("normal_hours_slippage_ticks", 1.0)
        self.illiquid_hours_slippage = config.get("illiquid_hours_slippage_ticks", 2.0)
        
        # Illiquid hours definition (typically overnight/early morning)
        self.illiquid_hours_start = config.get("illiquid_hours_start", time(0, 0))
        self.illiquid_hours_end = config.get("illiquid_hours_end", time(9, 30))
    
    def calculate_expected_slippage(self, quote: BidAskQuote, timestamp: datetime,
                                     spread_analyzer: SpreadAnalyzer) -> float:
        """
        Calculate expected slippage based on current market conditions.
        
        Args:
            quote: Current bid/ask quote
            timestamp: Current time
            spread_analyzer: Spread analyzer for time-of-day patterns
        
        Returns:
            Expected slippage in ticks
        """
        # Base slippage from time of day
        current_time = timestamp.time()
        
        if self.illiquid_hours_start <= current_time < self.illiquid_hours_end:
            base_slippage = self.illiquid_hours_slippage
        else:
            base_slippage = self.normal_hours_slippage
        
        # Adjust based on current spread vs normal
        expected_spread = spread_analyzer.get_expected_spread_for_time(timestamp)
        
        if expected_spread and expected_spread > 0:
            spread_ratio = quote.spread / expected_spread
            
            # If spread is wider than normal, expect more slippage
            if spread_ratio > 1.5:
                base_slippage *= 1.5
            elif spread_ratio > 1.2:
                base_slippage *= 1.2
        
        # Cap maximum slippage
        max_slippage = self.config.get("max_slippage_ticks", 3.0)
        return min(base_slippage, max_slippage)
    
    def should_avoid_trading(self, quote: BidAskQuote, timestamp: datetime,
                             spread_analyzer: SpreadAnalyzer) -> Tuple[bool, str]:
        """
        Determine if trading should be avoided due to wide spreads.
        
        Args:
            quote: Current bid/ask quote
            timestamp: Current time
            spread_analyzer: Spread analyzer
        
        Returns:
            Tuple of (should_avoid, reason)
        """
        expected_spread = spread_analyzer.get_expected_spread_for_time(timestamp)
        
        if expected_spread:
            spread_ratio = quote.spread / expected_spread
            
            # Avoid if spread is >2x normal for this time
            if spread_ratio > 2.0:
                return True, f"Spread {spread_ratio:.1f}x normal for this time ({timestamp.strftime('%H:%M')})"
        
        return False, "Trading conditions acceptable"


class DynamicFillStrategy:
    """
    Manages dynamic fill strategy including mixed orders and timeout handling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dynamic fill strategy.
        
        Args:
            config: Bot configuration dictionary
        """
        self.config = config
        self.passive_timeout = config.get("passive_order_timeout", 10)
        self.use_mixed_orders = config.get("use_mixed_order_strategy", False)
        self.mixed_passive_ratio = config.get("mixed_passive_ratio", 0.5)
    
    def should_use_mixed_strategy(self, contracts: int, market_volatility: str = "normal") -> Tuple[bool, int, int]:
        """
        Determine if mixed strategy should be used and calculate split.
        
        Args:
            contracts: Total number of contracts to trade
            market_volatility: Market volatility level ("calm", "normal", "high")
        
        Returns:
            Tuple of (use_mixed, passive_contracts, aggressive_contracts)
        """
        # Don't use mixed for single contract
        if contracts == 1:
            return False, 0, 0
        
        # Only use mixed if enabled in config
        if not self.use_mixed_orders:
            return False, 0, 0
        
        # Calculate split based on config ratio
        passive_contracts = int(contracts * self.mixed_passive_ratio)
        aggressive_contracts = contracts - passive_contracts
        
        # Ensure at least 1 contract on each side
        if passive_contracts == 0 or aggressive_contracts == 0:
            return False, 0, 0
        
        return True, passive_contracts, aggressive_contracts
    
    def get_retry_strategy(self, attempt: int, max_attempts: int = 3) -> Dict[str, Any]:
        """
        Get retry strategy parameters for failed passive orders.
        
        Args:
            attempt: Current attempt number (1-indexed)
            max_attempts: Maximum retry attempts
        
        Returns:
            Strategy parameters dictionary
        """
        if attempt >= max_attempts:
            return {
                "strategy": "aggressive",
                "timeout": 0,
                "reason": "Max passive attempts reached"
            }
        
        # Exponentially decrease timeout for retries
        timeout = self.passive_timeout / (2 ** (attempt - 1))
        
        return {
            "strategy": "passive",
            "timeout": max(timeout, 2),  # Minimum 2 seconds
            "reason": f"Retry attempt {attempt}/{max_attempts}"
        }


class MarketConditionClassifier:
    """
    Classifies current market conditions for optimal order routing.
    Requirement 11: Market Condition Classification
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize market condition classifier.
        
        Args:
            config: Bot configuration dictionary
        """
        self.config = config
        
        # Thresholds for classification
        self.tight_spread_multiplier = config.get("tight_spread_multiplier", 1.2)
        self.wide_spread_multiplier = config.get("wide_spread_multiplier", 2.0)
        self.extreme_spread_multiplier = config.get("extreme_spread_multiplier", 3.0)
        self.low_volume_threshold = config.get("low_volume_threshold", 0.5)
    
    def classify_market(self, quote: BidAskQuote, spread_analyzer: SpreadAnalyzer) -> Tuple[str, str]:
        """
        Classify current market condition.
        
        Args:
            quote: Current bid/ask quote
            spread_analyzer: Spread analyzer instance
        
        Returns:
            Tuple of (condition, reason)
            Conditions: "normal", "volatile", "illiquid", "stressed"
        """
        spread_stats = spread_analyzer.get_spread_stats()
        avg_spread = spread_stats.get("average_spread")
        
        if avg_spread is None or avg_spread == 0:
            return "normal", "Building baseline"
        
        current_spread = quote.spread
        spread_ratio = current_spread / avg_spread
        
        # Stressed: Extremely wide spreads
        if spread_ratio >= self.extreme_spread_multiplier:
            return "stressed", f"Extreme spread: {spread_ratio:.1f}x average - skip trading"
        
        # Illiquid: Wide spreads + low size
        if spread_ratio >= self.wide_spread_multiplier:
            if quote.bid_size < 10 or quote.ask_size < 10:
                return "illiquid", f"Wide spread ({spread_ratio:.1f}x) + low volume - reduce size or avoid"
            else:
                return "volatile", f"Wide spread ({spread_ratio:.1f}x) + fast movement - use aggressive"
        
        # Volatile: Spread widening
        is_widening, _ = spread_analyzer.is_spread_widening()
        if is_widening:
            return "volatile", "Spread widening - use aggressive orders"
        
        # Normal: Tight spreads, good liquidity
        if spread_ratio <= self.tight_spread_multiplier:
            return "normal", f"Tight spread ({spread_ratio:.1f}x) + good liquidity - use passive"
        
        return "normal", "Normal market conditions"


class FillProbabilityEstimator:
    """
    Estimates probability of passive order fill.
    Requirement 12: Fill Probability Estimation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize fill probability estimator.
        
        Args:
            config: Bot configuration dictionary
        """
        self.config = config
        self.tick_size = config.get("tick_size", 0.25)
    
    def estimate_fill_probability(self, quote: BidAskQuote, side: str, 
                                   price_momentum: Optional[float] = None) -> Tuple[float, float, str]:
        """
        Estimate likelihood of passive fill.
        
        Args:
            quote: Current bid/ask quote
            side: Trade side ('long' or 'short')
            price_momentum: Recent price change (optional)
        
        Returns:
            Tuple of (fill_probability, expected_wait_seconds, reason)
        """
        # Base probability from spread and sizes
        if side == "long":
            # Joining bid - need sellers to come
            depth_ratio = quote.ask_size / max(quote.bid_size, 1)
            base_prob = 0.7  # 70% base probability
        else:
            # Joining ask - need buyers to come
            depth_ratio = quote.bid_size / max(quote.ask_size, 1)
            base_prob = 0.7
        
        # Adjust for depth imbalance
        if depth_ratio > 2.0:
            # Lots of pressure on opposite side - higher fill probability
            fill_prob = min(base_prob * 1.3, 0.95)
            expected_wait = 3.0
            reason = "High fill probability - strong opposite-side pressure"
        elif depth_ratio < 0.5:
            # Pressure on our side - lower fill probability
            fill_prob = base_prob * 0.6
            expected_wait = 15.0
            reason = "Lower fill probability - pressure on our side"
        else:
            fill_prob = base_prob
            expected_wait = 7.0
            reason = "Normal fill probability"
        
        # Adjust for momentum if available
        if price_momentum is not None:
            if side == "long" and price_momentum > 0:
                # Price moving up, less likely to fill at bid
                fill_prob *= 0.7
                expected_wait *= 1.5
                reason += " - price moving away"
            elif side == "short" and price_momentum < 0:
                # Price moving down, less likely to fill at ask
                fill_prob *= 0.7
                expected_wait *= 1.5
                reason += " - price moving away"
            elif side == "long" and price_momentum < 0:
                # Price moving down toward our bid
                fill_prob *= 1.2
                expected_wait *= 0.7
                reason += " - price moving toward us"
            elif side == "short" and price_momentum > 0:
                # Price moving up toward our ask
                fill_prob *= 1.2
                expected_wait *= 0.7
                reason += " - price moving toward us"
        
        return min(fill_prob, 1.0), expected_wait, reason
    
    def should_wait_for_passive(self, fill_probability: float, expected_wait: float) -> Tuple[bool, str]:
        """
        Decide if worth waiting for passive fill.
        
        Args:
            fill_probability: Estimated fill probability (0-1)
            expected_wait: Expected wait time in seconds
        
        Returns:
            Tuple of (should_wait, reason)
        """
        max_wait = self.config.get("passive_order_timeout", 10)
        min_probability = self.config.get("min_fill_probability", 0.5)
        
        if fill_probability < min_probability:
            return False, f"Fill probability too low ({fill_probability:.1%} < {min_probability:.1%})"
        
        if expected_wait > max_wait:
            return False, f"Expected wait too long ({expected_wait:.1f}s > {max_wait}s)"
        
        return True, f"Good odds: {fill_probability:.1%} probability in {expected_wait:.1f}s"


class PostTradeAnalyzer:
    """
    Analyzes trade execution quality for continuous improvement.
    Requirement 13: Post-Trade Analysis
    """
    
    def __init__(self):
        """Initialize post-trade analyzer."""
        self.trade_records: List[Dict[str, Any]] = []
    
    def record_trade(self, signal_price: float, fill_price: float, side: str,
                     order_type: str, spread_at_order: float, fill_time_seconds: float,
                     estimated_costs: Dict[str, float], actual_costs: Dict[str, float]) -> None:
        """
        Record trade execution details.
        
        Args:
            signal_price: Original signal price
            fill_price: Actual fill price
            side: Trade side ('long' or 'short')
            order_type: 'passive' or 'aggressive'
            spread_at_order: Spread at time of order
            fill_time_seconds: Time from signal to fill
            estimated_costs: Estimated transaction costs
            actual_costs: Actual transaction costs
        """
        variance = fill_price - signal_price
        
        record = {
            "timestamp": datetime.now(),
            "signal_price": signal_price,
            "fill_price": fill_price,
            "side": side,
            "variance": variance,
            "order_type": order_type,
            "spread_at_order": spread_at_order,
            "spread_saved": spread_at_order if order_type == "passive" else -spread_at_order,
            "fill_time_seconds": fill_time_seconds,
            "estimated_costs": estimated_costs,
            "actual_costs": actual_costs,
            "cost_variance": actual_costs.get("total", 0) - estimated_costs.get("total", 0)
        }
        
        self.trade_records.append(record)
    
    def get_learning_insights(self, market_condition: str) -> Dict[str, Any]:
        """
        Analyze which conditions favor passive vs aggressive.
        
        Args:
            market_condition: Current market condition
        
        Returns:
            Learning insights dictionary
        """
        if not self.trade_records:
            return {}
        
        # Recent trades (last 20)
        recent_trades = self.trade_records[-20:]
        
        passive_trades = [t for t in recent_trades if t["order_type"] == "passive"]
        aggressive_trades = [t for t in recent_trades if t["order_type"] == "aggressive"]
        
        insights = {
            "total_trades": len(recent_trades),
            "passive_count": len(passive_trades),
            "aggressive_count": len(aggressive_trades),
            "passive_avg_fill_time": statistics.mean([t["fill_time_seconds"] for t in passive_trades]) if passive_trades else 0,
            "aggressive_avg_fill_time": statistics.mean([t["fill_time_seconds"] for t in aggressive_trades]) if aggressive_trades else 0,
            "passive_avg_savings": statistics.mean([t["spread_saved"] for t in passive_trades]) if passive_trades else 0,
            "aggressive_avg_cost": statistics.mean([abs(t["spread_saved"]) for t in aggressive_trades]) if aggressive_trades else 0,
        }
        
        return insights


class SpreadAwarePositionSizer:
    """
    Adjusts position size based on spread costs.
    Requirement 10: Spread-Aware Position Sizing
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize spread-aware position sizer.
        
        Args:
            config: Bot configuration dictionary
        """
        self.config = config
        self.tick_size = config.get("tick_size", 0.25)
        self.tick_value = config.get("tick_value", 12.50)
        self.max_transaction_cost_pct = config.get("max_transaction_cost_pct", 0.15)  # 15% of expected profit
    
    def calculate_position_size(self, quote: BidAskQuote, base_contracts: int,
                                 expected_profit_ticks: float, slippage_ticks: float,
                                 commission_per_contract: float) -> Tuple[int, Dict[str, float]]:
        """
        Calculate position size accounting for spread costs.
        
        Args:
            quote: Current bid/ask quote
            base_contracts: Initial position size calculation
            expected_profit_ticks: Expected profit in ticks
            slippage_ticks: Expected slippage
            commission_per_contract: Commission cost
        
        Returns:
            Tuple of (adjusted_contracts, cost_breakdown)
        """
        # Calculate total transaction costs
        spread_ticks = quote.spread / self.tick_size
        total_cost_ticks = spread_ticks + slippage_ticks
        total_cost_dollars = (total_cost_ticks * self.tick_value) + commission_per_contract
        
        expected_profit_dollars = expected_profit_ticks * self.tick_value
        
        # Check if cost is acceptable percentage of profit
        cost_pct = (total_cost_dollars / expected_profit_dollars) if expected_profit_dollars > 0 else 1.0
        
        cost_breakdown = {
            "spread_ticks": spread_ticks,
            "slippage_ticks": slippage_ticks,
            "total_cost_ticks": total_cost_ticks,
            "total_cost_dollars": total_cost_dollars,
            "commission": commission_per_contract,
            "expected_profit_dollars": expected_profit_dollars,
            "cost_percentage": cost_pct * 100
        }
        
        # Reduce size if spread is wide
        if cost_pct > self.max_transaction_cost_pct:
            # Calculate how many contracts we can afford
            max_affordable = int(base_contracts * (self.max_transaction_cost_pct / cost_pct))
            adjusted_contracts = max(1, max_affordable)  # At least 1 contract
            
            logger.warning(f"Reducing position size due to high transaction costs: "
                          f"{base_contracts} -> {adjusted_contracts} contracts "
                          f"(cost {cost_pct*100:.1f}% > max {self.max_transaction_cost_pct*100:.1f}%)")
            
            return adjusted_contracts, cost_breakdown
        
        return base_contracts, cost_breakdown


class ExitOrderOptimizer:
    """
    Optimizes exit orders using bid/ask logic.
    Requirement 9: Exit Order Optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize exit order optimizer.
        
        Args:
            config: Bot configuration dictionary
        """
        self.config = config
        self.tick_size = config.get("tick_size", 0.25)
    
    def get_exit_strategy(self, exit_type: str, quote: BidAskQuote, side: str,
                          urgency: str = "normal") -> Dict[str, Any]:
        """
        Determine optimal exit order strategy.
        
        Args:
            exit_type: Type of exit ("target", "stop", "time_flatten", "partial")
            quote: Current bid/ask quote
            side: Current position side ('long' or 'short')
            urgency: Urgency level ("low", "normal", "high")
        
        Returns:
            Exit strategy parameters
        """
        # HIGH URGENCY: Always use aggressive for fast exit
        if urgency == "high":
            if side == "long":
                return {
                    "order_type": "aggressive",
                    "limit_price": quote.bid_price,
                    "timeout": 0,
                    "reason": f"High urgency ({exit_type}) - aggressive market order"
                }
            else:  # short
                return {
                    "order_type": "aggressive",
                    "limit_price": quote.ask_price,
                    "timeout": 0,
                    "reason": f"High urgency ({exit_type}) - aggressive market order"
                }
        
        if exit_type == "target":
            # Taking profit at target - use passive to collect spread
            if side == "long":
                # Selling at target - place at ask to collect spread
                return {
                    "order_type": "passive",
                    "limit_price": quote.ask_price,
                    "timeout": self.config.get("passive_order_timeout", 10),
                    "reason": "Profit target - passive limit at ask to collect spread"
                }
            else:  # short
                # Buying to close short - place at bid to collect spread
                return {
                    "order_type": "passive",
                    "limit_price": quote.bid_price,
                    "timeout": self.config.get("passive_order_timeout", 10),
                    "reason": "Profit target - passive limit at bid to collect spread"
                }
        
        elif exit_type == "stop":
            # Stop loss hit - use aggressive for fast exit
            if side == "long":
                # Selling to stop - hit the bid
                return {
                    "order_type": "aggressive",
                    "limit_price": quote.bid_price,
                    "timeout": 0,
                    "reason": "Stop loss - aggressive market order for fast exit"
                }
            else:  # short
                # Buying to close short - pay the ask
                return {
                    "order_type": "aggressive",
                    "limit_price": quote.ask_price,
                    "timeout": 0,
                    "reason": "Stop loss - aggressive market order for fast exit"
                }
        
        elif exit_type == "time_flatten":
            # Time-based flatten - priority is closing before cutoff
            if side == "long":
                return {
                    "order_type": "aggressive",
                    "limit_price": quote.bid_price,
                    "timeout": 0,
                    "reason": "Time flatten - aggressive to ensure close before cutoff"
                }
            else:
                return {
                    "order_type": "aggressive",
                    "limit_price": quote.ask_price,
                    "timeout": 0,
                    "reason": "Time flatten - aggressive to ensure close before cutoff"
                }
        
        elif exit_type == "partial":
            # Partial exit - try passive first, aggressive if not filled
            if side == "long":
                return {
                    "order_type": "passive",
                    "limit_price": quote.ask_price,
                    "timeout": 5,  # Shorter timeout for partials
                    "fallback_price": quote.bid_price,
                    "reason": "Partial exit - passive first, aggressive if not filled in 5s"
                }
            else:
                return {
                    "order_type": "passive",
                    "limit_price": quote.bid_price,
                    "timeout": 5,
                    "fallback_price": quote.ask_price,
                    "reason": "Partial exit - passive first, aggressive if not filled in 5s"
                }
        
        # Default to aggressive
        return {
            "order_type": "aggressive",
            "limit_price": quote.bid_price if side == "long" else quote.ask_price,
            "timeout": 0,
            "reason": "Default aggressive exit"
        }


class BidAskManager:
    """
    Complete bid/ask trading manager that coordinates quote tracking, spread analysis,
    and intelligent order placement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize bid/ask manager.
        
        Args:
            config: Bot configuration dictionary
        """
        self.config = config
        self.quotes: Dict[str, BidAskQuote] = {}
        self.spread_analyzers: Dict[str, SpreadAnalyzer] = {}
        self.order_strategy = OrderPlacementStrategy(config)
        self.fill_strategy = DynamicFillStrategy(config)
        
        # New components (Requirements 5-8)
        self.spread_cost_tracker = SpreadCostTracker()
        self.queue_monitor = QueuePositionMonitor(config)
        self.rejection_validator = OrderRejectionValidator(config)
        self.slippage_model = AdaptiveSlippageModel(config)
        
        # Advanced components (Requirements 9-13)
        self.market_classifier = MarketConditionClassifier(config)
        self.fill_estimator = FillProbabilityEstimator(config)
        self.post_trade_analyzer = PostTradeAnalyzer()
        self.position_sizer = SpreadAwarePositionSizer(config)
        self.exit_optimizer = ExitOrderOptimizer(config)
        
        logger.info("Bid/Ask Manager initialized")
        logger.info(f"  Passive order timeout: {config.get('passive_order_timeout', 10)}s")
        logger.info(f"  Abnormal spread multiplier: {config.get('abnormal_spread_multiplier', 2.0)}x")
        logger.info(f"  Mixed order strategy: {config.get('use_mixed_order_strategy', False)}")
        logger.info(f"  Max queue size: {config.get('max_queue_size', 100)}")
        logger.info(f"  Min bid/ask size: {config.get('min_bid_ask_size', 1)}")
        logger.info(f"  Market condition classification: Enabled")
        logger.info(f"  Fill probability estimation: Enabled")
        logger.info(f"  Spread-aware position sizing: Enabled")
    
    def update_quote(self, symbol: str, bid_price: float, ask_price: float,
                     bid_size: int, ask_size: int, last_price: float, timestamp: int) -> None:
        """
        Update bid/ask quote for a symbol.
        
        Args:
            symbol: Instrument symbol
            bid_price: Current bid price
            ask_price: Current ask price
            bid_size: Bid size (contracts)
            ask_size: Ask size (contracts)
            last_price: Last trade price
            timestamp: Quote timestamp (milliseconds)
        """
        quote = BidAskQuote(
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            last_trade_price=last_price,
            timestamp=timestamp
        )
        
        self.quotes[symbol] = quote
        
        # Update spread analyzer ONLY if quote is valid (no corrupted data)
        if symbol not in self.spread_analyzers:
            self.spread_analyzers[symbol] = SpreadAnalyzer(
                lookback_periods=self.config.get("spread_lookback_periods", 100),
                abnormal_multiplier=self.config.get("abnormal_spread_multiplier", 2.0)
            )
        
        # Validate quote before updating spread history to prevent data corruption
        is_valid, validation_reason = quote.is_valid()
        if is_valid:
            # Update spread with timestamp for time-of-day tracking
            from datetime import datetime
            import pytz
            tz = pytz.timezone(self.config.get("timezone", "US/Eastern"))
            dt = datetime.fromtimestamp(timestamp / 1000, tz=tz)
            self.spread_analyzers[symbol].update(quote.spread, dt)
        else:
            # Skip invalid quotes to prevent spread history corruption
            logger.debug(f"Skipping spread update for {symbol}: {validation_reason}")
        
        logger.debug(f"Quote updated for {symbol}: Bid={bid_price:.2f}x{bid_size} "
                    f"Ask={ask_price:.2f}x{ask_size} Spread={quote.spread:.4f}")
    
    def get_current_quote(self, symbol: str) -> Optional[BidAskQuote]:
        """Get current quote for symbol."""
        return self.quotes.get(symbol)
    
    def validate_entry_spread(self, symbol: str) -> Tuple[bool, str]:
        """
        Validate that spread is acceptable for entry with enhanced rejection logic.
        
        Args:
            symbol: Instrument symbol
        
        Returns:
            Tuple of (is_acceptable, reason)
        """
        quote = self.quotes.get(symbol)
        if quote is None:
            return False, "No bid/ask quote available"
        
        analyzer = self.spread_analyzers.get(symbol)
        if analyzer is None:
            return False, "Spread analyzer not initialized"
        
        # Use enhanced rejection validator
        return self.rejection_validator.validate_order_entry(quote, analyzer)
    
    def get_entry_order_params(self, symbol: str, side: str, contracts: int,
                               signal_strength: str = "normal") -> Dict[str, Any]:
        """
        Get intelligent order parameters for entry.
        
        Args:
            symbol: Instrument symbol
            side: Trade side ("long" or "short")
            contracts: Number of contracts
            signal_strength: Signal strength ("strong", "normal", "weak")
        
        Returns:
            Order parameters dictionary with strategy details
        """
        quote = self.quotes.get(symbol)
        if quote is None:
            raise ValueError(f"No quote available for {symbol}")
        
        analyzer = self.spread_analyzers.get(symbol)
        if analyzer is None:
            raise ValueError(f"No spread analyzer for {symbol}")
        
        # ===== Gap #3: Bid/Ask Imbalance Detection =====
        imbalance_enabled = self.config.get("imbalance_detection_enabled", True)
        imbalance_signal = "balanced"
        imbalance_ratio = 1.0
        
        if imbalance_enabled:
            imbalance_threshold = self.config.get("imbalance_threshold_ratio", 3.0)
            imbalance_signal = quote.get_imbalance_signal(imbalance_threshold)
            imbalance_ratio = quote.imbalance_ratio
            
            if imbalance_signal != "balanced":
                logger.info(f"  📊 Imbalance detected: {imbalance_signal} (ratio: {imbalance_ratio:.2f})")
                
                # Adjust urgency based on imbalance and side
                if side == "long" and imbalance_signal == "strong_bid":
                    # Heavy buying pressure - be more aggressive on long entries
                    logger.info(f"  ⚡ Strong buying pressure - increasing long entry urgency")
                    signal_strength = "strong"  # Override to force aggressive
                elif side == "short" and imbalance_signal == "strong_ask":
                    # Heavy selling pressure - be more aggressive on short entries
                    logger.info(f"  ⚡ Strong selling pressure - increasing short entry urgency")
                    signal_strength = "strong"  # Override to force aggressive
                elif side == "long" and imbalance_signal == "strong_ask":
                    # Heavy selling against our long - be more patient
                    logger.info(f"  ⏸️ Strong selling pressure - reducing long entry urgency")
                    signal_strength = "weak"  # Try passive first
                elif side == "short" and imbalance_signal == "strong_bid":
                    # Heavy buying against our short - be more patient
                    logger.info(f"  ⏸️ Strong buying pressure - reducing short entry urgency")
                    signal_strength = "weak"  # Try passive first
        
        # Determine passive vs aggressive strategy
        use_passive, passive_reason = self.order_strategy.should_use_passive_entry(
            quote, analyzer, signal_strength
        )
        
        # Check for mixed order strategy
        use_mixed, passive_qty, aggressive_qty = self.fill_strategy.should_use_mixed_strategy(contracts)
        
        if use_mixed:
            # Mixed strategy: split between passive and aggressive
            passive_price = self.order_strategy.calculate_passive_entry_price(side, quote)
            aggressive_price = self.order_strategy.calculate_aggressive_entry_price(side, quote)
            
            return {
                "strategy": "mixed",
                "passive_contracts": passive_qty,
                "aggressive_contracts": aggressive_qty,
                "passive_price": passive_price,
                "aggressive_price": aggressive_price,
                "timeout": self.fill_strategy.passive_timeout,
                "reason": f"Mixed strategy: {passive_qty} passive + {aggressive_qty} aggressive",
                "quote": quote,
                "imbalance_signal": imbalance_signal,
                "imbalance_ratio": imbalance_ratio
            }
        elif use_passive:
            # Pure passive strategy
            passive_price = self.order_strategy.calculate_passive_entry_price(side, quote)
            
            return {
                "strategy": "passive",
                "contracts": contracts,
                "limit_price": passive_price,
                "timeout": self.fill_strategy.passive_timeout,
                "reason": passive_reason,
                "quote": quote,
                "fallback_price": self.order_strategy.calculate_aggressive_entry_price(side, quote),
                "imbalance_signal": imbalance_signal,
                "imbalance_ratio": imbalance_ratio
            }
        else:
            # Pure aggressive strategy
            aggressive_price = self.order_strategy.calculate_aggressive_entry_price(side, quote)
            
            return {
                "strategy": "aggressive",
                "contracts": contracts,
                "limit_price": aggressive_price,
                "timeout": 0,  # No timeout for aggressive
                "reason": passive_reason,  # Reason explains why not passive
                "quote": quote,
                "imbalance_signal": imbalance_signal,
                "imbalance_ratio": imbalance_ratio
            }
    
    def get_spread_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get spread statistics for a symbol."""
        analyzer = self.spread_analyzers.get(symbol)
        if analyzer is None:
            return {}
        
        return analyzer.get_spread_stats()
    
    def record_trade_execution(self, symbol: str, side: str, signal_price: float,
                               fill_price: float, quantity: int, order_type: str) -> None:
        """
        Record a trade execution for spread cost tracking.
        
        Args:
            symbol: Instrument symbol
            side: Trade side ('long' or 'short')
            signal_price: Original signal price
            fill_price: Actual fill price
            quantity: Number of contracts
            order_type: 'passive' or 'aggressive'
        """
        quote = self.quotes.get(symbol)
        if quote is None:
            logger.warning(f"Cannot record execution for {symbol}: no quote data")
            return
        
        execution = TradeExecution(
            symbol=symbol,
            side=side,
            signal_price=signal_price,
            spread_at_order=quote.spread,
            fill_price=fill_price,
            quantity=quantity,
            order_type=order_type,
            timestamp=datetime.now()
        )
        
        self.spread_cost_tracker.record_execution(execution)
        
        logger.info(f"Recorded execution: {order_type} {side} {quantity} @ {fill_price:.2f}, "
                   f"spread_saved={execution.spread_saved:.4f}")
    
    def get_spread_cost_stats(self) -> Dict[str, Any]:
        """Get cumulative spread cost statistics."""
        return self.spread_cost_tracker.get_statistics()
    
    def get_expected_slippage(self, symbol: str, timestamp: datetime) -> Optional[float]:
        """
        Get expected slippage for current market conditions.
        
        Args:
            symbol: Instrument symbol
            timestamp: Current time
        
        Returns:
            Expected slippage in ticks, or None if not enough data
        """
        quote = self.quotes.get(symbol)
        analyzer = self.spread_analyzers.get(symbol)
        
        if quote is None or analyzer is None:
            return None
        
        return self.slippage_model.calculate_expected_slippage(quote, timestamp, analyzer)
    
    def should_jump_queue(self, symbol: str, side: str, queue_position: int) -> Tuple[bool, float, str]:
        """
        Determine if order should jump queue.
        
        Args:
            symbol: Instrument symbol
            side: Trade side
            queue_position: Current position in queue
        
        Returns:
            Tuple of (should_jump, new_price, reason)
        """
        quote = self.quotes.get(symbol)
        if quote is None:
            return False, 0.0, "No quote available"
        
        return self.queue_monitor.should_jump_queue(quote, side, queue_position)
    
    def classify_market_condition(self, symbol: str) -> Tuple[str, str]:
        """
        Classify current market condition.
        Requirement 11: Market Condition Classification
        
        Args:
            symbol: Instrument symbol
        
        Returns:
            Tuple of (condition, reason)
            Conditions: "normal", "volatile", "illiquid", "stressed"
        """
        quote = self.quotes.get(symbol)
        analyzer = self.spread_analyzers.get(symbol)
        
        if quote is None or analyzer is None:
            return "normal", "No data available"
        
        return self.market_classifier.classify_market(quote, analyzer)
    
    def estimate_fill_probability(self, symbol: str, side: str, 
                                   price_momentum: Optional[float] = None) -> Tuple[float, float, str]:
        """
        Estimate probability of passive order fill.
        Requirement 12: Fill Probability Estimation
        
        Args:
            symbol: Instrument symbol
            side: Trade side ('long' or 'short')
            price_momentum: Recent price change (optional)
        
        Returns:
            Tuple of (fill_probability, expected_wait_seconds, reason)
        """
        quote = self.quotes.get(symbol)
        if quote is None:
            return 0.0, 0.0, "No quote available"
        
        return self.fill_estimator.estimate_fill_probability(quote, side, price_momentum)
    
    def record_post_trade_analysis(self, signal_price: float, fill_price: float, side: str,
                                    order_type: str, spread_at_order: float, fill_time_seconds: float,
                                    estimated_costs: Dict[str, float], actual_costs: Dict[str, float]) -> None:
        """
        Record trade for post-trade analysis.
        Requirement 13: Post-Trade Analysis
        
        Args:
            signal_price: Original signal price
            fill_price: Actual fill price
            side: Trade side
            order_type: 'passive' or 'aggressive'
            spread_at_order: Spread at time of order
            fill_time_seconds: Time from signal to fill
            estimated_costs: Estimated transaction costs
            actual_costs: Actual transaction costs
        """
        self.post_trade_analyzer.record_trade(
            signal_price, fill_price, side, order_type, spread_at_order,
            fill_time_seconds, estimated_costs, actual_costs
        )
    
    def get_learning_insights(self, market_condition: str) -> Dict[str, Any]:
        """
        Get learning insights from post-trade analysis.
        Requirement 13: Post-Trade Analysis
        
        Args:
            market_condition: Current market condition
        
        Returns:
            Learning insights dictionary
        """
        return self.post_trade_analyzer.get_learning_insights(market_condition)
    
    def calculate_spread_aware_position_size(self, symbol: str, base_contracts: int,
                                              expected_profit_ticks: float) -> Tuple[int, Dict[str, float]]:
        """
        Calculate position size accounting for spread costs.
        Requirement 10: Spread-Aware Position Sizing
        
        Args:
            symbol: Instrument symbol
            base_contracts: Initial position size calculation
            expected_profit_ticks: Expected profit in ticks
        
        Returns:
            Tuple of (adjusted_contracts, cost_breakdown)
        """
        quote = self.quotes.get(symbol)
        if quote is None:
            return base_contracts, {}
        
        # Get expected slippage
        slippage_ticks = self.get_expected_slippage(symbol, datetime.now()) or 1.0
        commission = self.config.get("commission_per_contract", 2.50)
        
        return self.position_sizer.calculate_position_size(
            quote, base_contracts, expected_profit_ticks, slippage_ticks, commission
        )
    
    def get_exit_order_strategy(self, exit_type: str, symbol: str, side: str,
                                 urgency: str = "normal") -> Dict[str, Any]:
        """
        Get optimal exit order strategy.
        Requirement 9: Exit Order Optimization
        
        Args:
            exit_type: Type of exit ("target", "stop", "time_flatten", "partial")
            symbol: Instrument symbol
            side: Current position side ('long' or 'short')
            urgency: Urgency level ("low", "normal", "high")
        
        Returns:
            Exit strategy parameters
        """
        quote = self.quotes.get(symbol)
        if quote is None:
            return {
                "order_type": "aggressive",
                "limit_price": 0.0,
                "timeout": 0,
                "reason": "No quote available - default aggressive"
            }
        
        return self.exit_optimizer.get_exit_strategy(exit_type, quote, side, urgency)
