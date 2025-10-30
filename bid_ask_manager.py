"""
Bid/Ask Manager - Complete Trading Strategy with No Cutting Corners
Manages real-time bid/ask quotes, spread analysis, and intelligent order placement.
"""

import logging
from typing import Dict, Any, Optional, Tuple, Deque
from collections import deque
from datetime import datetime
from dataclasses import dataclass
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
    
    def update(self, spread: float) -> None:
        """
        Update spread history with new spread value.
        
        Args:
            spread: Current bid/ask spread
        """
        self.spread_history.append(spread)
        
        # Recalculate statistics if we have enough data
        if len(self.spread_history) >= 20:  # Minimum 20 samples for stats
            self.average_spread = statistics.mean(self.spread_history)
            if len(self.spread_history) >= 2:
                self.std_dev_spread = statistics.stdev(self.spread_history)
    
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
        self.tick_size = config["tick_size"]
        
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
        
        logger.info("Bid/Ask Manager initialized")
        logger.info(f"  Passive order timeout: {config.get('passive_order_timeout', 10)}s")
        logger.info(f"  Abnormal spread multiplier: {config.get('abnormal_spread_multiplier', 2.0)}x")
        logger.info(f"  Mixed order strategy: {config.get('use_mixed_order_strategy', False)}")
    
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
        
        # Update spread analyzer
        if symbol not in self.spread_analyzers:
            self.spread_analyzers[symbol] = SpreadAnalyzer(
                lookback_periods=self.config.get("spread_lookback_periods", 100),
                abnormal_multiplier=self.config.get("abnormal_spread_multiplier", 2.0)
            )
        
        self.spread_analyzers[symbol].update(quote.spread)
        
        logger.debug(f"Quote updated for {symbol}: Bid={bid_price:.2f}x{bid_size} "
                    f"Ask={ask_price:.2f}x{ask_size} Spread={quote.spread:.4f}")
    
    def get_current_quote(self, symbol: str) -> Optional[BidAskQuote]:
        """Get current quote for symbol."""
        return self.quotes.get(symbol)
    
    def validate_entry_spread(self, symbol: str) -> Tuple[bool, str]:
        """
        Validate that spread is acceptable for entry.
        
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
        
        return analyzer.is_spread_acceptable(quote.spread)
    
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
                "quote": quote
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
                "fallback_price": self.order_strategy.calculate_aggressive_entry_price(side, quote)
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
                "quote": quote
            }
    
    def get_spread_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get spread statistics for a symbol."""
        analyzer = self.spread_analyzers.get(symbol)
        if analyzer is None:
            return {}
        
        return analyzer.get_spread_stats()
