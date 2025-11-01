"""
Adaptive Exit Management System
================================
Dynamic exit parameters that adjust to real-time market conditions.

Instead of static thresholds, this system adapts based on:
- Current volatility (ATR)
- Market regime (trending vs choppy)
- Trade performance
- Time of day
- Position holding duration
- Volume confirmation
- Consecutive win/loss streaks
- Session context (Asian/European/US)
"""

import logging
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class AdaptiveExitManager:
    """
    Manages adaptive exit parameters based on real-time market conditions.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize adaptive exit manager.
        
        Args:
            config: Bot configuration dictionary
        """
        self.config = config
        self.tick_size = config["tick_size"]
        self.tick_value = config["tick_value"]
        
        # Base parameters (fallback if adaptive disabled)
        self.base_breakeven_threshold = config.get("breakeven_profit_threshold_ticks", 8)
        self.base_breakeven_offset = config.get("breakeven_stop_offset_ticks", 1)
        self.base_trailing_distance = config.get("trailing_stop_distance_ticks", 8)
        self.base_trailing_min_profit = config.get("trailing_stop_min_profit_ticks", 12)
        
        # Adaptive scaling factors
        self.volatility_scaling_enabled = config.get("adaptive_volatility_scaling", True)
        self.regime_adaptation_enabled = config.get("adaptive_regime_detection", True)
        self.performance_adaptation_enabled = config.get("adaptive_performance_based", True)
        
        # Track recent trade results for streak detection
        self.recent_trades = deque(maxlen=10)  # Last 10 trades
        
        logger.info("Adaptive Exit Manager initialized")
    
    def calculate_market_volatility(self, bars: list, period: int = 14) -> float:
        """
        Calculate current market volatility using ATR.
        
        Args:
            bars: List of OHLC bars (or deque)
            period: ATR period
            
        Returns:
            Current ATR value
        """
        # Convert deque to list if needed
        if not isinstance(bars, list):
            bars = list(bars)
        
        if len(bars) < period + 1:
            return self._estimate_volatility(bars)
        
        recent_bars = bars[-(period + 1):]
        trs = []
        
        for i in range(1, len(recent_bars)):
            high = recent_bars[i]["high"]
            low = recent_bars[i]["low"]
            prev_close = recent_bars[i-1]["close"]
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            trs.append(tr)
        
        atr = np.mean(trs)
        return atr
    
    def _estimate_volatility(self, bars: list) -> float:
        """Estimate volatility from limited bars."""
        # Convert deque to list if needed
        if not isinstance(bars, list):
            bars = list(bars)
        
        if len(bars) < 2:
            return 2.0  # Default ATR for ES
        
        ranges = [bar["high"] - bar["low"] for bar in bars[-min(10, len(bars)):]]
        return np.mean(ranges)
    
    def detect_market_regime(self, bars: list, ema_period: int = 20) -> str:
        """
        Detect current market regime: trending or choppy.
        
        Args:
            bars: List of OHLC bars (or deque)
            ema_period: EMA period for trend detection
            
        Returns:
            'trending' or 'choppy'
        """
        # Convert deque to list if needed
        if not isinstance(bars, list):
            bars = list(bars)
        
        if len(bars) < ema_period:
            return 'choppy'  # Default to choppy with limited data
        
        # Calculate EMA
        closes = [bar["close"] for bar in bars[-ema_period:]]
        ema = self._calculate_ema(closes, ema_period)
        current_price = bars[-1]["close"]
        
        # Calculate ADX-like metric (directional strength)
        price_moves = []
        for i in range(len(bars) - min(14, len(bars)), len(bars) - 1):
            move = bars[i+1]["close"] - bars[i]["close"]
            price_moves.append(abs(move))
        
        avg_move = np.mean(price_moves) if price_moves else 0
        
        # Trending if price clearly away from EMA AND consistent directional moves
        distance_from_ema = abs(current_price - ema)
        atr = self.calculate_market_volatility(bars)
        
        # Trending criteria: price > 1 ATR from EMA and consistent moves
        if distance_from_ema > atr and avg_move > atr * 0.3:
            return 'trending'
        else:
            return 'choppy'
    
    def _calculate_ema(self, values: list, period: int) -> float:
        """Calculate EMA of values."""
        if not values:
            return 0.0
        
        multiplier = 2 / (period + 1)
        ema = values[0]
        
        for value in values[1:]:
            ema = (value - ema) * multiplier + ema
        
        return ema
    
    def get_adaptive_breakeven_params(
        self,
        bars: list,
        position: Dict,
        current_price: float
    ) -> Tuple[int, int]:
        """
        Calculate adaptive breakeven parameters.
        
        Args:
            bars: Recent OHLC bars
            position: Current position info
            current_price: Current market price
            
        Returns:
            (threshold_ticks, offset_ticks)
        """
        threshold = self.base_breakeven_threshold
        offset = self.base_breakeven_offset
        
        # Volatility-based scaling
        if self.volatility_scaling_enabled and len(bars) >= 14:
            atr = self.calculate_market_volatility(bars)
            atr_ticks = atr / self.tick_size
            
            # Scale threshold by volatility
            # High volatility: wider threshold (need more profit before breakeven)
            # Low volatility: tighter threshold (activate sooner)
            volatility_ratio = atr_ticks / 8.0  # 8 ticks = baseline
            threshold = int(self.base_breakeven_threshold * max(0.6, min(1.4, volatility_ratio)))
            
            logger.debug(f"Volatility-adjusted breakeven: ATR={atr:.2f} ({atr_ticks:.1f}t), "
                        f"threshold={threshold}t (base={self.base_breakeven_threshold}t)")
        
        # Regime-based adjustment
        if self.regime_adaptation_enabled and len(bars) >= 20:
            regime = self.detect_market_regime(bars)
            
            if regime == 'trending':
                # Trending: Be more patient, wider threshold
                threshold = int(threshold * 1.2)
                logger.debug(f"Trending market: Widened breakeven to {threshold}t")
            else:
                # Choppy: Lock in profits faster
                threshold = int(threshold * 0.8)
                logger.debug(f"Choppy market: Tightened breakeven to {threshold}t")
        
        # Performance-based: If trade is really winning, activate sooner
        if self.performance_adaptation_enabled and position["active"]:
            entry_price = position["entry_price"]
            side = position["side"]
            
            if side == "long":
                profit_ticks = (current_price - entry_price) / self.tick_size
            else:
                profit_ticks = (entry_price - current_price) / self.tick_size
            
            # If already in big profit (>15 ticks), activate breakeven ASAP
            if profit_ticks > 15:
                threshold = max(4, int(threshold * 0.6))
                logger.debug(f"Strong profit ({profit_ticks:.1f}t): Early breakeven at {threshold}t")
        
        # Volume-based adjustment
        volume_strength = self.calculate_volume_strength(bars)
        if volume_strength == 'low':
            # Low volume: tighten up, moves may be unreliable
            threshold = int(threshold * 0.8)
            logger.info(f" ADAPTIVE FACTOR: Low volume  tightened breakeven to {threshold}t")
        elif volume_strength == 'high':
            # High volume: give it more room, strong move
            threshold = int(threshold * 1.1)
            logger.info(f" ADAPTIVE FACTOR: High volume  widened breakeven to {threshold}t")
        else:
            logger.info(f" ADAPTIVE FACTOR: Normal volume  no adjustment")
        
        # Streak-based adjustment
        streak_length, streak_type = self.get_consecutive_pnl_streak()
        if streak_length >= 3:
            if streak_type == 'win':
                # Hot streak: protect profits faster (avoid giving back gains)
                threshold = int(threshold * 0.75)
                logger.info(f" ADAPTIVE FACTOR: Win streak ({streak_length})  tightened breakeven to {threshold}t")
            elif streak_type == 'loss':
                # Cold streak: might be choppy period, slightly wider
                threshold = int(threshold * 1.15)
                logger.info(f" ADAPTIVE FACTOR: Loss streak ({streak_length})  widened breakeven to {threshold}t")
        else:
            logger.info(f" ADAPTIVE FACTOR: No significant streak (length: {streak_length})  no adjustment")
        
        # Session context (influences exits only, not entries)
        session = self.get_session_context(datetime.now())
        if session == 'asian':
            # Asian session: lower volume, tighter exits
            threshold = int(threshold * 0.9)
            logger.info(f" ADAPTIVE FACTOR: Asian session  tightened breakeven to {threshold}t")
        elif session == 'us_open':
            # US open: high volatility, wider exits
            threshold = int(threshold * 1.1)
            logger.info(f" ADAPTIVE FACTOR: US open  widened breakeven to {threshold}t")
        elif session == 'us_midday':
            # Lunch: low volume, tighter
            threshold = int(threshold * 0.85)
            logger.info(f" ADAPTIVE FACTOR: US midday  tightened breakeven to {threshold}t")
        else:
            logger.info(f" ADAPTIVE FACTOR: Session {session}  no adjustment")
        
        # Ensure reasonable bounds
        threshold = max(3, min(15, threshold))
        
        # ========================================================================
        # COMMON SENSE OVERRIDES - Prevent conflicts & stupid decisions
        # ========================================================================
        
        # Rule 1: If we're already deep in profit, ALWAYS tighten regardless of other factors
        if self.performance_adaptation_enabled and position["active"]:
            entry_price = position["entry_price"]
            side = position["side"]
            
            if side == "long":
                profit_ticks = (current_price - entry_price) / self.tick_size
            else:
                profit_ticks = (entry_price - current_price) / self.tick_size
            
            # If profit > 30 ticks (substantial), override everything - lock it in!
            if profit_ticks > 30:
                threshold = min(threshold, 5)  # Force tight breakeven
                logger.info(f" COMMON SENSE: Deep profit ({profit_ticks:.1f}t) - forcing tight breakeven at {threshold}t")
        
        # Rule 2: Don't be stupid during low volume AND choppy - double danger!
        if len(bars) >= 20:
            regime = self.detect_market_regime(bars)
            volume = self.calculate_volume_strength(bars)
            
            if regime == 'choppy' and volume == 'low':
                # Very dangerous condition - be aggressive
                threshold = min(threshold, 5)
                logger.info(f" COMMON SENSE: Choppy + Low volume = DANGER - forcing tight breakeven at {threshold}t")
        
        # Rule 3: If losing streak + currently losing trade, don't make it worse
        streak_length, streak_type = self.get_consecutive_pnl_streak()
        if streak_length >= 4 and streak_type == 'loss':  # Need 4+ losses (more selective)
            if position["active"]:
                entry_price = position["entry_price"]
                side = position["side"]
                
                if side == "long":
                    current_profit = (current_price - entry_price) / self.tick_size
                else:
                    current_profit = (entry_price - current_price) / self.tick_size
                
                # If we're in a loss streak AND this trade is barely winning, protect it!
                if 0 < current_profit < 8:
                    threshold = min(threshold, 4)
                    logger.info(f" COMMON SENSE: Loss streak ({streak_length}) + small profit - protect at {threshold}t")
        
        # Rule 4: End of day + Friday = GET OUT with profits
        current_time = datetime.now()
        if current_time.hour >= 16 and current_time.weekday() == 4:  # 4pm+ on Friday (more selective)
            threshold = min(threshold, 5)
            logger.info(f" COMMON SENSE: Friday late afternoon - forcing tight breakeven at {threshold}t")
        
        return threshold, offset
    
    def get_adaptive_trailing_params(
        self,
        bars: list,
        position: Dict,
        current_price: float
    ) -> Tuple[int, int]:
        """
        Calculate adaptive trailing stop parameters.
        
        Args:
            bars: Recent OHLC bars
            position: Current position info
            current_price: Current market price
            
        Returns:
            (distance_ticks, min_profit_ticks)
        """
        distance = self.base_trailing_distance
        min_profit = self.base_trailing_min_profit
        
        # Volatility-based scaling
        if self.volatility_scaling_enabled and len(bars) >= 14:
            atr = self.calculate_market_volatility(bars)
            atr_ticks = atr / self.tick_size
            
            # Scale trailing distance by volatility
            # High volatility: trail wider (give it room)
            # Low volatility: trail tighter (protect gains)
            volatility_ratio = atr_ticks / 8.0
            distance = int(self.base_trailing_distance * max(0.5, min(2.0, volatility_ratio)))
            min_profit = int(self.base_trailing_min_profit * max(0.7, min(1.5, volatility_ratio)))
            
            logger.debug(f"Volatility-adjusted trailing: ATR={atr:.2f} ({atr_ticks:.1f}t), "
                        f"distance={distance}t, min_profit={min_profit}t")
        
        # Regime-based adjustment
        if self.regime_adaptation_enabled and len(bars) >= 20:
            regime = self.detect_market_regime(bars)
            
            if regime == 'trending':
                # Trending: Give it more room, trail wider
                distance = int(distance * 1.3)
                min_profit = int(min_profit * 1.2)
                logger.debug(f"Trending market: Wider trail distance={distance}t, min={min_profit}t")
            else:
                # Choppy: Protect gains, trail tighter
                distance = int(distance * 0.7)
                min_profit = int(min_profit * 0.8)
                logger.debug(f"Choppy market: Tighter trail distance={distance}t, min={min_profit}t")
        
        # Time-based: Trail tighter as we hold longer
        if position["active"] and position.get("entry_time"):
            entry_time = position["entry_time"]
            current_time = datetime.now(entry_time.tzinfo) if entry_time.tzinfo else datetime.now()
            hold_duration_minutes = (current_time - entry_time).total_seconds() / 60
            
            # After 2+ hours, tighten trailing by 20%
            if hold_duration_minutes > 120:
                distance = int(distance * 0.8)
                logger.debug(f"Long hold ({hold_duration_minutes:.0f}min): Tightened trail to {distance}t")
        
        # Volume-based adjustment
        volume_strength = self.calculate_volume_strength(bars)
        if volume_strength == 'low':
            # Low volume: trail tighter, lock in gains
            distance = int(distance * 0.8)
            min_profit = int(min_profit * 0.9)
            logger.debug(f"Low volume: Tighter trail distance={distance}t, min={min_profit}t")
        elif volume_strength == 'high':
            # High volume: give it more room
            distance = int(distance * 1.15)
            min_profit = int(min_profit * 1.1)
            logger.debug(f"High volume: Wider trail distance={distance}t, min={min_profit}t")
        
        # Streak-based adjustment
        streak_length, streak_type = self.get_consecutive_pnl_streak()
        if streak_length >= 3:
            if streak_type == 'win':
                # Hot streak: trail tighter, protect the gains
                distance = int(distance * 0.8)
                min_profit = int(min_profit * 0.85)
                logger.debug(f"Win streak ({streak_length}): Tighter trail distance={distance}t")
            elif streak_type == 'loss':
                # Cold streak: give more room, might reverse
                distance = int(distance * 1.1)
                min_profit = int(min_profit * 1.05)
                logger.debug(f"Loss streak ({streak_length}): Wider trail distance={distance}t")
        
        # Session context
        session = self.get_session_context(datetime.now())
        if session == 'asian':
            # Asian: lower volume, tighter trail
            distance = int(distance * 0.85)
            logger.debug(f"Asian session: Tighter trail distance={distance}t")
        elif session == 'us_open':
            # US open: wider trail for volatility
            distance = int(distance * 1.1)
            logger.debug(f"US open: Wider trail distance={distance}t")
        
        # Ensure reasonable bounds
        distance = max(4, min(20, distance))
        min_profit = max(6, min(25, min_profit))
        
        # ========================================================================
        # COMMON SENSE OVERRIDES - Smart trailing decisions
        # ========================================================================
        
        # Rule 1: If we're in big profit, don't give too much back
        if position["active"]:
            entry_price = position["entry_price"]
            side = position["side"]
            
            if side == "long":
                profit_ticks = (current_price - entry_price) / self.tick_size
            else:
                profit_ticks = (entry_price - current_price) / self.tick_size
            
            # If profit > 35 ticks (really big), trail TIGHT - we won big, don't blow it
            if profit_ticks > 35:
                distance = min(distance, 6)
                logger.info(f" COMMON SENSE: Big profit ({profit_ticks:.1f}t) - trailing tight at {distance}t")
            
            # If held 4+ hours (very long), probably at risk of reversal
            if position.get("entry_time"):
                entry_time = position["entry_time"]
                current_time = datetime.now(entry_time.tzinfo) if entry_time.tzinfo else datetime.now()
                hold_duration_hours = (current_time - entry_time).total_seconds() / 3600
                
                if hold_duration_hours >= 4:  # More selective - 4+ hours instead of 3
                    distance = min(distance, 7)
                    logger.info(f" COMMON SENSE: Held {hold_duration_hours:.1f}h - trailing tight at {distance}t")
        
        # Rule 2: Win streak + choppy market = lock in gains NOW
        if len(bars) >= 20:
            regime = self.detect_market_regime(bars)
            streak_length, streak_type = self.get_consecutive_pnl_streak()
            
            if streak_type == 'win' and streak_length >= 4 and regime == 'choppy':  # Need 4+ wins
                distance = min(distance, 5)
                min_profit = min(min_profit, 8)
                logger.info(f" COMMON SENSE: Hot streak ({streak_length}) in choppy market - protecting at {distance}t")
        
        # Rule 3: Massive volatility spike = danger, trail tight
        if len(bars) >= 14:
            current_atr = self.calculate_market_volatility(bars)
            atr_ticks = current_atr / self.tick_size
            
            # If ATR > 20 ticks (extreme volatility), be defensive
            if atr_ticks > 20:  # More selective - 20 instead of 15
                distance = min(distance, 8)
                logger.info(f" COMMON SENSE: Extreme volatility (ATR {atr_ticks:.1f}t) - tight trail at {distance}t")
        
        # Rule 4: End of day + any decent profit = lock it in
        current_time = datetime.now()
        if current_time.hour >= 16:  # After 4pm (more selective)
            if position["active"]:
                entry_price = position["entry_price"]
                side = position["side"]
                
                if side == "long":
                    profit_ticks = (current_price - entry_price) / self.tick_size
                else:
                    profit_ticks = (entry_price - current_price) / self.tick_size
                
                if profit_ticks > 8:  # Decent profit (raised from 5)
                    distance = min(distance, 5)
                    logger.info(f" COMMON SENSE: End of day with profit - tight trail at {distance}t")
        
        return distance, min_profit
    
    def get_adaptive_partial_exit_targets(
        self,
        bars: list,
        position: Dict
    ) -> list:
        """
        Calculate adaptive partial exit targets.
        
        Args:
            bars: Recent OHLC bars
            position: Current position info
            
        Returns:
            List of (percentage, r_multiple) tuples
        """
        # Base targets
        targets = [
            (0.50, 2.0),  # 50% at 2R
            (0.30, 3.0),  # 30% at 3R
        ]
        
        if len(bars) < 14:
            return targets
        
        # Volatility-based adjustment
        if self.volatility_scaling_enabled:
            atr = self.calculate_market_volatility(bars)
            atr_ticks = atr / self.tick_size
            
            # High volatility: targets further out
            # Low volatility: targets closer in
            volatility_ratio = atr_ticks / 8.0
            
            adjusted_targets = []
            for pct, r_mult in targets:
                adjusted_r = r_mult * max(0.7, min(1.5, volatility_ratio))
                adjusted_targets.append((pct, adjusted_r))
            
            logger.debug(f"Volatility-adjusted targets: {adjusted_targets}")
            return adjusted_targets
        
        # Regime-based adjustment
        if self.regime_adaptation_enabled and len(bars) >= 20:
            regime = self.detect_market_regime(bars)
            
            if regime == 'trending':
                # Trending: targets further, let it run
                targets = [
                    (0.40, 2.5),  # Take less, further out
                    (0.30, 4.0),
                ]
                logger.debug(f"Trending market: Extended targets {targets}")
            else:
                # Choppy: targets closer, take profits
                targets = [
                    (0.60, 1.5),  # Take more, sooner
                    (0.30, 2.5),
                ]
                logger.debug(f"Choppy market: Tightened targets {targets}")
        
        return targets
    
    def calculate_volume_strength(self, bars: list, lookback: int = 20) -> str:
        """
        Calculate relative volume strength.
        
        Args:
            bars: Recent OHLC bars
            lookback: Period for average volume
            
        Returns:
            'high', 'normal', or 'low'
        """
        # Convert deque to list if needed
        if not isinstance(bars, list):
            bars = list(bars)
        
        if len(bars) < lookback + 1:
            return 'normal'
        
        recent_bars = bars[-lookback:]
        
        # Get current bar volume (if available)
        current_volume = recent_bars[-1].get("volume", 0)
        if current_volume == 0:
            return 'normal'  # Volume data not available
        
        # Calculate average volume
        volumes = [bar.get("volume", 0) for bar in recent_bars[:-1]]
        avg_volume = np.mean([v for v in volumes if v > 0])
        
        if avg_volume == 0:
            return 'normal'
        
        # Compare current to average
        volume_ratio = current_volume / avg_volume
        
        if volume_ratio > 1.5:
            return 'high'  # 50%+ above average
        elif volume_ratio < 0.6:
            return 'low'   # 40%+ below average
        else:
            return 'normal'
    
    def get_consecutive_pnl_streak(self) -> Tuple[int, str]:
        """
        Get current win/loss streak.
        
        Returns:
            (streak_length, streak_type) where type is 'win', 'loss', or 'none'
        """
        if len(self.recent_trades) < 2:
            return 0, 'none'
        
        # Get last trade result
        last_pnl = self.recent_trades[-1]
        if last_pnl == 0:
            return 0, 'none'
        
        streak_type = 'win' if last_pnl > 0 else 'loss'
        streak_length = 1
        
        # Count backwards
        for i in range(len(self.recent_trades) - 2, -1, -1):
            pnl = self.recent_trades[i]
            if pnl == 0:
                break
            if (pnl > 0 and streak_type == 'win') or (pnl < 0 and streak_type == 'loss'):
                streak_length += 1
            else:
                break
        
        return streak_length, streak_type
    
    def get_session_context(self, current_time: datetime) -> str:
        """
        Determine current trading session.
        Note: This is for EXIT MANAGEMENT only, not entry filtering.
        
        Args:
            current_time: Current datetime
            
        Returns:
            'asian', 'european', 'us_open', 'us_midday', 'us_close'
        """
        hour = current_time.hour
        
        # EST/EDT times (assuming US timezone)
        if 0 <= hour < 3:
            return 'asian'       # Tokyo session
        elif 3 <= hour < 9:
            return 'european'    # London session
        elif 9 <= hour < 12:
            return 'us_open'     # US open (high volatility)
        elif 12 <= hour < 14:
            return 'us_midday'   # Lunch (lower volume)
        else:
            return 'us_close'    # Afternoon (EOD positioning)
    
    def record_trade_result(self, pnl: float):
        """
        Record a trade result for streak tracking.
        
        Args:
            pnl: Trade P&L in dollars
        """
        self.recent_trades.append(pnl)
        logger.debug(f"Trade recorded: ${pnl:.2f}, Recent trades: {len(self.recent_trades)}")
    
    def should_use_aggressive_exits(self, bars: list, time_of_day: datetime) -> bool:
        """
        Determine if aggressive exit management should be used.
        
        More aggressive during:
        - High volatility
        - Choppy markets
        - End of day
        - Fridays
        
        Args:
            bars: Recent OHLC bars
            time_of_day: Current time
            
        Returns:
            True if should be aggressive
        """
        reasons = []
        
        # High volatility check
        if len(bars) >= 14:
            atr = self.calculate_market_volatility(bars)
            atr_ticks = atr / self.tick_size
            
            if atr_ticks > 12:  # High volatility
                reasons.append("high_volatility")
        
        # Choppy market check
        if len(bars) >= 20:
            regime = self.detect_market_regime(bars)
            if regime == 'choppy':
                reasons.append("choppy_market")
        
        # Time of day check
        hour = time_of_day.hour
        if hour >= 15:  # After 3 PM
            reasons.append("end_of_day")
        
        # Friday check
        if time_of_day.weekday() == 4:  # Friday
            reasons.append("friday")
        
        is_aggressive = len(reasons) >= 2  # Need 2+ factors
        
        if is_aggressive:
            logger.info(f"Using AGGRESSIVE exits due to: {', '.join(reasons)}")
        
        return is_aggressive


def get_adaptive_exit_params(
    bars: list,
    position: Dict,
    current_price: float,
    config: Dict,
    adaptive_manager: Optional[AdaptiveExitManager] = None
) -> Dict:
    """
    Get all adaptive exit parameters in one call.
    
    Args:
        bars: Recent OHLC bars
        position: Current position info
        current_price: Current market price
        config: Bot configuration
        adaptive_manager: Optional pre-initialized manager
        
    Returns:
        Dictionary with all adaptive parameters
    """
    if adaptive_manager is None:
        adaptive_manager = AdaptiveExitManager(config)
    
    # Get adaptive parameters
    breakeven_threshold, breakeven_offset = adaptive_manager.get_adaptive_breakeven_params(
        bars, position, current_price
    )
    
    trailing_distance, trailing_min_profit = adaptive_manager.get_adaptive_trailing_params(
        bars, position, current_price
    )
    
    partial_targets = adaptive_manager.get_adaptive_partial_exit_targets(bars, position)
    
    # Determine if aggressive mode
    current_time = datetime.now()
    is_aggressive = adaptive_manager.should_use_aggressive_exits(bars, current_time)
    
    # ========================================================================
    # FINAL COMMON SENSE CHECK - Prevent conflicting signals
    # ========================================================================
    
    # Rule: Breakeven should NEVER be wider than trailing min profit
    # (That would be stupid - you'd trail before activating breakeven!)
    if breakeven_threshold > trailing_min_profit:
        logger.warning(f" CONFLICT DETECTED: Breakeven ({breakeven_threshold}t) > Trailing min ({trailing_min_profit}t)")
        breakeven_threshold = max(4, trailing_min_profit - 3)
        logger.info(f" FIXED: Adjusted breakeven to {breakeven_threshold}t")
    
    # Rule: Trailing distance should be reasonable relative to min profit
    # (Don't trail 20 ticks when min profit is 8 - you'd never trail!)
    if trailing_distance > trailing_min_profit * 2.0:  # More lenient - 2x instead of 1.5x
        logger.warning(f" CONFLICT DETECTED: Trail distance ({trailing_distance}t) too wide for min profit ({trailing_min_profit}t)")
        trailing_distance = int(trailing_min_profit * 1.5)
        logger.info(f" FIXED: Adjusted trail distance to {trailing_distance}t")
    
    # Rule: If in huge profit, ensure we CAN actually lock it in
    if position["active"]:
        entry_price = position["entry_price"]
        side = position["side"]
        
        if side == "long":
            profit_ticks = (current_price - entry_price) / config["tick_size"]
        else:
            profit_ticks = (entry_price - current_price) / config["tick_size"]
        
        # If profit > 40 ticks (huge!) but breakeven still not active, force it
        if profit_ticks > 40 and not position.get("breakeven_active"):  # More selective - 40 instead of 20
            if breakeven_threshold > 8:
                logger.warning(f" CONFLICT: Huge profit ({profit_ticks:.1f}t) but breakeven still at {breakeven_threshold}t!")
                breakeven_threshold = 6
                logger.info(f" FIXED: Forcing breakeven to {breakeven_threshold}t to lock in gains")
    
    return {
        "breakeven_threshold_ticks": breakeven_threshold,
        "breakeven_offset_ticks": breakeven_offset,
        "trailing_distance_ticks": trailing_distance,
        "trailing_min_profit_ticks": trailing_min_profit,
        "partial_exit_targets": partial_targets,
        "is_aggressive_mode": is_aggressive,
        "market_regime": adaptive_manager.detect_market_regime(bars) if len(bars) >= 20 else "unknown",
        "current_volatility_atr": adaptive_manager.calculate_market_volatility(bars) if len(bars) >= 14 else None
    }
