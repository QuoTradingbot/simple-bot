"""
Daily Reversal Detection System
========================================
Detects big moves and reversal exhaustion signals for daily trading.

THE EDGE:
Catch daily reversals when price makes a significant move and shows signs
of exhaustion. You wait for proof that the move is losing momentum, then 
enter in the opposite direction to ride the snapback.

LONG SIGNAL CONDITIONS (AFTER FLUSH DOWN) - ALL 9 MUST BE TRUE:
1. Flush Happened - Range of last 7 bars >= 8 ticks (2.0 points)
2. Flush Was Fast - Velocity >= 1.5 ticks per bar minimum speed
3. We Are Near The Bottom - Within 12 ticks (3 points) of flush low
4. RSI Is Oversold - RSI < 45
5. Volume Spiked - Current volume >= 1.2x 20-bar average
6. Flush Stopped Making New Lows - Current bar low >= previous bar low
7. Reversal Candle - Current bar closes green (close > open)
8. Price Is Below VWAP - Current close < VWAP
9. Regime Allows Trading - HIGH_VOL or NORMAL regimes (blocks LOW_VOL)

SHORT SIGNAL CONDITIONS (AFTER FLUSH UP) - ALL 9 MUST BE TRUE:
1. Pump Happened - Range of last 7 bars >= 8 ticks (2.0 points)
2. Pump Was Fast - Velocity >= 1.5 ticks per bar minimum speed
3. We Are Near The Top - Within 12 ticks (3 points) of flush high
4. RSI Is Overbought - RSI > 55
5. Volume Spiked - Current volume >= 1.2x 20-bar average
6. Pump Stopped Making New Highs - Current bar high <= previous bar high
7. Reversal Candle - Current bar closes red (close < open)
8. Price Is Above VWAP - Current close > VWAP
9. Regime Allows Trading - HIGH_VOL or NORMAL regimes (blocks LOW_VOL)

STOP LOSS:
- Long: 2 ticks below flush low
- Short: 2 ticks above flush high
- Emergency max: User's max loss setting (GUI configurable)
- Use tighter stop: whichever is tighter wins (flush-based or GUI max)

EXIT STRATEGY (TRAILING STOP MANAGES ALL EXITS):
- NO fixed VWAP target - trailing stop handles all profit-taking
- VWAP is SAFETY NET: Only used if price reaches VWAP before trailing activates
- Once trailing is active (15+ ticks), ignore VWAP and let it ride
- Trailing stop eventually exits you, either at VWAP or beyond

TRADE MANAGEMENT:
- Breakeven: Move stop to entry + 1 tick after 12 ticks profit
- Trailing: Trail 8 ticks behind peak after 15 ticks profit
- Time Stop (Optional): Exit after 20 bars if not at target/stop
"""

import logging
from typing import Dict, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FlushEvent:
    """Represents a detected flush (capitulation) event."""
    direction: str  # "DOWN" or "UP"
    flush_low: float  # Lowest point during flush
    flush_high: float  # Highest point during flush
    flush_size_ticks: float  # Total move in ticks
    flush_velocity: float  # Ticks per bar
    bar_count: int  # Number of bars in the flush
    

class CapitulationDetector:
    """
    Detects capitulation (panic) flushes and reversal signals.
    
    Uses exact 9-condition specification for entry signals.
    All 9 conditions must be TRUE to enter a trade.
    """
    
    # Configuration constants - AGGRESSIVE TO CATCH 5-6 MOVES PER DAY
    # These thresholds are tuned for higher frequency with quality filters
    MIN_FLUSH_TICKS = 8  # Minimum 8 ticks (2 dollars on ES) - catches more V-reversals
    MIN_VELOCITY_TICKS_PER_BAR = 1.5  # Flush must be at least 1.5 ticks per bar - more lenient
    FLUSH_LOOKBACK_BARS = 7  # Look at last 7 one-minute bars
    NEAR_EXTREME_TICKS = 12  # Must be within 12 ticks of flush extreme - more entry flexibility
    VOLUME_SPIKE_THRESHOLD = 1.2  # 1.2x 20-bar average volume - easier to trigger
    RSI_OVERSOLD_EXTREME = 45  # RSI < 45 for long entry - catches more reversals
    RSI_OVERBOUGHT_EXTREME = 55  # RSI > 55 for short entry - catches more reversals
    
    # Stop loss configuration
    STOP_BUFFER_TICKS = 2  # 2 ticks beyond flush extreme
    
    # Trade management - user can configure these
    BREAKEVEN_TRIGGER_TICKS = 12  # Move stop to entry after 12 ticks profit
    BREAKEVEN_OFFSET_TICKS = 1  # Entry + 1 tick buffer
    TRAILING_TRIGGER_TICKS = 15  # Start trailing after 15 ticks profit
    TRAILING_DISTANCE_TICKS = 8  # Trail 8 ticks behind peak
    MAX_HOLD_BARS = 20  # Time stop after 20 bars (optional)
    
    def __init__(self, tick_size: float = 0.25, tick_value: float = 12.50):
        """
        Initialize the capitulation detector.
        
        Args:
            tick_size: Price movement per tick (0.25 for ES)
            tick_value: Dollar value per tick ($12.50 for ES)
        """
        self.tick_size = tick_size
        self.tick_value = tick_value
        
        # State tracking
        self.last_flush: Optional[FlushEvent] = None
        self.bars_since_flush = 0
        
    def check_all_long_conditions(
        self,
        bars: deque,
        current_bar: Dict[str, Any],
        prev_bar: Dict[str, Any],
        rsi: Optional[float],
        avg_volume_20: float,
        current_price: float,
        vwap: float,
        regime: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check ALL 9 conditions for a LONG entry signal.
        
        Returns:
            Tuple of (all_conditions_met, condition_details)
        """
        conditions = {}
        
        # Get recent bars for flush analysis
        if len(bars) < self.FLUSH_LOOKBACK_BARS:
            return False, {"reason": f"Insufficient bars ({len(bars)}/{self.FLUSH_LOOKBACK_BARS})"}
        
        recent_bars = list(bars)[-self.FLUSH_LOOKBACK_BARS:]
        
        # Calculate flush metrics
        highest_high = max(bar["high"] for bar in recent_bars)
        lowest_low = min(bar["low"] for bar in recent_bars)
        flush_range = highest_high - lowest_low
        flush_range_ticks = flush_range / self.tick_size
        
        # CONDITION 1: Flush Happened (range >= 20 ticks)
        conditions["1_flush_happened"] = flush_range_ticks >= self.MIN_FLUSH_TICKS
        
        # CONDITION 2: Flush Was Fast (velocity >= 4 ticks per bar)
        bar_count = len(recent_bars)
        velocity = flush_range_ticks / bar_count if bar_count > 0 else 0
        conditions["2_flush_fast"] = velocity >= self.MIN_VELOCITY_TICKS_PER_BAR
        
        # CONDITION 3: We Are Near The Bottom (within 5 ticks of flush low)
        distance_from_low = (current_price - lowest_low) / self.tick_size
        conditions["3_near_bottom"] = distance_from_low <= self.NEAR_EXTREME_TICKS
        
        # CONDITION 4: RSI Is Extreme Oversold (RSI < 25)
        if rsi is not None:
            conditions["4_rsi_oversold"] = rsi < self.RSI_OVERSOLD_EXTREME
        else:
            conditions["4_rsi_oversold"] = False
        
        # CONDITION 5: Volume Spiked (current volume >= 1.2x 20-bar average)
        current_volume = current_bar.get("volume", 0)
        conditions["5_volume_spike"] = current_volume >= (avg_volume_20 * self.VOLUME_SPIKE_THRESHOLD)
        
        # CONDITION 6: Flush Stopped Making New Lows (current bar low >= prev bar low)
        conditions["6_stopped_new_lows"] = current_bar["low"] >= prev_bar["low"]
        
        # CONDITION 7: Reversal Candle (current bar closes green - close > open)
        conditions["7_reversal_candle"] = current_bar["close"] > current_bar["open"]
        
        # CONDITION 8: Price Is Below VWAP (buying at discount)
        conditions["8_below_vwap"] = current_price < vwap
        
        # CONDITION 9: Regime Allows Trading - RELAXED to allow more regimes
        # Now allows NORMAL regimes too, to catch daily reversals
        tradeable_regimes = {"HIGH_VOL_TRENDING", "HIGH_VOL_CHOPPY", "NORMAL_TRENDING", "NORMAL_CHOPPY", "NORMAL"}
        conditions["9_regime_allows"] = regime in tradeable_regimes
        
        # ALL 9 CONDITIONS MUST BE TRUE
        all_passed = all(conditions.values())
        
        # Build result details
        details = {
            "conditions": conditions,
            "flush_range_ticks": flush_range_ticks,
            "velocity": velocity,
            "distance_from_low_ticks": distance_from_low,
            "rsi": rsi,
            "volume_ratio": current_volume / avg_volume_20 if avg_volume_20 > 0 else 0,
            "flush_low": lowest_low,
            "flush_high": highest_high,
            "vwap": vwap,
            "regime": regime,
            "current_volume": current_volume,
            "avg_volume_20": avg_volume_20
        }
        
        if all_passed:
            # Store flush info for stop calculation
            self.last_flush = FlushEvent(
                direction="DOWN",
                flush_low=lowest_low,
                flush_high=highest_high,
                flush_size_ticks=flush_range_ticks,
                flush_velocity=velocity,
                bar_count=bar_count
            )
            details["stop_price"] = lowest_low - (self.STOP_BUFFER_TICKS * self.tick_size)
            details["target_price"] = vwap
            
            # Log the entry signal
            logger.info("=" * 60)
            logger.info("🚨 LONG ENTRY SIGNAL - ALL 9 CONDITIONS MET")
            logger.info("=" * 60)
            for key, value in conditions.items():
                status = "✅" if value else "❌"
                logger.info(f"  {status} {key}: {value}")
            logger.info(f"  Flush: {flush_range_ticks:.0f} ticks DOWN")
            logger.info(f"  Velocity: {velocity:.1f} ticks/bar")
            logger.info(f"  RSI: {rsi:.1f}" if rsi else "  RSI: N/A")
            logger.info(f"  Stop: ${details['stop_price']:.2f} | Target: ${vwap:.2f}")
            logger.info("=" * 60)
        else:
            # Find which conditions failed
            failed = [k for k, v in conditions.items() if not v]
            details["failed_conditions"] = failed
            details["reason"] = f"Failed conditions: {', '.join(failed)}"
            # Log when CLOSE to signal (8 or 9 conditions passed)
            passed_count = 9 - len(failed)
            if passed_count >= 8:
                logger.info(f"🎯 CLOSE TO LONG SIGNAL! {passed_count}/9 passed. Failed: {', '.join(failed)}")
                rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
                logger.info(f"   flush={flush_range_ticks:.1f}t, vel={velocity:.2f}, rsi={rsi_str}, vol={current_volume:.0f} (avg={avg_volume_20:.0f}, ratio={current_volume/avg_volume_20 if avg_volume_20 > 0 else 0:.2f}x)")
                logger.info(f"   dist_from_low={distance_from_low:.1f}t, reversal={current_bar['close']:.2f}>{current_bar['open']:.2f}, below_vwap={current_bar['close']:.2f}<{vwap:.2f}")
            
            # ENHANCED DIAGNOSTIC: Log ALL 9 conditions with actual values every 10 checks
            # This helps diagnose what's different between backtest and live
            import random
            # Check if shutdown is in progress (avoid logging during shutdown)
            try:
                from quotrading_engine import _shutdown_in_progress as shutdown_flag
                skip_diagnostic = shutdown_flag
            except (ImportError, AttributeError):
                skip_diagnostic = False
            
            diagnostic_sample = (random.random() < 0.1 or passed_count >= 7) and not skip_diagnostic  # 10% sample OR close to signal
            if diagnostic_sample:
                rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
                print(f"\n🔍 SIGNAL CHECK DIAGNOSTIC (Passed: {passed_count}/9)")
                print(f"   1. Flush Size: {flush_range_ticks:.1f}t (need >={self.MIN_FLUSH_TICKS}) {'✅' if conditions.get('1_flush_happened') else '❌'}")
                print(f"   2. Velocity: {velocity:.2f} t/bar (need >={self.MIN_VELOCITY_TICKS_PER_BAR}) {'✅' if conditions.get('2_fast_flush') else '❌'}")
                print(f"   3. Near Extreme: {distance_from_low:.1f}t from low (need <={self.NEAR_EXTREME_TICKS}) {'✅' if conditions.get('3_near_bottom') else '❌'}")
                print(f"   4. RSI: {rsi_str} (need <{self.RSI_OVERSOLD_EXTREME}) {'✅' if conditions.get('4_rsi_oversold') else '❌'}")
                print(f"   5. Volume Spike: {current_volume:.0f} / {avg_volume_20:.0f} = {current_volume/avg_volume_20 if avg_volume_20 > 0 else 0:.2f}x (need >={self.VOLUME_SPIKE_THRESHOLD}x) {'✅' if conditions.get('5_volume_spike') else '❌'}")
                print(f"   6. Stopped New Lows: cur_low={current_bar['low']:.2f} >= prev_low={prev_bar['low']:.2f} {'✅' if conditions.get('6_stopped_new_lows') else '❌'}")
                print(f"   7. Reversal Candle: close={current_bar['close']:.2f} > open={current_bar['open']:.2f} {'✅' if conditions.get('7_reversal_candle') else '❌'}")
                print(f"   8. Below VWAP: close={current_bar['close']:.2f} < vwap={vwap:.2f} {'✅' if conditions.get('8_below_vwap') else '❌'}")
                print(f"   9. Regime: {regime} (allowed: HIGH_VOL*, NORMAL*) {'✅' if conditions.get('9_regime_allows') else '❌'}")
                if passed_count >= 7:
                    print(f"   ⚠️  VERY CLOSE! Only {9-passed_count} condition(s) away from signal!")
            
            # DIAGNOSTIC: Log ALL near-misses (8 or 9 conditions) to help debug why 0 signals
            if passed_count >= 8:
                rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
                print(f"⚠️ Near-miss LONG: {passed_count}/9 passed. Failed: {', '.join(failed)}")
                print(f"   Bar: close={current_bar['close']:.2f}, open={current_bar['open']:.2f}, vol={current_volume:.0f}")
                print(f"   Flush: {flush_range_ticks:.1f}t (need {self.MIN_FLUSH_TICKS}+), vel={velocity:.2f} (need {self.MIN_VELOCITY_TICKS_PER_BAR}+)")
                print(f"   RSI: {rsi_str} (need <{self.RSI_OVERSOLD_EXTREME})")
                print(f"   Volume: {current_volume:.0f} vs avg={avg_volume_20:.0f} (need {current_volume}/{avg_volume_20:.0f} >= {self.VOLUME_SPIKE_THRESHOLD})")
                print(f"   Distance from low: {distance_from_low:.1f}t (need <={self.NEAR_EXTREME_TICKS})")
                print(f"   VWAP: close={current_bar['close']:.2f} vs {vwap:.2f} (need below)")
        
        return all_passed, details
    
    def check_all_short_conditions(
        self,
        bars: deque,
        current_bar: Dict[str, Any],
        prev_bar: Dict[str, Any],
        rsi: Optional[float],
        avg_volume_20: float,
        current_price: float,
        vwap: float,
        regime: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check ALL 9 conditions for a SHORT entry signal.
        
        Returns:
            Tuple of (all_conditions_met, condition_details)
        """
        conditions = {}
        
        # Get recent bars for flush analysis
        if len(bars) < self.FLUSH_LOOKBACK_BARS:
            return False, {"reason": f"Insufficient bars ({len(bars)}/{self.FLUSH_LOOKBACK_BARS})"}
        
        recent_bars = list(bars)[-self.FLUSH_LOOKBACK_BARS:]
        
        # Calculate flush metrics
        highest_high = max(bar["high"] for bar in recent_bars)
        lowest_low = min(bar["low"] for bar in recent_bars)
        flush_range = highest_high - lowest_low
        flush_range_ticks = flush_range / self.tick_size
        
        # CONDITION 1: Pump Happened (range >= 20 ticks)
        conditions["1_pump_happened"] = flush_range_ticks >= self.MIN_FLUSH_TICKS
        
        # CONDITION 2: Pump Was Fast (velocity >= 4 ticks per bar)
        bar_count = len(recent_bars)
        velocity = flush_range_ticks / bar_count if bar_count > 0 else 0
        conditions["2_pump_fast"] = velocity >= self.MIN_VELOCITY_TICKS_PER_BAR
        
        # CONDITION 3: We Are Near The Top (within 5 ticks of flush high)
        distance_from_high = (highest_high - current_price) / self.tick_size
        conditions["3_near_top"] = distance_from_high <= self.NEAR_EXTREME_TICKS
        
        # CONDITION 4: RSI Is Extreme Overbought (RSI > 75)
        if rsi is not None:
            conditions["4_rsi_overbought"] = rsi > self.RSI_OVERBOUGHT_EXTREME
        else:
            conditions["4_rsi_overbought"] = False
        
        # CONDITION 5: Volume Spiked (current volume >= 1.2x 20-bar average)
        current_volume = current_bar.get("volume", 0)
        conditions["5_volume_spike"] = current_volume >= (avg_volume_20 * self.VOLUME_SPIKE_THRESHOLD)
        
        # CONDITION 6: Pump Stopped Making New Highs (current bar high <= prev bar high)
        conditions["6_stopped_new_highs"] = current_bar["high"] <= prev_bar["high"]
        
        # CONDITION 7: Reversal Candle (current bar closes red - close < open)
        conditions["7_reversal_candle"] = current_bar["close"] < current_bar["open"]
        
        # CONDITION 8: Price Is Above VWAP (selling at premium)
        conditions["8_above_vwap"] = current_price > vwap
        
        # CONDITION 9: Regime Allows Trading - RELAXED to allow more regimes
        # Now allows NORMAL regimes too, to catch daily reversals
        tradeable_regimes = {"HIGH_VOL_TRENDING", "HIGH_VOL_CHOPPY", "NORMAL_TRENDING", "NORMAL_CHOPPY", "NORMAL"}
        conditions["9_regime_allows"] = regime in tradeable_regimes
        
        # ALL 9 CONDITIONS MUST BE TRUE
        all_passed = all(conditions.values())
        
        # Build result details
        details = {
            "conditions": conditions,
            "flush_range_ticks": flush_range_ticks,
            "velocity": velocity,
            "distance_from_high_ticks": distance_from_high,
            "rsi": rsi,
            "volume_ratio": current_volume / avg_volume_20 if avg_volume_20 > 0 else 0,
            "flush_low": lowest_low,
            "flush_high": highest_high,
            "vwap": vwap,
            "regime": regime
        }
        
        if all_passed:
            # Store flush info for stop calculation
            self.last_flush = FlushEvent(
                direction="UP",
                flush_low=lowest_low,
                flush_high=highest_high,
                flush_size_ticks=flush_range_ticks,
                flush_velocity=velocity,
                bar_count=bar_count
            )
            details["stop_price"] = highest_high + (self.STOP_BUFFER_TICKS * self.tick_size)
            details["target_price"] = vwap
            
            # Log the entry signal
            logger.info("=" * 60)
            logger.info("🚨 SHORT ENTRY SIGNAL - ALL 9 CONDITIONS MET")
            logger.info("=" * 60)
            for key, value in conditions.items():
                status = "✅" if value else "❌"
                logger.info(f"  {status} {key}: {value}")
            logger.info(f"  Pump: {flush_range_ticks:.0f} ticks UP")
            logger.info(f"  Velocity: {velocity:.1f} ticks/bar")
            logger.info(f"  RSI: {rsi:.1f}" if rsi else "  RSI: N/A")
            logger.info(f"  Stop: ${details['stop_price']:.2f} | Target: ${vwap:.2f}")
            logger.info("=" * 60)
        else:
            # Find which conditions failed
            failed = [k for k, v in conditions.items() if not v]
            details["failed_conditions"] = failed
            details["reason"] = f"Failed conditions: {', '.join(failed)}"
            # Log when CLOSE to signal (8 or 9 conditions passed)
            passed_count = 9 - len(failed)
            if passed_count >= 8:
                logger.info(f"🎯 CLOSE TO SHORT SIGNAL! {passed_count}/9 passed. Failed: {', '.join(failed)}")
                rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
                logger.info(f"   pump={flush_range_ticks:.1f}t, vel={velocity:.2f}, rsi={rsi_str}, vol={current_volume:.0f} (avg={avg_volume_20:.0f})")
            
            # ENHANCED DIAGNOSTIC: Log ALL 9 conditions with actual values every 10 checks
            import random
            # Check if shutdown is in progress (avoid logging during shutdown)
            try:
                from quotrading_engine import _shutdown_in_progress as shutdown_flag
                skip_diagnostic = shutdown_flag
            except (ImportError, AttributeError):
                skip_diagnostic = False
            
            diagnostic_sample = (random.random() < 0.1 or passed_count >= 7) and not skip_diagnostic  # 10% sample OR close to signal
            if diagnostic_sample:
                rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
                print(f"\n🔍 SHORT SIGNAL CHECK DIAGNOSTIC (Passed: {passed_count}/9)")
                print(f"   1. Pump Size: {flush_range_ticks:.1f}t (need >={self.MIN_FLUSH_TICKS}) {'✅' if conditions.get('1_pump_happened') else '❌'}")
                print(f"   2. Velocity: {velocity:.2f} t/bar (need >={self.MIN_VELOCITY_TICKS_PER_BAR}) {'✅' if conditions.get('2_fast_pump') else '❌'}")
                print(f"   3. Near Extreme: {distance_from_high:.1f}t from high (need <={self.NEAR_EXTREME_TICKS}) {'✅' if conditions.get('3_near_top') else '❌'}")
                print(f"   4. RSI: {rsi_str} (need >{self.RSI_OVERBOUGHT_EXTREME}) {'✅' if conditions.get('4_rsi_overbought') else '❌'}")
                print(f"   5. Volume Spike: {current_volume:.0f} / {avg_volume_20:.0f} = {current_volume/avg_volume_20 if avg_volume_20 > 0 else 0:.2f}x (need >={self.VOLUME_SPIKE_THRESHOLD}x) {'✅' if conditions.get('5_volume_spike') else '❌'}")
                print(f"   6. Stopped New Highs: cur_high={current_bar['high']:.2f} <= prev_high={prev_bar['high']:.2f} {'✅' if conditions.get('6_stopped_new_highs') else '❌'}")
                print(f"   7. Reversal Candle: close={current_bar['close']:.2f} < open={current_bar['open']:.2f} {'✅' if conditions.get('7_reversal_candle') else '❌'}")
                print(f"   8. Above VWAP: close={current_bar['close']:.2f} > vwap={vwap:.2f} {'✅' if conditions.get('8_above_vwap') else '❌'}")
                print(f"   9. Regime: {regime} (allowed: HIGH_VOL*, NORMAL*) {'✅' if conditions.get('9_regime_allows') else '❌'}")
                if passed_count >= 7:
                    print(f"   ⚠️  VERY CLOSE! Only {9-passed_count} condition(s) away from signal!")
        
        return all_passed, details
    
    def calculate_stop_price(self, side: str, flush_low: float, flush_high: float) -> float:
        """
        Calculate stop loss price based on flush extreme.
        
        Stop placement:
        - For LONG: 2 ticks below flush low
        - For SHORT: 2 ticks above flush high
        
        Args:
            side: "long" or "short"
            flush_low: Lowest price during flush
            flush_high: Highest price during flush
        
        Returns:
            Stop loss price
        """
        buffer = self.STOP_BUFFER_TICKS * self.tick_size
        
        if side == "long":
            return flush_low - buffer
        else:  # short
            return flush_high + buffer
    
    def should_activate_breakeven(self, current_price: float, entry_price: float,
                                  side: str) -> Tuple[bool, float]:
        """
        Check if breakeven should be activated (12 ticks profit).
        Move stop to entry + 1 tick.
        
        Args:
            current_price: Current market price
            entry_price: Entry price
            side: "long" or "short"
        
        Returns:
            Tuple of (should_activate, new_stop_price)
        """
        if side == "long":
            profit_ticks = (current_price - entry_price) / self.tick_size
            new_stop = entry_price + (self.BREAKEVEN_OFFSET_TICKS * self.tick_size)
        else:
            profit_ticks = (entry_price - current_price) / self.tick_size
            new_stop = entry_price - (self.BREAKEVEN_OFFSET_TICKS * self.tick_size)
        
        return profit_ticks >= self.BREAKEVEN_TRIGGER_TICKS, new_stop
    
    def should_activate_trailing(self, current_price: float, entry_price: float,
                                  side: str) -> bool:
        """
        Check if trailing stop should be activated (15+ ticks profit).
        
        Args:
            current_price: Current market price
            entry_price: Entry price
            side: "long" or "short"
        
        Returns:
            True if trailing should be activated
        """
        if side == "long":
            profit_ticks = (current_price - entry_price) / self.tick_size
        else:
            profit_ticks = (entry_price - current_price) / self.tick_size
        
        return profit_ticks >= self.TRAILING_TRIGGER_TICKS
    
    def calculate_trailing_stop(self, peak_price: float, side: str) -> float:
        """
        Calculate trailing stop price (8 ticks behind peak).
        
        Args:
            peak_price: Peak price reached (highest for long, lowest for short)
            side: "long" or "short"
        
        Returns:
            Trailing stop price
        """
        trail_distance = self.TRAILING_DISTANCE_TICKS * self.tick_size
        
        if side == "long":
            return peak_price - trail_distance
        else:
            return peak_price + trail_distance
    
    def check_time_stop(self, bars_held: int) -> bool:
        """
        Check if time stop should trigger (20 bars).
        This is optional - dead trades tie up capital.
        
        Args:
            bars_held: Number of bars since entry
        
        Returns:
            True if time stop triggered
        """
        return bars_held >= self.MAX_HOLD_BARS
    
    def reset(self):
        """Reset detector state for new session."""
        self.last_flush = None
        self.bars_since_flush = 0


# Singleton instance
_detector: Optional[CapitulationDetector] = None


def get_capitulation_detector(tick_size: float = 0.25, 
                              tick_value: float = 12.50) -> CapitulationDetector:
    """
    Get the global capitulation detector instance.
    
    Args:
        tick_size: Price movement per tick
        tick_value: Dollar value per tick
    
    Returns:
        CapitulationDetector instance
    """
    global _detector
    if _detector is None:
        _detector = CapitulationDetector(tick_size, tick_value)
    return _detector


def reset_capitulation_detector():
    """Reset the global capitulation detector."""
    global _detector
    if _detector is not None:
        _detector.reset()
