"""
Market Regime Detection System
================================
Detects and classifies market regimes based on volatility and price action.

Seven Regimes:
1. NORMAL - Baseline (ATR within 15% of average, choppy)
2. NORMAL_TRENDING - Normal volatility, trending
3. NORMAL_CHOPPY - Normal volatility, explicitly choppy
4. HIGH_VOL_CHOPPY - High volatility (ATR > 115% avg), choppy
5. HIGH_VOL_TRENDING - High volatility, trending
6. LOW_VOL_RANGING - Low volatility (ATR < 85% avg), ranging
7. LOW_VOL_TRENDING - Low volatility, trending

Each regime has specific parameters for stop loss, breakeven, and trailing stops.
"""

import logging
from typing import Dict, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)

# Constants for regime detection
MIN_BARS_FOR_REGIME_DETECTION = 114  # 100 for baseline + 14 for current ATR
BASELINE_BARS_START = 114  # Start of baseline window
BASELINE_BARS_END = 14     # End of baseline window (from end)
PRICE_ACTION_LOOKBACK = 20  # Bars to analyze for price action classification


class RegimeParameters:
    """Parameters for a specific market regime."""
    
    def __init__(self, name: str, stop_mult: float, breakeven_mult: float, 
                 trailing_mult: float, sideways_timeout: int, underwater_timeout: int):
        self.name = name
        self.stop_mult = stop_mult
        self.breakeven_mult = breakeven_mult
        self.trailing_mult = trailing_mult
        self.sideways_timeout = sideways_timeout
        self.underwater_timeout = underwater_timeout
    
    def __repr__(self):
        return f"RegimeParameters({self.name}, stop={self.stop_mult}x, be={self.breakeven_mult}x, trail={self.trailing_mult}x)"


# Seven regime definitions with exact parameters from problem statement
REGIME_DEFINITIONS = {
    "NORMAL": RegimeParameters(
        name="NORMAL",
        stop_mult=1.25,
        breakeven_mult=1.0,
        trailing_mult=1.0,
        sideways_timeout=10,
        underwater_timeout=7
    ),
    "NORMAL_TRENDING": RegimeParameters(
        name="NORMAL_TRENDING",
        stop_mult=1.62,
        breakeven_mult=1.0,
        trailing_mult=1.15,
        sideways_timeout=14,
        underwater_timeout=8
    ),
    "NORMAL_CHOPPY": RegimeParameters(
        name="NORMAL_CHOPPY",
        stop_mult=1.20,
        breakeven_mult=0.95,
        trailing_mult=0.95,
        sideways_timeout=8,
        underwater_timeout=6
    ),
    "HIGH_VOL_CHOPPY": RegimeParameters(
        name="HIGH_VOL_CHOPPY",
        stop_mult=1.50,
        breakeven_mult=0.75,
        trailing_mult=0.85,
        sideways_timeout=12,
        underwater_timeout=8
    ),
    "HIGH_VOL_TRENDING": RegimeParameters(
        name="HIGH_VOL_TRENDING",
        stop_mult=1.90,
        breakeven_mult=0.85,
        trailing_mult=1.25,
        sideways_timeout=18,
        underwater_timeout=10
    ),
    "LOW_VOL_RANGING": RegimeParameters(
        name="LOW_VOL_RANGING",
        stop_mult=1.10,
        breakeven_mult=1.0,
        trailing_mult=0.90,
        sideways_timeout=8,
        underwater_timeout=6
    ),
    "LOW_VOL_TRENDING": RegimeParameters(
        name="LOW_VOL_TRENDING",
        stop_mult=1.40,
        breakeven_mult=1.0,
        trailing_mult=1.10,
        sideways_timeout=15,
        underwater_timeout=8
    ),
}


class RegimeDetector:
    """
    Detects market regimes based on ATR and price action.
    
    Uses last 20 bars to determine:
    - Volatility level (high/normal/low) based on current ATR vs 20-bar average
    - Price action (trending/choppy/ranging) based on directional move vs price range
    """
    
    def __init__(self):
        self.atr_threshold = 0.15  # 15% threshold for volatility classification
        self.trend_threshold = 0.60  # 60% directional move for trending classification
    
    def detect_regime(self, bars: deque, current_atr: float, atr_period: int = 14) -> RegimeParameters:
        """
        Detect current market regime from recent bars.
        
        CRITICAL: Pass 15-minute bars here (not 1-minute bars) to reduce noise
        and get accurate regime classification. The current_atr should also be
        calculated from 15-minute bars using quotrading_engine.calculate_atr().
        
        Args:
            bars: Recent 15-minute price bars (OHLCV data)
            current_atr: Current ATR value from 15-minute bars (last 14 bars)
            atr_period: Period for ATR calculation (default 14)
        
        Returns:
            RegimeParameters for the detected regime
        """
        if len(bars) < MIN_BARS_FOR_REGIME_DETECTION:
            # Not enough data - need minimum bars for baseline + current
            logger.debug(f"Insufficient bars ({len(bars)}/{MIN_BARS_FOR_REGIME_DETECTION}) for regime detection, using NORMAL")
            return REGIME_DEFINITIONS["NORMAL"]
        
        # Get bars for analysis
        # - Baseline: bars 15-114 from end (100 bars) for average ATR
        # - Recent: last 20 bars for price action (trending/choppy/ranging)
        all_bars = list(bars)
        baseline_bars = all_bars[-MIN_BARS_FOR_REGIME_DETECTION:-BASELINE_BARS_END]  # Last 114 to 14 bars
        recent_bars = all_bars[-PRICE_ACTION_LOOKBACK:]  # Last 20 bars
        
        # Calculate baseline ATR from earlier period (NOT including current 14 bars)
        avg_atr = self._calculate_average_atr(baseline_bars, atr_period)
        
        if avg_atr == 0:
            logger.debug("Average ATR is 0, using NORMAL regime")
            return REGIME_DEFINITIONS["NORMAL"]
        
        # Classify volatility: high, normal, or low
        atr_ratio = current_atr / avg_atr
        
        if atr_ratio > (1.0 + self.atr_threshold):  # > 1.15
            volatility = "HIGH"
        elif atr_ratio < (1.0 - self.atr_threshold):  # < 0.85
            volatility = "LOW"
        else:  # Within 15% of average
            volatility = "NORMAL"
        
        # Classify price action: trending, choppy, or ranging
        price_action = self._classify_price_action(recent_bars)
        
        # Map to regime
        regime = self._map_to_regime(volatility, price_action)
        
        logger.debug(f"Regime detected: {regime.name} (ATR ratio: {atr_ratio:.2f}, "
                    f"volatility: {volatility}, action: {price_action})")
        
        return regime
    
    def _calculate_average_atr(self, bars: list, period: int = 14) -> float:
        """
        Calculate average ATR over the given bars.
        
        This is used internally by regime detection to calculate baseline ATR.
        In production, pass 15-minute bars here (from quotrading_engine.calculate_atr()).
        
        Args:
            bars: List of bars (must have 'high', 'low', 'close')
                  Should be 15-minute bars for production use (less noise)
            period: ATR period
        
        Returns:
            Average ATR value
        """
        if len(bars) < period + 1:
            return 0.0
        
        true_ranges = []
        
        for i in range(1, len(bars)):
            high = bars[i]["high"]
            low = bars[i]["low"]
            prev_close = bars[i-1]["close"]
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        if len(true_ranges) < period:
            return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0
        
        # Average of last 'period' true ranges
        return sum(true_ranges[-period:]) / period
    
    def _classify_price_action(self, bars: list) -> str:
        """
        Classify price action as trending, choppy, or ranging.
        
        Uses directional move as percentage of total price range:
        - Trending: Directional move > 60% of range
        - Choppy/Ranging: Directional move < 60% of range
        
        Args:
            bars: List of bars (must have 'high', 'low', 'open', 'close')
        
        Returns:
            "TRENDING", "CHOPPY", or "RANGING"
        """
        if not bars:
            return "CHOPPY"
        
        # Calculate price range (highest high - lowest low)
        highest = max(bar["high"] for bar in bars)
        lowest = min(bar["low"] for bar in bars)
        price_range = highest - lowest
        
        if price_range == 0:
            return "RANGING"
        
        # Calculate directional move (net change from first to last)
        first_close = bars[0]["close"]
        last_close = bars[-1]["close"]
        directional_move = abs(last_close - first_close)
        
        # Calculate percentage of range that's directional
        directional_pct = directional_move / price_range
        
        if directional_pct > self.trend_threshold:
            return "TRENDING"
        else:
            # For low directional move, distinguish between choppy and ranging
            # Ranging typically has tighter price action
            return "CHOPPY"
    
    def _map_to_regime(self, volatility: str, price_action: str) -> RegimeParameters:
        """
        Map volatility and price action to a specific regime.
        
        Args:
            volatility: "HIGH", "NORMAL", or "LOW"
            price_action: "TRENDING", "CHOPPY", or "RANGING"
        
        Returns:
            RegimeParameters for the mapped regime
        """
        if volatility == "HIGH":
            if price_action == "TRENDING":
                return REGIME_DEFINITIONS["HIGH_VOL_TRENDING"]
            else:  # CHOPPY or RANGING
                return REGIME_DEFINITIONS["HIGH_VOL_CHOPPY"]
        
        elif volatility == "LOW":
            if price_action == "TRENDING":
                return REGIME_DEFINITIONS["LOW_VOL_TRENDING"]
            else:  # CHOPPY or RANGING
                return REGIME_DEFINITIONS["LOW_VOL_RANGING"]
        
        else:  # NORMAL volatility
            if price_action == "TRENDING":
                return REGIME_DEFINITIONS["NORMAL_TRENDING"]
            elif price_action == "CHOPPY":
                return REGIME_DEFINITIONS["NORMAL_CHOPPY"]
            else:  # Default to baseline NORMAL
                return REGIME_DEFINITIONS["NORMAL"]
    
    def check_regime_change(self, entry_regime: str, current_regime: RegimeParameters) -> Tuple[bool, Optional[RegimeParameters]]:
        """
        Check if regime has changed.
        
        Args:
            entry_regime: Name of regime when position was entered
            current_regime: Currently detected regime parameters
        
        Returns:
            Tuple of (has_changed, new_regime)
        """
        if entry_regime == current_regime.name:
            return False, None
        
        # Regime has changed - use pure regime multipliers (no confidence scaling)
        logger.info(f"REGIME CHANGE: {entry_regime} → {current_regime.name}")
        logger.info(f"  Regime multipliers: stop={current_regime.stop_mult:.2f}x, "
                   f"trailing={current_regime.trailing_mult:.2f}x")
        
        return True, current_regime


# Singleton instance
_detector = None


def get_regime_detector() -> RegimeDetector:
    """Get the global regime detector instance."""
    global _detector
    if _detector is None:
        _detector = RegimeDetector()
    return _detector
