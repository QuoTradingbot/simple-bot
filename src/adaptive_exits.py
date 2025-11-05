"""
Adaptive Exit Management - Dynamic Exit Parameters with RL Learning
====================================================================
Adjusts breakeven, trailing stops, and exit timing based on:
- Current market volatility (ATR)
- Market regime (trending vs choppy)
- Trade performance and holding duration
- LEARNED optimal parameters from past outcomes

Provides smarter profit protection than static parameters.
"""

import logging
import json
import os
from typing import Dict, Any, Optional
from collections import deque
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)


class AdaptiveExitManager:
    """
    Manages adaptive exit parameters that adjust to market conditions.
    Maintains state across trades for regime detection.
    LEARNS optimal exit parameters from past outcomes.
    """
    
    def __init__(self, config: Dict, experience_file: str = "data/exit_experience.json"):
        """Initialize adaptive exit manager with RL learning."""
        self.config = config
        self.experience_file = experience_file
        
        # Track recent ATR values for regime detection
        self.recent_atr_values = deque(maxlen=20)
        self.recent_volatility_regime = "NORMAL"
        
        # Track recent trade durations for adaptive timing
        self.recent_trade_durations = deque(maxlen=10)
        
        # RL Learning for exit parameters
        self.exit_experiences = []  # All past exit outcomes
        
        # Learned optimal parameters per regime (updated from experiences)
        # MUST be defined BEFORE load_experiences() since it uses it as default
        self.learned_params = {
            'HIGH_VOL_CHOPPY': {'breakeven_mult': 0.75, 'trailing_mult': 0.7},
            'HIGH_VOL_TRENDING': {'breakeven_mult': 0.85, 'trailing_mult': 1.1},
            'LOW_VOL_RANGING': {'breakeven_mult': 1.0, 'trailing_mult': 1.0},
            'LOW_VOL_TRENDING': {'breakeven_mult': 1.0, 'trailing_mult': 1.15},
            'NORMAL': {'breakeven_mult': 1.0, 'trailing_mult': 1.0},
            'NORMAL_TRENDING': {'breakeven_mult': 1.0, 'trailing_mult': 1.1},
            'NORMAL_CHOPPY': {'breakeven_mult': 0.95, 'trailing_mult': 0.95}
        }
        
        # Load experiences and learned params from file (may override above defaults)
        self.load_experiences()
        
        logger.info(f"[ADAPTIVE] Exit Manager initialized with RL learning ({len(self.exit_experiences)} past exits)")
    
    def update_market_state(self, current_atr: float, bars: list):
        """
        Update internal market state tracking.
        
        Args:
            current_atr: Current ATR value
            bars: Recent price bars
        """
        self.recent_atr_values.append(current_atr)
        
        # Detect volatility regime
        if len(self.recent_atr_values) >= 5:
            avg_atr = statistics.mean(self.recent_atr_values)
            
            # High volatility: ATR > 1.2x average
            if current_atr > avg_atr * 1.2:
                self.recent_volatility_regime = "HIGH_VOL"
            # Low volatility: ATR < 0.8x average
            elif current_atr < avg_atr * 0.8:
                self.recent_volatility_regime = "LOW_VOL"
            else:
                self.recent_volatility_regime = "NORMAL"
    
    def record_trade_duration(self, duration_minutes: int):
        """Record how long a trade lasted for adaptive timing."""
        self.recent_trade_durations.append(duration_minutes)
    
    def record_exit_outcome(self, regime: str, exit_params: Dict, trade_outcome: Dict):
        """
        Record exit outcome for RL learning.
        
        Args:
            regime: Market regime when exit occurred
            exit_params: Exit parameters used (breakeven_threshold, trailing_distance, etc.)
            trade_outcome: Trade result (pnl, duration, exit_reason, win/loss)
        """
        experience = {
            'timestamp': datetime.now().isoformat(),
            'regime': regime,
            'exit_params': exit_params,
            'outcome': trade_outcome,
            'situation': {
                'time_of_day': datetime.now().strftime('%H:%M'),
                'volatility_atr': exit_params.get('current_atr', 0),
                'trend_strength': trade_outcome.get('trend_strength', 0)
            }
        }
        
        self.exit_experiences.append(experience)
        
        # Save every 3 exits
        if len(self.exit_experiences) % 3 == 0:
            self.save_experiences()
            # Re-learn optimal parameters
            self.update_learned_parameters()
        
        logger.info(f"[EXIT RL] LEARNED: {regime} | {exit_params['breakeven_threshold_ticks']}t BE, "
                   f"{exit_params['trailing_distance_ticks']}t Trail | "
                   f"P&L: ${trade_outcome['pnl']:.2f} | {trade_outcome['exit_reason']}")
    
    def update_learned_parameters(self):
        """
        Update learned optimal parameters based on past exit outcomes.
        Analyzes which parameter combos → best results per regime.
        """
        if len(self.exit_experiences) < 10:
            return  # Need minimum data
        
        # Group by regime
        regime_outcomes = {}
        for exp in self.exit_experiences:
            regime = exp['regime']
            if regime not in regime_outcomes:
                regime_outcomes[regime] = []
            regime_outcomes[regime].append(exp)
        
        # Analyze each regime
        for regime, outcomes in regime_outcomes.items():
            if len(outcomes) < 5:
                continue  # Need minimum per regime
            
            # Initialize regime if not in learned_params
            if regime not in self.learned_params:
                self.learned_params[regime] = {'breakeven_mult': 1.0, 'trailing_mult': 1.0}
                logger.info(f"[EXIT RL] New regime discovered: {regime}, initializing with 1.0x multipliers")
            
            # Calculate average P&L for different parameter ranges
            tight_exits = [o for o in outcomes if o['exit_params']['breakeven_threshold_ticks'] <= 6]
            standard_exits = [o for o in outcomes if 6 < o['exit_params']['breakeven_threshold_ticks'] <= 8]
            loose_exits = [o for o in outcomes if o['exit_params']['breakeven_threshold_ticks'] > 8]
            
            tight_pnl = sum(o['outcome']['pnl'] for o in tight_exits) / max(1, len(tight_exits))
            standard_pnl = sum(o['outcome']['pnl'] for o in standard_exits) / max(1, len(standard_exits))
            loose_pnl = sum(o['outcome']['pnl'] for o in loose_exits) / max(1, len(loose_exits))
            
            # Adjust multipliers based on what worked best
            if tight_pnl > standard_pnl and tight_pnl > loose_pnl:
                # Tight exits work best for this regime
                self.learned_params[regime]['breakeven_mult'] *= 0.95  # Tighten more
                logger.info(f"[EXIT RL] LEARNED: {regime} prefers TIGHT exits (avg P&L: ${tight_pnl:.2f})")
            elif loose_pnl > standard_pnl and loose_pnl > tight_pnl:
                # Loose exits work best
                self.learned_params[regime]['breakeven_mult'] *= 1.05  # Loosen more
                logger.info(f"[EXIT RL] LEARNED: {regime} prefers LOOSE exits (avg P&L: ${loose_pnl:.2f})")
            
            # Clamp to reasonable ranges
            self.learned_params[regime]['breakeven_mult'] = max(0.6, min(1.3, self.learned_params[regime]['breakeven_mult']))
            self.learned_params[regime]['trailing_mult'] = max(0.6, min(1.3, self.learned_params[regime]['trailing_mult']))
    
    def load_experiences(self):
        """Load past exit experiences from file."""
        logger.info(f"[DEBUG] Attempting to load exit experiences from: {self.experience_file}")
        logger.info(f"[DEBUG] File exists check: {os.path.exists(self.experience_file)}")
        
        if os.path.exists(self.experience_file):
            try:
                logger.info(f"[DEBUG] Opening exit file...")
                with open(self.experience_file, 'r') as f:
                    logger.info(f"[DEBUG] Loading exit JSON...")
                    data = json.load(f)
                    logger.info(f"[DEBUG] Exit JSON loaded successfully. Keys: {list(data.keys())}")
                    self.exit_experiences = data.get('exit_experiences', [])
                    self.learned_params = data.get('learned_params', self.learned_params)
                    logger.info(f"Loaded {len(self.exit_experiences)} past exit experiences")
            except Exception as e:
                logger.error(f"Failed to load exit experiences: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning(f"[DEBUG] Exit experience file not found: {self.experience_file}")
    
    def save_experiences(self):
        """Save exit experiences to file."""
        try:
            with open(self.experience_file, 'w') as f:
                json.dump({
                    'exit_experiences': self.exit_experiences,
                    'learned_params': self.learned_params,
                    'total_exits': len(self.exit_experiences)
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save exit experiences: {e}")


def detect_market_regime(bars: list, current_atr: float) -> str:
    """
    Detect current market regime (trending vs choppy).
    
    Args:
        bars: Recent 1-min bars (list of dicts, at least 20)
        current_atr: Current ATR value
    
    Returns:
        Regime string: HIGH_VOL_TRENDING, HIGH_VOL_CHOPPY, LOW_VOL_TRENDING, 
                      LOW_VOL_RANGING, or NORMAL
    """
    if len(bars) < 20:
        return "NORMAL"
    
    # Bars are guaranteed to be list of dicts (converted in get_adaptive_exit_params)
    recent_bars = bars[-20:]
    highs = [b["high"] for b in recent_bars]
    lows = [b["low"] for b in recent_bars]
    closes = [b["close"] for b in recent_bars]
    
    price_range = max(highs) - min(lows)
    avg_close = statistics.mean(closes)
    
    # Detect trend strength
    first_close = closes[0]
    last_close = closes[-1]
    directional_move = abs(last_close - first_close)
    
    # Trending if directional move > 60% of total range
    is_trending = directional_move > (price_range * 0.6)
    
    # Volatility classification
    # Calculate average ATR from recent bars if available
    avg_atr = current_atr  # Simplified - could calculate from bars
    
    if current_atr > avg_atr * 1.15:
        vol_regime = "HIGH_VOL"
    elif current_atr < avg_atr * 0.85:
        vol_regime = "LOW_VOL"
    else:
        vol_regime = "NORMAL"
    
    # Combine volatility + trend
    if vol_regime == "HIGH_VOL":
        return "HIGH_VOL_TRENDING" if is_trending else "HIGH_VOL_CHOPPY"
    elif vol_regime == "LOW_VOL":
        return "LOW_VOL_TRENDING" if is_trending else "LOW_VOL_RANGING"
    else:
        return "NORMAL_TRENDING" if is_trending else "NORMAL"


def get_adaptive_exit_params(bars: list, position: Dict, current_price: float, 
                             config: Dict, adaptive_manager: Optional[AdaptiveExitManager] = None) -> Dict:
    """
    Calculate adaptive exit parameters based on current market conditions.
    
    Args:
        bars: Recent 1-min bars (can be DataFrame or list)
        position: Current position state
        current_price: Current market price
        config: Bot configuration
        adaptive_manager: Optional manager instance for state persistence
    
    Returns:
        Dict with adaptive parameters:
        - breakeven_threshold_ticks: When to move to breakeven
        - breakeven_offset_ticks: Where to place breakeven stop
        - trailing_distance_ticks: Trailing stop distance
        - trailing_min_profit_ticks: Min profit before trailing activates
        - market_regime: Detected regime
        - current_volatility_atr: Current ATR
        - is_aggressive_mode: Whether in aggressive profit-taking mode
    """
    # Convert DataFrame or deque to list of dicts (prevent slicing errors)
    if hasattr(bars, 'iloc'):
        # Pandas DataFrame - convert to list of dicts for consistent handling
        bars = bars.to_dict('records')
    elif hasattr(bars, 'popleft'):
        # collections.deque - convert to list for slicing support
        bars = list(bars)
    
    # Base parameters from config
    base_breakeven_threshold = config.get("breakeven_profit_threshold_ticks", 8)
    base_breakeven_offset = config.get("breakeven_stop_offset_ticks", 1)
    base_trailing_distance = config.get("trailing_stop_distance_ticks", 8)
    base_trailing_min_profit = config.get("trailing_stop_min_profit_ticks", 12)
    
    # Calculate current ATR (simplified)
    # NOW bars is guaranteed to be a list of dicts
    if len(bars) > 0 and "atr" in bars[-1]:
        current_atr = bars[-1]["atr"]
    elif len(bars) >= 14:
        recent_ranges = [(b["high"] - b["low"]) for b in bars[-14:]]
        current_atr = statistics.mean(recent_ranges)
        
        # CRITICAL FIX: Many bars have only 1 tick, giving high==low (range=0)
        # Use a sensible minimum ATR based on typical ES/MES volatility
        # ES typically moves 1-3 points per minute, MES is 1/10th of ES
        if current_atr < 1.0:  # Less than 1 point ATR is unrealistic
            tick_size = config.get("tick_size", 0.25)
            min_atr_ticks = 8  # Minimum 8 ticks (~2 points for ES, ~0.5 for MES)
            current_atr = max(current_atr, min_atr_ticks * tick_size)
            logger.debug(f"[ADAPTIVE] ATR too low, using minimum: {current_atr:.2f}")
    else:
        current_atr = 5.0  # Default fallback
        logger.info(f"[ADAPTIVE] Using default ATR=5.0 (only {len(bars)} bars available)")
    
    # Update manager state if provided
    if adaptive_manager:
        adaptive_manager.update_market_state(current_atr, bars)
    
    # Detect market regime
    market_regime = detect_market_regime(bars, current_atr)
    
    # Calculate trade duration
    entry_time = position.get("entry_time")
    if entry_time and hasattr(entry_time, 'timestamp'):
        from datetime import datetime
        # FIX: Use bar timestamp instead of wall-clock time for backtest accuracy
        if len(bars) > 0 and "timestamp" in bars[-1]:
            current_time = bars[-1]["timestamp"]
            if isinstance(current_time, str):
                current_time = datetime.fromisoformat(current_time.replace('Z', '+00:00'))
        else:
            current_time = datetime.now()  # Fallback for live trading
        
        duration_minutes = (current_time.timestamp() - entry_time.timestamp()) / 60
        current_hour = current_time.hour
    else:
        duration_minutes = 0
        current_hour = 12  # Assume midday if unknown
    
    # ========================================================================
    # SITUATION AWARENESS - What's happening RIGHT NOW?
    # ========================================================================
    situation_factors = {
        'is_choppy': "CHOPPY" in market_regime,
        'is_high_vol': "HIGH_VOL" in market_regime,
        'is_trending': "TRENDING" in market_regime,
        'is_morning': 9 <= current_hour < 11,  # 9-11 AM volatile
        'is_lunch': 11 <= current_hour < 14,   # 11 AM - 2 PM slow
        'is_close': current_hour >= 15,        # After 3 PM rushes/reversals
        'is_old_trade': duration_minutes > 30,  # Been in too long
        'is_quick_trade': duration_minutes < 5, # Very recent entry
    }
    
    # Determine if we should be aggressive (take profits quicker)
    is_aggressive_mode = False
    aggression_reasons = []
    
    # Be aggressive in choppy/high volatility - protect gains!
    if situation_factors['is_choppy']:
        is_aggressive_mode = True
        aggression_reasons.append("CHOPPY")
    
    if situation_factors['is_high_vol']:
        is_aggressive_mode = True
        aggression_reasons.append("HIGH_VOL")
    
    # Be aggressive near market close - lock in profits before EOD
    if situation_factors['is_close']:
        is_aggressive_mode = True
        aggression_reasons.append("NEAR_CLOSE")
    
    # Be aggressive if trade has been open a while (>30 min) - don't let it reverse
    if situation_factors['is_old_trade']:
        is_aggressive_mode = True
        aggression_reasons.append("OLD_TRADE")
    
    # Be PATIENT in strong trends - let winners run!
    is_patient_mode = False
    if situation_factors['is_trending'] and not situation_factors['is_choppy']:
        is_patient_mode = True
    
    # Log situation awareness
    if aggression_reasons:
        logger.info(f"[SITUATION] AGGRESSIVE mode: {', '.join(aggression_reasons)}")
    if is_patient_mode:
        logger.info(f"[SITUATION] PATIENT mode: Strong trend detected, giving room")
    
    # ========================================================================
    # ADAPTIVE ADJUSTMENTS - Learn from past + React to current situation
    # ========================================================================
    
    # Default multipliers
    breakeven_threshold_multiplier = 1.0
    trailing_distance_multiplier = 1.0
    trailing_min_profit_multiplier = 1.0
    
    # Get LEARNED parameters from manager (if available)
    if adaptive_manager and hasattr(adaptive_manager, 'learned_params'):
        learned = adaptive_manager.learned_params.get(market_regime, {})
        base_breakeven_mult = learned.get('breakeven_mult', 1.0)
        base_trailing_mult = learned.get('trailing_mult', 1.0)
        logger.debug(f"[LEARNING] Using learned params for {market_regime}: BE={base_breakeven_mult:.2f}x, Trail={base_trailing_mult:.2f}x")
    else:
        # Fallback to hardcoded if no learning available
        base_breakeven_mult = 1.0
        base_trailing_mult = 1.0
    
    # Adjust based on regime (using LEARNED multipliers as base)
    if market_regime == "HIGH_VOL_CHOPPY":
        # Tighten exits in choppy conditions - protect profits!
        breakeven_threshold_multiplier = 0.70 * base_breakeven_mult  # Move to BE fast
        trailing_distance_multiplier = 0.65 * base_trailing_mult     # Tight trailing
        trailing_min_profit_multiplier = 0.75  # Trail early
        logger.debug(f"[REGIME] HIGH_VOL_CHOPPY: Tight exits (protect gains)")
        
    elif market_regime == "HIGH_VOL_TRENDING":
        # Give trends room to breathe but protect against reversals
        breakeven_threshold_multiplier = 0.80 * base_breakeven_mult
        trailing_distance_multiplier = 1.2 * base_trailing_mult   # Wider trailing in trends
        trailing_min_profit_multiplier = 0.85
        logger.debug(f"[REGIME] HIGH_VOL_TRENDING: Room to run, protected")
        
    elif market_regime == "LOW_VOL_RANGING":
        # Standard exits work well in calm ranges
        breakeven_threshold_multiplier = 1.0 * base_breakeven_mult
        trailing_distance_multiplier = 0.9 * base_trailing_mult   # Slightly tighter
        trailing_min_profit_multiplier = 1.0
        logger.debug(f"[REGIME] LOW_VOL_RANGING: Standard exits")
        
    elif market_regime == "LOW_VOL_TRENDING":
        # Let winners run in calm trends - this is ideal!
        breakeven_threshold_multiplier = 1.0 * base_breakeven_mult
        trailing_distance_multiplier = 1.3 * base_trailing_mult   # Lots of room
        trailing_min_profit_multiplier = 0.95
        logger.debug(f"[REGIME] LOW_VOL_TRENDING: Maximum room for winners!")
    else:
        # NORMAL or other regimes
        breakeven_threshold_multiplier = 1.0 * base_breakeven_mult
        trailing_distance_multiplier = 1.0 * base_trailing_mult
        trailing_min_profit_multiplier = 1.0
    
    # Aggressive mode: tighten everything (override regime settings)
    if is_aggressive_mode:
        breakeven_threshold_multiplier *= 0.80  # Move to breakeven FAST
        trailing_distance_multiplier *= 0.75    # TIGHT trailing
        trailing_min_profit_multiplier *= 0.80  # Trail EARLY
        logger.debug(f"[AGGRESSIVE] Tightening all exits by 20-25%")
    
    # Patient mode: give more room (but don't override if already aggressive)
    elif is_patient_mode:
        trailing_distance_multiplier *= 1.2    # Extra room in trends
        logger.debug(f"[PATIENT] Giving 20% more room for trend to develop")
    
    # Calculate final parameters (rounded to whole ticks)
    adaptive_breakeven_threshold = max(4, round(base_breakeven_threshold * breakeven_threshold_multiplier))
    adaptive_breakeven_offset = base_breakeven_offset  # Keep at 1 tick
    adaptive_trailing_distance = max(4, round(base_trailing_distance * trailing_distance_multiplier))
    adaptive_trailing_min_profit = max(6, round(base_trailing_min_profit * trailing_min_profit_multiplier))
    
    # ========================================================================
    # FINAL DECISION LOGGING - Show what bot decided and WHY
    # ========================================================================
    decision_summary = "[SMART EXIT DECISION]"
    logger.info(f"\n{'='*70}")
    logger.info(f"{decision_summary}")
    logger.info(f"{'='*70}")
    logger.info(f"[SITUATION] {market_regime} | ATR: {current_atr:.2f} | Duration: {duration_minutes:.0f}m")
    
    if aggression_reasons:
        logger.info(f"[MODE] AGGRESSIVE ({', '.join(aggression_reasons)})")
    elif is_patient_mode:
        logger.info(f"[MODE] PATIENT (letting trend develop)")
    else:
        logger.info(f"[MODE] BALANCED")
    
    # Show parameter changes
    be_change = ((adaptive_breakeven_threshold - base_breakeven_threshold) / base_breakeven_threshold * 100)
    trail_change = ((adaptive_trailing_distance - base_trailing_distance) / base_trailing_distance * 100)
    
    logger.info(f"[BREAKEVEN] {adaptive_breakeven_threshold}t ({be_change:+.0f}% vs base {base_breakeven_threshold}t)")
    logger.info(f"[TRAILING] {adaptive_trailing_distance}t @ {adaptive_trailing_min_profit}t ({trail_change:+.0f}% vs base {base_trailing_distance}t)")
    
    if adaptive_manager and base_breakeven_mult != 1.0:
        logger.info(f"[LEARNED] BE mult={base_breakeven_mult:.2f}x, Trail mult={base_trailing_mult:.2f}x")
    
    logger.info(f"{'='*70}\n")
    
    return {
        "breakeven_threshold_ticks": adaptive_breakeven_threshold,
        "breakeven_offset_ticks": adaptive_breakeven_offset,
        "trailing_distance_ticks": adaptive_trailing_distance,
        "trailing_min_profit_ticks": adaptive_trailing_min_profit,
        "market_regime": market_regime,
        "current_volatility_atr": current_atr,
        "is_aggressive_mode": is_aggressive_mode,
        "situation_factors": situation_factors,
        "decision_reasons": aggression_reasons if aggression_reasons else ["balanced"],
        "duration_minutes": duration_minutes,
        "learned_multiplier": base_breakeven_mult  # Include learned multiplier for tracking
    }
