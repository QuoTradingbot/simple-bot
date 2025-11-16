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
import random
from typing import Dict, Any, Optional
from collections import deque
from datetime import datetime
import statistics
import requests
import time
from exit_param_utils import extract_all_exit_params, extract_execution_data

logger = logging.getLogger(__name__)


class AdaptiveExitManager:
    """
    Manages adaptive exit parameters that adjust to market conditions.
    Maintains state across trades for regime detection.
    LEARNS optimal exit parameters from past outcomes.
    """
    
    def __init__(self, config: Dict, experience_file: str = "data/local_experiences/exit_experiences_v2.json", cloud_api_url: Optional[str] = None):
        """
        Initialize adaptive exit manager with RL learning and neural network support.
        
        Args:
            config: Bot configuration
            experience_file: Local file path for fallback (if cloud unavailable)
            cloud_api_url: Cloud API URL for fetching/saving experiences (e.g., "https://quotrading-signals.icymeadow-86b2969e.eastus.azurecontainerapps.io")
        """
        self.config = config
        self.experience_file = experience_file
        self.cloud_api_url = cloud_api_url  # NEW: Cloud API endpoint
        self.use_cloud = cloud_api_url is not None  # Use cloud if URL provided
        
        # REMOVED: Neural network is now cloud-only (protected model)
        # Users no longer have direct access to exit_model.pth
        
        # CACHE: Store cloud exit params to avoid rate limiting
        self.cloud_exit_params_cache = {}  # {regime: {params, timestamp}}
        self.cloud_cache_duration = 60  # Cache for 60 seconds (1 minute)
        
        # Track recent ATR values for regime detection
        self.recent_atr_values = deque(maxlen=20)
        self.recent_volatility_regime = "NORMAL"
        
        # Track recent trade durations for adaptive timing
        self.recent_trade_durations = deque(maxlen=10)
        
        # RL Learning for exit parameters
        self.exit_experiences = []  # All past exit outcomes
        
        # MAE/MFE tracking for learning optimal exit timing
        self.active_trades_mae_mfe = {}  # Trade ID -> {mae, mfe, entry_price}
        
        # Learned optimal parameters per regime (updated from experiences)
        # MUST be defined BEFORE load_experiences() since it uses it as default
        # NOW LEARNS: stops, breakeven, trailing, partial exits, sideways timeout
        self.learned_params = {
            'HIGH_VOL_CHOPPY': {
                'breakeven_mult': 0.75, 'trailing_mult': 0.7, 'stop_mult': 4.0,
                'partial_1_r': 2.0, 'partial_1_pct': 0.50,  # 50% @ 2R
                'partial_2_r': 3.0, 'partial_2_pct': 0.30,  # 30% @ 3R
                'partial_3_r': 5.0, 'partial_3_pct': 0.20,  # 20% @ 5R (runner)
                'sideways_timeout_minutes': 15,  # Exit runner if sideways 15 min
                'runner_hold_criteria': {'min_r_multiple': 6.0, 'min_duration_minutes': 30, 'max_drawdown_pct': 0.25}
            },
            'HIGH_VOL_TRENDING': {
                'breakeven_mult': 0.85, 'trailing_mult': 1.1, 'stop_mult': 4.2,
                'partial_1_r': 2.5, 'partial_1_pct': 0.40,  # Let trends run more
                'partial_2_r': 4.0, 'partial_2_pct': 0.30,
                'partial_3_r': 6.0, 'partial_3_pct': 0.30,
                'sideways_timeout_minutes': 20,
                'runner_hold_criteria': {'min_r_multiple': 8.0, 'min_duration_minutes': 40, 'max_drawdown_pct': 0.20}
            },
            'LOW_VOL_RANGING': {
                'breakeven_mult': 1.0, 'trailing_mult': 1.0, 'stop_mult': 3.2,
                'partial_1_r': 1.5, 'partial_1_pct': 0.60,  # Take profits quick in ranges
                'partial_2_r': 2.5, 'partial_2_pct': 0.30,
                'partial_3_r': 4.0, 'partial_3_pct': 0.10,
                'sideways_timeout_minutes': 10,
                'runner_hold_criteria': {'min_r_multiple': 4.0, 'min_duration_minutes': 20, 'max_drawdown_pct': 0.30}
            },
            'LOW_VOL_TRENDING': {
                'breakeven_mult': 1.0, 'trailing_mult': 1.15, 'stop_mult': 3.4,
                'partial_1_r': 2.0, 'partial_1_pct': 0.40,
                'partial_2_r': 3.5, 'partial_2_pct': 0.30,
                'partial_3_r': 5.5, 'partial_3_pct': 0.30,
                'sideways_timeout_minutes': 18,
                'runner_hold_criteria': {'min_r_multiple': 5.5, 'min_duration_minutes': 28, 'max_drawdown_pct': 0.26}
            },
            'NORMAL': {
                'breakeven_mult': 1.0, 'trailing_mult': 1.0, 'stop_mult': 3.6,
                'partial_1_r': 2.0, 'partial_1_pct': 0.50,
                'partial_2_r': 3.0, 'partial_2_pct': 0.30,
                'partial_3_r': 5.0, 'partial_3_pct': 0.20,
                'sideways_timeout_minutes': 12,
                'runner_hold_criteria': {'min_r_multiple': 5.0, 'min_duration_minutes': 25, 'max_drawdown_pct': 0.27}
            },
            'NORMAL_TRENDING': {
                'breakeven_mult': 1.0, 'trailing_mult': 1.1, 'stop_mult': 3.6,
                'partial_1_r': 2.2, 'partial_1_pct': 0.45,
                'partial_2_r': 3.5, 'partial_2_pct': 0.30,
                'partial_3_r': 5.5, 'partial_3_pct': 0.25,
                'sideways_timeout_minutes': 15,
                'runner_hold_criteria': {'min_r_multiple': 6.0, 'min_duration_minutes': 30, 'max_drawdown_pct': 0.25}
            },
            'NORMAL_CHOPPY': {
                'breakeven_mult': 0.95, 'trailing_mult': 0.95, 'stop_mult': 3.4,
                'partial_1_r': 1.8, 'partial_1_pct': 0.55,
                'partial_2_r': 2.8, 'partial_2_pct': 0.30,
                'partial_3_r': 4.5, 'partial_3_pct': 0.15,
                'sideways_timeout_minutes': 10,
                'runner_hold_criteria': {'min_r_multiple': 4.0, 'min_duration_minutes': 18, 'max_drawdown_pct': 0.29}
            }
        }
        
        # Load experiences and learned params from file (may override above defaults)
        self.load_experiences()
        
        logger.info(f"[ADAPTIVE] Exit Manager initialized with {'CLOUD' if self.use_cloud else 'LOCAL'} RL learning ({len(self.exit_experiences)} past exits)")
    
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
    
    def capture_complete_exit_state(self, 
                                   # Market Context (8)
                                   market_regime: str = 'NORMAL',
                                   rsi: float = 50.0,
                                   volume_ratio: float = 1.0,
                                   atr: float = 2.0,
                                   vix: float = 15.0,
                                   volatility_regime_change: bool = False,
                                   volume_at_exit: float = 1.0,
                                   market_state: int = 0,
                                   # Trade Context (5)
                                   entry_confidence: float = 0.5,
                                   side: str = 'LONG',
                                   session: str = 'NY',
                                   commission_cost: float = 0.0,
                                   regime: str = 'NORMAL',
                                   # Time Features (5)
                                   hour: int = 12,
                                   day_of_week: int = 0,
                                   duration: float = 10.0,
                                   time_in_breakeven_bars: int = 0,
                                   bars_until_breakeven: int = 0,
                                   # Performance Metrics (5)
                                   mae: float = 0.0,
                                   mfe: float = 0.0,
                                   max_r_achieved: float = 0.0,
                                   min_r_achieved: float = 0.0,
                                   r_multiple: float = 0.0,
                                   # Exit Strategy State (6)
                                   breakeven_activated: bool = False,
                                   trailing_activated: bool = False,
                                   stop_hit: bool = False,
                                   exit_param_update_count: int = 0,
                                   stop_adjustment_count: int = 0,
                                   bars_until_trailing: int = 0,
                                   # Results (5)
                                   pnl: float = 0.0,
                                   outcome: str = 'LOSS',
                                   win: bool = False,
                                   exit_reason: str = 'stop_loss',
                                   max_profit_reached: float = 0.0,
                                   # Advanced (8)
                                   atr_change_percent: float = 0.0,
                                   avg_atr_during_trade: float = 2.0,
                                   peak_r_multiple: float = 0.0,
                                   profit_drawdown_from_peak: float = 0.0,
                                   high_volatility_bars: int = 0,
                                   wins_in_last_5_trades: int = 0,
                                   losses_in_last_5_trades: int = 0,
                                   minutes_until_close: float = 240.0,
                                   # Temporal (5)
                                   entry_hour: int = 12,
                                   entry_minute: int = 0,
                                   exit_hour: int = 12,
                                   exit_minute: int = 0,
                                   bars_held: int = 0,
                                   # Position Tracking (3)
                                   entry_bar: int = 0,
                                   exit_bar: int = 0,
                                   contracts: int = 1,
                                   # Trade Context (3)
                                   trade_number_in_session: int = 0,
                                   cumulative_pnl_before_trade: float = 0.0,
                                   entry_price: float = 6500.0,
                                   # Performance (4)
                                   peak_unrealized_pnl: float = 0.0,
                                   opportunity_cost: float = 0.0,
                                   max_drawdown_percent: float = 0.0,
                                   drawdown_bars: int = 0,
                                   # Strategy Milestones (4)
                                   breakeven_activation_bar: int = 0,
                                   trailing_activation_bar: int = 0,
                                   duration_bars: int = 0,
                                   held_through_sessions: bool = False,
                                   # Additional metadata
                                   symbol: str = 'ES',
                                   timestamp: str = None,
                                   # Daily Loss Limit Features (3) - NEW
                                   daily_pnl_before_trade: float = 0.0,
                                   daily_loss_limit: float = 1000.0,
                                   daily_loss_proximity_pct: float = 0.0) -> Dict:
        """
        Capture COMPLETE exit state with ALL 62 features for neural network.
        (59 base features + 3 daily loss limit features, removed 2 backtest-incompatible features)
        
        Returns:
            Dictionary with all 62 features properly formatted
        """
        from datetime import datetime
        import pytz
        
        if timestamp is None:
            timestamp = datetime.now(pytz.UTC).isoformat()
        
        # Mappings
        regime_map = {
            'NORMAL': 0, 'NORMAL_TRENDING': 1, 'HIGH_VOL_TRENDING': 2,
            'HIGH_VOL_CHOPPY': 3, 'LOW_VOL_TRENDING': 4, 'LOW_VOL_RANGING': 5,
            'UNKNOWN': 0
        }
        session_map = {'Asia': 0, 'London': 1, 'NY': 2}
        outcome_map = {'WIN': 1, 'LOSS': 0}
        exit_reason_map = {
            'take_profit': 0, 'stop_loss': 1, 'trailing_stop': 2,
            'time_limit': 3, 'partial_exit': 4
        }
        
        return {
            # Market Context (8)
            'market_regime': market_regime,
            'regime_encoded': regime_map.get(market_regime, 0),
            'rsi': round(rsi, 2),
            'volume_ratio': round(volume_ratio, 4),
            'atr': round(atr, 4),
            'vix': round(vix, 2),
            'volatility_regime_change': int(volatility_regime_change),
            'volume_at_exit': round(volume_at_exit, 4),
            'market_state': market_state,
            
            # Trade Context (5)
            'entry_confidence': round(entry_confidence, 4),
            'side': side,
            'side_encoded': 1 if side == 'SHORT' else 0,
            'session': session,
            'session_encoded': session_map.get(session, 2),
            'commission_cost': round(commission_cost, 2),
            'regime': regime,
            
            # Time Features (5)
            'hour': hour,
            'day_of_week': day_of_week,
            'duration': round(duration, 2),
            'time_in_breakeven_bars': time_in_breakeven_bars,
            'bars_until_breakeven': bars_until_breakeven,
            
            # Performance Metrics (5)
            'mae': round(mae, 4),
            'mfe': round(mfe, 4),
            'max_r_achieved': round(max_r_achieved, 4),
            'min_r_achieved': round(min_r_achieved, 4),
            'r_multiple': round(r_multiple, 4),
            
            # Exit Strategy State (6)
            'breakeven_activated': int(breakeven_activated),
            'trailing_activated': int(trailing_activated),
            'stop_hit': int(stop_hit),
            'exit_param_update_count': exit_param_update_count,
            'stop_adjustment_count': stop_adjustment_count,
            'bars_until_trailing': bars_until_trailing,
            
            # Results (5)
            'pnl': round(pnl, 2),
            'outcome': outcome,
            'outcome_encoded': outcome_map.get(outcome, 0),
            'win': int(win),
            'exit_reason': exit_reason,
            'exit_reason_encoded': exit_reason_map.get(exit_reason, 1),
            'max_profit_reached': round(max_profit_reached, 2),
            
            # Advanced (8)
            'atr_change_percent': round(atr_change_percent, 4),
            'avg_atr_during_trade': round(avg_atr_during_trade, 4),
            'peak_r_multiple': round(peak_r_multiple, 4),
            'profit_drawdown_from_peak': round(profit_drawdown_from_peak, 4),
            'high_volatility_bars': high_volatility_bars,
            'wins_in_last_5_trades': wins_in_last_5_trades,
            'losses_in_last_5_trades': losses_in_last_5_trades,
            'minutes_until_close': round(minutes_until_close, 2),
            
            # Temporal (5)
            'entry_hour': entry_hour,
            'entry_minute': entry_minute,
            'exit_hour': exit_hour,
            'exit_minute': exit_minute,
            'bars_held': bars_held,
            
            # Position Tracking (3)
            'entry_bar': entry_bar,
            'exit_bar': exit_bar,
            'contracts': contracts,
            
            # Trade Context (3)
            'trade_number_in_session': trade_number_in_session,
            'cumulative_pnl_before_trade': round(cumulative_pnl_before_trade, 2),
            'entry_price': round(entry_price, 2),
            
            # Performance (4)
            'peak_unrealized_pnl': round(peak_unrealized_pnl, 2),
            'opportunity_cost': round(opportunity_cost, 2),
            'max_drawdown_percent': round(max_drawdown_percent, 4),
            'drawdown_bars': drawdown_bars,
            
            # Strategy Milestones (4)
            'breakeven_activation_bar': breakeven_activation_bar,
            'trailing_activation_bar': trailing_activation_bar,
            'duration_bars': duration_bars,
            'held_through_sessions': int(held_through_sessions),
            
            # Daily Loss Limit Features (3) - NEW
            'daily_pnl_before_trade': round(daily_pnl_before_trade, 2),
            'daily_loss_limit': round(daily_loss_limit, 2),
            'daily_loss_proximity_pct': round(daily_loss_proximity_pct, 2),
            
            # Metadata
            'symbol': symbol,
            'timestamp': timestamp,
            'experience_type': 'exit'
        }
    
    def record_exit_outcome(self, regime: str, exit_params: Dict, trade_outcome: Dict, 
                           market_state: Dict = None, partial_exits: list = None, backtest_mode: bool = False):
        """
        Record exit outcome for RL learning WITH COMPLETE 62-FEATURE CAPTURE + 131 EXIT PARAMETERS.
        
        Args:
            regime: Market regime when exit occurred
            exit_params: Exit parameters used (ALL 131 parameters)
            trade_outcome: Trade result (pnl, duration, exit_reason, win/loss, PLUS all 45 tracked features)
            market_state: Optional dict with RSI, volume_ratio, hour, day_of_week, streak, recent_pnl, vix, vwap_distance, atr
            partial_exits: List of partial exit decisions (level, r_multiple, contracts, percentage)
            backtest_mode: If True, collect for bulk save at end (don't POST immediately)
        """
        # Extract daily loss limit features from trade_outcome (NEW)
        daily_pnl_before = trade_outcome.get('daily_pnl_before_trade', 0.0)
        daily_loss_lim = trade_outcome.get('daily_loss_limit', 1000.0)
        daily_loss_prox = trade_outcome.get('daily_loss_proximity_pct', 0.0)
        
        # Use COMPLETE 64-feature capture method (was 62, added 3 daily limit features)
        experience = self.capture_complete_exit_state(
            regime=regime,
            exit_params=exit_params,
            trade_outcome=trade_outcome,
            market_state=market_state,
            partial_exits=partial_exits,
            # Pass daily loss limit features (NEW)
            daily_pnl_before_trade=daily_pnl_before,
            daily_loss_limit=daily_loss_lim,
            daily_loss_proximity_pct=daily_loss_prox
        )
        
        # ADD: All 131 exit parameters that were USED for this trade
        # This is what the neural network will learn to predict
        experience['exit_params_used'] = extract_all_exit_params(exit_params)
        
        # ADD: Execution tracking - what actually happened
        experience['execution_tracking'] = extract_execution_data(trade_outcome)
        
        self.exit_experiences.append(experience)
        
        # BULK SAVE MODE: Collect for later (backtest)
        if backtest_mode:
            return  # Don't POST immediately, backtest will bulk save at end
        
        # LIVE MODE: Save to cloud API immediately WITH ALL 64 FEATURES
        if self.use_cloud:
            try:
                # Prepare cloud payload
                cloud_payload = {
                    "user_id": "default_user",  # TODO: Get real user_id
                    "symbol": "ES",
                    "experience_type": "exit",
                    "experience": experience  # Complete 62-feature dict (59 base + 3 daily loss limit)
                }
                
                response = requests.post(
                    f"{self.cloud_api_url}/api/ml/save_experience",  # Updated endpoint
                    json=cloud_payload,
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('saved'):
                        logger.info(f"✅ [CLOUD] Saved exit experience with 62 features to cloud pool ({data.get('total_experiences', 0):,} total)")
                    else:
                        logger.warning(f"[CLOUD] Failed to save: {data.get('error', 'Unknown error')}")
                else:
                    logger.warning(f"[CLOUD] Save failed with status {response.status_code}")
                    
            except Exception as e:
                logger.error(f"[CLOUD] Error saving to cloud: {e}")
                logger.warning("[CLOUD] Experience saved locally but not in cloud pool")
        
        # Save every 3 exits (local file OR trigger cloud re-learning)
        if len(self.exit_experiences) % 3 == 0:
            if not self.use_cloud:
                self.save_experiences()  # Only save to file if not using cloud
            # Re-learn optimal parameters
            self.update_learned_parameters()
        
        # Log with market context
        market_ctx = experience.get('market_context', {})
        logger.info(f"[EXIT RL] LEARNED: {regime} | {exit_params.get('breakeven_threshold_ticks', 0)}t BE, "
                   f"{exit_params.get('trailing_distance_ticks', 0)}t Trail | "
                   f"P&L: ${trade_outcome['pnl']:.2f} | {trade_outcome['exit_reason']} | "
                   f"RSI:{market_ctx.get('rsi', 0):.1f} Vol:{market_ctx.get('volume_ratio', 0):.2f} "
                   f"Hour:{market_ctx.get('hour', 0)} Streak:{market_ctx.get('streak', 0)}")
    
    def update_learned_parameters(self):
        """
        Update learned optimal parameters based on past exit outcomes.
        Analyzes which parameter combos → best results per regime AND market context.
        
        NOW LEARNS FROM:
        - Regime (choppy/trending/high_vol)
        - RSI levels (overbought/oversold)
        - Volume conditions (high/low)
        - Time of day patterns
        - Win/loss streaks
        - Recent P&L
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
                self.learned_params[regime] = {'breakeven_mult': 1.0, 'trailing_mult': 1.0, 'stop_mult': 3.6}
                logger.info(f"[EXIT RL] New regime discovered: {regime}, initializing with defaults")
            
            # LEARN OPTIMAL STOP LOSS MULTIPLIER
            # Analyze what stop distance worked best in this regime
            wide_stops = [o for o in outcomes if o['exit_params'].get('stop_mult', 3.6) >= 4.0]
            normal_stops = [o for o in outcomes if 3.2 <= o['exit_params'].get('stop_mult', 3.6) < 4.0]
            tight_stops = [o for o in outcomes if o['exit_params'].get('stop_mult', 3.6) < 3.2]
            
            if len(wide_stops) >= 3 and len(tight_stops) >= 3:
                wide_pnl = sum(o['outcome']['pnl'] for o in wide_stops) / len(wide_stops)
                tight_pnl = sum(o['outcome']['pnl'] for o in tight_stops) / len(tight_stops)
                
                if wide_pnl > tight_pnl + 50:  # Wide stops significantly better
                    self.learned_params[regime]['stop_mult'] = min(4.5, self.learned_params[regime].get('stop_mult', 3.6) * 1.05)
                    logger.info(f"[EXIT RL] {regime}: WIDE stops work better (${wide_pnl:.0f} vs ${tight_pnl:.0f})")
                elif tight_pnl > wide_pnl + 50:  # Tight stops significantly better
                    self.learned_params[regime]['stop_mult'] = max(2.8, self.learned_params[regime].get('stop_mult', 3.6) * 0.95)
                    logger.info(f"[EXIT RL] {regime}: TIGHT stops work better (${tight_pnl:.0f} vs ${wide_pnl:.0f})")
            
            # LEARN BREAKEVEN TIMING
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
            
            # NEW: Learn optimal partial exit parameters
            self._learn_partial_exit_params(regime, outcomes)
            
            # NEW: Learn sideways timeout
            self._learn_sideways_timeout(regime, outcomes)
            
            # NEW: Learn from market context patterns
            self._learn_from_market_patterns(regime, outcomes)
            
            # NEW: Learn optimal scaling strategies from outcomes
            self._learn_scaling_strategies(regime, outcomes)
            
            # NEW: Learn runner hold criteria
            self._learn_runner_hold_criteria(regime, outcomes)
    
    def _learn_partial_exit_params(self, regime: str, outcomes: list):
        """
        Learn optimal R-multiples and percentages for partial exits.
        
        Analyzes which partial exit strategies produced best total P&L:
        - Should we take 50% @ 2R or 40% @ 2.5R?
        - Should runners be 20% or 30%?
        - What R-multiple captures the most profit without giving back?
        """
        if len(outcomes) < 15:
            return  # Need minimum data
        
        # Analyze outcomes by their partial exit settings
        # Group by first partial R-multiple
        early_partials = [o for o in outcomes if o.get('exit_params', {}).get('partial_1_r', 2.0) < 2.2]  # ~2R
        mid_partials = [o for o in outcomes if 2.2 <= o.get('exit_params', {}).get('partial_1_r', 2.0) < 2.8]  # ~2.5R
        late_partials = [o for o in outcomes if o.get('exit_params', {}).get('partial_1_r', 2.0) >= 2.8]  # ~3R+
        
        if len(early_partials) >= 5:
            avg_pnl = sum(o['outcome']['pnl'] for o in early_partials) / len(early_partials)
            avg_r = sum(o['outcome'].get('r_multiple', 0) for o in early_partials) / len(early_partials)
            logger.debug(f"[PARTIAL RL] {regime} Early partials (~2R): ${avg_pnl:.2f} avg, {avg_r:.2f}R")
        
        if len(mid_partials) >= 5:
            avg_pnl = sum(o['outcome']['pnl'] for o in mid_partials) / len(mid_partials)
            avg_r = sum(o['outcome'].get('r_multiple', 0) for o in mid_partials) / len(mid_partials)
            logger.debug(f"[PARTIAL RL] {regime} Mid partials (~2.5R): ${avg_pnl:.2f} avg, {avg_r:.2f}R")
        
        if len(late_partials) >= 5:
            avg_pnl = sum(o['outcome']['pnl'] for o in late_partials) / len(late_partials)
            avg_r = sum(o['outcome'].get('r_multiple', 0) for o in late_partials) / len(late_partials)
            logger.debug(f"[PARTIAL RL] {regime} Late partials (~3R): ${avg_pnl:.2f} avg, {avg_r:.2f}R")
        
        # Learn: Which timing worked best?
        best_strategy = None
        best_pnl = -999999
        
        if len(early_partials) >= 5:
            pnl = sum(o['outcome']['pnl'] for o in early_partials) / len(early_partials)
            if pnl > best_pnl:
                best_pnl, best_strategy = pnl, ('early', 2.0)
        
        if len(mid_partials) >= 5:
            pnl = sum(o['outcome']['pnl'] for o in mid_partials) / len(mid_partials)
            if pnl > best_pnl:
                best_pnl, best_strategy = pnl, ('mid', 2.5)
        
        if len(late_partials) >= 5:
            pnl = sum(o['outcome']['pnl'] for o in late_partials) / len(late_partials)
            if pnl > best_pnl:
                best_pnl, best_strategy = pnl, ('late', 3.0)
        
        # Adjust learned params toward best strategy
        if best_strategy:
            timing, target_r = best_strategy
            current_r = self.learned_params[regime].get('partial_1_r', 2.0)
            
            # Move 10% toward optimal
            new_r = current_r * 0.9 + target_r * 0.1
            self.learned_params[regime]['partial_1_r'] = max(1.5, min(3.5, new_r))
            
            logger.info(f"[PARTIAL RL] {regime}: {timing.upper()} partials work best (${best_pnl:.2f}), adjusting to {new_r:.2f}R")
        
        # Learn percentage splits
        # Analyze: Did aggressive scaling (60%+ first partial) or patient (40%) work better?
        aggressive_pct = [o for o in outcomes if o.get('exit_params', {}).get('partial_1_pct', 0.5) >= 0.55]
        patient_pct = [o for o in outcomes if o.get('exit_params', {}).get('partial_1_pct', 0.5) <= 0.45]
        
        if len(aggressive_pct) >= 5 and len(patient_pct) >= 5:
            agg_pnl = sum(o['outcome']['pnl'] for o in aggressive_pct) / len(aggressive_pct)
            pat_pnl = sum(o['outcome']['pnl'] for o in patient_pct) / len(patient_pct)
            
            current_pct = self.learned_params[regime].get('partial_1_pct', 0.50)
            
            if agg_pnl > pat_pnl + 25:  # Aggressive significantly better
                new_pct = min(0.70, current_pct * 1.05)
                self.learned_params[regime]['partial_1_pct'] = new_pct
                logger.info(f"[PARTIAL RL] {regime}: AGGRESSIVE scaling better (${agg_pnl:.2f} vs ${pat_pnl:.2f}), increasing to {new_pct:.0%}")
            elif pat_pnl > agg_pnl + 25:  # Patient significantly better
                new_pct = max(0.30, current_pct * 0.95)
                self.learned_params[regime]['partial_1_pct'] = new_pct
                logger.info(f"[PARTIAL RL] {regime}: PATIENT scaling better (${pat_pnl:.2f} vs ${agg_pnl:.2f}), decreasing to {new_pct:.0%}")
    
    def _learn_sideways_timeout(self, regime: str, outcomes: list):
        """
        Learn optimal timeout for sideways/stalling runners.
        
        Analyzes trades that stalled vs continued:
        - If runner stalled 10+ min then reversed, learn to exit earlier
        - If runner stalled but then ran to 8R, learn to hold longer
        """
        if len(outcomes) < 20:
            return
        
        # Find trades that achieved 3R+ (had a runner)
        runner_trades = [o for o in outcomes if o['outcome'].get('r_multiple', 0) >= 3.0]
        
        if len(runner_trades) < 10:
            return
        
        # Analyze by duration and outcome
        quick_winners = [o for o in runner_trades if o.get('duration_minutes', 999) <= 15 and o['outcome']['pnl'] > 0]
        slow_winners = [o for o in runner_trades if o.get('duration_minutes', 999) > 20 and o['outcome']['pnl'] > 0]
        stalled_losers = [o for o in runner_trades if o.get('duration_minutes', 999) > 20 and o['outcome']['pnl'] < 0]
        
        if quick_winners and slow_winners:
            quick_avg = sum(o['outcome']['pnl'] for o in quick_winners) / len(quick_winners)
            slow_avg = sum(o['outcome']['pnl'] for o in slow_winners) / len(slow_winners)
            
            logger.debug(f"[TIMEOUT RL] {regime}: Quick (<15min): ${quick_avg:.2f} ({len(quick_winners)} trades), "
                        f"Slow (>20min): ${slow_avg:.2f} ({len(slow_winners)} trades)")
        
        # If many stalled losers, tighten timeout
        if len(stalled_losers) >= 5:
            stalled_avg_loss = sum(o['outcome']['pnl'] for o in stalled_losers) / len(stalled_losers)
            current_timeout = self.learned_params[regime].get('sideways_timeout_minutes', 15)
            
            # If losing trades stalled too long, reduce timeout
            if stalled_avg_loss < -50 and current_timeout > 8:
                new_timeout = max(8, current_timeout - 2)
                self.learned_params[regime]['sideways_timeout_minutes'] = new_timeout
                logger.info(f"[TIMEOUT RL] {regime}: {len(stalled_losers)} stalled losers (avg ${stalled_avg_loss:.2f}), "
                           f"reducing timeout to {new_timeout} minutes")
        
        # If slow winners exist and profitable, loosen timeout
        elif len(slow_winners) >= 5:
            slow_avg = sum(o['outcome']['pnl'] for o in slow_winners) / len(slow_winners)
            current_timeout = self.learned_params[regime].get('sideways_timeout_minutes', 15)
            
            if slow_avg > 100 and current_timeout < 25:
                new_timeout = min(25, current_timeout + 2)
                self.learned_params[regime]['sideways_timeout_minutes'] = new_timeout
                logger.info(f"[TIMEOUT RL] {regime}: {len(slow_winners)} slow winners (avg ${slow_avg:.2f}), "
                           f"increasing timeout to {new_timeout} minutes")
    
    def _learn_scaling_strategies(self, regime: str, outcomes: list):
        """
        Analyze exit outcomes to learn WHEN to scale aggressively vs hold full position.
        
        Tracks which scaling strategies produced best results in different market contexts:
        - Did aggressive scaling (70% @ 2R) work better when RSI > 70?
        - Did holding full position work better during trending + volume?
        - Should we exit 100% early during afternoon chop?
        """
        if len(outcomes) < 20:
            return  # Need more data to learn scaling patterns
        
        # Group by scaling behavior (from partial_exits field)
        aggressive_exits = []  # Took 70%+ @ 2R
        patient_exits = []     # Held past 3R or took < 30% @ 2R
        full_early_exits = []  # Exited 100% before 4R
        
        for outcome in outcomes:
            partials = outcome.get('partial_exits', [])
            market = outcome.get('market_state', {})
            pnl = outcome['outcome']['pnl']
            r_multiple = outcome['outcome'].get('r_multiple', 0)
            
            if not partials:
                continue
            
            # Classify scaling behavior
            first_partial = partials[0] if partials else None
            if first_partial:
                first_r = first_partial.get('r_multiple', 0)
                first_contracts = first_partial.get('contracts', 0)
                total_contracts = outcome.get('outcome', {}).get('contracts', 1)
                first_pct = first_contracts / total_contracts if total_contracts > 0 else 0
                
                # Aggressive: Took 70%+ at or before 2.5R
                if first_r <= 2.5 and first_pct >= 0.65:
                    aggressive_exits.append({
                        'pnl': pnl,
                        'r_multiple': r_multiple,
                        'rsi': market.get('rsi', 50),
                        'volume_ratio': market.get('volume_ratio', 1.0),
                        'hour': market.get('hour', 12),
                        'regime': regime
                    })
                
                # Patient: First partial after 3R OR took < 30% initially
                elif first_r >= 3.0 or first_pct <= 0.30:
                    patient_exits.append({
                        'pnl': pnl,
                        'r_multiple': r_multiple,
                        'rsi': market.get('rsi', 50),
                        'volume_ratio': market.get('volume_ratio', 1.0),
                        'hour': market.get('hour', 12),
                        'regime': regime
                    })
            
            # Full early exit: All contracts closed before 4R
            if r_multiple < 4.0 and len(partials) >= 1:
                last_partial = partials[-1]
                if last_partial.get('r_multiple', 0) < 4.0:
                    full_early_exits.append({
                        'pnl': pnl,
                        'r_multiple': r_multiple,
                        'rsi': market.get('rsi', 50),
                        'hour': market.get('hour', 12),
                        'regime': regime
                    })
        
        # Analyze which strategies worked best in which conditions
        if len(aggressive_exits) >= 5:
            avg_pnl_aggressive = sum(e['pnl'] for e in aggressive_exits) / len(aggressive_exits)
            avg_r_aggressive = sum(e['r_multiple'] for e in aggressive_exits) / len(aggressive_exits)
            
            # When did aggressive scaling work BEST?
            high_rsi_aggressive = [e for e in aggressive_exits if e['rsi'] > 65]
            choppy_aggressive = [e for e in aggressive_exits if 'CHOPPY' in e['regime']]
            afternoon_aggressive = [e for e in aggressive_exits if 13 <= e['hour'] <= 15]
            
            logger.info(f"[SCALING LEARNING] {regime} Aggressive (70%+ @ 2R): {len(aggressive_exits)} trades, "
                       f"${avg_pnl_aggressive:.2f} avg, {avg_r_aggressive:.2f}R avg")
            
            if high_rsi_aggressive:
                avg_pnl = sum(e['pnl'] for e in high_rsi_aggressive) / len(high_rsi_aggressive)
                logger.info(f"  └─ @ RSI>65: {len(high_rsi_aggressive)} trades, ${avg_pnl:.2f} avg")
            
            if choppy_aggressive:
                avg_pnl = sum(e['pnl'] for e in choppy_aggressive) / len(choppy_aggressive)
                logger.info(f"  └─ @ Choppy: {len(choppy_aggressive)} trades, ${avg_pnl:.2f} avg")
        
        if len(patient_exits) >= 5:
            avg_pnl_patient = sum(e['pnl'] for e in patient_exits) / len(patient_exits)
            avg_r_patient = sum(e['r_multiple'] for e in patient_exits) / len(patient_exits)
            
            # When did patient holding work BEST?
            trending_patient = [e for e in patient_exits if 'TRENDING' in e['regime']]
            high_vol_patient = [e for e in patient_exits if e['volume_ratio'] > 1.3]
            
            logger.info(f"[SCALING LEARNING] {regime} Patient (hold past 3R): {len(patient_exits)} trades, "
                       f"${avg_pnl_patient:.2f} avg, {avg_r_patient:.2f}R avg")
            
            if trending_patient:
                avg_pnl = sum(e['pnl'] for e in trending_patient) / len(trending_patient)
                avg_r = sum(e['r_multiple'] for e in trending_patient) / len(trending_patient)
                logger.info(f"  └─ @ Trending: {len(trending_patient)} trades, ${avg_pnl:.2f} avg, {avg_r:.2f}R")
            
            if high_vol_patient:
                avg_pnl = sum(e['pnl'] for e in high_vol_patient) / len(high_vol_patient)
                logger.info(f"  └─ @ High Volume: {len(high_vol_patient)} trades, ${avg_pnl:.2f} avg")
        
        # Compare: Which strategy was better overall?
        if len(aggressive_exits) >= 5 and len(patient_exits) >= 5:
            aggressive_avg = sum(e['pnl'] for e in aggressive_exits) / len(aggressive_exits)
            patient_avg = sum(e['pnl'] for e in patient_exits) / len(patient_exits)
            
            better_strategy = "AGGRESSIVE" if aggressive_avg > patient_avg else "PATIENT"
            diff = abs(aggressive_avg - patient_avg)
            
            logger.info(f"[SCALING INSIGHT] {regime}: {better_strategy} scaling was ${diff:.2f} better on average")
    
    
    
    def _learn_from_market_patterns(self, regime: str, outcomes: list):
        """
        Analyze exit outcomes by market context (RSI, volume, time, etc.)
        to learn when tight vs loose exits work best.
        """
        if len(outcomes) < 10:
            return
        
        # Group by market conditions
        high_rsi_exits = [o for o in outcomes if o.get('market_state', {}).get('rsi', 50) > 65]
        low_rsi_exits = [o for o in outcomes if o.get('market_state', {}).get('rsi', 50) < 35]
        
        high_vol_exits = [o for o in outcomes if o.get('market_state', {}).get('volume_ratio', 1.0) > 1.5]
        low_vol_exits = [o for o in outcomes if o.get('market_state', {}).get('volume_ratio', 1.0) < 0.7]
        
        afternoon_exits = [o for o in outcomes if 13 <= o.get('market_state', {}).get('hour', 12) <= 15]
        morning_exits = [o for o in outcomes if 9 <= o.get('market_state', {}).get('hour', 12) <= 11]
        
        # Analyze what worked best in each condition
        if len(high_rsi_exits) >= 5:
            avg_pnl = sum(o['outcome']['pnl'] for o in high_rsi_exits) / len(high_rsi_exits)
            tight_count = sum(1 for o in high_rsi_exits if o['exit_params']['trailing_distance_ticks'] < 11)
            logger.info(f"[EXIT RL PATTERN] {regime} @ RSI>65: ${avg_pnl:.2f} avg | "
                       f"{tight_count}/{len(high_rsi_exits)} used tight trailing")
        
        if len(high_vol_exits) >= 5:
            avg_pnl = sum(o['outcome']['pnl'] for o in high_vol_exits) / len(high_vol_exits)
            wide_count = sum(1 for o in high_vol_exits if o['exit_params']['trailing_distance_ticks'] > 12)
            logger.info(f"[EXIT RL PATTERN] {regime} @ High Volume: ${avg_pnl:.2f} avg | "
                       f"{wide_count}/{len(high_vol_exits)} used wide trailing")
        
        if len(afternoon_exits) >= 5:
            avg_pnl = sum(o['outcome']['pnl'] for o in afternoon_exits) / len(afternoon_exits)
            quick_be = sum(1 for o in afternoon_exits if o['exit_params']['breakeven_threshold_ticks'] <= 7)
            logger.info(f"[EXIT RL PATTERN] {regime} @ Afternoon (13-15h): ${avg_pnl:.2f} avg | "
                       f"{quick_be}/{len(afternoon_exits)} used quick breakeven")

    
    def _learn_runner_hold_criteria(self, regime: str, outcomes: list):
        """
        Learn optimal runner hold criteria (when to let runner run vs exit early).
        
        Analyzes runner trades for:
        - Minimum R-multiple to justify holding
        - Minimum duration to hold runner
        - Maximum drawdown percentage before exit
        """
        if len(outcomes) < 20:
            return
        
        # Find trades that achieved 3R+ (had a runner)
        runner_trades = [o for o in outcomes if o['outcome'].get('r_multiple', 0) >= 3.0]
        
        if len(runner_trades) < 10:
            return
        
        # Analyze min R-multiple for best runners
        high_runners = [o for o in runner_trades if o['outcome'].get('r_multiple', 0) >= 6.0]
        med_runners = [o for o in runner_trades if 4.0 <= o['outcome'].get('r_multiple', 0) < 6.0]
        low_runners = [o for o in runner_trades if 3.0 <= o['outcome'].get('r_multiple', 0) < 4.0]
        
        best_group = None
        best_pnl = -999999
        
        for group, trades in [('high', high_runners), ('med', med_runners), ('low', low_runners)]:
            if len(trades) >= 5:
                avg_pnl = sum(o['outcome']['pnl'] for o in trades) / len(trades)
                if avg_pnl > best_pnl:
                    best_pnl, best_group = avg_pnl, group
        
        # Adjust min_r_multiple toward best group
        if best_group and 'runner_hold_criteria' in self.learned_params[regime]:
            current_min_r = self.learned_params[regime]['runner_hold_criteria']['min_r_multiple']
            
            if best_group == 'high':
                new_min_r = min(10.0, current_min_r * 0.9 + 6.0 * 0.1)
                logger.info(f"[RUNNER RL] {regime}: HIGH runners (6R+) most profitable, targeting {new_min_r:.1f}R")
            elif best_group == 'med':
                new_min_r = min(8.0, current_min_r * 0.9 + 4.5 * 0.1)
                logger.info(f"[RUNNER RL] {regime}: MED runners (4-6R) most profitable, targeting {new_min_r:.1f}R")
            else:
                new_min_r = max(3.0, current_min_r * 0.9 + 3.5 * 0.1)
                logger.info(f"[RUNNER RL] {regime}: LOW runners (3-4R) most profitable, targeting {new_min_r:.1f}R")
            
            self.learned_params[regime]['runner_hold_criteria']['min_r_multiple'] = new_min_r
        
        # Analyze min duration to hold runner
        long_holds = [o for o in runner_trades if o.get('duration_minutes', 0) >= 30 and o['outcome']['pnl'] > 0]
        short_holds = [o for o in runner_trades if o.get('duration_minutes', 0) < 20 and o['outcome']['pnl'] > 0]
        
        if len(long_holds) >= 5 and len(short_holds) >= 5:
            long_avg = sum(o['outcome']['pnl'] for o in long_holds) / len(long_holds)
            short_avg = sum(o['outcome']['pnl'] for o in short_holds) / len(short_holds)
            
            if 'runner_hold_criteria' in self.learned_params[regime]:
                current_min_dur = self.learned_params[regime]['runner_hold_criteria']['min_duration_minutes']
                
                if long_avg > short_avg + 25:  # Long holds significantly better
                    new_min_dur = min(60, current_min_dur + 5)
                    self.learned_params[regime]['runner_hold_criteria']['min_duration_minutes'] = new_min_dur
                    logger.info(f"[RUNNER RL] {regime}: Long holds (30+ min) better (${long_avg:.2f} vs ${short_avg:.2f}), "
                               f"increasing min duration to {new_min_dur} min")
                elif short_avg > long_avg + 25:  # Short holds significantly better
                    new_min_dur = max(10, current_min_dur - 5)
                    self.learned_params[regime]['runner_hold_criteria']['min_duration_minutes'] = new_min_dur
                    logger.info(f"[RUNNER RL] {regime}: Short holds (<20 min) better (${short_avg:.2f} vs ${long_avg:.2f}), "
                               f"decreasing min duration to {new_min_dur} min")
        
        # Analyze max drawdown percentage before exit
        drawdown_trades = [o for o in runner_trades if o.get('max_drawdown_pct', 0.0) > 0.25 and o['outcome']['pnl'] < 0]
        
        if len(drawdown_trades) >= 5 and 'runner_hold_criteria' in self.learned_params[regime]:
            avg_loss = sum(o['outcome']['pnl'] for o in drawdown_trades) / len(drawdown_trades)
            current_dd = self.learned_params[regime]['runner_hold_criteria']['max_drawdown_pct']
            
            # If many losing trades had high drawdowns, tighten max drawdown
            if avg_loss < -50:
                new_dd = max(0.15, current_dd * 0.95)
                self.learned_params[regime]['runner_hold_criteria']['max_drawdown_pct'] = new_dd
                logger.info(f"[RUNNER RL] {regime}: {len(drawdown_trades)} high-drawdown losers (avg ${avg_loss:.2f}), "
                           f"tightening max drawdown to {new_dd:.2%}")

    
    def load_experiences(self):
        """Load past exit experiences from v2 format (shared with backtest), cloud, or local file."""
        
        # Try v2 format FIRST (shared with backtest - highest priority for dev)
        v2_file = "data/local_experiences/exit_experiences_v2.json"
        if os.path.exists(v2_file):
            try:
                with open(v2_file, 'r') as f:
                    data = json.load(f)
                    self.exit_experiences = data.get('experiences', [])
                    logger.info(f"✅ Loaded {len(self.exit_experiences)} exit experiences from v2 format (shared with backtest)")
                    
                    # Re-learn from loaded data
                    if len(self.exit_experiences) > 10:
                        self.update_learned_parameters()
                    return
            except Exception as e:
                logger.warning(f"Failed to load v2 exit experiences: {e}, trying cloud/local...")
        
        # Try cloud if configured
        if self.use_cloud:
            logger.info(f"[CLOUD] Fetching exit experiences from: {self.cloud_api_url}/api/ml/get_exit_experiences")
            try:
                response = requests.get(
                    f"{self.cloud_api_url}/api/ml/get_exit_experiences",
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        self.exit_experiences = data.get('exit_experiences', [])
                        logger.info(f"✅ [CLOUD] Loaded {len(self.exit_experiences):,} exit experiences from cloud pool")
                        
                        # Extract learned params if cloud stored them
                        if self.exit_experiences and len(self.exit_experiences) > 10:
                            self.update_learned_parameters()  # Re-learn from cloud data
                        return
                    else:
                        logger.warning(f"[CLOUD] API returned error: {data.get('error', 'Unknown error')}")
                else:
                    logger.warning(f"[CLOUD] API returned status {response.status_code}")
                    
            except Exception as e:
                logger.error(f"[CLOUD] Failed to fetch from cloud: {e}")
                logger.warning("[CLOUD] Falling back to local file...")
        
        # Fallback to local legacy file
        logger.info(f"[LOCAL] Loading exit experiences from: {self.experience_file}")
        
        if os.path.exists(self.experience_file):
            try:
                with open(self.experience_file, 'r') as f:
                    data = json.load(f)
                    self.exit_experiences = data.get('exit_experiences', [])
                    self.learned_params = data.get('learned_params', self.learned_params)
                    
                    logger.info(f"[LOCAL] Loaded {len(self.exit_experiences)} past exit experiences from local file")
            except Exception as e:
                logger.error(f"[LOCAL] Failed to load exit experiences: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning(f"[LOCAL] No exit experience files found - starting fresh")
    
    
    def get_cloud_exit_params(self, regime: str, market_state: Dict, position: Dict, entry_confidence: float = 0.75) -> Optional[Dict]:
        """
        Query cloud API for REAL-TIME exit parameter recommendations.
        
        This is called MID-TRADE to adapt exit strategy as market conditions change.
        Unlike load_experiences() which loads once at startup, this queries the cloud
        on every bar update to get fresh exit recommendations based on current conditions.
        
        CACHED: To avoid rate limiting, results are cached for 60 seconds per regime.
        
        Args:
            regime: Current market regime (HIGH_VOL_CHOPPY, etc.)
            market_state: Current market conditions (RSI, ATR, VWAP distance, etc.)
            position: Current position state (side, duration, unrealized P&L, etc.)
            entry_confidence: Signal RL confidence score (0.0-1.0)
            
        Returns:
            Dict with cloud-recommended exit parameters, or None if cloud unavailable
            {
                "breakeven_threshold_ticks": 9,
                "breakeven_offset_ticks": 2,
                "trailing_distance_ticks": 10,
                "partial_1_r": 2.0,
                "partial_1_pct": 0.50,
                "stop_mult": 3.8,
                "recommendation": "TIGHTEN (68% win rate in 342 similar exits)"
            }
        """
        if not self.use_cloud:
            return None  # Cloud disabled, use local learned params
        
        # Check cache first (avoid rate limiting)
        import time
        current_time = time.time()
        if regime in self.cloud_exit_params_cache:
            cached = self.cloud_exit_params_cache[regime]
            cache_age = current_time - cached['timestamp']
            if cache_age < self.cloud_cache_duration:
                logger.debug(f"[CLOUD EXIT RL] Using cached params for {regime} (age: {cache_age:.1f}s)")
                return cached['params']
        
        try:
            # Build request payload
            request_data = {
                "regime": regime,
                "market_state": market_state,
                "position": position,
                "entry_confidence": entry_confidence
            }
            
            # Query cloud API
            response = requests.post(
                f"{self.cloud_api_url}/api/ml/get_adaptive_exit_params",
                json=request_data,
                timeout=15  # Longer timeout for large experience database
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    params = data.get('params', {})
                    similar_count = data.get('similar_exits_analyzed', 0)
                    recommendation = data.get('recommendation', 'N/A')
                    
                    logger.info(f"🎯 [CLOUD EXIT RL] {recommendation} (analyzed {similar_count} similar exits)")
                    
                    # Add metadata for debugging
                    params['_cloud_metadata'] = {
                        'similar_exits': similar_count,
                        'avg_pnl': data.get('avg_pnl_similar', 0.0),
                        'win_rate': data.get('win_rate_similar', 0.0),
                        'recommendation': recommendation
                    }
                    
                    # Cache the result
                    self.cloud_exit_params_cache[regime] = {
                        'params': params,
                        'timestamp': current_time
                    }
                    
                    return params
                else:
                    logger.warning(f"[CLOUD EXIT RL] Cloud returned no success: {data}")
                    return None
            elif response.status_code == 429:
                logger.warning("[CLOUD EXIT RL] Rate limited - using cached/local params")
                return None
            else:
                logger.warning(f"[CLOUD EXIT RL] HTTP {response.status_code}")
                return None
                
        except requests.Timeout:
            logger.warning("[CLOUD EXIT RL] Timeout (>15s) - using local params")
            return None
        except Exception as e:
            logger.error(f"[CLOUD EXIT RL] Error querying cloud: {e}")
            return None
    
    
    def get_stop_multiplier(self, regime: str, recent_exits: list = None) -> float:
        """
        Get the learned stop loss multiplier for the current regime.
        Automatically widens stops after cluster of stop-outs.
        
        Args:
            regime: Current market regime
            recent_exits: Optional list of recent exit outcomes
            
        Returns: Stop loss multiplier (ATR multiple, typically 2.8x-4.5x)
        """
        params = self.learned_params.get(regime, self.learned_params.get('NORMAL', {}))
        stop_mult = params.get('stop_mult', 3.6)  # Default 3.6x if not found
        
        # ADAPTIVE STOP WIDENING: Check for cluster of recent stop-outs
        if recent_exits and len(recent_exits) >= 3:
            # Check last 3 trades
            last_3 = recent_exits[-3:]
            stop_outs = [e for e in last_3 if e.get('exit_reason') == 'stop_loss']
            
            # If 2+ of last 3 were stopped out, widen stop
            if len(stop_outs) >= 2:
                # Widen by 15% (e.g., 3.6 → 4.14)
                widened_mult = stop_mult * 1.15
                widened_mult = min(5.0, widened_mult)  # Cap at 5.0x
                
                logger.warning(f"⚠️ [ADAPTIVE STOP] {len(stop_outs)}/3 recent stop-outs in {regime} → "
                             f"widening stop from {stop_mult:.1f}x to {widened_mult:.1f}x ATR (+15%)")
                return widened_mult
            
            # If all 3 were stopped out, widen even more
            if len(stop_outs) == 3:
                widened_mult = stop_mult * 1.25  # +25%
                widened_mult = min(5.5, widened_mult)  # Cap at 5.5x
                
                logger.critical(f"🚨 [ADAPTIVE STOP] 3/3 consecutive stop-outs in {regime} → "
                               f"widening stop from {stop_mult:.1f}x to {widened_mult:.1f}x ATR (+25%)")
                return widened_mult
        
        logger.info(f"[EXIT RL] Using {regime} stop multiplier: {stop_mult:.1f}x ATR")
        return stop_mult
    
    def choose_exit_order_type(self, exit_reason: str, pnl: float, urgency: str = 'normal', 
                               current_spread_ticks: float = 1.0) -> str:
        """
        Learn and choose optimal exit order type based on conditions.
        
        Args:
            exit_reason: Why exiting ('profit_target', 'stop_loss', 'timeout', 'runner_exit')
            pnl: Current P&L in dollars
            urgency: 'low', 'normal', 'high'
            current_spread_ticks: Current bid/ask spread
            
        Returns: 'limit' or 'market'
        
        Learning logic:
        - Stop loss: Always market (get out NOW)
        - Profit target + low urgency: Limit order (patient)
        - Runner exit + high profit: Limit order (don't give back to slippage)
        - High urgency: Market order
        - Wide spread: Limit order (avoid bad price)
        """
        # Hard rules first (safety)
        if exit_reason == 'stop_loss':
            return 'market'  # Get out immediately on stop
        
        if urgency == 'high':
            return 'market'  # Urgent exits need speed
        
        # Learn from past exit slippage outcomes
        if len(self.exit_experiences) >= 20:
            # Find similar exits
            similar_limit = [
                exp for exp in self.exit_experiences
                if exp.get('execution', {}).get('exit_order_type') == 'limit'
                and exp.get('outcome', {}).get('exit_reason') == exit_reason
            ]
            
            similar_market = [
                exp for exp in self.exit_experiences
                if exp.get('execution', {}).get('exit_order_type') == 'market'
                and exp.get('outcome', {}).get('exit_reason') == exit_reason
            ]
            
            if len(similar_limit) >= 5 and len(similar_market) >= 5:
                # Compare average exit slippage
                limit_slippage = sum(
                    exp.get('execution', {}).get('exit_slippage_ticks', 0)
                    for exp in similar_limit
                ) / len(similar_limit)
                
                market_slippage = sum(
                    exp.get('execution', {}).get('exit_slippage_ticks', 0)
                    for exp in similar_market
                ) / len(similar_market)
                
                # Compare final P&L (did slippage cost us?)
                limit_pnl = sum(exp['outcome']['pnl'] for exp in similar_limit) / len(similar_limit)
                market_pnl = sum(exp['outcome']['pnl'] for exp in similar_market) / len(similar_market)
                
                # Log learning
                if abs(limit_pnl - market_pnl) > 10:
                    logger.info(f"📊 [EXIT EXECUTION] {exit_reason}: Limit={limit_slippage:.2f}t slip (${limit_pnl:.2f} avg) "
                               f"vs Market={market_slippage:.2f}t slip (${market_pnl:.2f} avg)")
                
                # Choose better option (prefer limit if similar P&L, less slippage)
                if limit_pnl >= market_pnl - 5:  # Limit at least as good
                    return 'limit'
        
        # Default heuristics if not enough data
        if exit_reason == 'profit_target' and pnl > 100:
            return 'limit'  # Protect profit with limit
        
        if current_spread_ticks > 2.0:
            return 'limit'  # Wide spread = use limit
        
        if urgency == 'low':
            return 'limit'  # No rush = patient limit
        
        return 'market'  # Default to market for safety
    
    def get_dynamic_partial_percentage(self, regime: str, current_r_multiple: float, 
                                      partial_level: int, market_conditions: Dict = None) -> float:
        """
        Dynamically adjust partial exit percentage based on real-time conditions.
        
        Args:
            regime: Current market regime
            current_r_multiple: Current R-multiple achieved
            partial_level: Which partial (1, 2, or 3)
            market_conditions: Optional dict with rsi, volume_spike, reversal_risk
            
        Returns: Percentage to exit (0.0-1.0)
        
        Example: If at 2R and market shows reversal, exit 80% instead of learned 50%
        """
        if market_conditions is None:
            market_conditions = {}
        
        # Get base learned percentage
        param_key = f'partial_{partial_level}_pct'
        base_pct = self.learned_params.get(regime, {}).get(param_key, 0.50)
        
        # Adjust for real-time conditions
        adjusted_pct = base_pct
        
        # Reversal risk detection
        rsi = market_conditions.get('rsi', 50)
        volume_spike = market_conditions.get('volume_spike', False)
        
        if partial_level == 1:  # First partial
            # If showing reversal signs at 2R, take MORE
            if current_r_multiple >= 2.0:
                if rsi > 75 or rsi < 25:  # Overbought/oversold
                    adjusted_pct = min(0.90, base_pct * 1.5)
                    logger.info(f"📊 [DYNAMIC PARTIAL] RSI={rsi:.0f} at {current_r_multiple:.1f}R → "
                               f"increasing partial 1 to {adjusted_pct:.0%} (from {base_pct:.0%})")
                elif volume_spike:  # Climax volume
                    adjusted_pct = min(0.85, base_pct * 1.4)
                    logger.info(f"📊 [DYNAMIC PARTIAL] Volume spike at {current_r_multiple:.1f}R → "
                               f"increasing partial 1 to {adjusted_pct:.0%}")
        
        elif partial_level == 2:  # Second partial
            # If deep in profit and trending, take LESS (let it run)
            if current_r_multiple >= 4.0:
                if 40 < rsi < 60:  # Healthy trend
                    adjusted_pct = max(0.20, base_pct * 0.7)
                    logger.info(f"📊 [DYNAMIC PARTIAL] Healthy trend at {current_r_multiple:.1f}R → "
                               f"decreasing partial 2 to {adjusted_pct:.0%} (let it run)")
        
        return min(1.0, max(0.1, adjusted_pct))
    
    def get_exit_urgency(self, current_time, entry_time, r_multiple: float) -> str:
        """
        Determine exit urgency based on time of day and trade duration.
        
        Args:
            current_time: Current datetime
            entry_time: Entry datetime
            r_multiple: Current R-multiple
            
        Returns: 'low', 'normal', or 'high'
        
        Time-based urgency:
        - 7:45-8:00 PM UTC: HIGH (market closing soon - exit everything)
        - 4:00-5:00 PM UTC: HIGH if losing (lunch chop)
        - Normal hours: NORMAL unless conditions warrant
        """
        hour = current_time.hour
        minute = current_time.minute
        
        # CRITICAL: Near market close (7:45-8:00 PM UTC)
        if hour == 19 and minute >= 45:  # 7:45-8:00 PM UTC
            return 'high'  # Get out NOW
        elif hour >= 20:  # After 8 PM UTC
            return 'high'
        
        # HIGH URGENCY: Lunch hour with losing trade
        if 12 <= hour <= 13 and r_multiple < 0:
            return 'high'  # Choppy lunch period - exit losers fast
        
        # NORMAL: Standard trading hours
        return 'normal'
    
    def check_profit_lock(self, current_r_multiple: float, peak_r_multiple: float, 
                         current_profit_ticks: float, direction: str) -> Dict:
        """
        Check if profit should be locked (ratchet mechanism).
        Once at high R-multiple, never let it fall below lower threshold.
        
        Args:
            current_r_multiple: Current R-multiple (profit/risk)
            peak_r_multiple: Highest R-multiple achieved in trade
            current_profit_ticks: Current profit in ticks
            direction: 'long' or 'short'
            
        Returns:
            Dict with lock_profit (bool), min_acceptable_r (float), reason (str)
        """
        # Define profit lock zones (once achieved X, never go below Y)
        lock_zones = [
            {'achieved': 5.0, 'min_lock': 3.5, 'description': '5R → Lock 3.5R'},
            {'achieved': 4.0, 'min_lock': 3.0, 'description': '4R → Lock 3R'},
            {'achieved': 3.0, 'min_lock': 2.0, 'description': '3R → Lock 2R'},
            {'achieved': 2.5, 'min_lock': 1.5, 'description': '2.5R → Lock 1.5R'},
        ]
        
        # Check which zone applies based on peak R
        for zone in lock_zones:
            if peak_r_multiple >= zone['achieved']:
                # We achieved this level - enforce the lock
                if current_r_multiple < zone['min_lock']:
                    return {
                        'lock_profit': True,
                        'min_acceptable_r': zone['min_lock'],
                        'reason': f"🔒 PROFIT LOCK: Peaked at {peak_r_multiple:.1f}R, "
                                 f"current {current_r_multiple:.1f}R < lock threshold {zone['min_lock']}R "
                                 f"({zone['description']})"
                    }
                
                # Still above lock threshold - no action needed yet
                return {
                    'lock_profit': False,
                    'min_acceptable_r': zone['min_lock'],
                    'reason': f"Above lock ({current_r_multiple:.1f}R > {zone['min_lock']}R)"
                }
        
        # No lock zone reached yet
        return {
            'lock_profit': False,
            'min_acceptable_r': 0.0,
            'reason': f"No lock zone (peak {peak_r_multiple:.1f}R)"
        }
    
    def detect_adverse_momentum(self, recent_bars: list, direction: str, 
                                entry_price: float, current_price: float,
                                position_size: int) -> Dict:
        """
        Detect when price momentum has shifted strongly against the position.
        Exit FAST when market reversing hard.
        
        Args:
            recent_bars: Last 3-5 bars
            direction: 'long' or 'short'
            entry_price: Entry price
            current_price: Current price
            position_size: Number of contracts
            
        Returns:
            Dict with adverse_detected (bool), severity ('low'/'medium'/'high'), reason (str)
        """
        if len(recent_bars) < 3:
            return {
                'adverse_detected': False,
                'severity': 'low',
                'reason': 'Insufficient bar data'
            }
        
        # Get last 3 bars for momentum check
        bars = recent_bars[-3:]
        
        # Calculate price momentum
        if direction == 'long':
            # For LONG: Adverse = rapid downward movement
            bar_directions = []
            for bar in bars:
                if bar['close'] < bar['open']:
                    bar_directions.append('down')
                else:
                    bar_directions.append('up')
            
            # Check for consecutive red bars with increasing ranges
            consecutive_red = all(d == 'down' for d in bar_directions)
            ranges = [(bar['high'] - bar['low']) for bar in bars]
            expanding_range = ranges[-1] > ranges[0] * 1.5  # Last bar 50% bigger
            
            # Current drawdown from entry
            drawdown_pct = ((entry_price - current_price) / entry_price) * 100
            
            if consecutive_red and expanding_range and drawdown_pct > 0.5:
                return {
                    'adverse_detected': True,
                    'severity': 'high',
                    'reason': f"⚠️ ADVERSE MOMENTUM: 3 consecutive red bars with expanding range, "
                             f"drawdown {drawdown_pct:.1f}%"
                }
            elif consecutive_red and drawdown_pct > 0.3:
                return {
                    'adverse_detected': True,
                    'severity': 'medium',
                    'reason': f"Adverse: 3 red bars, drawdown {drawdown_pct:.1f}%"
                }
        
        else:  # SHORT
            # For SHORT: Adverse = rapid upward movement
            bar_directions = []
            for bar in bars:
                if bar['close'] > bar['open']:
                    bar_directions.append('up')
                else:
                    bar_directions.append('down')
            
            # Check for consecutive green bars
            consecutive_green = all(d == 'up' for d in bar_directions)
            ranges = [(bar['high'] - bar['low']) for bar in bars]
            expanding_range = ranges[-1] > ranges[0] * 1.5
            
            # Current drawdown from entry
            drawdown_pct = ((current_price - entry_price) / entry_price) * 100
            
            if consecutive_green and expanding_range and drawdown_pct > 0.5:
                return {
                    'adverse_detected': True,
                    'severity': 'high',
                    'reason': f"⚠️ ADVERSE MOMENTUM: 3 consecutive green bars with expanding range, "
                             f"drawdown {drawdown_pct:.1f}%"
                }
            elif consecutive_green and drawdown_pct > 0.3:
                return {
                    'adverse_detected': True,
                    'severity': 'medium',
                    'reason': f"Adverse: 3 green bars, drawdown {drawdown_pct:.1f}%"
                }
        
        return {
            'adverse_detected': False,
            'severity': 'low',
            'reason': 'No adverse momentum detected'
        }
    
    def check_volume_exhaustion(self, recent_bars: list, r_multiple: float, 
                                avg_volume: float = None) -> Dict:
        """
        Detect when volume is drying up during a profit run (exhaustion signal).
        Exit runners before trend reverses.
        
        Args:
            recent_bars: Last 5-10 bars
            r_multiple: Current R-multiple (only relevant if in profit)
            avg_volume: Average volume for comparison (optional)
            
        Returns:
            Dict with exhaustion_detected (bool), volume_trend (str), reason (str)
        """
        if len(recent_bars) < 5:
            return {
                'exhaustion_detected': False,
                'volume_trend': 'unknown',
                'reason': 'Insufficient bar data'
            }
        
        # Get last 5 bars
        bars = recent_bars[-5:]
        volumes = [bar.get('volume', 0) for bar in bars]
        
        # Calculate volume trend (comparing recent vs earlier)
        early_avg = statistics.mean(volumes[:3])  # First 3 bars
        recent_avg = statistics.mean(volumes[3:])  # Last 2 bars
        
        # Calculate average volume if not provided
        if avg_volume is None:
            avg_volume = statistics.mean([bar.get('volume', 0) for bar in recent_bars[-20:]])
        
        # Volume dropping during profit run = exhaustion
        if r_multiple > 2.0:  # Only relevant when in good profit
            if recent_avg < early_avg * 0.6:  # Volume dropped 40%+
                return {
                    'exhaustion_detected': True,
                    'volume_trend': 'declining',
                    'reason': f"📉 VOLUME EXHAUSTION: Profit at {r_multiple:.1f}R but volume dropped "
                             f"{(1 - recent_avg/early_avg)*100:.0f}% → Exit runner before reversal"
                }
            elif recent_avg < avg_volume * 0.5:  # Recent volume 50% below average
                return {
                    'exhaustion_detected': True,
                    'volume_trend': 'very_low',
                    'reason': f"Volume very low ({recent_avg:.0f} vs {avg_volume:.0f} avg) at {r_multiple:.1f}R"
                }
        
        return {
            'exhaustion_detected': False,
            'volume_trend': 'normal',
            'reason': f"Volume OK (recent {recent_avg:.0f} vs early {early_avg:.0f})"
        }
    
    def detect_failed_breakout(self, recent_bars: list, direction: str,
                               target_hit: bool, entry_price: float,
                               current_price: float) -> Dict:
        """
        Detect when price hit target but immediately reversed (trap/failed breakout).
        Exit FAST before full reversal.
        
        Args:
            recent_bars: Last 2-3 bars
            direction: 'long' or 'short'
            target_hit: Whether target was reached
            entry_price: Entry price
            current_price: Current price
            
        Returns:
            Dict with failed_breakout (bool), severity (str), reason (str)
        """
        if not target_hit or len(recent_bars) < 2:
            return {
                'failed_breakout': False,
                'severity': 'low',
                'reason': 'Target not hit or insufficient data'
            }
        
        # Get last 2 bars
        bars = recent_bars[-2:]
        last_bar = bars[-1]
        prev_bar = bars[-2] if len(bars) > 1 else bars[0]
        
        if direction == 'long':
            # For LONG: Failed breakout = hit high then closed near low
            bar_range = last_bar['high'] - last_bar['low']
            close_from_high = last_bar['high'] - last_bar['close']
            
            # If closed in bottom 30% of range after hitting target
            if bar_range > 0 and (close_from_high / bar_range) > 0.7:
                # Check if actually reversing (current price below close)
                if current_price < last_bar['close'] * 0.999:
                    return {
                        'failed_breakout': True,
                        'severity': 'high',
                        'reason': f"⚠️ FAILED BREAKOUT: Hit target but closed in bottom 70% of range, "
                                 f"now reversing → Exit NOW"
                    }
                else:
                    return {
                        'failed_breakout': True,
                        'severity': 'medium',
                        'reason': f"Possible failed breakout: Closed near lows after target"
                    }
        
        else:  # SHORT
            # For SHORT: Failed breakout = hit low then closed near high
            bar_range = last_bar['high'] - last_bar['low']
            close_from_low = last_bar['close'] - last_bar['low']
            
            # If closed in top 30% of range after hitting target
            if bar_range > 0 and (close_from_low / bar_range) > 0.7:
                # Check if actually reversing
                if current_price > last_bar['close'] * 1.001:
                    return {
                        'failed_breakout': True,
                        'severity': 'high',
                        'reason': f"⚠️ FAILED BREAKOUT: Hit target but closed in top 70% of range, "
                                 f"now reversing → Exit NOW"
                    }
                else:
                    return {
                        'failed_breakout': True,
                        'severity': 'medium',
                        'reason': f"Possible failed breakout: Closed near highs after target"
                    }
        
        return {
            'failed_breakout': False,
            'severity': 'low',
            'reason': 'No failed breakout detected'
        }
    
    def track_mae_mfe(self, trade_id: str, entry_price: float, current_price: float, 
                     direction: str, position_size: int):
        """
        Track Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE).
        This data helps learn optimal exit timing.
        
        Args:
            trade_id: Unique trade identifier
            entry_price: Entry price
            current_price: Current price
            direction: 'long' or 'short'
            position_size: Number of contracts
        """
        # Initialize tracking if new trade
        if trade_id not in self.active_trades_mae_mfe:
            self.active_trades_mae_mfe[trade_id] = {
                'mae': 0.0,  # Max drawdown
                'mfe': 0.0,  # Max profit
                'entry_price': entry_price,
                'direction': direction,
                'position_size': position_size
            }
        
        trade_data = self.active_trades_mae_mfe[trade_id]
        
        # Calculate current P&L
        if direction == 'long':
            pnl = current_price - entry_price
        else:  # short
            pnl = entry_price - current_price
        
        # Update MAE (max adverse - worst drawdown)
        if pnl < trade_data['mae']:
            trade_data['mae'] = pnl
        
        # Update MFE (max favorable - best profit)
        if pnl > trade_data['mfe']:
            trade_data['mfe'] = pnl
    
    def get_mae_mfe_stats(self, trade_id: str) -> Dict:
        """
        Get MAE/MFE statistics for a trade.
        
        Returns:
            Dict with mae, mfe, mae_pct, mfe_pct, efficiency_ratio
        """
        if trade_id not in self.active_trades_mae_mfe:
            return {
                'mae': 0.0,
                'mfe': 0.0,
                'mae_pct': 0.0,
                'mfe_pct': 0.0,
                'efficiency_ratio': 0.0
            }
        
        trade_data = self.active_trades_mae_mfe[trade_id]
        entry_price = trade_data['entry_price']
        
        # Calculate percentages
        mae_pct = (trade_data['mae'] / entry_price) * 100 if entry_price > 0 else 0
        mfe_pct = (trade_data['mfe'] / entry_price) * 100 if entry_price > 0 else 0
        
        # Efficiency ratio: How much of max profit did we capture?
        # 1.0 = exited at peak, 0.0 = gave it all back
        efficiency = 0.0
        if trade_data['mfe'] > 0:
            # This will be calculated when trade exits
            # efficiency = (exit_pnl / mfe)
            efficiency = 0.0  # Placeholder until exit
        
        return {
            'mae': trade_data['mae'],
            'mfe': trade_data['mfe'],
            'mae_pct': mae_pct,
            'mfe_pct': mfe_pct,
            'efficiency_ratio': efficiency
        }
    
    def analyze_mae_mfe_patterns(self):
        """
        Analyze MAE/MFE patterns from past trades to learn optimal exit timing.
        Helps answer:
        - How much profit do we typically give back?
        - At what point should we lock in profits?
        - What's typical MAE before a winner?
        """
        if len(self.exit_experiences) < 20:
            return
        
        # Separate winners and losers
        winners = [exp for exp in self.exit_experiences if exp['outcome'].get('win', False)]
        losers = [exp for exp in self.exit_experiences if not exp['outcome'].get('win', False)]
        
        if len(winners) >= 10:
            # Calculate average MFE for winners
            avg_mfe = statistics.mean([exp.get('mfe', 0) for exp in winners if 'mfe' in exp])
            avg_exit_pnl = statistics.mean([exp['outcome']['pnl'] for exp in winners])
            
            # Efficiency: How much of peak profit do we capture?
            if avg_mfe > 0:
                efficiency = (avg_exit_pnl / avg_mfe) * 100
                logger.info(f"📊 [MAE/MFE ANALYSIS] Winners: Avg MFE ${avg_mfe:.2f}, "
                           f"Avg Exit ${avg_exit_pnl:.2f} → {efficiency:.0f}% efficiency")
                
                if efficiency < 60:
                    logger.warning(f"⚠️ [EXIT TIMING] Only capturing {efficiency:.0f}% of peak profit! "
                                  f"Consider tighter trailing stops.")
        
        if len(losers) >= 10:
            # Typical MAE before stop-out
            avg_mae = statistics.mean([abs(exp.get('mae', 0)) for exp in losers if 'mae' in exp])
            logger.info(f"📊 [MAE/MFE ANALYSIS] Losers: Avg MAE ${avg_mae:.2f}")
    
    def check_profit_velocity(self, entry_time, current_time, r_multiple: float, 
                              peak_r: float) -> dict:
        """
        Detect rapid profit moves that should be protected with tighter management.
        
        Args:
            entry_time: When trade entered
            current_time: Current time
            r_multiple: Current R-multiple
            peak_r: Peak R-multiple achieved
            
        Returns: dict with velocity_detected, recommended_action
        
        Examples:
        - Hit 5R in 10 minutes → Tighten trail to lock profit
        - Hit 3R in 5 minutes → Move to breakeven immediately
        """
        from datetime import timedelta
        
        duration = current_time - entry_time
        duration_minutes = duration.total_seconds() / 60
        
        result = {
            'velocity_detected': False,
            'recommended_action': None,
            'reason': ''
        }
        
        # Avoid division by zero
        if duration_minutes < 1:
            return result
        
        # Calculate R per minute
        r_per_minute = r_multiple / duration_minutes
        
        # FAST PROFIT: 5R in < 10 minutes (0.5 R/min)
        if r_multiple >= 5.0 and duration_minutes < 10:
            result['velocity_detected'] = True
            result['recommended_action'] = 'tighten_trail'
            result['reason'] = f"🚀 Hit {r_multiple:.1f}R in {duration_minutes:.0f} min ({r_per_minute:.2f} R/min) - LOCK PROFIT"
            return result
        
        # MEDIUM VELOCITY: 3R in < 5 minutes (0.6 R/min)
        if r_multiple >= 3.0 and duration_minutes < 5:
            result['velocity_detected'] = True
            result['recommended_action'] = 'move_to_breakeven'
            result['reason'] = f"⚡ Hit {r_multiple:.1f}R in {duration_minutes:.0f} min - protect profit"
            return result
        
        # EXTREME VELOCITY: 2R in < 2 minutes (1.0 R/min)
        if r_multiple >= 2.0 and duration_minutes < 2:
            result['velocity_detected'] = True
            result['recommended_action'] = 'tighten_trail'
            result['reason'] = f"💨 Hit {r_multiple:.1f}R in {duration_minutes:.0f} min (parabolic) - LOCK IT"
            return result
        
        return result
    
    
    def should_skip_trade(self, symbol: str, symbol_state: dict) -> tuple:
        """
        Determine if trade should be skipped based on learned patterns.
        
        Args:
            symbol: Trading symbol
            symbol_state: Current state dict for symbol with bars_1min, etc.
            
        Returns: (should_skip: bool, reason: str)
        """
        # Extract current market conditions
        try:
            # Get hour (UTC)
            from datetime import datetime
            import pytz
            hour = datetime.now(pytz.UTC).hour
            
            # Get RSI if available
            rsi = symbol_state.get('rsi', 50)
            
            # Calculate volume ratio if bars available
            volume_ratio = 1.0
            bars = symbol_state.get('bars_1min', [])
            if len(bars) >= 20:
                recent_vol = sum(b['volume'] for b in bars[-5:]) / 5
                avg_vol = sum(b['volume'] for b in bars[-20:]) / 20
                if avg_vol > 0:
                    volume_ratio = recent_vol / avg_vol
        except:
            # If we can't extract conditions, don't skip
            return (False, "")
        
        # Check for low-probability market conditions
        # Choppy + low volume = typically poor performance
        if 45 <= rsi <= 55 and volume_ratio < 0.8:
            # Check if bot has learned this is bad
            choppy_low_vol = [exp for exp in self.exit_experiences[-100:] 
                            if 45 <= exp.get('market_state', {}).get('rsi', 50) <= 55
                            and exp.get('market_state', {}).get('volume_ratio', 1.0) < 0.8]
            
            if len(choppy_low_vol) >= 10:
                avg_pnl = sum(e['outcome']['pnl'] for e in choppy_low_vol) / len(choppy_low_vol)
                if avg_pnl < -20:  # Learned this loses money
                    return (True, f"SKIP: Choppy + low volume = ${avg_pnl:.0f} avg (learned from {len(choppy_low_vol)} trades)")
        
        return (False, "")
    
    def should_exit_early(self, position: dict, current_market: dict, current_price: float) -> tuple:
        """
        Determine if we should exit entire position early based on learned deterioration patterns.
        
        Args:
            position: Current position dict with entry_price, side, time_in_trade_minutes, etc.
            current_market: Current market state (rsi, volume_ratio, regime, etc.)
            current_price: Current market price
            
        Returns: (should_exit: bool, reason: str)
        """
        if len(self.exit_experiences) < 50:
            return (False, "")  # Need enough data
        
        # Get position details
        side = position.get('side', 'long')
        entry_price = position.get('entry_price', current_price)
        time_in_trade = position.get('time_in_trade_minutes', 0)
        regime = current_market.get('regime', 'NORMAL')
        rsi = current_market.get('rsi', 50)
        
        # Calculate current P&L in ticks
        tick_size = position.get('tick_size', 0.25)
        if side == 'long':
            unrealized_ticks = (current_price - entry_price) / tick_size
        else:
            unrealized_ticks = (entry_price - current_price) / tick_size
        
        # Find similar situations that became losses
        similar_bad_exits = []
        for exp in self.exit_experiences:
            exp_outcome = exp.get('outcome', {})
            exp_market = exp.get('market_state', {})
            
            # Only look at losses
            if exp_outcome.get('pnl', 0) >= 0:
                continue
            
            # Match regime
            if exp.get('regime', '') != regime:
                continue
            
            # Similar RSI
            rsi_diff = abs(exp_market.get('rsi', 50) - rsi)
            if rsi_diff > 15:
                continue
            
            # Similar time in trade
            exp_duration = exp_outcome.get('duration', 0)
            if abs(exp_duration - time_in_trade) > 30:
                continue
            
            similar_bad_exits.append(exp_outcome)
        
        # If 10+ similar situations lost, exit early
        if len(similar_bad_exits) >= 10:
            avg_loss = sum(e.get('pnl', 0) for e in similar_bad_exits) / len(similar_bad_exits)
            
            if unrealized_ticks < 15:  # Not deeply profitable
                reason = f"Early exit - {len(similar_bad_exits)} similar → ${avg_loss:.0f} avg loss"
                logger.warning(f"[EXIT RL] {reason}")
                return (True, reason)
        
        return (False, "")
    
    def should_hold_runner(self, position: dict, current_market: dict, r_multiple: float) -> bool:
        """
        Determine if runner should be held longer based on learned patterns.
        
        Args:
            position: Position dict
            current_market: Current market state
            r_multiple: Current R-multiple
            
        Returns: True to hold runner, False to exit
        """
        if len(self.exit_experiences) < 30:
            return False
        
        regime = current_market.get('regime', 'NORMAL')
        
        # Find big winners in this regime
        big_winners = []
        for exp in self.exit_experiences:
            if exp.get('regime', '') != regime:
                continue
            
            outcome = exp.get('outcome', {})
            achieved_r = outcome.get('r_multiple', 0)
            
            if achieved_r > 5.0:
                big_winners.append(achieved_r)
        
        # If 15+ examples of 5R+ winners
        if len(big_winners) >= 15:
            avg_big_r = sum(big_winners) / len(big_winners)
            
            # Hold if below 80% of average big winner
            if r_multiple < avg_big_r * 0.8:
                logger.info(f"[EXIT RL] Hold runner - {len(big_winners)} similar averaged {avg_big_r:.1f}R")
                return True
        
        return False
    
    
    def save_experiences(self):
        """
        Save exit experiences to v2 format (shared with backtest) AND cloud API.
        Priority: v2 local file (for dev/training) + cloud (for production users).
        """
        
        # ALWAYS save to v2 format for backtest compatibility (DEV MODE)
        try:
            v2_dir = "data/local_experiences"
            os.makedirs(v2_dir, exist_ok=True)
            v2_file = os.path.join(v2_dir, "exit_experiences_v2.json")
            
            with open(v2_file, 'w') as f:
                json.dump({
                    'experiences': self.exit_experiences,
                    'metadata': {
                        'total_experiences': len(self.exit_experiences),
                        'last_updated': datetime.now().isoformat(),
                        'source': 'live_trading'
                    }
                }, f, indent=2)
            
            logger.info(f"✅ Saved {len(self.exit_experiences)} exit experiences to v2 format (shared with backtest)")
        except Exception as e:
            logger.error(f"Failed to save v2 exit experiences: {e}")
        
        # ALSO save to cloud if configured (for production users)
        if self.use_cloud:
            logger.info(f"[CLOUD] ✅ {len(self.exit_experiences)} exit experiences saved to cloud API individually")
        else:
            logger.debug(f"[INFO] Cloud API not configured - using local v2 format only (dev mode)")


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


def get_recommended_scaling_strategy(market_state: Dict, regime: str, adaptive_manager: Optional['AdaptiveExitManager'] = None) -> Dict:
    """
    Recommend partial exit scaling strategy based on LEARNED PATTERNS from past experiences.
    
    RETRIEVES similar past exits and uses their outcomes to decide:
    - WHEN to scale (which R-multiples)
    - HOW MUCH to scale (percentages)
    - WHETHER to scale at all (hold full position)
    
    Returns dict with:
        - partial_1_r: First partial R-multiple (e.g., 2.0)
        - partial_1_pct: First partial percentage (0.0-1.0)
        - partial_2_r: Second partial R-multiple (e.g., 3.0)
        - partial_2_pct: Second partial percentage (0.0-1.0)
        - partial_3_r: Third partial R-multiple (e.g., 5.0)
        - partial_3_pct: Third partial percentage (0.0-1.0)
        - strategy: Description of chosen strategy
        - similar_count: Number of similar past experiences found
        - avg_pnl: Average P&L from similar experiences
    """
    rsi = market_state.get('rsi', 50)
    volume_ratio = market_state.get('volume_ratio', 1.0)
    hour = market_state.get('hour', 12)
    streak = market_state.get('streak', 0)
    vwap_distance = market_state.get('vwap_distance', 0.0)
    
    # STEP 1: Try to learn from similar past experiences
    if adaptive_manager and hasattr(adaptive_manager, 'exit_experiences') and len(adaptive_manager.exit_experiences) >= 20:
        similar_exits = _find_similar_exit_experiences(
            market_state=market_state,
            regime=regime,
            all_experiences=adaptive_manager.exit_experiences,
            min_similarity=0.60  # 60% similarity threshold
        )
        
        if len(similar_exits) >= 5:
            # LEARN: What scaling worked best in similar situations?
            strategy = _extract_scaling_strategy_from_experiences(similar_exits, market_state, regime)
            strategy['similar_count'] = len(similar_exits)
            strategy['learning_mode'] = 'EXPERIENCE_BASED'
            return strategy
    
    # STEP 2: Fallback to rule-based strategy (for cold start or low similarity)
    # These rules will be overridden as bot learns from experience
    
    # AGGRESSIVE SCALING: Get out quick (choppy/overbought/afternoon)
    if (rsi > 65 and "CHOPPY" in regime) or (13 <= hour <= 15 and "CHOPPY" in regime):
        return {
            'partial_1_r': 2.0,
            'partial_1_pct': 0.70,  # 70% @ 2R (secure most profit)
            'partial_2_r': 3.0,
            'partial_2_pct': 0.25,  # 25% @ 3R
            'partial_3_r': 5.0,
            'partial_3_pct': 0.05,  # 5% runner
            'strategy': f'AGGRESSIVE_SCALE (RSI:{rsi:.0f}, {regime}, Hour:{hour})'
        }
    
    # HOLD FULL POSITION: Let it run (trending + momentum)
    elif ("TRENDING" in regime and volume_ratio > 1.3) or (streak >= 3 and "TRENDING" in regime):
        return {
            'partial_1_r': 3.0,
            'partial_1_pct': 0.0,   # Hold all @ 2R
            'partial_2_r': 4.0,
            'partial_2_pct': 0.30,  # Take some @ 4R
            'partial_3_r': 6.0,
            'partial_3_pct': 0.70,  # Most @ 6R (ride the trend!)
            'strategy': f'HOLD_FULL (TRENDING, Vol:{volume_ratio:.1f}x, Streak:{streak:+d})'
        }
    
    # BALANCED SCALING: Standard approach (normal conditions)
    else:
        return {
            'partial_1_r': 2.0,
            'partial_1_pct': 0.50,  # 50% @ 2R
            'partial_2_r': 3.0,
            'partial_2_pct': 0.30,  # 30% @ 3R
            'partial_3_r': 5.0,
            'partial_3_pct': 0.20,  # 20% @ 5R
            'strategy': f'BALANCED ({regime}, RSI:{rsi:.0f})',
            'similar_count': 0,
            'learning_mode': 'RULE_BASED'
        }


def _find_similar_exit_experiences(market_state: Dict, regime: str, all_experiences: list, min_similarity: float = 0.60) -> list:
    """
    Find past exit experiences that are similar to current market conditions.
    Uses 9-feature similarity scoring like signal RL.
    
    Returns: List of similar experiences sorted by similarity score
    """
    similar = []
    
    for exp in all_experiences:
        exp_market = exp.get('market_state', {})
        exp_regime = exp.get('regime', '')
        
        # Calculate similarity score (0.0 to 1.0)
        score = 0.0
        weights_sum = 0.0
        
        # RSI similarity (25% weight)
        rsi_diff = abs(market_state.get('rsi', 50) - exp_market.get('rsi', 50))
        rsi_similarity = max(0, 1.0 - (rsi_diff / 100.0))  # Normalize by RSI range
        score += rsi_similarity * 0.25
        weights_sum += 0.25
        
        # Volume ratio similarity (15% weight)
        vol_diff = abs(market_state.get('volume_ratio', 1.0) - exp_market.get('volume_ratio', 1.0))
        vol_similarity = max(0, 1.0 - (vol_diff / 2.0))  # Normalize by typical range
        score += vol_similarity * 0.15
        weights_sum += 0.15
        
        # Hour similarity (10% weight)
        hour_diff = abs(market_state.get('hour', 12) - exp_market.get('hour', 12))
        hour_similarity = max(0, 1.0 - (hour_diff / 12.0))  # Normalize by half-day
        score += hour_similarity * 0.10
        weights_sum += 0.10
        
        # VWAP distance similarity (15% weight)
        vwap_diff = abs(market_state.get('vwap_distance', 0) - exp_market.get('vwap_distance', 0))
        vwap_similarity = max(0, 1.0 - (vwap_diff / 2.0))
        score += vwap_similarity * 0.15
        weights_sum += 0.15
        
        # Streak similarity (10% weight)
        streak_diff = abs(market_state.get('streak', 0) - exp_market.get('streak', 0))
        streak_similarity = max(0, 1.0 - (streak_diff / 10.0))
        score += streak_similarity * 0.10
        weights_sum += 0.10
        
        # Regime similarity (25% weight) - exact match or partial
        if exp_regime == regime:
            regime_similarity = 1.0
        elif regime.split('_')[0] in exp_regime:  # Same volatility regime
            regime_similarity = 0.5
        else:
            regime_similarity = 0.0
        score += regime_similarity * 0.25
        weights_sum += 0.25
        
        # Normalize score
        final_score = score / weights_sum if weights_sum > 0 else 0
        
        if final_score >= min_similarity:
            similar.append({
                'experience': exp,
                'similarity': final_score
            })
    
    # Sort by similarity (highest first)
    similar.sort(key=lambda x: x['similarity'], reverse=True)
    
    return similar


def _extract_scaling_strategy_from_experiences(similar_exits: list, market_state: Dict, regime: str) -> Dict:
    """
    Extract optimal scaling strategy from similar past experiences.
    
    Analyzes what actually worked (high P&L, high R-multiple) vs what didn't.
    
    Returns: Recommended scaling strategy dict
    """
    if not similar_exits:
        return None
    
    # Group experiences by their scaling behavior
    aggressive_exits = []  # Took 60%+ @ <= 2.5R
    patient_exits = []     # First partial @ >= 3R OR took < 40%
    
    total_pnl = 0
    total_r = 0
    
    for item in similar_exits:
        exp = item['experience']
        partials = exp.get('partial_exits', [])
        outcome = exp.get('outcome', {})
        pnl = outcome.get('pnl', 0)
        r_mult = outcome.get('r_multiple', 0)
        
        total_pnl += pnl
        total_r += r_mult
        
        if partials:
            first_partial = partials[0]
            first_r = first_partial.get('r_multiple', 0)
            first_pct = first_partial.get('percentage', 0)
            
            if first_r <= 2.5 and first_pct >= 0.60:
                aggressive_exits.append({'pnl': pnl, 'r': r_mult, 'exp': exp})
            elif first_r >= 3.0 or first_pct <= 0.40:
                patient_exits.append({'pnl': pnl, 'r': r_mult, 'exp': exp})
    
    avg_pnl = total_pnl / len(similar_exits)
    avg_r = total_r / len(similar_exits)
    
    # Calculate optimal single-contract exit target from winners
    winning_exits = [item for item in similar_exits if item['experience'].get('outcome', {}).get('pnl', 0) > 0]
    if winning_exits:
        avg_winning_r = sum(item['experience'].get('outcome', {}).get('r_multiple', 0) for item in winning_exits) / len(winning_exits)
        single_contract_target = max(2.0, min(8.0, avg_winning_r))  # Between 2R-8R based on what worked
    else:
        single_contract_target = 3.0  # Conservative default
    
    # Decide strategy: Which performed better in similar conditions?
    if len(aggressive_exits) >= 3 and len(patient_exits) >= 3:
        aggressive_avg_pnl = sum(e['pnl'] for e in aggressive_exits) / len(aggressive_exits)
        patient_avg_pnl = sum(e['pnl'] for e in patient_exits) / len(patient_exits)
        
        if aggressive_avg_pnl > patient_avg_pnl:
            # Aggressive worked better
            return {
                'partial_1_r': 2.0,
                'partial_1_pct': 0.70,
                'partial_2_r': 3.0,
                'partial_2_pct': 0.25,
                'partial_3_r': 5.0,
                'partial_3_pct': 0.05,
                'single_contract_target': single_contract_target,  # Learned from winners
                'strategy': f'LEARNED_AGGRESSIVE (${aggressive_avg_pnl:.0f} avg from {len(aggressive_exits)} similar)',
                'avg_pnl': avg_pnl,
                'avg_r': avg_r
            }
        else:
            # Patient worked better
            return {
                'partial_1_r': 3.0,
                'partial_1_pct': 0.0,
                'partial_2_r': 4.0,
                'partial_2_pct': 0.30,
                'partial_3_r': 6.0,
                'partial_3_pct': 0.70,
                'single_contract_target': single_contract_target,  # Learned from winners
                'strategy': f'LEARNED_PATIENT (${patient_avg_pnl:.0f} avg from {len(patient_exits)} similar)',
                'avg_pnl': avg_pnl,
                'avg_r': avg_r
            }
    
    # Default: Use average R-multiple from similar exits to guide strategy
    if avg_r > 3.5:
        # Similar situations led to big wins → be patient
        return {
            'partial_1_r': 3.0,
            'partial_1_pct': 0.0,
            'partial_2_r': 4.0,
            'partial_2_pct': 0.40,
            'partial_3_r': 6.0,
            'partial_3_pct': 0.60,
            'single_contract_target': single_contract_target,
            'strategy': f'LEARNED_HOLD ({avg_r:.1f}R avg from {len(similar_exits)} similar)',
            'avg_pnl': avg_pnl,
            'avg_r': avg_r
        }
    elif avg_r < 2.5:
        # Similar situations led to quick exits → be aggressive
        return {
            'partial_1_r': 2.0,
            'partial_1_pct': 0.80,
            'partial_2_r': 3.0,
            'partial_2_pct': 0.20,
            'partial_3_r': 5.0,
            'partial_3_pct': 0.0,
            'single_contract_target': single_contract_target,
            'strategy': f'LEARNED_QUICK ({avg_r:.1f}R avg from {len(similar_exits)} similar)',
            'avg_pnl': avg_pnl,
            'avg_r': avg_r
        }
    else:
        # Balanced approach from similar situations
        return {
            'partial_1_r': 2.0,
            'partial_1_pct': 0.50,
            'partial_2_r': 3.0,
            'partial_2_pct': 0.30,
            'partial_3_r': 5.0,
            'partial_3_pct': 0.20,
            'single_contract_target': single_contract_target,
            'strategy': f'LEARNED_BALANCED ({avg_r:.1f}R avg from {len(similar_exits)} similar)',
            'avg_pnl': avg_pnl,
            'avg_r': avg_r
        }


def get_adaptive_exit_params(bars: list, position: Dict, current_price: float, 
                             config: Dict, adaptive_manager: Optional[AdaptiveExitManager] = None,
                             entry_confidence: float = 0.75) -> Dict:
    """
    Calculate adaptive exit parameters based on current market conditions.
    
    Args:
        bars: Recent 1-min bars (can be DataFrame or list)
        position: Current position state
        current_price: Current market price
        config: Bot configuration
        adaptive_manager: Optional manager instance for state persistence
        entry_confidence: Confidence score from Signal RL (0.0-1.0) - affects exit tightness
    
    Returns:
        Dict with adaptive parameters:
        - breakeven_threshold_ticks: When to move to breakeven
        - breakeven_offset_ticks: Where to place breakeven stop
        - trailing_distance_ticks: Trailing stop distance
        - trailing_min_profit_ticks: Min profit before trailing activates
        - market_regime: Detected regime
        - current_volatility_atr: Current ATR
        - is_aggressive_mode: Whether in aggressive profit-taking mode
        - confidence_adjusted: Whether params were tightened due to low confidence
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
    
    # ========================================================================
    # NEURAL NETWORK EXIT PREDICTION - Cloud API (protected model)
    # ========================================================================
    if adaptive_manager and hasattr(adaptive_manager, 'use_cloud') and adaptive_manager.use_cloud and adaptive_manager.cloud_api_url:
        try:
            import numpy as np
            
            # Extract ALL 45 features for cloud neural network API
            latest_bar = bars[-1] if len(bars) > 0 else {}
            entry_time = position.get('entry_time', datetime.now())
            if not isinstance(entry_time, datetime):
                entry_time = datetime.now()
            duration_bars = position.get('duration_bars', 1)
            
            # Market Context (8 features)
            regime_map = {'NORMAL': 0, 'NORMAL_TRENDING': 1, 'HIGH_VOL_TRENDING': 2, 
                          'HIGH_VOL_CHOPPY': 3, 'LOW_VOL_TRENDING': 4, 'LOW_VOL_RANGING': 5, 'UNKNOWN': 0}
            market_regime_enc = regime_map.get(market_regime, 0) / 5.0
            rsi = latest_bar.get('rsi', 50.0) / 100.0
            volume_ratio = np.clip(latest_bar.get('volume', 1.0) / latest_bar.get('avg_volume', 1.0) if 'avg_volume' in latest_bar else 1.0, 0, 3) / 3.0
            atr_norm = np.clip(current_atr / 10.0, 0, 1)
            vix = np.clip(latest_bar.get('vix', 15.0) / 40.0, 0, 1)
            volatility_regime_change = 1.0 if position.get('volatility_regime_change', False) else 0.0
            volume_at_exit = np.clip(latest_bar.get('volume', 1.0) / latest_bar.get('avg_volume', 1.0) if 'avg_volume' in latest_bar else 1.0, 0, 3) / 3.0
            market_state_enc = 0.5
            
            # Trade Context (5 features) - removed bid_ask_spread and slippage (live-only, not in backtest)
            entry_conf = entry_confidence
            side = 1.0 if position.get('side', 'long').lower() == 'short' else 0.0
            session = latest_bar.get('session', 0) / 2.0
            commission = np.clip(2.0 / 10.0, 0, 1)
            regime_enc = market_regime_enc
            
            # Time Features (5 features)
            hour = latest_bar.get('hour', 12) / 24.0
            day_of_week = latest_bar.get('day_of_week', 2) / 6.0
            duration = np.clip(duration_bars / 500.0, 0, 1)
            time_in_breakeven = np.clip(position.get('time_in_breakeven_bars', 0) / 100.0, 0, 1)
            bars_until_breakeven = np.clip(position.get('bars_until_breakeven', 999) / 100.0, 0, 1)
            
            # Performance Metrics (5 features) - calculable in real-time
            entry_price = position.get('entry_price', current_price)
            tick_size = config.get('tick_size', 0.25)
            current_pnl = (current_price - entry_price) * position.get('quantity', 1) * (50 if 'MES' in config.get('symbol', 'MES') else 50) * (1 if position.get('side', 'long').lower() == 'long' else -1)
            mae = np.clip(position.get('mae', 0) / 1000.0, -1, 0)
            mfe = np.clip(position.get('mfe', 0) / 2000.0, 0, 1)
            risk = abs(entry_price - position.get('stop_price', entry_price - current_atr)) * position.get('quantity', 1) * 50
            max_r = np.clip(position.get('max_r_achieved', 0) / 10.0, 0, 1)
            min_r = np.clip(position.get('min_r_achieved', 0) / 5.0, -1, 1)
            r_multiple = np.clip((current_pnl / risk if risk > 0 else 0) / 10.0, -1, 1)
            
            # Exit Strategy State (6 features) - tracking state
            breakeven_activated = 1.0 if position.get('breakeven_activated', False) else 0.0
            trailing_activated = 1.0 if position.get('trailing_activated', False) else 0.0
            stop_hit = 0.0  # Not hit yet
            exit_param_updates = np.clip(position.get('exit_param_update_count', 0) / 50.0, 0, 1)
            stop_adjustments = np.clip(position.get('stop_adjustment_count', 0) / 20.0, 0, 1)
            bars_until_trailing = np.clip(position.get('bars_until_trailing', 999) / 100.0, 0, 1)
            
            # Results (5 features) - use CURRENT values (not final)
            pnl_norm = np.clip(current_pnl / 2000.0, -1, 1)
            outcome_current = 1.0 if current_pnl > 0 else 0.0
            win_current = 1.0 if current_pnl > 0 else 0.0
            exit_reason = 0.0  # Unknown yet
            max_profit = np.clip(position.get('mfe', 0) / 2000.0, 0, 1)
            
            # ADVANCED (8 features)
            entry_atr = position.get('entry_atr', current_atr)
            atr_change_pct = np.clip((current_atr - entry_atr) / entry_atr * 100.0 / 100.0 if entry_atr > 0 else 0.0, -1, 1)
            avg_atr_trade = np.clip(position.get('avg_atr_during_trade', current_atr) / 10.0, 0, 1)
            peak_r = np.clip(position.get('peak_r_multiple', max_r * 10.0) / 10.0, 0, 1)
            profit_dd = np.clip(position.get('profit_drawdown_from_peak', 0) / 2000.0, 0, 1)
            high_vol_bars = np.clip(position.get('high_volatility_bars', 0) / 100.0, 0, 1)
            recent_wins = np.clip(position.get('wins_in_last_5_trades', 0) / 5.0, 0, 1)
            recent_losses = np.clip(position.get('losses_in_last_5_trades', 0) / 5.0, 0, 1)
            from datetime import datetime
            current_time = datetime.now()
            close_time = current_time.replace(hour=16, minute=0, second=0)
            mins_to_close = np.clip((close_time - current_time).total_seconds() / 60 / 480.0, 0, 1)
            
            # Build request payload for cloud neural network API
            api_request = {
                "market_regime": market_regime,
                "rsi": latest_bar.get('rsi', 50.0),
                "volume_ratio": latest_bar.get('volume', 1.0) / latest_bar.get('avg_volume', 1.0) if 'avg_volume' in latest_bar else 1.0,
                "atr": current_atr,
                "vix": latest_bar.get('vix', 15.0),
                "volatility_regime_change": position.get('volatility_regime_change', False),
                "entry_confidence": entry_confidence,
                "side": position.get('side', 'long'),
                "session": latest_bar.get('session', 0),
                "hour": latest_bar.get('hour', 12),
                "day_of_week": latest_bar.get('day_of_week', 2),
                "duration_bars": duration_bars,
                "time_in_breakeven_bars": position.get('time_in_breakeven_bars', 0),
                "bars_until_breakeven": position.get('bars_until_breakeven', 999),
                "mae": position.get('mae', 0),
                "mfe": position.get('mfe', 0),
                "max_r_achieved": position.get('max_r_achieved', 0),
                "min_r_achieved": position.get('min_r_achieved', 0),
                "r_multiple": (current_pnl / risk if risk > 0 else 0),
                "breakeven_activated": position.get('breakeven_activated', False),
                "trailing_activated": position.get('trailing_activated', False),
                "exit_param_update_count": position.get('exit_param_update_count', 0),
                "stop_adjustment_count": position.get('stop_adjustment_count', 0),
                "bars_until_trailing": position.get('bars_until_trailing', 999),
                "current_pnl": current_pnl,
                "entry_atr": entry_atr,
                "avg_atr_during_trade": position.get('avg_atr_during_trade', current_atr),
                "profit_drawdown_from_peak": position.get('profit_drawdown_from_peak', 0),
                "high_volatility_bars": position.get('high_volatility_bars', 0),
                "wins_in_last_5_trades": position.get('wins_in_last_5_trades', 0),
                "losses_in_last_5_trades": position.get('losses_in_last_5_trades', 0),
                "minutes_to_close": (close_time - current_time).total_seconds() / 60
            }
            
            # Call cloud neural network API
            import requests
            api_url = f"{adaptive_manager.cloud_api_url}/api/ml/predict_exit_params"
            response = requests.post(api_url, json=api_request, timeout=2.0)  # 2 second timeout
            
            if response.status_code == 200:
                api_response = response.json()
                if api_response.get('success'):
                    exit_params_dict = api_response.get('exit_params', {})
                    
                    logger.info(f"🧠 ☁️  [CLOUD EXIT NN] Predicted params: breakeven={exit_params_dict['breakeven_threshold_ticks']:.1f}, "
                               f"trailing={exit_params_dict['trailing_distance_ticks']:.1f}, "
                               f"stop_mult={exit_params_dict['stop_mult']:.2f}x, "
                               f"partials={exit_params_dict['partial_1_r']:.1f}R/{exit_params_dict['partial_2_r']:.1f}R/{exit_params_dict['partial_3_r']:.1f}R, "
                               f"RL_BE={exit_params_dict.get('should_activate_breakeven', 0.8):.2f}, "
                               f"RL_TRAIL={exit_params_dict.get('should_activate_trailing', 0.8):.2f}, "
                               f"EXIT_NOW={exit_params_dict.get('should_exit_now', 0.0):.2f} "
                               f"({api_response.get('prediction_time_ms', 0):.1f}ms)")
                    
                    # Convert to final format - ALL 130 parameters from neural network
                    result = {
                        'breakeven_threshold_ticks': int(exit_params_dict['breakeven_threshold_ticks']),
                        'breakeven_offset_ticks': 1,  # Standard offset
                        'trailing_distance_ticks': int(exit_params_dict['trailing_distance_ticks']),
                        'trailing_min_profit_ticks': int(exit_params_dict['breakeven_threshold_ticks'] * 1.5),
                        'market_regime': market_regime,
                        'current_volatility_atr': current_atr,
                        'is_aggressive_mode': entry_confidence < 0.6,
                        'confidence_adjusted': entry_confidence < 0.6,
                        'partial_1_r': exit_params_dict['partial_1_r'],
                        'partial_1_pct': 0.50,
                        'partial_2_r': exit_params_dict['partial_2_r'],
                        'partial_2_pct': 0.30,
                        'partial_3_r': exit_params_dict['partial_3_r'],
                        'partial_3_pct': 0.20,
                        'stop_mult': exit_params_dict['stop_mult'],
                        'prediction_source': 'cloud_neural_network',
                        
                        # RL STRATEGY CONTROL (16 params)
                        'should_activate_breakeven': exit_params_dict.get('should_activate_breakeven', 0.8),
                        'breakeven_activation_profit_threshold': exit_params_dict.get('breakeven_activation_profit_threshold', 10),
                        'breakeven_activation_min_bars': exit_params_dict.get('breakeven_activation_min_bars', 3),
                        'breakeven_activation_r_threshold': exit_params_dict.get('breakeven_activation_r_threshold', 1.0),
                        'should_activate_trailing': exit_params_dict.get('should_activate_trailing', 0.8),
                        'trailing_activation_profit_threshold': exit_params_dict.get('trailing_activation_profit_threshold', 15),
                        'trailing_activation_r_threshold': exit_params_dict.get('trailing_activation_r_threshold', 1.5),
                        'trailing_wait_after_breakeven_bars': exit_params_dict.get('trailing_wait_after_breakeven_bars', 5),
                        'should_adjust_stop': exit_params_dict.get('should_adjust_stop', 0.7),
                        'stop_adjustment_frequency_bars': exit_params_dict.get('stop_adjustment_frequency_bars', 3),
                        'max_stop_adjustments_per_trade': exit_params_dict.get('max_stop_adjustments_per_trade', 10),
                        'should_update_exit_params': exit_params_dict.get('should_update_exit_params', 0.6),
                        'exit_param_update_frequency_bars': exit_params_dict.get('exit_param_update_frequency_bars', 10),
                        'max_exit_param_updates_per_trade': exit_params_dict.get('max_exit_param_updates_per_trade', 5),
                        'exit_strategy_aggressiveness': exit_params_dict.get('exit_strategy_aggressiveness', 0.5),
                        'dynamic_strategy_adaptation_rate': exit_params_dict.get('dynamic_strategy_adaptation_rate', 0.3),
                        
                        # IMMEDIATE ACTIONS (4 params)
                        'should_exit_now': exit_params_dict.get('should_exit_now', 0.0),
                        'should_take_partial_1': exit_params_dict.get('should_take_partial_1', 0.0),
                        'should_take_partial_2': exit_params_dict.get('should_take_partial_2', 0.0),
                        'should_take_partial_3': exit_params_dict.get('should_take_partial_3', 0.0),
                        
                        # RUNNER MANAGEMENT (2 params)
                        'runner_percentage': exit_params_dict.get('runner_percentage', 0.25),
                        'runner_target_r': exit_params_dict.get('runner_target_r', 5.0),
                        
                        # TIME-BASED (2 params)
                        'time_stop_max_bars': exit_params_dict.get('time_stop_max_bars', 60.0),
                        'time_decay_rate': exit_params_dict.get('time_decay_rate', 0.5),
                        
                        # ADVERSE CONDITIONS (2 params)
                        'regime_change_immediate_exit': exit_params_dict.get('regime_change_immediate_exit', 0.0),
                        'failed_breakout_exit_speed': exit_params_dict.get('failed_breakout_exit_speed', 0.5),
                        
                        # DEAD TRADE (6 params)
                        'should_exit_dead_trade': exit_params_dict.get('should_exit_dead_trade', 0.0),
                        'dead_trade_max_loss_ticks': exit_params_dict.get('dead_trade_max_loss_ticks', 8.0),
                        'dead_trade_max_loss_r': exit_params_dict.get('dead_trade_max_loss_r', 1.0),
                        'dead_trade_detection_bars': exit_params_dict.get('dead_trade_detection_bars', 10.0),
                        'dead_trade_acceptable_loss_pct': exit_params_dict.get('dead_trade_acceptable_loss_pct', 0.5),
                        'dead_trade_early_cut_enabled': exit_params_dict.get('dead_trade_early_cut_enabled', 1.0),
                        
                        # SIDEWAYS MARKETS (8 params)
                        'sideways_market_exit_enabled': exit_params_dict.get('sideways_market_exit_enabled', 1.0),
                        'sideways_detection_range_pct': exit_params_dict.get('sideways_detection_range_pct', 0.005),
                        'sideways_detection_bars': exit_params_dict.get('sideways_detection_bars', 20.0),
                        'sideways_max_loss_r': exit_params_dict.get('sideways_max_loss_r', 0.5),
                        'sideways_stop_tightening_mult': exit_params_dict.get('sideways_stop_tightening_mult', 0.6),
                        'sideways_exit_aggressiveness': exit_params_dict.get('sideways_exit_aggressiveness', 0.7),
                        'sideways_avoid_new_entry': exit_params_dict.get('sideways_avoid_new_entry', 1.0),
                        'sideways_breakout_confirmation': exit_params_dict.get('sideways_breakout_confirmation', 3.0),
                        
                        # PROFIT PROTECTION (2 params)
                        'profit_lock_activation_r': exit_params_dict.get('profit_lock_activation_r', 2.0),
                        'profit_protection_aggressiveness': exit_params_dict.get('profit_protection_aggressiveness', 0.5),
                        
                        # VOLATILITY RESPONSE (1 param)
                        'volatility_spike_adaptive_exit': exit_params_dict.get('volatility_spike_adaptive_exit', 2.5),
                        
                        # FALSE BREAKOUT RECOVERY (1 param)
                        'false_breakout_recovery_enabled': exit_params_dict.get('false_breakout_recovery_enabled', 0.0),
                        
                        # ACCOUNT PROTECTION (4 params)
                        'consecutive_loss_emergency_exit': exit_params_dict.get('consecutive_loss_emergency_exit', 5.0),
                        'drawdown_tightening_threshold': exit_params_dict.get('drawdown_tightening_threshold', 0.10),
                        'drawdown_exit_aggressiveness': exit_params_dict.get('drawdown_exit_aggressiveness', 0.5),
                        'recovery_mode_sensitivity': exit_params_dict.get('recovery_mode_sensitivity', 0.7),
                        
                        # LOSS ACCEPTANCE (3 params)
                        'acceptable_loss_for_bad_entry': exit_params_dict.get('acceptable_loss_for_bad_entry', 0.5),
                        'acceptable_loss_for_good_entry': exit_params_dict.get('acceptable_loss_for_good_entry', 2.0),
                        'entry_quality_threshold': exit_params_dict.get('entry_quality_threshold', 0.7),
                    }
                    return result
                else:
                    logger.warning(f"⚠️  Cloud exit NN returned error: {api_response.get('error', 'unknown')}")
            else:
                logger.warning(f"⚠️  Cloud exit NN API returned status {response.status_code}")
            
        except Exception as e:
            logger.warning(f"⚠️  Cloud exit neural network prediction failed: {e}, falling back to pattern matching")
            # Fall through to cloud/pattern matching below
    
    # ========================================================================
    # CLOUD EXIT RL - Query cloud for REAL-TIME exit recommendations
    # ========================================================================
    # Try to get cloud-recommended exit params based on current market conditions
    cloud_params = None
    if adaptive_manager and hasattr(adaptive_manager, 'get_cloud_exit_params'):
        # Build market state dict for cloud query
        market_state_for_cloud = {}
        if len(bars) > 0:
            latest_bar = bars[-1]
            market_state_for_cloud = {
                'rsi': latest_bar.get('rsi', 50.0),
                'atr': current_atr,
                'vwap_distance': abs(current_price - latest_bar.get('vwap', current_price)) if 'vwap' in latest_bar else 0.0,
                'volume_ratio': latest_bar.get('volume', 1.0) / latest_bar.get('avg_volume', 1.0) if 'avg_volume' in latest_bar else 1.0,
                'hour': current_hour,
                'day_of_week': latest_bar.get('day_of_week', 0) if 'day_of_week' in latest_bar else 0,
                'streak': position.get('streak', 0) if 'streak' in position else 0,
                'recent_pnl': position.get('recent_pnl', 0.0) if 'recent_pnl' in position else 0.0,
                'vix': latest_bar.get('vix', 15.0) if 'vix' in latest_bar else 15.0
            }
        
        # Build position state for cloud query
        position_for_cloud = {
            'side': position.get('side', 'long'),
            'duration_minutes': duration_minutes,
            'unrealized_pnl': position.get('unrealized_pnl', 0.0) if 'unrealized_pnl' in position else 0.0,
            'entry_price': position.get('entry_price', current_price),
            'r_multiple': position.get('r_multiple', 0.0) if 'r_multiple' in position else 0.0
        }
        
        # Query cloud for real-time exit params
        cloud_params = adaptive_manager.get_cloud_exit_params(
            regime=market_regime,
            market_state=market_state_for_cloud,
            position=position_for_cloud,
            entry_confidence=entry_confidence
        )
        
        # If cloud returns params, use them directly (they're already optimized)
        if cloud_params:
            logger.info(f"🎯 [CLOUD EXIT RL] Using cloud-recommended params: {cloud_params.get('_cloud_metadata', {}).get('recommendation', 'N/A')}")
            return {
                "breakeven_threshold_ticks": cloud_params.get('breakeven_threshold_ticks', base_breakeven_threshold),
                "breakeven_offset_ticks": cloud_params.get('breakeven_offset_ticks', base_breakeven_offset),
                "trailing_distance_ticks": cloud_params.get('trailing_distance_ticks', base_trailing_distance),
                "trailing_min_profit_ticks": cloud_params.get('partial_1_r', 2.0) * base_trailing_distance,  # Estimate
                "market_regime": market_regime,
                "current_volatility_atr": current_atr,
                "is_aggressive_mode": "TIGHTEN" in cloud_params.get('_cloud_metadata', {}).get('recommendation', ''),
                "confidence_adjusted": False,  # Cloud already factors in confidence
                "cloud_recommendation": cloud_params.get('_cloud_metadata', {})
            }
        else:
            logger.debug(f"[CLOUD EXIT RL] Cloud unavailable, using local learned params")
    
    # If cloud query failed or not available, continue with local learned parameters below
    # ========================================================================
    
    # Calculate trade duration
    entry_time = position.get("entry_time")
    if entry_time and hasattr(entry_time, 'timestamp'):
        from datetime import datetime
        import pytz
        # FIX: Use bar timestamp instead of wall-clock time for backtest accuracy
        if len(bars) > 0 and "timestamp" in bars[-1]:
            current_time = bars[-1]["timestamp"]
            if isinstance(current_time, str):
                current_time = datetime.fromisoformat(current_time.replace('Z', '+00:00'))
        else:
            current_time = datetime.now(pytz.UTC)  # Fallback for live trading (UTC)
        
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
        'is_morning': 9 <= current_hour < 11,  # 9-11 AM UTC volatile
        'is_lunch': 11 <= current_hour < 14,   # 11 AM - 2 PM UTC slow
        'is_close': current_hour >= 19,        # After 7 PM UTC rushes/reversals
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
    # The learned params ARE the optimal values - don't override them with hardcoded logic!
    if adaptive_manager and hasattr(adaptive_manager, 'learned_params'):
        learned = adaptive_manager.learned_params.get(market_regime, {})
        breakeven_threshold_multiplier = learned.get('breakeven_mult', 1.0)
        trailing_distance_multiplier = learned.get('trailing_mult', 1.0)
        
        # EXPLORATION: 30% of the time, use randomized multipliers to discover better settings
        exploration_rate = config.get('exploration_rate', 0.0)  # Default 0 for live trading
        if exploration_rate > 0 and random.random() < exploration_rate:
            # Randomize multipliers within safe ranges (±20% variation)
            breakeven_threshold_multiplier *= random.uniform(0.8, 1.2)
            trailing_distance_multiplier *= random.uniform(0.8, 1.2)
            
            # Clamp to safe bounds (same as learning algorithm)
            breakeven_threshold_multiplier = max(0.6, min(1.3, breakeven_threshold_multiplier))
            trailing_distance_multiplier = max(0.6, min(1.3, trailing_distance_multiplier))
            
            logger.info(f"[EXIT EXPLORATION] Randomized params for {market_regime}: BE={breakeven_threshold_multiplier:.2f}x, Trail={trailing_distance_multiplier:.2f}x")
        else:
            logger.debug(f"[RL LEARNED] {market_regime}: BE={breakeven_threshold_multiplier:.2f}x, Trail={trailing_distance_multiplier:.2f}x")
    else:
        # Fallback: Use regime-specific defaults only when NO learning available
        logger.debug(f"[NO LEARNING] Using fallback defaults for {market_regime}")
        
        if market_regime == "HIGH_VOL_CHOPPY":
            breakeven_threshold_multiplier = 0.75  # Tight BE
            trailing_distance_multiplier = 0.70    # Tight trailing
            trailing_min_profit_multiplier = 0.75
        elif market_regime == "HIGH_VOL_TRENDING":
            breakeven_threshold_multiplier = 0.85
            trailing_distance_multiplier = 1.10   # Room for trends
            trailing_min_profit_multiplier = 0.85
        elif market_regime == "LOW_VOL_RANGING":
            breakeven_threshold_multiplier = 1.0
            trailing_distance_multiplier = 1.0
            trailing_min_profit_multiplier = 1.0
        elif market_regime == "LOW_VOL_TRENDING":
            breakeven_threshold_multiplier = 1.0
            trailing_distance_multiplier = 1.15   # More room
            trailing_min_profit_multiplier = 0.95
        else:
            # NORMAL or unknown regimes
            breakeven_threshold_multiplier = 1.0
            trailing_distance_multiplier = 1.0
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
    
    # ========================================================================
    # CONFIDENCE CORRELATION - Low confidence entries get TIGHTER exits
    # ========================================================================
    confidence_adjusted = False
    if entry_confidence < 0.7:
        # Low confidence entry - tighten everything to limit damage
        confidence_penalty = (0.7 - entry_confidence) / 0.7  # 0.0 to 1.0 scale
        tighten_factor = 1.0 - (confidence_penalty * 0.3)    # Max 30% tighter
        
        breakeven_threshold_multiplier *= tighten_factor
        trailing_distance_multiplier *= tighten_factor
        trailing_min_profit_multiplier *= tighten_factor
        
        confidence_adjusted = True
        logger.info(f"⚠️ [CONFIDENCE CORRELATION] Low entry confidence {entry_confidence:.1%} "
                   f"→ Tightening exits by {(1-tighten_factor)*100:.0f}%")
    elif entry_confidence >= 0.85:
        # High confidence entry - give more room to run
        confidence_bonus = (entry_confidence - 0.85) / 0.15  # 0.0 to 1.0 scale
        loosen_factor = 1.0 + (confidence_bonus * 0.15)      # Max 15% looser
        
        trailing_distance_multiplier *= loosen_factor
        trailing_min_profit_multiplier *= 0.95  # Activate trailing slightly earlier
        
        confidence_adjusted = True
        logger.info(f"✅ [CONFIDENCE CORRELATION] High entry confidence {entry_confidence:.1%} "
                   f"→ Giving {(loosen_factor-1)*100:.0f}% more room")
    
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
    
    if adaptive_manager and breakeven_threshold_multiplier != 1.0:
        logger.info(f"[LEARNED] BE mult={breakeven_threshold_multiplier:.2f}x, Trail mult={trailing_distance_multiplier:.2f}x")
    
    logger.info(f"{'='*70}\n")
    
    return {
        "breakeven_threshold_ticks": adaptive_breakeven_threshold,
        "breakeven_offset_ticks": adaptive_breakeven_offset,
        "trailing_distance_ticks": adaptive_trailing_distance,
        "trailing_min_profit_ticks": adaptive_trailing_min_profit,
        "market_regime": market_regime,
        "current_volatility_atr": current_atr,
        "is_aggressive_mode": is_aggressive_mode,
        "confidence_adjusted": confidence_adjusted,
        "entry_confidence": entry_confidence,
        "situation_factors": situation_factors,
        "decision_reasons": aggression_reasons if aggression_reasons else ["balanced"],
        "duration_minutes": duration_minutes,
        "learned_multiplier": breakeven_threshold_multiplier  # Include learned multiplier for tracking
    }
