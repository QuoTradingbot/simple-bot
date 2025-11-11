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

logger = logging.getLogger(__name__)


class AdaptiveExitManager:
    """
    Manages adaptive exit parameters that adjust to market conditions.
    Maintains state across trades for regime detection.
    LEARNS optimal exit parameters from past outcomes.
    """
    
    def __init__(self, config: Dict, experience_file: str = "data/exit_experience.json", cloud_api_url: Optional[str] = None):
        """
        Initialize adaptive exit manager with RL learning.
        
        Args:
            config: Bot configuration
            experience_file: Local file path for fallback (if cloud unavailable)
            cloud_api_url: Cloud API URL for fetching/saving experiences (e.g., "https://quotrading-signals.icymeadow-86b2969e.eastus.azurecontainerapps.io")
        """
        self.config = config
        self.experience_file = experience_file
        self.cloud_api_url = cloud_api_url  # NEW: Cloud API endpoint
        self.use_cloud = cloud_api_url is not None  # Use cloud if URL provided
        
        # Track recent ATR values for regime detection
        self.recent_atr_values = deque(maxlen=20)
        self.recent_volatility_regime = "NORMAL"
        
        # Track recent trade durations for adaptive timing
        self.recent_trade_durations = deque(maxlen=10)
        
        # RL Learning for exit parameters
        self.exit_experiences = []  # All past exit outcomes
        
        # Learned optimal parameters per regime (updated from experiences)
        # MUST be defined BEFORE load_experiences() since it uses it as default
        # NOW LEARNS: stops, breakeven, trailing, partial exits, sideways timeout
        self.learned_params = {
            'HIGH_VOL_CHOPPY': {
                'breakeven_mult': 0.75, 'trailing_mult': 0.7, 'stop_mult': 4.0,
                'partial_1_r': 2.0, 'partial_1_pct': 0.50,  # 50% @ 2R
                'partial_2_r': 3.0, 'partial_2_pct': 0.30,  # 30% @ 3R
                'partial_3_r': 5.0, 'partial_3_pct': 0.20,  # 20% @ 5R (runner)
                'sideways_timeout_minutes': 15  # Exit runner if sideways 15 min
            },
            'HIGH_VOL_TRENDING': {
                'breakeven_mult': 0.85, 'trailing_mult': 1.1, 'stop_mult': 4.2,
                'partial_1_r': 2.5, 'partial_1_pct': 0.40,  # Let trends run more
                'partial_2_r': 4.0, 'partial_2_pct': 0.30,
                'partial_3_r': 6.0, 'partial_3_pct': 0.30,
                'sideways_timeout_minutes': 20
            },
            'LOW_VOL_RANGING': {
                'breakeven_mult': 1.0, 'trailing_mult': 1.0, 'stop_mult': 3.2,
                'partial_1_r': 1.5, 'partial_1_pct': 0.60,  # Take profits quick in ranges
                'partial_2_r': 2.5, 'partial_2_pct': 0.30,
                'partial_3_r': 4.0, 'partial_3_pct': 0.10,
                'sideways_timeout_minutes': 10
            },
            'LOW_VOL_TRENDING': {
                'breakeven_mult': 1.0, 'trailing_mult': 1.15, 'stop_mult': 3.4,
                'partial_1_r': 2.0, 'partial_1_pct': 0.40,
                'partial_2_r': 3.5, 'partial_2_pct': 0.30,
                'partial_3_r': 5.5, 'partial_3_pct': 0.30,
                'sideways_timeout_minutes': 18
            },
            'NORMAL': {
                'breakeven_mult': 1.0, 'trailing_mult': 1.0, 'stop_mult': 3.6,
                'partial_1_r': 2.0, 'partial_1_pct': 0.50,
                'partial_2_r': 3.0, 'partial_2_pct': 0.30,
                'partial_3_r': 5.0, 'partial_3_pct': 0.20,
                'sideways_timeout_minutes': 12
            },
            'NORMAL_TRENDING': {
                'breakeven_mult': 1.0, 'trailing_mult': 1.1, 'stop_mult': 3.6,
                'partial_1_r': 2.2, 'partial_1_pct': 0.45,
                'partial_2_r': 3.5, 'partial_2_pct': 0.30,
                'partial_3_r': 5.5, 'partial_3_pct': 0.25,
                'sideways_timeout_minutes': 15
            },
            'NORMAL_CHOPPY': {
                'breakeven_mult': 0.95, 'trailing_mult': 0.95, 'stop_mult': 3.4,
                'partial_1_r': 1.8, 'partial_1_pct': 0.55,
                'partial_2_r': 2.8, 'partial_2_pct': 0.30,
                'partial_3_r': 4.5, 'partial_3_pct': 0.15,
                'sideways_timeout_minutes': 10
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
    
    def record_exit_outcome(self, regime: str, exit_params: Dict, trade_outcome: Dict, 
                           market_state: Dict = None, partial_exits: list = None):
        """
        Record exit outcome for RL learning (including scaling decisions).
        
        Args:
            regime: Market regime when exit occurred
            exit_params: Exit parameters used (breakeven_threshold, trailing_distance, etc.)
            trade_outcome: Trade result (pnl, duration, exit_reason, win/loss)
            market_state: Optional dict with RSI, volume_ratio, hour, day_of_week, streak, recent_pnl, vix, vwap_distance, atr
            partial_exits: List of partial exit decisions (level, r_multiple, contracts, percentage)
        """
        # Build market context (use provided state or defaults)
        if market_state is None:
            market_state = {}
        
        experience = {
            'timestamp': datetime.now().isoformat(),
            'regime': regime,
            'exit_params': exit_params,
            'outcome': trade_outcome,
            'situation': {
                'time_of_day': datetime.now().strftime('%H:%M'),
                'volatility_atr': exit_params.get('current_atr', 0),
                'trend_strength': trade_outcome.get('trend_strength', 0)
            },
            # NEW: Add 9-feature market state for context-aware exit learning
            'market_state': {
                'rsi': market_state.get('rsi', 50.0),
                'volume_ratio': market_state.get('volume_ratio', 1.0),
                'hour': market_state.get('hour', 12),
                'day_of_week': market_state.get('day_of_week', 0),
                'streak': market_state.get('streak', 0),
                'recent_pnl': market_state.get('recent_pnl', 0.0),
                'vix': market_state.get('vix', 15.0),
                'vwap_distance': market_state.get('vwap_distance', 0.0),
                'atr': market_state.get('atr', exit_params.get('current_atr', 0))
            },
            # NEW: Store partial exit decisions for scaling strategy learning
            'partial_exits': partial_exits if partial_exits else []
        }
        
        self.exit_experiences.append(experience)
        
        # Save to cloud API immediately if configured
        if self.use_cloud:
            try:
                # Convert boolean values to int for JSON serialization
                cloud_experience = {
                    **experience,
                    'outcome': {
                        **experience['outcome'],
                        'win': int(experience['outcome']['win'])  # bool → int
                    }
                }
                
                response = requests.post(
                    f"{self.cloud_api_url}/api/ml/save_exit_experience",
                    json=cloud_experience,
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('saved'):
                        logger.info(f"✅ [CLOUD] Saved exit experience to cloud pool ({data.get('total_exit_experiences', 0):,} total)")
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
        market_ctx = experience.get('market_state', {})
        logger.info(f"[EXIT RL] LEARNED: {regime} | {exit_params['breakeven_threshold_ticks']}t BE, "
                   f"{exit_params['trailing_distance_ticks']}t Trail | "
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

    
    def load_experiences(self):
        """Load past exit experiences from cloud API or local file (fallback)."""
        
        # Try cloud first if configured
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
        
        # Fallback to local file
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
            logger.warning(f"[LOCAL] Exit experience file not found: {self.experience_file}")
    
    
    def get_stop_multiplier(self, regime: str) -> float:
        """
        Get the learned stop loss multiplier for the current regime.
        
        Args:
            regime: Current market regime
            
        Returns: Stop loss multiplier (ATR multiple, typically 2.8x-4.5x)
        """
        params = self.learned_params.get(regime, self.learned_params.get('NORMAL', {}))
        stop_mult = params.get('stop_mult', 3.6)  # Default 3.6x if not found
        
        logger.info(f"[EXIT RL] Using {regime} stop multiplier: {stop_mult:.1f}x ATR")
        return stop_mult
    
    
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
            # Get hour
            from datetime import datetime
            hour = datetime.now().hour
            
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
        """Save exit experiences to cloud API or local file (fallback)."""
        
        # Try cloud first if configured
        if self.use_cloud:
            # Note: Individual experiences already saved via record_exit_outcome()
            # This method is for batch saves or fallback
            logger.info(f"[CLOUD] Using cloud API - experiences saved individually on record")
            return
        
        # Fallback to local file
        try:
            import json
            
            # Use custom JSON encoder that handles numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if hasattr(obj, 'item'):  # numpy scalar
                        return obj.item()
                    elif hasattr(obj, 'tolist'):  # numpy array
                        return obj.tolist()
                    return super().default(obj)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.experience_file), exist_ok=True)
            
            # Save with custom encoder
            with open(self.experience_file, 'w') as f:
                json.dump({
                    'exit_experiences': self.exit_experiences,
                    'learned_params': self.learned_params,
                    'total_exits': len(self.exit_experiences)
                }, f, indent=2, cls=NumpyEncoder)
                
            logger.info(f"[LOCAL] Saved {len(self.exit_experiences)} exit experiences to local file")
        except Exception as e:
            logger.error(f"[LOCAL] Failed to save exit experiences: {e}")


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
        "situation_factors": situation_factors,
        "decision_reasons": aggression_reasons if aggression_reasons else ["balanced"],
        "duration_minutes": duration_minutes,
        "learned_multiplier": breakeven_threshold_multiplier  # Include learned multiplier for tracking
    }
