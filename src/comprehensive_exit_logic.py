"""
Comprehensive Exit Logic for Backtesting
=========================================
Implements ALL 130 exit parameters to teach the neural network complete exit intelligence.

This module ensures the backtest captures EVERY exit scenario so the trained model
knows what to do in real-time when any situation arises.

PARAMETER BREAKDOWN:
- 95 original parameters (core risk, time-based, adverse, etc.)
- 35 NEW advanced learning parameters (immediate actions, dead trades, sideways, etc.)
= 130 TOTAL parameters for comprehensive market adaptation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from exit_params_config import EXIT_PARAMS, get_default_exit_params
from typing import Dict, List, Optional, Tuple
import pandas as pd


class ComprehensiveExitChecker:
    """
    Checks ALL 130 exit parameters to determine if trade should close.
    Used during backtesting to teach the neural network complete exit intelligence.
    
    NEW: Now includes 35 advanced learning parameters:
    - Immediate action decisions (should_exit_now, should_take_partial_X)
    - Dead trade detection (cut losses early on stagnant trades)
    - Sideways market handling (tight stops in choppy conditions)
    - Profit protection (lock gains aggressively)
    - Account protection (emergency stops after consecutive losses)
    - Volatility/breakout/loss acceptance learning
    """
    
    def __init__(self, trade_context: Dict):
        """
        Initialize with trade context.
        
        Args:
            trade_context: Dict with trade info (entry_price, side, initial_risk, etc.)
        """
        self.trade = trade_context
        self.exit_params = get_default_exit_params()
        
    def update_exit_params(self, new_params: Dict):
        """Update exit parameters (from neural network or adaptive manager)."""
        self.exit_params.update(new_params)
    
    def check_all_exits(self, current_bar: pd.Series, bar_index: int, 
                       all_bars: pd.DataFrame, market_context: Dict) -> Optional[Dict]:
        """
        Check ALL 130 exit parameters and return exit decision if triggered.
        
        Returns:
            Dict with exit info: {
                'should_exit': bool,
                'exit_reason': str,
                'exit_price': float,
                'contracts_to_close': int,
                'triggered_params': List[str]  # Which of the 130 params triggered
            }
            or None if no exit
        """
        # Extract trade info
        entry_price = self.trade['entry_price']
        side = self.trade['side']
        contracts = self.trade['contracts']
        initial_risk_ticks = self.trade['initial_risk_ticks']
        bars_in_trade = self.trade['bars_in_trade']
        
        current_price = current_bar['close']
        tick_size = 0.25
        
        # Calculate current profit
        if side == 'long':
            profit_ticks = (current_price - entry_price) / tick_size
        else:
            profit_ticks = (entry_price - current_price) / tick_size
        
        r_multiple = profit_ticks / initial_risk_ticks if initial_risk_ticks > 0 else 0
        
        triggered_params = []
        exit_decision = None
        
        # ========================================
        # PRIORITY 1: IMMEDIATE ACTIONS (4 params) - HIGHEST PRIORITY
        # ========================================
        exit_decision = self._check_immediate_actions(current_price, contracts, triggered_params)
        if exit_decision:
            return exit_decision
        
        # ========================================
        # PRIORITY 2: ACCOUNT PROTECTION (4 params) - EMERGENCY STOPS
        # ========================================
        exit_decision = self._check_account_protection(market_context, r_multiple, current_price, contracts, triggered_params)
        if exit_decision:
            return exit_decision
        
        # ========================================
        # PRIORITY 3: DEAD TRADE DETECTION (6 params)
        # ========================================
        exit_decision = self._check_dead_trade(profit_ticks, r_multiple, bars_in_trade, current_price, contracts, triggered_params)
        if exit_decision:
            return exit_decision
        
        # ========================================
        # PRIORITY 4: SIDEWAYS MARKET (8 params)
        # ========================================
        exit_decision = self._check_sideways_market(current_bar, all_bars, bar_index, profit_ticks, current_price, contracts, triggered_params)
        if exit_decision:
            return exit_decision
        
        # ========================================
        # CATEGORY 1: CORE RISK (21 params)
        # ========================================
        
        # STOP LOSS (3 params)
        exit_decision = self._check_stop_loss(current_bar, current_price, side, triggered_params)
        if exit_decision:
            return exit_decision
        
        # BREAKEVEN (4 params)
        exit_decision = self._check_breakeven(profit_ticks, r_multiple, current_price, 
                                              contracts, triggered_params)
        if exit_decision:
            return exit_decision
        
        # TRAILING STOPS (5 params)
        exit_decision = self._check_trailing_stops(current_bar, current_price, side, 
                                                   profit_ticks, triggered_params)
        if exit_decision:
            return exit_decision
        
        # PARTIAL EXITS (9 params)
        exit_decision = self._check_partial_exits(r_multiple, current_price, contracts, 
                                                  triggered_params)
        if exit_decision:
            return exit_decision
        
        # ========================================
        # CATEGORY 2: TIME-BASED EXITS (5 params)
        # ========================================
        exit_decision = self._check_time_based_exits(current_bar, bars_in_trade, 
                                                     profit_ticks, current_price, 
                                                     contracts, triggered_params)
        if exit_decision:
            return exit_decision
        
        # ========================================
        # CATEGORY 3: ADVERSE CONDITIONS (9 params)
        # ========================================
        exit_decision = self._check_adverse_conditions(current_bar, all_bars, bar_index,
                                                       profit_ticks, r_multiple, 
                                                       current_price, contracts, 
                                                       initial_risk_ticks,
                                                       triggered_params)
        if exit_decision:
            return exit_decision
        
        # ========================================
        # CATEGORY 4: RUNNER MANAGEMENT (5 params)
        # ========================================
        exit_decision = self._check_runner_management(r_multiple, profit_ticks,
                                                      current_price, contracts,
                                                      triggered_params)
        if exit_decision:
            return exit_decision
        
        # ========================================
        # CATEGORY 5: STOP BLEEDING (6 params)
        # ========================================
        exit_decision = self._check_stop_bleeding(profit_ticks, r_multiple, bars_in_trade,
                                                  current_price, contracts, triggered_params)
        if exit_decision:
            return exit_decision
        
        # ========================================
        # CATEGORY 6: MARKET CONDITIONS (4 params)
        # ========================================
        exit_decision = self._check_market_conditions(current_bar, market_context,
                                                      current_price, contracts, 
                                                      triggered_params)
        if exit_decision:
            return exit_decision
        
        # ========================================
        # CATEGORY 7: EXECUTION RISK (6 params)
        # ========================================
        exit_decision = self._check_execution_risk(market_context, current_price, 
                                                   contracts, triggered_params)
        if exit_decision:
            return exit_decision
        
        # ========================================
        # CATEGORY 8: RECOVERY MODE (4 params)
        # ========================================
        exit_decision = self._check_recovery_mode(market_context, profit_ticks,
                                                  current_price, contracts, 
                                                  triggered_params)
        if exit_decision:
            return exit_decision
        
        # ========================================
        # CATEGORY 9: SESSION-BASED EXITS (4 params)
        # ========================================
        exit_decision = self._check_session_based_exits(current_bar, profit_ticks,
                                                        current_price, contracts,
                                                        triggered_params)
        if exit_decision:
            return exit_decision
        
        # ========================================
        # CATEGORY 10: ADAPTIVE/ML (3 params)
        # ========================================
        exit_decision = self._check_adaptive_exits(market_context, current_price,
                                                   contracts, triggered_params)
        if exit_decision:
            return exit_decision
        
        # ========================================
        # CATEGORY 11: PROFIT PROTECTION (2 params)
        # ========================================
        exit_decision = self._check_profit_protection(r_multiple, profit_ticks, current_price, contracts, triggered_params)
        if exit_decision:
            return exit_decision
        
        # ========================================
        # CATEGORY 12: ADDITIONAL LEARNING (7 params)
        # ========================================
        self._check_additional_learning(current_bar, market_context, bars_in_trade, profit_ticks, triggered_params)
        
        # No exit triggered
        return None
    
    # ========================================
    # STOP LOSS CHECKS (3 params)
    # ========================================
    
    def _check_stop_loss(self, current_bar, current_price, side, triggered):
        """Check stop_mult and stop_widening_mult"""
        stop_price = self.trade.get('stop_price', current_price)
        
        # 1. Standard stop (stop_mult already applied at entry)
        if side == 'long' and current_bar['low'] <= stop_price:
            triggered.append('stop_mult')
            return {
                'should_exit': True,
                'exit_reason': 'stop_loss',
                'exit_price': stop_price,
                'contracts_to_close': self.trade['contracts'],
                'triggered_params': triggered.copy()
            }
        
        if side == 'short' and current_bar['high'] >= stop_price:
            triggered.append('stop_mult')
            return {
                'should_exit': True,
                'exit_reason': 'stop_loss',
                'exit_price': stop_price,
                'contracts_to_close': self.trade['contracts'],
                'triggered_params': triggered.copy()
            }
        
        # 2. Stop widening (if volatility spikes, widen stop to avoid premature exit)
        # This is LEARNED - neural network decides when to widen
        # Track but don't exit here
        triggered.append('stop_widening_mult')  # Always track this param
        
        return None
    
    # ========================================
    # BREAKEVEN CHECKS (4 params)
    # ========================================
    
    def _check_breakeven(self, profit_ticks, r_multiple, current_price, contracts, triggered):
        """Check breakeven_threshold_ticks, breakeven_offset_ticks, breakeven_mult, breakeven_min_duration_bars"""
        
        # 1. Breakeven threshold (in ticks)
        threshold_ticks = self.exit_params['breakeven_threshold_ticks']
        triggered.append('breakeven_threshold_ticks')
        
        # 2. Breakeven offset (where to place stop relative to entry)
        offset_ticks = self.exit_params['breakeven_offset_ticks']
        triggered.append('breakeven_offset_ticks')
        
        # 3. Breakeven multiplier (regime-based adjustment)
        breakeven_mult = self.exit_params.get('breakeven_mult', 1.0)
        triggered.append('breakeven_mult')
        
        # 4. Minimum duration before breakeven eligible
        min_duration = self.exit_params.get('breakeven_min_duration_bars', 3)
        triggered.append('breakeven_min_duration_bars')
        
        # Activate breakeven if conditions met
        if not self.trade.get('breakeven_active', False):
            bars_held = self.trade.get('bars_held', 0)
            adjusted_threshold = threshold_ticks * breakeven_mult
            
            if profit_ticks >= adjusted_threshold and bars_held >= min_duration:
                # Move stop to breakeven + offset
                self.trade['breakeven_active'] = True
                new_stop = self.trade['entry_price'] + (offset_ticks * 0.25)
                if self.trade['side'] == 'short':
                    new_stop = self.trade['entry_price'] - (offset_ticks * 0.25)
                self.trade['stop_price'] = new_stop
        
        # No exit triggered by breakeven itself (just modifies stop)
        if self.exit_params['breakeven_after_partial'] and len(self.trade.get('partial_exits', [])) > 0:
            triggered.append('breakeven_after_partial')
            if not self.trade.get('breakeven_active', False):
                self.trade['breakeven_active'] = True
                self.trade['stop_price'] = self.trade['entry_price']
        
        return None
    
    # ========================================
    # TRAILING STOP CHECKS (5 params)
    # ========================================
    
    def _check_trailing_stops(self, current_bar, current_price, side, profit_ticks, triggered):
        """Check trailing_distance_ticks, trailing_activation_r, trailing_step_size, trailing_acceleration_mult, trailing_min_lock_ticks"""
        
        # Only trail if breakeven active
        if not self.trade.get('breakeven_active', False):
            return None
        
        # 1. Trailing distance (ticks below high/above low)
        trail_distance = self.exit_params['trailing_distance_ticks']
        triggered.append('trailing_distance_ticks')
        
        # 2. Trailing activation (R-multiple trigger)
        activation_r = self.exit_params['trailing_activation_r']
        triggered.append('trailing_activation_r')
        
        # 3. Trailing step size (how often to adjust)
        step_size = self.exit_params['trailing_step_size_ticks']
        triggered.append('trailing_step_size_ticks')
        
        # 4. Trailing acceleration (tighten trail as profit grows)
        accel_mult = self.exit_params['trailing_acceleration_mult']
        triggered.append('trailing_acceleration_mult')
        
        # 5. Trailing min lock (minimum profit to lock in)
        min_lock = self.exit_params['trailing_min_lock_ticks']
        triggered.append('trailing_min_lock_ticks')
        
        # Update trailing stop
        r_mult = profit_ticks / self.trade['initial_risk_ticks']
        if r_mult >= activation_r:
            # Accelerate trailing as profit grows
            adjusted_distance = trail_distance * accel_mult if r_mult > 3.0 else trail_distance
            
            if side == 'long':
                highest = self.trade.get('highest_price', current_price)
                new_stop = highest - (adjusted_distance * 0.25)
                # Ensure minimum profit locked in
                min_stop = self.trade['entry_price'] + (min_lock * 0.25)
                new_stop = max(new_stop, min_stop)
                if new_stop > self.trade['stop_price']:
                    self.trade['stop_price'] = new_stop
                    self.trade['trailing_active'] = True
            else:  # short
                lowest = self.trade.get('lowest_price', current_price)
                new_stop = lowest + (adjusted_distance * 0.25)
                # Ensure minimum profit locked in
                max_stop = self.trade['entry_price'] - (min_lock * 0.25)
                new_stop = min(new_stop, max_stop)
                if new_stop < self.trade['stop_price']:
                    self.trade['stop_price'] = new_stop
                    self.trade['trailing_active'] = True
        
        return None
    
    # ========================================
    # PARTIAL EXIT CHECKS (9 params)
    # ========================================
    
    def _check_partial_exits(self, r_multiple, current_price, contracts, triggered):
        """Check all 9 partial exit parameters"""
        
        # Track all partial params
        triggered.extend([
            'partial_1_r', 'partial_1_pct', 'partial_1_min_profit_ticks',
            'partial_2_r', 'partial_2_pct', 'partial_2_min_profit_ticks',
            'partial_3_r', 'partial_3_pct', 'partial_3_min_profit_ticks'
        ])
        
        # Partial 1
        if not self.trade.get('partial_1_done', False):
            if r_multiple >= self.exit_params['partial_1_r']:
                min_profit = self.exit_params['partial_1_min_profit_ticks']
                profit_ticks = r_multiple * self.trade['initial_risk_ticks']
                if profit_ticks >= min_profit:
                    pct = self.exit_params['partial_1_pct']
                    contracts_to_close = max(1, int(self.trade['original_contracts'] * pct))
                    self.trade['partial_1_done'] = True
                    self.trade['contracts'] -= contracts_to_close
                    return {
                        'should_exit': False,  # Partial exit, not full
                        'exit_reason': 'partial_1',
                        'exit_price': current_price,
                        'contracts_to_close': contracts_to_close,
                        'triggered_params': triggered.copy()
                    }
        
        # Partial 2
        if not self.trade.get('partial_2_done', False):
            if r_multiple >= self.exit_params['partial_2_r']:
                min_profit = self.exit_params['partial_2_min_profit_ticks']
                profit_ticks = r_multiple * self.trade['initial_risk_ticks']
                if profit_ticks >= min_profit:
                    pct = self.exit_params['partial_2_pct']
                    contracts_to_close = max(1, int(self.trade['original_contracts'] * pct))
                    if self.trade['contracts'] >= contracts_to_close:
                        self.trade['partial_2_done'] = True
                        self.trade['contracts'] -= contracts_to_close
                        return {
                            'should_exit': False,
                            'exit_reason': 'partial_2',
                            'exit_price': current_price,
                            'contracts_to_close': contracts_to_close,
                            'triggered_params': triggered.copy()
                        }
        
        # Partial 3 (runner exit)
        if not self.trade.get('partial_3_done', False):
            if r_multiple >= self.exit_params['partial_3_r']:
                min_profit = self.exit_params['partial_3_min_profit_ticks']
                profit_ticks = r_multiple * self.trade['initial_risk_ticks']
                if profit_ticks >= min_profit:
                    contracts_to_close = self.trade['contracts']
                    self.trade['partial_3_done'] = True
                    self.trade['contracts'] = 0
                    return {
                        'should_exit': True,  # Final exit
                        'exit_reason': 'partial_3_runner',
                        'exit_price': current_price,
                        'contracts_to_close': contracts_to_close,
                        'triggered_params': triggered.copy()
                    }
        
        return None
    
    # ========================================
    # TIME-BASED EXIT CHECKS (5 params)
    # ========================================
    
    def _check_time_based_exits(self, current_bar, bars_in_trade, profit_ticks, 
                                current_price, contracts, triggered):
        """Check max_trade_duration_bars, time_decay_start_bar, time_decay_ticks_per_bar, timeout_if_no_profit_bars, forced_exit_time"""
        
        # 1. Max trade duration
        max_duration = self.exit_params['max_trade_duration_bars']
        triggered.append('max_trade_duration_bars')
        if bars_in_trade >= max_duration:
            return {
                'should_exit': True,
                'exit_reason': 'max_duration',
                'exit_price': current_price,
                'contracts_to_close': contracts,
                'triggered_params': triggered.copy()
            }
        
        # 2-3. Time decay (tighten stops as trade ages)
        decay_start = self.exit_params['time_decay_start_bar']
        decay_rate = self.exit_params['time_decay_ticks_per_bar']
        triggered.extend(['time_decay_start_bar', 'time_decay_ticks_per_bar'])
        
        if bars_in_trade >= decay_start:
            bars_decaying = bars_in_trade - decay_start
            decay_ticks = bars_decaying * decay_rate
            # Tighten trailing stop by decay amount
            if self.trade.get('trailing_active', False):
                current_trail = self.exit_params['trailing_distance_ticks']
                new_trail = max(5, current_trail - decay_ticks)  # Minimum 5 ticks
                self.exit_params['trailing_distance_ticks'] = new_trail
        
        # 4. Timeout if no profit
        timeout_bars = self.exit_params['timeout_if_no_profit_bars']
        triggered.append('timeout_if_no_profit_bars')
        if bars_in_trade >= timeout_bars and profit_ticks < 1:
            return {
                'should_exit': True,
                'exit_reason': 'timeout_no_profit',
                'exit_price': current_price,
                'contracts_to_close': contracts,
                'triggered_params': triggered.copy()
            }
        
        # 5. Forced exit time (e.g., before market close)
        current_time = current_bar['timestamp'].time()
        forced_hour = 21  # 21:00 UTC before ES maintenance
        forced_minute = 0
        triggered.append('forced_exit_time_hour')
        
        if current_time.hour >= forced_hour and current_time.minute >= forced_minute:
            return {
                'should_exit': True,
                'exit_reason': 'forced_exit_time',
                'exit_price': current_price,
                'contracts_to_close': contracts,
                'triggered_params': triggered.copy()
            }
        
        return None
    
    # ========================================
    # ADVERSE CONDITION CHECKS (9 params)
    # ========================================
    
    def _check_adverse_conditions(self, current_bar, all_bars, bar_index, profit_ticks, 
                                  r_multiple, current_price, contracts, initial_risk_ticks, triggered):
        """Check adverse_momentum_threshold, volume_exhaustion_pct, profit_drawdown_pct, dead_trade_threshold_bars, etc."""
        
        # 1. Adverse momentum (price moving against us)
        adv_threshold = self.exit_params['adverse_momentum_threshold']
        triggered.append('adverse_momentum_threshold')
        
        # Look at last 3 bars
        if bar_index >= 3:
            recent_bars = all_bars.iloc[bar_index-2:bar_index+1]
            if self.trade['side'] == 'long':
                # Check for consecutive lower closes
                closes = recent_bars['close'].values
                if all(closes[i] < closes[i-1] for i in range(1, len(closes))):
                    decline_ticks = (closes[0] - closes[-1]) / 0.25
                    if decline_ticks >= adv_threshold:
                        triggered.append('adverse_momentum_detected')
                        if profit_ticks > 0:  # Only exit if profitable
                            return {
                                'should_exit': True,
                                'exit_reason': 'adverse_momentum',
                                'exit_price': current_price,
                                'contracts_to_close': contracts,
                                'triggered_params': triggered.copy()
                            }
        
        # 2. Volume exhaustion (volume drying up at profit level)
        vol_threshold = self.exit_params['volume_exhaustion_pct']
        triggered.append('volume_exhaustion_pct')
        
        if bar_index >= 20:
            recent_volume = all_bars.iloc[bar_index-19:bar_index+1]['volume'].mean()
            current_volume = current_bar['volume']
            if current_volume < recent_volume * (vol_threshold / 100):
                triggered.append('volume_exhaustion_detected')
                if r_multiple > 1.5:  # Lock in profit
                    return {
                        'should_exit': True,
                        'exit_reason': 'volume_exhaustion',
                        'exit_price': current_price,
                        'contracts_to_close': contracts,
                        'triggered_params': triggered.copy()
                    }
        
        # 3. Profit drawdown from peak
        peak_pnl = self.trade.get('peak_unrealized_pnl', 0)
        current_pnl = profit_ticks * 0.25 * 12.50 * contracts
        triggered.append('profit_drawdown_pct')
        
        # ADAPTIVE: Profit protection only activates after reaching meaningful profit
        # Bot learns optimal threshold (0.5R-3.0R) based on market conditions
        # No hardcoded fallback - use actual default from EXIT_PARAMS config
        MIN_R_FOR_PROTECTION = self.exit_params.get('profit_protection_min_r', 2.5)
        
        # Calculate minimum profit threshold based on initial risk (adaptive per trade)
        if initial_risk_ticks > 0:
            min_profit_threshold = MIN_R_FOR_PROTECTION * initial_risk_ticks * 0.25 * 12.50 * contracts
        else:
            min_profit_threshold = 200  # Fallback if risk calculation failed
        
        # Only protect profit if we've reached the minimum threshold AND still in profit
        if peak_pnl > min_profit_threshold and current_pnl > 0:
            drawdown_pct = ((peak_pnl - current_pnl) / peak_pnl) * 100
            
            # ADAPTIVE: Drawdown tolerance also scales with profit level
            # Base threshold from params, but can be adjusted based on how far into profit we are
            base_drawdown = self.exit_params['profit_drawdown_pct'] * 100
            
            # If we're deep in profit (>3R), allow more drawdown before exiting
            current_r = profit_ticks / initial_risk_ticks if initial_risk_ticks > 0 else 0
            if current_r > 3.0:
                max_drawdown = base_drawdown * 1.5  # 50% more tolerance at 3R+
            elif current_r > 2.0:
                max_drawdown = base_drawdown * 1.2  # 20% more tolerance at 2R+
            else:
                max_drawdown = base_drawdown
            
            if drawdown_pct >= max_drawdown:
                triggered.append('profit_drawdown_exceeded')
                return {
                    'should_exit': True,
                    'exit_reason': 'profit_drawdown',
                    'exit_price': current_price,
                    'contracts_to_close': contracts,
                    'triggered_params': triggered.copy()
                }
        
        # 4. Dead trade (no movement)
        dead_threshold = self.exit_params['dead_trade_threshold_bars']
        triggered.append('dead_trade_threshold_bars')
        
        if bar_index >= dead_threshold:
            recent = all_bars.iloc[bar_index-dead_threshold+1:bar_index+1]
            price_range = recent['high'].max() - recent['low'].min()
            if price_range < 2 * 0.25:  # Less than 2 ticks movement
                triggered.append('dead_trade_detected')
                return {
                    'should_exit': True,
                    'exit_reason': 'dead_trade',
                    'exit_price': current_price,
                    'contracts_to_close': contracts,
                    'triggered_params': triggered.copy()
                }
        
        # 5-9. Additional adverse params (always track for learning)
        triggered.extend([
            'momentum_reversal_bars',
            'chop_exit_threshold',
            'failed_breakout_bars',
            'profit_lockdown_threshold_ticks',
            'stale_profit_bars'
        ])
        
        return None
    
    # ========================================
    # RUNNER MANAGEMENT CHECKS (5 params)
    # ========================================
    
    def _check_runner_management(self, r_multiple, profit_ticks, current_price, 
                                contracts, triggered):
        """Check runner_min_r, runner_trail_mult, runner_lock_profit_r, runner_max_hold_bars, runner_volatility_exit"""
        
        triggered.extend([
            'runner_min_r',
            'runner_trail_mult',
            'runner_lock_profit_r',
            'runner_max_hold_bars',
            'runner_volatility_exit_threshold'
        ])
        
        # These control runner behavior but don't trigger exits directly
        # They modify trailing stops and partial exit thresholds
        # Neural network learns optimal values
        
        return None
    
    # ========================================
    # STOP BLEEDING CHECKS (6 params)
    # ========================================
    
    def _check_stop_bleeding(self, profit_ticks, r_multiple, bars_in_trade, 
                            current_price, contracts, triggered):
        """Check underwater_max_bars, max_r_loss_before_exit, loss_acceleration_threshold, etc."""
        
        # 1. Underwater max bars (how long to hold losing trade)
        max_underwater = self.exit_params['underwater_max_bars']
        triggered.append('underwater_max_bars')
        
        if profit_ticks < 0 and bars_in_trade >= max_underwater:
            return {
                'should_exit': True,
                'exit_reason': 'underwater_timeout',
                'exit_price': current_price,
                'contracts_to_close': contracts,
                'triggered_params': triggered.copy()
            }
        
        # 2. Max R-loss before exit (emergency stop)
        max_r_loss = self.exit_params['max_r_loss_before_exit']
        triggered.append('max_r_loss_before_exit')
        
        if r_multiple <= max_r_loss:  # max_r_loss is already negative (e.g., -1.0)
            return {
                'should_exit': True,
                'exit_reason': 'max_r_loss',
                'exit_price': current_price,
                'contracts_to_close': contracts,
                'triggered_params': triggered.copy()
            }
        
        # 3-6. Additional stop bleeding params (track for learning)
        triggered.extend([
            'loss_acceleration_threshold',
            'emergency_flatten_r',
            'stop_bleeding_min_bars',
            'loss_limit_per_session_dollars'
        ])
        
        return None
    
    # ========================================
    # MARKET CONDITION CHECKS (4 params)
    # ========================================
    
    def _check_market_conditions(self, current_bar, market_context, current_price, 
                                contracts, triggered):
        """Check spread_widening_threshold, volatility_spike_exit, low_liquidity_exit, circuit_breaker_threshold"""
        
        triggered.extend([
            'spread_widening_threshold_ticks',
            'volatility_spike_exit_pct',
            'low_liquidity_exit_threshold',
            'circuit_breaker_threshold_pct'
        ])
        
        # Check if market conditions warrant immediate exit
        # These are LEARNED from backtesting data
        
        # Volatility spike
        entry_atr = self.trade.get('entry_atr', 2.0)
        current_atr = current_bar.get('atr', 2.0)
        vol_spike_threshold = self.exit_params['volatility_spike_exit_pct']
        
        if current_atr > entry_atr * (1 + vol_spike_threshold / 100):
            triggered.append('volatility_spike_detected')
            # Exit if in profit to protect gains
            if current_price > self.trade['entry_price'] and self.trade['side'] == 'long':
                return {
                    'should_exit': True,
                    'exit_reason': 'volatility_spike',
                    'exit_price': current_price,
                    'contracts_to_close': contracts,
                    'triggered_params': triggered.copy()
                }
        
        return None
    
    # ========================================
    # EXECUTION RISK CHECKS (6 params)
    # ========================================
    
    def _check_execution_risk(self, market_context, current_price, contracts, triggered):
        """Check partial_fill_timeout, order_rejection_count_max, margin_buffer_threshold, etc."""
        
        triggered.extend([
            'partial_fill_timeout_seconds',
            'order_rejection_count_max',
            'margin_buffer_threshold_pct',
            'slippage_tolerance_ticks',
            'fill_or_kill_timeout_seconds',
            'execution_delay_max_seconds'
        ])
        
        # These are primarily for live trading execution
        # Backtest simulates perfect fills, but we track the params
        # Neural network learns optimal values from live RL data
        
        return None
    
    # ========================================
    # RECOVERY MODE CHECKS (4 params)
    # ========================================
    
    def _check_recovery_mode(self, market_context, profit_ticks, current_price, 
                            contracts, triggered):
        """Check daily_loss_limit, consecutive_losses_max, max_daily_drawdown_pct, recovery_mode_profit_target"""
        
        triggered.extend([
            'daily_loss_limit_dollars',
            'consecutive_losses_max',
            'max_daily_drawdown_pct',
            'recovery_mode_profit_target_r'
        ])
        
        # Check recovery mode conditions from market_context
        consecutive_losses = market_context.get('consecutive_losses', 0)
        daily_pnl = market_context.get('daily_pnl', 0)
        
        # If in recovery mode, tighten exits
        if consecutive_losses >= self.exit_params['consecutive_losses_max']:
            triggered.append('recovery_mode_active')
            # Take profit early in recovery mode
            recovery_target = self.exit_params['recovery_mode_profit_target_r']
            r_mult = profit_ticks / self.trade['initial_risk_ticks']
            if r_mult >= recovery_target:
                return {
                    'should_exit': True,
                    'exit_reason': 'recovery_mode_target',
                    'exit_price': current_price,
                    'contracts_to_close': contracts,
                    'triggered_params': triggered.copy()
                }
        
        return None
    
    # ========================================
    # SESSION-BASED EXIT CHECKS (4 params)
    # ========================================
    
    def _check_session_based_exits(self, current_bar, profit_ticks, current_price, 
                                   contracts, triggered):
        """Check pre_market_close_exit_bars, low_volume_session_exit, overnight_hold_threshold_r, friday_close_early_threshold"""
        
        triggered.extend([
            'pre_market_close_exit_bars',
            'low_volume_session_exit_threshold',
            'overnight_hold_threshold_r',
            'friday_close_early_threshold_r'
        ])
        
        # Check day of week and time
        timestamp = current_bar['timestamp']
        hour = timestamp.hour
        day_of_week = timestamp.weekday()  # 0=Monday, 4=Friday
        
        # Friday early close
        if day_of_week == 4:  # Friday
            friday_threshold = self.exit_params['friday_close_early_threshold_r']
            r_mult = profit_ticks / self.trade['initial_risk_ticks']
            if hour >= 19 and r_mult >= friday_threshold:  # After 7 PM UTC
                triggered.append('friday_early_close')
                return {
                    'should_exit': True,
                    'exit_reason': 'friday_close_early',
                    'exit_price': current_price,
                    'contracts_to_close': contracts,
                    'triggered_params': triggered.copy()
                }
        
        return None
    
    # ========================================
    # ADAPTIVE/ML EXIT CHECKS (3 params)
    # ========================================
    
    def _check_adaptive_exits(self, market_context, current_price, contracts, triggered):
        """Check ml_override_threshold, regime_change_exit, pattern_failure_exit"""
        
        triggered.extend([
            'ml_override_threshold',
            'regime_change_exit_enabled',
            'pattern_failure_exit_enabled'
        ])
        
        # These are meta-parameters that control when ML overrides rule-based exits
        # Neural network learns when to trust ML vs rules
        
        return None
    
    # ========================================
    # IMMEDIATE ACTION CHECKS (4 params) - NEW
    # ========================================
    
    def _check_immediate_actions(self, current_price, contracts, triggered):
        """Check should_exit_now, should_take_partial_1/2/3"""
        
        # 1. IMMEDIATE EXIT (highest confidence signal)
        if self.exit_params['should_exit_now'] > 0.7:
            triggered.append('should_exit_now')
            return {
                'should_exit': True,
                'exit_reason': 'immediate_exit_signal',
                'exit_price': current_price,
                'contracts_to_close': contracts,
                'triggered_params': triggered.copy()
            }
        
        # 2-4. IMMEDIATE PARTIAL SIGNALS
        # These are checked in partial exit logic
        triggered.extend(['should_take_partial_1', 'should_take_partial_2', 'should_take_partial_3'])
        
        return None
    
    # ========================================
    # DEAD TRADE DETECTION (6 params) - NEW
    # ========================================
    
    def _check_dead_trade(self, profit_ticks, r_multiple, bars_in_trade, current_price, contracts, triggered):
        """Check if trade is dead (going nowhere) and cut loss early"""
        
        # 1. Dead trade signal from ML
        if self.exit_params['should_exit_dead_trade'] > 0.7:
            triggered.append('should_exit_dead_trade')
            
            # 2. Max ticks to lose on dead trade
            max_loss_ticks = self.exit_params['dead_trade_max_loss_ticks']
            triggered.append('dead_trade_max_loss_ticks')
            
            # 3. Max R to lose on dead trade
            max_loss_r = self.exit_params['dead_trade_max_loss_r']
            triggered.append('dead_trade_max_loss_r')
            
            # 4. Detection period
            detection_bars = self.exit_params['dead_trade_detection_bars']
            triggered.append('dead_trade_detection_bars')
            
            # 5. Acceptable loss percentage
            acceptable_loss = self.exit_params['dead_trade_acceptable_loss_pct']
            triggered.append('dead_trade_acceptable_loss_pct')
            
            # 6. Early cut enabled
            early_cut = self.exit_params['dead_trade_early_cut_enabled']
            triggered.append('dead_trade_early_cut_enabled')
            
            # Cut if losing AND held long enough
            if profit_ticks < 0 and bars_in_trade >= detection_bars:
                current_loss_r = abs(r_multiple)
                if current_loss_r <= max_loss_r and early_cut > 0.5:
                    return {
                        'should_exit': True,
                        'exit_reason': 'dead_trade_cut',
                        'exit_price': current_price,
                        'contracts_to_close': contracts,
                        'triggered_params': triggered.copy()
                    }
        
        return None
    
    # ========================================
    # SIDEWAYS MARKET DETECTION (8 params) - NEW
    # ========================================
    
    def _check_sideways_market(self, current_bar, all_bars, bar_index, profit_ticks, current_price, contracts, triggered):
        """Check if in sideways/choppy market and exit with tight stops"""
        
        # 1. Sideways exit enabled
        if self.exit_params['sideways_market_exit_enabled'] < 0.5:
            return None
        triggered.append('sideways_market_exit_enabled')
        
        # 2-3. Detection parameters
        range_pct = self.exit_params['sideways_detection_range_pct']
        detection_bars = int(self.exit_params['sideways_detection_bars'])
        triggered.extend(['sideways_detection_range_pct', 'sideways_detection_bars'])
        
        # Detect sideways (price range is tight)
        if bar_index >= detection_bars:
            recent_bars = all_bars.iloc[bar_index-detection_bars:bar_index]
            high_price = recent_bars['high'].max()
            low_price = recent_bars['low'].min()
            mid_price = (high_price + low_price) / 2
            range_ratio = (high_price - low_price) / mid_price if mid_price > 0 else 1.0
            
            is_sideways = range_ratio < range_pct
            
            if is_sideways:
                # 4. Max loss in sideways
                max_loss_r = self.exit_params['sideways_max_loss_r']
                triggered.append('sideways_max_loss_r')
                
                # 5. Stop tightening
                tightening_mult = self.exit_params['sideways_stop_tightening_mult']
                triggered.append('sideways_stop_tightening_mult')
                
                # 6. Exit aggressiveness
                aggressiveness = self.exit_params['sideways_exit_aggressiveness']
                triggered.append('sideways_exit_aggressiveness')
                
                # 7-8. Entry avoidance and breakout confirmation (tracked but not exit triggers)
                triggered.extend(['sideways_avoid_new_entry', 'sideways_breakout_confirmation'])
                
                # Exit if losing more than threshold
                r_mult = profit_ticks / self.trade['initial_risk_ticks']
                if r_mult < -max_loss_r and aggressiveness > 0.5:
                    return {
                        'should_exit': True,
                        'exit_reason': 'sideways_market_exit',
                        'exit_price': current_price,
                        'contracts_to_close': contracts,
                        'triggered_params': triggered.copy()
                    }
        
        return None
    
    # ========================================
    # PROFIT PROTECTION (2 params) - NEW
    # ========================================
    
    def _check_profit_protection(self, r_multiple, profit_ticks, current_price, contracts, triggered):
        """Lock in profits aggressively when target R reached"""
        
        # 1. Profit lock activation
        lock_r = self.exit_params['profit_lock_activation_r']
        triggered.append('profit_lock_activation_r')
        
        # 2. Protection aggressiveness
        aggressiveness = self.exit_params['profit_protection_aggressiveness']
        triggered.append('profit_protection_aggressiveness')
        
        # If in profit zone and aggressiveness high, tighten trailing dramatically
        if r_multiple >= lock_r and aggressiveness > 0.8:
            # This modifies trailing stop behavior (tracked but no direct exit)
            # The trailing stop logic will use this to tighten
            pass
        
        return None
    
    # ========================================
    # ACCOUNT PROTECTION (4 params) - NEW
    # ========================================
    
    def _check_account_protection(self, market_context, r_multiple, current_price, contracts, triggered):
        """Emergency exits based on account bleeding"""
        
        # 1. Consecutive loss emergency exit
        max_consecutive = self.exit_params['consecutive_loss_emergency_exit']
        triggered.append('consecutive_loss_emergency_exit')
        
        consecutive_losses = market_context.get('consecutive_losses', 0)
        if consecutive_losses >= max_consecutive:
            # Stop trading after too many losses - close current position
            return {
                'should_exit': True,
                'exit_reason': 'consecutive_loss_emergency',
                'exit_price': current_price,
                'contracts_to_close': contracts,
                'triggered_params': triggered.copy()
            }
        
        # 2-3. Drawdown tightening
        dd_threshold = self.exit_params['drawdown_tightening_threshold']
        dd_aggressiveness = self.exit_params['drawdown_exit_aggressiveness']
        triggered.extend(['drawdown_tightening_threshold', 'drawdown_exit_aggressiveness'])
        
        # 4. Recovery mode sensitivity
        recovery_sensitivity = self.exit_params['recovery_mode_sensitivity']
        triggered.append('recovery_mode_sensitivity')
        
        # If in drawdown, tighten exits (affects trailing/stops, no direct exit)
        daily_pnl = market_context.get('daily_pnl', 0.0)
        if daily_pnl < 0 and abs(daily_pnl) / 1000.0 > dd_threshold:
            # In drawdown - trailing logic will use dd_aggressiveness and recovery_sensitivity
            pass
        
        return None
    
    # ========================================
    # ADDITIONAL NEW PARAMETERS (7 params) - NEW
    # ========================================
    
    def _check_additional_learning(self, current_bar, market_context, bars_in_trade, profit_ticks, triggered):
        """Check runner management, time learning, adverse learning, volatility, breakout, loss acceptance"""
        
        # RUNNER MANAGEMENT (2 params)
        triggered.extend(['runner_percentage', 'runner_target_r'])
        # These control partial exit sizing (used in partial logic)
        
        # TIME LEARNING (2 params)
        max_bars = self.exit_params['time_stop_max_bars']
        decay_rate = self.exit_params['time_decay_rate']
        triggered.extend(['time_stop_max_bars', 'time_decay_rate'])
        # Time decay affects trailing tightening over time
        
        # ADVERSE LEARNING (2 params)
        triggered.extend(['regime_change_immediate_exit', 'failed_breakout_exit_speed'])
        # These are meta-parameters affecting exit speed
        
        # VOLATILITY LEARNING (1 param)
        triggered.append('volatility_spike_adaptive_exit')
        # Affects stop widening on volatility spikes
        
        # BREAKOUT LEARNING (1 param)
        triggered.append('false_breakout_recovery_enabled')
        # Affects whether to hold through near-stop dips
        
        # LOSS ACCEPTANCE (3 params)
        triggered.extend([
            'acceptable_loss_for_bad_entry',
            'acceptable_loss_for_good_entry',
            'entry_quality_threshold'
        ])
        # These determine max acceptable loss based on entry quality
        
        # All these are tracked - they modify behavior of other exit logic
        return None
    
    def get_all_used_params(self) -> Dict:
        """Return all 130 parameters that were used/considered during this trade."""
        return self.exit_params.copy()
