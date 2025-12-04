"""
Signal Confidence - RL Layer for Capitulation Reversal Signals
================================================================
Learns which capitulation reversal signals to trust vs skip.

Keeps your hardcoded entry logic, but adds intelligence:
- Should I take this signal? (confidence scoring)
- How much to risk? (position sizing)
- When to exit? (profit taking)

Learns from every trade outcome to improve decision-making.
"""

import logging
import json
import os
from typing import Dict, Optional, Tuple
from datetime import datetime
from collections import deque
import random

logger = logging.getLogger(__name__)


class SignalConfidenceRL:
    """
    Reinforcement learning layer that decides whether to trust Capitulation Reversal signals.
    
    NOTE: For production deployments, RL should be hosted in the cloud.
    Local RL experience files are only used for backtesting and development.
    
    State: Market conditions when signal triggers
    Action: Take trade (yes/no) + position size + exit params
    Reward: Profit/loss from trade outcome
    """
    
    def __init__(self, experience_file: str = None, backtest_mode: bool = False, confidence_threshold: Optional[float] = None, exploration_rate: Optional[float] = None, min_exploration: Optional[float] = None, exploration_decay: Optional[float] = None, save_local: bool = True):
        """
        Initialize RL confidence scorer.
        
        Args:
            experience_file: Path to experience file (None = no local RL, cloud-based only)
            backtest_mode: Whether in backtest mode
            confidence_threshold: Optional fixed threshold (0.1-1.0). 
                                 - For LIVE/SHADOW mode: If None, defaults to 0.5 (50%). User's GUI setting always used.
                                 - For BACKTEST mode: If None, calculates adaptive threshold from experiences.
            exploration_rate: Percentage of random exploration (0.0-1.0). Default: 0.05 (5%)
            min_exploration: Minimum exploration rate (0.0-1.0). Default: 0.05 (5%)
            exploration_decay: Decay factor for exploration rate. Default: 0.995
            save_local: Whether to save experiences locally (False for live mode cloud-only saving)
        """
        # Default to no local experience file for production (cloud-based RL)
        # Only load local file for backtesting or if explicitly provided
        if experience_file is None and backtest_mode:
            self.experience_file = "data/signal_experience.json"
        else:
            self.experience_file = experience_file
        self.experiences = []  # All past (state, action, reward) tuples
        self.experience_keys = set()  # Set for O(1) duplicate detection
        self.recent_trades = deque(maxlen=20)  # Last 20 outcomes
        self.backtest_mode = backtest_mode
        self.freeze_learning = False  # LEARNING ENABLED - Brain 2 learns during backtests
        self.save_local = save_local  # Whether to save experiences locally
        
        # Random exploration enabled - no fixed seed for natural learning
        
        # Learning parameters - use config values or defaults
        self.exploration_rate = exploration_rate if exploration_rate is not None else 0.05  # Default 5%
        self.min_exploration = min_exploration if min_exploration is not None else 0.05  # Default 5%
        self.exploration_decay = exploration_decay if exploration_decay is not None else 0.995
        
        # User-configured threshold
        # For LIVE/SHADOW mode: Always use user threshold (default 50% if not set)
        # For BACKTEST mode: Use user threshold if set, otherwise calculate adaptive
        if not backtest_mode and confidence_threshold is None:
            # LIVE/SHADOW mode with no user config - use safe default of 50%
            self.user_threshold = 0.5
            pass  # Silent - live mode configuration
        else:
            self.user_threshold = confidence_threshold
            pass  # Silent - RL brain configuration
        
        # Cached optimal threshold (only used in backtest mode when user_threshold is None)
        self.cached_threshold = None
        self.last_threshold_calc_signal_count = 0
        self.last_threshold_calc_exp_count = 0
        
        # Performance tracking
        self.total_signals = 0
        self.signals_taken = 0
        self.signals_skipped = 0
        
        # Win/loss streak tracking (for adaptive behavior)
        self.current_win_streak = 0
        self.current_loss_streak = 0
        
        # Regime-specific multipliers (reserved for future use)
        self.regime_multipliers = {
            'HIGH_VOL_CHOPPY': 0.7,    # Reduce size in choppy high vol
            'HIGH_VOL_TRENDING': 0.85,  # Slightly reduce in volatile trends
            'LOW_VOL_RANGING': 1.0,     # Standard in calm range
            'LOW_VOL_TRENDING': 1.15,   # Increase in calm trends
            'NORMAL': 1.0               # Standard
        }
        
        self.load_experience()
        pass  # Silent - RL brain initialized
        
        # Log threshold configuration
        if self.user_threshold is not None:
            if self.backtest_mode:
                pass  # Silent - threshold configuration
            else:
                pass  # Silent - threshold configuration
        else:
            # Only happens in backtest mode now
            pass  # Silent - threshold will be calculated
        
        # Log exploration mode
        if self.backtest_mode:
            pass  # Silent - exploration mode
        else:
            pass  # Silent - live mode exploitation
    
    def _generate_experience_key(self, experience: Dict, execution_data: Optional[Dict] = None) -> str:
        """
        Generate a unique key for duplicate detection.
        Uses ALL significant fields to identify truly identical experiences.
        
        This prevents incorrectly removing valid experiences that have:
        - Same timestamp but different outcomes (P&L, duration, exit reason)
        - Same signal but different execution quality (MFE/MAE, slippage)
        
        Args:
            experience: The experience dictionary with market state and outcome data
            execution_data: Optional execution data dict (used during record_outcome)
        
        Returns:
            Hash string for O(1) duplicate detection
        """
        import hashlib
        
        # ALL fields that make an experience unique
        # If any of these differ, experiences are NOT duplicates
        # NOTE: exploration_rate is EXCLUDED because it's metadata about HOW
        # the signal was taken, not WHAT the signal outcome was. Including it
        # would cause the same trading signal at the same timestamp with the
        # same outcome to be stored twice just because the exploration rate
        # was different during collection.
        key_fields = [
            # Core identification fields
            'timestamp', 'symbol', 'price',
            # Outcome fields - these make each trade unique
            'pnl', 'duration', 'took_trade', 'exit_reason',
            # Market state fields (match actual field names in experience file)
            'flush_size_ticks', 'flush_velocity', 'flush_direction',
            'distance_from_flush_low', 'rsi', 'volume_climax_ratio',
            'vwap_distance_ticks', 'atr', 'regime', 'hour', 'session',
            # Execution quality fields
            'mfe', 'mae', 'order_type_used', 'entry_slippage_ticks',
            # Risk parameters
            'stop_distance_ticks', 'target_distance_ticks', 'risk_reward_ratio',
            # Binary confirmation flags
            'reversal_candle', 'no_new_extreme',
            # 'exploration_rate' - EXCLUDED: metadata about collection, not signal quality
        ]
        
        # Build key from all significant values
        # Handle exit_reason specially - check execution_data first, then experience dict
        values = []
        for field in key_fields:
            if field == 'exit_reason':
                # Check execution_data first (for new experiences being recorded)
                if execution_data and 'exit_reason' in execution_data:
                    val = execution_data['exit_reason']
                elif 'exit_reason' in experience:
                    val = experience['exit_reason']
                else:
                    val = None
            elif field in experience:
                val = experience[field]
            else:
                val = None
            
            # Round floats to 6 decimals to avoid precision issues
            if isinstance(val, float):
                val = round(val, 6)
            values.append(str(val) if val is not None else '')
        
        # Create hash from concatenated values
        key_string = '|'.join(values)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def capture_signal_state(self, rsi: float, vwap_distance: float, 
                            atr: float, volume_ratio: float,
                            hour: int, day_of_week: int,
                            recent_pnl: float, streak: int) -> Dict:
        """
        Capture market state when VWAP signal triggers.
        
        Args:
            rsi: Current RSI value
            vwap_distance: Distance from VWAP in std devs
            atr: Current ATR (volatility)
            volume_ratio: Current volume vs average
            hour: Hour of day (0-23)
            day_of_week: 0=Monday, 4=Friday
            recent_pnl: P&L from last 3 trades
            streak: Win/loss streak (positive=wins, negative=losses)
        """
        return {
            'rsi': round(rsi, 1),
            'vwap_distance': round(vwap_distance, 2),
            'atr': round(atr, 2),
            'volume_ratio': round(volume_ratio, 2),
            'hour': hour,
            'day_of_week': day_of_week,
            'recent_pnl': round(recent_pnl, 2),
            'streak': streak
        }
    
    def should_take_signal(self, state: Dict) -> Tuple[bool, float, str]:
        """
        Decide whether to take this VWAP signal.
        
        Returns:
            (take_trade, confidence, reason)
            - take_trade: True/False
            - confidence: 0.0-1.0 (how confident)
            - reason: Why this decision
        """
        self.total_signals += 1
        
        # SMART EXPLORATION: Only in backtest mode!
        # LIVE MODE: 0% exploration (pure exploitation of learned intelligence)
        # BACKTEST MODE: Use configured exploration_rate (default 5%)
        effective_exploration = self.exploration_rate if self.backtest_mode else 0.0
        
        # ALWAYS calculate confidence from experiences
        confidence, reason = self.calculate_confidence(state)
        
        # Determine which threshold to use
        if self.user_threshold is not None:
            # User has configured a specific threshold (or default 50% for live mode)
            optimal_threshold = self.user_threshold
        else:
            # No user threshold - only happens in BACKTEST mode
            # Calculate adaptive optimal threshold from experiences
            # CACHED ADAPTIVE THRESHOLD: Only recalculate every 100 signals or when experiences grow by 50+
            should_recalc = (
                self.cached_threshold is None or
                self.total_signals - self.last_threshold_calc_signal_count >= 100 or
                len(self.experiences) - self.last_threshold_calc_exp_count >= 50
            )
            
            if should_recalc:
                self.cached_threshold = self._calculate_optimal_threshold()
                self.last_threshold_calc_signal_count = self.total_signals
                self.last_threshold_calc_exp_count = len(self.experiences)
            
            optimal_threshold = self.cached_threshold
        
        # FILTER BASED ON LEARNED THRESHOLD (calculate first)
        take = confidence > optimal_threshold
        
        # Exploration: Give rejected signals a chance to be taken
        # This allows the system to learn from signals it would normally skip
        if not take and random.random() < effective_exploration:
            # This signal was rejected, but exploration gives it a chance
            take = True
            threshold_source = "User" if self.user_threshold is not None else "Learned"
            reason = f"Exploring ({effective_exploration*100:.0f}% chance for rejected signals, {len(self.experiences)} exp) | Threshold: {optimal_threshold:.1%} ({threshold_source})"
            self.signals_taken += 1
            return take, confidence, reason
        
        # Normal behavior: use threshold decision
        if take:
            self.signals_taken += 1
            reason += f" APPROVED ({confidence:.1%} > {optimal_threshold:.1%})"
        else:
            self.signals_skipped += 1
            reason += f" REJECTED ({confidence:.1%} < {optimal_threshold:.1%})"
        
        # Decay exploration over time
        self.exploration_rate = max(self.min_exploration, 
                                   self.exploration_rate * self.exploration_decay)
        
        return take, confidence, reason
    
    def calculate_confidence(self, current_state: Dict) -> Tuple[float, str]:
        """
        Calculate confidence based on similar past experiences.
        
        CONFIDENCE FORMULA:
        ==================
        Step 1: Find 10 most similar past trades
        Step 2: Calculate from those similar trades:
          - Win Rate = Winners / Total
          - Average Profit = Sum of profits / Count
          - Profit Score = min(Average Profit / 300, 1.0)
          - Final Confidence = (Win Rate × 90%) + (Profit Score × 10%)
        Step 3: If average profit is negative → Auto reject (0% confidence)
        
        Example: 8 wins out of 10 = 80% WR, $120 avg profit
          Profit Score = 120/300 = 0.40
          Confidence = (0.80 × 0.90) + (0.40 × 0.10) = 0.72 + 0.04 = 76%
        
        Returns:
            (confidence, reason)
        """
        # Need at least 10 experiences before using them for decisions
        if len(self.experiences) < 10:
            return 0.65, f"Limited experience ({len(self.experiences)} trades) - optimistic"
        
        # Step 1: Find 10 most similar past trades
        similar = self.find_similar_states(current_state, max_results=10)
        
        if not similar:
            return 0.5, "No similar situations - neutral confidence"
        
        # Step 2: Calculate metrics from similar trades
        # Win Rate = Winners / Total
        wins = sum(1 for exp in similar if exp.get('pnl', 0) > 0)
        win_rate = wins / len(similar)
        
        # Average Profit = Sum of profits / Count
        avg_profit = sum(exp.get('pnl', 0) for exp in similar) / len(similar)
        
        # Step 3: If average profit is negative → Auto reject (0% confidence)
        if avg_profit < 0:
            reason = f"{len(similar)} similar: {win_rate*100:.0f}% WR, ${avg_profit:.0f} avg (NEGATIVE EV - REJECTED)"
            return 0.0, reason
        
        # Profit Score = min(Average Profit / 300, 1.0)
        profit_score = min(avg_profit / 300.0, 1.0)
        
        # Final Confidence = (Win Rate × 90%) + (Profit Score × 10%)
        confidence = (win_rate * 0.90) + (profit_score * 0.10)
        confidence = max(0.0, min(1.0, confidence))
        
        reason = f"{len(similar)} similar: {win_rate*100:.0f}% WR, ${avg_profit:.0f} avg"
        
        return confidence, reason
    
    def find_similar_states(self, current: Dict, max_results: int = 10) -> list:
        """
        Find past experiences with similar market states.
        
        SIMPLIFIED PATTERN MATCHING (7 features):
        ==========================================
        - Flush Size (25%) - How big was the panic move in ticks
        - Velocity (20%) - How fast was the flush in ticks per bar
        - RSI (15%) - How extreme was RSI at entry
        - Volume Spike (15%) - How much volume spiked vs average
        - Distance From Extreme (10%) - How close to the flush low/high
        - Regime Match (10%) - Same market regime or not
        - Hour of Day (5%) - Same time of day or not
        
        Pick the 10 most similar past trades.
        
        EXCLUDED (outcomes/metadata):
          ❌ timestamp, symbol, price, pnl, duration, took_trade,
          ❌ mfe, mae, exit_reason, session, flush_direction,
          ❌ bars_since_flush_start, target_distance_ticks, vwap, atr
        """
        if not self.experiences:
            return []
        
        # Calculate similarity score for each past experience
        scored = []
        for exp in self.experiences:
            # FLAT FORMAT: All fields are at top level
            past = exp
            
            # Calculate distance for each feature (normalized to 0-1 range)
            # Lower score = more similar
            
            # Flush Size (25%) - How big was the panic move
            flush_size_diff = abs(current.get('flush_size_ticks', 0) - past.get('flush_size_ticks', 0)) / 50  # Typical range 0-50 ticks
            
            # Velocity (20%) - How fast was the flush
            flush_velocity_diff = abs(current.get('flush_velocity', 0) - past.get('flush_velocity', 0)) / 10  # Typical range 0-10 ticks/bar
            
            # RSI (15%) - How extreme was RSI at entry
            rsi_diff = abs(current.get('rsi', 50) - past.get('rsi', 50)) / 100  # 0-100 scale
            
            # Volume Spike (15%) - How much volume spiked vs average
            volume_ratio_diff = abs(current.get('volume_climax_ratio', 1) - past.get('volume_climax_ratio', 1)) / 3  # Typical range 0-3
            
            # Distance From Extreme (10%) - How close to the flush low/high
            distance_from_low_diff = abs(current.get('distance_from_flush_low', 0) - past.get('distance_from_flush_low', 0)) / 20  # Typical range 0-20 ticks
            
            # Regime Match (10%) - Binary: same regime or not
            regime_match = 0.0 if current.get('regime', 'NORMAL') == past.get('regime', 'NORMAL') else 1.0
            
            # Hour of Day (5%) - Same time of day or not
            hour_diff = abs(current.get('hour', 12) - past.get('hour', 12)) / 24  # 0-24 scale
            
            # Weighted similarity score (lower is more similar)
            similarity = (
                flush_size_diff * 0.25 +      # Flush Size (25%)
                flush_velocity_diff * 0.20 +  # Velocity (20%)
                rsi_diff * 0.15 +             # RSI (15%)
                volume_ratio_diff * 0.15 +    # Volume Spike (15%)
                distance_from_low_diff * 0.10 + # Distance From Extreme (10%)
                regime_match * 0.10 +         # Regime Match (10%)
                hour_diff * 0.05              # Hour of Day (5%)
            )
            
            scored.append((similarity, exp))
        
        # Sort by similarity (most similar first)
        scored.sort(key=lambda x: x[0])
        
        # Return top N most similar (default 10)
        return [exp for _, exp in scored[:max_results]]
    
    def _calculate_optimal_threshold(self) -> float:
        """
        Learn the optimal confidence threshold from past experiences.
        Strategy: For different threshold levels, calculate what the expected profit would be.
        Choose the threshold that maximizes profit PER TRADE (quality), not total volume.
        SMART TRADING: Be selective, not aggressive. Quality over quantity.
        UPDATED for FLAT FORMAT: experiences have fields at top level.
        """
        if len(self.experiences) < 50:
            # Not enough data - use conservative default (50% minimum confidence)
            return 0.50
        
        # Test different thresholds and see which gives best profit per trade
        # CONSERVATIVE APPROACH: Only test higher thresholds (50%+) for quality over quantity
        threshold_results = {}
        
        # OPTIMIZATION: Pre-calculate confidences once instead of for each threshold
        # This avoids O(n²) complexity by doing the expensive work upfront
        experience_confidences = []
        for exp in self.experiences:
            # FLAT FORMAT: 'took_trade' is at top level
            if not exp.get('took_trade', False):
                continue  # Skip experiences where trade wasn't taken
            
            # Calculate what confidence this trade would have had
            # (based on similar past trades at the time)
            # FLAT FORMAT: All fields are at top level
            state = exp
            
            similar_before = [
                e for e in self.experiences 
                if e.get('timestamp', '') < exp.get('timestamp', '') and 
                   abs(e.get('rsi', 50) - state.get('rsi', 50)) < 10 and
                   abs(e.get('vwap_distance', 1.0) - state.get('vwap_distance', 1.0)) < 0.5
            ]
            
            if len(similar_before) < 5:
                continue  # Not enough similar trades to calculate confidence
            
            # Calculate win rate of similar trades
            # FLAT FORMAT: 'pnl' is at top level (not 'reward')
            wins = sum(1 for e in similar_before if e.get('pnl', e.get('reward', 0)) > 0)
            confidence = wins / len(similar_before) if similar_before else 0.5
            
            # FLAT FORMAT: 'pnl' is at top level (not 'reward')
            pnl_value = exp.get('pnl', exp.get('reward', 0))
            
            experience_confidences.append({
                'confidence': confidence,
                'reward': pnl_value  # Keep 'reward' key for consistency in threshold calculation
            })
        
        
        # Now test each threshold quickly using pre-calculated confidences
        # CONSERVATIVE RANGE: Start at 50% minimum (don't test low thresholds that lead to overtrading)
        for test_threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
            trades_at_threshold = []
            
            # Would we have taken this trade with the test threshold?
            for exp_conf in experience_confidences:
                if exp_conf['confidence'] >= test_threshold:
                    trades_at_threshold.append(exp_conf['reward'])
            
            # Calculate expected value at this threshold
            if len(trades_at_threshold) >= 10:  # Need minimum sample
                win_rate = sum(1 for r in trades_at_threshold if r > 0) / len(trades_at_threshold)
                avg_profit = sum(trades_at_threshold) / len(trades_at_threshold)
                expected_value = avg_profit  # Simpler: just use average profit per trade
                
                threshold_results[test_threshold] = {
                    'expected_value': expected_value,
                    'trades': len(trades_at_threshold),
                    'win_rate': win_rate,
                    'avg_profit': avg_profit
                }
        
        # Find threshold that maximizes PROFIT PER TRADE (quality), not total volume
        # We want the threshold that gives us:
        # - SMART TRADING: ~1-2 trades/day average (not 3-5/day)
        # - HIGH QUALITY: 70%+ win rate (not just 65%)
        # - BEST PROFIT: Maximum average profit per trade
        
        # QUALITY REQUIREMENTS: High standards for signal selection
        valid_thresholds = {
            t: r for t, r in threshold_results.items() 
            if r['trades'] >= 10 and r['win_rate'] >= 0.70  # Min 10 trades, 70%+ WR for quality
        }
        
        if not valid_thresholds:
            # No threshold meets criteria - use conservative default (50% minimum)
            pass  # Silent - using default threshold
            return 0.50
        
        # Choose threshold that maximizes AVERAGE PROFIT PER TRADE (not total profit)
        # This ensures we're selective and only take high-quality setups
        best_threshold = max(
            valid_thresholds.items(), 
            key=lambda x: x[1]['avg_profit']  # Pure quality: best average profit per trade
        )[0]
        best_result = valid_thresholds[best_threshold]
        
        total_profit_potential = best_result['avg_profit'] * best_result['trades']
        pass  # Silent - optimal threshold learned
        
        return best_threshold
    
    def get_position_size_multiplier(self, confidence: float) -> float:
        """
        Get position size multiplier based on confidence.
        Uses smooth interpolation for aggressive scaling with high confidence.
        
        Returns a multiplier (0-1) that scales with user's max_contracts:
        - VERY LOW confidence (0-20%): 20% of max_contracts (minimum viable)
        - LOW confidence (20-40%): 20-40% of max_contracts
        - MEDIUM confidence (40-60%): 40-60% of max_contracts
        - HIGH confidence (60-80%): 60-80% of max_contracts
        - VERY HIGH confidence (80-100%): 80-100% of max_contracts
        
        Examples with max_contracts=25:
        - 10% confidence: 5 contracts (20%)
        - 30% confidence: 7 contracts (30%)
        - 50% confidence: 12 contracts (50%)
        - 70% confidence: 17 contracts (70%)
        - 90% confidence: 22 contracts (90%)
        - 95%+ confidence: 25 contracts (100%)
        
        Examples with max_contracts=3:
        - Low confidence: 1 contract
        - Medium confidence: 2 contracts
        - High confidence: 3 contracts
        
        Args:
            confidence: Confidence level (0-1)
        
        Returns:
            Multiplier value 0.2-1.0 (multiply by max_contracts to get actual size)
        """
        # Smooth linear scaling: confidence directly maps to position size %
        # Minimum 20% (even at 0% confidence, take at least something)
        # Maximum 100% (full confidence = full position)
        
        # Linear interpolation: 0% conf ΓåÆ 20% size, 100% conf ΓåÆ 100% size
        multiplier = 0.2 + (confidence * 0.8)
        
        # Cap between 0.2 and 1.0
        multiplier = max(0.2, min(1.0, multiplier))
        
        return multiplier
    
    def record_outcome(self, state: Dict, took_trade: bool, 
                      pnl: float, duration_minutes: int, 
                      execution_data: Optional[Dict] = None):
        """
        Record the outcome of this signal for learning.
        FLAT FORMAT: All fields at top level (no nested state/action/reward structure).
        
        Args:
            state: Market state when signal triggered (flat dict with 16+ fields)
            took_trade: Whether we took the trade
            pnl: Profit/loss (0 if skipped)
            duration_minutes: How long trade lasted
            execution_data: Execution quality metrics (CRITICAL for RL learning)
                CRITICAL FIELDS (must always include):
                - exit_reason: How trade closed (target_hit/stop_hit/time_exit/regime_change)
                  * RL learns: "Was this a good exit or did we panic?"
                  * Pattern example: "When RSI=70 + regime=RANGING → time exits win 60%"
                - order_type_used: "passive", "aggressive", "mixed"
                  * RL learns: "When volatility is HIGH → use limit orders"
                  * Helps optimize entry execution strategy
                - entry_slippage_ticks: Actual slippage in ticks
                  * RL learns: "Avoid trading during high slippage times"
                  * Critical for live P&L vs theoretical P&L analysis
                
                IMPORTANT FIELDS:
                - mfe: Max Favorable Excursion (dollars) - execution quality
                - mae: Max Adverse Excursion (dollars) - risk management
                - partial_fill: Whether partial fill occurred
                - fill_ratio: Percentage filled (0.66 = 2 of 3)
                - held_full_duration: Whether hit target/stop vs time exit
        """
        # FLAT FORMAT: Merge all fields at top level
        # Start with market state (16 fields from capture_market_state)
        if not isinstance(state, dict):
            logger.error(f"Invalid state type: {type(state)}. Expected dict, skipping experience recording.")
            return
        
        experience = state.copy()
        
        # Add outcome fields at top level (not nested)
        experience['pnl'] = pnl
        experience['duration'] = duration_minutes
        experience['took_trade'] = took_trade
        experience['exploration_rate'] = self.exploration_rate
        
        # Add MFE/MAE if available
        if execution_data:
            if 'mfe' in execution_data:
                experience['mfe'] = execution_data['mfe']
            if 'mae' in execution_data:
                experience['mae'] = execution_data['mae']
            
            # Add other execution metrics at top level
            if 'order_type_used' in execution_data:
                experience['order_type_used'] = execution_data['order_type_used']
            if 'entry_slippage_ticks' in execution_data:
                experience['entry_slippage_ticks'] = execution_data['entry_slippage_ticks']
            if 'partial_fill' in execution_data:
                experience['partial_fill'] = execution_data['partial_fill']
            if 'fill_ratio' in execution_data:
                experience['fill_ratio'] = execution_data['fill_ratio']
            if 'exit_reason' in execution_data:
                experience['exit_reason'] = execution_data['exit_reason']
        
        # Add to memory (learning enabled)
        # USER REQUEST: Only save trades that were actually taken
        if took_trade:
            # DUPLICATE PREVENTION: Check if this experience already exists
            # Use helper method to generate consistent key
            exp_key = self._generate_experience_key(experience, execution_data)
            
            # Check if this exact experience already exists (O(1) lookup with set)
            if exp_key in self.experience_keys:
                pass  # Silent - duplicate prevention working
                # Early return - don't update any state for duplicates
                # Duplicates should not affect recent_trades, streaks, or trigger saves
                return
            
            # Not a duplicate - add to experiences and update all related state
            self.experience_keys.add(exp_key)
            self.experiences.append(experience)
            self.recent_trades.append(pnl)
            
            # Update win/loss streaks for non-duplicate trades
            if pnl > 0:
                self.current_win_streak += 1
                self.current_loss_streak = 0
            else:
                self.current_loss_streak += 1
                self.current_win_streak = 0
            
            # Save every 5 unique trades (auto-save enabled)
            if len(self.experiences) % 5 == 0:
                self.save_experience()
            
            # Log learning progress with execution details
            outcome = "WIN" if pnl > 0 else "LOSS"
            log_msg = f"πΎ [FLAT FORMAT] Recorded {outcome}: ${pnl:.2f} in {duration_minutes}min | Streak: W{self.current_win_streak}/L{self.current_loss_streak}"
            
            # Add MFE/MAE info if available
            if execution_data and ('mfe' in execution_data or 'mae' in execution_data):
                exec_notes = []
                if 'mfe' in execution_data:
                    exec_notes.append(f"MFE: ${execution_data['mfe']:.2f}")
                if 'mae' in execution_data:
                    exec_notes.append(f"MAE: ${execution_data['mae']:.2f}")
                
                # Add other execution quality info if available
                if execution_data.get("order_type_used"):
                    exec_notes.append(f"Order: {execution_data['order_type_used']}")
                if execution_data.get("entry_slippage_ticks", 0) > 0:
                    exec_notes.append(f"Slippage: {execution_data['entry_slippage_ticks']:.1f}t")
                if execution_data.get("partial_fill"):
                    exec_notes.append(f"Partial: {execution_data.get('fill_ratio', 0):.0%}")
                
                if exec_notes:
                    log_msg += f" | {', '.join(exec_notes)}"
            
            pass  # Silent - learning progress is internal (not customer-facing)
    
    def get_stats(self) -> Dict:
        """Get current performance statistics."""
        if not self.recent_trades:
            return {
                'total_signals': self.total_signals,
                'taken': self.signals_taken,
                'skipped': self.signals_skipped,
                'take_rate': 0,
                'recent_pnl': 0,
                'recent_win_rate': 0
            }
        
        wins = sum(1 for pnl in self.recent_trades if pnl > 0)
        total_pnl = sum(self.recent_trades)
        
        return {
            'total_signals': self.total_signals,
            'taken': self.signals_taken,
            'skipped': self.signals_skipped,
            'take_rate': (self.signals_taken / max(1, self.total_signals)) * 100,
            'recent_pnl': total_pnl,
            'recent_win_rate': (wins / len(self.recent_trades)) * 100,
            'exploration_rate': self.exploration_rate * 100
        }
    
    def load_experience(self):
        """Load past experiences from file."""
        # Skip loading if no experience file is configured (cloud-based RL)
        if self.experience_file is None:
            return
        
        if os.path.exists(self.experience_file):
            try:
                with open(self.experience_file, 'r') as f:
                    data = json.load(f)
                    self.experiences = data.get('experiences', [])
                    
                    # Populate experience_keys set for O(1) duplicate detection
                    # Use helper method to ensure consistency with record_outcome
                    self.experience_keys = set()
                    for exp in self.experiences:
                        exp_key = self._generate_experience_key(exp)
                        self.experience_keys.add(exp_key)
                    
                    pass  # Silent - experiences loaded
            except Exception as e:
                logger.error(f"Failed to load experiences: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    def save_experience(self):
        """Save experiences to file."""
        # Skip saving if disabled (e.g., live mode with cloud-only saving)
        if not self.save_local:
            return
        
        # Skip saving if no experience file is configured (cloud-based RL)
        if self.experience_file is None:
            return
        
        try:
            with open(self.experience_file, 'w') as f:
                json.dump({
                    'experiences': self.experiences,
                    'stats': self.get_stats()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save experiences: {e}")
