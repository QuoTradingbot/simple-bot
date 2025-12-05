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
    
    def _generate_experience_key(self, experience: Dict) -> str:
        """
        Generate a unique key for duplicate detection using 16-field structure.
        
        Args:
            experience: The experience dictionary with 16 fields
        
        Returns:
            Hash string for O(1) duplicate detection
        """
        import hashlib
        
        # ALL 16 fields that make an experience unique
        key_fields = [
            # The 12 Pattern Matching Fields
            'flush_size_ticks', 'flush_velocity', 'volume_climax_ratio', 'flush_direction',
            'rsi', 'distance_from_flush_low', 'reversal_candle', 'no_new_extreme',
            'vwap_distance_ticks', 'regime', 'session', 'hour',
            # The 4 Metadata Fields
            'symbol', 'timestamp', 'pnl', 'took_trade'
        ]
        
        # Build key from all significant values
        values = []
        for field in key_fields:
            val = experience.get(field)
            
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
        
        # LOG CONFIDENCE FOR ALL SIGNALS
        threshold_source = "User" if self.user_threshold is not None else "Learned"
        logger.info(f"[RL Confidence] Signal confidence: {confidence:.1%} vs threshold {optimal_threshold:.1%} ({threshold_source}) - {reason}")
        
        # Print for diagnostics
        print(f"[RL Decision Check] Confidence {confidence*100:.1f}% vs Threshold {optimal_threshold*100:.1f}% = {'PASS' if take else 'FAIL'}")
        
        # Exploration: Give rejected signals a chance to be taken
        # This allows the system to learn from signals it would normally skip
        if not take and random.random() < effective_exploration:
            # This signal was rejected, but exploration gives it a chance
            take = True
            reason = f"Exploring ({effective_exploration*100:.0f}% chance for rejected signals, {len(self.experiences)} exp) | Threshold: {optimal_threshold:.1%} ({threshold_source})"
            self.signals_taken += 1
            logger.info(f"[RL Decision] EXPLORATION TRADE TAKEN - {reason}")
            print(f"[RL Decision] ✅ EXPLORATION TRADE (was rejected but exploring)")
            return take, confidence, reason
        
        # Normal behavior: use threshold decision
        if take:
            self.signals_taken += 1
            reason += f" APPROVED ({confidence:.1%} > {optimal_threshold:.1%})"
            logger.info(f"[RL Decision] ✅ SIGNAL APPROVED - {reason}")
            print(f"[RL Decision] ✅ TRADE APPROVED (confidence > threshold)")
        else:
            self.signals_skipped += 1
            reason += f" REJECTED ({confidence:.1%} < {optimal_threshold:.1%})"
            logger.info(f"[RL Decision] ❌ SIGNAL REJECTED - {reason}")
            print(f"[RL Decision] ❌ TRADE REJECTED (confidence < threshold)")
        
        # Decay exploration over time
        self.exploration_rate = max(self.min_exploration, 
                                   self.exploration_rate * self.exploration_decay)
        
        return take, confidence, reason
    
    def calculate_confidence(self, current_state: Dict) -> Tuple[float, str]:
        """
        Calculate confidence based on similar past experiences.
        
        CONFIDENCE FORMULA (80/20 Rule):
        ==================
        Step 1: Find 10 most similar past trades
        Step 2: Calculate from those similar trades:
          - Win Rate = Winners / Total
          - Average Profit = Sum of profits / Count
          - Profit Score = min(Average Profit / 300, 1.0)
          - Final Confidence = (Win Rate × 80%) + (Profit Score × 20%)
        Step 3: If average profit is negative → Auto reject (0% confidence)
        
        Example: 8 wins out of 10 = 80% WR, $120 avg profit
          Profit Score = 120/300 = 0.40
          Confidence = (0.80 × 0.80) + (0.40 × 0.20) = 0.64 + 0.08 = 72%
        
        Returns:
            (confidence, reason)
        """
        # Need at least 10 experiences before using them for decisions
        if len(self.experiences) < 10:
            logger.debug(f"[RL] Limited experience: {len(self.experiences)}/10 required - using safety default 35%")
            return 0.35, f"Limited experience ({len(self.experiences)} trades) - safety default"
        
        # Step 1: Find 10 most similar past trades
        similar = self.find_similar_states(current_state, max_results=10)
        
        if not similar:
            logger.warning(f"[RL] No similar trades found despite {len(self.experiences)} experiences - pattern matching may be too strict")
            logger.debug(f"[RL] Current state: flush_size={current_state.get('flush_size_ticks')}, "
                        f"velocity={current_state.get('flush_velocity')}, "
                        f"rsi={current_state.get('rsi')}, "
                        f"regime={current_state.get('regime')}")
            # Print to console for diagnostics
            print(f"[RL Confidence] 35.0% (DEFAULT) - No similar trades found despite {len(self.experiences)} experiences")
            return 0.35, "No similar situations - safety default"
        
        # Step 2: Calculate metrics from similar trades
        # Win Rate = Winners / Total
        wins = sum(1 for exp in similar if exp.get('pnl', 0) > 0)
        win_rate = wins / len(similar)
        
        # Average Profit = Sum of profits / Count
        avg_profit = sum(exp.get('pnl', 0) for exp in similar) / len(similar)
        
        logger.debug(f"[RL] Found {len(similar)} similar trades: {wins} wins, {len(similar)-wins} losses, "
                    f"WR={win_rate*100:.0f}%, avg_profit=${avg_profit:.2f}")
        
        # Step 3: If average profit is negative → Auto reject (0% confidence)
        if avg_profit < 0:
            reason = f"{len(similar)} similar: {win_rate*100:.0f}% WR, ${avg_profit:.0f} avg (NEGATIVE EV - REJECTED)"
            logger.info(f"[RL] Signal REJECTED due to negative expected value: {reason}")
            return 0.0, reason
        
        # Profit Score = min(Average Profit / 300, 1.0)
        profit_score = min(avg_profit / 300.0, 1.0)
        
        # Final Confidence = (Win Rate × 80%) + (Profit Score × 20%)
        confidence = (win_rate * 0.80) + (profit_score * 0.20)
        confidence = max(0.0, min(1.0, confidence))
        
        reason = f"{len(similar)} similar: {win_rate*100:.0f}% WR, ${avg_profit:.0f} avg"
        logger.debug(f"[RL] Calculated confidence: {confidence:.1%} - {reason}")
        
        # Print to console for diagnostics (shows in backtest)
        print(f"[RL Confidence] {confidence*100:.1f}% - {reason}")
        
        return confidence, reason
    
    def find_similar_states(self, current: Dict, max_results: int = 10) -> list:
        """
        Find past experiences with similar market states.
        
        UPDATED PATTERN MATCHING (11 features for live and backtesting):
        ================================================================
        
        Primary Flush Signals (50% total):
        - Flush Size (20%) - How big was the panic move in ticks
        - Velocity (15%) - How fast was the flush in ticks per bar
        - Volume Climax (10%) - How much volume spiked vs average
        - Flush Direction (5%) - Binary: same direction or not
        
        Entry Quality (25% total):
        - RSI (8%) - How extreme was RSI at entry
        - Distance From Flush Low (7%) - How close to the flush low/high
        - Reversal Candle (5%) - Binary: both have reversal candle or not
        - No New Extreme (5%) - Binary: both have no new extreme or not
        
        Market Context (15% total):
        - VWAP Distance (8%) - Distance from VWAP in ticks
        - Regime Match (7%) - Binary: same market regime or not
        
        Time Context (10% total):
        - Session (6%) - Binary: both ETH or both RTH
        - Hour of Day (4%) - Same time of day or not
        
        EXCLUDED (outcomes/metadata):
          ❌ timestamp, symbol, price, pnl, duration, took_trade,
          ❌ mfe, mae, exit_reason, bars_since_flush_start, 
          ❌ stop_distance_ticks, target_distance_ticks, risk_reward_ratio, atr
        """
        if not self.experiences:
            return []
        
        # DEBUG: Log current state for diagnosis
        logger.debug(f"[RL Pattern Matching] Searching for similar trades among {len(self.experiences)} experiences")
        logger.debug(f"[RL Pattern Matching] Current: flush_size={current.get('flush_size_ticks', 0):.1f}, "
                    f"velocity={current.get('flush_velocity', 0):.1f}, "
                    f"direction={current.get('flush_direction', 'NONE')}, "
                    f"rsi={current.get('rsi', 50):.1f}, "
                    f"regime={current.get('regime', 'NORMAL')}")
        
        # Calculate similarity score for each past experience
        scored = []
        for exp in self.experiences:
            # FLAT FORMAT: All fields are at top level
            past = exp
            
            # Calculate distance for each feature (normalized to 0-1 range)
            # Lower score = more similar
            
            # === PRIMARY FLUSH SIGNALS (50% total) ===
            
            # Flush Size (20%) - Normalize by dividing difference by 50
            flush_size_diff = abs(current.get('flush_size_ticks', 0) - past.get('flush_size_ticks', 0)) / 50.0
            
            # Velocity (15%) - Normalize by dividing difference by 10
            flush_velocity_diff = abs(current.get('flush_velocity', 0) - past.get('flush_velocity', 0)) / 10.0
            
            # Volume Climax (10%) - Normalize by dividing difference by 3
            volume_climax_diff = abs(current.get('volume_climax_ratio', 1) - past.get('volume_climax_ratio', 1)) / 3.0
            
            # Flush Direction (5%) - Score 0 if match, score 1 if different
            flush_direction_match = 0.0 if current.get('flush_direction', 'NEUTRAL') == past.get('flush_direction', 'NEUTRAL') else 1.0
            
            # === ENTRY QUALITY (25% total) ===
            
            # RSI (8%) - Normalize by dividing difference by 100
            rsi_diff = abs(current.get('rsi', 50) - past.get('rsi', 50)) / 100.0
            
            # Distance From Flush Low (7%) - Normalize by dividing difference by 20
            distance_from_low_diff = abs(current.get('distance_from_flush_low', 0) - past.get('distance_from_flush_low', 0)) / 20.0
            
            # Reversal Candle (5%) - Score 0 if both match, score 1 if different
            reversal_candle_match = 0.0 if current.get('reversal_candle', False) == past.get('reversal_candle', False) else 1.0
            
            # No New Extreme (5%) - Score 0 if both match, score 1 if different
            no_new_extreme_match = 0.0 if current.get('no_new_extreme', False) == past.get('no_new_extreme', False) else 1.0
            
            # === MARKET CONTEXT (15% total) ===
            
            # VWAP Distance (8%) - Normalize by dividing absolute difference by 100
            vwap_distance_diff = abs(current.get('vwap_distance_ticks', 0) - past.get('vwap_distance_ticks', 0)) / 100.0
            
            # Regime Match (7%) - Score 0 if regimes match, score 1 if different
            regime_match = 0.0 if current.get('regime', 'NORMAL') == past.get('regime', 'NORMAL') else 1.0
            
            # === TIME CONTEXT (10% total) ===
            
            # Session (6%) - Score 0 if both ETH or both RTH, score 1 if different
            session_match = 0.0 if current.get('session', 'RTH') == past.get('session', 'RTH') else 1.0
            
            # Hour (4%) - Normalize by dividing difference by 24
            hour_diff = abs(current.get('hour', 12) - past.get('hour', 12)) / 24.0
            
            # Weighted similarity score (lower is more similar)
            similarity = (
                # Primary Flush Signals (50%)
                flush_size_diff * 0.20 +           # 20%
                flush_velocity_diff * 0.15 +       # 15%
                volume_climax_diff * 0.10 +        # 10%
                flush_direction_match * 0.05 +     # 5%
                # Entry Quality (25%)
                rsi_diff * 0.08 +                  # 8%
                distance_from_low_diff * 0.07 +    # 7%
                reversal_candle_match * 0.05 +     # 5%
                no_new_extreme_match * 0.05 +      # 5%
                # Market Context (15%)
                vwap_distance_diff * 0.08 +        # 8%
                regime_match * 0.07 +              # 7%
                # Time Context (10%)
                session_match * 0.06 +             # 6%
                hour_diff * 0.04                   # 4%
            )
            
            scored.append((similarity, exp))
        
        # Sort by similarity (most similar first)
        scored.sort(key=lambda x: x[0])
        
        # DEBUG: Show similarity scores of top matches
        if scored:
            top_5 = scored[:min(5, len(scored))]
            logger.debug(f"[RL Pattern Matching] Top 5 similarity scores: {[f'{s:.3f}' for s, _ in top_5]}")
            best_match = top_5[0][1]
            logger.debug(f"[RL Pattern Matching] Best match: flush_size={best_match.get('flush_size_ticks', 0):.1f}, "
                        f"velocity={best_match.get('flush_velocity', 0):.1f}, "
                        f"direction={best_match.get('flush_direction', 'NONE')}, "
                        f"rsi={best_match.get('rsi', 50):.1f}, "
                        f"pnl=${best_match.get('pnl', 0):.2f}")
        else:
            logger.warning(f"[RL Pattern Matching] No scored experiences - this should not happen!")
        
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
        SIMPLIFIED 16-FIELD STRUCTURE.
        
        Args:
            state: Market state when signal triggered (14 fields: 12 pattern matching + 2 metadata)
            took_trade: Whether we took the trade
            pnl: Profit/loss (0 if skipped)
            duration_minutes: IGNORED - not stored in simplified structure
            execution_data: IGNORED - not stored in simplified structure
        """
        # FLAT FORMAT: Merge all fields at top level
        # Start with market state (14 fields from capture_market_state)
        if not isinstance(state, dict):
            logger.error(f"Invalid state type: {type(state)}. Expected dict, skipping experience recording.")
            return
        
        experience = state.copy()
        
        # Add outcome fields at top level (only pnl and took_trade)
        experience['pnl'] = pnl
        experience['took_trade'] = took_trade
        
        # Add to memory (learning enabled)
        # USER REQUEST: Only save trades that were actually taken
        if took_trade:
            # DUPLICATE PREVENTION: Check if this experience already exists
            # Use helper method to generate consistent key
            exp_key = self._generate_experience_key(experience)
            
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
            
            # Log learning progress
            outcome = "WIN" if pnl > 0 else "LOSS"
            log_msg = f"πΎ [16-FIELD] Recorded {outcome}: ${pnl:.2f} | Streak: W{self.current_win_streak}/L{self.current_loss_streak}"
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
