"""
Signal Confidence - RL Layer for VWAP Signals
==============================================
Learns which VWAP bounce signals to trust vs skip.

Keeps your hardcoded VWAP/RSI entry logic, but adds intelligence:
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
    Reinforcement learning layer that decides whether to trust VWAP signals.
    
    State: Market conditions when signal triggers
    Action: Take trade (yes/no) + position size + exit params
    Reward: Profit/loss from trade outcome
    """
    
    def __init__(self, experience_file: str = "data/signal_experience.json", backtest_mode: bool = False):
        """Initialize RL confidence scorer."""
        self.experience_file = experience_file
        self.experiences = []  # All past (state, action, reward) tuples
        self.recent_trades = deque(maxlen=20)  # Last 20 outcomes
        self.backtest_mode = backtest_mode
        self.freeze_learning = False  # LEARNING ENABLED - Brain 2 learns during backtests
        
        # Random exploration enabled - no fixed seed for natural learning
        
        # Learning parameters
        self.exploration_rate = 0.30  # 30% random exploration for learning
        self.min_exploration = 0.05  # Never go below 5%
        self.exploration_decay = 0.995
        
        # Cached optimal threshold (recalculate only when needed)
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
        logger.info(f" Signal Confidence RL initialized: {len(self.experiences)} past experiences")
        
        # Log exploration mode
        if self.backtest_mode:
            logger.info(f" BACKTEST MODE: 5% exploration enabled (learning mode)")
        else:
            logger.info(f" LIVE MODE: 0% exploration (pure exploitation - NO RANDOM TRADES!)")
    
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
        # BACKTEST MODE: 5% exploration (keeps learning without hurting performance)
        effective_exploration = 0.05 if self.backtest_mode else 0.0
        
        # ALWAYS calculate confidence from experiences
        confidence, reason = self.calculate_confidence(state)
        
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
        
        # Exploration: Sometimes use random decision to explore
        if random.random() < effective_exploration:
            take = random.choice([True, False])
            reason = f"Exploring ({effective_exploration*100:.0f}% random, {len(self.experiences)} exp) | Threshold: {optimal_threshold:.1%}"
            
            if take:
                self.signals_taken += 1
            else:
                self.signals_skipped += 1
            
            return take, confidence, reason
        
        # FILTER BASED ON LEARNED THRESHOLD
        take = confidence > optimal_threshold
        
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
        
        Returns:
            (confidence, reason)
        """
        # Need at least 10 experiences before using them for decisions
        # Otherwise, a few early losses will make Brain 2 reject everything!
        if len(self.experiences) < 10:
            return 0.65, f"🆕 Limited experience ({len(self.experiences)} trades) - optimistic"
        
        # Find similar past situations
        similar = self.find_similar_states(current_state, max_results=10)
        
        if not similar:
            return 0.5, " No similar situations - neutral confidence"
        
        # Calculate win rate from similar situations
        wins = sum(1 for exp in similar if exp['reward'] > 0)
        win_rate = wins / len(similar)
        
        # Average profit from similar situations
        avg_profit = sum(exp['reward'] for exp in similar) / len(similar)
        
        # SAFETY CHECK: Reject signals with negative expected value
        if avg_profit < 0:
            reason = f" {len(similar)} similar: {win_rate*100:.0f}% WR, ${avg_profit:.0f} avg (NEGATIVE EV - REJECTED)"
            return 0.0, reason
        
        # MORE AGGRESSIVE confidence formula to match best-run quality
        # Win rate is king (90% weight), profit is secondary (10% weight)
        # Examples: 80% WR + $100 avg = 73% confidence
        #          100% WR + $150 avg = 93% confidence
        #           50% WR + $45 avg = 45% confidence (was 37.7% - now higher bar)
        confidence = (win_rate * 0.9) + (min(avg_profit / 300, 1.0) * 0.1)
        confidence = max(0.0, min(1.0, confidence))
        
        reason = f" {len(similar)} similar: {win_rate*100:.0f}% WR, ${avg_profit:.0f} avg"
        
        return confidence, reason
    
    def find_similar_states(self, current: Dict, max_results: int = 10) -> list:
        """Find past experiences with similar market states."""
        if not self.experiences:
            return []
        
        # Calculate similarity score for each past experience
        scored = []
        for exp in self.experiences:
            past = exp['state']
            
            # Calculate distance in each dimension (with safety checks for missing keys)
            rsi_diff = abs(current.get('rsi', 50) - past.get('rsi', 50)) / 100
            vwap_diff = abs(current.get('vwap_distance', 0) - past.get('vwap_distance', 0)) / 5
            atr_diff = abs(current.get('atr', 1) - past.get('atr', 1)) / 20
            volume_diff = abs(current.get('volume_ratio', 1) - past.get('volume_ratio', 1)) / 3
            hour_diff = abs(current.get('hour', 12) - past.get('hour', 12)) / 24
            streak_diff = abs(current.get('streak', 0) - past.get('streak', 0)) / 10
            
            # Weighted similarity score (lower is more similar)
            similarity = (
                rsi_diff * 0.25 +
                vwap_diff * 0.25 +
                atr_diff * 0.20 +
                volume_diff * 0.15 +
                hour_diff * 0.10 +
                streak_diff * 0.05
            )
            
            scored.append((similarity, exp))
        
        # Sort by similarity (most similar first)
        scored.sort(key=lambda x: x[0])
        
        # Return top N most similar
        return [exp for _, exp in scored[:max_results]]
    
    def _calculate_optimal_threshold(self) -> float:
        """
        Learn the optimal confidence threshold from past experiences.
        Strategy: For different threshold levels, calculate what the expected profit would be.
        Choose the threshold that maximizes total expected profit.
        BIAS TOWARDS MORE TRADES to match best-run performance.
        """
        if len(self.experiences) < 50:
            # Not enough data - use conservative default
            return 0.30
        
        # Test different thresholds and see which gives best expected value
        # EXPANDED RANGE: Test lower thresholds (30%, 35%, 40%) to capture more trades
        threshold_results = {}
        
        # OPTIMIZATION: Pre-calculate confidences once instead of for each threshold
        # This avoids O(n²) complexity by doing the expensive work upfront
        experience_confidences = []
        for exp in self.experiences:
            if not exp.get('action', {}).get('took_trade', False):
                continue  # Skip experiences where trade wasn't taken
            
            # Calculate what confidence this trade would have had
            # (based on similar past trades at the time)
            state = exp['state']
            similar_before = [
                e for e in self.experiences 
                if e['timestamp'] < exp['timestamp'] and 
                   abs(e['state'].get('rsi', 50) - state.get('rsi', 50)) < 10 and
                   abs(e['state'].get('vwap_distance', 1.0) - state.get('vwap_distance', 1.0)) < 0.5 and
                   e['state'].get('side') == state.get('side')
            ]
            
            if len(similar_before) < 5:
                continue  # Not enough similar trades to calculate confidence
            
            # Calculate win rate of similar trades
            wins = sum(1 for e in similar_before if e['reward'] > 0)
            confidence = wins / len(similar_before) if similar_before else 0.5
            
            experience_confidences.append({
                'confidence': confidence,
                'reward': exp['reward']
            })
        
        # Now test each threshold quickly using pre-calculated confidences
        for test_threshold in [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
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
        
        # Find threshold with MAXIMUM TOTAL PROFIT (not just high win rate)
        # We want the threshold that gives us the best balance of:
        # - Enough trades (volume) - TARGET 20-30 trades for best-run performance
        # - Good win rate (accuracy) - MAINTAIN 65%+ to match best run quality
        # - Positive expected value (profitability)
        
        # BALANCED REQUIREMENTS: Quality over quantity, but not too conservative
        valid_thresholds = {
            t: r for t, r in threshold_results.items() 
            if r['trades'] >= 15 and r['win_rate'] >= 0.65  # Min 15 trades, 65%+ WR for quality
        }
        
        if not valid_thresholds:
            # No threshold meets criteria - use moderate default
            logger.info("No valid thresholds found, using default 35%")
            return 0.35
        
        # Choose threshold that maximizes TOTAL expected profit (avg_profit * num_trades)
        # Slight bonus for more trades, but prioritize profit quality
        best_threshold = max(
            valid_thresholds.items(), 
            key=lambda x: (x[1]['expected_value'] * x[1]['trades']) + (x[1]['trades'] * 20)  # Small bonus $20 per trade
        )[0]
        best_result = valid_thresholds[best_threshold]
        
        total_profit_potential = best_result['avg_profit'] * best_result['trades']
        logger.info(f"LEARNED OPTIMAL THRESHOLD: {best_threshold*100:.0f}%")
        logger.info(f"   Expected: {best_result['trades']} trades, {best_result['win_rate']*100:.1f}% WR, ${best_result['avg_profit']:.0f} avg, ${total_profit_potential:.0f} total potential")
        
        return best_threshold
    
    def get_position_size_multiplier(self, confidence: float) -> float:
        """
        Get position size multiplier based on confidence.
        Uses smooth interpolation for better sizing precision.
        
        Returns a multiplier (0-1) that scales with user's max_contracts:
        - LOW confidence (0-40%): ~33% of max_contracts
        - MEDIUM confidence (40-70%): ~33-67% of max_contracts (interpolated)
        - HIGH confidence (70-100%): ~67-100% of max_contracts (interpolated)
        
        Examples:
        - max_contracts=3: LOW=1, MEDIUM=2, HIGH=3
        - max_contracts=10: LOW=3, MEDIUM=6, HIGH=10
        - max_contracts=1: Always 1 (confidence just validates entry)
        
        Args:
            confidence: Confidence level (0-1)
        
        Returns:
            Multiplier value 0.33-1.0 (multiply by max_contracts to get actual size)
        """
        # Smooth interpolation based on confidence tiers
        # confidence 0-40%: 1 unit (33% of max)
        # confidence 40-70%: 1-2 units (33-67% of max, interpolated)
        # confidence 70-100%: 2-3 units (67-100% of max, interpolated)
        
        if confidence < 0.4:
            # Low confidence: ~33% of max
            contracts = 1.0
        elif confidence < 0.7:
            # Medium confidence: interpolate 33% → 67% of max
            range_pct = (confidence - 0.4) / 0.3
            contracts = 1.0 + range_pct
        else:
            # High confidence: interpolate 67% → 100% of max
            range_pct = (confidence - 0.7) / 0.3
            contracts = 2.0 + range_pct
        
        # Convert to multiplier (1 unit = 0.33, 2 units = 0.67, 3 units = 1.0)
        multiplier = contracts / 3.0
        
        return multiplier
    
    def record_outcome(self, state: Dict, took_trade: bool, 
                      pnl: float, duration_minutes: int, 
                      execution_data: Optional[Dict] = None):
        """
        Record the outcome of this signal for learning.
        
        Args:
            state: Market state when signal triggered
            took_trade: Whether we took the trade
            pnl: Profit/loss (0 if skipped)
            duration_minutes: How long trade lasted
            execution_data: Optional execution quality metrics (for live trading learning)
                - order_type_used: "passive", "aggressive", "mixed"
                - entry_slippage_ticks: Actual slippage in ticks
                - partial_fill: Whether partial fill occurred
                - fill_ratio: Percentage filled (0.66 = 2 of 3)
                - exit_reason: How trade closed
                - held_full_duration: Whether hit target/stop vs time exit
        """
        experience = {
            'timestamp': datetime.now().isoformat(),
            'state': state,
            'action': {
                'took_trade': took_trade,
                'exploration_rate': self.exploration_rate
            },
            'reward': pnl,
            'duration': duration_minutes,
            'execution': execution_data or {}  # Store execution quality data
        }
        
        # Add to memory (learning enabled)
        self.experiences.append(experience)
        self.recent_trades.append(pnl)
        
        # Update win/loss streaks
        if took_trade:
            if pnl > 0:
                self.current_win_streak += 1
                self.current_loss_streak = 0
            else:
                self.current_loss_streak += 1
                self.current_win_streak = 0
        
        # Save every 5 trades (auto-save enabled)
        if len(self.experiences) % 5 == 0:
            self.save_experience()
        
        # Log learning progress with execution details
        if took_trade:
            outcome = "WIN" if pnl > 0 else "LOSS"
            log_msg = f"Recorded {outcome}: ${pnl:.2f} in {duration_minutes}min | Streak: W{self.current_win_streak}/L{self.current_loss_streak}"
            
            # Add execution quality info if available
            if execution_data:
                exec_notes = []
                if execution_data.get("order_type_used"):
                    exec_notes.append(f"Order: {execution_data['order_type_used']}")
                if execution_data.get("entry_slippage_ticks", 0) > 0:
                    exec_notes.append(f"Slippage: {execution_data['entry_slippage_ticks']:.1f}t")
                if execution_data.get("partial_fill"):
                    exec_notes.append(f"Partial: {execution_data.get('fill_ratio', 0):.0%}")
                
                if exec_notes:
                    log_msg += f" | Exec: {', '.join(exec_notes)}"
            
            logger.info(log_msg)
    
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
        logger.info(f"[DEBUG] Attempting to load experiences from: {self.experience_file}")
        logger.info(f"[DEBUG] File exists check: {os.path.exists(self.experience_file)}")
        
        if os.path.exists(self.experience_file):
            try:
                logger.info(f"[DEBUG] Opening file...")
                with open(self.experience_file, 'r') as f:
                    logger.info(f"[DEBUG] Loading JSON...")
                    data = json.load(f)
                    logger.info(f"[DEBUG] JSON loaded successfully. Keys: {list(data.keys())}")
                    self.experiences = data.get('experiences', [])
                    logger.info(f" Loaded {len(self.experiences)} past signal experiences")
            except Exception as e:
                logger.error(f"Failed to load experiences: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning(f"[DEBUG] Experience file not found: {self.experience_file}")
    
    def save_experience(self):
        """Save experiences to file."""
        try:
            with open(self.experience_file, 'w') as f:
                json.dump({
                    'experiences': self.experiences,
                    'stats': self.get_stats()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save experiences: {e}")
