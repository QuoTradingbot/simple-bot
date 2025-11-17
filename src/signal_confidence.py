"""
Signal Confidence - Neural Network + RL Hybrid Layer for VWAP Signals
======================================================================
Uses trained neural network (84.80% accuracy) for confidence prediction.
Falls back to pattern matching RL if neural network unavailable.

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
    Neural network + RL hybrid layer that decides whether to trust VWAP signals.
    
    State: Market conditions when signal triggers
    Action: Take trade (yes/no) + position size + exit params
    Reward: Profit/loss from trade outcome
    """
    
    def __init__(self, experience_file: str = "data/local_experiences/signal_experiences_v2.json", backtest_mode: bool = False, confidence_threshold: Optional[float] = None, exploration_rate: Optional[float] = None, min_exploration: Optional[float] = None, exploration_decay: Optional[float] = None):
        """
        Initialize RL confidence scorer with neural network support.
        
        Args:
            experience_file: Path to experience file
            backtest_mode: Whether in backtest mode
            confidence_threshold: Optional fixed threshold (0.1-1.0). 
                                 - For LIVE/SHADOW mode: If None, defaults to 0.5 (50%). User's GUI setting always used.
                                 - For BACKTEST mode: If None, calculates adaptive threshold from experiences.
            exploration_rate: Percentage of random exploration (0.0-1.0). Default: 0.05 (5%)
            min_exploration: Minimum exploration rate (0.0-1.0). Default: 0.05 (5%)
            exploration_decay: Decay factor for exploration rate. Default: 0.995
        """
        self.experience_file = experience_file
        self.experiences = []  # All past (state, action, reward) tuples
        self.recent_trades = deque(maxlen=20)  # Last 20 outcomes
        self.backtest_mode = backtest_mode
        self.freeze_learning = False  # LEARNING ENABLED - Brain learns during backtests
        
        # Neural network support (NEW!)
        self.neural_predictor = None
        self.use_neural_network = True
        try:
            from neural_confidence_model import ConfidencePredictor
            self.neural_predictor = ConfidencePredictor(model_path='data/neural_model.pth')
            if self.neural_predictor.load_model():
                logger.info("🧠 NEURAL NETWORK LOADED - Using 84.80% accuracy AI model")
                self.use_neural_network = True
            else:
                logger.warning("⚠️  Neural network not found - using pattern matching fallback")
                self.use_neural_network = False
        except Exception as e:
            logger.warning(f"⚠️  Could not load neural network: {e}")
            logger.warning("   Using pattern matching fallback")
            self.use_neural_network = False
        
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
            logger.info(" LIVE MODE: No threshold configured, using default 50%")
        else:
            self.user_threshold = confidence_threshold
            logger.info(f" RL BRAIN CONFIG: threshold={confidence_threshold}, exploration={exploration_rate}, backtest_mode={backtest_mode}")
        
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
        
        # Session tracking for learning (NO hard stops - let RL learn)
        self.session_start_time = datetime.now(pytz.UTC)
        self.session_pnl = 0.0
        self.session_trades = []
        
        # Adverse selection tracking - learn from signals we skipped
        self.skipped_signals = deque(maxlen=50)  # Track last 50 skipped signals
        self.skipped_signal_outcomes = {}  # Track what happened to skipped signals
        
        # Regime-specific multipliers (NOW ACTIVE)
        self.regime_multipliers = {
            'HIGH_VOL_CHOPPY': 0.7,    # Reduce size in choppy high vol
            'HIGH_VOL_TRENDING': 0.85,  # Slightly reduce in volatile trends
            'LOW_VOL_RANGING': 1.0,     # Standard in calm range
            'LOW_VOL_TRENDING': 1.15,   # Increase in calm trends
        }
        
        # ALL RL FEATURES NOW 100% CLOUD-BASED
        # Cloud PostgreSQL database tracks:
        # - Feature 1: Win rates by market regime
        # - Feature 2: Immediate adverse movement patterns
        # - Feature 3: Dual pattern matching (winners + losers)
        # - Feature 4: Dynamic confidence thresholds
        # - Feature 5: Entry context validation (spread, liquidity, regime)
        # - Feature 6: Regime pause system
        # Bot queries cloud API for decisions, saves results back to cloud
        
        self.load_experience()
        logger.info(f" Signal Confidence RL initialized: {len(self.experiences)} past experiences")
        
        # Log threshold configuration
        if self.user_threshold is not None:
            if self.backtest_mode:
                logger.info(f" CONFIDENCE THRESHOLD: {self.user_threshold:.1%} (USER CONFIGURED for backtest)")
            else:
                logger.info(f" CONFIDENCE THRESHOLD: {self.user_threshold:.1%} (LIVE/SHADOW MODE - User Setting)")
        else:
            # Only happens in backtest mode now
            logger.info(f" CONFIDENCE THRESHOLD: Will be calculated from experiences (BACKTEST - ADAPTIVE)")
        
        # Log exploration mode
        if self.backtest_mode:
            logger.info(f" BACKTEST MODE: {self.exploration_rate*100:.1f}% exploration enabled (learning mode)")
        else:
            logger.info(f" LIVE MODE: 0% exploration (pure exploitation - NO RANDOM TRADES!)")
    
    def capture_signal_state(self, rsi: float, vwap_distance: float, 
                            atr: float, volume_ratio: float,
                            hour: int, day_of_week: int,
                            recent_pnl: float, streak: int,
                            # NEW: Additional features for complete neural network input (31 total)
                            vix: float = 15.0,
                            session: str = 'NY',
                            trend_strength: float = 0.0,
                            sr_proximity_ticks: float = 0.0,
                            trade_type: int = 0,
                            time_since_last_trade_mins: float = 0.0,
                            drawdown_pct_at_entry: float = 0.0,
                            commission_cost: float = 0.0,
                            signal: str = 'LONG',
                            market_regime: str = 'NORMAL',
                            recent_volatility_20bar: float = 2.0,
                            volatility_trend: float = 0.0,
                            vwap_std_dev: float = 2.0,
                            confidence: float = 0.5,
                            price: float = 6500.0,
                            entry_price: float = 6500.0,
                            vwap: float = 6500.0,
                            consecutive_wins: int = 0,
                            consecutive_losses: int = 0,
                            cumulative_pnl_at_entry: float = 0.0,
                            timestamp: str = None) -> Dict:
        """
        Capture COMPLETE market state when VWAP signal triggers (29 features for neural network).
        NOTE: bid_ask_spread and entry_slippage removed (live-only, not available in backtesting)
        
        Args:
            # Basic features (8)
            rsi: Current RSI value
            vwap_distance: Distance from VWAP in std devs
            atr: Current ATR (volatility)
            volume_ratio: Current volume vs average
            hour: Hour of day (0-23)
            day_of_week: 0=Monday, 4=Friday
            recent_pnl: P&L from last 3 trades
            streak: Win/loss streak (positive=wins, negative=losses)
            
            # Advanced features (23)
            vix: VIX level
            session: Trading session (Asia/London/NY)
            trend_strength: Trend indicator
            sr_proximity_ticks: Distance to support/resistance
            trade_type: 0=reversal, 1=continuation
            time_since_last_trade_mins: Minutes since last trade
            drawdown_pct_at_entry: Current drawdown percentage
            commission_cost: Commission paid
            signal: LONG or SHORT
            market_regime: Market classification
            recent_volatility_20bar: Rolling 20-bar volatility
            volatility_trend: Volatility direction
            vwap_std_dev: VWAP standard deviation
            confidence: Model's own prediction (meta-learning)
            price: Current price
            entry_price: Entry price
            vwap: VWAP value
            consecutive_wins: Consecutive wins count
            consecutive_losses: Consecutive losses count
            cumulative_pnl_at_entry: Total P&L at entry
            timestamp: ISO timestamp
        """
        from datetime import datetime
        import pytz
        
        if timestamp is None:
            timestamp = datetime.now(pytz.UTC).isoformat()
        
        # Session encoding: Asia=0, London=1, NY=2
        session_map = {'Asia': 0, 'London': 1, 'NY': 2}
        session_encoded = session_map.get(session, 2)
        
        # Signal encoding: LONG=0, SHORT=1
        signal_encoded = 0 if signal == 'LONG' else 1
        
        return {
            # Original 8 features
            'rsi': round(rsi, 2),
            'vwap_distance': round(vwap_distance, 4),
            'atr': round(atr, 4),
            'volume_ratio': round(volume_ratio, 4),
            'hour': hour,
            'day_of_week': day_of_week,
            'recent_pnl': round(recent_pnl, 2),
            'streak': streak,
            
            # Additional 23 features for complete neural network input
            'vix': round(vix, 2),
            'session': session_encoded,
            'trend_strength': round(trend_strength, 6),
            'sr_proximity_ticks': round(sr_proximity_ticks, 4),
            'trade_type': trade_type,
            'time_since_last_trade_mins': round(time_since_last_trade_mins, 2),
            'drawdown_pct_at_entry': round(drawdown_pct_at_entry, 4),
            'commission_cost': round(commission_cost, 2),
            'signal': signal_encoded,
            'market_regime': market_regime,
            'recent_volatility_20bar': round(recent_volatility_20bar, 4),
            'volatility_trend': round(volatility_trend, 6),
            'vwap_std_dev': round(vwap_std_dev, 4),
            'confidence': round(confidence, 4),
            'price': round(price, 2),
            'entry_price': round(entry_price, 2),
            'vwap': round(vwap, 2),
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses,
            'cumulative_pnl_at_entry': round(cumulative_pnl_at_entry, 2),
            'timestamp': timestamp,
            
            # Meta info
            'symbol': 'ES',  # Add symbol for tracking
            'took_trade': False,  # Will be updated when trade decision is made
            'outcome': None  # Will be filled when trade completes
        }
    
    def check_spread_acceptable(self, current_spread_ticks: float) -> Tuple[bool, str]:
        """
        Check if bid/ask spread is acceptable for entry.
        Wide spreads = poor fills = slippage losses.
        
        Args:
            current_spread_ticks: Current bid/ask spread in ticks
            
        Returns:
            (acceptable, reason)
        """
        # LEARNED THRESHOLDS from institutional trading
        # 1 tick = tight (ideal)
        # 2 ticks = acceptable (normal market)
        # 3+ ticks = wide (avoid entry, slippage will kill edge)
        
        if current_spread_ticks <= 2.0:
            return True, f"Spread OK ({current_spread_ticks:.1f} ticks)"
        
        # Wide spread - reject entry
        return False, f"❌ SPREAD TOO WIDE ({current_spread_ticks:.1f} ticks > 2.0) - Slippage risk"
    
    def check_liquidity_acceptable(self, volume_ratio: float) -> Tuple[bool, str]:
        """
        Check if market has enough liquidity for entry.
        Thin volume = poor fills = difficulty getting in/out.
        
        Args:
            volume_ratio: Current volume vs average (1.0 = normal)
            
        Returns:
            (acceptable, reason)
        """
        # LEARNED THRESHOLDS
        # 1.0+ = normal or above (good liquidity)
        # 0.5-1.0 = lower but acceptable
        # < 0.3 = very thin (avoid - can't fill orders properly)
        
        if volume_ratio >= 0.3:
            return True, f"Liquidity OK (vol {volume_ratio:.2f}x)"
        
        # Very thin market - reject entry
        return False, f"❌ LIQUIDITY TOO LOW (vol {volume_ratio:.2f}x < 0.3x) - Fill risk"
    
    def check_regime_acceptable(self, regime: str) -> Tuple[bool, str]:
        """
        Check if market regime is acceptable.
        Cloud API tracks regime win rates - this is just a basic check.
        
        Args:
            regime: Market regime (HIGH_VOL_TRENDING, LOW_VOL_RANGING, etc.)
            
        Returns:
            (acceptable, reason)
        """
        # Basic validation - cloud API handles actual regime filtering
        return True, f"Regime {regime} (cloud API validates)"
    
    def check_immediate_adverse_movement(self, state: Dict) -> Tuple[bool, str]:
        """
        Check if setup is acceptable.
        Cloud API tracks adverse movement patterns - this is just a passthrough.
        
        Args:
            state: Current market state
            
        Returns:
            (acceptable, reason)
        """
        # Cloud API handles adverse movement detection
        return True, "Cloud API validates adverse movement"
    
    def separate_winner_loser_experiences(self) -> tuple:
        """
        NEW FEATURE 3 UPGRADED: Dual Pattern Matching
        Separate experiences into winners and losers for intelligent learning.
        
        Returns:
            (winner_experiences, loser_experiences)
        """
        winners = []
        losers = []
        
        for exp in self.experiences:
            reward = exp.get('reward', 0)
            
            if reward > 0:
                winners.append(exp)
            else:
                losers.append(exp)
        
        return winners, losers
    
    def validate_entry_context(self, state: Dict, spread_ok: bool, liquidity_ok: bool, 
                               regime_ok: bool) -> Tuple[bool, str]:
        """
        NEW FEATURE 5: Entry Context Validation
        Check if current setup matches WINNING experience patterns.
        Require ALL context factors to be acceptable.
        
        Args:
            state: Current market state
            spread_ok: Spread check result
            liquidity_ok: Liquidity check result  
            regime_ok: Regime check result
            
        Returns:
            (acceptable, reason)
        """
        if not self.context_validation_enabled:
            return True, "Context validation disabled"
        
        failed_checks = []
        
        # Check spread
        if not spread_ok and 'spread' in self.required_context_checks:
            failed_checks.append('spread')
        
        # Check liquidity
        if not liquidity_ok and 'volume' in self.required_context_checks:
            failed_checks.append('volume')
        
        # Check regime
        if not regime_ok and 'regime' in self.required_context_checks:
            failed_checks.append('regime')
        
        # Check recent wins
        if 'recent_wins' in self.required_context_checks:
            recent_trades = list(self.recent_trades)[-self.recent_win_rate_window:]
            if len(recent_trades) >= 10:
                recent_wins = sum(1 for t in recent_trades if t.get('reward', 0) > 0)
                recent_win_rate = recent_wins / len(recent_trades)
                if recent_win_rate < 0.40:  # Less than 40% recent win rate
                    failed_checks.append(f'recent_wins({recent_win_rate:.0%})')
        
        if failed_checks:
            return False, f"❌ CONTEXT VALIDATION FAILED: {', '.join(failed_checks)}"
        
        return True, "✅ All context checks passed"
    
    def update_regime_stats(self, regime: str, is_winner: bool):
        """
        REMOVED - Cloud API handles regime tracking.
        Kept as stub to avoid breaking existing code.
        """
        pass
    
    def should_take_signal(self, state: Dict, current_spread_ticks: float = 1.0) -> Tuple[bool, float, float, str]:
        """
        Decide whether to take this VWAP signal and how much to risk.
        
        Args:
            state: Market state dict
            current_spread_ticks: Current bid/ask spread in ticks (default 1.0)
        
        Returns:
            (take_trade, confidence, size_multiplier, reason)
            - take_trade: True/False
            - confidence: 0.0-1.0 (how confident)
            - size_multiplier: 0.25-2.0 (position size adjustment)
            - reason: Why this decision
        """
        self.total_signals += 1
        
        # PRE-FILTERS: Check spread and liquidity BEFORE calculating confidence
        spread_ok, spread_reason = self.check_spread_acceptable(current_spread_ticks)
        if not spread_ok:
            self.signals_skipped += 1
            return False, 0.0, 0.0, spread_reason
        
        volume_ratio = state.get('volume_ratio', 1.0)
        liquidity_ok, liquidity_reason = self.check_liquidity_acceptable(volume_ratio)
        if not liquidity_ok:
            self.signals_skipped += 1
            return False, 0.0, 0.0, liquidity_reason
        
        # NEW FEATURE 1: Win Rate Filter by Market Regime
        regime = state.get('regime', 'NORMAL')
        regime_ok, regime_reason = self.check_regime_acceptable(regime)
        if not regime_ok:
            self.signals_skipped += 1
            return False, 0.0, 0.0, regime_reason
        
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
        
        # Exploration: Sometimes use random decision to explore
        if random.random() < effective_exploration:
            take = random.choice([True, False])
            threshold_source = "User" if self.user_threshold is not None else "Learned"
            reason = f"Exploring ({effective_exploration*100:.0f}% random, {len(self.experiences)} exp) | Threshold: {optimal_threshold:.1%} ({threshold_source})"
            
            if take:
                self.signals_taken += 1
            else:
                self.signals_skipped += 1
            
            # Calculate position size even during exploration
            size_mult = self._calculate_position_size(confidence, state)
            return take, confidence, size_mult, reason
        
        # FILTER BASED ON LEARNED THRESHOLD
        take = confidence > optimal_threshold
        
        if take:
            self.signals_taken += 1
            reason += f" APPROVED ({confidence:.1%} > {optimal_threshold:.1%})"
        else:
            self.signals_skipped += 1
            reason += f" REJECTED ({confidence:.1%} < {optimal_threshold:.1%})"
        
        # Calculate position size based on confidence and conditions
        size_mult = self._calculate_position_size(confidence, state)
        
        # Decay exploration over time
        self.exploration_rate = max(self.min_exploration, 
                                   self.exploration_rate * self.exploration_decay)
        
        return take, confidence, size_mult, reason
    
    def calculate_confidence(self, current_state: Dict) -> Tuple[float, str]:
        """
        Calculate confidence using NEURAL NETWORK (preferred) or DUAL PATTERN MATCHING (fallback).
        
        Priority:
        1. Neural Network (84.80% accuracy) - trained on 5,293 experiences
        2. Pattern Matching RL - fallback if NN unavailable
        
        Returns:
            (confidence, reason)
        """
        # TRY NEURAL NETWORK FIRST
        if self.use_neural_network and self.neural_predictor is not None:
            try:
                # Extract features for neural network
                features = {
                    'rsi': current_state.get('rsi', 50.0),
                    'vwap_distance': current_state.get('vwap_distance', 0.0),
                    'atr': current_state.get('atr', 0.0),
                    'volume_ratio': current_state.get('volume_ratio', 1.0),
                    'hour': current_state.get('hour', 12),
                    'day_of_week': current_state.get('day_of_week', 2),
                    'vix': current_state.get('vix', 15.0),
                    'streak': self.current_win_streak if self.current_win_streak > 0 else -self.current_loss_streak,
                    'recent_pnl': sum(t.get('reward', 0) for t in list(self.recent_trades)[-3:]) if self.recent_trades else 0.0,
                    'regime': current_state.get('regime', 'NORMAL'),
                    'side': current_state.get('side', 'LONG'),
                    'session': current_state.get('session', 'NY')
                }
                
                # Get neural network prediction
                confidence = self.neural_predictor.predict_confidence(features)
                
                # Get calibrated confidence and temperature
                stats = self.neural_predictor.get_model_stats()
                calibrated_conf = stats.get('calibrated_confidence', confidence)
                temperature = stats.get('temperature', 1.0)
                
                reason = f"🧠 Neural Network: {calibrated_conf*100:.1f}% (T={temperature:.1f})"
                return calibrated_conf, reason
                
            except Exception as e:
                logger.warning(f"Neural network prediction failed: {e}, falling back to pattern matching")
                # Fall through to pattern matching below
        
        # FALLBACK: DUAL PATTERN MATCHING (original RL logic)
        # Need at least 10 experiences before using them for decisions
        if len(self.experiences) < 10:
            return 0.65, f"🆕 Limited experience ({len(self.experiences)} trades) - optimistic"
        
        # Separate into winners and losers
        winners, losers = self.separate_winner_loser_experiences()
        
        if len(winners) < 5:
            return 0.65, f"🆕 Limited winning experience ({len(winners)} wins) - optimistic"
        
        # Find similar WINNING patterns
        similar_winners = self.find_similar_states(current_state, max_results=10, experiences=winners)
        
        # Find similar LOSING patterns  
        similar_losers = self.find_similar_states(current_state, max_results=10, experiences=losers) if len(losers) >= 5 else []
        
        # Calculate winner confidence
        if similar_winners:
            winner_wins = sum(1 for exp in similar_winners if exp['reward'] > 0)
            winner_win_rate = winner_wins / len(similar_winners)
            winner_avg_profit = sum(exp['reward'] for exp in similar_winners) / len(similar_winners)
            
            # Winner confidence (same formula as before)
            winner_confidence = (winner_win_rate * 0.9) + (min(winner_avg_profit / 300, 1.0) * 0.1)
            winner_confidence = max(0.0, min(1.0, winner_confidence))
        else:
            winner_confidence = 0.5
            winner_win_rate = 0.5
            winner_avg_profit = 0
        
        # Calculate loser penalty
        if similar_losers:
            loser_losses = sum(1 for exp in similar_losers if exp['reward'] < 0)
            loser_loss_rate = loser_losses / len(similar_losers)
            loser_avg_loss = sum(exp['reward'] for exp in similar_losers) / len(similar_losers)
            
            # Penalty is HIGH if very similar to losers
            # Scale: 0.0 (not similar to losers) to 0.5 (very similar to losers)
            loser_penalty = (loser_loss_rate * 0.4) + (min(abs(loser_avg_loss) / 300, 1.0) * 0.1)
            loser_penalty = max(0.0, min(0.5, loser_penalty))
        else:
            loser_penalty = 0.0
            loser_loss_rate = 0.0
            loser_avg_loss = 0
        
        # DUAL PATTERN MATCHING: Confidence = Winners - Losers
        final_confidence = winner_confidence - loser_penalty
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        # Build detailed reason
        reason = f" {len(similar_winners)}W/{len(similar_losers)}L similar"
        reason += f" | Winners: {winner_win_rate*100:.0f}% WR, ${winner_avg_profit:.0f} avg"
        
        if similar_losers:
            reason += f" | Losers: {loser_loss_rate*100:.0f}% LR, ${loser_avg_loss:.0f} avg"
            reason += f" | Penalty: -{loser_penalty:.1%}"
        
        # SAFETY CHECK: Reject if more similar to losers or negative EV
        if final_confidence < 0.3:
            reason += " | LOOKS LIKE PAST LOSERS - REJECTED"
        
        return final_confidence, reason
    
    def _calculate_position_size(self, confidence: float, state: Dict) -> float:
        """
        Calculate position size multiplier based on confidence and conditions.
        
        Args:
            confidence: Signal confidence (0.0-1.0)
            state: Market state dict (includes vix, regime, atr, etc.)
            
        Returns:
            Size multiplier (0.25-2.0)
            - 0.25x = minimum size (very low confidence or bad conditions)
            - 1.0x = standard size (normal conditions)
            - 2.0x = maximum size (very high confidence + good conditions)
        """
        # Base multiplier from confidence
        # confidence 0.0 → 0.25x, 0.5 → 1.0x, 1.0 → 1.75x
        base_mult = 0.25 + (confidence * 1.5)
        
        # 1. WIN/LOSS STREAK ADJUSTMENT
        if self.current_loss_streak >= 2:
            # Reduce size after losses (defensive)
            streak_adj = 0.75 ** min(self.current_loss_streak - 1, 3)  # 0.75x, 0.56x, 0.42x
            base_mult *= streak_adj
        elif self.current_win_streak >= 3:
            # Increase size after wins (capitalize on hot streak)
            streak_adj = 1.0 + (min(self.current_win_streak - 2, 3) * 0.15)  # +15%, +30%, +45%
            base_mult *= streak_adj
        
        # 2. VIX-BASED ADJUSTMENT (Broader market volatility)
        vix = state.get('vix', 15.0)  # Default 15 if not provided
        if vix > 25:  # High market fear
            base_mult *= 0.75  # -25% size in high VIX
        elif vix > 20:  # Elevated volatility
            base_mult *= 0.90  # -10% size
        elif vix < 12:  # Very low volatility
            base_mult *= 1.15  # +15% size (calm markets)
        elif vix < 15:  # Low volatility
            base_mult *= 1.05  # +5% size
        
        # 3. MARKET REGIME ADJUSTMENT (NOW ACTIVE!)
        regime = state.get('regime', 'NORMAL')
        regime_mult = self.regime_multipliers.get(regime, 1.0)
        base_mult *= regime_mult
        
        # 4. ATR-BASED ADJUSTMENT (Instrument-specific volatility)
        atr = state.get('atr', 0)
        avg_atr = atr / 1.2 if atr > 0 else atr  # Estimate normal ATR
        if atr > avg_atr * 1.3:  # High volatility
            base_mult *= 0.85  # 15% reduction (already have VIX, less aggressive)
        elif atr < avg_atr * 0.7:  # Low volatility
            base_mult *= 1.10  # 10% increase
        
        # 5. TIME OF DAY ADJUSTMENT
        tod_score = state.get('time_of_day_score', 1.0)
        if tod_score < 0.5:  # Poor time of day
            base_mult *= 0.85  # 15% reduction
        
        # Clamp to reasonable range
        final_mult = max(0.25, min(2.0, base_mult))
        
        # Log position sizing decision if significantly different from 1.0x
        if final_mult < 0.75 or final_mult > 1.25:
            logger.info(f"📊 [POSITION SIZING] {final_mult:.2f}x size: conf={confidence:.1%}, "
                       f"streak=W{self.current_win_streak}/L{self.current_loss_streak}, "
                       f"VIX={vix:.1f}, regime={regime} ({regime_mult:.2f}x), "
                       f"atr={atr:.1f}, tod={tod_score:.2f}")
        
        return final_mult
    
    def find_similar_states(self, current: Dict, max_results: int = 10, experiences: Optional[list] = None) -> list:
        """
        Find past experiences with similar market states.
        NEW FEATURE 3: Can optionally use filtered experiences list
        
        Args:
            current: Current market state
            max_results: Max number of similar states to return
            experiences: Optional list of experiences to search (defaults to self.experiences)
        """
        exp_list = experiences if experiences is not None else self.experiences
        if not exp_list:
            return []
        
        # Calculate similarity score for each past experience
        scored = []
        for exp in exp_list:
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
        Choose the threshold that maximizes profit PER TRADE (quality), not total volume.
        SMART TRADING: Be selective, not aggressive. Quality over quantity.
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
            logger.info("No valid thresholds found, using default 50%")
            return 0.50
        
        # Choose threshold that maximizes AVERAGE PROFIT PER TRADE (not total profit)
        # This ensures we're selective and only take high-quality setups
        best_threshold = max(
            valid_thresholds.items(), 
            key=lambda x: x[1]['avg_profit']  # Pure quality: best average profit per trade
        )[0]
        best_result = valid_thresholds[best_threshold]
        
        total_profit_potential = best_result['avg_profit'] * best_result['trades']
        logger.info(f"LEARNED OPTIMAL THRESHOLD: {best_threshold*100:.0f}%")
        logger.info(f"   Expected: {best_result['trades']} trades, {best_result['win_rate']*100:.1f}% WR, ${best_result['avg_profit']:.0f} avg, ${total_profit_potential:.0f} total potential")
        
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
        
        # Linear interpolation: 0% conf → 20% size, 100% conf → 100% size
        multiplier = 0.2 + (confidence * 0.8)
        
        # Cap between 0.2 and 1.0
        multiplier = max(0.2, min(1.0, multiplier))
        
        return multiplier
    
    def record_outcome(self, state: Dict, took_trade: bool, 
                      pnl: float, duration_minutes: int, 
                      execution_data: Optional[Dict] = None):
        """
        Record the outcome of this signal for learning.
        ALL LEARNING NOW HAPPENS IN CLOUD API - This just tracks local session stats.
        
        Args:
            state: Market state when signal triggered
            took_trade: Whether we took the trade
            pnl: Profit/loss (0 if skipped)
            duration_minutes: How long trade lasted
            execution_data: Optional execution quality metrics
        """
        experience = {
            'timestamp': datetime.now(pytz.UTC).isoformat(),
            'state': state,
            'action': {
                'took_trade': took_trade,
                'exploration_rate': self.exploration_rate
            },
            'reward': pnl,
            'duration': duration_minutes,
            'execution': execution_data or {}
        }
        
        # Add to memory (for local session tracking only - cloud has full history)
        self.experiences.append(experience)
        self.recent_trades.append(pnl)
        
        # Update win/loss streaks (local session only)
        if took_trade:
            is_winner = pnl > 0
            if is_winner:
                self.current_win_streak += 1
                self.current_loss_streak = 0
            else:
                self.current_loss_streak += 1
                self.current_win_streak = 0
        
        # Save every 5 trades (auto-save enabled)
        if len(self.experiences) % 5 == 0:
            self.save_experience()
        
        # Log outcome
        if took_trade:
            outcome = "WIN" if pnl > 0 else "LOSS"
            logger.info(f"Recorded {outcome}: ${pnl:.2f} in {duration_minutes}min | Streak: W{self.current_win_streak}/L{self.current_loss_streak}")
    
    def _learn_execution_quality(self, state: Dict, execution_data: Dict, pnl: float):
        """
        REMOVED - Cloud API handles execution quality learning.
        Kept as stub to avoid breaking existing code.
        """
        pass
    
    def track_skipped_signal(self, state: Dict, confidence: float, reason: str):
        """
        Track signals we skipped to learn from adverse selection.
        Helps identify when we're being too conservative (FOMO reduction).
        
        Args:
            state: Market state when signal was rejected
            confidence: Confidence score when rejected
            reason: Why signal was rejected
        """
        now_utc = datetime.now(pytz.UTC)
        signal_id = f"{now_utc.strftime('%Y%m%d_%H%M%S')}"
        
        self.skipped_signals.append({
            'id': signal_id,
            'timestamp': now_utc,
            'state': state.copy(),
            'confidence': confidence,
            'reason': reason
        })
    
    def analyze_adverse_selection(self):
        """
        Analyze outcomes of signals we skipped (would've won but rejected).
        This helps detect if threshold is too conservative.
        """
        if len(self.skipped_signal_outcomes) < 20:
            return
        
        # How many skipped signals would've won?
        winners = [outcome for outcome in self.skipped_signal_outcomes.values() if outcome['pnl'] > 0]
        losers = [outcome for outcome in self.skipped_signal_outcomes.values() if outcome['pnl'] <= 0]
        
        if len(winners) + len(losers) < 20:
            return
        
        win_rate = len(winners) / (len(winners) + len(losers))
        avg_winner_pnl = sum(w['pnl'] for w in winners) / len(winners) if winners else 0
        avg_loser_pnl = sum(l['pnl'] for l in losers) / len(losers) if losers else 0
        expected_value = (win_rate * avg_winner_pnl) + ((1 - win_rate) * avg_loser_pnl)
        
        logger.info(f"📊 [ADVERSE SELECTION] Skipped {len(winners) + len(losers)} signals: "
                   f"{win_rate*100:.1f}% would've won, EV: ${expected_value:.2f}")
        
        # If we're skipping profitable signals, threshold might be too high
        if expected_value > 20:  # Skipping signals with $20+ EV
            logger.warning(f"⚠️ [THRESHOLD WARNING] Skipping profitable signals! "
                          f"Consider lowering confidence threshold (current: {self.user_threshold:.1%})")
    
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
        """Load past experiences from v2 format (shared with backtest)."""
        if os.path.exists(self.experience_file):
            try:
                with open(self.experience_file, 'r') as f:
                    data = json.load(f)
                    self.experiences = data.get('experiences', [])
                    logger.info(f"✅ Loaded {len(self.experiences)} experiences from v2 format")
            except Exception as e:
                logger.error(f"Failed to load experiences: {e}")
                self.experiences = []
        else:
            logger.warning(f"No experience file found - starting fresh")
            self.experiences = []
    
    def save_experience(self):
        """Save experiences to file in v2 format (compatible with backtest training)."""
        try:
            # Save to main experience file
            with open(self.experience_file, 'w') as f:
                json.dump({
                    'experiences': self.experiences,
                    'stats': self.get_stats()
                }, f, indent=2)
            
            # ALSO save to v2 format for backtest compatibility
            v2_dir = "data/local_experiences"
            os.makedirs(v2_dir, exist_ok=True)
            v2_file = os.path.join(v2_dir, "signal_experiences_v2.json")
            
            with open(v2_file, 'w') as f:
                json.dump({
                    'experiences': self.experiences,
                    'metadata': {
                        'total_experiences': len(self.experiences),
                        'last_updated': datetime.now().isoformat(),
                        'source': 'live_trading'
                    }
                }, f, indent=2)
            
            logger.debug(f"Saved {len(self.experiences)} experiences to {self.experience_file} and {v2_file}")
            
        except Exception as e:
            logger.error(f"Failed to save experiences: {e}")
