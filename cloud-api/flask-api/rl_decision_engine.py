"""
Cloud RL Decision Engine
=========================
Makes trading decisions based on the collective RL brain (7,559+ experiences).
This runs ONLY on the cloud - user bots never see this logic.
"""

import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import math

logger = logging.getLogger(__name__)


class CloudRLDecisionEngine:
    """
    Cloud-side RL decision engine that analyzes market states and decides
    whether user bots should take trades.
    
    User bots send market conditions → This analyzes with RL brain → Returns decision
    """
    
    def __init__(self, experiences: List[Dict]):
        """
        Initialize with collective RL brain.
        
        Args:
            experiences: List of all trade experiences from Azure Blob Storage
        """
        self.experiences = experiences
        self.default_confidence_threshold = 0.5  # 50% default
        
        logger.info(f"🧠 Cloud RL Engine initialized with {len(experiences)} experiences")
    
    def should_take_signal(self, state: Dict) -> Tuple[bool, float, str]:
        """
        Decide if a user bot should take this trade based on market state.
        
        Args:
            state: Current market conditions {rsi, vwap_distance, atr, volume_ratio, 
                   hour, day_of_week, recent_pnl, streak, side, price}
        
        Returns:
            (take_trade, confidence, reason)
        """
        # Calculate confidence using dual pattern matching
        confidence, reason = self.calculate_confidence(state)
        
        # Decision: take if confidence > threshold
        take_trade = confidence >= self.default_confidence_threshold
        
        if take_trade:
            decision_reason = f"✅ TAKE ({confidence:.1%} confidence) - {reason}"
        else:
            decision_reason = f"❌ SKIP ({confidence:.1%} confidence) - {reason}"
        
        return take_trade, confidence, decision_reason
    
    def calculate_confidence(self, current_state: Dict) -> Tuple[float, str]:
        """
        Calculate confidence using SIMPLE PATTERN MATCHING (matches running local bot).
        
        Formula: Find 10 most similar experiences, calculate their win rate directly.
        Using 10 samples provides good balance between statistical significance and relevance.
        
        Returns:
            (confidence, reason)
        """
        num_samples = 10  # Number of similar experiences to analyze
        
        # Need at least num_samples experiences before using them for decisions
        if len(self.experiences) < num_samples:
            return 0.65, f"🆕 Limited experience ({len(self.experiences)} trades) - optimistic"
        
        # Find most similar experiences (regardless of win/loss)
        similar = self.find_similar_states(current_state, max_results=num_samples)
        
        if len(similar) < num_samples:
            return 0.65, f"🆕 Limited similar experience ({len(similar)} trades) - optimistic"
        
        # Calculate win rate and average PNL from similar trades
        wins = sum(1 for exp in similar if exp.get('reward', 0) > 0)
        win_rate = wins / len(similar)
        avg_pnl = sum(exp.get('reward', 0) for exp in similar) / len(similar)
        
        # Simple confidence = win rate directly
        confidence = win_rate
        
        # Build reason
        reason = f"{len(similar)} similar: {win_rate*100:.0f}% WR, ${avg_pnl:.0f} avg"
        
        # Safety check: Reject if negative expected value
        if avg_pnl < 0:
            confidence = 0.0
            reason += " (NEGATIVE EV - REJECTED)"
        
        return confidence, reason
    
    def find_similar_states(self, current_state: Dict, max_results: int = 10, 
                           experiences: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Find past experiences with similar market conditions.
        
        USES SAME FORMULA AS LOCAL BOT:
        - Weighted similarity score (NOT Euclidean distance)
        - Lower score = more similar
        - Weights: RSI 25%, VWAP 25%, ATR 20%, Volume 15%, Hour 10%, Streak 5%
        
        Args:
            current_state: Current market state
            max_results: Max number of similar experiences to return
            experiences: Optional subset of experiences to search (for winner/loser filtering)
        
        Returns:
            List of similar experiences, sorted by similarity (most similar first)
        """
        exp_list = experiences if experiences is not None else self.experiences
        
        if not exp_list:
            return []
        
        # Calculate similarity score for each past experience
        scored = []
        for exp in exp_list:
            past = exp.get('state', {})
            
            # Calculate distance in each dimension (with safety checks for missing keys)
            # EXACT SAME FORMULA AS LOCAL BOT
            rsi_diff = abs(current_state.get('rsi', 50) - past.get('rsi', 50)) / 100
            vwap_diff = abs(current_state.get('vwap_distance', 0) - past.get('vwap_distance', 0)) / 5
            atr_diff = abs(current_state.get('atr', 1) - past.get('atr', 1)) / 20
            volume_diff = abs(current_state.get('volume_ratio', 1) - past.get('volume_ratio', 1)) / 3
            hour_diff = abs(current_state.get('hour', 12) - past.get('hour', 12)) / 24
            streak_diff = abs(current_state.get('streak', 0) - past.get('streak', 0)) / 10
            
            # Weighted similarity score (lower is more similar)
            # EXACT WEIGHTS AS LOCAL BOT: RSI 25%, VWAP 25%, ATR 20%, Volume 15%, Hour 10%, Streak 5%
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
    
    def record_outcome(self, state: Dict, took_trade: bool, pnl: float, duration: float) -> Dict:
        """
        Record a trade outcome to the RL brain.
        This gets called after user bot executes and reports results.
        
        Args:
            state: Market state when signal occurred
            took_trade: Whether trade was taken
            pnl: Profit/loss in dollars
            duration: Trade duration in seconds
        
        Returns:
            Experience dictionary to be saved to Azure Blob
        """
        experience = {
            'timestamp': datetime.now().isoformat(),
            'state': state,
            'action': {
                'took_trade': took_trade,
                'exploration_rate': 0.0  # Cloud never explores (always exploitation)
            },
            'reward': pnl,
            'duration': duration
        }
        
        return experience
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the RL brain.
        
        Returns:
            Dictionary with win rate, avg reward, total experiences, etc.
        """
        if not self.experiences:
            return {
                'total_experiences': 0,
                'win_rate': 0.0,
                'avg_reward': 0.0,
                'total_reward': 0.0
            }
        
        trades_taken = [exp for exp in self.experiences if exp.get('action', {}).get('took_trade', False)]
        
        if not trades_taken:
            return {
                'total_experiences': len(self.experiences),
                'win_rate': 0.0,
                'avg_reward': 0.0,
                'total_reward': 0.0
            }
        
        winners = sum(1 for exp in trades_taken if exp.get('reward', 0) > 0)
        total_reward = sum(exp.get('reward', 0) for exp in trades_taken)
        
        return {
            'total_experiences': len(self.experiences),
            'trades_taken': len(trades_taken),
            'win_rate': winners / len(trades_taken) if trades_taken else 0.0,
            'avg_reward': total_reward / len(trades_taken) if trades_taken else 0.0,
            'total_reward': total_reward
        }
