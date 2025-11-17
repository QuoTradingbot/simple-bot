"""
Hybrid Model V1: Pattern-Matching Confidence Booster
=====================================================
Combines neural network predictions with pattern matching to improve signal approval rate.

Strategy:
- Use neural network as primary predictor
- If neural network gives low confidence (< 30%), check pattern matching
- If pattern matching finds similar winning patterns, boost confidence
- This prevents missing good opportunities when neural network is uncertain

Benefits:
- Maintains neural network's learning capabilities
- Reduces false negatives (missed opportunities)
- Leverages historical pattern matching as safety net
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from signal_confidence import SignalConfidenceRL
import json
from typing import Dict, Tuple, Optional


class HybridConfidenceV1(SignalConfidenceRL):
    """
    Hybrid V1: Neural Network + Pattern Matching Booster
    
    Workflow:
    1. Get neural network prediction
    2. If confidence < boost_threshold (default 30%), check pattern matching
    3. Find similar historical patterns
    4. If similar winning patterns exist, boost confidence
    5. Return max(neural_confidence, boosted_confidence)
    """
    
    def __init__(self, experience_file: str = "data/local_experiences/signal_experiences_v2.json", 
                 backtest_mode: bool = False, 
                 confidence_threshold: Optional[float] = None,
                 exploration_rate: Optional[float] = None,
                 boost_threshold: float = 0.30,  # Boost if NN confidence < 30%
                 pattern_match_threshold: float = 0.70):  # Pattern similarity threshold
        """
        Initialize Hybrid V1 model.
        
        Args:
            boost_threshold: If NN confidence below this, try pattern matching
            pattern_match_threshold: Similarity threshold for pattern matching (0-1)
        """
        super().__init__(experience_file, backtest_mode, confidence_threshold, exploration_rate)
        self.boost_threshold = boost_threshold
        self.pattern_match_threshold = pattern_match_threshold
        self.boosts_applied = 0
        self.total_predictions = 0
        
        print(f"🔀 HYBRID MODEL V1 INITIALIZED")
        print(f"   Boost threshold: {boost_threshold:.0%}")
        print(f"   Pattern match threshold: {pattern_match_threshold:.0%}")
    
    def get_signal_confidence(self, rl_state: Dict, side: str, exploration_rate: float = 0.0) -> Tuple[bool, float, str]:
        """
        Get confidence using hybrid approach.
        
        Returns:
            (take_signal, confidence, reason)
        """
        self.total_predictions += 1
        
        # Step 1: Get neural network prediction
        nn_take_signal, nn_confidence, nn_reason = super().get_signal_confidence(rl_state, side, exploration_rate)
        
        # Step 2: If NN confidence is low, try pattern matching boost
        if nn_confidence < self.boost_threshold:
            # Find similar historical patterns
            similar_patterns = self._find_similar_patterns(rl_state, side)
            
            if similar_patterns:
                # Calculate win rate of similar patterns
                wins = sum(1 for p in similar_patterns if p.get('outcome') == 'WIN' or p.get('pnl', 0) > 0)
                win_rate = wins / len(similar_patterns)
                
                # Boost confidence based on pattern win rate
                if win_rate >= 0.55:  # If similar patterns have 55%+ win rate
                    boosted_confidence = min(0.65, win_rate)  # Cap at 65%
                    
                    if boosted_confidence > nn_confidence:
                        self.boosts_applied += 1
                        reason = f"hybrid_boost (NN:{nn_confidence:.0%} → Pattern:{boosted_confidence:.0%}, {len(similar_patterns)} matches)"
                        
                        # Check against threshold
                        threshold = self.user_threshold if self.user_threshold is not None else self.get_optimal_threshold()
                        take_signal = boosted_confidence >= threshold
                        
                        return (take_signal, boosted_confidence, reason)
        
        # Use neural network result
        return (nn_take_signal, nn_confidence, nn_reason)
    
    def _find_similar_patterns(self, rl_state: Dict, side: str, max_matches: int = 20) -> list:
        """
        Find similar historical patterns using feature matching.
        
        Args:
            rl_state: Current market state
            side: 'long' or 'short'
            max_matches: Maximum number of similar patterns to return
        
        Returns:
            List of similar experiences
        """
        if not self.experiences:
            return []
        
        # Key features for pattern matching
        key_features = ['rsi', 'vwap_distance', 'vix', 'atr', 'volume_ratio', 'hour', 'day_of_week']
        
        # Calculate similarity scores
        similarities = []
        for exp in self.experiences:
            if not isinstance(exp, dict):
                continue
            
            # Must match side
            exp_signal = exp.get('signal', '').lower()
            if exp_signal != side.lower():
                continue
            
            # Calculate feature similarity
            similarity = self._calculate_similarity(rl_state, exp, key_features)
            
            if similarity >= self.pattern_match_threshold:
                similarities.append({
                    'experience': exp,
                    'similarity': similarity
                })
        
        # Sort by similarity and return top matches
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return [s['experience'] for s in similarities[:max_matches]]
    
    def _calculate_similarity(self, state1: Dict, state2: Dict, features: list) -> float:
        """
        Calculate similarity between two states (0 = different, 1 = identical).
        
        Uses weighted feature matching with tolerance ranges.
        """
        total_weight = 0
        matched_weight = 0
        
        feature_weights = {
            'rsi': 1.5,  # RSI is important
            'vwap_distance': 2.0,  # VWAP distance is very important
            'vix': 1.2,  # VIX indicates market regime
            'atr': 1.0,
            'volume_ratio': 0.8,
            'hour': 1.3,  # Time of day matters
            'day_of_week': 0.5  # Day of week less important
        }
        
        for feature in features:
            weight = feature_weights.get(feature, 1.0)
            total_weight += weight
            
            val1 = state1.get(feature)
            val2 = state2.get(feature)
            
            if val1 is None or val2 is None:
                continue
            
            # Calculate feature similarity with tolerance
            if feature == 'rsi':
                # RSI within 10 points is similar
                if abs(val1 - val2) <= 10:
                    matched_weight += weight * (1 - abs(val1 - val2) / 10)
            elif feature == 'vwap_distance':
                # VWAP distance within 0.01 (1%) is similar
                if abs(val1 - val2) <= 0.01:
                    matched_weight += weight * (1 - abs(val1 - val2) / 0.01)
            elif feature == 'vix':
                # VIX within 5 points is similar
                if abs(val1 - val2) <= 5:
                    matched_weight += weight * (1 - abs(val1 - val2) / 5)
            elif feature == 'atr':
                # ATR within 1.0 is similar
                if abs(val1 - val2) <= 1.0:
                    matched_weight += weight * (1 - abs(val1 - val2) / 1.0)
            elif feature == 'volume_ratio':
                # Volume ratio within 0.5 is similar
                if abs(val1 - val2) <= 0.5:
                    matched_weight += weight * (1 - abs(val1 - val2) / 0.5)
            elif feature == 'hour':
                # Same hour or adjacent hour
                if abs(val1 - val2) <= 1:
                    matched_weight += weight * (1 - abs(val1 - val2) / 1)
            elif feature == 'day_of_week':
                # Same day of week
                if val1 == val2:
                    matched_weight += weight
        
        return matched_weight / total_weight if total_weight > 0 else 0.0
    
    def get_stats(self) -> Dict:
        """Get hybrid model statistics."""
        stats = super().get_stats() if hasattr(super(), 'get_stats') else {}
        stats['hybrid_v1'] = {
            'total_predictions': self.total_predictions,
            'boosts_applied': self.boosts_applied,
            'boost_rate': f"{self.boosts_applied / self.total_predictions * 100:.1f}%" if self.total_predictions > 0 else "0%"
        }
        return stats
