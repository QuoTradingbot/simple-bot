"""
Hybrid Model V2: Adaptive Threshold with Performance Feedback
===============================================================
Dynamically adjusts confidence threshold based on recent trading performance.

Strategy:
- Start with base neural network predictions
- Track recent performance (last 10 trades)
- If winning streak: Lower threshold slightly (take more trades)
- If losing streak: Raise threshold (be more selective)
- Adapts to changing market conditions in real-time

Benefits:
- Self-correcting based on actual results
- Prevents over-trading during unfavorable conditions
- Capitalizes on favorable conditions
- Maintains neural network intelligence while adding adaptability
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from signal_confidence import SignalConfidenceRL
import json
from typing import Dict, Tuple, Optional
from collections import deque


class HybridConfidenceV2(SignalConfidenceRL):
    """
    Hybrid V2: Adaptive Threshold Model
    
    Key Features:
    - Tracks rolling window of recent trades (default: last 10)
    - Calculates recent win rate and P&L
    - Adjusts threshold based on performance:
      * Winning streak (≥60% win rate) → Lower threshold by 10%
      * Losing streak (<40% win rate) → Raise threshold by 10%
      * Neutral (40-60%) → Use base threshold
    - Prevents threshold from going too extreme (20% - 80% range)
    """
    
    def __init__(self, experience_file: str = "data/local_experiences/signal_experiences_v2.json",
                 backtest_mode: bool = False,
                 confidence_threshold: Optional[float] = None,
                 exploration_rate: Optional[float] = None,
                 window_size: int = 10,  # Look at last 10 trades
                 adjustment_factor: float = 0.10):  # Adjust threshold by 10%
        """
        Initialize Hybrid V2 model.
        
        Args:
            window_size: Number of recent trades to track for adaptation
            adjustment_factor: How much to adjust threshold (0.10 = 10%)
        """
        super().__init__(experience_file, backtest_mode, confidence_threshold, exploration_rate)
        self.window_size = window_size
        self.adjustment_factor = adjustment_factor
        self.recent_outcomes = deque(maxlen=window_size)  # Store recent trade outcomes
        self.threshold_history = []  # Track threshold changes
        self.current_adaptive_threshold = confidence_threshold or 0.50
        
        print(f"📊 HYBRID MODEL V2 INITIALIZED")
        print(f"   Window size: {window_size} trades")
        print(f"   Adjustment factor: {adjustment_factor:.0%}")
        print(f"   Base threshold: {self.current_adaptive_threshold:.0%}")
    
    def record_trade_outcome(self, pnl: float, confidence: float):
        """
        Record a trade outcome to adapt threshold.
        
        Args:
            pnl: Trade P&L (positive = win, negative = loss)
            confidence: Confidence level of the trade
        """
        self.recent_outcomes.append({
            'pnl': pnl,
            'win': pnl > 0,
            'confidence': confidence
        })
        
        # Recalculate adaptive threshold
        self._update_adaptive_threshold()
    
    def _update_adaptive_threshold(self):
        """
        Update the adaptive threshold based on recent performance.
        """
        if len(self.recent_outcomes) < 3:  # Need at least 3 trades
            return
        
        # Calculate recent win rate
        wins = sum(1 for outcome in self.recent_outcomes if outcome['win'])
        win_rate = wins / len(self.recent_outcomes)
        
        # Calculate recent P&L
        total_pnl = sum(outcome['pnl'] for outcome in self.recent_outcomes)
        avg_pnl = total_pnl / len(self.recent_outcomes)
        
        # Get base threshold
        base_threshold = self.user_threshold if self.user_threshold is not None else 0.50
        
        # Adjust threshold based on performance
        new_threshold = base_threshold
        adjustment_reason = "neutral"
        
        if win_rate >= 0.60 and avg_pnl > 0:
            # Winning streak - lower threshold to take more trades
            new_threshold = base_threshold * (1 - self.adjustment_factor)
            adjustment_reason = f"winning_streak (WR:{win_rate:.0%}, PnL:${avg_pnl:.0f})"
        elif win_rate < 0.40 or avg_pnl < -50:
            # Losing streak - raise threshold to be more selective
            new_threshold = base_threshold * (1 + self.adjustment_factor)
            adjustment_reason = f"losing_streak (WR:{win_rate:.0%}, PnL:${avg_pnl:.0f})"
        
        # Clamp threshold to reasonable range (20% - 80%)
        new_threshold = max(0.20, min(0.80, new_threshold))
        
        # Only update if changed
        if new_threshold != self.current_adaptive_threshold:
            self.threshold_history.append({
                'old': self.current_adaptive_threshold,
                'new': new_threshold,
                'reason': adjustment_reason,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'sample_size': len(self.recent_outcomes)
            })
            
            self.current_adaptive_threshold = new_threshold
    
    def get_signal_confidence(self, rl_state: Dict, side: str, exploration_rate: float = 0.0) -> Tuple[bool, float, str]:
        """
        Get confidence using adaptive threshold.
        
        Returns:
            (take_signal, confidence, reason)
        """
        # Get neural network prediction
        _, nn_confidence, nn_reason = super().get_signal_confidence(rl_state, side, exploration_rate)
        
        # Use adaptive threshold instead of fixed threshold
        take_signal = nn_confidence >= self.current_adaptive_threshold
        
        # Build reason string
        if self.threshold_history and len(self.recent_outcomes) >= 3:
            last_adjustment = self.threshold_history[-1]
            reason = f"adaptive_threshold ({last_adjustment['reason']}, threshold:{self.current_adaptive_threshold:.0%})"
        else:
            reason = f"adaptive_threshold (base:{self.current_adaptive_threshold:.0%})"
        
        return (take_signal, nn_confidence, reason)
    
    def get_current_threshold(self) -> float:
        """Get the current adaptive threshold."""
        return self.current_adaptive_threshold
    
    def get_stats(self) -> Dict:
        """Get hybrid model statistics."""
        stats = super().get_stats() if hasattr(super(), 'get_stats') else {}
        
        # Recent performance
        recent_stats = {}
        if len(self.recent_outcomes) > 0:
            wins = sum(1 for o in self.recent_outcomes if o['win'])
            recent_stats = {
                'window_size': len(self.recent_outcomes),
                'win_rate': f"{wins / len(self.recent_outcomes) * 100:.1f}%",
                'total_pnl': f"${sum(o['pnl'] for o in self.recent_outcomes):.2f}",
                'avg_pnl': f"${sum(o['pnl'] for o in self.recent_outcomes) / len(self.recent_outcomes):.2f}"
            }
        
        stats['hybrid_v2'] = {
            'current_threshold': f"{self.current_adaptive_threshold:.0%}",
            'threshold_adjustments': len(self.threshold_history),
            'recent_performance': recent_stats
        }
        
        return stats
    
    def print_threshold_history(self):
        """Print the history of threshold adjustments."""
        if not self.threshold_history:
            print("No threshold adjustments yet")
            return
        
        print(f"\n📊 THRESHOLD ADJUSTMENT HISTORY:")
        print(f"{'='*80}")
        for i, adjustment in enumerate(self.threshold_history[-10:], 1):  # Show last 10
            print(f"{i}. {adjustment['old']:.0%} → {adjustment['new']:.0%}")
            print(f"   Reason: {adjustment['reason']}")
            print(f"   Win Rate: {adjustment['win_rate']:.0%} | Avg P&L: ${adjustment['avg_pnl']:.2f}")
            print(f"   Sample: {adjustment['sample_size']} trades")
            print()
