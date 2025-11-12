"""
Local Experience Manager for Fast Backtesting
Loads experiences from local JSON instead of cloud API calls
"""
import json
import os
from typing import Dict, List
from datetime import datetime

class LocalExperienceManager:
    """Manages local experiences for fast backtesting without API calls"""
    
    def __init__(self):
        self.signal_experiences = []
        self.exit_experiences = []
        self.local_dir = "data/local_experiences"
        self.loaded = False
        self.new_signal_experiences = []  # NEW: Track experiences added during backtest
        self.new_exit_experiences = []    # NEW: Track exit experiences added during backtest
        
    def load_experiences(self) -> bool:
        """Load experiences from local JSON files (v2 format with full structure)"""
        if self.loaded:
            return True
            
        try:
            # Use V2 files with full structure
            signal_file = os.path.join(self.local_dir, "signal_experiences_v2.json")
            exit_file = os.path.join(self.local_dir, "exit_experiences_v2.json")
            
            # Fallback to old files if v2 don't exist yet
            if not os.path.exists(signal_file):
                signal_file = os.path.join(self.local_dir, "signal_experiences.json")
            if not os.path.exists(exit_file):
                exit_file = os.path.join(self.local_dir, "exit_experiences.json")
            
            if not os.path.exists(signal_file) or not os.path.exists(exit_file):
                print(f"\n⚠️  Local experiences not found in {self.local_dir}/")
                print(f"   Creating new v2 experience files...")
                return False
            
            # Load signal experiences
            with open(signal_file, 'r') as f:
                data = json.load(f)
                self.signal_experiences = data.get('experiences', [])
            
            # Load exit experiences
            with open(exit_file, 'r') as f:
                data = json.load(f)
                self.exit_experiences = data.get('experiences', [])
            
            self.loaded = True
            print(f"✅ Loaded {len(self.signal_experiences)} signal experiences from local files")
            return True
        except Exception as e:
            print(f"⚠️  Error loading local experiences: {e}")
            return False
    
    def _get_learned_confidence_threshold(self) -> float:
        """
        Learn optimal confidence threshold from historical win rates.
        Finds the threshold that maximizes profit while maintaining acceptable win rate.
        """
        if len(self.signal_experiences) < 50:
            return 0.50  # Default until we have enough data
        
        # Bucket experiences by confidence ranges
        conf_buckets = {
            0.5: [],  # 50-60%
            0.6: [],  # 60-70%
            0.7: [],  # 70-80%
            0.8: [],  # 80-90%
            0.9: []   # 90-95%
        }
        
        for exp in self.signal_experiences:
            if exp.get('took_trade', False):
                confidence = exp.get('confidence', 0.5)  # Get saved confidence value
                pnl = exp.get('pnl', 0)
                
                # Bucket by actual confidence level
                if confidence >= 0.9:
                    conf_buckets[0.9].append(pnl)
                elif confidence >= 0.8:
                    conf_buckets[0.8].append(pnl)
                elif confidence >= 0.7:
                    conf_buckets[0.7].append(pnl)
                elif confidence >= 0.6:
                    conf_buckets[0.6].append(pnl)
                else:
                    conf_buckets[0.5].append(pnl)
        
        # Calculate metrics for each threshold
        best_threshold = 0.50
        best_score = -999999
        
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            trades = conf_buckets[threshold]
            if len(trades) < 10:
                continue
            
            wins = sum(1 for pnl in trades if pnl > 0)
            win_rate = wins / len(trades)
            avg_pnl = sum(trades) / len(trades)
            
            # Score = (win_rate * avg_pnl) - penalize low sample size
            score = (win_rate * avg_pnl) * (len(trades) / 50)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold
    
    def get_signal_confidence(self, rl_state: Dict, signal: str) -> tuple:
        """
        Get ML confidence from local experiences (no API call)
        Returns: (take_signal, confidence, reason)
        """
        if not self.loaded:
            if not self.load_experiences():
                return (False, 0.0, "Local experiences not loaded")
        
        # LEARN FEATURE WEIGHTS from winning vs losing signals
        winners = [e for e in self.signal_experiences if e.get('signal') == signal and e.get('pnl', 0) > 0]
        losers = [e for e in self.signal_experiences if e.get('signal') == signal and e.get('pnl', 0) <= 0]
        
        if len(winners) > 20 and len(losers) > 20:
            # Compare feature distributions
            win_rsi_avg = sum(e.get('rsi', 50) for e in winners) / len(winners)
            lose_rsi_avg = sum(e.get('rsi', 50) for e in losers) / len(losers)
            rsi_importance = abs(win_rsi_avg - lose_rsi_avg) / 100.0
            
            win_vix_avg = sum(e.get('vix', 15) for e in winners) / len(winners)
            lose_vix_avg = sum(e.get('vix', 15) for e in losers) / len(losers)
            vix_importance = abs(win_vix_avg - lose_vix_avg) / 35.0
            
            # Hour distribution difference
            win_hours = {}
            for e in winners:
                h = e.get('hour', 12)
                win_hours[h] = win_hours.get(h, 0) + 1
            lose_hours = {}
            for e in losers:
                h = e.get('hour', 12)
                lose_hours[h] = lose_hours.get(h, 0) + 1
            
            hour_diff = sum(abs(win_hours.get(h, 0)/len(winners) - lose_hours.get(h, 0)/len(losers)) for h in range(24))
            hour_importance = hour_diff / 24.0
            
            # Normalize weights
            total = rsi_importance + vix_importance + hour_importance
            if total > 0:
                rsi_weight = rsi_importance / total
                vix_weight = vix_importance / total
                hour_weight = hour_importance / total
            else:
                rsi_weight = 0.4
                vix_weight = 0.3
                hour_weight = 0.3
        else:
            # Defaults
            rsi_weight = 0.4
            vix_weight = 0.3
            hour_weight = 0.3
        
        # Simple pattern matching (mimics cloud API logic but faster)
        similar_experiences = []
        
        for exp in self.signal_experiences:
            # Filter by signal type
            if exp.get('signal') != signal:
                continue
            
            # Calculate similarity score (LEARNED WEIGHTS)
            rsi_sim = 1.0 - abs(exp.get('rsi', 50) - rl_state.get('rsi', 50)) / 100.0
            vix_sim = 1.0 - abs(exp.get('vix', 15) - rl_state.get('vix', 15)) / 35.0
            hour_sim = 1.0 if exp.get('hour') == rl_state.get('hour') else 0.5
            
            total_sim = rsi_sim * rsi_weight + vix_sim * vix_weight + hour_sim * hour_weight
            
            # LEARNED THRESHOLD
            if len(winners) > 10:
                win_sims = []
                for w in winners[:100]:  # Sample
                    w_rsi_sim = 1.0 - abs(w.get('rsi', 50) - rl_state.get('rsi', 50)) / 100.0
                    w_vix_sim = 1.0 - abs(w.get('vix', 15) - rl_state.get('vix', 15)) / 35.0
                    w_hour_sim = 1.0 if w.get('hour') == rl_state.get('hour') else 0.5
                    w_sim = w_rsi_sim * rsi_weight + w_vix_sim * vix_weight + w_hour_sim * hour_weight
                    win_sims.append(w_sim)
                
                win_sims.sort()
                threshold = win_sims[len(win_sims) // 4] if len(win_sims) > 0 else 0.5
                threshold = max(0.3, min(0.7, threshold))
            else:
                threshold = 0.5
            
            if total_sim > threshold:  # LEARNED threshold
                similar_experiences.append((total_sim, exp))
        
        if not similar_experiences:
            return (False, 0.50, "No similar experiences found")
        
        # Use ALL similar experiences (no limit - use all the data we have!)
        similar_experiences.sort(key=lambda x: x[0], reverse=True)
        top_experiences = [exp for _, exp in similar_experiences]  # Use ALL similar ones
        
        # Calculate win rate
        wins = sum(1 for exp in top_experiences if exp.get('pnl', 0) > 0)
        total = len(top_experiences)
        win_rate = wins / total if total > 0 else 0.5
        
        # Calculate average PnL
        avg_pnl = sum(exp.get('pnl', 0) for exp in top_experiences) / total if total > 0 else 0
        
        # LEARN confidence adjustment from data
        # Instead of fixed 1.1/0.9, learn optimal boost/penalty
        if total > 20:
            # Calculate PnL distribution
            pnls = [exp.get('pnl', 0) for exp in top_experiences]
            pnls.sort()
            
            # If PnL is strongly positive, boost more
            # If PnL is mediocre, boost less
            median_pnl = pnls[len(pnls) // 2]
            
            # Learn boost factor from PnL quality
            if avg_pnl > 200:  # Strong winners
                boost_factor = 1.15
            elif avg_pnl > 100:  # Good winners
                boost_factor = 1.10
            elif avg_pnl > 0:  # Marginal winners
                boost_factor = 1.05
            else:  # Losers
                penalty_factor = 0.85 if avg_pnl < -100 else 0.90
                boost_factor = penalty_factor
        else:
            # Defaults
            boost_factor = 1.1 if avg_pnl > 0 else 0.9
        
        # LEARN TIME-OF-DAY QUALITY - adjust confidence based on hour performance
        current_hour = rl_state.get('hour', 12)
        hour_adjustment = 1.0
        
        if total > 50:
            # Calculate win rate for this specific hour
            hour_trades = [exp for exp in top_experiences if exp.get('hour') == current_hour]
            
            if len(hour_trades) > 5:
                hour_wins = sum(1 for exp in hour_trades if exp.get('pnl', 0) > 0)
                hour_win_rate = hour_wins / len(hour_trades)
                
                # Compare to overall win rate
                if hour_win_rate > win_rate * 1.15:  # 15% better than average
                    hour_adjustment = 1.10  # Boost confidence
                elif hour_win_rate > win_rate * 1.05:  # 5% better
                    hour_adjustment = 1.05
                elif hour_win_rate < win_rate * 0.85:  # 15% worse
                    hour_adjustment = 0.90  # Reduce confidence
                elif hour_win_rate < win_rate * 0.95:  # 5% worse
                    hour_adjustment = 0.95
        
        # Apply LEARNED adjustments (PnL quality + time-of-day)
        # Use ADDITIVE adjustments to prevent over-stacking
        # Example: 60% win_rate + 15% PnL boost + 10% hour boost = 76% (controlled)
        # vs: 60% × 1.15 × 1.10 = 76% (same result but clearer logic)
        confidence = win_rate
        
        # Calculate total adjustment
        pnl_boost = (boost_factor - 1.0)  # e.g., 1.15 → 0.15 (15% boost)
        hour_boost = (hour_adjustment - 1.0)  # e.g., 1.10 → 0.10 (10% boost)
        
        # Apply combined boost (additive to prevent over-stacking)
        if avg_pnl > 0:
            confidence = win_rate * (1.0 + pnl_boost + hour_boost)
            confidence = min(0.95, confidence)  # Cap at 95%
        elif avg_pnl < 0:
            confidence = win_rate * (1.0 + pnl_boost + hour_boost)
            confidence = max(0.05, confidence)  # Floor at 5%
        
        # LEARN OPTIMAL CONFIDENCE THRESHOLD from historical data
        optimal_threshold = self._get_learned_confidence_threshold()
        take_signal = confidence >= optimal_threshold
        reason = f"Local: {wins}W/{total-wins}L from {total} similar ({win_rate:.0%} WR, avg ${avg_pnl:.0f})"
        
        return (take_signal, confidence, reason)
    
    def add_signal_experience(self, rl_state: Dict, took_trade: bool, outcome: Dict):
        """Add new signal experience from backtest (will save to file at end)"""
        experience = {
            # ORIGINAL FIELDS
            'rsi': float(rl_state.get('rsi', 50)),
            'vwap_distance': float(rl_state.get('vwap_distance', 0)),
            'vix': float(rl_state.get('vix', 15.0)),
            'hour': int(rl_state.get('hour', 12)),
            'day_of_week': int(rl_state.get('day_of_week', 0)),
            'volume_ratio': float(rl_state.get('volume_ratio', 1.0)),
            'atr': float(rl_state.get('atr', 2.0)),
            'recent_pnl': float(rl_state.get('recent_pnl', 0.0)),
            'streak': int(rl_state.get('streak', 0)),
            'price': float(rl_state.get('price', 0)),
            'vwap': float(rl_state.get('vwap', 0)),
            'signal': str(rl_state.get('side', 'LONG')).upper(),
            'timestamp': datetime.now().isoformat(),
            'pnl': float(outcome.get('pnl', 0)),
            'outcome': 'WIN' if float(outcome.get('pnl', 0)) > 0 else 'LOSS',
            'took_trade': bool(took_trade),
            'confidence': float(outcome.get('confidence', 0.50)),  # CRITICAL: Save confidence for learning
            # NEW PSYCHOLOGICAL FIELDS
            'cumulative_pnl_at_entry': float(rl_state.get('cumulative_pnl_at_entry', 0.0)),
            'consecutive_wins': int(rl_state.get('consecutive_wins', 0)),
            'consecutive_losses': int(rl_state.get('consecutive_losses', 0)),
            'drawdown_pct_at_entry': float(rl_state.get('drawdown_pct_at_entry', 0.0)),
            'time_since_last_trade_mins': float(rl_state.get('time_since_last_trade_mins', 0.0)),
            # NEW MARKET CONTEXT FIELDS
            'session': str(rl_state.get('session', 'NY')),
            'trend_strength': float(rl_state.get('trend_strength', 0.0)),
            'sr_proximity_ticks': float(rl_state.get('sr_proximity_ticks', 0.0)),
            'trade_type': str(rl_state.get('trade_type', 'continuation')),
            'entry_slippage_ticks': float(rl_state.get('entry_slippage_ticks', 0.0)),
            'commission_cost': float(rl_state.get('commission_cost', 0.0)),
            'bid_ask_spread_ticks': float(rl_state.get('bid_ask_spread_ticks', 0.5)),
        }
        self.new_signal_experiences.append(experience)
    
    def save_new_experiences_to_file(self):
        """Save new experiences accumulated during backtest to local JSON files (v2 format)"""
        if len(self.new_signal_experiences) == 0:
            print("No new signal experiences to save")
            return
        
        # Use V2 file with full structure
        signal_file = os.path.join(self.local_dir, "signal_experiences_v2.json")
        
        # Load existing
        with open(signal_file, 'r') as f:
            data = json.load(f)
            existing_experiences = data.get('experiences', [])
        
        # Add new ones
        all_experiences = existing_experiences + self.new_signal_experiences
        
        # Save back with v2 metadata
        with open(signal_file, 'w') as f:
            json.dump({
                'experiences': all_experiences,
                'count': len(all_experiences),
                'version': '2.0',
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"✅ Saved {len(self.new_signal_experiences)} new signal experiences to local file")
        print(f"   Total experiences now: {len(all_experiences):,}")
        
        # Update in-memory list
        self.signal_experiences = all_experiences
        self.new_signal_experiences = []
    
    def save_experience(self, experience: Dict):
        """Save new experience to local list (will bulk upload later)"""
        exp_type = experience.get('experience_type', 'SIGNAL')
        if exp_type == 'SIGNAL':
            self.signal_experiences.append(experience)
        else:
            self.exit_experiences.append(experience)
    
    def get_experience_count(self) -> Dict:
        """Get count of local experiences"""
        return {
            'signal': len(self.signal_experiences),
            'exit': len(self.exit_experiences),
            'total': len(self.signal_experiences) + len(self.exit_experiences)
        }

# Global instance
local_manager = LocalExperienceManager()
