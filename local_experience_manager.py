"""
Local Experience Manager for Fast Backtesting
Loads experiences from local JSON instead of cloud API calls
"""
import json
import os
from typing import Dict, List
from datetime import datetime
import pytz

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
            
            # Calculate similarity score using ALL 22 available features!
            # Core technical indicators
            rsi_sim = 1.0 - min(abs(exp.get('rsi', 50) - rl_state.get('rsi', 50)) / 100.0, 1.0)
            vix_sim = 1.0 - min(abs(exp.get('vix', 15) - rl_state.get('vix', 15)) / 35.0, 1.0)
            hour_sim = 1.0 if exp.get('hour') == rl_state.get('hour') else 0.5
            
            # Technical indicators
            atr_sim = 1.0 - min(abs(exp.get('atr', 2.0) - rl_state.get('atr', 2.0)) / 5.0, 1.0)
            volume_sim = 1.0 - min(abs(exp.get('volume_ratio', 1.0) - rl_state.get('volume_ratio', 1.0)) / 2.0, 1.0)
            vwap_dist_sim = 1.0 - min(abs(exp.get('vwap_distance', 0.0) - rl_state.get('vwap_distance', 0.0)) / 0.1, 1.0)
            
            # Psychological state
            streak_sim = 1.0 - min(abs(exp.get('streak', 0) - rl_state.get('streak', 0)) / 10.0, 1.0)
            cons_wins_sim = 1.0 - min(abs(exp.get('consecutive_wins', 0) - rl_state.get('consecutive_wins', 0)) / 5.0, 1.0)
            cons_losses_sim = 1.0 - min(abs(exp.get('consecutive_losses', 0) - rl_state.get('consecutive_losses', 0)) / 5.0, 1.0)
            
            # P&L context similarity
            pnl_diff = abs(exp.get('cumulative_pnl_at_entry', 0) - rl_state.get('cumulative_pnl_at_entry', 0))
            pnl_sim = 1.0 - min(pnl_diff / 1000.0, 1.0)
            
            # Session matching
            session_sim = 1.0 if exp.get('session', 'NY') == rl_state.get('session', 'NY') else 0.6
            
            # NEW: Market structure fields
            trend_diff = abs(exp.get('trend_strength', 0.0) - rl_state.get('trend_strength', 0.0))
            trend_sim = 1.0 - min(trend_diff / 2.0, 1.0)
            
            sr_prox_diff = abs(exp.get('sr_proximity_ticks', 0.0) - rl_state.get('sr_proximity_ticks', 0.0))
            sr_prox_sim = 1.0 - min(sr_prox_diff / 20.0, 1.0)
            
            trade_type_sim = 1.0 if exp.get('trade_type', 'continuation') == rl_state.get('trade_type', 'continuation') else 0.3
            
            # NEW: Timing and execution quality
            time_since_diff = abs(exp.get('time_since_last_trade_mins', 0.0) - rl_state.get('time_since_last_trade_mins', 0.0))
            time_since_sim = 1.0 - min(time_since_diff / 60.0, 1.0)
            
            spread_diff = abs(exp.get('bid_ask_spread_ticks', 0.5) - rl_state.get('bid_ask_spread_ticks', 0.5))
            spread_sim = 1.0 - min(spread_diff / 2.0, 1.0)
            
            drawdown_diff = abs(exp.get('drawdown_pct_at_entry', 0.0) - rl_state.get('drawdown_pct_at_entry', 0.0))
            drawdown_sim = 1.0 - min(drawdown_diff / 20.0, 1.0)
            
            # NEW: Day of week patterns (some strategies work better on certain days)
            day_sim = 1.0 if exp.get('day_of_week', 0) == rl_state.get('day_of_week', 0) else 0.7
            
            # NEW: Recent P&L momentum (rolling recent performance)
            recent_pnl_diff = abs(exp.get('recent_pnl', 0.0) - rl_state.get('recent_pnl', 0.0))
            recent_pnl_sim = 1.0 - min(recent_pnl_diff / 500.0, 1.0)
            
            # Weighted similarity using ALL 22 fields
            # Core (45%) + Contextual (20%) + Psychological (15%) + Market Structure (12%) + Execution (8%)
            total_sim = (
                rsi_sim * rsi_weight +          # Core: RSI (learned weight)
                vix_sim * vix_weight +          # Core: VIX (learned weight)  
                hour_sim * hour_weight +        # Core: Hour (learned weight)
                atr_sim * 0.07 +                # Volatility context
                volume_sim * 0.06 +             # Market activity
                vwap_dist_sim * 0.06 +          # Price vs VWAP
                streak_sim * 0.04 +             # Win/loss streak
                cons_wins_sim * 0.03 +          # Consecutive wins
                cons_losses_sim * 0.03 +        # Consecutive losses
                pnl_sim * 0.05 +                # Account state
                session_sim * 0.04 +            # Trading session
                trend_sim * 0.04 +              # Trend strength
                sr_prox_sim * 0.03 +            # S/R proximity
                trade_type_sim * 0.04 +         # Trade type
                time_since_sim * 0.03 +         # Time since last trade
                spread_sim * 0.02 +             # Bid/ask spread
                drawdown_sim * 0.03 +           # Drawdown state
                day_sim * 0.02 +                # Day of week patterns
                recent_pnl_sim * 0.02           # Recent P&L momentum
            )
            
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
        
        # NEW: CONSECUTIVE LOSS ADJUSTMENT - Be more conservative after losses
        consecutive_losses = rl_state.get('consecutive_losses', 0)
        loss_adjustment = 1.0
        
        if total > 30:
            # Analyze how trades after 2+ losses performed
            after_loss_trades = [exp for exp in top_experiences if exp.get('consecutive_losses', 0) >= 2]
            fresh_trades = [exp for exp in top_experiences if exp.get('consecutive_losses', 0) == 0]
            
            if len(after_loss_trades) > 5 and len(fresh_trades) > 5:
                after_loss_wins = sum(1 for exp in after_loss_trades if exp.get('pnl', 0) > 0)
                after_loss_wr = after_loss_wins / len(after_loss_trades)
                
                fresh_wins = sum(1 for exp in fresh_trades if exp.get('pnl', 0) > 0)
                fresh_wr = fresh_wins / len(fresh_trades)
                
                # If after-loss trades underperform, reduce confidence
                if after_loss_wr < fresh_wr * 0.85:  # 15% worse
                    loss_adjustment = 0.85 if consecutive_losses >= 3 else 0.92
                elif after_loss_wr < fresh_wr * 0.95:  # 5% worse
                    loss_adjustment = 0.97
        
        # NEW: DRAWDOWN ADJUSTMENT - Be conservative in deep drawdown
        drawdown_pct = rl_state.get('drawdown_pct_at_entry', 0.0)
        drawdown_adjustment = 1.0
        
        if drawdown_pct < -10.0:  # In 10%+ drawdown
            # Check if trades taken in drawdown perform worse
            if total > 30:
                dd_trades = [exp for exp in top_experiences if exp.get('drawdown_pct_at_entry', 0) < -5.0]
                normal_trades = [exp for exp in top_experiences if exp.get('drawdown_pct_at_entry', 0) >= -5.0]
                
                if len(dd_trades) > 5 and len(normal_trades) > 5:
                    dd_wins = sum(1 for exp in dd_trades if exp.get('pnl', 0) > 0)
                    dd_wr = dd_wins / len(dd_trades)
                    
                    normal_wins = sum(1 for exp in normal_trades if exp.get('pnl', 0) > 0)
                    normal_wr = normal_wins / len(normal_trades)
                    
                    # If drawdown trades underperform, be more selective
                    if dd_wr < normal_wr * 0.85:
                        drawdown_adjustment = 0.90  # 10% penalty in drawdown
                    elif dd_wr < normal_wr * 0.95:
                        drawdown_adjustment = 0.97  # 3% penalty
        
        # NEW: TREND STRENGTH ADJUSTMENT - Boost confidence in strong trends
        trend_strength = rl_state.get('trend_strength', 0.0)
        trend_adjustment = 1.0
        
        if total > 30:
            # Analyze performance in strong vs weak trends
            strong_trend_trades = [exp for exp in top_experiences if abs(exp.get('trend_strength', 0)) > 0.7]
            weak_trend_trades = [exp for exp in top_experiences if abs(exp.get('trend_strength', 0)) < 0.3]
            
            if len(strong_trend_trades) > 5 and len(weak_trend_trades) > 5:
                strong_wins = sum(1 for exp in strong_trend_trades if exp.get('pnl', 0) > 0)
                strong_wr = strong_wins / len(strong_trend_trades)
                
                weak_wins = sum(1 for exp in weak_trend_trades if exp.get('pnl', 0) > 0)
                weak_wr = weak_wins / len(weak_trend_trades)
                
                # If strong trends outperform, boost confidence when trend is strong
                if abs(trend_strength) > 0.7:
                    if strong_wr > weak_wr * 1.15:  # 15% better
                        trend_adjustment = 1.10
                    elif strong_wr > weak_wr * 1.05:  # 5% better
                        trend_adjustment = 1.05
                # If weak trends underperform, penalize weak trend setups
                elif abs(trend_strength) < 0.3:
                    if weak_wr < strong_wr * 0.85:  # 15% worse
                        trend_adjustment = 0.88
                    elif weak_wr < strong_wr * 0.95:  # 5% worse
                        trend_adjustment = 0.95
        
        # NEW: SUPPORT/RESISTANCE PROXIMITY ADJUSTMENT - Caution near key levels
        sr_proximity = rl_state.get('sr_proximity_ticks', 0.0)
        sr_adjustment = 1.0
        
        if total > 30:
            # Analyze performance when close to S/R vs far from S/R
            near_sr_trades = [exp for exp in top_experiences if exp.get('sr_proximity_ticks', 0) < 5]
            far_sr_trades = [exp for exp in top_experiences if exp.get('sr_proximity_ticks', 0) > 15]
            
            if len(near_sr_trades) > 5 and len(far_sr_trades) > 5:
                near_wins = sum(1 for exp in near_sr_trades if exp.get('pnl', 0) > 0)
                near_wr = near_wins / len(near_sr_trades)
                
                far_wins = sum(1 for exp in far_sr_trades if exp.get('pnl', 0) > 0)
                far_wr = far_wins / len(far_sr_trades)
                
                # If near S/R performs better (bounces work), boost when close
                if sr_proximity < 5:
                    if near_wr > far_wr * 1.10:
                        sr_adjustment = 1.08  # S/R bounce is proven
                    elif near_wr < far_wr * 0.90:
                        sr_adjustment = 0.92  # S/R breakout failures
        
        # NEW: TRADE TYPE ADJUSTMENT - Learn continuation vs reversal quality
        trade_type = rl_state.get('trade_type', 'continuation')
        trade_type_adjustment = 1.0
        
        if total > 30:
            cont_trades = [exp for exp in top_experiences if exp.get('trade_type') == 'continuation']
            rev_trades = [exp for exp in top_experiences if exp.get('trade_type') == 'reversal']
            
            if len(cont_trades) > 5 and len(rev_trades) > 5:
                cont_wins = sum(1 for exp in cont_trades if exp.get('pnl', 0) > 0)
                cont_wr = cont_wins / len(cont_trades)
                
                rev_wins = sum(1 for exp in rev_trades if exp.get('pnl', 0) > 0)
                rev_wr = rev_wins / len(rev_trades)
                
                # Boost whichever type performs better
                if trade_type == 'continuation' and cont_wr > rev_wr * 1.10:
                    trade_type_adjustment = 1.07
                elif trade_type == 'continuation' and cont_wr < rev_wr * 0.90:
                    trade_type_adjustment = 0.93
                elif trade_type == 'reversal' and rev_wr > cont_wr * 1.10:
                    trade_type_adjustment = 1.07
                elif trade_type == 'reversal' and rev_wr < cont_wr * 0.90:
                    trade_type_adjustment = 0.93
        
        # NEW: SPREAD QUALITY ADJUSTMENT - Penalize wide spreads
        spread = rl_state.get('bid_ask_spread_ticks', 0.5)
        spread_adjustment = 1.0
        
        if total > 20:
            tight_spread_trades = [exp for exp in top_experiences if exp.get('bid_ask_spread_ticks', 0.5) <= 1.0]
            wide_spread_trades = [exp for exp in top_experiences if exp.get('bid_ask_spread_ticks', 0.5) > 1.5]
            
            if len(tight_spread_trades) > 5 and len(wide_spread_trades) > 5:
                tight_wins = sum(1 for exp in tight_spread_trades if exp.get('pnl', 0) > 0)
                tight_wr = tight_wins / len(tight_spread_trades)
                
                wide_wins = sum(1 for exp in wide_spread_trades if exp.get('pnl', 0) > 0)
                wide_wr = wide_wins / len(wide_spread_trades)
                
                # Penalize wide spreads if they hurt performance
                if spread > 1.5 and wide_wr < tight_wr * 0.90:
                    spread_adjustment = 0.94  # Wide spread penalty
        
        # NEW: PRICE LEVEL ADJUSTMENT - Do certain price levels/ranges perform better?
        current_price = rl_state.get('price', 5000)
        price_adjustment = 1.0
        
        if total > 40:
            # Group by price ranges (e.g., round numbers, quarters)
            # ES example: 5000, 5025, 5050, 5075, 5100 (25-point levels)
            price_bucket = round(current_price / 25) * 25  # Round to nearest 25
            
            # Get trades near this price level vs others
            near_price_trades = [exp for exp in top_experiences if abs(exp.get('price', 5000) - price_bucket) < 15]
            other_trades = [exp for exp in top_experiences if abs(exp.get('price', 5000) - price_bucket) >= 30]
            
            if len(near_price_trades) > 5 and len(other_trades) > 5:
                near_wins = sum(1 for exp in near_price_trades if exp.get('pnl', 0) > 0)
                near_wr = near_wins / len(near_price_trades)
                
                other_wins = sum(1 for exp in other_trades if exp.get('pnl', 0) > 0)
                other_wr = other_wins / len(other_trades)
                
                # If this price level has better/worse history
                if near_wr > other_wr * 1.12:
                    price_adjustment = 1.06  # This price level is favorable
                elif near_wr < other_wr * 0.88:
                    price_adjustment = 0.94  # This price level is unfavorable

        # ADJUSTMENT #10: DIRECTIONAL BIAS - analyze LONG vs SHORT performance patterns
        # Use aggregate historical patterns to favor better-performing direction
        direction_adjustment = 1.0
        
        current_signal = rl_state.get('signal', 'LONG')
        if len(top_experiences) >= 20:
            # Separate experiences by direction
            long_trades = [exp for exp in top_experiences if exp.get('signal') == 'LONG']
            short_trades = [exp for exp in top_experiences if exp.get('signal') == 'SHORT']
            
            if len(long_trades) >= 10 and len(short_trades) >= 10:
                # Calculate win rates by direction
                long_wins = sum(1 for exp in long_trades if exp.get('pnl', 0) > 0)
                long_wr = long_wins / len(long_trades)
                long_avg_pnl = sum(exp.get('pnl', 0) for exp in long_trades) / len(long_trades)
                
                short_wins = sum(1 for exp in short_trades if exp.get('pnl', 0) > 0)
                short_wr = short_wins / len(short_trades)
                short_avg_pnl = sum(exp.get('pnl', 0) for exp in short_trades) / len(short_trades)
                
                # Create composite score (WR + profitability)
                long_score = long_wr * (1 + min(long_avg_pnl / 100, 0.5))
                short_score = short_wr * (1 + min(short_avg_pnl / 100, 0.5))
                
                # If current signal matches better-performing direction
                if current_signal == 'LONG' and long_score > short_score * 1.15:
                    direction_adjustment = 1.08  # Favor LONG based on history
                elif current_signal == 'SHORT' and short_score > long_score * 1.15:
                    direction_adjustment = 1.08  # Favor SHORT based on history
                # If current signal is weaker direction
                elif current_signal == 'LONG' and short_score > long_score * 1.15:
                    direction_adjustment = 0.92  # LONG underperforms historically
                elif current_signal == 'SHORT' and long_score > short_score * 1.15:
                    direction_adjustment = 0.92  # SHORT underperforms historically
        
        # Apply ALL LEARNED adjustments (10 total factors!)
        # Use multiplicative for independent adjustments
        confidence = win_rate
        
        # Calculate combined adjustment (all 10 factors are multiplicative since they're independent)
        total_adjustment = (boost_factor * hour_adjustment * loss_adjustment * drawdown_adjustment * 
                          trend_adjustment * sr_adjustment * trade_type_adjustment * spread_adjustment * 
                          price_adjustment * direction_adjustment)
        
        # Apply adjustment
        if avg_pnl > 0:
            confidence = win_rate * total_adjustment
            confidence = min(0.95, confidence)  # Cap at 95%
        else:
            confidence = win_rate * total_adjustment
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
            'timestamp': datetime.now(pytz.UTC).isoformat(),
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
            # ADDITIONAL FIELDS (for future use - multi-symbol support and price-based learning)
            'symbol': str(rl_state.get('symbol', 'ES')),
            'entry_price': float(rl_state.get('entry_price', rl_state.get('price', 0))),
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
                'last_updated': datetime.now(pytz.UTC).isoformat()
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
