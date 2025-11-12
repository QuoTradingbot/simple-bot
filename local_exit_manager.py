"""
Local Exit Manager for Fast Backtesting
Provides same adaptive exit logic as cloud API but using local experiences
"""
import json
import os
from datetime import datetime
from typing import Dict, Optional, List
import pytz

class LocalExitManager:
    """Manages adaptive exits using local exit experiences (no API calls)"""
    
    def __init__(self, verbose=False):
        self.exit_experiences = []
        self.local_dir = "data/local_experiences"
        self.loaded = False
        self.new_exit_experiences = []  # NEW: Track new exit experiences from backtest
        self.verbose = verbose  # Control learning output
        self.learned_insights = []  # Collect insights
        
    def load_experiences(self) -> bool:
        """Load exit experiences from local JSON file (v2 format with full structure)"""
        if self.loaded:
            return True
            
        try:
            # Use V2 file with full structure
            exit_file = os.path.join(self.local_dir, "exit_experiences_v2.json")
            
            # Fallback to old file if v2 doesn't exist yet
            if not os.path.exists(exit_file):
                exit_file = os.path.join(self.local_dir, "exit_experiences.json")
            
            if not os.path.exists(exit_file):
                print(f"\n⚠️  Local exit experiences not found: {exit_file}")
                print(f"   Creating new v2 exit experience file...")
                return False
            
            with open(exit_file, 'r') as f:
                data = json.load(f)
                self.exit_experiences = data.get('experiences', [])
            
            self.loaded = True
            print(f"✅ Loaded {len(self.exit_experiences):,} exit experiences from local files")
            return True
            
        except Exception as e:
            print(f"❌ Error loading local exit experiences: {e}")
            return False
    
    def get_adaptive_exit_params(self, market_state: Dict, position: Dict, 
                                  entry_confidence: float) -> Optional[Dict]:
        """
        Get adaptive exit parameters using local pattern matching.
        Mimics cloud API behavior but uses local data.
        
        Args:
            market_state: Current market conditions (vix, atr, hour, etc.)
            position: Position info (entry_price, side, entry_time)
            entry_confidence: ML confidence when entering trade
            
        Returns:
            Dict with exit parameters or None if no match
        """
        if not self.loaded:
            if not self.load_experiences():
                return self._get_default_params()  # Return defaults if can't load
        
        if len(self.exit_experiences) == 0:
            return self._get_default_params()  # Return defaults if no data yet
        
        # LEARN FEATURE WEIGHTS - determine which factors predict wins
        side = position.get('side')
        winners = [e for e in self.exit_experiences if e.get('outcome', {}).get('win', False) and e.get('side') == side]
        losers = [e for e in self.exit_experiences if not e.get('outcome', {}).get('win', False) and e.get('side') == side]
        
        if len(winners) > 20 and len(losers) > 20:
            # Compare feature distributions between winners and losers
            # Features with bigger differences get higher weights
            
            win_vix_avg = sum(e.get('vix', 18) for e in winners) / len(winners)
            lose_vix_avg = sum(e.get('vix', 18) for e in losers) / len(losers)
            vix_importance = abs(win_vix_avg - lose_vix_avg) / 35.0
            
            win_atr_avg = sum(e.get('atr', 2) for e in winners) / len(winners)
            lose_atr_avg = sum(e.get('atr', 2) for e in losers) / len(losers)
            atr_importance = abs(win_atr_avg - lose_atr_avg) / 5.0
            
            # Hour importance - how different are the hour distributions?
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
            
            # Normalize to sum to 1.0
            total = vix_importance + atr_importance + hour_importance
            if total > 0:
                vix_weight = vix_importance / total
                atr_weight = atr_importance / total
                hour_weight = hour_importance / total
            else:
                vix_weight = 0.4
                atr_weight = 0.3
                hour_weight = 0.3
        else:
            # Not enough data - use balanced weights
            vix_weight = 0.4
            atr_weight = 0.3
            hour_weight = 0.3
        
        # Find similar exit experiences
        similar_exits = []
        
        for exp in self.exit_experiences:
            # Match by side
            if exp.get('side') != position.get('side'):
                continue
            
            # Calculate similarity based on market conditions
            vix_sim = 1.0 - min(abs(exp.get('vix', 18) - market_state.get('vix', 18)) / 35.0, 1.0)
            atr_sim = 1.0 - min(abs(exp.get('atr', 2.0) - market_state.get('atr', 2.0)) / 5.0, 1.0)
            hour_sim = 1.0 if exp.get('hour') == market_state.get('hour') else 0.5
            
            # Overall similarity (LEARNED WEIGHTS)
            similarity = vix_sim * vix_weight + atr_sim * atr_weight + hour_sim * hour_weight
            
            # LEARNED THRESHOLD - what similarity do winners typically have?
            if len(winners) > 10:
                # Calculate similarity distribution for winners
                win_sims = []
                for w in winners[:100]:  # Sample for performance
                    w_vix_sim = 1.0 - min(abs(w.get('vix', 18) - market_state.get('vix', 18)) / 35.0, 1.0)
                    w_atr_sim = 1.0 - min(abs(w.get('atr', 2) - market_state.get('atr', 2)) / 5.0, 1.0)
                    w_hour_sim = 1.0 if w.get('hour') == market_state.get('hour') else 0.5
                    w_sim = w_vix_sim * vix_weight + w_atr_sim * atr_weight + w_hour_sim * hour_weight
                    win_sims.append(w_sim)
                
                # Use 25th percentile (keep top 75% of winners)
                win_sims.sort()
                threshold = win_sims[len(win_sims) // 4] if len(win_sims) > 0 else 0.5
                threshold = max(0.3, min(0.7, threshold))  # Reasonable bounds
            else:
                threshold = 0.5
            
            if similarity >= threshold:  # LEARNED threshold
                similar_exits.append({
                    'exp': exp,
                    'similarity': similarity,
                    'win': exp.get('outcome', {}).get('win', False)
                })
        
        if len(similar_exits) == 0:
            # No similar experiences - return default params
            return self._get_default_params()
        
        # Sort by similarity and USE ALL similar exits (no limit!)
        similar_exits.sort(key=lambda x: x['similarity'], reverse=True)
        top_exits = similar_exits  # Use ALL similar ones, not just top 100
        
        # Calculate win rate from ALL similar exits
        wins = sum(1 for e in top_exits if e['win'])
        win_rate = wins / len(top_exits) if len(top_exits) > 0 else 0.5
        
        # LEARN ACTUAL PARAMETERS from winning vs losing exits
        # Instead of using fixed formulas, calculate what ACTUALLY worked
        
        # Separate winners and losers
        winning_exits = [e['exp'] for e in top_exits if e['win']]
        losing_exits = [e['exp'] for e in top_exits if not e['win']]
        
        # Calculate MEDIAN parameters from WINNING exits (robust to outliers)
        if len(winning_exits) > 5:  # Need at least 5 for statistical significance
            # Extract all values
            be_values = [e.get('exit_params', {}).get('breakeven_threshold_ticks', 12) for e in winning_exits]
            trail_values = [e.get('exit_params', {}).get('trailing_distance_ticks', 15) for e in winning_exits]
            stop_values = [e.get('exit_params', {}).get('stop_mult', 4.0) for e in winning_exits]
            
            # Use MEDIAN (not mean) - naturally filters outliers
            be_values.sort()
            trail_values.sort()
            stop_values.sort()
            
            avg_winner_be = be_values[len(be_values) // 2]  # Median
            avg_winner_trail = trail_values[len(trail_values) // 2]  # Median
            avg_winner_stop = stop_values[len(stop_values) // 2]  # Median
        else:
            # Fallback to defaults (not enough data yet)
            avg_winner_be = 12
            avg_winner_trail = 15
            avg_winner_stop = 4.0
        
        # Adjust based on confidence
        # LEARN optimal confidence scaling from winning exits
        # Instead of fixed 80-120%, learn what aggressiveness works best
        confidence_adjustment = entry_confidence  # 0.0 to 1.0
        
        if len(winning_exits) > 10:
            # Calculate what confidence levels actually won with
            winner_confidences = []
            for e in winning_exits:
                # Try to extract entry confidence (may not exist in old data)
                entry_conf = e.get('entry_confidence', 0.5)
                winner_confidences.append(entry_conf)
            
            # If high-confidence trades won more, be more aggressive with scaling
            # If low-confidence trades also won, be more conservative
            avg_winner_conf = sum(winner_confidences) / len(winner_confidences)
            
            # Learn scaling range based on winner confidence distribution
            if avg_winner_conf > 0.7:
                # Winners were high confidence - be more aggressive
                min_scale = 0.7
                max_scale = 1.3  # 70-130% range
            elif avg_winner_conf > 0.5:
                # Winners were medium confidence - balanced
                min_scale = 0.8
                max_scale = 1.2  # 80-120% range
            else:
                # Winners were low confidence - be conservative
                min_scale = 0.85
                max_scale = 1.15  # 85-115% range
        else:
            # Defaults
            min_scale = 0.8
            max_scale = 1.2
        
        # Apply LEARNED scaling range
        scale_range = max_scale - min_scale
        scale_factor = min_scale + (confidence_adjustment * scale_range)
        
        breakeven_threshold_ticks = int(avg_winner_be * scale_factor)
        trailing_distance_ticks = int(avg_winner_trail * scale_factor)
        stop_mult = avg_winner_stop * (0.9 + confidence_adjustment * 0.2)  # Keep stop conservative
        
        # Partial targets - learn from actual R-multiples achieved
        if len(winning_exits) > 5:  # Need enough data
            # Calculate what R-multiples winners actually hit
            r_multiples = []
            for e in winning_exits:
                outcome = e.get('outcome', {})
                if isinstance(outcome, dict) and 'r_multiple' in outcome:
                    r_value = outcome['r_multiple']
                    # Only include reasonable R values (filter obvious errors)
                    if 0.5 <= r_value <= 10.0:
                        r_multiples.append(r_value)
            
            if len(r_multiples) > 3:
                # Use percentiles of actual R achieved
                r_multiples.sort()
                partial_1_r = r_multiples[int(len(r_multiples) * 0.33)]  # 33rd percentile
                partial_2_r = r_multiples[int(len(r_multiples) * 0.66)]  # 66th percentile  
                partial_3_r = r_multiples[int(len(r_multiples) * 0.90)]  # 90th percentile
            else:
                # Fallback
                partial_1_r = 1.5
                partial_2_r = 2.5
                partial_3_r = 4.0
        else:
            # Defaults if no winning data
            partial_1_r = 1.5
            partial_2_r = 2.5
            partial_3_r = 4.0
        
        # NO HARD LIMITS - trust the data!
        # Median naturally filters outliers, confidence scaling prevents extremes
        
        # LEARN PARTIAL PERCENTAGES from winning exits
        # Calculate how winners actually split their position
        if len(winning_exits) > 5:
            partial_pcts = []
            for e in winning_exits:
                params = e.get('exit_params', {})
                if 'partial_1_pct' in params:
                    partial_pcts.append((
                        params.get('partial_1_pct', 0.5),
                        params.get('partial_2_pct', 0.3),
                        params.get('partial_3_pct', 0.2)
                    ))
            
            if len(partial_pcts) > 3:
                # Use median percentages
                p1_values = sorted([p[0] for p in partial_pcts])
                p2_values = sorted([p[1] for p in partial_pcts])
                p3_values = sorted([p[2] for p in partial_pcts])
                
                partial_1_pct = p1_values[len(p1_values) // 2]
                partial_2_pct = p2_values[len(p2_values) // 2]
                partial_3_pct = p3_values[len(p3_values) // 2]
            else:
                # Fallback
                partial_1_pct = 0.5
                partial_2_pct = 0.3
                partial_3_pct = 0.2
        else:
            # Defaults
            partial_1_pct = 0.5
            partial_2_pct = 0.3
            partial_3_pct = 0.2
        
        return {
            'breakeven_threshold_ticks': breakeven_threshold_ticks,
            'trailing_distance_ticks': trailing_distance_ticks,
            'stop_mult': stop_mult,
            'partial_1_r': partial_1_r,
            'partial_1_pct': partial_1_pct,  # LEARNED from data
            'partial_2_r': partial_2_r,
            'partial_2_pct': partial_2_pct,  # LEARNED from data
            'partial_3_r': partial_3_r,
            'partial_3_pct': partial_3_pct,  # LEARNED from data
        }
    
    def _get_default_params(self) -> Dict:
        """Return default exit parameters when no experiences available"""
        return {
            'breakeven_threshold_ticks': 12,
            'trailing_distance_ticks': 15,
            'stop_mult': 4.0,
            'partial_1_r': 1.5,
            'partial_1_pct': 0.5,
            'partial_2_r': 2.5,
            'partial_2_pct': 0.3,
            'partial_3_r': 4.0,
            'partial_3_pct': 0.2,
        }
    
    def record_exit_outcome(self, regime: str, exit_params: Dict, trade_outcome: Dict, 
                           market_state: Dict = None, backtest_mode: bool = False, 
                           partial_exits: List = None):
        """
        Record exit outcome for local learning (adds to new experiences list).
        Compatible with cloud AdaptiveExitManager interface.
        Saves ~30 fields per exit for comprehensive learning.
        """
        if market_state is None:
            market_state = {}
        if partial_exits is None:
            partial_exits = []
        
        # Convert exit_params to ensure JSON serializable (handle numpy types)
        safe_exit_params = {}
        if exit_params:
            for k, v in exit_params.items():
                # Convert to native Python types
                if hasattr(v, 'item'):  # numpy types
                    v = v.item()
                if isinstance(v, bool):
                    safe_exit_params[k] = bool(v)
                elif isinstance(v, float):
                    safe_exit_params[k] = float(v)
                elif isinstance(v, int):
                    safe_exit_params[k] = int(v)
                else:
                    safe_exit_params[k] = v
        
        # Convert outcome to ensure JSON serializable (handle numpy types)
        safe_outcome = {}
        if trade_outcome:
            for k, v in trade_outcome.items():
                # Convert numpy types to native Python
                if hasattr(v, 'item'):  # numpy types
                    v = v.item()
                if isinstance(v, bool):
                    safe_outcome[k] = bool(v)
                elif isinstance(v, float):
                    safe_outcome[k] = float(v)
                elif isinstance(v, int):
                    safe_outcome[k] = int(v)
                else:
                    safe_outcome[k] = str(v) if v is not None else ''
        
        # Convert partial_exits to ensure JSON serializable
        safe_partial_exits = []
        for partial in partial_exits:
            safe_partial = {}
            for k, v in partial.items():
                if hasattr(v, 'item'):  # numpy types
                    v = v.item()
                if isinstance(v, bool):
                    safe_partial[k] = bool(v)
                elif isinstance(v, float):
                    safe_partial[k] = float(v)
                elif isinstance(v, int):
                    safe_partial[k] = int(v)
                else:
                    safe_partial[k] = str(v) if v is not None else ''
            safe_partial_exits.append(safe_partial)
        
        # Convert advanced tracking arrays (NEW)
        safe_exit_param_updates = []
        exit_param_updates = trade_outcome.get('exit_param_updates', [])
        if exit_param_updates:
            for update in exit_param_updates:
                safe_update = {}
                for k, v in update.items():
                    if hasattr(v, 'item'):
                        v = v.item()
                    if isinstance(v, (int, float)):
                        safe_update[k] = float(v)
                    else:
                        safe_update[k] = str(v) if v is not None else ''
                safe_exit_param_updates.append(safe_update)
        
        safe_stop_adjustments = []
        stop_adjustments = trade_outcome.get('stop_adjustments', [])
        if stop_adjustments:
            for adjustment in stop_adjustments:
                safe_adjustment = {}
                for k, v in adjustment.items():
                    if hasattr(v, 'item'):
                        v = v.item()
                    if isinstance(v, (int, float)):
                        safe_adjustment[k] = float(v)
                    else:
                        safe_adjustment[k] = str(v) if v is not None else ''
                safe_stop_adjustments.append(safe_adjustment)
        
        experience = {
            'regime': str(regime) if regime else 'NORMAL',
            'exit_params': safe_exit_params,
            'outcome': safe_outcome,
            'market_state': {
                'rsi': float(market_state.get('rsi', 50.0)),
                'volume_ratio': float(market_state.get('volume_ratio', 1.0)),
                'hour': int(market_state.get('hour', 12)),
                'day_of_week': int(market_state.get('day_of_week', 0)),
                'streak': int(market_state.get('streak', 0)),
                'recent_pnl': float(market_state.get('recent_pnl', 0.0)),
                'vix': float(market_state.get('vix', 15.0)),
                'vwap_distance': float(market_state.get('vwap_distance', 0.0)),
                'atr': float(market_state.get('atr', 2.0))
            },
            'partial_exits': safe_partial_exits,  # Track partial exit history
            'timestamp': datetime.now(pytz.UTC).isoformat(),
            'side': str(trade_outcome.get('side', 'LONG')),
            'pnl': float(trade_outcome.get('pnl', 0)),
            'duration': int(trade_outcome.get('duration', 0)),
            'exit_reason': str(trade_outcome.get('exit_reason', 'unknown')),
            'win': bool(trade_outcome.get('win', False)),
            'entry_confidence': float(trade_outcome.get('entry_confidence', 0.5)),
            'r_multiple': float(trade_outcome.get('r_multiple', 0.0)),
            'mae': float(trade_outcome.get('mae', 0.0)),
            'mfe': float(trade_outcome.get('mfe', 0.0)),
            # NEW: In-trade behavior tracking
            'max_r_achieved': float(trade_outcome.get('max_r_achieved', 0.0)),
            'min_r_achieved': float(trade_outcome.get('min_r_achieved', 0.0)),
            'exit_param_update_count': int(trade_outcome.get('exit_param_update_count', 0)),
            'stop_adjustment_count': int(trade_outcome.get('stop_adjustment_count', 0)),
            'breakeven_activation_bar': int(trade_outcome.get('breakeven_activation_bar', 0)),
            'trailing_activation_bar': int(trade_outcome.get('trailing_activation_bar', 0)),
            'bars_until_breakeven': int(trade_outcome.get('bars_until_breakeven', 0)),
            'bars_until_trailing': int(trade_outcome.get('bars_until_trailing', 0)),
            'breakeven_activated': bool(trade_outcome.get('breakeven_activated', False)),
            'trailing_activated': bool(trade_outcome.get('trailing_activated', False)),
            # Advanced tracking arrays
            'exit_param_updates': safe_exit_param_updates,
            'stop_adjustments': safe_stop_adjustments,
            # NEW: Execution quality tracking
            'slippage_ticks': float(trade_outcome.get('slippage_ticks', 0.0)),
            'commission_cost': float(trade_outcome.get('commission_cost', 0.0)),
            'bid_ask_spread_ticks': float(trade_outcome.get('bid_ask_spread_ticks', 0.5)),
            # NEW: Market context tracking
            'session': str(trade_outcome.get('session', 'NY')),
            'volume_at_exit': float(trade_outcome.get('volume_at_exit', 0.0)),
            'volatility_regime_change': bool(trade_outcome.get('volatility_regime_change', False)),
            # NEW: Exit quality tracking
            'time_in_breakeven_bars': int(trade_outcome.get('time_in_breakeven_bars', 0)),
            'rejected_partial_count': int(trade_outcome.get('rejected_partial_count', 0)),
            'stop_hit': bool(trade_outcome.get('stop_hit', False)),
        }
        
        self.new_exit_experiences.append(experience)
    
    def save_new_experiences_to_file(self):
        """Save new exit experiences accumulated during backtest to local JSON file (v2 format)"""
        if len(self.new_exit_experiences) == 0:
            print("No new exit experiences to save")
            return
        
        # Use V2 file with full structure
        exit_file = os.path.join(self.local_dir, "exit_experiences_v2.json")
        
        # Load existing
        with open(exit_file, 'r') as f:
            data = json.load(f)
            existing_experiences = data.get('experiences', [])
        
        # Add new ones
        all_experiences = existing_experiences + self.new_exit_experiences
        
        # Save back with v2 metadata
        with open(exit_file, 'w') as f:
            json.dump({
                'experiences': all_experiences,
                'count': len(all_experiences),
                'version': '2.0',
                'last_updated': datetime.now(pytz.UTC).isoformat()
            }, f, indent=2)
        
        print(f"✅ Saved {len(self.new_exit_experiences)} new exit experiences to local file")
        print(f"   Total exit experiences now: {len(all_experiences):,}")
        
        # Update in-memory list
        self.exit_experiences = all_experiences
        self.new_exit_experiences = []

    
    def print_learned_summary(self):
        """Print summary of all learned insights from this backtest run"""
        if len(self.learned_insights) > 0:
            print(f"\n{'='*80}")
            print(f"📚 LEARNED INSIGHTS SUMMARY ({len(self.learned_insights)} adjustments)")
            print(f"{'='*80}")
            for insight in self.learned_insights:
                print(f"  {insight}")
            print(f"{'='*80}\n")

# Global instance for import
local_exit_manager = LocalExitManager()
