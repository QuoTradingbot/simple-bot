# pyright: reportGeneralTypeIssues=false
"""
Local Exit Manager for Fast Backtesting
Provides same adaptive exit logic as cloud API but using local experiences
NOW WITH NEURAL NETWORK PREDICTIONS!
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, List

JSONDict = Dict[str, Any]
import pytz
import numpy as np

# Try to import torch - gracefully handle if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - exit neural network disabled")

try:
    from neural_exit_model import ExitParamsNet, denormalize_exit_params
    NEURAL_MODEL_AVAILABLE = True
except ImportError:
    NEURAL_MODEL_AVAILABLE = False
    
from exit_param_utils import extract_all_exit_params

class LocalExitManager:
    """Manages adaptive exits using neural network predictions"""
    
    def __init__(self, verbose: bool = False):
        self.exit_experiences: List[JSONDict] = []
        # Use absolute path relative to this file's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.local_dir = os.path.join(script_dir, "..", "data", "local_experiences")
        self.loaded = False
        self.new_exit_experiences: List[JSONDict] = []  # NEW: Track new exit experiences
        self.verbose = verbose  # Control learning output
        self.learned_insights: List[str] = []  # Collect insights
        
        # Load neural network model (if torch available)
        self.exit_model = None
        self.device = None
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._load_neural_model()
        else:
            print("⚠️  PyTorch not available - using rule-based exits only")
    
    def get_adaptive_exit_params(self, market_state: Optional[JSONDict] = None,
                                 position: Optional[JSONDict] = None,
                                 entry_confidence: float = 0.5) -> JSONDict:
        """Public API matching cloud adaptive exit manager."""
        if market_state is None:
            market_state = {}
        if position is None:
            position = {}

        try:
            conf = float(entry_confidence)
        except (TypeError, ValueError):
            conf = 0.5
        entry_confidence = float(np.clip(conf, 0.0, 1.0))

        if not self.loaded:
            self.load_experiences()

        params = self._predict_with_neural_network(market_state, position, entry_confidence)
        params.setdefault('source', 'default')
        params.setdefault('market_regime', market_state.get('regime', 'UNKNOWN'))
        return params

    def _load_neural_model(self):
        """Load the trained exit neural network"""
        if not TORCH_AVAILABLE or not NEURAL_MODEL_AVAILABLE:
            return False
            
        try:
            # Use absolute path from this file's location
            model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'exit_model.pth')
            model_path = os.path.abspath(model_path)
            if not os.path.exists(model_path):
                print(f"⚠️  Exit neural network not found at {model_path}")
                print(f"   Falling back to rule-based exits")
                return False
            
            checkpoint = torch.load(model_path, map_location=self.device)
            input_size = checkpoint.get('input_size', 45)
            hidden_size = checkpoint.get('hidden_size', 64)
            self.exit_model = ExitParamsNet(
                input_size=input_size,
                hidden_size=hidden_size
            ).to(self.device)
            self.exit_model.load_state_dict(checkpoint['model_state_dict'])
            self.exit_model.eval()
            
            print(f"🧠 Exit neural network loaded (Val Loss: {checkpoint.get('val_loss', 0):.4f})")
            return True
            
        except Exception as e:
            print(f"❌ Error loading exit neural network: {e}")
            print(f"   Falling back to rule-based exits")
            return False
    
    def _predict_with_neural_network(self, market_state: JSONDict, position: JSONDict,
                                      entry_confidence: float) -> JSONDict:
        """Use 45-feature exit network; fall back to experiential logic if NN fails."""

        # Detect regime from market state (mirrors adaptive_exits.py)
        rsi = market_state.get('rsi', 50.0)
        volume_ratio = market_state.get('volume_ratio', 1.0)
        if rsi > 70 and volume_ratio > 1.5:
            regime = 'OVERBOUGHT'
        elif rsi < 30 and volume_ratio > 1.5:
            regime = 'OVERSOLD'
        elif volume_ratio > 2.0:
            regime = 'CHOPPY'
        else:
            regime = market_state.get('regime', 'NORMAL')

        regime_map = {
            'NORMAL': 0,
            'NORMAL_TRENDING': 1,
            'HIGH_VOL_TRENDING': 2,
            'HIGH_VOL_CHOPPY': 3,
            'LOW_VOL_TRENDING': 4,
            'LOW_VOL_RANGING': 5,
            'UNKNOWN': 0,
            'OVERBOUGHT': 1,
            'OVERSOLD': 1,
            'CHOPPY': 3,
        }
        market_regime_enc = regime_map.get(regime, 0) / 5.0

        # Market Context (8 features)
        avg_volume = market_state.get('avg_volume', max(market_state.get('volume', 1.0), 1.0))
        vol_ratio_norm = np.clip(volume_ratio / 3.0, 0, 1)
        atr_value = market_state.get('atr', 2.0)
        atr_norm = np.clip(atr_value / 10.0, 0, 1)
        vix_norm = np.clip(market_state.get('vix', 15.0) / 40.0, 0, 1)
        volatility_regime_change = 1.0 if market_state.get('volatility_regime_change') else 0.0
        volume_at_exit = vol_ratio_norm
        market_state_enc = market_state.get('market_state_enc', 0.5)

        # Trade Context (7 features)
        side = 1.0 if position.get('side', 'long').lower() == 'short' else 0.0
        session_hour = market_state.get('hour', 12)
        if 0 <= session_hour < 8:
            session = 0
        elif 8 <= session_hour < 14:
            session = 1
        else:
            session = 2
        session_norm = session / 2.0
        bid_ask_spread = np.clip(market_state.get('bid_ask_spread_ticks', 1.0) / 5.0, 0, 1)
        commission = np.clip(market_state.get('commission_cost', 2.0) / 10.0, 0, 1)
        slippage = np.clip(market_state.get('slippage_ticks', 1.0) / 5.0, 0, 1)
        regime_enc = market_regime_enc

        # Time Features (5)
        hour_norm = session_hour / 24.0
        day_norm = market_state.get('day_of_week', 2) / 6.0
        duration = np.clip(position.get('duration_bars', 1) / 500.0, 0, 1)
        time_in_breakeven = np.clip(position.get('time_in_breakeven_bars', 0) / 100.0, 0, 1)
        bars_until_breakeven = np.clip(position.get('bars_until_breakeven', 999) / 100.0, 0, 1)

        # Performance Metrics (5)
        mae = np.clip(position.get('mae', 0.0) / 1000.0, -1, 0)
        mfe = np.clip(position.get('mfe', 0.0) / 2000.0, 0, 1)
        max_r = np.clip(position.get('max_r_achieved', 0.0) / 10.0, 0, 1)
        min_r = np.clip(position.get('min_r_achieved', 0.0) / 5.0, -1, 1)
        r_multiple = np.clip(position.get('r_multiple', 0.0) / 10.0, -1, 1)

        # Exit Strategy State (7)
        breakeven_activated = 1.0 if position.get('breakeven_activated', False) else 0.0
        trailing_activated = 1.0 if position.get('trailing_activated', False) else 0.0
        stop_hit = 1.0 if position.get('stop_hit', False) else 0.0
        exit_param_updates = np.clip(position.get('exit_param_update_count', 0) / 50.0, 0, 1)
        stop_adjustments = np.clip(position.get('stop_adjustment_count', 0) / 20.0, 0, 1)
        rejected_partials = np.clip(position.get('rejected_partial_count', 0) / 10.0, 0, 1)
        bars_until_trailing = np.clip(position.get('bars_until_trailing', 999) / 100.0, 0, 1)

        # Results (5)
        current_pnl = position.get('unrealized_pnl', 0.0)
        pnl_norm = np.clip(current_pnl / 2000.0, -1, 1)
        outcome_current = 1.0 if current_pnl > 0 else 0.0
        win_current = 1.0 if current_pnl > 0 else 0.0
        exit_reason = 0.0
        max_profit = np.clip(position.get('mfe', 0.0) / 2000.0, 0, 1)

        # Advanced (8)
        entry_atr = position.get('entry_atr', atr_value)
        atr_change_pct = 0.0
        if entry_atr > 0:
            atr_change_pct = np.clip(((atr_value - entry_atr) / entry_atr) , -1, 1)
        avg_atr_trade = np.clip(position.get('avg_atr_during_trade', atr_value) / 10.0, 0, 1)
        peak_r = np.clip(position.get('peak_r_multiple', max_r * 10.0) / 10.0, 0, 1)
        profit_dd = np.clip(position.get('profit_drawdown_from_peak', 0.0) / 2000.0, 0, 1)
        high_vol_bars = np.clip(position.get('high_volatility_bars', 0) / 100.0, 0, 1)
        recent_wins = np.clip(market_state.get('wins_in_last_5_trades', 0) / 5.0, 0, 1)
        recent_losses = np.clip(market_state.get('losses_in_last_5_trades', 0) / 5.0, 0, 1)
        minutes_until_close = market_state.get('minutes_until_close', 180)
        mins_to_close = np.clip(minutes_until_close / 480.0, 0, 1)

        features = [
            # Market Context (8)
            market_regime_enc, rsi / 100.0, vol_ratio_norm, atr_norm, vix_norm,
            volatility_regime_change, volume_at_exit, market_state_enc,
            # Trade Context (7)
            entry_confidence, side, session_norm, bid_ask_spread, commission, slippage, regime_enc,
            # Time Features (5)
            hour_norm, day_norm, duration, time_in_breakeven, bars_until_breakeven,
            # Performance Metrics (5)
            mae, mfe, max_r, min_r, r_multiple,
            # Exit Strategy State (7)
            breakeven_activated, trailing_activated, stop_hit, exit_param_updates,
            stop_adjustments, rejected_partials, bars_until_trailing,
            # Results (5)
            pnl_norm, outcome_current, win_current, exit_reason, max_profit,
            # Advanced (8)
            atr_change_pct, avg_atr_trade, peak_r, profit_dd, high_vol_bars,
            recent_wins, recent_losses, mins_to_close,
        ]

        if self.exit_model is not None:
            try:
                with torch.no_grad():
                    feature_tensor = torch.FloatTensor([features]).to(self.device)
                    normalized_output = self.exit_model(feature_tensor)
                exit_params = denormalize_exit_params(normalized_output[0].cpu())
                return {
                    'breakeven_threshold_ticks': exit_params['breakeven_threshold_ticks'],
                    'trailing_distance_ticks': exit_params['trailing_distance_ticks'],
                    'stop_mult': exit_params['stop_mult'],
                    'partial_1_r': exit_params['partial_1_r'],
                    'partial_2_r': exit_params['partial_2_r'],
                    'partial_3_r': exit_params['partial_3_r'],
                    'source': 'neural'
                }
            except Exception as err:
                if self.verbose:
                    print(f"⚠️  Neural exit prediction failed: {err}")

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
            exp_vix = float(exp.get('vix', 18.0))
            state_vix = float(market_state.get('vix', 18.0))
            vix_sim = 1.0 - min(abs(exp_vix - state_vix) / 35.0, 1.0)
            exp_atr = float(exp.get('atr', 2.0))
            state_atr = float(market_state.get('atr', 2.0))
            atr_sim = 1.0 - min(abs(exp_atr - state_atr) / 5.0, 1.0)
            hour_sim = 1.0 if exp.get('hour') == market_state.get('hour') else 0.5
            
            # Overall similarity (LEARNED WEIGHTS)
            similarity = vix_sim * vix_weight + atr_sim * atr_weight + hour_sim * hour_weight
            
            # LEARNED THRESHOLD - what similarity do winners typically have?
            if len(winners) > 10:
                # Calculate similarity distribution for winners
                win_sims = []
                for w in winners[:100]:  # Sample for performance
                    win_vix = float(w.get('vix', 18.0))
                    w_vix_sim = 1.0 - min(abs(win_vix - state_vix) / 35.0, 1.0)
                    win_atr = float(w.get('atr', 2.0))
                    w_atr_sim = 1.0 - min(abs(win_atr - state_atr) / 5.0, 1.0)
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
        
        # NEW: Separate losses into two categories
        # 1. Never profitable (bad entry) - ignore these for exit learning
        # 2. Was profitable, gave it back (bad exit) - LEARN FROM THESE
        gave_back_exits = []
        protected_exits = []
        
        for e in top_exits:
            exp = e['exp']
            outcome = exp.get('outcome', {})
            max_profit = exp.get('max_profit_reached', exp.get('mfe', 0))
            final_pnl = outcome.get('pnl', 0)
            
            # Categorize trades that went profitable
            if max_profit > 0:
                if final_pnl < 0:
                    # Was green, went red - PROFIT GIVEBACK
                    gave_back_exits.append(exp)
                else:
                    # Stayed green - PROTECTED PROFIT
                    protected_exits.append(exp)
        
        # Calculate MEDIAN parameters from PROTECTED exits (trades that stayed profitable)
        # IGNORE givebacks - their parameters were TOO LOOSE
        if len(protected_exits) > 5:  # Need at least 5 for statistical significance
            # Extract all values from PROTECTED exits only
            be_values = [e.get('exit_params', {}).get('breakeven_threshold_ticks', 12) for e in protected_exits]
            trail_values = [e.get('exit_params', {}).get('trailing_distance_ticks', 15) for e in protected_exits]
            stop_values = [e.get('exit_params', {}).get('stop_mult', 4.0) for e in protected_exits]
            
            # Use MEDIAN (not mean) - naturally filters outliers
            be_values.sort()
            trail_values.sort()
            stop_values.sort()
            
            avg_protected_be = be_values[len(be_values) // 2]  # Median
            avg_protected_trail = trail_values[len(trail_values) // 2]  # Median
            avg_protected_stop = stop_values[len(stop_values) // 2]  # Median
            
            # If we have giveback data, make parameters TIGHTER than givebacks used
            if len(gave_back_exits) > 10:
                # Calculate what givebacks used
                giveback_be = [e.get('exit_params', {}).get('breakeven_threshold_ticks', 12) for e in gave_back_exits]
                giveback_trail = [e.get('exit_params', {}).get('trailing_distance_ticks', 15) for e in gave_back_exits]
                giveback_be.sort()
                giveback_trail.sort()
                
                median_giveback_be = giveback_be[len(giveback_be) // 2]
                median_giveback_trail = giveback_trail[len(giveback_trail) // 2]
                
                # Learn to use TIGHTER than givebacks (85% of giveback value)
                learned_be = int(min(avg_protected_be, median_giveback_be * 0.85))
                learned_trail = int(min(avg_protected_trail, median_giveback_trail * 0.85))
                
                avg_winner_be = learned_be
                avg_winner_trail = learned_trail
                avg_winner_stop = avg_protected_stop
            else:
                # Just use protected medians
                avg_winner_be = avg_protected_be
                avg_winner_trail = avg_protected_trail
                avg_winner_stop = avg_protected_stop
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
            'source': 'experiences',
        }
    
    def load_experiences(self):
        """Load local exit experiences once for offline learning."""
        if self.loaded and self.exit_experiences:
            return

        exit_path = os.path.join(self.local_dir, "exit_experiences_v2.json")
        if not os.path.exists(exit_path):
            print(f"⚠️  Exit experiences file not found at {exit_path}")
            print("   Creating empty exit experiences file for this backtest...")
            # Create empty file so save_new_experiences_to_file() can append to it
            with open(exit_path, 'w') as f:
                json.dump({
                    'experiences': [],
                    'count': 0,
                    'version': '2.0',
                    'last_updated': datetime.now(pytz.UTC).isoformat()
                }, f, indent=2)
            print("   ✅ Empty exit experiences file created - will collect data during backtest")
            self.exit_experiences = []
            self.loaded = True
            return

        with open(exit_path, 'r') as infile:
            data = json.load(infile)

        if isinstance(data, dict):
            self.exit_experiences = data.get('experiences', [])
        else:
            self.exit_experiences = data

        self.loaded = True
        print(f"✅ Loaded {len(self.exit_experiences):,} exit experiences from local files")

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
            'source': 'default',
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
                elif isinstance(v, (list, dict)):
                    safe_outcome[k] = v  # Keep lists/dicts as-is for JSON serialization
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
        
        # Helper to handle NaN values
        import math
        def safe_float(value, default):
            try:
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    return default
                return float(value)
            except (ValueError, TypeError):
                return default
        
        experience: JSONDict = {
            'regime': str(regime) if regime else 'NORMAL',
            'exit_params': safe_exit_params,  # Original params for compatibility
            'exit_params_used': safe_exit_params,  # ALL 131 params the bot actually used
            'outcome': safe_outcome,
            'market_state': {
                'rsi': safe_float(market_state.get('rsi'), 50.0),
                'volume_ratio': safe_float(market_state.get('volume_ratio'), 1.0),
                'hour': int(market_state.get('hour', 12)),
                'day_of_week': int(market_state.get('day_of_week', 0)),
                'streak': int(market_state.get('streak', 0)),
                'recent_pnl': safe_float(market_state.get('recent_pnl'), 0.0),
                'vix': safe_float(market_state.get('vix'), 15.0),
                'vwap_distance': safe_float(market_state.get('vwap_distance'), 0.0),
                'atr': safe_float(market_state.get('atr'), 2.0),
                'peak_pnl': safe_float(market_state.get('peak_pnl'), 0.0)  # CRITICAL: Peak unrealized PNL for drawdown analysis
            },
            'partial_exits': safe_partial_exits,  # Track partial exit history
            'timestamp': datetime.now(pytz.UTC).isoformat(),
            'side': str(trade_outcome.get('side', 'LONG')),
            'pnl': safe_float(trade_outcome.get('pnl'), 0.0),
            'duration': int(trade_outcome.get('duration', 0)),
            'exit_reason': str(trade_outcome.get('exit_reason', 'unknown')),
            'win': bool(trade_outcome.get('win', False)),
            'entry_confidence': safe_float(trade_outcome.get('entry_confidence'), 0.5),
            'r_multiple': safe_float(trade_outcome.get('r_multiple'), 0.0),
            'mae': safe_float(trade_outcome.get('mae'), 0.0),
            'mfe': safe_float(trade_outcome.get('mfe'), 0.0),
            'max_profit_reached': safe_float(trade_outcome.get('mfe'), 0.0),  # Track max profit (same as MFE)
            # NEW: In-trade behavior tracking
            'max_r_achieved': safe_float(trade_outcome.get('max_r_achieved'), 0.0),
            'min_r_achieved': safe_float(trade_outcome.get('min_r_achieved'), 0.0),
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
            'slippage_ticks': safe_float(trade_outcome.get('slippage_ticks'), 0.0),
            'commission_cost': safe_float(trade_outcome.get('commission_cost'), 0.0),
            # NEW: Market context tracking (save as INTEGER, not string!)
            'session': int(trade_outcome.get('session', 2)),  # NY=2
            'volume_at_exit': safe_float(trade_outcome.get('volume_at_exit'), 0.0),
            'volatility_regime_change': bool(trade_outcome.get('volatility_regime_change', False)),
            # NEW: Exit quality tracking
            'time_in_breakeven_bars': int(trade_outcome.get('time_in_breakeven_bars', 0)),
            'rejected_partial_count': int(trade_outcome.get('rejected_partial_count', 0)),
            'stop_hit': bool(trade_outcome.get('stop_hit', False)),
        }
        
        self.new_exit_experiences.append(experience)
        
        # DEBUG: Confirm experience was added
        if len(self.new_exit_experiences) % 10 == 0:
            print(f"   📊 Exit experiences collected: {len(self.new_exit_experiences)}")
    
    def save_new_experiences_to_file(self):
        """Save new exit experiences accumulated during backtest to local JSON file (v2 format)"""
        if len(self.new_exit_experiences) == 0:
            print("No new exit experiences to save")
            return
        
        # Use V2 file with full structure
        exit_file = os.path.join(self.local_dir, "exit_experiences_v2.json")
        
        # Load existing (if file exists)
        existing_experiences = []
        if os.path.exists(exit_file):
            try:
                with open(exit_file, 'r') as f:
                    data = json.load(f)
                    existing_experiences = data.get('experiences', [])
            except Exception as e:
                print(f"   Warning: Could not load existing exit experiences: {e}")
                existing_experiences = []
        
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
