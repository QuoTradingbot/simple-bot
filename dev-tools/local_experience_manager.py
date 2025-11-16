"""
Local Experience Manager for Neural Network Backtesting
Loads experiences from local JSON and uses trained neural network for predictions
"""
import json
import os
from typing import Dict, List
from datetime import datetime
import pytz

class LocalExperienceManager:
    """Manages local experiences and neural network predictions for backtesting"""
    
    def __init__(self, use_neural_network=True, confidence_threshold=None):
        self.signal_experiences = []
        self.exit_experiences = []
        # Use absolute path relative to this file's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.local_dir = os.path.join(script_dir, "..", "data", "local_experiences")
        self.loaded = False
        self.new_signal_experiences = []  # Track experiences added during backtest
        self.new_exit_experiences = []    # Track exit experiences added during backtest
        
        # Confidence threshold (if None, will use learned adaptive threshold)
        self.confidence_threshold = confidence_threshold
        
        # Neural network support
        self.use_neural_network = use_neural_network
        self.neural_predictor = None
        if use_neural_network:
            try:
                from neural_confidence_model import ConfidencePredictor
                # Use absolute path from this file's location
                model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'neural_model.pth')
                model_path = os.path.abspath(model_path)
                self.neural_predictor = ConfidencePredictor(model_path=model_path)
                if self.neural_predictor.load_model():
                    print("🧠 Neural network loaded - using AI for confidence prediction")
                else:
                    print("⚠️  Neural network not found - train with: python train_model.py")
                    self.use_neural_network = False
            except Exception as e:
                print(f"⚠️  Could not load neural network: {e}")
                print("   Train with: python train_model.py")
                self.use_neural_network = False
        
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
            
            # Check if at least signal experiences exist
            if not os.path.exists(signal_file):
                print(f"\n⚠️  Local experiences not found in {self.local_dir}/")
                print(f"   Creating new v2 experience files...")
                return False
            
            # Load signal experiences (required)
            with open(signal_file, 'r') as f:
                data = json.load(f)
                self.signal_experiences = data.get('experiences', [])
            
            # Load exit experiences (optional - may not exist yet)
            if os.path.exists(exit_file):
                with open(exit_file, 'r') as f:
                    data = json.load(f)
                    self.exit_experiences = data.get('experiences', [])
            else:
                print(f"   Exit experiences not found - will be created during backtest")
                self.exit_experiences = []
            
            self.loaded = True
            print(f"✅ Loaded {len(self.signal_experiences)} signal experiences from local files")
            return True
        except Exception as e:
            print(f"⚠️  Error loading local experiences: {e}")
            return False
    
    def _get_learned_confidence_threshold(self) -> float:
        """
        ADAPTIVE threshold - learns optimal confidence from NEW model predictions.
        Re-predicts all historical trades with current model, then finds threshold that maximizes profit.
        Completely ignores old buggy 'confidence' field.
        """
        if len(self.signal_experiences) < 50:
            return 0.20  # Start low - let bot explore and learn (adaptive mode)
        
        # Collect all taken trades and RE-PREDICT confidence with current model
        trades_with_conf = []
        for exp in self.signal_experiences:
            if exp.get('took_trade', False):
                # Rebuild rl_state from experience
                rl_state = {k: exp.get(k, 0) for k in exp.keys() 
                           if k not in ['took_trade', 'outcome', 'timestamp', 'symbol', 'pnl', 'confidence']}
                rl_state['signal'] = 0 if exp.get('signal') == 'LONG' else 1
                
                # Re-predict with CURRENT model
                if self.neural_predictor is not None:
                    try:
                        confidence = self.neural_predictor.predict(rl_state)
                    except:
                        confidence = 0.5  # Fallback
                else:
                    confidence = 0.5
                
                pnl = exp.get('pnl', 0)
                trades_with_conf.append((confidence, pnl))
        
        if len(trades_with_conf) < 20:
            return 0.20  # Need minimum data - use exploratory threshold
        
        # Test thresholds from 10% to 80% in 5% increments (FULL ADAPTIVE RANGE)
        test_thresholds = [i/100 for i in range(10, 85, 5)]
        
        best_threshold = 0.20  # Safe exploratory default
        best_score = -999999
        
        for threshold in test_thresholds:
            # Get trades that would pass this threshold
            qualifying_trades = [pnl for conf, pnl in trades_with_conf if conf >= threshold]
            
            if len(qualifying_trades) < 10:  # Need minimum sample
                continue
            
            # Maximize profit while ensuring reasonable trade frequency
            total_pnl = sum(qualifying_trades)
            trade_frequency = len(qualifying_trades) / len(trades_with_conf)
            
            # Score = profit * sqrt(frequency) - rewards both profit AND trading
            # This prevents setting threshold too high and missing opportunities
            score = total_pnl * (trade_frequency ** 0.5)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        # Return learned threshold - let bot be fully adaptive
        # No artificial floor - trust the model's predictions
        return best_threshold
    
    def get_signal_confidence(self, rl_state: Dict, signal: str, exploration_rate: float = 0.0) -> tuple:
        """
        Get ML confidence using neural network ONLY.
        Neural network must be trained - no fallback to pattern matching.
        
        Args:
            rl_state: Market state features
            signal: LONG or SHORT
            exploration_rate: Probability (0.0-1.0) of taking trade regardless of confidence (for learning)
        
        Returns: (take_signal, confidence, reason)
        """
        # Neural network only - no fallback
        if self.use_neural_network and self.neural_predictor is not None:
            # Try to load experiences if not loaded (for saving new ones later)
            if not self.loaded:
                self.load_experiences()  # Doesn't matter if this fails - we have neural network
            
            try:
                return self._get_confidence_neural(rl_state, signal, exploration_rate)
            except Exception as e:
                print(f"❌ Neural network error: {e}")
                print(f"   Train the model with: python train_model.py")
                return (False, 0.0, "Neural network failed - train model first")
        
        # No neural network available - return random confidence for initial data collection
        # Use full range (0-100%) for true exploration, then apply threshold
        import random
        random_confidence = random.uniform(0.30, 0.95)  # Random between 30-95% for exploration
        
        # Apply the configured confidence threshold (from CONFIG or default 0.70)
        # This ensures some signals get rejected even without neural network
        threshold = 0.70  # Default threshold when no NN trained
        take_signal = random_confidence >= threshold
        
        reason = f"random_exploration (NN not trained): {random_confidence:.0%} vs threshold {threshold:.0%}"
        return (take_signal, random_confidence, reason)
    
    def _get_confidence_neural(self, rl_state: Dict, signal: str, exploration_rate: float = 0.0) -> tuple:
        """
        Get confidence prediction from trained neural network.
        Neural network uses 32 features to predict R-multiple (converted to confidence).
        
        Args:
            rl_state: Market state features
            signal: LONG or SHORT
            exploration_rate: Probability (0.0-1.0) of taking trade regardless of confidence (for learning)
        """
        # Signal is already encoded in rl_state (LONG=0, SHORT=1)
        # Don't overwrite it with string
        
        # Get prediction from neural network
        confidence = self.neural_predictor.predict(rl_state)
        
        # Use configured threshold if provided, otherwise use learned adaptive threshold
        if self.confidence_threshold is not None:
            threshold = self.confidence_threshold
            threshold_source = "configured"
        else:
            threshold = self._get_learned_confidence_threshold()
            threshold_source = "learned"
        
        # Apply exploration: randomly take exploration_rate% of signals regardless of confidence
        import random
        explore = random.random() < exploration_rate if exploration_rate > 0 else False
        
        if explore:
            take_signal = True
            reason = f"🎲 EXPLORATION: {confidence:.0%} confidence (exploration {exploration_rate:.0%})"
        else:
            take_signal = confidence >= threshold
            # Generate reason string
            wins = sum(1 for e in self.signal_experiences 
                      if e.get('took_trade') and e.get('pnl', 0) > 0)
            total = len([e for e in self.signal_experiences if e.get('took_trade')])
            reason = f"Neural: {confidence:.0%} confidence ({threshold_source} threshold: {threshold:.0%}, trained on {total} trades)"
        
        return (take_signal, confidence, reason)
    
    def add_signal_experience(self, rl_state: Dict, took_trade: bool, outcome: Dict):
        """Add new signal experience from backtest (will save to file at end)"""
        import math
        
        # Helper to handle NaN values
        def safe_float(value, default):
            try:
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    return default
                return float(value)
            except (ValueError, TypeError):
                return default
        
        # Extract timestamp for time-based features
        timestamp = datetime.now(pytz.UTC)
        minute = timestamp.minute
        hour = int(rl_state.get('hour', timestamp.hour))
        
        # Calculate time to session close (22:00 UTC = 17:00 EST = ES futures close)
        hour_decimal = hour + minute / 60.0
        time_to_close = max(0, 22.0 - hour_decimal) * 60  # minutes to close
        
        # Calculate price proximity to round 50-point levels
        price = safe_float(rl_state.get('price'), 0.0)
        price_mod_50 = (price % 50) / 50.0 if price > 0 else 0.0  # Distance to nearest 50-point level (0-1)
        
        experience = {
            # ORIGINAL FIELDS (with NaN handling)
            'rsi': safe_float(rl_state.get('rsi'), 50.0),
            'vwap_distance': safe_float(rl_state.get('vwap_distance'), 0.0),
            'vix': safe_float(rl_state.get('vix'), 15.0),
            'hour': hour,
            'day_of_week': int(rl_state.get('day_of_week', 0)),
            'volume_ratio': safe_float(rl_state.get('volume_ratio'), 1.0),
            'atr': safe_float(rl_state.get('atr'), 2.0),
            'recent_pnl': safe_float(rl_state.get('recent_pnl'), 0.0),
            'streak': int(rl_state.get('streak', 0)),
            'price': price,
            'vwap': safe_float(rl_state.get('vwap'), 0.0),
            'signal': str(rl_state.get('side', 'LONG')).upper(),
            'timestamp': timestamp.isoformat(),
            'pnl': safe_float(outcome.get('pnl'), 0.0),
            'outcome': 'WIN' if safe_float(outcome.get('pnl'), 0.0) > 0 else 'LOSS',
            'took_trade': bool(took_trade),
            'confidence': safe_float(outcome.get('confidence'), 0.50),
            # NEW PSYCHOLOGICAL FIELDS
            'cumulative_pnl_at_entry': safe_float(rl_state.get('cumulative_pnl_at_entry'), 0.0),
            'consecutive_wins': int(rl_state.get('consecutive_wins', 0)),
            'consecutive_losses': int(rl_state.get('consecutive_losses', 0)),
            'drawdown_pct_at_entry': safe_float(rl_state.get('drawdown_pct_at_entry'), 0.0),
            'time_since_last_trade_mins': safe_float(rl_state.get('time_since_last_trade_mins'), 0.0),
            # NEW MARKET CONTEXT FIELDS (save as integers, not strings!)
            'session': int(rl_state.get('session', 2)),  # NY=2
            'trend_strength': safe_float(rl_state.get('trend_strength'), 0.0),
            'sr_proximity_ticks': safe_float(rl_state.get('sr_proximity_ticks'), 0.0),
            'trade_type': int(rl_state.get('trade_type', 1)),  # continuation=1
            'entry_slippage_ticks': safe_float(rl_state.get('entry_slippage_ticks'), 0.0),
            'commission_cost': safe_float(rl_state.get('commission_cost'), 0.0),
            'bid_ask_spread_ticks': safe_float(rl_state.get('bid_ask_spread_ticks'), 0.5),
            # ADDITIONAL FIELDS (for future use - multi-symbol support and price-based learning)
            'symbol': str(rl_state.get('symbol', 'ES')),
            'entry_price': safe_float(rl_state.get('entry_price', rl_state.get('price', 0)), 0.0),
            'entry_bar': int(rl_state.get('entry_bar', 0)),  # NEW: Bar index at signal
            # ADVANCED ML FEATURES (market regime + volatility clustering)
            'market_regime': str(rl_state.get('market_regime', 'NORMAL')),
            'recent_volatility_20bar': safe_float(rl_state.get('recent_volatility_20bar'), 2.0),
            'volatility_trend': safe_float(rl_state.get('volatility_trend'), 0.0),
            'vwap_std_dev': safe_float(rl_state.get('vwap_std_dev'), 2.0),
            # PRE-CALCULATED TEMPORAL/PRICE FEATURES (for neural network)
            'minute': minute,  # Minute of hour (0-59)
            'time_to_close': time_to_close,  # Minutes until session close
            'price_mod_50': price_mod_50,  # Distance to round 50-level (0-1)
            # POSITION SIZING (critical for risk-adjusted learning)
            'contracts': int(outcome.get('contracts', rl_state.get('contracts', 1))),  # Number of contracts traded
        }
        self.new_signal_experiences.append(experience)
    
    def save_new_experiences_to_file(self):
        """Save new experiences accumulated during backtest to local JSON files (v2 format)"""
        if len(self.new_signal_experiences) == 0:
            print("No new signal experiences to save")
            return
        
        # Use V2 file with full structure
        signal_file = os.path.join(self.local_dir, "signal_experiences_v2.json")
        
        # Load existing (if file exists)
        existing_experiences = []
        if os.path.exists(signal_file):
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

# Global instance - threshold will be set from full_backtest CONFIG
local_manager = LocalExperienceManager(confidence_threshold=0.10)  # Default 10%, will be overridden by CONFIG
