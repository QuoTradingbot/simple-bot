"""
Neural Network Confidence Scorer for Cloud API
Loads your trained neural_model.pth and provides same predictions as backtest
"""
import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SignalConfidenceNet(nn.Module):
    """
    Neural network that predicts trade success probability.
    EXACT SAME architecture as dev-tools/neural_confidence_model.py
    """
    
    def __init__(self, input_size=31):
        super(SignalConfidenceNet, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1: 31 → 64
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2: 64 → 32
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Output: 32 → 1
            nn.Linear(32, 1),
            nn.Sigmoid()  # 0-1 confidence
        )
        
    def forward(self, x):
        return self.network(x)


class NeuralConfidenceScorer:
    """
    Cloud-ready neural network scorer
    Loads your trained model and provides predictions
    """
    
    def __init__(self, model_path: str = "neural_model.pth"):
        self.model = None
        self.model_path = model_path
        self.device = torch.device('cpu')  # Cloud runs on CPU
        
        # Load model if exists
        if os.path.exists(model_path):
            self.load_model()
        else:
            logger.warning(f"⚠️  Neural model not found: {model_path}")
            logger.warning(f"   Will use pattern matching fallback")
    
    def load_model(self):
        """Load trained neural network"""
        try:
            self.model = SignalConfidenceNet(input_size=31)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            # Handle both checkpoint format and direct state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()  # Set to evaluation mode
            logger.info(f"✅ Neural model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load neural model: {e}")
            self.model = None
    
    def prepare_features(self, state: Dict) -> np.ndarray:
        """
        Convert market state to neural network input features
        EXACT SAME 33 features as train_model.py (updated architecture)
        """
        from datetime import datetime
        
        # Session, signal, trade type, regime mappings
        session_map = {'Asia': 0, 'London': 1, 'NY': 2}
        signal_map = {'LONG': 0, 'SHORT': 1}
        trade_type_map = {'reversal': 0, 'continuation': 1}
        regime_map = {
            'NORMAL': 0,
            'NORMAL_TRENDING': 1,
            'HIGH_VOL_TRENDING': 2,
            'HIGH_VOL_CHOPPY': 3,
            'LOW_VOL_TRENDING': 4,
            'LOW_VOL_RANGING': 5,
            'UNKNOWN': 0
        }
        
        # Extract timestamp-based features
        timestamp_str = state.get('timestamp', '')
        minute = 0
        time_to_close = 240  # default 4 hours
        if timestamp_str:
            try:
                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                minute = dt.minute
                hour_decimal = dt.hour + dt.minute / 60.0
                # Time to session close (16:00 UTC = market close)
                time_to_close = max(0, 16.0 - hour_decimal) * 60  # minutes to close
            except:
                pass
        
        # Price level features
        price = state.get('price', state.get('entry_price', 6500.0))
        price_mod_50 = (price % 50) / 50.0  # Distance to nearest 50-point level (0-1)
        
        # Build 33-feature vector (EXACT order from train_model.py)
        features = [
            state.get('rsi', 50.0),                          # 0
            state.get('vix', 15.0),                          # 1
            state.get('hour', 12),                           # 2
            state.get('atr', 2.0),                           # 3
            state.get('volume_ratio', 1.0),                  # 4
            state.get('vwap_distance', 0.0),                 # 5
            state.get('streak', 0),                          # 6
            state.get('consecutive_wins', 0),                # 7
            state.get('consecutive_losses', 0),              # 8
            state.get('cumulative_pnl_at_entry', 0.0),       # 9
            session_map.get(state.get('session', 'NY'), 2),  # 10
            state.get('trend_strength', 0.0),                # 11
            state.get('sr_proximity_ticks', 0.0),            # 12
            trade_type_map.get(state.get('trade_type', 'reversal'), 0),  # 13
            state.get('time_since_last_trade_mins', 0.0),    # 14
            state.get('bid_ask_spread_ticks', 0.5),          # 15
            state.get('drawdown_pct_at_entry', 0.0),         # 16
            state.get('day_of_week', 0),                     # 17
            state.get('recent_pnl', 0.0),                    # 18
            state.get('entry_slippage_ticks', 0.0),          # 19
            state.get('commission_cost', 0.0),               # 20
            signal_map.get(state.get('signal', 'LONG'), 0),  # 21
            # ADVANCED ML FEATURES (4 features)
            regime_map.get(state.get('market_regime', 'NORMAL'), 0),  # 22
            state.get('recent_volatility_20bar', 2.0),       # 23
            state.get('volatility_trend', 0.0),              # 24
            state.get('vwap_std_dev', 2.0),                  # 25
            # NEW TEMPORAL/PRICE FEATURES (3 features - total 31)
            minute / 60.0,                                   # 26: Minute of hour (0-1)
            time_to_close / 240.0,                           # 27: Time to close (0-1)
            price_mod_50,                                    # 28: Distance to round 50 (0-1)
        ]
        
        return np.array(features, dtype=np.float32)
    
    def predict(self, state: Dict, signal: str) -> Dict:
        """
        Get neural network prediction
        
        Args:
            state: Market state dict with RSI, VWAP, VIX, etc.
            signal: 'LONG' or 'SHORT'
        
        Returns:
            Dict with confidence, should_trade, reason
        """
        if self.model is None:
            # Fallback to pattern matching if no model
            return {
                'confidence': 0.5,
                'should_trade': False,
                'size_multiplier': 1.0,
                'reason': 'Neural model not loaded, using fallback',
                'model_used': 'fallback'
            }
        
        try:
            # Set signal direction (one-hot encoding)
            state['signal_long'] = 1 if signal == 'LONG' else 0
            state['signal_short'] = 1 if signal == 'SHORT' else 0
            
            # Prepare features
            features = self.prepare_features(state)
            
            # Convert to tensor
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                confidence = self.model(x).item()
            
            # Apply temperature scaling (same as backtest)
            temperature = 1.0
            confidence = confidence ** (1.0 / temperature)
            
            # Cap at 95% (never 100% confident)
            confidence = min(confidence, 0.95)
            
            # Decision threshold (same as backtest)
            threshold = 0.5  # User configurable
            should_trade = confidence >= threshold
            
            # Size multiplier based on confidence (same as backtest)
            if confidence >= 0.85:
                size_mult = 1.5  # Very confident
            elif confidence >= 0.70:
                size_mult = 1.25  # Confident
            elif confidence >= 0.55:
                size_mult = 1.0  # Normal
            else:
                size_mult = 0.75  # Less confident
            
            reason = f"Neural network: {confidence:.1%} confidence (threshold: {threshold:.0%})"
            
            return {
                'confidence': confidence,
                'should_trade': should_trade,
                'size_multiplier': size_mult,
                'reason': reason,
                'model_used': 'neural_network',
                'threshold': threshold
            }
            
        except Exception as e:
            logger.error(f"❌ Neural prediction error: {e}")
            return {
                'confidence': 0.5,
                'should_trade': False,
                'size_multiplier': 1.0,
                'reason': f'Error: {str(e)}',
                'model_used': 'error'
            }


# Global neural scorer instance (loaded once at startup)
neural_scorer: Optional[NeuralConfidenceScorer] = None


def init_neural_scorer(model_path: str = "neural_model.pth"):
    """Initialize neural scorer at API startup"""
    global neural_scorer
    neural_scorer = NeuralConfidenceScorer(model_path)
    return neural_scorer


def get_neural_prediction(state: Dict, signal: str) -> Dict:
    """
    Get neural network prediction (convenience function)
    
    Args:
        state: Market state with RSI, VWAP, VIX, etc.
        signal: 'LONG' or 'SHORT'
    
    Returns:
        Dict with confidence, should_trade, size_multiplier, reason
    """
    global neural_scorer
    
    if neural_scorer is None:
        init_neural_scorer()
    
    return neural_scorer.predict(state, signal)
