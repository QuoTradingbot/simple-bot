"""
Neural Network for Signal Confidence Prediction
Replaces manual similarity calculation with learned patterns
"""
import torch
import torch.nn as nn
import numpy as np
import os

class SignalConfidenceNet(nn.Module):
    """
    Neural network that learns to predict trade reward quality (R-multiple).
    
    Architecture (RISK-ADJUSTED REWARD REGRESSION):
    - Input: 32 features (RSI, VIX, hour, ATR, volume, market_regime, volatility, 
             price levels, temporal, position sizing)
    - Hidden Layer 1: 64 neurons with ReLU activation
    - Hidden Layer 2: 32 neurons with ReLU activation  
    - Output: 1 neuron with LINEAR activation (R-multiple prediction -3 to +3)
    
    Dropout: 0.3 to prevent overfitting
    """
    
    def __init__(self, input_size=32):
        super(SignalConfidenceNet, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1: 32 → 64
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2: 64 → 32
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Output: 32 → 1 (LINEAR for R-multiple regression)
            nn.Linear(32, 1)
            # NO ACTIVATION - outputs raw R-multiple (-3 to +3)
        )
    
    def forward(self, x):
        return self.network(x)


class ConfidencePredictor:
    """
    Wrapper for the neural network with easy predict() interface.
    Handles model loading, feature normalization, and prediction.
    """
    
    def __init__(self, model_path='data/neural_model.pth'):
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Feature normalization parameters (will be set during training)
        self.feature_means = None
        self.feature_stds = None
        
        # Temperature scaling for calibration (learned from validation set)
        self.temperature = 1.5  # Will be optimized during training
        
    def load_model(self):
        """Load trained model from file"""
        if not os.path.exists(self.model_path):
            print(f"⚠️  Neural model not found at {self.model_path}")
            print(f"   Run train_model.py first to create the model")
            return False
            
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            self.model = SignalConfidenceNet()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Load normalization parameters
            self.feature_means = checkpoint.get('feature_means')
            self.feature_stds = checkpoint.get('feature_stds')
            
            # Load optimized temperature (if available)
            self.temperature = checkpoint.get('temperature', 1.5)
            
            print(f"✅ Neural network loaded from {self.model_path}")
            print(f"   Training MAE: {checkpoint.get('train_mae', 0):.3f}R")
            print(f"   Validation MAE: {checkpoint.get('val_mae', 0):.3f}R")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading neural model: {e}")
            return False
    
    def _normalize_features(self, features):
        """Normalize features using training statistics"""
        if self.feature_means is None or self.feature_stds is None:
            return features  # No normalization if stats not available
        
        import numpy as np
        # Use same logic as training: if std < 0.01, don't scale (use std=1.0)
        safe_stds = np.where(self.feature_stds < 0.01, 1.0, self.feature_stds)
        normalized = (features - self.feature_means) / safe_stds
        return normalized
    
    def predict(self, rl_state):
        """
        Predict confidence for a signal using R-multiple regression model.
        
        Args:
            rl_state: Dict with 32 features (RSI, VIX, hour, market_regime, volatility_clustering, 
                     position sizing, etc.)
            
        Returns:
            confidence: Float 0-1.0 (converted from predicted R-multiple)
                       R > 0 → confidence > 0.5 (profitable expected)
                       R < 0 → confidence < 0.5 (losing expected)
                       R = 0 → confidence = 0.5 (break-even)
        """
        if self.model is None:
            # Fallback to 50% if model not loaded
            return 0.50
        
        # Market regime encoding
        regime_map = {
            'NORMAL': 0,
            'NORMAL_TRENDING': 1,
            'HIGH_VOL_TRENDING': 2,
            'HIGH_VOL_CHOPPY': 3,
            'LOW_VOL_TRENDING': 4,
            'LOW_VOL_RANGING': 5,
            'UNKNOWN': 0  # Map unknown to NORMAL
        }
        market_regime_str = rl_state.get('market_regime', 'NORMAL')
        market_regime_encoded = regime_map.get(market_regime_str, 0)
        
        # Extract features in correct order (must match training!)
        # Calculate temporal features
        from datetime import datetime
        minute = rl_state.get('minute', 0)
        time_to_close = rl_state.get('time_to_close', 240.0)
        
        # Price level features
        price = rl_state.get('price', rl_state.get('entry_price', 6500.0))
        price_mod_50 = (price % 50) / 50.0
        
        features = np.array([
            rl_state.get('rsi', 50.0),
            rl_state.get('vix', 15.0),
            rl_state.get('hour', 12),
            rl_state.get('atr', 2.0),
            rl_state.get('volume_ratio', 1.0),
            rl_state.get('vwap_distance', 0.0),
            rl_state.get('streak', 0),
            rl_state.get('consecutive_wins', 0),
            rl_state.get('consecutive_losses', 0),
            rl_state.get('cumulative_pnl_at_entry', 0.0),
            rl_state.get('session', 0),  # Convert to numeric: Asia=0, London=1, NY=2
            rl_state.get('trend_strength', 0.0),
            rl_state.get('sr_proximity_ticks', 0.0),
            rl_state.get('trade_type', 0),  # reversal=0, continuation=1
            rl_state.get('time_since_last_trade_mins', 0.0),
            rl_state.get('bid_ask_spread_ticks', 0.5),
            rl_state.get('drawdown_pct_at_entry', 0.0),
            rl_state.get('day_of_week', 0),
            rl_state.get('recent_pnl', 0.0),
            rl_state.get('entry_slippage_ticks', 0.0),
            rl_state.get('commission_cost', 0.0),
            rl_state.get('signal', 0),  # LONG=0, SHORT=1
            # ADVANCED ML FEATURES (4 features)
            market_regime_encoded,  # Market regime (0-5)
            rl_state.get('recent_volatility_20bar', 2.0),  # Rolling 20-bar price std
            rl_state.get('volatility_trend', 0.0),  # Is volatility increasing
            rl_state.get('vwap_std_dev', 2.0),  # VWAP standard deviation
            # NEW TEMPORAL/PRICE FEATURES (3 features)
            minute / 60.0,  # Minute of hour (0-1)
            time_to_close / 240.0,  # Time to close normalized (0-1, 4hrs max)
            price_mod_50,  # Distance to round 50-level (0-1)
            # ADDITIONAL FEATURES (2 features - padding to match training)
            0.0,  # Reserved feature 1
            0.0,  # Reserved feature 2  
            # POSITION SIZING (1 feature - brings total to 32)
            rl_state.get('contracts', 1),  # Position size (1-3 contracts)
        ], dtype=np.float32)
        
        # Normalize features
        features = self._normalize_features(features)
        
        # Convert to tensor
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Predict R-multiple
        with torch.no_grad():
            r_multiple_raw = self.model(x).item()
        
        # FIX: Model outputs unbounded values, but training used tanh compression to -3/+3
        # Apply tanh to compress to expected range, then scale by typical magnitude (divide by 2)
        # This maps raw outputs to reasonable R-multiple range
        r_multiple = 3.0 * np.tanh(r_multiple_raw / 6.0)
        
        # Convert R-multiple to confidence (0-1 scale)
        # Use sigmoid-like transformation: confidence = 1 / (1 + exp(-r_multiple))
        # This maps: R=-3 → 5%, R=-1 → 27%, R=0 → 50%, R=+1 → 73%, R=+3 → 95%
        confidence = 1.0 / (1.0 + np.exp(-r_multiple))
        
        # Clip to valid range
        confidence = np.clip(confidence, 0.01, 0.99)
        
        # DETAILED LOGGING for debugging
        print(f"  🧠 Neural prediction: R-multiple={r_multiple:.3f} (raw={r_multiple_raw:.1f}) → confidence={confidence:.1%}")
        
        return confidence
    
    def predict_batch(self, rl_states):
        """
        Predict confidence for multiple signals at once (faster).
        
        Args:
            rl_states: List of rl_state dicts
            
        Returns:
            confidences: List of floats 0-1.0 (converted from predicted R-multiples)
        """
        if self.model is None:
            return [0.50] * len(rl_states)
        
        # Market regime encoding
        regime_map = {
            'NORMAL': 0,
            'NORMAL_TRENDING': 1,
            'HIGH_VOL_TRENDING': 2,
            'HIGH_VOL_CHOPPY': 3,
            'LOW_VOL_TRENDING': 4,
            'LOW_VOL_RANGING': 5,
            'UNKNOWN': 0  # Map unknown to NORMAL
        }
        
        # Extract features for all states
        features_list = []
        for rl_state in rl_states:
            market_regime_str = rl_state.get('market_regime', 'NORMAL')
            market_regime_encoded = regime_map.get(market_regime_str, 0)
            
            # Calculate temporal and price features
            minute = rl_state.get('minute', 0)
            time_to_close = rl_state.get('time_to_close', 240.0)
            price = rl_state.get('price', rl_state.get('entry_price', 6500.0))
            price_mod_50 = (price % 50) / 50.0
            
            features = np.array([
                rl_state.get('rsi', 50.0),
                rl_state.get('vix', 15.0),
                rl_state.get('hour', 12),
                rl_state.get('atr', 2.0),
                rl_state.get('volume_ratio', 1.0),
                rl_state.get('vwap_distance', 0.0),
                rl_state.get('streak', 0),
                rl_state.get('consecutive_wins', 0),
                rl_state.get('consecutive_losses', 0),
                rl_state.get('cumulative_pnl_at_entry', 0.0),
                rl_state.get('session', 0),
                rl_state.get('trend_strength', 0.0),
                rl_state.get('sr_proximity_ticks', 0.0),
                rl_state.get('trade_type', 0),
                rl_state.get('time_since_last_trade_mins', 0.0),
                rl_state.get('bid_ask_spread_ticks', 0.5),
                rl_state.get('drawdown_pct_at_entry', 0.0),
                rl_state.get('day_of_week', 0),
                rl_state.get('recent_pnl', 0.0),
                rl_state.get('entry_slippage_ticks', 0.0),
                rl_state.get('commission_cost', 0.0),
                rl_state.get('signal', 0),
                # ADVANCED ML FEATURES
                market_regime_encoded,
                rl_state.get('recent_volatility_20bar', 2.0),
                rl_state.get('volatility_trend', 0.0),
                rl_state.get('vwap_std_dev', 2.0),
                # TEMPORAL/PRICE FEATURES
                minute / 60.0,
                time_to_close / 240.0,
                price_mod_50,
                # ADDITIONAL FEATURES (5 features - total 32)
                rl_state.get('price_momentum', 0.0),
                rl_state.get('volume_momentum', 1.0),
                rl_state.get('spread_normalized', 0.0),
                rl_state.get('liquidity_score', 1.0),
                rl_state.get('contracts', 1),  # Position size
            ], dtype=np.float32)
            features_list.append(features)
        
        # Stack into batch
        features_batch = np.stack(features_list)
        features_batch = self._normalize_features(features_batch)
        
        # Convert to tensor
        x = torch.FloatTensor(features_batch).to(self.device)
        
        # Predict R-multiples
        with torch.no_grad():
            r_multiples = self.model(x).squeeze().cpu().numpy()
        
        # Convert R-multiples to confidences using sigmoid transformation
        confidences = 1.0 / (1.0 + np.exp(-r_multiples))
        confidences = np.clip(confidences, 0.01, 0.99)
        
        # Handle single prediction case
        if len(rl_states) == 1:
            return [float(confidences)]
        
        return confidences.tolist()


# Global predictor instance (will be loaded once at startup)
neural_predictor = ConfidencePredictor()
