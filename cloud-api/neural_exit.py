"""
Exit Neural Network Model for Cloud API
Predicts optimal exit parameters based on 62 market features
"""
import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict
import logging
from exit_params_config import EXIT_PARAMS, get_param_ranges, get_default_exit_params

logger = logging.getLogger(__name__)


class ExitParamsNet(nn.Module):
    """
    Neural network to predict optimal exit parameters
    
    Inputs (44 features - see _extract_feature_vector method for actual implementation):
        Market Context (8):
        - market_regime (encoded 0-4)
        - rsi (1)
        - volume_ratio (1) 
        - atr (1)
        - vix (1)
        - volatility_regime_change (1: boolean)
        - volume_at_exit (1)
        - market_state (1)
        
        Trade Context (5):
        - entry_confidence (1)
        - side (1: 0=LONG, 1=SHORT)
        - session (1: 0=Asia, 1=London, 2=NY)
        - commission_cost (1)
        - regime (1: encoded)
        
        Time Features (5):
        - hour_of_day (1)
        - day_of_week (1)
        - time_in_trade (1)
        - time_to_breakeven (1)
        - time_to_trailing (1)
        
        Performance Metrics (5):
        - mae_ticks (1: max adverse excursion)
        - mfe_ticks (1: max favorable excursion)
        - max_r_multiple (1)
        - min_r_multiple (1)
        - current_r_multiple (1)
        
        Strategy State (7):
        - breakeven_activated (1: boolean)
        - trailing_activated (1: boolean)
        - stop_hit (1: boolean)
        - num_breakeven_updates (1)
        - num_trailing_updates (1)
        - num_partials (1)
        - time_to_partial (1)
        
        Results (5):
        - pnl (1)
        - win_loss (1: boolean)
        - exit_reason (1: encoded)
        - max_profit_at_exit (1)
        - max_r_profit (1)
        
        Advanced (24):
        - atr_evolution (1)
        - avg_volatility (1)
        - peak_r_multiple (1)
        - profit_drawdown (1)
        - bars_high_volatility (1)
        - recent_wins (1)
        - recent_losses (1)
        - avg_time_to_close (1)
        - breakeven_threshold_ticks (1)
        - trailing_distance_ticks (1)
        - stop_mult (1)
        - partial_1_pct (1)
        - partial_2_pct (1)
        - partial_3_pct (1)
        - partial_1_r (1)
        - partial_2_r (1)
        - partial_3_r (1)
        - partial_on_profit (1)
        - partial_scale_atr (1)
        - partial_scale_volume (1)
        - max_partials (1)
        - partial_distance (1)
        - min_partial_size (1)
        - combine_partials (1)
        
        Daily Loss Limit (3):
        - daily_pnl_before_trade (1)
        - daily_loss_limit (1)
        - daily_loss_proximity_pct (1)
    
    Outputs (79 exit parameters):
        All backtest-learnable exit parameters
    """
    
    def __init__(self, input_size=44, hidden_size=128):
        super(ExitParamsNet, self).__init__()
        
        # Architecture: 44 inputs → 128 → 128 → 131 outputs
        # 44 inputs: market/trade context features (see _extract_feature_vector)
        # 131 outputs: comprehensive exit parameters (all learnable parameters)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 128)
        self.fc3 = nn.Linear(128, 131)  # 131 exit parameters
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))  # Normalized 0-1
        return x


def denormalize_exit_params(normalized_params):
    """
    Convert normalized [0-1] outputs back to real exit parameters
    
    Args:
        normalized_params: Tensor of shape [batch, 131] with values 0-1
    
    Returns:
        dict with actual exit parameter values (all 131 parameters)
    """
    # Get ranges from central config
    ranges = get_param_ranges()
    param_names = list(EXIT_PARAMS.keys())
    
    # Handle both single predictions and batches
    if len(normalized_params.shape) == 1:
        # Single prediction
        result = {}
        for i, name in enumerate(param_names):
            min_val, max_val = ranges[name]
            value = min_val + normalized_params[i].item() * (max_val - min_val)
            result[name] = value
        return result
    else:
        # Batch predictions
        results = []
        for batch_idx in range(normalized_params.shape[0]):
            result = {}
            for i, name in enumerate(param_names):
                min_val, max_val = ranges[name]
                value = min_val + normalized_params[batch_idx, i].item() * (max_val - min_val)
                result[name] = value
            results.append(result)
        return results


class NeuralExitPredictor:
    """
    Cloud-ready exit parameter predictor
    Loads trained exit model and provides real-time predictions
    """
    
    def __init__(self, model_path: str = "exit_model.pth"):
        """
        Initialize exit predictor with trained model
        
        Args:
            model_path: Path to exit_model.pth file
        """
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cpu")  # Cloud runs on CPU
        
        # Load model at initialization
        self.load_model()
    
    def load_model(self):
        """Load the trained exit neural network"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"❌ Exit model not found: {self.model_path}")
                return False
            
            # Create model architecture
            self.model = ExitParamsNet(input_size=44, hidden_size=64)
            
            # Load trained weights
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()  # Set to evaluation mode
            
            logger.info(f"✅ Exit neural network loaded from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load exit model: {e}")
            self.model = None
            return False
    
    def predict(self, features: Dict) -> Dict:
        """
        Predict optimal exit parameters for current trade state
        
        Args:
            features: Dict with 45 features (market context, trade state, etc.)
        
        Returns:
            Dict with predicted exit parameters
        """
        if self.model is None:
            logger.error("❌ Exit model not loaded, cannot predict")
            # Return safe defaults
            return {
                "breakeven_threshold_ticks": 8.0,
                "trailing_distance_ticks": 10.0,
                "stop_mult": 3.5,
                "partial_1_r": 2.0,
                "partial_2_r": 3.0,
                "partial_3_r": 5.0
            }
        
        try:
            # Extract 45 features in correct order
            feature_vector = self._extract_feature_vector(features)
            
            # Convert to tensor
            feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                normalized_output = self.model(feature_tensor)
            
            # Denormalize to actual parameter values
            exit_params = denormalize_exit_params(normalized_output.squeeze(0))
            
            logger.info(f"🧠 Exit NN prediction: BE={exit_params['breakeven_threshold_ticks']:.1f}t, "
                       f"Trail={exit_params['trailing_distance_ticks']:.1f}t, "
                       f"Partials={exit_params['partial_1_r']:.2f}R/{exit_params['partial_2_r']:.2f}R/{exit_params['partial_3_r']:.2f}R")
            
            return exit_params
            
        except Exception as e:
            logger.error(f"❌ Exit prediction error: {e}")
            # Return safe defaults
            return {
                "breakeven_threshold_ticks": 8.0,
                "trailing_distance_ticks": 10.0,
                "stop_mult": 3.5,
                "partial_1_r": 2.0,
                "partial_2_r": 3.0,
                "partial_3_r": 5.0
            }
    
    def _extract_feature_vector(self, features: Dict) -> list:
        """
        Extract 44 features in correct order for neural network
        
        Args:
            features: Dict with feature values
        
        Returns:
            List of 45 normalized feature values
        """
        # Market Context (8 features)
        regime_map = {'NORMAL': 0, 'NORMAL_TRENDING': 1, 'HIGH_VOL_TRENDING': 2, 
                      'HIGH_VOL_CHOPPY': 3, 'LOW_VOL_TRENDING': 4, 'LOW_VOL_RANGING': 5, 'UNKNOWN': 0}
        market_regime_enc = regime_map.get(features.get('market_regime', 'NORMAL'), 0) / 5.0
        rsi = features.get('rsi', 50.0) / 100.0
        volume_ratio = np.clip(features.get('volume_ratio', 1.0), 0, 3) / 3.0
        atr_norm = np.clip(features.get('atr', 5.0) / 10.0, 0, 1)
        vix = np.clip(features.get('vix', 15.0) / 40.0, 0, 1)
        volatility_regime_change = 1.0 if features.get('volatility_regime_change', False) else 0.0
        volume_at_exit = volume_ratio  # Same as volume_ratio
        market_state_enc = 0.5  # Default mid-range
        
        # Trade Context (7 features)
        entry_conf = features.get('entry_confidence', 0.75)
        side = 1.0 if features.get('side', 'long').lower() == 'short' else 0.0
        session = features.get('session', 0) / 2.0
        bid_ask_spread = np.clip(features.get('bid_ask_spread_ticks', 0.5) / 3.0, 0, 1)
        commission = np.clip(2.0 / 10.0, 0, 1)
        slippage = np.clip(1.0 / 5.0, 0, 1)
        regime_enc = market_regime_enc
        
        # Time Features (5 features)
        hour = features.get('hour', 12) / 24.0
        day_of_week = features.get('day_of_week', 2) / 6.0
        duration = np.clip(features.get('duration_bars', 1) / 500.0, 0, 1)
        time_in_breakeven = np.clip(features.get('time_in_breakeven_bars', 0) / 100.0, 0, 1)
        bars_until_breakeven = np.clip(features.get('bars_until_breakeven', 999) / 100.0, 0, 1)
        
        # Performance Metrics (5 features)
        mae = np.clip(features.get('mae', 0) / 1000.0, -1, 0)
        mfe = np.clip(features.get('mfe', 0) / 2000.0, 0, 1)
        max_r = np.clip(features.get('max_r_achieved', 0) / 10.0, 0, 1)
        min_r = np.clip(features.get('min_r_achieved', 0) / 5.0, -1, 1)
        r_multiple = np.clip(features.get('r_multiple', 0) / 10.0, -1, 1)
        
        # Exit Strategy State (6 features)
        breakeven_activated = 1.0 if features.get('breakeven_activated', False) else 0.0
        trailing_activated = 1.0 if features.get('trailing_activated', False) else 0.0
        stop_hit = 0.0  # Not hit yet (still in trade)
        exit_param_updates = np.clip(features.get('exit_param_update_count', 0) / 50.0, 0, 1)
        stop_adjustments = np.clip(features.get('stop_adjustment_count', 0) / 20.0, 0, 1)
        bars_until_trailing = np.clip(features.get('bars_until_trailing', 999) / 100.0, 0, 1)
        
        # Results (5 features) - use current values
        current_pnl = features.get('current_pnl', 0)
        pnl_norm = np.clip(current_pnl / 2000.0, -1, 1)
        outcome_current = 1.0 if current_pnl > 0 else 0.0
        win_current = 1.0 if current_pnl > 0 else 0.0
        exit_reason = 0.0  # Unknown yet
        max_profit = mfe  # Same as MFE
        
        # ADVANCED (8 features)
        entry_atr = features.get('entry_atr', features.get('atr', 5.0))
        current_atr = features.get('atr', 5.0)
        atr_change_pct = np.clip((current_atr - entry_atr) / entry_atr * 100.0 / 100.0 if entry_atr > 0 else 0.0, -1, 1)
        avg_atr_trade = np.clip(features.get('avg_atr_during_trade', current_atr) / 10.0, 0, 1)
        peak_r = max_r  # Same as max_r
        profit_dd = np.clip(features.get('profit_drawdown_from_peak', 0) / 2000.0, 0, 1)
        high_vol_bars = np.clip(features.get('high_volatility_bars', 0) / 100.0, 0, 1)
        recent_wins = np.clip(features.get('wins_in_last_5_trades', 0) / 5.0, 0, 1)
        recent_losses = np.clip(features.get('losses_in_last_5_trades', 0) / 5.0, 0, 1)
        mins_to_close = np.clip(features.get('minutes_to_close', 240) / 480.0, 0, 1)
        
        # Return all 44 features in order
        return [
            # Market Context (8)
            market_regime_enc, rsi, volume_ratio, atr_norm, vix, volatility_regime_change, volume_at_exit, market_state_enc,
            # Trade Context (7)
            entry_conf, side, session, bid_ask_spread, commission, slippage, regime_enc,
            # Time Features (5)
            hour, day_of_week, duration, time_in_breakeven, bars_until_breakeven,
            # Performance Metrics (5)
            mae, mfe, max_r, min_r, r_multiple,
            # Exit Strategy State (6)
            breakeven_activated, trailing_activated, stop_hit, exit_param_updates, stop_adjustments, bars_until_trailing,
            # Results (5)
            pnl_norm, outcome_current, win_current, exit_reason, max_profit,
            # ADVANCED (8)
            atr_change_pct, avg_atr_trade, peak_r, profit_dd, high_vol_bars, recent_wins, recent_losses, mins_to_close
        ]
