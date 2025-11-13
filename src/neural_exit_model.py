"""
Exit Neural Network Model
Predicts optimal exit parameters based on market context
"""
import torch
import torch.nn as nn

class ExitParamsNet(nn.Module):
    """
    Neural network to predict optimal exit parameters
    
    Inputs (15 features):
        - market_regime (one-hot: 3)
        - rsi (1)
        - volume_ratio (1) 
        - atr (1)
        - entry_confidence (1)
        - hour (1)
        - day_of_week (1)
        - vix (1)
        - vwap_distance (1)
        - streak (1)
        - recent_pnl (1)
        - side (1: 0=LONG, 1=SHORT)
        - session (1: 0=Asia, 1=London, 2=NY)
    
    Outputs (6 exit parameters):
        - breakeven_threshold_ticks (normalized)
        - trailing_distance_ticks (normalized)
        - stop_mult (normalized)
        - partial_1_r (normalized 2R target)
        - partial_2_r (normalized 3R target)
        - partial_3_r (normalized 5R target)
    """
    
    def __init__(self, input_size=15, hidden_size=32):
        super(ExitParamsNet, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            
            nn.Linear(16, 6),  # 6 exit parameters
            nn.Sigmoid()  # Output 0-1, will denormalize later
        )
    
    def forward(self, x):
        return self.network(x)


def denormalize_exit_params(normalized_params):
    """
    Convert normalized [0-1] outputs back to real exit parameters
    
    Args:
        normalized_params: Tensor of shape [batch, 6] with values 0-1
    
    Returns:
        dict with actual exit parameter values
    """
    # Denormalization ranges (min, max for each param)
    ranges = {
        'breakeven_threshold_ticks': (6, 18),    # 6-18 ticks
        'trailing_distance_ticks': (8, 24),      # 8-24 ticks
        'stop_mult': (2.5, 5.0),                 # 2.5-5.0x ATR
        'partial_1_r': (1.5, 3.0),               # 1.5-3.0R
        'partial_2_r': (2.5, 4.5),               # 2.5-4.5R
        'partial_3_r': (4.0, 8.0),               # 4.0-8.0R
    }
    
    param_names = [
        'breakeven_threshold_ticks',
        'trailing_distance_ticks', 
        'stop_mult',
        'partial_1_r',
        'partial_2_r',
        'partial_3_r'
    ]
    
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


def normalize_exit_params(exit_params):
    """
    Normalize exit parameters to [0-1] range for training
    
    Args:
        exit_params: dict with exit parameter values
    
    Returns:
        list of 6 normalized values [0-1]
    """
    ranges = {
        'breakeven_threshold_ticks': (6, 18),
        'trailing_distance_ticks': (8, 24),
        'stop_mult': (2.5, 5.0),
        'partial_1_r': (1.5, 3.0),
        'partial_2_r': (2.5, 4.5),
        'partial_3_r': (4.0, 8.0),
    }
    
    param_names = [
        'breakeven_threshold_ticks',
        'trailing_distance_ticks',
        'stop_mult',
        'partial_1_r',
        'partial_2_r',
        'partial_3_r'
    ]
    
    normalized = []
    for name in param_names:
        min_val, max_val = ranges[name]
        value = exit_params.get(name, (min_val + max_val) / 2)  # Use midpoint if missing
        norm = (value - min_val) / (max_val - min_val)
        norm = max(0.0, min(1.0, norm))  # Clip to [0, 1]
        normalized.append(norm)
    
    return normalized
