"""
Exit Neural Network Model - 79 BACKTEST-LEARNABLE PARAMETERS
Predicts optimal exit parameters based on market context
Comprehensive exit management covering all trading scenarios
"""
import torch
import torch.nn as nn
import sys
import os

# Import parameter definitions
sys.path.append(os.path.dirname(__file__))
from exit_params_config import EXIT_PARAMS, get_param_ranges, get_default_exit_params

class ExitParamsNet(nn.Module):
    """
    Neural network to predict optimal exit parameters
    
    Inputs (62 features - COMPLETE feature coverage):
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
        - hour (1)
        - day_of_week (1)
        - duration (1)
        - time_in_breakeven_bars (1)
        - bars_until_breakeven (1)
        
        Performance Metrics (5):
        - mae (1: max adverse excursion)
        - mfe (1: max favorable excursion)
        - max_r_achieved (1)
        - min_r_achieved (1)
        - r_multiple (1)
        
        Exit Strategy State (6):
        - breakeven_activated (1: boolean)
        - trailing_activated (1: boolean)
        - stop_hit (1: boolean)
        - exit_param_update_count (1)
        - stop_adjustment_count (1)
        - bars_until_trailing (1)
        
        Results (5):
        - pnl (1)
        - outcome (1: WIN/LOSS encoded)
        - win (1: boolean)
        - exit_reason (1: encoded)
        - max_profit_reached (1)
        
        ADVANCED (8):
        - atr_change_percent (ATR evolution during trade)
        - avg_atr_during_trade (average volatility)
        - peak_r_multiple (maximum profit achieved)
        - profit_drawdown_from_peak (profit given back)
        - high_volatility_bars (volatile period count)
        - wins_in_last_5_trades (recent performance)
        - losses_in_last_5_trades (recent performance)
        - minutes_until_close (time pressure)
        
        TEMPORAL (5):
        - entry_hour (1: hour of trade entry)
        - entry_minute (1: minute of trade entry)
        - exit_hour (1: hour of trade exit)
        - exit_minute (1: minute of trade exit)
        - bars_held (1: number of bars trade was held)
        
        POSITION TRACKING (3):
        - entry_bar (1: bar index when entered)
        - exit_bar (1: bar index when exited)
        - contracts (1: number of contracts)
        
        TRADE CONTEXT (3):
        - trade_number_in_session (1: sequence number)
        - cumulative_pnl_before_trade (1: session P&L context)
        - entry_price (1: price at entry)
        
        PERFORMANCE (4):
        - peak_unrealized_pnl (1: highest unrealized profit)
        - opportunity_cost (1: profit left on table)
        - max_drawdown_percent (1: worst drawdown %)
        - drawdown_bars (1: bars in drawdown)
        
        STRATEGY MILESTONES (4):
        - breakeven_activation_bar (1: when BE triggered)
        - trailing_activation_bar (1: when trailing triggered)
        - duration_bars (1: total bars held)
        - held_through_sessions (1: crossed session boundary)
    
    Outputs (131 exit parameters - COMPREHENSIVE):
        Core Risk (21): stops, breakeven, trailing, partials
        Time-Based (5): timeouts, time decay
        Adverse (9): momentum, profit protection, dead trades
        Runner (5): runner optimization
        Stop Bleeding (5): loss control (removed fixed $ stop)
        Market Conditions (4): spread, volatility, liquidity
        Execution (6): fills, rejections, margin
        Recovery (4): daily limits, drawdown
        Session (4): pre-close, low volume, overnight, Friday
        Adaptive (3): ML overrides, regime changes
        Exit Strategy Control (16): RL-controlled breakeven/trailing/stop decisions
        Additional Parameters (49): extended exit conditions and thresholds
    """
    
    def __init__(self, input_size=205, hidden_size=256):
        super(ExitParamsNet, self).__init__()
        
        # Architecture: 205 inputs → hidden_size → hidden_size → 131 outputs
        # 205 inputs: 10 market_state + 63 outcome + 132 exit_params
        # 131 outputs: comprehensive exit parameters
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size),  # Use hidden_size consistently
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, 131),  # 131 exit parameters
            nn.Sigmoid()  # Output 0-1, will denormalize later
        )
    
    def forward(self, x):
        return self.network(x)


def denormalize_exit_params(normalized_params):
    """
    Convert normalized [0-1] outputs back to real exit parameters
    
    Args:
        normalized_params: Tensor of shape [batch, 68] with values 0-1
    
    Returns:
        dict with actual exit parameter values (59 backtest-learnable params)
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


def normalize_exit_params(exit_params):
    """
    Normalize exit parameters to [0-1] range for training
    
    Args:
        exit_params: dict with exit parameter values
    
    Returns:
        list of 68 normalized values [0-1]
    """
    ranges = get_param_ranges()
    defaults = get_default_exit_params()
    param_names = list(EXIT_PARAMS.keys())
    
    normalized = []
    for name in param_names:
        min_val, max_val = ranges[name]
        # Use default if missing
        value = exit_params.get(name, defaults[name])
        norm = (value - min_val) / (max_val - min_val)
        norm = max(0.0, min(1.0, norm))  # Clip to [0, 1]
        normalized.append(norm)
    
    return normalized
