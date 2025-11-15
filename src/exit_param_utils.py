"""
Exit Parameter Extraction Utilities
Helper functions for extracting and tracking all 131 exit parameters
"""

import sys
import os

# Import parameter definitions
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dev-tools'))
from exit_params_config import EXIT_PARAMS, get_default_exit_params


def extract_all_exit_params(exit_params: dict) -> dict:
    """
    Extract all 131 exit parameters from exit_params dict.
    Uses defaults from exit_params_config if parameter not provided.
    
    Args:
        exit_params: Dict with exit parameters (may be incomplete)
        
    Returns:
        Complete dict with all 131 backtest-learnable parameters
    """
    defaults = get_default_exit_params()
    
    # Extract all 59 backtest-learnable parameters, using provided value or default
    all_params = {}
    for param_name in EXIT_PARAMS.keys():
        all_params[param_name] = exit_params.get(param_name, defaults[param_name])
    
    return all_params


def extract_execution_data(trade_outcome: dict) -> dict:
    """
    Extract execution tracking data from trade outcome.
    Tracks what actually happened during the trade.
    
    Args:
        trade_outcome: Trade outcome dictionary
        
    Returns:
        Dict with execution tracking data
    """
    return {
        # Exit triggers (which conditions actually fired)
        'hit_breakeven': trade_outcome.get('breakeven_activated', False),
        'hit_trailing': trade_outcome.get('trailing_activated', False),
        'hit_stop': trade_outcome.get('stop_hit', False),
        'partials_taken': trade_outcome.get('partial_count', 0),
        
        # Adverse condition triggers
        'adverse_momentum_detected': trade_outcome.get('adverse_momentum', False),
        'dead_trade_timeout': trade_outcome.get('sideways_timeout', False),
        'volume_exhaustion_detected': trade_outcome.get('volume_exhaustion', False),
        'volatility_spike': trade_outcome.get('volatility_spike', False),
        
        # Time-based triggers
        'time_decay_triggered': trade_outcome.get('time_decay_level', 0),  # 0, 50, 75, 90
        'session_exit': trade_outcome.get('session_based_exit', False),
        'friday_exit': trade_outcome.get('friday_exit', False),
        
        # Execution reality (live data only, estimated in backtest)
        'actual_slippage_ticks': trade_outcome.get('slippage_ticks', 0.5),
        'partial_fill_occurred': trade_outcome.get('partial_fill', False),
        'order_rejections': trade_outcome.get('rejection_count', 0),
        'fill_delay_seconds': trade_outcome.get('fill_delay', 0.0),
        'spread_at_entry': trade_outcome.get('bid_ask_spread_ticks', 1.0),
        'spread_at_exit': trade_outcome.get('exit_spread_ticks', 1.0),
        
        # Profit protection
        'profit_locked': trade_outcome.get('profit_lock_triggered', False),
        'max_drawdown_from_peak': trade_outcome.get('profit_drawdown_from_peak', 0.0),
        
        # Runner management
        'runner_held': trade_outcome.get('runner_position', False),
        'runner_exit_reason': trade_outcome.get('runner_exit_reason', ''),
    }


def validate_exit_params_complete(exit_params_used: dict) -> tuple:
    """
    Validate that all 131 exit parameters are present and within range.
    
    Args:
        exit_params_used: Dict with exit parameters
        
    Returns:
        (is_complete: bool, missing_params: list, out_of_range: list)
    """
    all_param_names = set(EXIT_PARAMS.keys())
    provided_params = set(exit_params_used.keys())
    
    missing = list(all_param_names - provided_params)
    out_of_range = []
    
    for name, value in exit_params_used.items():
        if name in EXIT_PARAMS:
            config = EXIT_PARAMS[name]
            if not (config['min'] <= value <= config['max']):
                out_of_range.append(f"{name}={value} (range: [{config['min']}, {config['max']}])")
    
    is_complete = len(missing) == 0 and len(out_of_range) == 0
    return is_complete, missing, out_of_range
