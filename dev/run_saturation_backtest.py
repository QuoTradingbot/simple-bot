#!/usr/bin/env python3
"""
Saturation Backtest Script - Run RL Learning Until Experience Saturation

This script runs the backtest repeatedly on historical data until no new unique
experiences can be added (saturation). Uses the existing RL duplicate prevention
which checks all significant fields including timestamp, price, P&L, duration,
exit reason, MFE/MAE, etc.

Features:
- Uses configurable exploration rate (default 30%) to discover new experiences
- Runs backtests in a loop until saturation is reached
- Tracks progress: new experiences per iteration
- Uses SignalConfidenceRL's hash-based duplicate detection
- Saves experiences after each iteration

Usage:
    python dev/run_saturation_backtest.py                    # Default: 30% exploration
    python dev/run_saturation_backtest.py --exploration 0.5  # Custom 50% exploration
    python dev/run_saturation_backtest.py --max-iterations 20  # Limit iterations
    python dev/run_saturation_backtest.py --symbol ES        # Specify symbol
"""

import argparse
import sys
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from types import ModuleType
import pytz
import json

# CRITICAL: Set backtest mode BEFORE any imports that load the bot module
os.environ['BOT_BACKTEST_MODE'] = 'true'
os.environ['USE_CLOUD_SIGNALS'] = 'false'

# Add parent directory to path to import from src/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# Import backtesting framework from dev
from backtesting import BacktestConfig, BacktestEngine
from backtest_reporter import reset_reporter, get_reporter

# Import production bot modules
from config import load_config
from signal_confidence import SignalConfidenceRL


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Run backtests repeatedly until RL experience saturation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default 30% exploration until saturation
    python dev/run_saturation_backtest.py
    
    # Run with 50% exploration
    python dev/run_saturation_backtest.py --exploration 0.5
    
    # Limit to 10 iterations
    python dev/run_saturation_backtest.py --max-iterations 10
    
    # Specify symbol
    python dev/run_saturation_backtest.py --symbol ES
        """
    )
    
    parser.add_argument(
        '--exploration',
        type=float,
        default=0.30,
        help='Exploration rate (0.0-1.0). Default: 0.30 (30%%)'
    )
    
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=100,
        help='Maximum number of backtest iterations. Default: 100'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='ES',
        help='Trading symbol. Default: ES'
    )
    
    parser.add_argument(
        '--min-new-experiences',
        type=int,
        default=0,
        help='Stop when fewer than this many new experiences are added. Default: 0 (only stop at 0)'
    )
    
    parser.add_argument(
        '--consecutive-zero',
        type=int,
        default=3,
        help='Number of consecutive zero-gain iterations before stopping. Default: 3'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='WARNING',
        help='Logging level. Default: WARNING'
    )
    
    return parser.parse_args()


def initialize_rl_brain(symbol: str, exploration_rate: float) -> Tuple[SignalConfidenceRL, ModuleType]:
    """
    Initialize RL brain with specified exploration rate.
    
    Args:
        symbol: Trading symbol (e.g., 'ES')
        exploration_rate: Exploration rate (0.0-1.0)
    
    Returns:
        Tuple of (rl_brain, bot_module)
    """
    # Import the bot module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "quotrading_engine",
        os.path.join(PROJECT_ROOT, "src/quotrading_engine.py")
    )
    bot_module = importlib.util.module_from_spec(spec)
    sys.modules['quotrading_engine'] = bot_module
    sys.modules['capitulation_reversal_bot'] = bot_module
    
    # Load the module
    spec.loader.exec_module(bot_module)
    
    # Initialize RL brain with specified exploration rate
    signal_exp_file = os.path.join(PROJECT_ROOT, f"experiences/{symbol}/signal_experience.json")
    
    rl_brain = SignalConfidenceRL(
        experience_file=signal_exp_file,
        backtest_mode=True,
        confidence_threshold=None,  # Use adaptive threshold
        exploration_rate=exploration_rate,
        min_exploration=exploration_rate,  # Keep exploration constant
        exploration_decay=1.0  # No decay
    )
    
    # Set the global rl_brain in the bot module
    bot_module.__dict__['rl_brain'] = rl_brain
    
    return rl_brain, bot_module


def run_single_backtest(bot_module: ModuleType, rl_brain: SignalConfidenceRL, 
                        symbol: str, bot_config: Dict[str, Any],
                        start_date: datetime, end_date: datetime) -> int:
    """
    Run a single backtest and return the number of new experiences added.
    
    Args:
        bot_module: The loaded trading engine module
        rl_brain: The RL brain instance
        symbol: Trading symbol
        bot_config: Bot configuration dict
        start_date: Backtest start date
        end_date: Backtest end date
    
    Returns:
        Number of new experiences added
    """
    import io
    import contextlib
    
    # Count experiences before
    initial_count = len(rl_brain.experiences)
    
    # Create backtest config
    data_path = os.path.join(PROJECT_ROOT, "data/historical_data")
    
    backtest_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_equity=50000.0,
        symbols=[symbol],
        data_path=data_path,
        use_tick_data=False
    )
    
    # Create backtest engine
    engine = BacktestEngine(backtest_config, bot_config)
    
    # Import bot functions
    initialize_state = bot_module.initialize_state
    inject_complete_bar = bot_module.inject_complete_bar
    check_daily_reset = bot_module.check_daily_reset
    check_vwap_reset = bot_module.check_vwap_reset
    state = bot_module.state
    
    # Initialize bot state
    initialize_state(symbol)
    
    # Set bot instance for RL tracking
    class BotRLReferences:
        def __init__(self):
            self.signal_rl = rl_brain
    
    engine.set_bot_instance(BotRLReferences())
    
    # Eastern timezone for daily reset checks
    eastern_tz = pytz.timezone("US/Eastern")
    
    # Track position state
    prev_position_active = False
    last_exit_reason = 'bot_exit'
    
    def strategy_func(bars_1min: List[Dict[str, Any]], bars_15min: List[Dict[str, Any]]) -> None:
        nonlocal prev_position_active, last_exit_reason
        
        for bar in bars_1min:
            timestamp = bar['timestamp']
            timestamp_eastern = timestamp.astimezone(eastern_tz)
            
            # Check for new trading day
            check_daily_reset(symbol, timestamp_eastern)
            check_vwap_reset(symbol, timestamp_eastern)
            
            # Inject bar data
            inject_complete_bar(symbol, bar)
            
            # Track position state
            if symbol in state and 'position' in state[symbol]:
                pos = state[symbol]['position']
                current_active = pos.get('active', False)
                
                if current_active or (not current_active and prev_position_active):
                    if 'last_exit_reason' in state[symbol]:
                        last_exit_reason = state[symbol]['last_exit_reason']
                
                prev_position_active = current_active
                
                # Update backtest engine position tracking
                if pos.get('active') and engine.current_position is None:
                    engine.current_position = {
                        'symbol': symbol,
                        'side': pos['side'],
                        'quantity': pos.get('quantity', 1),
                        'entry_price': pos['entry_price'],
                        'entry_time': pos.get('entry_time', timestamp),
                        'stop_price': pos.get('stop_price'),
                        'target_price': pos.get('target_price')
                    }
                elif not pos.get('active') and engine.current_position is not None:
                    exit_price = bar['close']
                    engine._close_position(timestamp, exit_price, last_exit_reason)
                    last_exit_reason = 'bot_exit'
    
    # Run backtest with suppressed output
    # Capture and discard all stdout/stderr from the backtest engine
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        engine.run_with_strategy(strategy_func)
    
    # Save experiences
    rl_brain.save_experience()
    
    # Count new experiences
    final_count = len(rl_brain.experiences)
    new_experiences = final_count - initial_count
    
    return new_experiences


def main():
    """Main entry point for saturation backtest"""
    args = parse_arguments()
    
    # Configure logging - suppress all verbose output for clean progress display
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(level=logging.CRITICAL, format='%(message)s')
    
    # Suppress ALL verbose loggers for clean saturation output
    for logger_name in ['quotrading_engine', 'backtesting', 'signal_confidence', 
                        'regime_detection', 'capitulation_detector', 'root', '__main__',
                        'asyncio', 'urllib3', 'requests', 'httpx']:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    
    # Suppress the root logger too
    logging.getLogger().setLevel(logging.CRITICAL)
    
    # Load configuration
    bot_config = load_config(backtest_mode=True)
    bot_config.instrument = args.symbol
    bot_config.account_size = 50000.0
    bot_config.max_contracts = 1
    bot_config_dict = bot_config.to_dict()
    
    # Get date range from historical data
    data_path = os.path.join(PROJECT_ROOT, "data/historical_data")
    csv_path = os.path.join(data_path, f"{args.symbol}_1min.csv")
    
    if not os.path.exists(csv_path):
        print(f"ERROR: Historical data file not found: {csv_path}")
        sys.exit(1)
    
    # Read date range from CSV
    tz = pytz.timezone('US/Eastern')
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            print("ERROR: Historical data file is empty")
            sys.exit(1)
        
        # First data line
        first_line = lines[1]
        first_timestamp = first_line.split(',')[0]
        if '+' in first_timestamp:
            first_timestamp = first_timestamp.split('+')[0]
        start_date = datetime.strptime(first_timestamp, '%Y-%m-%d %H:%M:%S')
        start_date = tz.localize(start_date.replace(hour=0, minute=0, second=0))
        
        # Last data line
        last_line = lines[-1]
        last_timestamp = last_line.split(',')[0]
        if '+' in last_timestamp:
            last_timestamp = last_timestamp.split('+')[0]
        end_date = datetime.strptime(last_timestamp, '%Y-%m-%d %H:%M:%S')
        end_date = tz.localize(end_date.replace(hour=23, minute=59, second=59))
    
    # Count unique days
    unique_days = set()
    for line in lines[1:]:
        date_str = line.split(',')[0].split(' ')[0]
        unique_days.add(date_str)
    
    # Print header
    print("=" * 70)
    print("  SATURATION BACKTEST - RL Experience Learning")
    print("=" * 70)
    print(f"  Symbol:           {args.symbol}")
    print(f"  Date Range:       {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"  Trading Days:     {len(unique_days)}")
    print(f"  Total Bars:       {len(lines) - 1:,}")
    print(f"  Exploration Rate: {args.exploration * 100:.0f}%")
    print(f"  Max Iterations:   {args.max_iterations}")
    print(f"  Consecutive Zero Stop: {args.consecutive_zero}")
    print("=" * 70)
    print()
    
    # Load initial experience count
    exp_file = os.path.join(PROJECT_ROOT, f"experiences/{args.symbol}/signal_experience.json")
    initial_experiences = 0
    if os.path.exists(exp_file):
        with open(exp_file, 'r') as f:
            data = json.load(f)
            initial_experiences = len(data.get('experiences', []))
    
    print(f"  Starting Experiences: {initial_experiences}")
    print()
    print("  Iteration Progress:")
    print("  " + "-" * 60)
    
    # Track iterations
    consecutive_zero = 0
    total_new_experiences = 0
    iteration = 0
    
    # Run iterations
    while iteration < args.max_iterations:
        iteration += 1
        
        # Re-initialize RL brain for each iteration
        # This ensures experiences from previous iterations are loaded
        rl_brain, bot_module = initialize_rl_brain(args.symbol, args.exploration)
        
        experiences_before = len(rl_brain.experiences)
        
        # Run backtest
        new_experiences = run_single_backtest(
            bot_module, rl_brain, args.symbol, bot_config_dict,
            start_date, end_date
        )
        
        total_new_experiences += new_experiences
        current_total = len(rl_brain.experiences)
        
        # Print progress
        status = "✓" if new_experiences > 0 else "○"
        print(f"  {status} Iteration {iteration:3d}: {experiences_before:4d} → {current_total:4d} (+{new_experiences:3d} new)")
        
        # Check for saturation
        if new_experiences <= args.min_new_experiences:
            consecutive_zero += 1
            if consecutive_zero >= args.consecutive_zero:
                print()
                print(f"  ✓ SATURATION REACHED after {consecutive_zero} consecutive iterations with 0 new experiences")
                break
        else:
            consecutive_zero = 0
    
    # Print summary
    print()
    print("  " + "-" * 60)
    print("  SATURATION SUMMARY")
    print("  " + "-" * 60)
    
    # Load final experience count
    final_experiences = 0
    if os.path.exists(exp_file):
        with open(exp_file, 'r') as f:
            data = json.load(f)
            final_experiences = len(data.get('experiences', []))
    
    print(f"  Initial Experiences:  {initial_experiences}")
    print(f"  Final Experiences:    {final_experiences}")
    print(f"  New Experiences:      {final_experiences - initial_experiences}")
    print(f"  Total Iterations:     {iteration}")
    print(f"  Exploration Rate:     {args.exploration * 100:.0f}%")
    
    if iteration >= args.max_iterations:
        print(f"  Status:               Max iterations reached (may not be saturated)")
    else:
        print(f"  Status:               SATURATED (no new unique experiences)")
    
    print("=" * 70)
    
    sys.exit(0)


if __name__ == '__main__':
    main()
