#!/usr/bin/env python3
"""
Development Backtesting Environment for VWAP Bounce Bot

This is the development/testing environment that:
- Runs backtests to test bot performance
- Loads signal RL locally from data/signal_experience.json
- Includes pattern matching and all trading logic
- Handles all regimes (HIGH_VOL_TRENDING, HIGH_VOL_CHOPPY, etc.)
- Follows same UTC maintenance and flatten rules as production
- Does everything the live bot does with all trade management

Separated from production bot for clean architecture.
"""

import argparse
import sys
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from types import ModuleType
import pytz

# Add parent directory to path to import from src/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# Import backtesting framework from dev
from backtesting import BacktestConfig, BacktestEngine, ReportGenerator

# Import production bot modules
from config import load_config
from monitoring import setup_logging
from signal_confidence import SignalConfidenceRL


def parse_arguments():
    """Parse command-line arguments for backtest"""
    parser = argparse.ArgumentParser(
        description='VWAP Bounce Bot - Development Backtesting Environment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest for last 30 days (bar-by-bar with 1-minute bars)
  python dev/run_backtest.py --days 30
  
  # Run backtest with specific date range
  python dev/run_backtest.py --start 2024-01-01 --end 2024-01-31
  
  # Run backtest with tick-by-tick replay (requires tick data)
  python dev/run_backtest.py --days 7 --use-tick-data
  
  # Save backtest report to file
  python dev/run_backtest.py --days 30 --report backtest_results.txt
        """
    )
    
    parser.add_argument(
        '--start',
        type=str,
        help='Backtest start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        help='Backtest end date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        help='Backtest for last N days (alternative to --start/--end)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to historical data directory (default: <project_root>/data/historical_data)'
    )
    
    parser.add_argument(
        '--initial-equity',
        type=float,
        default=50000.0,
        help='Initial equity for backtesting (default: 50000)'
    )
    
    parser.add_argument(
        '--report',
        type=str,
        help='Save backtest report to specified file'
    )
    
    parser.add_argument(
        '--use-tick-data',
        action='store_true',
        help='Use tick-by-tick replay instead of bar-by-bar (requires tick data files)'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        help='Override trading symbol (default: MES)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def initialize_rl_brains_for_backtest() -> Tuple[Any, ModuleType]:
    """
    Initialize RL brain (signal confidence) for backtest mode.
    This ensures experience files are loaded before the backtest runs.
    
    Returns:
        Tuple of (rl_brain, bot_module) where rl_brain is the SignalConfidenceRL 
        instance and bot_module is the loaded trading engine module
    """
    logger = logging.getLogger('backtest')
    
    # Import the bot module to access its RL brain
    # Note: We need to import it dynamically since quotrading_engine is the actual module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "quotrading_engine",
        os.path.join(PROJECT_ROOT, "src/quotrading_engine.py")
    )
    bot_module = importlib.util.module_from_spec(spec)
    sys.modules['quotrading_engine'] = bot_module
    
    # Also make it available as vwap_bounce_bot for compatibility
    sys.modules['vwap_bounce_bot'] = bot_module
    
    # Load the module
    spec.loader.exec_module(bot_module)
    
    # Initialize RL brain with experience file
    signal_exp_file = os.path.join(PROJECT_ROOT, "data/signal_experience.json")
    rl_brain = SignalConfidenceRL(
        experience_file=signal_exp_file,
        backtest_mode=True
    )
    
    # Set it on the bot module if it has rl_brain attribute
    if hasattr(bot_module, 'rl_brain'):
        bot_module.rl_brain = rl_brain
    
    logger.info(f"✅ RL BRAIN INITIALIZED for backtest - {len(rl_brain.experiences)} signal experiences loaded")
    
    return rl_brain, bot_module


def run_backtest(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run backtesting mode - completely independent of broker API.
    Uses historical data to replay market conditions and simulate trading.
    
    This backtest environment:
    - Loads signal RL from data/signal_experience.json
    - Uses pattern matching for signal detection
    - Handles all market regimes
    - Follows UTC maintenance and flatten rules
    - Executes all trade management logic
    - Everything the live bot does
    
    Args:
        args: Parsed command-line arguments from argparse
        
    Returns:
        Dictionary with backtest performance metrics
    """
    logger = logging.getLogger('backtest')
    logger.info("="*60)
    logger.info("STARTING BACKTEST MODE (Development Environment)")
    logger.info("Backtesting does NOT use broker API - runs on historical data only")
    logger.info("="*60)
    
    # Set backtest mode environment variable
    os.environ['BOT_BACKTEST_MODE'] = 'true'
    
    # Load configuration
    bot_config = load_config(backtest_mode=True)
    
    # Override symbol if specified
    if args.symbol:
        bot_config.instrument = args.symbol
    
    # Determine date range
    tz = pytz.timezone(bot_config.timezone)
    
    if args.start and args.end:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    elif args.days:
        end_date = datetime.now(tz)
        start_date = end_date - timedelta(days=args.days)
    else:
        # Default: last 7 days
        end_date = datetime.now(tz)
        start_date = end_date - timedelta(days=7)
        
    # Create backtest configuration
    data_path = args.data_path if args.data_path else os.path.join(PROJECT_ROOT, "data/historical_data")
    
    backtest_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_equity=args.initial_equity,
        symbols=[args.symbol] if args.symbol else [bot_config.instrument],
        data_path=data_path,
        use_tick_data=args.use_tick_data
    )
    
    logger.info(f"Backtest Configuration:")
    logger.info(f"  Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"  Initial Equity: ${backtest_config.initial_equity:,.2f}")
    logger.info(f"  Symbols: {', '.join(backtest_config.symbols)}")
    logger.info(f"  Data Path: {backtest_config.data_path}")
    logger.info(f"  Replay Mode: {'Tick-by-tick' if args.use_tick_data else 'Bar-by-bar (1-minute bars)'}")
    
    # Create backtest engine
    bot_config_dict = bot_config.to_dict()
    engine = BacktestEngine(backtest_config, bot_config_dict)
    
    # Initialize RL brain and bot module
    rl_brain, bot_module = initialize_rl_brains_for_backtest()
    
    # Import bot functions from the loaded module
    initialize_state = bot_module.initialize_state
    on_tick = bot_module.on_tick
    check_for_signals = bot_module.check_for_signals
    check_exit_conditions = bot_module.check_exit_conditions
    check_daily_reset = bot_module.check_daily_reset
    state = bot_module.state
    
    # Initialize bot state for backtesting
    symbol = bot_config_dict['instrument']
    initialize_state(symbol)
    
    # Create a simple object to hold RL brain reference for tracking
    class BotRLReferences:
        def __init__(self):
            self.signal_rl = rl_brain
    
    # Set bot instance for RL tracking
    bot_ref = BotRLReferences()
    engine.set_bot_instance(bot_ref)
    
    # Use Eastern timezone for daily reset checks (follows production rules)
    eastern_tz = pytz.timezone("US/Eastern")
    
    def vwap_strategy_backtest(bars_1min: List[Dict[str, Any]], bars_15min: List[Dict[str, Any]]) -> None:
        """
        Actual VWAP Bounce strategy integrated with backtest engine.
        Processes historical data through the real bot logic.
        
        This executes:
        - Signal RL for confidence scoring
        - Pattern matching for signal detection
        - Regime detection for market adaptation
        - All trade management (stops, targets, breakeven, trailing)
        - UTC maintenance and flatten rules
        """
        for bar in bars_1min:
            # Extract bar data
            timestamp = bar['timestamp']
            price = bar['close']
            volume = bar['volume']
            timestamp_ms = int(timestamp.timestamp() * 1000)
            
            # Check for new trading day (resets daily counters following production rules)
            timestamp_eastern = timestamp.astimezone(eastern_tz)
            check_daily_reset(symbol, timestamp_eastern)
            
            # Process tick through actual bot logic
            # This includes all signal detection, pattern matching, regime handling
            on_tick(symbol, price, volume, timestamp_ms)
            
            # Check for entry signals after each bar
            # Uses RL confidence, pattern matching, and regime-aware logic
            check_for_signals(symbol)
            
            # Check for exit signals
            # Handles stops, targets, breakeven, trailing, time decay
            check_exit_conditions(symbol)
            
            # Update backtest engine with current position from bot state
            if symbol in state and 'position' in state[symbol]:
                pos = state[symbol]['position']
                
                # If bot has active position and backtest engine doesn't have it, record it
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
                    logger.info(f"Backtest: {pos['side'].upper()} position entered at {pos['entry_price']}")
                    
                # If bot closed position (active=False), close it in backtest engine too
                elif not pos.get('active') and engine.current_position is not None:
                    exit_price = price
                    exit_time = timestamp
                    exit_reason = 'bot_exit'
                    engine._close_position(exit_time, exit_price, exit_reason)
                    logger.info(f"Backtest: Position closed at {exit_price}, reason: {exit_reason}")
        
    # Run backtest with integrated strategy
    results = engine.run_with_strategy(vwap_strategy_backtest)
    
    # Save RL experiences after backtest completion
    print("\nSaving RL experiences...")
    try:
        if rl_brain is not None and hasattr(rl_brain, 'save_experience'):
            rl_brain.save_experience()
            print("✅ Signal RL experiences saved")
    except Exception as e:
        print(f"⚠️ Failed to save signal RL experiences: {e}")
    
    # Generate report
    report_gen = ReportGenerator(engine.metrics)
    
    # Track RL learning progress
    print("\n" + "="*60)
    print("RL BRAIN LEARNING SUMMARY")
    print("="*60)
    
    # Read the experience files directly
    try:
        import json
        signal_exp_file = os.path.join(PROJECT_ROOT, "data/signal_experience.json")
        with open(signal_exp_file, 'r') as f:
            signal_data = json.load(f)
            signal_count = len(signal_data['experiences'])
            signal_wins = len([e for e in signal_data['experiences'] if e['reward'] > 0])
            signal_losses = len([e for e in signal_data['experiences'] if e['reward'] < 0])
            signal_wr = (signal_wins / signal_count * 100) if signal_count > 0 else 0
            
            print(f"[SIGNALS] {signal_count} total experiences")
            print(f"  Wins: {signal_wins} | Losses: {signal_losses} | Win Rate: {signal_wr:.1f}%")
    except Exception as e:
        print(f"Could not load signal experiences: {e}")
        
    print("="*60)
    
    # Print results to console
    logger.info("\n" + "="*60)
    logger.info("BACKTEST RESULTS")
    logger.info("="*60)
    logger.info(report_gen.generate_trade_breakdown())
    logger.info("\n")
    
    # Save report if requested
    if args.report:
        report_gen.save_report(args.report)
        logger.info(f"Report saved to: {args.report}")
    
    logger.info("="*60)
    logger.info("BACKTEST COMPLETE")
    logger.info("="*60)
    
    return results


def main():
    """Main entry point for development backtesting"""
    args = parse_arguments()
    
    # Setup logging
    config_dict = {'log_directory': os.path.join(PROJECT_ROOT, 'logs')}
    logger = setup_logging(config_dict)
    
    # Set log level
    logger.setLevel(getattr(logging, args.log_level))
    
    logger.info("="*60)
    logger.info("VWAP Bounce Bot - Development Backtest Environment")
    logger.info("="*60)
    logger.info("Features:")
    logger.info("  ✅ Signal RL loaded from data/signal_experience.json")
    logger.info("  ✅ Pattern matching for signal detection")
    logger.info("  ✅ Regime detection and adaptation")
    logger.info("  ✅ All trade management (stops, targets, breakeven, trailing)")
    logger.info("  ✅ UTC maintenance and flatten rules")
    logger.info("  ✅ Everything the live bot does")
    logger.info("="*60 + "\n")
    
    # Run backtest
    try:
        results = run_backtest(args)
        
        # Exit with success/failure based on results
        if results and results.get('total_trades', 0) > 0:
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
