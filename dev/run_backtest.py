#!/usr/bin/env python3
"""
Development Backtesting Environment for Capitulation Reversal Bot

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

# CRITICAL: Set backtest mode BEFORE any imports that load the bot module
# This ensures config validation skips broker requirements
os.environ['BOT_BACKTEST_MODE'] = 'true'
# Disable cloud API calls during backtest (use local RL only)
os.environ['USE_CLOUD_SIGNALS'] = 'false'

# Add parent directory to path to import from src/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# Import backtesting framework from dev
from backtesting import BacktestConfig, BacktestEngine, ReportGenerator
from backtest_reporter import reset_reporter, get_reporter

# Import production bot modules
from config import load_config
from monitoring import setup_logging
from signal_confidence import SignalConfidenceRL


def parse_arguments():
    """Parse command-line arguments for backtest"""
    parser = argparse.ArgumentParser(
        description='Capitulation Reversal Bot - Development Backtesting Environment',
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

Note: All trading parameters (account_size, max_contracts, rl_exploration_rate, etc.) 
      are configured in data/config.json
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


def initialize_rl_brains_for_backtest(bot_config) -> Tuple[Any, ModuleType]:
    """
    Initialize RL brain (signal confidence) for backtest mode.
    This ensures experience files are loaded before the backtest runs.
    
    Args:
        bot_config: Bot configuration object with RL parameters
    
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
    
    # Also make it available as capitulation_reversal_bot for compatibility
    sys.modules['capitulation_reversal_bot'] = bot_module
    
    # Load the module
    spec.loader.exec_module(bot_module)
    
    # Get symbol for symbol-specific experience folder
    symbol = bot_config.instrument
    
    # Initialize RL brain with symbol-specific experience file
    # Using 30% exploration and 70% confidence threshold
    signal_exp_file = os.path.join(PROJECT_ROOT, f"experiences/{symbol}/signal_experience.json")
    rl_brain = SignalConfidenceRL(
        experience_file=signal_exp_file,
        backtest_mode=True,
        confidence_threshold=0.70,  # 70% confidence threshold
        exploration_rate=0.30,  # 30% exploration
        min_exploration=0.30,   # Keep at 30%
        exploration_decay=1.0  # No decay - maintain exploration rate
    )
    
    # Set the global rl_brain in the bot module's namespace
    # This is critical - the module uses 'global rl_brain' which looks up in module.__dict__
    bot_module.__dict__['rl_brain'] = rl_brain
    
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
    
    # Get the clean reporter
    reporter = get_reporter()
    
    # Backtest mode environment variables already set at module import
    # (see top of file - BOT_BACKTEST_MODE and USE_CLOUD_SIGNALS)
    
    # Load configuration - use defaults, don't load from GUI/live config files
    # IMPORTANT: Backtesting is completely isolated from live trading configuration
    bot_config = load_config(backtest_mode=True)
    
    # BACKTEST-SPECIFIC OVERRIDES - These are hardcoded defaults for backtesting
    # They do NOT affect live trading in any way
    bot_config.account_size = 50000.0  # Standard backtest account size
    bot_config.max_contracts = 1  # Single contract for backtesting (no position sizing)
    bot_config.daily_loss_limit = 1000.0  # Standard daily loss limit for testing
    bot_config.shadow_mode = False  # Backtesting always executes simulated trades
    
    # Override symbol if specified via command line
    if args.symbol:
        bot_config.instrument = args.symbol
    
    # Extract symbol once - used throughout this function
    symbol = bot_config.instrument
    
    # Convert config to dict early for header
    bot_config_dict = bot_config.to_dict()
    
    # Determine date range
    tz = pytz.timezone(bot_config.timezone)
    
    if args.start and args.end:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    elif args.days:
        # Load the CSV to get the actual end date of available data
        data_path = args.data_path if args.data_path else os.path.join(PROJECT_ROOT, "data/historical_data")
        csv_path = os.path.join(data_path, f"{symbol}_1min.csv")
        
        if os.path.exists(csv_path):
            # Read last line to get actual end date from data
            with open(csv_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:  # Skip header
                    last_line = lines[-1]
                    last_timestamp = last_line.split(',')[0]
                    # Handle timezone-aware timestamp format (e.g., "2025-11-27 18:00:00+00:00")
                    if '+' in last_timestamp:
                        last_timestamp = last_timestamp.split('+')[0]  # Remove timezone offset
                    end_date = datetime.strptime(last_timestamp, '%Y-%m-%d %H:%M:%S')
                    end_date = tz.localize(end_date.replace(hour=23, minute=59, second=59))
                else:
                    end_date = datetime.now(tz)
        else:
            end_date = datetime.now(tz)
        
        start_date = end_date - timedelta(days=args.days)
    else:
        # Default: last 7 days
        end_date = datetime.now(tz)
        start_date = end_date - timedelta(days=7)
    
    # Print clean header
    reporter.print_header(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        symbol=symbol,
        config=bot_config_dict
    )
        
    # Create backtest configuration
    data_path = args.data_path if args.data_path else os.path.join(PROJECT_ROOT, "data/historical_data")
    
    backtest_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_equity=bot_config.account_size,
        symbols=[symbol],
        data_path=data_path,
        use_tick_data=args.use_tick_data
    )
    
    # Suppress verbose logger output - keep only essential info
    logger.setLevel(logging.CRITICAL)
    
    # Create backtest engine
    engine = BacktestEngine(backtest_config, bot_config_dict)
    
    # Suppress engine logger warnings for clean output
    if hasattr(engine, 'logger'):
        engine.logger.setLevel(logging.CRITICAL)
    
    # Initialize RL brain and bot module with config values
    rl_brain, bot_module = initialize_rl_brains_for_backtest(bot_config)
    
    # Track initial experience count to show how many were added during backtest
    initial_experience_count = len(rl_brain.experiences) if rl_brain else 0
    
    # Import bot functions from the loaded module
    initialize_state = bot_module.initialize_state
    inject_complete_bar = bot_module.inject_complete_bar
    check_for_signals = bot_module.check_for_signals
    check_exit_conditions = bot_module.check_exit_conditions
    check_daily_reset = bot_module.check_daily_reset
    check_vwap_reset = bot_module.check_vwap_reset
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
    
    # Track previous position state to detect trade completions
    prev_position_active = False
    bars_processed = 0
    total_bars = 0
    
    # Track RL confidence for each trade
    trade_confidences = {}
    last_exit_reason = 'bot_exit'  # Track last exit reason
    prev_position_active = False
    
    def capitulation_strategy_backtest(bars_1min: List[Dict[str, Any]], bars_15min: List[Dict[str, Any]]) -> None:
        """
        Capitulation Reversal strategy integrated with backtest engine.
        Processes historical data through the real bot logic.
        
        This executes:
        - Signal RL for confidence scoring
        - Capitulation/flush pattern detection
        - Regime detection for market adaptation
        - All trade management (stops, targets, breakeven, trailing)
        - UTC maintenance and flatten rules
        """
        nonlocal prev_position_active, bars_processed, total_bars, last_exit_reason
        total_bars = len(bars_1min)
        
        for bar_idx, bar in enumerate(bars_1min):
            bars_processed = bar_idx + 1
            
            # Update progress less frequently - every 10% or every 500 bars (whichever is larger)
            progress_interval = max(500, total_bars // 10)  # Show 10 updates max
            if bars_processed % progress_interval == 0 or bars_processed == total_bars:
                reporter.update_progress(bars_processed, total_bars)
            
            # Extract bar data
            timestamp = bar['timestamp']
            timestamp_eastern = timestamp.astimezone(eastern_tz)
            
            # Check for new trading day (resets daily counters following production rules)
            check_daily_reset(symbol, timestamp_eastern)
            
            # Check for VWAP reset at 6PM ET (futures trading day start)
            check_vwap_reset(symbol, timestamp_eastern)
            
            # CRITICAL: Use inject_complete_bar to preserve OHLC data for accurate ATR calculation
            # This is essential for proper indicator calculations (ATR, RSI, etc.)
            # Using on_tick loses intrabar high/low which breaks ATR-based regime detection
            inject_complete_bar(symbol, bar)
            
            # Track previous position state
            if symbol in state and 'position' in state[symbol]:
                pos = state[symbol]['position']
                current_active = pos.get('active', False)
                
                # Capture exit reason while position is still active or just closed
                if current_active or (not current_active and prev_position_active):
                    # Check state for last_exit_reason (persists after position reset)
                    if 'last_exit_reason' in state[symbol]:
                        last_exit_reason = state[symbol]['last_exit_reason']
                
                # Capture confidence and regime when position opens
                if current_active and not prev_position_active:
                    # Position just opened - save the confidence and regime
                    entry_time = pos.get('entry_time', timestamp)
                    entry_time_key = str(entry_time)
                    confidence = state[symbol].get('entry_rl_confidence', 0.5)
                    # Convert to percentage
                    if confidence <= 1.0:
                        confidence = confidence * 100
                    trade_confidences[entry_time_key] = confidence
                    
                    # Track regime at entry
                    regime = state[symbol].get('current_regime', 'UNKNOWN')
                    trade_confidences[f"{entry_time_key}_regime"] = regime
                
                prev_position_active = current_active
                
                # Update backtest engine with current position from bot state
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
                    
                # If bot closed position (active=False), close it in backtest engine too
                elif not pos.get('active') and engine.current_position is not None:
                    exit_price = bar['close']  # Use bar close price for exit
                    exit_time = timestamp
                    # Use the last captured exit reason
                    engine._close_position(exit_time, exit_price, last_exit_reason)
                    last_exit_reason = 'bot_exit'  # Reset for next trade
        
        # Ensure final progress is shown
        print()  # New line after progress
        
    # Run backtest with integrated strategy
    results = engine.run_with_strategy(capitulation_strategy_backtest)
    
    # Get trades from engine metrics and add to reporter
    if hasattr(engine, 'metrics') and hasattr(engine.metrics, 'trades'):
        for trade in engine.metrics.trades:
            # Get RL confidence and regime from tracked data
            entry_time_key = str(trade.entry_time)
            confidence = trade_confidences.get(entry_time_key, 50)  # Default to 50% if not found
            regime = trade_confidences.get(f"{entry_time_key}_regime", "")  # Get regime if tracked
            
            # Convert Trade dataclass to dict for reporter
            trade_dict = {
                'side': trade.side,
                'quantity': trade.quantity,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'pnl': trade.pnl,
                'exit_reason': trade.exit_reason,
                'duration_minutes': trade.duration_minutes,
                'confidence': confidence,
                'regime': regime  # Add regime info
            }
            reporter.record_trade(trade_dict)
    
    # Update reporter totals from results
    if results:
        reporter.total_bars = total_bars
    
    # Print clean summary
    reporter.print_summary()
    
    # Save RL experiences at the end
    print("Saving RL experiences...")
    experience_path = f"experiences/{symbol}/signal_experience.json"
    if rl_brain is not None and hasattr(rl_brain, 'save_experience'):
        rl_brain.save_experience()
        final_experience_count = len(rl_brain.experiences)
        new_experiences = final_experience_count - initial_experience_count
        print(f"[OK] Signal RL experiences saved to {experience_path}")
        print(f"   Total experiences: {final_experience_count}")
        print(f"   New experiences this backtest: {new_experiences}")
    else:
        print("⚠️  No RL brain to save")
    
    # Return results
    return results


def main():
    """Main entry point for development backtesting"""
    args = parse_arguments()
    
    # Load configuration early to get account_size for reporter
    bot_config = load_config(backtest_mode=True)
    
    # BACKTEST-SPECIFIC OVERRIDES - These are hardcoded defaults for backtesting
    # They do NOT affect live trading in any way
    bot_config.account_size = 50000.0  # Standard backtest account size
    bot_config.max_contracts = 1  # Single contract for backtesting (no position sizing)
    bot_config.daily_loss_limit = 1000.0  # Standard daily loss limit for testing
    bot_config.shadow_mode = False  # Backtesting always executes simulated trades
    
    # Setup logging - suppress verbose output for clean backtest display
    config_dict = {'log_directory': os.path.join(PROJECT_ROOT, 'logs')}
    logger = setup_logging(config_dict)
    
    # Set log level - use WARNING to suppress INFO logs during backtest
    if args.log_level == 'INFO':
        # Override INFO to WARNING for cleaner output
        log_level = logging.WARNING
    else:
        log_level = getattr(logging, args.log_level)
    
    # Suppress unnecessary warnings for clean backtest output
    import warnings
    warnings.filterwarnings('ignore')
    
    # Suppress specific loggers completely
    logging.getLogger('root').setLevel(logging.CRITICAL)
    logging.getLogger('backtesting').setLevel(logging.CRITICAL)
    logging.getLogger('urllib3').setLevel(logging.CRITICAL)
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)
    
    logger.setLevel(log_level)
    
    # Suppress verbose logging from most loggers during backtest
    logging.getLogger('quotrading_engine').setLevel(logging.INFO)  # Need INFO level for RL messages
    logging.getLogger('backtesting').setLevel(logging.WARNING)
    logging.getLogger('capitulation_bot').setLevel(logging.ERROR)
    logging.getLogger('backtest').setLevel(logging.WARNING)
    logging.getLogger('regime_detection').setLevel(logging.WARNING)  # Suppress regime change spam
    logging.getLogger('signal_confidence').setLevel(logging.WARNING)  # Only show warnings and errors
    
    # Initialize clean reporter with account_size and max_contracts from config
    reporter = reset_reporter(
        starting_balance=bot_config.account_size,
        max_contracts=bot_config.max_contracts
    )
    
    # Create a custom filter to suppress signal spam and only track them
    class BacktestMessageFilter(logging.Filter):
        def filter(self, record):
            # Track RL signals for the reporter but suppress output
            msg = record.getMessage()
            if 'SIGNAL APPROVED' in msg:
                reporter.record_signal(approved=True)
                return False  # Suppress output
            elif 'Signal Declined' in msg:
                reporter.record_signal(approved=False)
                return False  # Suppress output
            elif 'Exploring' in msg:
                return False  # Suppress exploration messages
            elif 'LONG SIGNAL' in msg or 'SHORT SIGNAL' in msg:
                return False  # Suppress signal detection messages
            # Allow WARNING and above
            return record.levelno >= logging.WARNING
    
    # Add filter to quotrading_engine logger
    qte_logger = logging.getLogger('quotrading_engine')
    qte_logger.addFilter(BacktestMessageFilter())
    
    # Run backtest with clean output
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
