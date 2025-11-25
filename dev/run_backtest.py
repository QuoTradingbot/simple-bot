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
from regime_detection import MIN_BARS_FOR_REGIME_DETECTION


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
    
    # Also make it available as vwap_bounce_bot for compatibility
    sys.modules['vwap_bounce_bot'] = bot_module
    
    # Load the module
    spec.loader.exec_module(bot_module)
    
    # Initialize RL brain with experience file and config values
    signal_exp_file = os.path.join(PROJECT_ROOT, "data/signal_experience.json")
    
    # BACKTEST MODE: Force exploration to enable learning
    # In live mode, exploration=0 means pure exploitation of learned patterns
    # In backtest mode, we want to explore to build experience
    backtest_exploration_rate = max(bot_config.rl_exploration_rate, 0.15)  # Minimum 15% exploration
    
    rl_brain = SignalConfidenceRL(
        experience_file=signal_exp_file,
        backtest_mode=True,
        exploration_rate=backtest_exploration_rate,
        min_exploration=0.10,  # Keep some exploration throughout backtest
        exploration_decay=bot_config.rl_exploration_decay
    )
    
    # Log how many experiences were loaded
    experience_count = len(rl_brain.experiences) if rl_brain else 0
    logger.info("=" * 60)
    logger.info("RL BRAIN INITIALIZATION")
    logger.info("=" * 60)
    logger.info(f"Experience file: {signal_exp_file}")
    logger.info(f"Experiences loaded: {experience_count:,}")
    logger.info(f"Exploration rate: {backtest_exploration_rate*100:.1f}%")
    if experience_count < 100:
        logger.warning("⚠️  Low experience count! For best results, use 1000+ experiences")
        logger.warning("   Make sure your local data/signal_experience.json has your full experience database")
    logger.info("=" * 60)
    
    # CRITICAL: Set the global rl_brain variable in the bot module
    # This is what get_ml_confidence() checks when deciding signals in backtest mode.
    # The hasattr check was removed because:
    # 1. We just created rl_brain above, so it's guaranteed to exist
    # 2. The module might not have rl_brain as an attribute initially (fresh import)
    # 3. We need to unconditionally set it to ensure backtest uses local RL brain
    # instead of trying to call cloud API which would fail in backtest mode
    bot_module.rl_brain = rl_brain
    
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
        # Extend end_date to include the full day (23:59:59)
        end_date = end_date.replace(hour=23, minute=59, second=59)
    elif args.days:
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
        symbol=args.symbol if args.symbol else bot_config.instrument
    )
    
    # Show config values being used for backtest (from config.json)
    print(f"Backtest Configuration (from data/config.json):")
    print(f"   Account Size: ${bot_config.account_size:,.2f}")
    print(f"   Max Contracts: {bot_config.max_contracts}")
    print(f"   RL Exploration Rate: {bot_config.rl_exploration_rate*100:.1f}%")
    print(f"   RL Min Exploration: {bot_config.rl_min_exploration_rate*100:.1f}%")
    print()
        
    # Create backtest configuration
    data_path = args.data_path if args.data_path else os.path.join(PROJECT_ROOT, "data/historical_data")
    
    backtest_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_equity=bot_config.account_size,
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
    
    # Suppress engine logger warnings for clean output
    if hasattr(engine, 'logger'):
        engine.logger.setLevel(logging.CRITICAL)
    
    # Initialize RL brain and bot module with config values
    rl_brain, bot_module = initialize_rl_brains_for_backtest(bot_config)
    
    # Track initial experience count to show how many were added during backtest
    initial_experience_count = len(rl_brain.experiences) if rl_brain else 0
    
    # Import bot functions from the loaded module
    initialize_state = bot_module.initialize_state
    on_tick = bot_module.on_tick
    inject_complete_bar = bot_module.inject_complete_bar  # For historical bar replay
    inject_complete_bar_15min = bot_module.inject_complete_bar_15min  # For 15min bars
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
    
    # Track previous position state to detect trade completions
    prev_position_active = False
    bars_processed = 0
    total_bars = 0
    
    # Track RL confidence for each trade
    trade_confidences = {}
    last_exit_reason = 'bot_exit'  # Track last exit reason
    prev_position_active = False
    
    def vwap_strategy_backtest(bars_1min: List[Dict[str, Any]], bars_15min: List[Dict[str, Any]]) -> None:
        """
        Actual VWAP Bounce strategy integrated with backtest engine.
        Processes historical data through the real bot logic.
        
        This executes:
        - Signal RL for confidence scoring
        - Pattern matching for signal detection
        - Regime detection for market adaptation (uses ATR from OHLC bars)
        - All trade management (stops, targets, breakeven, trailing)
        - UTC maintenance and flatten rules
        """
        nonlocal prev_position_active, bars_processed, total_bars, last_exit_reason
        total_bars = len(bars_1min)
        
        if len(bars_1min) == 0:
            print("WARNING: No 1-minute bars loaded!")
            return
        
        # Pre-load all 15-minute bars before processing 1-minute bars
        # This ensures indicators (RSI, MACD, trend) are ready
        print(f"Pre-loading {len(bars_15min)} 15-minute bars for indicators...")
        
        # CRITICAL: Verify we have enough 15-min bars for regime detection
        if len(bars_15min) < MIN_BARS_FOR_REGIME_DETECTION:
            print(f"⚠️  WARNING: Only {len(bars_15min)} 15-min bars loaded (need {MIN_BARS_FOR_REGIME_DETECTION} for regime detection)")
            print(f"   Regime detection will use fallback 'NORMAL' until {MIN_BARS_FOR_REGIME_DETECTION} bars accumulated")
            print(f"   This may affect accuracy of early trades")
            print(f"   Consider extending backtest date range to get more historical data")
        
        for bar_15min in bars_15min:
            inject_complete_bar_15min(symbol, bar_15min)
        print(f"15-minute bars loaded, indicators ready")
        print(f"Processing {len(bars_1min)} 1-minute bars...")
        
        for bar_idx, bar in enumerate(bars_1min):
            bars_processed = bar_idx + 1
            
            # Update progress every 100 bars
            if bars_processed % 100 == 0 or bars_processed == total_bars:
                reporter.update_progress(bars_processed, total_bars)
            
            # Extract bar data
            timestamp = bar['timestamp']
            
            # CRITICAL: Set backtest simulation time before processing bar
            # This ensures all time-based logic (trading hours, flatten mode, etc.) uses historical time
            bot_module.backtest_current_time = timestamp
            
            # Check for new trading day (resets daily counters following production rules)
            timestamp_eastern = timestamp.astimezone(eastern_tz)
            check_daily_reset(symbol, timestamp_eastern)
            
            # CRITICAL FIX: Inject complete OHLC bar instead of using on_tick
            # This preserves high/low data for accurate ATR calculation
            # inject_complete_bar handles regime detection, VWAP, signals, and exits
            inject_complete_bar(symbol, bar)
        
        # Ensure final progress is shown
        print()  # New line after progress
        
    # Run backtest with integrated strategy
    results = engine.run_with_strategy(vwap_strategy_backtest)
    
    # IMPORTANT: Get trades from bot's session_stats instead of engine metrics
    # The bot tracks all trades internally even if they complete quickly
    if symbol in state and 'session_stats' in state[symbol]:
        stats = state[symbol]['session_stats']
        win_count = stats.get('win_count', 0)
        loss_count = stats.get('loss_count', 0)
        total_pnl = stats.get('total_pnl', 0.0)
        
        print(f"\nBot Internal Stats:")
        print(f"  Trades: {win_count + loss_count} ({win_count}W/{loss_count}L)")
        print(f"  Total P&L: ${total_pnl:.2f}")
        print(f"  Daily trade count: {state[symbol].get('daily_trade_count', 0)}")
        
        # Update reporter with bot's stats
        # Note: Bot doesn't store individual trade details in session_stats['trades']
        # So we'll use the aggregate stats
        reporter.total_bars = total_bars
        
        # Create summary results from bot stats
        results = {
            'total_trades': win_count + loss_count,
            'total_pnl': total_pnl,
            'win_rate': (win_count / (win_count + loss_count) * 100) if (win_count + loss_count) > 0 else 0.0,
            'average_win': stats.get('largest_win', 0.0),  # Approximate
            'average_loss': stats.get('largest_loss', 0.0),  # Approximate
            'final_equity': bot_config.account_size + total_pnl,
            'total_return': (total_pnl / bot_config.account_size) * 100
        }
    else:
        # Fallback to engine results if bot stats not available
        # Get trades from engine metrics and add to reporter
        if hasattr(engine, 'metrics') and hasattr(engine.metrics, 'trades'):
            for trade in engine.metrics.trades:
                # Get RL confidence from tracked confidences
                entry_time_key = str(trade.entry_time)
                confidence = trade_confidences.get(entry_time_key, 50)  # Default to 50% if not found
                
                # Convert Trade dataclass to dict for reporter
                trade_dict = {
                    'side': trade.side,
                    'quantity': trade.quantity,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'pnl': trade.pnl,
                    'exit_reason': trade.exit_reason,  # This comes from the engine
                    'exit_time': trade.exit_time,
                    'duration_minutes': trade.duration_minutes,
                    'confidence': confidence
                }
                reporter.record_trade(trade_dict)
        
        # Update reporter totals from results
        if results:
            reporter.total_bars = total_bars
    
    # Print clean summary
    reporter.print_summary()
    
    # Save RL experiences at the end
    print("Saving RL experiences...")
    if rl_brain is not None and hasattr(rl_brain, 'save_experience'):
        rl_brain.save_experience()
        final_experience_count = len(rl_brain.experiences)
        new_experiences = final_experience_count - initial_experience_count
        print(f"[OK] Signal RL experiences saved to data/signal_experience.json")
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
    logging.getLogger('vwap_bot').setLevel(logging.ERROR)
    logging.getLogger('backtest').setLevel(logging.WARNING)
    logging.getLogger('regime_detection').setLevel(logging.WARNING)  # Suppress regime change spam
    logging.getLogger('signal_confidence').setLevel(logging.WARNING)  # Only show warnings and errors
    
    # Initialize clean reporter with account_size from config
    reporter = reset_reporter(starting_balance=bot_config.account_size)
    
    # Create a custom filter to allow only specific INFO messages through AND track signals
    class BacktestMessageFilter(logging.Filter):
        def filter(self, record):
            # Track RL signals for the reporter
            msg = record.getMessage()
            if 'RL APPROVED' in msg:
                reporter.record_signal(approved=True)
                return True
            elif 'RL REJECTED' in msg:
                reporter.record_signal(approved=False)
                return False  # Don't show rejections (too much spam)
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
