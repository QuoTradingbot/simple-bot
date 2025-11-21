#!/usr/bin/env python3
"""
Main entry point for VWAP Bounce Bot
Supports both live trading and backtesting modes
"""

import argparse
import sys
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Union
import pytz
import pandas as pd
from dotenv import load_dotenv

# Determine project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
load_dotenv()

# Import bot modules
from config import load_config, log_config
from monitoring import setup_logging, HealthChecker, HealthCheckServer, MetricsCollector, AuditLogger
from backtesting import BacktestConfig, BacktestEngine, ReportGenerator


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='VWAP Bounce Bot - Mean Reversion Trading Strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in live trading mode (default)
  python main.py --mode live
  
  # Run in dry-run mode (paper trading)
  python main.py --mode live --dry-run
  
  # Run backtest for last 30 days (bar-by-bar with 1-minute bars)
  python main.py --mode backtest --days 30
  
  # Run backtest with specific date range
  python main.py --mode backtest --start 2024-01-01 --end 2024-01-31
  
  # Run backtest with tick-by-tick replay (requires tick data)
  python main.py --mode backtest --days 7 --use-tick-data
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['live', 'backtest'],
        default='live',
        help='Trading mode: live or backtest (default: live)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Enable dry-run mode (paper trading, no real orders)'
    )
    
    # Backtest-specific arguments
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
        default=None,  # Will be set to PROJECT_ROOT/data/historical_data if not specified
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
        '--strategy',
        choices=['vwap', 'ma-crossover'],
        default='vwap',
        help='Trading strategy: vwap (mean reversion) or ma-crossover (breakout) (default: vwap)'
    )
    
    # Configuration overrides
    parser.add_argument(
        '--symbol',
        type=str,
        help='Override trading symbol (default: MES)'
    )
    
    parser.add_argument(
        '--environment',
        type=str,
        choices=['development', 'staging', 'production'],
        help='Environment configuration to use'
    )
    
    parser.add_argument(
        '--health-check-port',
        type=int,
        default=8080,
        help='Port for health check HTTP server (default: 8080)'
    )
    
    parser.add_argument(
        '--no-health-check',
        action='store_true',
        help='Disable health check HTTP server'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--continuous-learn',
        action='store_true',
        help='Run continuous learning mode to optimize parameters through backtesting'
    )
    
    parser.add_argument(
        '--regime-learn',
        action='store_true',
        help='Run regime-aware learning (learns different strategies for different market conditions)'
    )
    
    parser.add_argument(
        '--learning-iterations',
        type=int,
        default=50,
        help='Number of learning iterations for continuous learning (default: 50)'
    )
    
    return parser.parse_args()


def initialize_rl_brains_for_backtest() -> Any:
    """
    Initialize RL brain (signal confidence) for backtest mode.
    This ensures experience files are loaded before the backtest runs.
    
    Note: Imports are done locally as these modules must be imported after 
    other bot components are initialized.
    
    Returns:
        Any: The initialized rl_brain instance
    """
    logger = logging.getLogger('main')
    import vwap_bounce_bot
    from signal_confidence import SignalConfidenceRL
    
    # Initialize RL brain with experience file using PROJECT_ROOT
    if vwap_bounce_bot.rl_brain is None:
        signal_exp_file = os.path.join(PROJECT_ROOT, "data/signal_experience.json")
        vwap_bounce_bot.rl_brain = SignalConfidenceRL(
            experience_file=signal_exp_file,
            backtest_mode=True
        )
        logger.info(f"✅ RL BRAIN INITIALIZED for backtest - {len(vwap_bounce_bot.rl_brain.experiences)} signal experiences loaded")
    
    return vwap_bounce_bot.rl_brain


def run_backtest_with_params(
    symbol: str, 
    days: int, 
    initial_equity: float, 
    params: Dict[str, Any], 
    return_bars: bool = False
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """
    Run a backtest with custom parameters and return results.
    This is a helper function for continuous learning.
    
    Args:
        symbol: Trading symbol (e.g., 'ES')
        days: Number of days to backtest
        initial_equity: Starting equity
        params: Dictionary of parameters to override in config
        return_bars: If True, return (results, bars) tuple for regime classification
        
    Returns:
        Dictionary with backtest results (pnl, win_rate, trades, etc.)
        OR (results_dict, bars_list) if return_bars=True
    """
    import config as cfg
    from backtesting import BacktestEngine, ReportGenerator
    from vwap_bounce_bot import on_tick, check_for_signals, check_exit_conditions, state
    
    logger = logging.getLogger('continuous_learner')
    
    # Create a copy of the config and override with params
    # Map learning parameter names to actual config variable names
    param_mapping = {
        # Core signal parameters
        'vwap_std_dev_1': 'vwap_std_dev_1',
        'vwap_std_dev_2': 'vwap_std_dev_2',
        'vwap_std_dev_3': 'vwap_std_dev_3',
        'rsi_period': 'rsi_period',
        'rsi_oversold': 'rsi_oversold',
        'rsi_overbought': 'rsi_overbought',
        
        # Risk management
        'stop_loss_atr_multiplier': 'stop_loss_atr_multiplier',
        'profit_target_atr_multiplier': 'profit_target_atr_multiplier',
        
        # Breakeven
        'breakeven_profit_threshold_ticks': 'breakeven_profit_threshold_ticks',
        'breakeven_stop_offset_ticks': 'breakeven_stop_offset_ticks',
        
        # Trailing
        'trailing_stop_distance_ticks': 'trailing_stop_distance_ticks',
        'trailing_stop_min_profit_ticks': 'trailing_stop_min_profit_ticks',
        
        # Time decay
        'time_decay_50_percent_tightening': 'time_decay_50_percent_tightening',
        'time_decay_75_percent_tightening': 'time_decay_75_percent_tightening',
        'time_decay_90_percent_tightening': 'time_decay_90_percent_tightening',
        
        # Partial exits
        'partial_exit_1_percentage': 'partial_exit_1_percentage',
        'partial_exit_1_r_multiple': 'partial_exit_1_r_multiple',
        'partial_exit_2_percentage': 'partial_exit_2_percentage',
        'partial_exit_2_r_multiple': 'partial_exit_2_r_multiple',
        
        # Filters
        'volume_multiplier_threshold': 'volume_multiplier_threshold',
        'trend_ema_period': 'trend_ema_period',
        'max_trades_per_day': 'max_trades_per_day',
    }
    
    original_values = {}
    for param_key, value in params.items():
        # Get config variable name
        config_var = param_mapping.get(param_key, param_key)
        
        if hasattr(cfg, config_var):
            original_values[config_var] = getattr(cfg, config_var)
            setattr(cfg, config_var, value)
            logger.debug(f"Set {config_var} = {value}")
    
    try:
        # Calculate date range
        tz = pytz.timezone("US/Eastern")  # Use UTC for CME futures
        end_date = datetime.now(tz)
        start_date = end_date - timedelta(days=days)
        
        # Create backtest config
        backtest_config = BacktestConfig(
            symbols=[symbol],
            start_date=start_date,
            end_date=end_date,
            initial_equity=initial_equity
        )
        
        # Get bot config - create instance and convert to dict
        bot_cfg_instance = cfg.load_config(backtest_mode=True)
        bot_config_dict = {k: getattr(bot_cfg_instance, k) for k in dir(bot_cfg_instance) if not k.startswith('_')}
        
        # Initialize backtest engine with proper parameters
        engine = BacktestEngine(config=backtest_config, bot_config=bot_config_dict)
        
        # Import bot functions and initialize state
        from vwap_bounce_bot import initialize_state, on_tick, check_for_signals, check_exit_conditions, state, check_daily_reset
        initialize_state(symbol)
        
        # Initialize RL brain (signal confidence) for backtest
        rl_brain = initialize_rl_brains_for_backtest()
        
        # Track starting experience count
        starting_signal_count = len(rl_brain.experiences) if rl_brain else 0
        
        et = tz  # Use same timezone
        
        # Store bars for regime classification if needed
        all_bars = []
        
        # Define strategy function
        def vwap_strategy_backtest(bars_1min, bars_15min=None):
            """Strategy function for backtest"""
            if return_bars:
                all_bars.extend(bars_1min)
            
            for bar in bars_1min:
                timestamp = bar['timestamp']
                timestamp_ms = int(timestamp.timestamp() * 1000)
                
                # Check for new trading day
                timestamp_et = timestamp.astimezone(et)
                check_daily_reset(symbol, timestamp_et)
                
                # CRITICAL FIX: Inject complete OHLCV bar to preserve high/low ranges for ATR calculation
                from vwap_bounce_bot import inject_complete_bar
                inject_complete_bar(symbol, bar)
                
                # Update engine with bot position
                if symbol in state and 'position' in state[symbol]:
                    pos = state[symbol]['position']
                    
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
                        engine._close_position(timestamp, bar['close'], 'bot_exit')
        
        # Run backtest
        results = engine.run_with_strategy(vwap_strategy_backtest)
        
        # Extract metrics using get_summary() method
        metrics = engine.metrics
        summary = metrics.get_summary()
        
        result_dict = {
            'total_pnl': summary['total_pnl'],
            'win_rate': summary['win_rate'],
            'profit_factor': summary['profit_factor'],
            'sharpe_ratio': summary['sharpe_ratio'],
            'max_drawdown': summary['max_drawdown_dollars'],
            'total_trades': summary['total_trades'],
            'winning_trades': len([t for t in metrics.trades if t.pnl > 0]),
            'losing_trades': len([t for t in metrics.trades if t.pnl < 0]),
            'avg_win': summary['average_win'],
            'avg_loss': summary['average_loss'],
            'largest_win': max([t.pnl for t in metrics.trades]) if metrics.trades else 0,
            'largest_loss': min([t.pnl for t in metrics.trades]) if metrics.trades else 0
        }
        
        logger.info(f"Backtest completed: {result_dict['total_trades']} trades, ${result_dict['total_pnl']:+,.2f}")
        
        # Save RL experiences after backtest completion
        try:
            if rl_brain is not None and hasattr(rl_brain, 'save_experience'):
                rl_brain.save_experience()
                logger.debug("Signal RL experiences saved")
        except Exception as save_error:
            logger.warning(f"Failed to save signal RL experiences: {save_error}")
        
        if return_bars:
            return result_dict, all_bars
        else:
            return result_dict
        
    except Exception as e:
        import traceback
        logger.error(f"Backtest failed with params {params}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        error_result = {
            'total_pnl': -99999,
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': -999,
            'max_drawdown': 99999,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0
        }
        
        if return_bars:
            return error_result, []
        else:
            return error_result
        
    finally:
        # Restore original config values
        for key, value in original_values.items():
            setattr(cfg, key, value)


def run_backtest(args: argparse.Namespace, bot_config: Any) -> Dict[str, Any]:
    """
    Run backtesting mode - completely independent of broker API.
    Uses historical data to replay market conditions and simulate trading.
    
    Args:
        args: Parsed command-line arguments from argparse
        bot_config: Bot configuration object
        
    Returns:
        Dictionary with backtest performance metrics
    """
    logger = logging.getLogger('main')
    logger.info("="*60)
    logger.info("STARTING BACKTEST MODE")
    logger.info("Backtesting does NOT use broker API - runs on historical data only")
    logger.info("="*60)
    
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
    # Use PROJECT_ROOT-based path if data_path not explicitly provided
    data_path = args.data_path if args.data_path else os.path.join(PROJECT_ROOT, "data/historical_data")
    
    backtest_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_equity=args.initial_equity,
        symbols=[args.symbol] if args.symbol else [bot_config.instrument],
        data_path=data_path,
        use_tick_data=args.use_tick_data  # Enable tick-by-tick if requested
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
    
    # CRITICAL FIX: Force reload of config and vwap_bounce_bot modules to get fresh values
    # This fixes subprocess caching issue where config changes weren't being picked up
    import sys
    import importlib
    if 'config' in sys.modules:
        importlib.reload(sys.modules['config'])
    if 'vwap_bounce_bot' in sys.modules:
        importlib.reload(sys.modules['vwap_bounce_bot'])
    
    # Integrate actual VWAP bot strategy for backtesting
    from vwap_bounce_bot import initialize_state, on_tick, check_for_signals, check_exit_conditions, check_daily_reset, state
    
    # Initialize bot state for backtesting
    symbol = bot_config_dict['instrument']
    initialize_state(symbol)
    
    # Initialize RL brain (signal confidence) for backtest
    rl_brain = initialize_rl_brains_for_backtest()
    
    # Create a simple object to hold RL brain reference for tracking
    class BotRLReferences:
        def __init__(self):
            self.signal_rl = rl_brain
    
    # Set bot instance for RL tracking
    bot_ref = BotRLReferences()
    engine.set_bot_instance(bot_ref)
    
    # Use UTC timezone for daily reset checks
    utc_tz = pytz.timezone("US/Eastern")
    
    def vwap_strategy_backtest(bars_1min: List[Dict[str, Any]], bars_15min: List[Dict[str, Any]]) -> None:
        """
        Actual VWAP Bounce strategy integrated with backtest engine.
        Processes historical data through the real bot logic.
        """
        for bar in bars_1min:
            # Extract bar data
            timestamp = bar['timestamp']
            price = bar['close']
            volume = bar['volume']
            timestamp_ms = int(timestamp.timestamp() * 1000)
            
            # Check for new trading day (resets daily counters at 23:00 UTC)
            timestamp_eastern = timestamp.astimezone(utc_tz)
            check_daily_reset(symbol, timestamp_eastern)
            
            # Process tick through actual bot logic
            on_tick(symbol, price, volume, timestamp_ms)
            
            # Check for entry signals after each bar
            check_for_signals(symbol)
            
            # Check for exit signals
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
        
    results = engine.run_with_strategy(vwap_strategy_backtest)
    
    # Save RL experiences after backtest completion
    print("\nSaving RL experiences...")
    try:
        if rl_brain is not None and hasattr(rl_brain, 'save_experience'):
            rl_brain.save_experience()
            print("Γ£ô Signal RL experiences saved")
    except Exception as e:
        print(f"Γ£ù Failed to save signal RL experiences: {e}")
    
    
    # Generate report
    report_gen = ReportGenerator(engine.metrics)
    
    # Track RL learning progress FIRST (before trade breakdown which might be long)
    print("\n" + "="*60)
    print("RL BRAIN LEARNING SUMMARY")
    print("="*60)
    
    # Read the experience files directly since module globals get cleared
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


def run_live_trading(args, bot_config):
    """Run live trading mode"""
    logger = logging.getLogger('main')
    logger.info("="*60)
    logger.info("STARTING LIVE TRADING MODE")
    logger.info("="*60)
    
    if args.dry_run:
        logger.info("DRY-RUN MODE ENABLED (Paper Trading)")
        bot_config.dry_run = True
    else:
        logger.warning("LIVE TRADING MODE - Real orders will be placed!")
        
        # Require confirmation for live trading
        if not os.getenv('CONFIRM_LIVE_TRADING'):
            logger.error("Live trading requires CONFIRM_LIVE_TRADING=1 environment variable")
            logger.error("Set this variable to confirm you understand the risks")
            return False
    
    # Check if multiple instruments are configured
    instruments = bot_config.instruments if hasattr(bot_config, 'instruments') else [bot_config.instrument]
    
    if len(instruments) > 1:
        logger.info(f"Multi-symbol mode detected: {', '.join(instruments)}")
        logger.info(f"Launching {len(instruments)} bot instances in parallel...")
        return run_multi_symbol_trading(args, bot_config, instruments)
    else:
        # Single symbol mode
        logger.info(f"Single-symbol mode: {instruments[0]}")
        return run_single_symbol_bot(args, bot_config, instruments[0])


def run_single_symbol_bot(args, bot_config, symbol):
    """Run bot for a single symbol"""
    logger = logging.getLogger('main')
    
    # Import bot modules
    from vwap_bounce_bot import main as bot_main, bot_status, CONFIG
    
    # Setup monitoring components
    config_dict = bot_config.to_dict()
    
    # Initialize health checker
    health_checker = HealthChecker(bot_status, config_dict)
    
    # Start health check server if enabled
    health_server = None
    if not args.no_health_check:
        health_server = HealthCheckServer(health_checker, port=args.health_check_port)
        health_server.start()
        logger.info(f"Health check endpoint: http://localhost:{args.health_check_port}/health")
    
    # Initialize metrics collector
    metrics_collector = MetricsCollector()
    
    # Initialize audit logger
    audit_logger = AuditLogger()
    
    try:
        # Run the bot with symbol
        logger.info(f"Starting bot for {symbol}...")
        bot_main(symbol_override=symbol)
        
    except KeyboardInterrupt:
        logger.info(f"[{symbol}] Received shutdown signal")
        
    except Exception as e:
        logger.error(f"[{symbol}] Bot error: {e}", exc_info=True)
        
    finally:
        # Cleanup
        if health_server:
            health_server.stop()
            
        logger.info(f"[{symbol}] Bot shutdown complete")
    
    return True


def _run_symbol_process(symbol):
    """Process target function for each symbol (must be at module level for Windows)"""
    # Re-import in subprocess
    from vwap_bounce_bot import main as bot_main
    import logging
    import multiprocessing
    
    process_logger = logging.getLogger(f'bot.{symbol}')
    process_logger.info(f"[{symbol}] Bot process starting (PID: {multiprocessing.current_process().pid})")
    
    try:
        # Run bot with this symbol
        bot_main(symbol_override=symbol)
    except KeyboardInterrupt:
        process_logger.info(f"[{symbol}] Shutdown signal received")
    except Exception as e:
        process_logger.error(f"[{symbol}] Bot error: {e}", exc_info=True)
    finally:
        process_logger.info(f"[{symbol}] Bot process exiting")


def run_multi_symbol_trading(args, bot_config, instruments):
    """Run multiple bot instances in parallel (one per symbol)"""
    from multiprocessing import Process
    logger = logging.getLogger('main')
    
    # Create a process for each symbol
    processes = []
    
    # Launch a process for each symbol
    for symbol in instruments:
        logger.info(f"Launching bot process for {symbol}...")
        p = Process(
            target=_run_symbol_process,
            args=(symbol,),
            name=f"Bot-{symbol}"
        )
        p.start()
        processes.append((symbol, p))
        logger.info(f"  ΓööΓöÇ {symbol} bot started (PID: {p.pid})")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"ALL {len(processes)} BOT INSTANCES RUNNING")
    logger.info(f"{'='*60}")
    for symbol, p in processes:
        logger.info(f"  [{symbol}] PID: {p.pid}")
    logger.info(f"{'='*60}\n")
    
    try:
        # Wait for all processes
        for symbol, p in processes:
            p.join()
            logger.info(f"[{symbol}] Bot process terminated (exit code: {p.exitcode})")
    
    except KeyboardInterrupt:
        logger.info("\nShutdown signal received - stopping all bot instances...")
        for symbol, p in processes:
            if p.is_alive():
                logger.info(f"  Terminating {symbol}...")
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    logger.warning(f"  Force killing {symbol}...")
                    p.kill()
    
    logger.info("All bot instances stopped")
    return True


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Load configuration with backtest mode flag
    backtest_mode = (args.mode == 'backtest' or args.continuous_learn)
    
    # Set environment variable so vwap_bounce_bot knows we're in backtest mode
    if backtest_mode:
        os.environ['BOT_BACKTEST_MODE'] = 'true'
    
    bot_config = load_config(environment=args.environment, backtest_mode=backtest_mode)
    
    # Apply learned parameters if available (unless we're learning new ones)
    if not args.continuous_learn:
        from config import apply_learned_parameters
        apply_learned_parameters(bot_config)
    
    # In backtest mode, we don't need API token at all
    # The backtest runs completely independently using historical data
    
    # Override symbol if specified
    if args.symbol:
        bot_config.instrument = args.symbol
    
    # Setup logging
    config_dict = bot_config.to_dict()
    config_dict['log_directory'] = './logs'
    logger = setup_logging(config_dict)
    
    # Set log level
    logger.setLevel(getattr(logging, args.log_level))
    
    # Log configuration
    log_config(bot_config, logger)
    
    # Run in selected mode
    if args.regime_learn:
        # REGIME-AWARE LEARNING MODE - Learn different strategies for different market conditions
        from regime_learner import RegimeAwareLearner
        
        logger.info("="*70)
        logger.info(" REGIME-AWARE LEARNING MODE ")
        logger.info("="*70)
        logger.info(f"Will run {args.learning_iterations} iterations")
        logger.info(f"Learning separate strategies for each market regime:")
        logger.info("  ΓÇó HIGH_VOL_TRENDING - High volatility + trending")
        logger.info("  ΓÇó HIGH_VOL_CHOPPY - High volatility + choppy")
        logger.info("  ΓÇó LOW_VOL_TRENDING - Low volatility + trending")
        logger.info("  ΓÇó LOW_VOL_RANGING - Low volatility + ranging")
        logger.info("  ΓÇó NORMAL - Normal conditions")
        logger.info("="*70 + "\n")
        
        # Create regime learner
        learner = RegimeAwareLearner(bot_config.__dict__)
        
        # Run learning iterations
        for i in range(args.learning_iterations):
            logger.info(f"\n{'='*70}")
            logger.info(f"ITERATION {i+1}/{args.learning_iterations}")
            logger.info(f"{'='*70}")
            
            # Run backtest and get bars for regime classification
            results, bars = run_backtest_with_params(
                symbol=bot_config.instrument,
                days=args.days if args.days else 30,
                initial_equity=args.initial_equity,
                params={},  # Will be set by learner
                return_bars=True  # Need bars for regime classification
            )
            
            if results is None:
                logger.warning(f"Backtest failed, skipping iteration")
                continue
            
            # Classify regime and record results
            regime = learner.classify_regime_from_backtest(bars)
            if regime:
                # Generate params for this regime
                params = learner.generate_parameter_set(regime)
                
                # Run backtest with regime-specific params
                results, bars = run_backtest_with_params(
                    symbol=bot_config.instrument,
                    days=args.days if args.days else 30,
                    initial_equity=args.initial_equity,
                    params=params,
                    return_bars=True
                )
                
                if results:
                    # Record results for this regime
                    learner.record_backtest(params, results, bars)
                    
                    # Save progress after each iteration
                    learner.save_regime_history()
        
        # Print final summary
        learner.print_regime_summary()
        
        logger.info("\n" + "="*70)
        logger.info(" REGIME-AWARE LEARNING COMPLETE!")
        logger.info("="*70)
        logger.info(f"Learned parameters saved to: regime_learning_history.json")
        logger.info(f"Total iterations: {learner.global_iteration}")
        logger.info("="*70 + "\n")
        
        sys.exit(0)
        
    elif args.continuous_learn:
        print(f"DEBUG: continuous_learn={args.continuous_learn}, iterations={args.learning_iterations}")
        # Continuous learning mode
        from continuous_learner import run_continuous_learning
        
        logger.info("="*60)
        logger.info("STARTING CONTINUOUS LEARNING MODE")
        logger.info(f"Will run {args.learning_iterations} learning iterations")
        logger.info("="*60)
        
        # Create backtest runner function for the learner
        def backtest_runner(params):
            return run_backtest_with_params(
                symbol=bot_config.instrument,
                days=args.days if args.days else 30,
                initial_equity=args.initial_equity,
                params=params
            )
        
        # Run continuous learning
        best_params, insights = run_continuous_learning(
            backtest_runner=backtest_runner,
            max_iterations=args.learning_iterations
        )
        
        logger.info("\n" + "="*60)
        logger.info("CONTINUOUS LEARNING COMPLETE")
        logger.info("="*60)
        logger.info(f"\nBest parameters found:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        
        logger.info(f"\nMarket insights:")
        for key, value in insights.items():
            logger.info(f"  {key}: {value}")
        
        sys.exit(0)
        
    elif args.mode == 'backtest':
        results = run_backtest(args, bot_config)
        
        # Exit with success/failure based on results
        if results and results.get('total_trades', 0) > 0:
            sys.exit(0)
        else:
            sys.exit(1)
            
    else:  # live mode
        success = run_live_trading(args, bot_config)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
