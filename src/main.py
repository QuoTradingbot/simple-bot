#!/usr/bin/env python3
"""
Main entry point for VWAP Bounce Bot - Production Trading Only

For backtesting and development, use: python dev/run_backtest.py
"""

import argparse
import sys
import os
import logging
from dotenv import load_dotenv

# Determine project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file in project root
env_path = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(dotenv_path=env_path)

# Import bot modules
from config import load_config, log_config
from monitoring import setup_logging, HealthChecker, HealthCheckServer, MetricsCollector, AuditLogger


def parse_arguments():
    """Parse command-line arguments for production trading"""
    parser = argparse.ArgumentParser(
        description='VWAP Bounce Bot - Production Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in live trading mode
  python src/main.py
  
  # Run in dry-run mode (paper trading)
  python src/main.py --dry-run
  
  # For backtesting, use the dev environment:
  python dev/run_backtest.py --days 30
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Enable dry-run mode (paper trading, no real orders)'
    )
    
    # Configuration overrides
    parser.add_argument(
        '--symbol',
        type=str,
        help='Override trading symbol (default: from config)'
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
    
    return parser.parse_args()


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
    from quotrading_engine import main as bot_main, bot_status, CONFIG
    
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
    from quotrading_engine import main as bot_main
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
    """Main entry point for production trading"""
    args = parse_arguments()
    
    # Load configuration for production
    bot_config = load_config(environment=args.environment, backtest_mode=False)
    
    # Apply learned parameters if available (feature not yet implemented)
    # from config import apply_learned_parameters
    # apply_learned_parameters(bot_config)
    
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
    
    # Run live trading
    success = run_live_trading(args, bot_config)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
