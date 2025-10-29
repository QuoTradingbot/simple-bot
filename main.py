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
import pytz

# Import bot modules
from config import load_config, log_config
from monitoring import setup_logging, HealthChecker, HealthCheckServer, MetricsCollector, AlertManager, AuditLogger
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
  
  # Run backtest for last 30 days
  python main.py --mode backtest --days 30
  
  # Run backtest with specific date range
  python main.py --mode backtest --start 2024-01-01 --end 2024-01-31
  
  # Run backtest with custom data path
  python main.py --mode backtest --days 7 --data-path ./data
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
        default='./historical_data',
        help='Path to historical data directory (default: ./historical_data)'
    )
    
    parser.add_argument(
        '--initial-equity',
        type=float,
        default=25000.0,
        help='Initial equity for backtesting (default: 25000)'
    )
    
    parser.add_argument(
        '--report',
        type=str,
        help='Save backtest report to specified file'
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
    
    return parser.parse_args()


def run_backtest(args, bot_config):
    """
    Run backtesting mode - completely independent of broker API.
    Uses historical data to replay market conditions and simulate trading.
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
    backtest_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_equity=args.initial_equity,
        symbols=[args.symbol] if args.symbol else [bot_config.instrument],
        data_path=args.data_path
    )
    
    logger.info(f"Backtest Configuration:")
    logger.info(f"  Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"  Initial Equity: ${backtest_config.initial_equity:,.2f}")
    logger.info(f"  Symbols: {', '.join(backtest_config.symbols)}")
    logger.info(f"  Data Path: {backtest_config.data_path}")
    
    # Create backtest engine
    bot_config_dict = bot_config.to_dict()
    engine = BacktestEngine(backtest_config, bot_config_dict)
    
    # Run backtest (placeholder strategy function)
    # In full integration, this would use the actual bot strategy
    def dummy_strategy():
        pass
        
    results = engine.run(dummy_strategy)
    
    # Generate report
    report_gen = ReportGenerator(engine.metrics)
    
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
    
    # Initialize alert manager
    alert_manager = AlertManager(config_dict)
    
    # Initialize audit logger
    audit_logger = AuditLogger()
    
    try:
        # Run the bot
        logger.info("Starting bot...")
        bot_main()
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        
    except Exception as e:
        logger.error(f"Bot error: {e}", exc_info=True)
        alert_manager.send_alert('critical', 'Bot Crash', str(e))
        
    finally:
        # Cleanup
        if health_server:
            health_server.stop()
            
        logger.info("Bot shutdown complete")
    
    return True


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Load configuration with backtest mode flag
    backtest_mode = (args.mode == 'backtest')
    bot_config = load_config(environment=args.environment, backtest_mode=backtest_mode)
    
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
    if args.mode == 'backtest':
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
