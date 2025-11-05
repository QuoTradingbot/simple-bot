#!/usr/bin/env python3
"""
Full Backtest Runner with Custom Configuration
Runs backtest with:
- No daily trade cap (bot decides based on signals)
- 50% confidence threshold
- Maximum 3 contracts
- Dynamic contract sizing based on confidence
- All RL/ML features enabled
"""

import sys
import os
import csv

# Set backtest mode BEFORE any other imports
os.environ['BOT_BACKTEST_MODE'] = 'true'

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import bot modules
from config import load_config, BotConfiguration
from backtesting import BacktestConfig, BacktestEngine, ReportGenerator

# Determine project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Constant for unlimited trades
UNLIMITED_TRADES = 999


def run_full_backtest():
    """Run full backtest with custom configuration"""
    
    # Import vwap_bounce_bot modules here (after env var is set)
    from vwap_bounce_bot import initialize_state, on_tick, check_for_signals, check_exit_conditions, check_daily_reset, state, inject_complete_bar
    from signal_confidence import SignalConfidenceRL
    from adaptive_exits import AdaptiveExitManager
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/backtest_full.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger('backtest')
    
    logger.info("="*80)
    logger.info("FULL BACKTEST - Custom Configuration")
    logger.info("="*80)
    logger.info("Configuration:")
    logger.info("  - No daily trade cap (bot decides)")
    logger.info("  - Confidence threshold: 50% (0.5)")
    logger.info("  - Max contracts: 3")
    logger.info("  - Dynamic contract sizing based on confidence")
    logger.info("  - All RL/ML features enabled")
    logger.info("="*80)
    
    # Load bot configuration in backtest mode
    bot_config = load_config(backtest_mode=True)
    
    # Apply custom settings as per requirements
    bot_config.max_contracts = 3  # Maximum of 3 contracts
    bot_config.max_trades_per_day = UNLIMITED_TRADES  # Remove cap - let bot decide based on signals
    bot_config.rl_confidence_threshold = 0.5  # 50% confidence threshold
    
    # Ensure RL is enabled for dynamic contract sizing
    bot_config.rl_enabled = True
    
    # Set contract sizing based on confidence (already in config, but make explicit)
    bot_config.rl_min_contracts = 1  # Low confidence
    bot_config.rl_medium_contracts = 2  # Medium confidence  
    bot_config.rl_max_contracts = 3  # High confidence
    
    logger.info(f"Bot configured:")
    logger.info(f"  Max Contracts: {bot_config.max_contracts}")
    logger.info(f"  Max Trades/Day: {bot_config.max_trades_per_day} (unlimited)")
    logger.info(f"  RL Enabled: {bot_config.rl_enabled}")
    logger.info(f"  RL Confidence Threshold: {bot_config.rl_confidence_threshold}")
    logger.info(f"  RL Contract Sizing: {bot_config.rl_min_contracts}/{bot_config.rl_medium_contracts}/{bot_config.rl_max_contracts}")
    
    # Determine date range from CSV data using csv module
    data_file = os.path.join(PROJECT_ROOT, 'data/historical_data/ES_1min.csv')
    
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        return None
    
    # Read first and last timestamp from CSV using csv.reader
    with open(data_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        first_row = next(reader)
        first_data = first_row[0]
        
        # Read last line efficiently
        for row in reader:
            last_data = row[0]
        
    start_date = datetime.fromisoformat(first_data)
    end_date = datetime.fromisoformat(last_data)
    
    # Make timezone-aware
    tz = pytz.timezone('America/New_York')
    if start_date.tzinfo is None:
        start_date = tz.localize(start_date)
    if end_date.tzinfo is None:
        end_date = tz.localize(end_date)
    
    logger.info("")
    logger.info(f"Data Range:")
    logger.info(f"  Start: {start_date}")
    logger.info(f"  End: {end_date}")
    logger.info(f"  Duration: {(end_date - start_date).days} days")
    
    # Count lines efficiently
    with open(data_file, 'r') as f:
        total_bars = sum(1 for line in f) - 1  # -1 for header
    
    logger.info(f"  Total bars: {total_bars:,}")
    
    # Create backtest configuration
    backtest_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_equity=50000.0,
        symbols=['ES'],
        data_path=os.path.join(PROJECT_ROOT, 'data/historical_data'),
        use_tick_data=False,  # Use 1-minute bars
        slippage_ticks=bot_config.slippage_ticks,
        commission_per_contract=bot_config.commission_per_contract
    )
    
    logger.info("")
    logger.info(f"Backtest Configuration:")
    logger.info(f"  Initial Equity: ${backtest_config.initial_equity:,.2f}")
    logger.info(f"  Symbol: ES")
    logger.info(f"  Data Mode: 1-minute bars")
    logger.info(f"  Slippage: {backtest_config.slippage_ticks} ticks")
    logger.info(f"  Commission: ${backtest_config.commission_per_contract} per contract")
    
    # Create backtest engine
    bot_config_dict = bot_config.to_dict()
    engine = BacktestEngine(backtest_config, bot_config_dict)
    
    # Initialize bot state and components
    logger.info("")
    logger.info("Initializing bot components...")
    
    # Initialize bot state
    symbol = 'ES'
    initialize_state(symbol)
    
    # Force initialize RL brain in backtest mode
    import vwap_bounce_bot
    
    # Initialize RL brain with experience file
    signal_exp_file = os.path.join(PROJECT_ROOT, "data/signal_experience.json")
    vwap_bounce_bot.rl_brain = SignalConfidenceRL(
        experience_file=signal_exp_file,
        backtest_mode=True
    )
    logger.info(f"✓ RL BRAIN INITIALIZED - {len(vwap_bounce_bot.rl_brain.experiences)} signal experiences loaded")
    
    # Initialize adaptive exit manager
    exit_exp_file = os.path.join(PROJECT_ROOT, "data/exit_experience.json")
    vwap_bounce_bot.adaptive_manager = AdaptiveExitManager(
        config=vwap_bounce_bot.CONFIG,
        experience_file=exit_exp_file
    )
    logger.info(f"✓ ADAPTIVE EXITS INITIALIZED - {len(vwap_bounce_bot.adaptive_manager.exit_experiences)} exit experiences loaded")
    
    # Track starting experience counts
    starting_signal_count = len(vwap_bounce_bot.rl_brain.experiences)
    starting_exit_count = len(vwap_bounce_bot.adaptive_manager.exit_experiences)
    
    # Create bot reference for RL tracking
    class BotRLReferences:
        def __init__(self):
            self.signal_rl = vwap_bounce_bot.rl_brain
            self.exit_manager = vwap_bounce_bot.adaptive_manager
    
    bot_ref = BotRLReferences()
    engine.set_bot_instance(bot_ref)
    
    # Get ET timezone for daily reset
    et = pytz.timezone('US/Eastern')
    
    logger.info("")
    logger.info("="*80)
    logger.info("STARTING BACKTEST EXECUTION")
    logger.info("="*80)
    
    def vwap_strategy_backtest(bars_1min, bars_15min=None):
        """
        VWAP Bounce strategy integrated with backtest engine.
        Processes historical data through real bot logic.
        """
        for bar in bars_1min:
            # Extract bar data
            timestamp = bar['timestamp']
            price = bar['close']
            volume = bar['volume']
            timestamp_ms = int(timestamp.timestamp() * 1000)
            
            # Check for new trading day (resets daily counters at 6 PM ET)
            timestamp_et = timestamp.astimezone(et)
            check_daily_reset(symbol, timestamp_et)
            
            # Inject complete OHLCV bar for ATR calculation
            inject_complete_bar(symbol, bar)
            
            # Process tick through actual bot logic
            on_tick(symbol, price, volume, timestamp_ms)
            
            # Check for entry signals
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
                    
                # If bot closed position, close it in backtest engine too
                elif not pos.get('active') and engine.current_position is not None:
                    exit_price = price
                    exit_time = timestamp
                    exit_reason = 'bot_exit'
                    engine._close_position(exit_time, exit_price, exit_reason)
    
    # Run backtest
    results = engine.run_with_strategy(vwap_strategy_backtest)
    
    # Generate report
    report_gen = ReportGenerator(engine.metrics)
    
    # Print RL learning summary
    logger.info("")
    logger.info("="*80)
    logger.info("RL BRAIN LEARNING SUMMARY")
    logger.info("="*80)
    
    final_signal_count = len(vwap_bounce_bot.rl_brain.experiences)
    final_exit_count = len(vwap_bounce_bot.adaptive_manager.exit_experiences)
    
    signal_growth = final_signal_count - starting_signal_count
    exit_growth = final_exit_count - starting_exit_count
    
    logger.info(f"Signal Experiences: {starting_signal_count} → {final_signal_count} (+{signal_growth})")
    logger.info(f"Exit Experiences: {starting_exit_count} → {final_exit_count} (+{exit_growth})")
    logger.info(f"Total RL Growth: +{signal_growth + exit_growth} new experiences")
    logger.info("="*80)
    
    # Print detailed results
    logger.info("")
    logger.info("="*80)
    logger.info("BACKTEST RESULTS")
    logger.info("="*80)
    logger.info(report_gen.generate_trade_breakdown())
    
    # Save report
    report_file = os.path.join(PROJECT_ROOT, 'logs/backtest_full_report.txt')
    report_gen.save_report(report_file)
    logger.info("")
    logger.info(f"Full report saved to: {report_file}")
    
    logger.info("")
    logger.info("="*80)
    logger.info("BACKTEST COMPLETE")
    logger.info("="*80)
    
    return results


if __name__ == "__main__":
    try:
        results = run_full_backtest()
        if results:
            print("\n✓ Backtest completed successfully!")
            print(f"Total P&L: ${results['total_pnl']:+,.2f}")
            print(f"Total Trades: {results['total_trades']}")
            print(f"Win Rate: {results['win_rate']:.2f}%")
        else:
            print("\n✗ Backtest failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nBacktest interrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
