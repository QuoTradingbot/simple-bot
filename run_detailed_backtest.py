#!/usr/bin/env python3
"""
Enhanced Trade Logging Backtest
Shows detailed trade logic including:
- Entry signals and RL confidence
- Take profit levels and runners
- Stop loss adjustments (breakeven, trailing)
- Partial exits
- Exit reasons for wins and losses
"""

import sys
import os

# Set backtest mode BEFORE any other imports
os.environ['BOT_BACKTEST_MODE'] = 'true'

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
from datetime import datetime
import pytz

# Determine project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    """Run backtest with enhanced trade logging"""
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Setup detailed logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/detailed_trade_log.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('detailed_backtest')
    
    logger.info("="*80)
    logger.info("DETAILED TRADE LOGGING BACKTEST")
    logger.info("Showing: Entry/Exit Logic, Take Profits, Runners, Stop Management")
    logger.info("="*80)
    
    # Import the backtest function
    from run_full_backtest import run_full_backtest
    
    # Run single backtest
    results = run_full_backtest()
    
    if results:
        logger.info("")
        logger.info("="*80)
        logger.info("BACKTEST SUMMARY")
        logger.info("="*80)
        logger.info(f"Total P&L: ${results['total_pnl']:+,.2f}")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Win Rate: {results['win_rate']:.2f}%")
        logger.info(f"Average Win: ${results.get('avg_win', 0):+,.2f}")
        logger.info(f"Average Loss: ${results.get('avg_loss', 0):+,.2f}")
        
        # Parse the log file to extract trade details
        logger.info("")
        logger.info("="*80)
        logger.info("DETAILED TRADE ANALYSIS")
        logger.info("="*80)
        
        try:
            with open('logs/backtest_full.log', 'r') as f:
                log_lines = f.readlines()
            
            # Find trade entries and exits
            entries = []
            exits = []
            partials = []
            stop_adjustments = []
            
            for line in log_lines:
                if 'ENTERING' in line or 'LONG POSITION ENTERED' in line or 'SHORT POSITION ENTERED' in line:
                    entries.append(line.strip())
                elif 'EXITING' in line or 'Position closed' in line:
                    exits.append(line.strip())
                elif 'PARTIAL EXIT' in line:
                    partials.append(line.strip())
                elif 'breakeven' in line.lower() or 'trailing' in line.lower():
                    stop_adjustments.append(line.strip())
            
            logger.info(f"\nFound {len(entries)} trade entries")
            logger.info(f"Found {len(exits)} trade exits")
            logger.info(f"Found {len(partials)} partial exits")
            logger.info(f"Found {len(stop_adjustments)} stop adjustments")
            
        except Exception as e:
            logger.warning(f"Could not parse log file: {e}")
        
        logger.info("")
        logger.info("="*80)
        logger.info("See logs/detailed_trade_log.log for complete details")
        logger.info("See logs/backtest_full.log for full bot logging")
        logger.info("="*80)
        
        return results
    else:
        logger.error("Backtest failed!")
        return None


if __name__ == "__main__":
    try:
        results = main()
        if results:
            print("\n✓ Detailed backtest completed!")
            print("Check logs/detailed_trade_log.log for trade-by-trade breakdown")
        else:
            print("\n✗ Backtest failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nBacktest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
