#!/usr/bin/env python3
"""
Run backtest and capture ACTUAL trade logs to prove functionality
"""

import sys
import os

# Set backtest mode BEFORE any other imports
os.environ['BOT_BACKTEST_MODE'] = 'true'

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging

# Setup file logging
os.makedirs('logs', exist_ok=True)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/PROOF_backtest.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('proof_backtest')

logger.info("="*80)
logger.info("RUNNING BACKTEST TO CAPTURE ACTUAL TRADE DETAILS")
logger.info("This will show REAL logs, not documentation")
logger.info("="*80)

# Run the backtest
from run_full_backtest import run_full_backtest

results = run_full_backtest()

if results:
    logger.info("")
    logger.info("="*80)
    logger.info("BACKTEST COMPLETE - Results saved to logs/PROOF_backtest.log")
    logger.info(f"Total P&L: ${results['total_pnl']:+,.2f}")
    logger.info(f"Total Trades: {results['total_trades']}")
    logger.info(f"Win Rate: {results['win_rate']:.2f}%")
    logger.info("="*80)
    print("\n✓ Check logs/PROOF_backtest.log for actual trade logs")
    print("✓ Check logs/backtest_full_report.txt for trade breakdown")
else:
    logger.error("Backtest failed")
    sys.exit(1)
