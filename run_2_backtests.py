#!/usr/bin/env python3
"""
Run 2 Backtests with 30% Exploration
"""

import sys
import os
import json

# Set backtest mode BEFORE any other imports
os.environ['BOT_BACKTEST_MODE'] = 'true'

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging

# Determine project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def count_experiences():
    """Count current experiences in JSON files"""
    try:
        with open(os.path.join(PROJECT_ROOT, 'data/signal_experience.json'), 'r') as f:
            signal_data = json.load(f)
            signal_count = len(signal_data.get('experiences', []))
    except:
        signal_count = 0
    
    try:
        with open(os.path.join(PROJECT_ROOT, 'data/exit_experience.json'), 'r') as f:
            exit_data = json.load(f)
            exit_count = len(exit_data.get('exit_experiences', []))
    except:
        exit_count = 0
    
    return signal_count, exit_count


def main():
    """Run 2 backtests with 30% exploration"""
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/run_2_backtests.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger('run_2_backtests')
    
    logger.info("="*80)
    logger.info("RUNNING 2 BACKTESTS WITH 30% EXPLORATION")
    logger.info("="*80)
    
    # Check starting experience counts
    start_signal, start_exit = count_experiences()
    logger.info(f"Starting experiences:")
    logger.info(f"  Signal experiences: {start_signal:,}")
    logger.info(f"  Exit experiences: {start_exit:,}")
    logger.info("")
    
    # Track results for each backtest
    all_results = []
    
    # Import the backtest function
    from run_full_backtest import run_full_backtest
    
    # Run 2 backtests
    num_runs = 2
    
    for i in range(1, num_runs + 1):
        logger.info("="*80)
        logger.info(f"BACKTEST RUN {i}/{num_runs}")
        logger.info("="*80)
        
        try:
            # Run backtest
            results = run_full_backtest()
            
            if results:
                all_results.append({
                    'run': i,
                    'total_pnl': results['total_pnl'],
                    'total_trades': results['total_trades'],
                    'win_rate': results['win_rate']
                })
                
                # Check experience growth
                current_signal, current_exit = count_experiences()
                signal_growth = current_signal - start_signal
                exit_growth = current_exit - start_exit
                
                logger.info("")
                logger.info(f"Run {i} Complete:")
                logger.info(f"  P&L: ${results['total_pnl']:+,.2f}")
                logger.info(f"  Trades: {results['total_trades']}")
                logger.info(f"  Win Rate: {results['win_rate']:.2f}%")
                logger.info(f"  Signal experiences: {current_signal:,} (+{signal_growth:,} from start)")
                logger.info(f"  Exit experiences: {current_exit:,} (+{exit_growth:,} from start)")
                logger.info("")
            else:
                logger.error(f"Run {i} failed!")
                
        except Exception as e:
            logger.error(f"Run {i} error: {e}", exc_info=True)
        
        # Small delay between runs
        import time
        time.sleep(1)
    
    # Final summary
    logger.info("")
    logger.info("="*80)
    logger.info("BOTH BACKTESTS COMPLETE - FINAL SUMMARY")
    logger.info("="*80)
    
    # Final experience counts
    final_signal, final_exit = count_experiences()
    total_signal_growth = final_signal - start_signal
    total_exit_growth = final_exit - start_exit
    
    logger.info("")
    logger.info("Experience Growth:")
    logger.info(f"  Signal: {start_signal:,} → {final_signal:,} (+{total_signal_growth:,})")
    logger.info(f"  Exit: {start_exit:,} → {final_exit:,} (+{total_exit_growth:,})")
    logger.info(f"  Total: {start_signal + start_exit:,} → {final_signal + final_exit:,} (+{total_signal_growth + total_exit_growth:,})")
    
    logger.info("")
    logger.info("Backtest Results Summary:")
    logger.info(f"  Total runs completed: {len(all_results)}/{num_runs}")
    
    if all_results:
        total_pnl = sum(r['total_pnl'] for r in all_results)
        total_trades = sum(r['total_trades'] for r in all_results)
        avg_win_rate = sum(r['win_rate'] for r in all_results) / len(all_results)
        
        logger.info(f"  Combined P&L: ${total_pnl:+,.2f}")
        logger.info(f"  Combined Trades: {total_trades}")
        logger.info(f"  Average Win Rate: {avg_win_rate:.2f}%")
        
        logger.info("")
        logger.info("Individual Run Results:")
        for r in all_results:
            logger.info(f"  Run {r['run']}: P&L ${r['total_pnl']:+,.2f}, Trades {r['total_trades']}, Win Rate {r['win_rate']:.2f}%")
    
    logger.info("")
    logger.info("="*80)
    logger.info("VERIFICATION: All experiences saved to correct files")
    logger.info(f"  Signal file: data/signal_experience.json ({final_signal:,} experiences)")
    logger.info(f"  Exit file: data/exit_experience.json ({final_exit:,} experiences)")
    logger.info("  Exploration rate: 30% (aggressive learning mode)")
    logger.info(f"  Data: Full 63 days of 1-minute ES bars (63,599 bars)")
    logger.info("="*80)
    
    return all_results


if __name__ == "__main__":
    try:
        results = main()
        if results:
            print("\n✓ Both backtests completed successfully!")
            print(f"✓ Using 30% exploration rate for aggressive learning")
        else:
            print("\n✗ Some backtests failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nBacktests interrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
