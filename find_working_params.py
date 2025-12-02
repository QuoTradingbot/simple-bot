#!/usr/bin/env python3
"""
Find working parameters by simplifying the strategy
Remove restrictive filters and test simpler configurations
"""
import os
os.environ['BOT_BACKTEST_MODE'] = 'true'

from datetime import datetime
import pytz
from config import BotConfiguration
from backtesting import BacktestConfig, BacktestEngine
import logging
logging.basicConfig(level=logging.ERROR)

def run_backtest(params_dict):
    """Run backtest with given parameters"""
    bot_config = BotConfiguration()
    bot_config.instrument = "ES"
    bot_config.tick_size = 0.25
    bot_config.tick_value = 12.50
    bot_config.backtest_mode = True
    
    # Apply parameters
    for key, value in params_dict.items():
        setattr(bot_config, key, value)
    
    et = pytz.timezone('America/New_York')
    start_date = datetime(2025, 8, 31, tzinfo=et)
    end_date = datetime(2025, 10, 29, tzinfo=et)
    
    backtest_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_equity=25000.0,
        symbols=["ES"],
        slippage_ticks=1.5,
        commission_per_contract=2.50,
        data_path="./historical_data"
    )
    
    bot_config_dict = bot_config.to_dict()
    engine = BacktestEngine(backtest_config, bot_config_dict)
    
    from vwap_bounce_bot import initialize_state, on_tick, check_for_signals, check_exit_conditions, check_daily_reset, state
    
    def vwap_strategy(bars_1min, bars_15min):
        symbol = "ES"
        initialize_state(symbol)
        et_tz = pytz.timezone('US/Eastern')
        
        for bar in bars_1min:
            timestamp = bar['timestamp']
            price = bar['close']
            volume = bar['volume']
            timestamp_ms = int(timestamp.timestamp() * 1000)
            timestamp_et = timestamp.astimezone(et_tz)
            
            check_daily_reset(symbol, timestamp_et)
            on_tick(symbol, price, volume, timestamp_ms)
            check_for_signals(symbol)
            check_exit_conditions(symbol)
    
    results = engine.run_with_strategy(vwap_strategy)
    return results

print("\n" + "="*80)
print("FINDING WORKING PARAMETERS - SIMPLIFIED STRATEGY")
print("="*80)
print("Removing restrictive filters to generate trades...")
print("="*80 + "\n")

all_results = []

# Test with ALL filters OFF (simplest configuration)
print("PHASE 1: Testing with ALL filters OFF")
print("-" * 80)

vwap_bands = [0.5, 0.75, 1.0, 1.25, 1.5]
risk_rewards = [1.5, 2.0, 2.5, 3.0]

test_num = 0
for vwap_band in vwap_bands:
    for rr in risk_rewards:
        test_num += 1
        
        params = {
            'vwap_std_dev_2': vwap_band,
            'use_rsi_filter': False,
            'use_vwap_direction_filter': False,
            'use_trend_filter': False,
            'use_volume_filter': False,
            'risk_reward_ratio': rr,
            'daily_loss_limit': 500.0,  # Increase to allow more trades
        }
        
        desc = f"VWAP:{vwap_band} std, R:R {rr}:1, All Filters OFF"
        
        print(f"[{test_num}/20] {desc}")
        
        try:
            results = run_backtest(params)
            all_results.append((desc, params, results))
            
            if results['total_trades'] > 0:
                print(f"         ✓ Trades: {results['total_trades']}, P&L: ${results['total_pnl']:,.0f}, "
                      f"Win%: {results['win_rate']:.1f}%, Sharpe: {results['sharpe_ratio']:.2f}")
            else:
                print(f"         ✗ No trades")
                
        except Exception as e:
            print(f"         ERROR: {e}")
            continue

print("\n" + "="*80)
print("RESULTS ANALYSIS")
print("="*80)

with_trades = [(d, p, r) for d, p, r in all_results if r['total_trades'] > 0]
profitable = [(d, p, r) for d, p, r in with_trades if r['total_pnl'] > 0]

print(f"\nConfigurations tested: {len(all_results)}")
print(f"Configurations with trades: {len(with_trades)}")
print(f"Profitable configurations: {len(profitable)}")

if len(profitable) > 0:
    print("\n" + "="*80)
    print("PROFITABLE CONFIGURATIONS (by Sharpe Ratio)")
    print("="*80)
    
    sorted_results = sorted(profitable, key=lambda x: x[2]['sharpe_ratio'], reverse=True)
    
    for i, (desc, params, r) in enumerate(sorted_results, 1):
        print(f"\n{i}. {desc}")
        print(f"   Sharpe: {r['sharpe_ratio']:.2f} | P&L: ${r['total_pnl']:,.0f} | "
              f"Return: {r['total_return']:.1f}% | Trades: {r['total_trades']} | "
              f"Win%: {r['win_rate']:.1f}%")
        print(f"   PF: {r['profit_factor']:.2f} | Avg Win: ${r['average_win']:.0f} | "
              f"Avg Loss: ${r['average_loss']:.0f}")
        print(f"   Max DD: ${r['max_drawdown_dollars']:,.0f} ({r['max_drawdown_percent']:.1f}%)")
    
    print("\n" + "="*80)
    print("⭐ BEST CONFIGURATION (Highest Sharpe Ratio)")
    print("="*80)
    best = sorted_results[0]
    print(f"\nConfiguration: {best[0]}\n")
    print("Performance:")
    print(f"  Total Return: {best[2]['total_return']:+.2f}%")
    print(f"  Total P&L: ${best[2]['total_pnl']:+,.2f}")
    print(f"  Sharpe Ratio: {best[2]['sharpe_ratio']:.2f}")
    print(f"  Profit Factor: {best[2]['profit_factor']:.2f}")
    print(f"  Win Rate: {best[2]['win_rate']:.1f}%")
    print(f"  Total Trades: {best[2]['total_trades']}")
    print(f"  Average Win: ${best[2]['average_win']:.2f}")
    print(f"  Average Loss: ${best[2]['average_loss']:.2f}")
    print(f"  Max Drawdown: ${best[2]['max_drawdown_dollars']:,.2f} ({best[2]['max_drawdown_percent']:.1f}%)")
    print(f"  Final Equity: ${best[2]['final_equity']:,.2f}")
    
    print("\nOptimal Parameters:")
    for key, value in best[1].items():
        print(f"  {key}: {value}")
    
    # Save best config
    print("\n" + "="*80)
    print("Saving best configuration to best_params.txt...")
    with open('best_params.txt', 'w') as f:
        f.write("# Best ES VWAP Parameters Found\n\n")
        f.write(f"Configuration: {best[0]}\n\n")
        f.write("Parameters:\n")
        for key, value in best[1].items():
            f.write(f"{key} = {value}\n")
        f.write(f"\nPerformance:\n")
        f.write(f"Total Return: {best[2]['total_return']:+.2f}%\n")
        f.write(f"Total P&L: ${best[2]['total_pnl']:+,.2f}\n")
        f.write(f"Sharpe Ratio: {best[2]['sharpe_ratio']:.2f}\n")
        f.write(f"Profit Factor: {best[2]['profit_factor']:.2f}\n")
        f.write(f"Win Rate: {best[2]['win_rate']:.1f}%\n")
        f.write(f"Total Trades: {best[2]['total_trades']}\n")
    print("Saved to best_params.txt")
    
elif len(with_trades) > 0:
    print("\n" + "="*80)
    print("CONFIGURATIONS WITH TRADES (but unprofitable)")
    print("="*80)
    
    sorted_results = sorted(with_trades, key=lambda x: x[2]['total_pnl'], reverse=True)
    
    for i, (desc, params, r) in enumerate(sorted_results, 1):
        print(f"\n{i}. {desc}")
        print(f"   P&L: ${r['total_pnl']:,.0f} | Trades: {r['total_trades']} | "
              f"Win%: {r['win_rate']:.1f}% | Sharpe: {r['sharpe_ratio']:.2f}")
else:
    print("\nNO TRADES GENERATED - Strategy needs fundamental redesign")

print("\n" + "="*80)
