#!/usr/bin/env python3
"""
Comprehensive iterative parameter optimization for ES VWAP strategy
Tests all important parameter combinations to find the best profitable settings
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
print("COMPREHENSIVE ES VWAP PARAMETER OPTIMIZATION")
print("="*80)
print("Testing all parameter combinations to find most profitable settings...")
print("="*80 + "\n")

all_results = []
test_num = 0

# Test different VWAP band widths with different filter combinations
vwap_bands = [0.5, 0.75, 1.0, 1.5, 2.0]
rsi_configs = [
    (False, None, None),  # RSI filter OFF
    (True, 20, 80),       # Very tight RSI
    (True, 25, 75),       # Tight RSI
    (True, 30, 70),       # Moderate RSI
    (True, 35, 65),       # Loose RSI
]
vwap_dir_filters = [False, True]
risk_rewards = [1.5, 2.0, 2.5, 3.0]

# Priority: Test looser configurations first (more likely to generate trades)
for vwap_band in vwap_bands:
    for use_rsi, rsi_os, rsi_ob in rsi_configs:
        for vwap_dir in vwap_dir_filters:
            for rr in risk_rewards:
                test_num += 1
                
                params = {
                    'vwap_std_dev_2': vwap_band,
                    'use_rsi_filter': use_rsi,
                    'use_vwap_direction_filter': vwap_dir,
                    'use_trend_filter': False,
                    'use_volume_filter': False,
                    'risk_reward_ratio': rr
                }
                
                if use_rsi:
                    params['rsi_oversold'] = rsi_os
                    params['rsi_overbought'] = rsi_ob
                
                desc = f"VWAP:{vwap_band}, RSI:{'OFF' if not use_rsi else f'{rsi_os}/{rsi_ob}'}, VWAPDir:{'ON' if vwap_dir else 'OFF'}, RR:{rr}"
                
                print(f"[{test_num}/80] Testing: {desc}")
                
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
print("OPTIMIZATION COMPLETE - ANALYZING RESULTS")
print("="*80)

# Filter to only profitable configurations
profitable = [(d, p, r) for d, p, r in all_results if r['total_pnl'] > 0 and r['total_trades'] > 0]
print(f"\nConfigurations tested: {len(all_results)}")
print(f"Configurations with trades: {len([r for d,p,r in all_results if r['total_trades'] > 0])}")
print(f"Profitable configurations: {len(profitable)}")

if len(profitable) > 0:
    print("\n" + "="*80)
    print("TOP 10 PROFITABLE CONFIGURATIONS (by Sharpe Ratio)")
    print("="*80)
    
    sorted_by_sharpe = sorted(profitable, key=lambda x: x[2]['sharpe_ratio'], reverse=True)[:10]
    
    for i, (desc, params, r) in enumerate(sorted_by_sharpe, 1):
        print(f"\n{i}. {desc}")
        print(f"   Sharpe: {r['sharpe_ratio']:.2f} | P&L: ${r['total_pnl']:,.0f} | "
              f"Trades: {r['total_trades']} | Win%: {r['win_rate']:.1f}% | "
              f"PF: {r['profit_factor']:.2f} | DD: ${r['max_drawdown_dollars']:,.0f}")
    
    print("\n" + "="*80)
    print("BEST CONFIGURATION (Highest Sharpe Ratio):")
    print("="*80)
    best = sorted_by_sharpe[0]
    print(f"\n{best[0]}\n")
    print(f"Performance Metrics:")
    print(f"  Total P&L: ${best[2]['total_pnl']:,.2f}")
    print(f"  Total Return: {best[2]['total_return']:.2f}%")
    print(f"  Win Rate: {best[2]['win_rate']:.1f}%")
    print(f"  Sharpe Ratio: {best[2]['sharpe_ratio']:.2f}")
    print(f"  Profit Factor: {best[2]['profit_factor']:.2f}")
    print(f"  Total Trades: {best[2]['total_trades']}")
    print(f"  Average Win: ${best[2]['average_win']:.2f}")
    print(f"  Average Loss: ${best[2]['average_loss']:.2f}")
    print(f"  Max Drawdown: ${best[2]['max_drawdown_dollars']:,.2f} ({best[2]['max_drawdown_percent']:.1f}%)")
    print(f"  Final Equity: ${best[2]['final_equity']:,.2f}")
    
    print(f"\nParameter Settings:")
    for key, value in best[1].items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*80)
    print("TOP 5 BY TOTAL PROFIT:")
    print("="*80)
    sorted_by_pnl = sorted(profitable, key=lambda x: x[2]['total_pnl'], reverse=True)[:5]
    for i, (desc, params, r) in enumerate(sorted_by_pnl, 1):
        print(f"{i}. P&L: ${r['total_pnl']:,.0f} | {desc}")
    
else:
    print("\n" + "="*80)
    print("NO PROFITABLE CONFIGURATIONS FOUND")
    print("="*80)
    
    # Show configurations that at least generated trades
    with_trades = [(d, p, r) for d, p, r in all_results if r['total_trades'] > 0]
    if len(with_trades) > 0:
        print(f"\nConfigurations that generated trades (but unprofitable):")
        for desc, params, r in with_trades[:10]:
            print(f"  {desc}")
            print(f"    P&L: ${r['total_pnl']:,.0f}, Trades: {r['total_trades']}, Win%: {r['win_rate']:.1f}%")
    else:
        print("\nNO configurations generated any trades!")
        print("The strategy may need fundamental changes or data quality issues exist.")

print("\n" + "="*80)
