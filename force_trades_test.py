#!/usr/bin/env python3
import os
os.environ['BOT_BACKTEST_MODE'] = 'true'

from datetime import datetime
import pytz
from config import BotConfiguration
from backtesting import BacktestConfig, BacktestEngine
import logging
logging.basicConfig(level=logging.ERROR)

def test_config(vwap_std=1.0, rsi_filter=False, vwap_dir_filter=False):
    bot_config = BotConfiguration()
    bot_config.instrument = "ES"
    bot_config.tick_size = 0.25
    bot_config.tick_value = 12.50
    bot_config.backtest_mode = True
    bot_config.use_rsi_filter = rsi_filter
    bot_config.use_trend_filter = False
    bot_config.use_vwap_direction_filter = vwap_dir_filter
    bot_config.use_volume_filter = False
    bot_config.vwap_std_dev_2 = vwap_std
    
    et = pytz.timezone('America/New_York')
    start_date = datetime(2025, 9, 1, tzinfo=et)
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

print("Testing different VWAP band configurations on full ES dataset...")
print("="*70)

configs = [
    (0.5, False, False, "0.5 std, no filters"),
    (0.75, False, False, "0.75 std, no filters"),
    (1.0, False, False, "1.0 std, no filters"),
    (1.5, False, False, "1.5 std, no filters"),
    (1.0, False, True, "1.0 std, VWAP direction filter"),
    (0.75, False, True, "0.75 std, VWAP direction filter"),
]

results_list = []

for std, rsi, vwap_dir, desc in configs:
    print(f"\nTesting: {desc}")
    r = test_config(std, rsi, vwap_dir)
    results_list.append((desc, r))
    print(f"  Trades: {r['total_trades']}, Win%: {r['win_rate']:.1f}%, "
          f"P&L: ${r['total_pnl']:,.0f}, Sharpe: {r['sharpe_ratio']:.2f}")

print("\n" + "="*70)
print("RANKED BY SHARPE RATIO:")
print("="*70)

sorted_results = sorted(results_list, key=lambda x: x[1]['sharpe_ratio'], reverse=True)
for i, (desc, r) in enumerate(sorted_results, 1):
    print(f"{i}. {desc}")
    print(f"   Sharpe: {r['sharpe_ratio']:.2f} | P&L: ${r['total_pnl']:,.0f} | "
          f"Trades: {r['total_trades']} | Win%: {r['win_rate']:.1f}% | PF: {r['profit_factor']:.2f}")

print("\n" + "="*70)
print("BEST CONFIGURATION:")
best = sorted_results[0]
print(f"  {best[0]}")
print(f"  Total P&L: ${best[1]['total_pnl']:,.2f}")
print(f"  Win Rate: {best[1]['win_rate']:.1f}%")
print(f"  Sharpe Ratio: {best[1]['sharpe_ratio']:.2f}")
print(f"  Profit Factor: {best[1]['profit_factor']:.2f}")
print(f"  Trades: {best[1]['total_trades']}")
print(f"  Max DD: ${best[1]['max_drawdown_dollars']:,.2f} ({best[1]['max_drawdown_percent']:.1f}%)")
print("="*70)
