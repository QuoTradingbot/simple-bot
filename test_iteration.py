#!/usr/bin/env python3
import os
import sys
os.environ['BOT_BACKTEST_MODE'] = 'true'

from datetime import datetime
import pytz
from config import BotConfiguration
from backtesting import BacktestConfig, BacktestEngine

def test_params(**params):
    """Test specific parameters"""
    # Setup ES config
    bot_config = BotConfiguration()
    bot_config.instrument = "ES"
    bot_config.tick_size = 0.25
    bot_config.tick_value = 12.50
    bot_config.backtest_mode = True
    
    # Apply test parameters
    for key, value in params.items():
        setattr(bot_config, key, value)
    
    # Backtest config
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
    
    print(f"\n{'='*60}")
    print(f"Parameters: {params}")
    print(f"Trades: {results['total_trades']}, Win Rate: {results['win_rate']:.1f}%, "
          f"P&L: ${results['total_pnl']:,.0f}, Sharpe: {results['sharpe_ratio']:.2f}")
    print(f"{'='*60}")
    
    return results

# Test configurations
print("\n" + "="*60)
print("STARTING ITERATIVE PARAMETER OPTIMIZATION FOR ES")
print("="*60)

results_log = []

# Test 1: Baseline with filters off
print("\n[TEST 1] Baseline - All filters OFF")
r1 = test_params(
    use_rsi_filter=False,
    use_trend_filter=False,
    use_vwap_direction_filter=False,
    use_volume_filter=False,
    vwap_std_dev_2=2.0
)
results_log.append(('Test 1: Baseline filters OFF, 2.0 std', r1))

# Test 2: Lower VWAP band
print("\n[TEST 2] Lower VWAP band to 1.5 std")
r2 = test_params(
    use_rsi_filter=False,
    use_trend_filter=False,
    use_vwap_direction_filter=False,
    use_volume_filter=False,
    vwap_std_dev_2=1.5
)
results_log.append(('Test 2: 1.5 std dev', r2))

# Test 3: Even lower VWAP band
print("\n[TEST 3] Even lower VWAP band to 1.0 std")
r3 = test_params(
    use_rsi_filter=False,
    use_trend_filter=False,
    use_vwap_direction_filter=False,
    use_volume_filter=False,
    vwap_std_dev_2=1.0
)
results_log.append(('Test 3: 1.0 std dev', r3))

# Test 4: Add VWAP direction filter back
print("\n[TEST 4] 1.5 std with VWAP direction filter")
r4 = test_params(
    use_rsi_filter=False,
    use_trend_filter=False,
    use_vwap_direction_filter=True,
    use_volume_filter=False,
    vwap_std_dev_2=1.5
)
results_log.append(('Test 4: 1.5 std + VWAP direction', r4))

# Test 5: Add RSI filter
print("\n[TEST 5] 1.5 std with RSI filter (28/72)")
r5 = test_params(
    use_rsi_filter=True,
    use_trend_filter=False,
    use_vwap_direction_filter=True,
    use_volume_filter=False,
    vwap_std_dev_2=1.5,
    rsi_oversold=28,
    rsi_overbought=72
)
results_log.append(('Test 5: 1.5 std + RSI 28/72', r5))

# Test 6: Wider RSI
print("\n[TEST 6] 1.5 std with wider RSI (25/75)")
r6 = test_params(
    use_rsi_filter=True,
    use_trend_filter=False,
    use_vwap_direction_filter=True,
    use_volume_filter=False,
    vwap_std_dev_2=1.5,
    rsi_oversold=25,
    rsi_overbought=75
)
results_log.append(('Test 6: 1.5 std + RSI 25/75', r6))

# Test 7: Higher risk/reward
print("\n[TEST 7] 1.5 std, best RSI, R:R 2.5:1")
best_rsi = (25, 75) if r6['total_pnl'] > r5['total_pnl'] else (28, 72)
r7 = test_params(
    use_rsi_filter=True,
    use_trend_filter=False,
    use_vwap_direction_filter=True,
    use_volume_filter=False,
    vwap_std_dev_2=1.5,
    rsi_oversold=best_rsi[0],
    rsi_overbought=best_rsi[1],
    risk_reward_ratio=2.5
)
results_log.append(('Test 7: Best + R:R 2.5', r7))

# Test 8: Even higher risk/reward
print("\n[TEST 8] 1.5 std, best RSI, R:R 3.0:1")
r8 = test_params(
    use_rsi_filter=True,
    use_trend_filter=False,
    use_vwap_direction_filter=True,
    use_volume_filter=False,
    vwap_std_dev_2=1.5,
    rsi_oversold=best_rsi[0],
    rsi_overbought=best_rsi[1],
    risk_reward_ratio=3.0
)
results_log.append(('Test 8: Best + R:R 3.0', r8))

# Print summary
print("\n" + "="*80)
print("OPTIMIZATION SUMMARY - RANKED BY SHARPE RATIO")
print("="*80)

sorted_results = sorted(results_log, key=lambda x: x[1]['sharpe_ratio'], reverse=True)
for i, (name, r) in enumerate(sorted_results, 1):
    print(f"{i}. {name}")
    print(f"   Sharpe: {r['sharpe_ratio']:.2f} | P&L: ${r['total_pnl']:,.0f} | "
          f"Trades: {r['total_trades']} | Win%: {r['win_rate']:.1f}% | "
          f"PF: {r['profit_factor']:.2f}")

print("\n" + "="*80)
print("BEST CONFIGURATION (by Sharpe Ratio):")
best = sorted_results[0]
print(f"  {best[0]}")
print(f"  Total P&L: ${best[1]['total_pnl']:,.2f}")
print(f"  Win Rate: {best[1]['win_rate']:.1f}%")
print(f"  Sharpe Ratio: {best[1]['sharpe_ratio']:.2f}")
print(f"  Profit Factor: {best[1]['profit_factor']:.2f}")
print(f"  Max Drawdown: ${best[1]['max_drawdown_dollars']:,.2f} ({best[1]['max_drawdown_percent']:.1f}%)")
print(f"  Total Trades: {best[1]['total_trades']}")
print("="*80)
