#!/usr/bin/env python3
"""
FINAL COMPREHENSIVE OPTIMIZATION
Tests ALL critical parameters to find absolute best profit and win rate.
Focused on high-impact combinations to complete in reasonable time.
"""

import os
import sys
from datetime import datetime
import pytz
import logging
import json
from itertools import product

os.environ['BOT_BACKTEST_MODE'] = 'true'

from config import load_config
from backtesting import BacktestConfig, BacktestEngine
from vwap_bounce_bot import initialize_state, on_tick, check_for_signals, check_exit_conditions, check_daily_reset, state

logging.basicConfig(level=logging.ERROR)

def run_backtest(params, symbol, start_date, end_date):
    """Run backtest with given parameters"""
    bot_config = load_config(backtest_mode=True)
    for key, value in params.items():
        setattr(bot_config, key, value)
    bot_config_dict = bot_config.to_dict()
    
    backtest_config = BacktestConfig(
        start_date=start_date, end_date=end_date, initial_equity=50000.0,
        symbols=[symbol], data_path='./historical_data', use_tick_data=False,
        slippage_ticks=bot_config.slippage_ticks,
        commission_per_contract=bot_config.commission_per_contract
    )
    
    engine = BacktestEngine(backtest_config, bot_config_dict)
    et = pytz.timezone('US/Eastern')
    last_active = [False]
    
    def strategy(bars_1min, bars_15min):
        if symbol in state:
            state.pop(symbol)
        initialize_state(symbol)
        last_active[0] = False
        
        for bar in bars_1min:
            timestamp, price, volume = bar['timestamp'], bar['close'], bar['volume']
            timestamp_ms = int(timestamp.timestamp() * 1000)
            check_daily_reset(symbol, timestamp.astimezone(et))
            on_tick(symbol, price, volume, timestamp_ms)
            check_for_signals(symbol)
            check_exit_conditions(symbol)
            
            if symbol in state and 'position' in state[symbol]:
                pos = state[symbol]['position']
                is_active = pos.get('active', False)
                
                if is_active and not last_active[0]:
                    engine.current_position = {
                        'symbol': symbol, 'side': pos['side'],
                        'quantity': pos.get('quantity', 1),
                        'entry_price': pos['entry_price'],
                        'entry_time': pos.get('entry_time', timestamp),
                        'stop_price': pos.get('stop_price'),
                        'target_price': pos.get('target_price')
                    }
                    last_active[0] = True
                elif not is_active and last_active[0]:
                    if engine.current_position:
                        engine._close_position(timestamp, price, 'bot_exit')
                    last_active[0] = False
    
    try:
        return engine.run_with_strategy(strategy)
    except:
        return None

def main():
    print("="*80)
    print("FINAL COMPREHENSIVE OPTIMIZATION - ALL PARAMETERS")
    print("="*80)
    
    symbol = 'ES'
    tz = pytz.timezone('America/New_York')
    # Use full data range
    end_date = datetime(2025, 10, 29, tzinfo=tz)
    start_date = datetime(2025, 8, 31, tzinfo=tz)  # Full ~60 days
    
    print(f"Period: {start_date.date()} to {end_date.date()} ({(end_date-start_date).days} days)")
    print()
    
    # Comprehensive parameter grid - most impactful combinations
    param_grid = {
        # Entry (proven critical)
        'vwap_std_dev_2': [1.5, 2.0],
        'rsi_oversold': [20, 25],
        'rsi_overbought': [70, 75],
        
        # Position sizing (critical for profit)
        'risk_per_trade': [0.01, 0.015, 0.02],
        'max_contracts': [2, 3],
        
        # Exit management (CRITICAL - never tested before!)
        'breakeven_enabled': [True, False],
        'trailing_stop_enabled': [True, False],
        'partial_exits_enabled': [True, False],
        
        # Advanced settings
        'risk_reward_ratio': [2.0, 2.5],
        'max_trades_per_day': [3, 5],
    }
    
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    combinations = list(product(*param_values))
    
    total = len(combinations)
    print(f"Testing {total} combinations including EXIT MANAGEMENT...")
    print()
    
    results = []
    best_pnl, best_wr, best_combo = -999999, 0, None
    
    for i, combo in enumerate(combinations, 1):
        params = dict(zip(param_names, combo))
        
        if i % 10 == 0:
            print(f"[{i}/{total}] Progress: {i*100//total}%")
        
        result = run_backtest(params, symbol, start_date, end_date)
        
        if result and result['total_trades'] > 0:
            results.append({'params': params, 'metrics': result})
            
            if result['total_pnl'] > best_pnl:
                best_pnl = result['total_pnl']
                best_combo = {'params': params, 'metrics': result}
            if result['win_rate'] > best_wr:
                best_wr = result['win_rate']
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)
    
    # Sort results
    by_pnl = sorted(results, key=lambda x: x['metrics']['total_pnl'], reverse=True)
    by_wr = sorted(results, key=lambda x: x['metrics']['win_rate'], reverse=True)
    by_sharpe = sorted(results, key=lambda x: x['metrics']['sharpe_ratio'], reverse=True)
    
    # Show top 5 by profit
    print("\n🏆 TOP 5 BY PROFIT:")
    print("-"*80)
    for i, r in enumerate(by_pnl[:5], 1):
        m, p = r['metrics'], r['params']
        print(f"\n#{i}: ${m['total_pnl']:+,.0f} | {m['win_rate']:.1f}% WR | {m['total_trades']} trades")
        print(f"  VWAP: {p['vwap_std_dev_2']}, RSI: {p['rsi_oversold']}/{p['rsi_overbought']}")
        print(f"  Risk: {p['risk_per_trade']}, Contracts: {p['max_contracts']}, MaxTrades: {p['max_trades_per_day']}")
        print(f"  Breakeven: {p['breakeven_enabled']}, Trailing: {p['trailing_stop_enabled']}, Partials: {p['partial_exits_enabled']}")
        print(f"  R:R: {p['risk_reward_ratio']}, PF: {m['profit_factor']:.2f}, Sharpe: {m['sharpe_ratio']:.2f}")
    
    # Show top 5 by win rate
    print("\n\n🎯 TOP 5 BY WIN RATE:")
    print("-"*80)
    for i, r in enumerate(by_wr[:5], 1):
        m, p = r['metrics'], r['params']
        print(f"\n#{i}: {m['win_rate']:.1f}% WR | ${m['total_pnl']:+,.0f} | {m['total_trades']} trades")
        print(f"  VWAP: {p['vwap_std_dev_2']}, RSI: {p['rsi_oversold']}/{p['rsi_overbought']}")
        print(f"  Risk: {p['risk_per_trade']}, Contracts: {p['max_contracts']}, MaxTrades: {p['max_trades_per_day']}")
        print(f"  Breakeven: {p['breakeven_enabled']}, Trailing: {p['trailing_stop_enabled']}, Partials: {p['partial_exits_enabled']}")
        print(f"  R:R: {p['risk_reward_ratio']}, PF: {m['profit_factor']:.2f}, Sharpe: {m['sharpe_ratio']:.2f}")
    
    # Save results
    output = {
        'test_period': {'start': start_date.isoformat(), 'end': end_date.isoformat(), 'days': (end_date-start_date).days},
        'total_combinations': total,
        'successful_runs': len(results),
        'best_by_profit': by_pnl[0] if by_pnl else None,
        'best_by_winrate': by_wr[0] if by_wr else None,
        'best_by_sharpe': by_sharpe[0] if by_sharpe else None,
        'top_10_profit': by_pnl[:10],
        'top_10_winrate': by_wr[:10],
        'all_results': results
    }
    
    with open('FINAL_RESULTS.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    # Show absolute best
    print("\n\n" + "="*80)
    print("🌟 ABSOLUTE BEST CONFIGURATION:")
    print("="*80)
    if by_pnl:
        best = by_pnl[0]
        m, p = best['metrics'], best['params']
        print(f"\n💰 PROFIT: ${m['total_pnl']:+,.2f}")
        print(f"📊 WIN RATE: {m['win_rate']:.1f}%")
        print(f"📈 TRADES: {m['total_trades']}")
        print(f"⚡ SHARPE: {m['sharpe_ratio']:.2f}")
        print(f"💪 PROFIT FACTOR: {m['profit_factor']:.2f}")
        print(f"📉 MAX DD: ${m['max_drawdown_dollars']:,.0f} ({m['max_drawdown_percent']:.1f}%)")
        print(f"\nPARAMETERS:")
        for k, v in p.items():
            print(f"  {k}: {v}")
    
    print(f"\nResults saved to: FINAL_RESULTS.json")
    print("="*80)

if __name__ == '__main__':
    main()
