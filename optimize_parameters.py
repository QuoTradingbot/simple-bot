#!/usr/bin/env python3
"""
Optimized parameter search for VWAP bot - finds best profitable settings
Tests multiple parameter combinations and saves the best one
"""

import os
import sys
from datetime import datetime
import pytz
import logging
import json
from itertools import product

# Set backtest mode BEFORE importing
os.environ['BOT_BACKTEST_MODE'] = 'true'

from config import load_config
from backtesting import BacktestConfig, BacktestEngine
from vwap_bounce_bot import initialize_state, on_tick, check_for_signals, check_exit_conditions, check_daily_reset, state

# Setup logging
logging.basicConfig(
    level=logging.ERROR,  # Only show errors
    format='%(levelname)s: %(message)s'
)

def run_backtest_with_params(param_dict, symbol, start_date, end_date):
    """Run a single backtest with given parameters"""
    
    # Load config and override with test parameters
    bot_config = load_config(backtest_mode=True)
    for key, value in param_dict.items():
        setattr(bot_config, key, value)
    bot_config_dict = bot_config.to_dict()
    
    # Setup backtest
    backtest_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_equity=50000.0,
        symbols=[symbol],
        data_path='./historical_data',
        use_tick_data=False,
        slippage_ticks=bot_config.slippage_ticks,
        commission_per_contract=bot_config.commission_per_contract
    )
    
    # Create engine
    engine = BacktestEngine(backtest_config, bot_config_dict)
    
    # Get ET timezone
    et = pytz.timezone('US/Eastern')
    
    # Track position state
    last_position_active = [False]  # Use list to modify in closure
    
    def vwap_strategy(bars_1min, bars_15min):
        """VWAP strategy with proper position tracking"""
        
        # Clear state fresh for each run
        if symbol in state:
            state.pop(symbol)
        initialize_state(symbol)
        
        last_position_active[0] = False
        
        for bar in bars_1min:
            timestamp = bar['timestamp']
            price = bar['close']
            volume = bar['volume']
            timestamp_ms = int(timestamp.timestamp() * 1000)
            
            # Reset daily counters
            timestamp_et = timestamp.astimezone(et)
            check_daily_reset(symbol, timestamp_et)
            
            # Process
            on_tick(symbol, price, volume, timestamp_ms)
            check_for_signals(symbol)
            check_exit_conditions(symbol)
            
            # Sync position state with engine
            if symbol in state and 'position' in state[symbol]:
                pos = state[symbol]['position']
                is_active = pos.get('active', False)
                
                # Position opened
                if is_active and not last_position_active[0]:
                    engine.current_position = {
                        'symbol': symbol,
                        'side': pos['side'],
                        'quantity': pos.get('quantity', 1),
                        'entry_price': pos['entry_price'],
                        'entry_time': pos.get('entry_time', timestamp),
                        'stop_price': pos.get('stop_price'),
                        'target_price': pos.get('target_price')
                    }
                    last_position_active[0] = True
                
                # Position closed
                elif not is_active and last_position_active[0]:
                    if engine.current_position is not None:
                        engine._close_position(timestamp, price, 'bot_exit')
                    last_position_active[0] = False
    
    # Run backtest
    try:
        results = engine.run_with_strategy(vwap_strategy)
        return results
    except Exception as e:
        print(f"ERROR in backtest: {e}")
        return None


def main():
    print("="*80)
    print("VWAP BOT PARAMETER OPTIMIZATION")
    print("="*80)
    print()
    
    # Date range - use all available data
    symbol = 'ES'
    tz = pytz.timezone('America/New_York')
    end_date = datetime(2025, 10, 29, tzinfo=tz)
    start_date = datetime(2025, 9, 1, tzinfo=tz)
    
    print(f"Testing Period: {start_date.date()} to {end_date.date()}")
    print(f"Symbol: {symbol}")
    print()
    
    # Define parameter grid - focus on most impactful parameters
    param_grid = {
        'vwap_std_dev_2': [1.5, 2.0, 2.5],  # Entry band
        'rsi_oversold': [20, 25, 30],  # Long threshold
        'rsi_overbought': [70, 75, 80],  # Short threshold
        'use_rsi_filter': [True, False],  # Filter on/off
        'risk_per_trade': [0.01, 0.012, 0.015],  # Position sizing
    }
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    combinations = list(product(*param_values))
    
    total = len(combinations)
    print(f"Testing {total} parameter combinations...")
    print()
    
    results = []
    best_pnl = -999999
    best_sharpe = -999
    best_params_pnl = None
    best_params_sharpe = None
    
    for i, combo in enumerate(combinations, 1):
        params = dict(zip(param_names, combo))
        
        print(f"[{i}/{total}] Testing: vwap={params['vwap_std_dev_2']}, rsi={params['rsi_oversold']}/{params['rsi_overbought']}, use_rsi={params['use_rsi_filter']}, risk={params['risk_per_trade']}")
        
        result = run_backtest_with_params(params, symbol, start_date, end_date)
        
        if result:
            results.append({
                'params': params,
                'metrics': result
            })
            
            print(f"  -> Trades: {result['total_trades']}, P&L: ${result['total_pnl']:+,.2f}, Sharpe: {result['sharpe_ratio']:.3f}, Win%: {result['win_rate']:.1f}%")
            
            # Track best by P&L
            if result['total_pnl'] > best_pnl:
                best_pnl = result['total_pnl']
                best_params_pnl = params.copy()
            
            # Track best by Sharpe
            if result['sharpe_ratio'] > best_sharpe:
                best_sharpe = result['sharpe_ratio']
                best_params_sharpe = params.copy()
        else:
            print(f"  -> FAILED")
        
        print()
    
    print("="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)
    print()
    
    # Sort results
    results_by_pnl = sorted(results, key=lambda x: x['metrics']['total_pnl'], reverse=True)
    results_by_sharpe = sorted(results, key=lambda x: x['metrics']['sharpe_ratio'], reverse=True)
    
    # Show top 5 by P&L
    print("TOP 5 MOST PROFITABLE (by P&L):")
    print("-"*80)
    for i, r in enumerate(results_by_pnl[:5], 1):
        m = r['metrics']
        p = r['params']
        print(f"\n#{i}: ${m['total_pnl']:+,.2f} P&L, {m['sharpe_ratio']:.3f} Sharpe, {m['win_rate']:.1f}% Win")
        print(f"    vwap_std_dev_2={p['vwap_std_dev_2']}, rsi={p['rsi_oversold']}/{p['rsi_overbought']}, ")
        print(f"    use_rsi_filter={p['use_rsi_filter']}, risk_per_trade={p['risk_per_trade']}")
        print(f"    Trades: {m['total_trades']}, Profit Factor: {m['profit_factor']:.2f}")
    
    print("\n" + "="*80)
    print("TOP 5 BEST RISK-ADJUSTED (by Sharpe Ratio):")
    print("-"*80)
    for i, r in enumerate(results_by_sharpe[:5], 1):
        m = r['metrics']
        p = r['params']
        print(f"\n#{i}: {m['sharpe_ratio']:.3f} Sharpe, ${m['total_pnl']:+,.2f} P&L, {m['win_rate']:.1f}% Win")
        print(f"    vwap_std_dev_2={p['vwap_std_dev_2']}, rsi={p['rsi_oversold']}/{p['rsi_overbought']}, ")
        print(f"    use_rsi_filter={p['use_rsi_filter']}, risk_per_trade={p['risk_per_trade']}")
        print(f"    Trades: {m['total_trades']}, Profit Factor: {m['profit_factor']:.2f}")
    
    # Save all results
    output = {
        'test_period': {
            'start': start_date.isoformat(),
            'end': end_date.isoformat(),
            'days': (end_date - start_date).days
        },
        'total_combinations': total,
        'successful_runs': len(results),
        'best_by_pnl': {
            'params': best_params_pnl,
            'metrics': results_by_pnl[0]['metrics'] if results_by_pnl else None
        },
        'best_by_sharpe': {
            'params': best_params_sharpe,
            'metrics': results_by_sharpe[0]['metrics'] if results_by_sharpe else None
        },
        'all_results_by_pnl': results_by_pnl[:10],  # Top 10
        'all_results_by_sharpe': results_by_sharpe[:10]  # Top 10
    }
    
    with open('optimization_results_final.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print("\n" + "="*80)
    print("BEST PARAMETERS TO USE:")
    print("="*80)
    if results_by_pnl:
        print("\nFor Maximum Profit (highest P&L):")
        print(f"  vwap_std_dev_2: {best_params_pnl['vwap_std_dev_2']}")
        print(f"  rsi_oversold: {best_params_pnl['rsi_oversold']}")
        print(f"  rsi_overbought: {best_params_pnl['rsi_overbought']}")
        print(f"  use_rsi_filter: {best_params_pnl['use_rsi_filter']}")
        print(f"  risk_per_trade: {best_params_pnl['risk_per_trade']}")
        print(f"  Expected P&L: ${results_by_pnl[0]['metrics']['total_pnl']:+,.2f}")
        
        print("\nFor Best Risk-Adjusted Returns (highest Sharpe):")
        print(f"  vwap_std_dev_2: {best_params_sharpe['vwap_std_dev_2']}")
        print(f"  rsi_oversold: {best_params_sharpe['rsi_oversold']}")
        print(f"  rsi_overbought: {best_params_sharpe['rsi_overbought']}")
        print(f"  use_rsi_filter: {best_params_sharpe['use_rsi_filter']}")
        print(f"  risk_per_trade: {best_params_sharpe['risk_per_trade']}")
        print(f"  Expected Sharpe: {results_by_sharpe[0]['metrics']['sharpe_ratio']:.3f}")
    
    print("\nResults saved to: optimization_results_final.json")
    print("="*80)


if __name__ == '__main__':
    main()
