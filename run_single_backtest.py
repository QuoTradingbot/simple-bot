#!/usr/bin/env python3
"""
Fixed backtest integration - properly sync bot state with engine
"""

import os
import sys
from datetime import datetime
import pytz
import logging

# Set backtest mode BEFORE importing
os.environ['BOT_BACKTEST_MODE'] = 'true'

from config import load_config
from backtesting import BacktestConfig, BacktestEngine, ReportGenerator
from vwap_bounce_bot import initialize_state, on_tick, check_for_signals, check_exit_conditions, check_daily_reset, state

# Setup logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Load config
bot_config = load_config(backtest_mode=True)
bot_config_dict = bot_config.to_dict()
symbol = bot_config_dict['instrument']

# Setup backtest - use the continuous data range
tz = pytz.timezone(bot_config.timezone)
end_date = datetime(2025, 10, 29, tzinfo=tz)
start_date = datetime(2025, 9, 1, tzinfo=tz)  # Full 2 months

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
last_position_active = False
position_entry_tracked = False

def vwap_strategy(bars_1min, bars_15min):
    """VWAP strategy with FIXED position tracking"""
    global last_position_active, position_entry_tracked
    
    # CRITICAL: Clear state fresh for each run
    if symbol in state:
        state.pop(symbol)
    initialize_state(symbol)
    
    last_position_active = False
    position_entry_tracked = False
    
    print(f"Processing {len(bars_1min)} bars from {bars_1min[0]['timestamp']} to {bars_1min[-1]['timestamp']}...")
    
    for i, bar in enumerate(bars_1min):
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
        
        # CRITICAL FIX: Properly sync position state with engine
        if symbol in state and 'position' in state[symbol]:
            pos = state[symbol]['position']
            is_active = pos.get('active', False)
            
            # Position just opened
            if is_active and not last_position_active:
                engine.current_position = {
                    'symbol': symbol,
                    'side': pos['side'],
                    'quantity': pos.get('quantity', 1),
                    'entry_price': pos['entry_price'],
                    'entry_time': pos.get('entry_time', timestamp),
                    'stop_price': pos.get('stop_price'),
                    'target_price': pos.get('target_price')
                }
                print(f"  [{timestamp.strftime('%m-%d %H:%M')}] {pos['side'].upper()} entry @ ${pos['entry_price']:.2f}")
                last_position_active = True
                position_entry_tracked = True
            
            # Position just closed
            elif not is_active and last_position_active:
                if engine.current_position is not None:
                    engine._close_position(timestamp, price, 'bot_exit')
                    print(f"  [{timestamp.strftime('%m-%d %H:%M')}] Position closed @ ${price:.2f}")
                last_position_active = False
                position_entry_tracked = False

# Run backtest
print("="*80)
print("RUNNING BACKTEST WITH FIXED INTEGRATION")
print("="*80)
print()

results = engine.run_with_strategy(vwap_strategy)

print()
print("="*80)
print("BACKTEST RESULTS")
print("="*80)
print(f"Total Trades: {results['total_trades']}")
print(f"Total P&L: ${results['total_pnl']:+,.2f}")
print(f"Total Return: {results['total_return']:+.2f}%")
print(f"Win Rate: {results['win_rate']:.2f}%")
print(f"Average Win: ${results['average_win']:+.2f}")
print(f"Average Loss: ${results['average_loss']:+.2f}")
print(f"Profit Factor: {results['profit_factor']:.2f}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
print(f"Max Drawdown: ${results['max_drawdown_dollars']:,.2f} ({results['max_drawdown_percent']:.2f}%)")
print(f"Final Equity: ${results['final_equity']:,.2f}")
print("="*80)

# Generate detailed report
report_gen = ReportGenerator(engine.metrics)
print()
print(report_gen.generate_trade_breakdown())

# Save results
import json
with open('backtest_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("\nResults saved to: backtest_results.json")
