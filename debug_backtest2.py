#!/usr/bin/env python3
import os
os.environ['BOT_BACKTEST_MODE'] = 'true'

from datetime import datetime
import pytz
from config import BotConfiguration
from backtesting import BacktestConfig, BacktestEngine
import logging
logging.basicConfig(level=logging.WARNING)

bot_config = BotConfiguration()
bot_config.instrument = "ES"
bot_config.tick_size = 0.25
bot_config.tick_value = 12.50
bot_config.backtest_mode = True
bot_config.use_rsi_filter = False
bot_config.use_trend_filter = False
bot_config.use_vwap_direction_filter = False
bot_config.use_volume_filter = False
bot_config.vwap_std_dev_2 = 1.5

et = pytz.timezone('America/New_York')
start_date = datetime(2025, 9, 3, tzinfo=et)
end_date = datetime(2025, 9, 5, tzinfo=et)

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

signals_generated = []

def vwap_strategy(bars_1min, bars_15min):
    symbol = "ES"
    initialize_state(symbol)
    et_tz = pytz.timezone('US/Eastern')
    
    print(f"Processing {len(bars_1min)} bars...")
    
    for i, bar in enumerate(bars_1min):
        timestamp = bar['timestamp']
        price = bar['close']
        volume = bar['volume']
        timestamp_ms = int(timestamp.timestamp() * 1000)
        timestamp_et = timestamp.astimezone(et_tz)
        
        check_daily_reset(symbol, timestamp_et)
        on_tick(symbol, price, volume, timestamp_ms)
        
        # Check if we have VWAP calculated
        if i == 200 and symbol in state:
            vwap = state[symbol].get('vwap')
            vwap_bands = state[symbol].get('vwap_bands', {})
            print(f"\nAfter 200 bars:")
            print(f"  VWAP: {vwap}")
            print(f"  Bands: {vwap_bands}")
            print(f"  Current price: {price}")
        
        # Before checking signals, inspect state
        prev_active = state.get(symbol, {}).get('position', {}).get('active', False)
        check_for_signals(symbol)
        check_exit_conditions(symbol)
        
        # Check if signal was generated
        if symbol in state:
            curr_active = state[symbol].get('position', {}).get('active', False)
            if curr_active and not prev_active:
                signals_generated.append({
                    'time': timestamp_et,
                    'price': price,
                    'side': state[symbol]['position']['side']
                })
                print(f"\n*** SIGNAL GENERATED at {timestamp_et}: {state[symbol]['position']['side']} at {price}")

print("Starting backtest...")
results = engine.run_with_strategy(vwap_strategy)

print(f"\n{'='*60}")
print(f"Signals generated: {len(signals_generated)}")
print(f"Trades: {results['total_trades']}")
print(f"P&L: ${results['total_pnl']:,.2f}")
print(f"{'='*60}")

if len(signals_generated) > 0:
    print("\nSignals:")
    for sig in signals_generated:
        print(f"  {sig['time']}: {sig['side']} at {sig['price']}")
