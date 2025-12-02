#!/usr/bin/env python3
import os
os.environ['BOT_BACKTEST_MODE'] = 'true'

from datetime import datetime
import pytz
from config import BotConfiguration
from backtesting import BacktestConfig, BacktestEngine

# Setup logging to see what's happening
import logging
logging.basicConfig(level=logging.INFO)

# Setup ES config
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

# Backtest config
et = pytz.timezone('America/New_York')
start_date = datetime(2025, 9, 1, tzinfo=et)  # Start from September
end_date = datetime(2025, 9, 10, tzinfo=et)   # Just 10 days for debugging

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

from vwap_bounce_bot import initialize_state, on_tick, check_for_signals, check_exit_conditions, check_daily_reset, state, CONFIG

def vwap_strategy(bars_1min, bars_15min):
    symbol = "ES"
    initialize_state(symbol)
    et_tz = pytz.timezone('US/Eastern')
    
    print(f"Total bars to process: {len(bars_1min)}")
    
    for i, bar in enumerate(bars_1min):
        timestamp = bar['timestamp']
        price = bar['close']
        volume = bar['volume']
        timestamp_ms = int(timestamp.timestamp() * 1000)
        timestamp_et = timestamp.astimezone(et_tz)
        
        check_daily_reset(symbol, timestamp_et)
        on_tick(symbol, price, volume, timestamp_ms)
        
        # Print VWAP status every 100 bars
        if i % 100 == 0 and symbol in state:
            vwap = state[symbol].get('vwap')
            vwap_bands = state[symbol].get('vwap_bands', {})
            print(f"Bar {i}: {timestamp_et.strftime('%Y-%m-%d %H:%M')} - "
                  f"Price: {price:.2f}, VWAP: {vwap:.2f if vwap else 'None'}, "
                  f"Lower_2: {vwap_bands.get('lower_2', 'None')}")
        
        check_for_signals(symbol)
        check_exit_conditions(symbol)
    
    print(f"\nFinal state check:")
    if symbol in state:
        print(f"  VWAP: {state[symbol].get('vwap')}")
        print(f"  VWAP Bands: {state[symbol].get('vwap_bands')}")
        print(f"  Trend: {state[symbol].get('trend_direction')}")
        print(f"  Bars 1min: {len(state[symbol].get('bars_1min', []))}")

print("Starting backtest...")
results = engine.run_with_strategy(vwap_strategy)

print(f"\n{'='*60}")
print(f"RESULTS:")
print(f"Trades: {results['total_trades']}")
print(f"P&L: ${results['total_pnl']:,.2f}")
print(f"{'='*60}")
