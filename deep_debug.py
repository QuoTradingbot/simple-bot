#!/usr/bin/env python3
import os
os.environ['BOT_BACKTEST_MODE'] = 'true'

from datetime import datetime
import pytz
from config import BotConfiguration
from backtesting import BacktestConfig, BacktestEngine
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

bot_config = BotConfiguration()
bot_config.instrument = "ES"
bot_config.tick_size = 0.25
bot_config.tick_value = 12.50
bot_config.backtest_mode = True
bot_config.use_rsi_filter = False
bot_config.use_trend_filter = False
bot_config.use_vwap_direction_filter = False
bot_config.use_volume_filter = False
bot_config.vwap_std_dev_2 = 1.0

et = pytz.timezone('America/New_York')
start_date = datetime(2025, 9, 3, tzinfo=et)
end_date = datetime(2025, 9, 4, tzinfo=et)  # Just 1 day for debugging

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

from vwap_bounce_bot import initialize_state, on_tick, check_for_signals, check_exit_conditions, check_daily_reset, state, check_long_signal_conditions, check_short_signal_conditions

touch_events = []

def vwap_strategy(bars_1min, bars_15min):
    symbol = "ES"
    initialize_state(symbol)
    et_tz = pytz.timezone('US/Eastern')
    
    print(f"\nProcessing {len(bars_1min)} bars...")
    print(f"Config: VWAP std={bot_config.vwap_std_dev_2}, All filters OFF\n")
    
    for i, bar in enumerate(bars_1min):
        timestamp = bar['timestamp']
        price = bar['close']
        volume = bar['volume']
        timestamp_ms = int(timestamp.timestamp() * 1000)
        timestamp_et = timestamp.astimezone(et_tz)
        
        check_daily_reset(symbol, timestamp_et)
        on_tick(symbol, price, volume, timestamp_ms)
        
        # Check VWAP band touches
        if symbol in state and 'vwap_bands' in state[symbol]:
            vwap_bands = state[symbol]['vwap_bands']
            if all(v is not None for v in vwap_bands.values()):
                lower_2 = vwap_bands['lower_2']
                upper_2 = vwap_bands['upper_2']
                
                # Check if price is touching bands
                if bar['low'] <= lower_2:
                    touch_events.append({
                        'time': timestamp_et,
                        'type': 'LOWER',
                        'price': price,
                        'band': lower_2,
                        'bar_low': bar['low']
                    })
                    print(f"[{i}] {timestamp_et.strftime('%H:%M')} - TOUCHED LOWER BAND: "
                          f"low={bar['low']:.2f} <= band={lower_2:.2f}, close={price:.2f}")
                    
                    # Check if next bar would trigger signal
                    if i + 1 < len(bars_1min):
                        next_bar = bars_1min[i+1]
                        if next_bar['close'] > lower_2:
                            print(f"    --> Next bar close={next_bar['close']:.2f} > band (SHOULD SIGNAL!)")
                        
                if bar['high'] >= upper_2:
                    touch_events.append({
                        'time': timestamp_et,
                        'type': 'UPPER',
                        'price': price,
                        'band': upper_2,
                        'bar_high': bar['high']
                    })
                    print(f"[{i}] {timestamp_et.strftime('%H:%M')} - TOUCHED UPPER BAND: "
                          f"high={bar['high']:.2f} >= band={upper_2:.2f}, close={price:.2f}")
                    
                    # Check if next bar would trigger signal
                    if i + 1 < len(bars_1min):
                        next_bar = bars_1min[i+1]
                        if next_bar['close'] < upper_2:
                            print(f"    --> Next bar close={next_bar['close']:.2f} < band (SHOULD SIGNAL!)")
        
        check_for_signals(symbol)
        check_exit_conditions(symbol)

print("Starting deep debug backtest...")
results = engine.run_with_strategy(vwap_strategy)

print(f"\n" + "="*80)
print(f"DEEP DEBUG RESULTS")
print(f"="*80)
print(f"Band touch events detected: {len(touch_events)}")
print(f"Trades executed: {results['total_trades']}")

if len(touch_events) > 0:
    print(f"\nFirst 10 band touches:")
    for i, event in enumerate(touch_events[:10], 1):
        print(f"{i}. {event['time'].strftime('%Y-%m-%d %H:%M')} - {event['type']} band "
              f"at {event['band']:.2f}, price={event['price']:.2f}")

print(f"\n" + "="*80)
