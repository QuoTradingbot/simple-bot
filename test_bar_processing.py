"""
Quick diagnostic backtest to verify bar processing
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datetime import datetime, timedelta
import pytz
import csv
from quotrading_engine import (
    calculate_vwap, calculate_rsi_1min, calculate_atr_1min,
    check_long_signal_conditions, check_short_signal_conditions,
    get_current_regime
)

# Load bars
bars = []
with open('data/historical_data/ES_1min.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        bars.append({
            'timestamp': datetime.fromisoformat(row['timestamp']),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': int(row['volume'])
        })

print(f"Loaded {len(bars):,} bars")
print(f"Date range: {bars[0]['timestamp']} to {bars[-1]['timestamp']}")

# Test on Nov 15-21 (last week of data)
eastern = pytz.timezone('US/Eastern')
start_date_utc = pytz.UTC.localize(datetime(2025, 11, 15, 0, 0, 0))
end_date_utc = pytz.UTC.localize(datetime(2025, 11, 22, 0, 0, 0))

test_bars = [b for b in bars if start_date_utc <= pytz.UTC.localize(b['timestamp']) < end_date_utc]
print(f"\nTest period: Nov 15-21")
print(f"Bars in period: {len(test_bars):,}")

# Process bars and check for signals
bar_buffer = []
signal_count = 0
long_signals = 0
short_signals = 0

print("\nProcessing bars bar-by-bar...")
print("=" * 100)

for i, bar in enumerate(test_bars):
    bar_buffer.append(bar)
    
    # Need at least 114 bars for indicators
    if len(bar_buffer) < 114:
        continue
    
    # Keep only last 500 bars
    if len(bar_buffer) > 500:
        bar_buffer.pop(0)
    
    # Convert to Eastern time
    bar_time_utc = pytz.UTC.localize(bar['timestamp'])
    bar_time_et = bar_time_utc.astimezone(eastern)
    
    # Check trading hours (6 PM - 4 PM ET entry window)
    hour = bar_time_et.hour
    if hour < 6 and hour >= 16:  # Before 6 AM or after 4 PM
        continue
    
    # Calculate indicators
    current_bar = bar_buffer[-1]
    prev_bar = bar_buffer[-2] if len(bar_buffer) > 1 else current_bar
    
    # Check for VWAP bounce signals
    try:
        # Check long signal
        if check_long_signal_conditions("ES", prev_bar, current_bar):
            long_signals += 1
            signal_count += 1
            
            # Get indicators
            vwap = calculate_vwap(bar_buffer)
            rsi = calculate_rsi_1min(bar_buffer)
            atr = calculate_atr_1min(bar_buffer)
            regime = get_current_regime("ES")
            
            vwap_dist = ((current_bar['close'] - vwap) / vwap) * 100
            
            print(f"\n🟢 LONG SIGNAL #{signal_count}")
            print(f"   Time: {bar_time_et.strftime('%Y-%m-%d %H:%M ET')} (bar {i+1}/{len(test_bars)})")
            print(f"   Price: {current_bar['close']:.2f} | VWAP: {vwap:.2f} (dist: {vwap_dist:+.2f}%)")
            print(f"   RSI: {rsi:.1f} | ATR: {atr:.2f} | Regime: {regime}")
        
        # Check short signal
        if check_short_signal_conditions("ES", prev_bar, current_bar):
            short_signals += 1
            signal_count += 1
            
            # Get indicators
            vwap = calculate_vwap(bar_buffer)
            rsi = calculate_rsi_1min(bar_buffer)
            atr = calculate_atr_1min(bar_buffer)
            regime = get_current_regime("ES")
            
            vwap_dist = ((current_bar['close'] - vwap) / vwap) * 100
            
            print(f"\n🔴 SHORT SIGNAL #{signal_count}")
            print(f"   Time: {bar_time_et.strftime('%Y-%m-%d %H:%M ET')} (bar {i+1}/{len(test_bars)})")
            print(f"   Price: {current_bar['close']:.2f} | VWAP: {vwap:.2f} (dist: {vwap_dist:+.2f}%)")
            print(f"   RSI: {rsi:.1f} | ATR: {atr:.2f} | Regime: {regime}")
            
    except Exception as e:
        # Skip bars with errors
        pass

print("\n" + "=" * 100)
print(f"\nSUMMARY:")
print(f"Total bars processed: {len(test_bars):,}")
print(f"Total signals detected: {signal_count}")
print(f"  Long signals: {long_signals}")
print(f"  Short signals: {short_signals}")
