import csv
from datetime import datetime
import pytz

# Read first 20 bars to see the structure
with open('data/historical_data/ES_1min.csv', 'r') as f:
    reader = csv.DictReader(f)
    bars = list(reader)[:20]
    
print('First 20 bars from CSV:')
print('=' * 100)
for i, bar in enumerate(bars):
    print(f"{i+1:3d}. {bar['timestamp']:23s} | O:{bar['open']:>8s} H:{bar['high']:>8s} L:{bar['low']:>8s} C:{bar['close']:>8s} V:{bar['volume']:>8s}")

# Check bar intervals
print('\n\nChecking time intervals between consecutive bars:')
print('=' * 100)
eastern = pytz.timezone('US/Eastern')
for i in range(1, min(10, len(bars))):
    dt1_utc = pytz.UTC.localize(datetime.fromisoformat(bars[i-1]['timestamp']))
    dt2_utc = pytz.UTC.localize(datetime.fromisoformat(bars[i]['timestamp']))
    dt1_et = dt1_utc.astimezone(eastern)
    dt2_et = dt2_utc.astimezone(eastern)
    diff_minutes = (dt2_utc - dt1_utc).total_seconds() / 60
    print(f"Bar {i} to {i+1}: {diff_minutes:.0f} min | {dt1_et.strftime('%Y-%m-%d %H:%M ET')} -> {dt2_et.strftime('%Y-%m-%d %H:%M ET')}")

# Read last 20 bars
with open('data/historical_data/ES_1min.csv', 'r') as f:
    reader = csv.DictReader(f)
    all_bars = list(reader)
    last_bars = all_bars[-20:]

print('\n\nLast 20 bars from CSV:')
print('=' * 100)
for i, bar in enumerate(last_bars):
    print(f"{len(all_bars)-19+i:6d}. {bar['timestamp']:23s} | O:{bar['open']:>8s} H:{bar['high']:>8s} L:{bar['low']:>8s} C:{bar['close']:>8s} V:{bar['volume']:>8s}")

print(f'\n\nTotal bars in CSV: {len(all_bars):,}')
print(f"First bar: {all_bars[0]['timestamp']}")
print(f"Last bar:  {all_bars[-1]['timestamp']}")

# Convert to Eastern time
first_utc = pytz.UTC.localize(datetime.fromisoformat(all_bars[0]['timestamp']))
last_utc = pytz.UTC.localize(datetime.fromisoformat(all_bars[-1]['timestamp']))
first_et = first_utc.astimezone(eastern)
last_et = last_utc.astimezone(eastern)
print(f"\nFirst bar (ET): {first_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"Last bar (ET):  {last_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
