import pandas as pd
from datetime import timedelta
from pathlib import Path
import pytz

# Get project root (parent of scripts folder)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / 'data' / 'historical_data' / 'ES_1min.csv'

# Load data
df = pd.read_csv(DATA_FILE)
# Parse timestamps with mixed formats, localize naive timestamps to UTC
df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"Total bars: {len(df)}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")

# Convert to ET for proper weekend/market hours checking
# ES futures trade: Sun 6pm ET - Fri 5pm ET (with daily 5-6pm maintenance)
et_tz = pytz.timezone('America/New_York')
df['timestamp_et'] = df['timestamp'].dt.tz_convert(et_tz)
df['day_of_week_et'] = df['timestamp_et'].dt.dayofweek
df['hour_et'] = df['timestamp_et'].dt.hour

# Check for actual weekend bars (Sat all day, Sun before 6pm)
saturday_bars = df[df['day_of_week_et'] == 5]  # Saturday
sunday_before_open = df[(df['day_of_week_et'] == 6) & (df['hour_et'] < 18)]  # Sunday before 6pm

weekend_bars = pd.concat([saturday_bars, sunday_before_open])

if len(weekend_bars) > 0:
    print(f"⚠️  Found {len(weekend_bars)} actual weekend bars (outside trading hours):")
    print(weekend_bars[['timestamp_et', 'day_of_week_et', 'hour_et']].head(10))
else:
    print("✓ No weekend bars found (all bars within Sun 6pm - Fri 5pm ET)")

# Check for gaps > 5 minutes (excluding expected maintenance gaps)
df['time_diff'] = df['timestamp'].diff()
large_gaps = df[df['time_diff'] > timedelta(minutes=5)]

# Maintenance gap patterns (5pm-6pm ET):
# - During EDT (before Nov 3): 4:59pm -> 7:00pm = 2 hours (6pm hour missing from yfinance)
# - During EST (after Nov 3): 4:59pm -> 6:00pm = 1 hour (correct)
# Weekend gap: Fri 5pm - Sun 6pm = ~49 hours
def is_expected_gap(time_diff_hours):
    # Maintenance gaps: 1 hour (EST) or 2 hours (EDT with missing 6pm data)
    if 0.9 <= time_diff_hours <= 2.1:
        return True
    # Weekend gap (48-52 hours)
    if 48 <= time_diff_hours <= 52:
        return True
    return False

unexpected_gaps = []
for idx, row in large_gaps.iterrows():
    gap_hours = row['time_diff'].total_seconds() / 3600
    if not is_expected_gap(gap_hours):
        if idx > 0:
            prev_row = df.loc[idx-1]
            unexpected_gaps.append({
                'prev_time': prev_row['timestamp_et'],
                'curr_time': row['timestamp_et'],
                'gap_hours': gap_hours
            })

if len(unexpected_gaps) > 0:
    print(f"\n⚠️  Found {len(unexpected_gaps)} unexpected gaps (not maintenance or weekend):")
    for gap in unexpected_gaps[:10]:
        print(f"  Gap: {gap['prev_time']} -> {gap['curr_time']} ({gap['gap_hours']:.1f} hours)")
else:
    print("\n✓ No unexpected gaps (all gaps are maintenance or weekend)")

# Check for bars during maintenance hour (Wed 5pm ET)
maintenance_bars = df[(df['day_of_week_et'] == 2) & (df['hour_et'] == 17)]  # Wednesday 5pm

if len(maintenance_bars) > 0:
    print(f"\n⚠️  Found {len(maintenance_bars)} bars during Wed 5pm maintenance hour")
    print("  (This is normal if maintenance didn't occur on those days)")
else:
    print("\n✓ No bars during Wednesday maintenance window")

# Check for duplicates
duplicates = df[df.duplicated(subset=['timestamp'], keep=False)]
if len(duplicates) > 0:
    print(f"\n⚠️  Found {len(duplicates)} duplicate timestamps:")
    print(duplicates[['timestamp']].head(10))
else:
    print("\n✓ No duplicate timestamps")

# Summary
print("\n" + "="*60)
print("VERIFICATION SUMMARY")
print("="*60)
print(f"Total bars: {len(df)}")
print(f"Weekend bars: {len(weekend_bars)}")
print(f"Unexpected gaps: {len(unexpected_gaps)}")
print(f"Duplicate timestamps: {len(duplicates)}")
print("="*60)
