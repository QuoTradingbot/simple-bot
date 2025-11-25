from datetime import datetime
import pytz

et = pytz.timezone('US/Eastern')
utc = pytz.UTC

# Corrected Iteration 3 times in ET
times = [
    ("6:00 PM ET", 18, 0),
    ("4:00 PM ET", 16, 0),
    ("4:45 PM ET", 16, 45),
]

print("Corrected Iteration 3 Trading Hours - ET to UTC Conversion:")
print("=" * 60)

for label, hour, minute in times:
    # Use Nov 25, 2025 (EST, not EDT)
    dt = datetime(2025, 11, 25, hour, minute)
    et_time = et.localize(dt)
    utc_time = et_time.astimezone(utc)
    print(f"{label:15s} = {utc_time.strftime('%H:%M')} UTC")
