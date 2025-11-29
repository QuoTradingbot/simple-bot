import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from pathlib import Path

# Get project root (parent of scripts folder)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / 'data' / 'historical_data' / 'ES_1min.csv'

# Load existing data
df_existing = pd.read_csv(DATA_FILE)
df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'])
last_date = df_existing['timestamp'].max()

print(f'Last date in file: {last_date}')

# Download from last date + 1 day to today
start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
end = datetime.now().strftime('%Y-%m-%d')

print(f'Downloading from {start} to {end}...')

# Download new data
ticker = yf.Ticker('ES=F')
df_new = ticker.history(start=start, end=end, interval='1m')

if len(df_new) > 0:
    df_new = df_new.reset_index()
    # Keep all columns from yfinance, just rename timestamp
    df_new = df_new.rename(columns={'Datetime': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    df_new = df_new[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Append to existing
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(DATA_FILE, index=False)
    print(f'✓ Added {len(df_new)} new bars')
    print(f'✓ Total bars now: {len(df_combined)}')
else:
    print('No new data available')
