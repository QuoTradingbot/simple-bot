"""
Daily Data Update Script
Run this every 24 hours to fetch the latest ES 1-minute bars and update your historical data.

Usage: python scripts/update_daily_data.py
"""

import pandas as pd
import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

async def update_daily_data():
    """Fetch yesterday's complete data and append to historical file"""
    
    data_file = Path('data/historical_data/ES_1min.csv')
    
    # Read existing data
    if not data_file.exists():
        print(f"Error: {data_file} not found!")
        return False
    
    df_existing = pd.read_csv(data_file)
    df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'])
    
    last_date = df_existing['timestamp'].max()
    print(f"Current data ends at: {last_date}")
    
    # Calculate date range to fetch
    # Fetch from last bar + 1 minute to current time
    start_time = last_date + timedelta(minutes=1)
    end_time = datetime.now()
    
    # Only fetch if we're missing data
    hours_missing = (end_time - start_time).total_seconds() / 3600
    
    if hours_missing < 1:
        print(f"Data is up to date (only {hours_missing:.1f} hours behind)")
        return True
    
    print(f"Fetching {hours_missing:.1f} hours of new data...")
    print(f"Range: {start_time} to {end_time}")
    
    try:
        # Import TopstepX SDK
        from project_x_py import ProjectX, ProjectXConfig
        
        # Get API credentials
        api_token = os.getenv('TOPSTEP_API_TOKEN')
        username = os.getenv('TOPSTEP_USERNAME')
        
        if not api_token or not username:
            print("Error: Missing TOPSTEP_API_TOKEN or TOPSTEP_USERNAME in .env file")
            return False
        
        # Connect and authenticate
        print("Connecting to TopstepX...")
        client = ProjectX(
            username=username,
            api_key=api_token,
            config=ProjectXConfig()
        )
        
        await client.authenticate()
        print("Authenticated successfully")
        
        # Fetch new bars
        print(f"Downloading 1-minute bars...")
        bars_df = await client.get_bars(
            symbol='ES',
            interval=1,
            unit=2,  # Minutes
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Max ~1 week of data
        )
        
        if bars_df is None or len(bars_df) == 0:
            print("No new data available (markets closed or weekend?)")
            return True
        
        # Convert to pandas if needed
        if hasattr(bars_df, 'to_pandas'):
            df_new = bars_df.to_pandas()
        else:
            df_new = bars_df
        
        # Ensure correct column names
        if 'time' in df_new.columns:
            df_new.rename(columns={'time': 'timestamp'}, inplace=True)
        
        print(f"Fetched {len(df_new)} new bars")
        print(f"  First: {df_new.iloc[0]['timestamp']}")
        print(f"  Last: {df_new.iloc[-1]['timestamp']}")
        
        # Combine with existing data
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        
        # Handle timezone differences - strip timezone without converting
        df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
        df_combined['timestamp'] = df_combined['timestamp'].astype(str).str.replace(r'[+-]\d{2}:\d{2}$', '', regex=True)
        df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
        
        # Remove duplicates and sort
        original_count = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset='timestamp', keep='last')
        df_combined = df_combined.sort_values('timestamp')
        duplicates_removed = original_count - len(df_combined)
        
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate bars")
        
        # Backup original file
        backup_file = data_file.with_suffix('.csv.backup')
        df_existing.to_csv(backup_file, index=False)
        print(f"Created backup: {backup_file}")
        
        # Save updated data
        df_combined.to_csv(data_file, index=False)
        
        new_bars = len(df_combined) - len(df_existing)
        print(f"\nUpdate complete!")
        print(f"  Total bars: {len(df_existing)} -> {len(df_combined)} (+{new_bars})")
        print(f"  Date range: {df_combined.iloc[0]['timestamp']} to {df_combined.iloc[-1]['timestamp']}")
        
        return True
        
    except ImportError as e:
        print(f"Error: Project-X SDK not installed: {e}")
        print("Install with: pip install project-x-py")
        return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("ES 1-Minute Data Daily Update")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    success = asyncio.run(update_daily_data())
    
    if success:
        print("\n✓ Update completed successfully")
        sys.exit(0)
    else:
        print("\n✗ Update failed")
        sys.exit(1)
