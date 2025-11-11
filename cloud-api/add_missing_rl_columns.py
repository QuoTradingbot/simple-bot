"""
Add missing RL context columns to support all 13 signal features
"""
import os
from sqlalchemy import create_engine, text

# Get database URL
database_url = os.getenv('DATABASE_URL', 'postgresql://quotadmin:QuoTrading2025!Secure@quotrading-db.postgres.database.azure.com/quotrading?sslmode=require')

print("🔧 Adding missing RL context columns to database...")
print(f"Database: {database_url.split('@')[1].split('/')[0]}")

engine = create_engine(database_url)

# Columns to add (if they don't exist)
new_columns = {
    'atr': 'FLOAT',
    'volume_ratio': 'FLOAT',
    'recent_pnl': 'FLOAT',
    'streak': 'INTEGER',
    'entry_price': 'FLOAT',
    'vwap': 'FLOAT',
    'price': 'FLOAT',
    'side': 'VARCHAR(10)'  # 'long' or 'short'
}

try:
    with engine.connect() as conn:
        # Check existing columns
        result = conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'rl_experiences'
        """))
        existing_columns = {row[0] for row in result}
        
        print(f"\n📊 Current columns in rl_experiences: {len(existing_columns)}")
        
        # Add missing columns
        added = 0
        for col_name, col_type in new_columns.items():
            if col_name not in existing_columns:
                print(f"  ➕ Adding column: {col_name} ({col_type})")
                conn.execute(text(f"ALTER TABLE rl_experiences ADD COLUMN {col_name} {col_type}"))
                conn.commit()
                added += 1
            else:
                print(f"  ✓ Column already exists: {col_name}")
        
        print(f"\n✅ Added {added} new columns")
        
        # Verify all columns exist
        result = conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'rl_experiences'
            ORDER BY column_name
        """))
        all_columns = [row[0] for row in result]
        
        print(f"\n📋 Total columns now: {len(all_columns)}")
        print("All RL context columns:")
        context_cols = ['rsi', 'vwap_distance', 'vix', 'day_of_week', 'hour_of_day', 
                       'atr', 'volume_ratio', 'recent_pnl', 'streak', 
                       'entry_price', 'vwap', 'price', 'side']
        for col in context_cols:
            status = "✓" if col in all_columns else "✗"
            print(f"  {status} {col}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    raise

print("\n✅ Database schema updated! Ready for full 13-feature migration.")
