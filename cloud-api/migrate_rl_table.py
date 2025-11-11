"""
Add missing columns to rl_experiences table for context-aware learning
"""
import os
from sqlalchemy import create_engine, text

# Get database URL from environment
database_url = os.getenv('DATABASE_URL', 'postgresql://quotadmin:QuoTrading2025!Secure@quotrading-db.postgres.database.azure.com/quotrading?sslmode=require')

print(f"Connecting to database...")
engine = create_engine(database_url)

# Add missing columns to rl_experiences table
migration_sql = """
-- Add context columns if they don't exist
ALTER TABLE rl_experiences 
ADD COLUMN IF NOT EXISTS rsi FLOAT,
ADD COLUMN IF NOT EXISTS vwap_distance FLOAT,
ADD COLUMN IF NOT EXISTS vix FLOAT,
ADD COLUMN IF NOT EXISTS day_of_week INTEGER,
ADD COLUMN IF NOT EXISTS hour_of_day INTEGER;

-- Verify columns exist
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'rl_experiences'
ORDER BY ordinal_position;
"""

try:
    with engine.connect() as conn:
        print("Adding missing columns to rl_experiences table...")
        conn.execute(text("""
            ALTER TABLE rl_experiences 
            ADD COLUMN IF NOT EXISTS rsi FLOAT,
            ADD COLUMN IF NOT EXISTS vwap_distance FLOAT,
            ADD COLUMN IF NOT EXISTS vix FLOAT,
            ADD COLUMN IF NOT EXISTS day_of_week INTEGER,
            ADD COLUMN IF NOT EXISTS hour_of_day INTEGER
        """))
        conn.commit()
        
        print("\n✅ Migration successful!")
        print("\nVerifying table schema:")
        result = conn.execute(text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'rl_experiences'
            ORDER BY ordinal_position
        """))
        
        for row in result:
            print(f"  - {row[0]}: {row[1]}")
        
        # Count existing experiences
        count_result = conn.execute(text("SELECT COUNT(*) FROM rl_experiences WHERE experience_type = 'SIGNAL'"))
        signal_count = count_result.scalar()
        print(f"\n📊 Found {signal_count:,} signal experiences in database")
        
except Exception as e:
    print(f"❌ Migration failed: {e}")
    raise

print("\n✅ Database schema updated successfully!")
print("   Signal RL can now use context-aware pattern matching")
