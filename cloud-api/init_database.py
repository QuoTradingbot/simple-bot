"""
Initialize PostgreSQL database with rl_experiences table.
Run this once to set up the database schema.
"""
import os
import psycopg2
from psycopg2 import sql

# Azure PostgreSQL connection details
DB_HOST = os.getenv("DATABASE_HOST", "quotrading-db.postgres.database.azure.com")
DB_NAME = os.getenv("DATABASE_NAME", "quotrading")
DB_USER = os.getenv("DATABASE_USER", "quotradingadmin")
DB_PASSWORD = os.getenv("DATABASE_PASSWORD") or input("Enter PostgreSQL password: ")

def init_database():
    """Create the rl_experiences table with all indexes."""
    
    # SQL commands embedded
    sql_commands = """
    CREATE TABLE IF NOT EXISTS rl_experiences (
        id SERIAL PRIMARY KEY,
        license_key VARCHAR(100) NOT NULL,
        symbol VARCHAR(20) NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        rsi FLOAT,
        vwap_distance FLOAT,
        atr FLOAT,
        volume_ratio FLOAT,
        side VARCHAR(10),
        regime VARCHAR(20),
        pnl FLOAT,
        duration_minutes FLOAT,
        win BOOLEAN,
        metadata JSONB
    );

    CREATE INDEX IF NOT EXISTS idx_license_key ON rl_experiences(license_key);
    CREATE INDEX IF NOT EXISTS idx_symbol ON rl_experiences(symbol);
    CREATE INDEX IF NOT EXISTS idx_created_at ON rl_experiences(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_took_trade ON rl_experiences(took_trade);
    CREATE INDEX IF NOT EXISTS idx_side ON rl_experiences(side);
    CREATE INDEX IF NOT EXISTS idx_regime ON rl_experiences(regime);
    CREATE INDEX IF NOT EXISTS idx_composite_similarity ON rl_experiences(symbol, rsi, vwap_distance, atr, volume_ratio, side, regime);

    GRANT ALL PRIVILEGES ON TABLE rl_experiences TO quotradingadmin;
    GRANT ALL PRIVILEGES ON SEQUENCE rl_experiences_id_seq TO quotradingadmin;
    """
    
    print(f"Connecting to {DB_HOST}...")
    # For Azure PostgreSQL Flexible Server, use plain username
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,  # Just quotradingadmin, no @servername
        password=DB_PASSWORD,
        sslmode='require'
    )
    
    try:
        cursor = conn.cursor()
        
        # Execute the SQL commands
        print("Creating rl_experiences table...")
        cursor.execute(sql_commands)
        
        conn.commit()
        print("✅ Database initialized successfully!")
        
        # Verify table exists
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_name = 'rl_experiences'
        """)
        count = cursor.fetchone()[0]
        print(f"Table verification: {count} table(s) found")
        
        cursor.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    init_database()
