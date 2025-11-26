"""
Database initialization script - Creates rl_experiences table
Run this once to set up PostgreSQL schema
"""
import os
import psycopg2

# Get credentials from Azure App Settings
DB_HOST = os.environ.get("DB_HOST", "quotrading-db.postgres.database.azure.com")
DB_NAME = os.environ.get("DB_NAME", "quotrading")
DB_USER = os.environ.get("DB_USER", "quotradingadmin")
DB_PASSWORD = os.environ.get("DB_PASSWORD")

def init_database():
    """Create rl_experiences table and indexes"""
    print("Initializing PostgreSQL database...")
    
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode='require'
    )
    cursor = conn.cursor()
    
    # Create table
    print("Creating rl_experiences table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rl_experiences (
            id SERIAL PRIMARY KEY,
            license_key VARCHAR(50) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            price DECIMAL(10,2) NOT NULL,
            returns DECIMAL(10,6) NOT NULL,
            vwap_distance DECIMAL(10,6) NOT NULL,
            vwap_slope DECIMAL(10,6) NOT NULL,
            atr DECIMAL(10,6) NOT NULL,
            atr_slope DECIMAL(10,6) NOT NULL,
            rsi DECIMAL(5,2) NOT NULL,
            macd_hist DECIMAL(10,6) NOT NULL,
            stoch_k DECIMAL(5,2) NOT NULL,
            volume_ratio DECIMAL(10,2) NOT NULL,
            volume_slope DECIMAL(10,2) NOT NULL,
            hour INTEGER NOT NULL,
            session VARCHAR(10) NOT NULL,
            regime VARCHAR(50) NOT NULL,
            volatility_regime VARCHAR(20) NOT NULL,
            pnl DECIMAL(10,2) NOT NULL,
            duration DECIMAL(10,2) NOT NULL,
            took_trade BOOLEAN NOT NULL,
            exploration_rate DECIMAL(5,2) NOT NULL,
            mfe DECIMAL(10,2) NOT NULL,
            mae DECIMAL(10,2) NOT NULL,
            order_type_used VARCHAR(20) NOT NULL,
            entry_slippage_ticks DECIMAL(5,2) NOT NULL,
            exit_reason VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    
    # Create indexes
    print("Creating indexes...")
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_rl_experiences_license ON rl_experiences(license_key)",
        "CREATE INDEX IF NOT EXISTS idx_rl_experiences_symbol ON rl_experiences(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_rl_experiences_created ON rl_experiences(created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_rl_experiences_took_trade ON rl_experiences(took_trade)",
        "CREATE INDEX IF NOT EXISTS idx_rl_experiences_regime ON rl_experiences(regime)",
        "CREATE INDEX IF NOT EXISTS idx_rl_experiences_timestamp ON rl_experiences(timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS idx_rl_experiences_similarity ON rl_experiences(symbol, regime, volatility_regime, rsi, vwap_distance, atr)"
    ]
    
    for idx in indexes:
        cursor.execute(idx)
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print("✅ Database initialized successfully!")

if __name__ == "__main__":
    init_database()
