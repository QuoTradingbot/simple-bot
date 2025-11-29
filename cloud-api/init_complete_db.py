"""
Complete PostgreSQL Database Initialization
Creates ALL required tables: users, rl_experiences, api_logs, heartbeats
"""
import os
import psycopg2

# Azure PostgreSQL connection
DB_HOST = os.getenv("DB_HOST", "quotrading-db.postgres.database.azure.com")
DB_NAME = os.getenv("DB_NAME", "quotrading")
DB_USER = os.getenv("DB_USER", "quotradingadmin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "QuoTrade2024!SecureDB")

def init_complete_database():
    """Create ALL tables needed for QuoTrading API"""
    
    print(f"🔗 Connecting to {DB_HOST}...")
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode='require'
    )
    
    cursor = conn.cursor()
    
    try:
        # 1. USERS TABLE (licenses)
        print("📋 Creating users table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                account_id VARCHAR(100) PRIMARY KEY,
                license_key VARCHAR(100) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                whop_user_id VARCHAR(100),
                whop_membership_id VARCHAR(100),
                license_type VARCHAR(50) NOT NULL DEFAULT 'Monthly',
                license_status VARCHAR(20) NOT NULL DEFAULT 'active',
                license_expiration TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_heartbeat TIMESTAMP,
                metadata JSONB DEFAULT '{}'
            )
        """)
        
        # 2. RL EXPERIENCES TABLE
        print("🧠 Creating rl_experiences table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rl_experiences (
                id SERIAL PRIMARY KEY,
                license_key VARCHAR(100) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                price DECIMAL(10,2),
                returns DECIMAL(10,6),
                vwap_distance DECIMAL(10,6),
                vwap_slope DECIMAL(10,6),
                atr DECIMAL(10,6),
                atr_slope DECIMAL(10,6),
                rsi DECIMAL(5,2),
                macd_hist DECIMAL(10,6),
                stoch_k DECIMAL(5,2),
                volume_ratio DECIMAL(10,2),
                volume_slope DECIMAL(10,2),
                hour INTEGER,
                session VARCHAR(10),
                regime VARCHAR(50),
                volatility_regime VARCHAR(20),
                pnl DECIMAL(10,2) NOT NULL,
                duration DECIMAL(10,2),
                took_trade BOOLEAN NOT NULL DEFAULT FALSE,
                exploration_rate DECIMAL(5,2),
                mfe DECIMAL(10,2),
                mae DECIMAL(10,2),
                order_type_used VARCHAR(20),
                entry_slippage_ticks DECIMAL(5,2),
                exit_reason VARCHAR(50),
                side VARCHAR(10),
                win BOOLEAN,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 3. API LOGS TABLE
        print("📊 Creating api_logs table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_logs (
                id SERIAL PRIMARY KEY,
                license_key VARCHAR(100),
                endpoint VARCHAR(255) NOT NULL,
                method VARCHAR(10) DEFAULT 'POST',
                request_data JSONB,
                response_data JSONB,
                status_code INTEGER,
                ip_address VARCHAR(45),
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 4. HEARTBEATS TABLE
        print("💓 Creating heartbeats table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS heartbeats (
                id SERIAL PRIMARY KEY,
                license_key VARCHAR(100) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                bot_version VARCHAR(50),
                status VARCHAR(20) DEFAULT 'online',
                metadata JSONB DEFAULT '{}'
            )
        """)
        
        # CREATE INDEXES
        print("🔍 Creating indexes...")
        
        # Users indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_license_key ON users(license_key)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_status ON users(license_status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_expiration ON users(license_expiration)")
        
        # RL experiences indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rl_license_key ON rl_experiences(license_key)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rl_symbol ON rl_experiences(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rl_timestamp ON rl_experiences(timestamp DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rl_created_at ON rl_experiences(created_at DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rl_took_trade ON rl_experiences(took_trade)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rl_regime ON rl_experiences(regime)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rl_similarity ON rl_experiences(symbol, regime, volatility_regime, rsi, vwap_distance, atr)")
        
        # API logs indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_license_key ON api_logs(license_key)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_endpoint ON api_logs(endpoint)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_created_at ON api_logs(created_at DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_status_code ON api_logs(status_code)")
        
        # Heartbeats indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_heartbeats_license_key ON heartbeats(license_key)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_heartbeats_timestamp ON heartbeats(timestamp DESC)")
        
        conn.commit()
        
        # VERIFY TABLES
        print("\n✅ Verifying tables...")
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """)
        tables = cursor.fetchall()
        
        print("📋 Tables created:")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
            count = cursor.fetchone()[0]
            print(f"  ✓ {table[0]}: {count} rows")
        
        print("\n🎉 Database initialization complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    init_complete_database()
