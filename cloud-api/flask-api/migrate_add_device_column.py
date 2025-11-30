"""
Add device_fingerprint column to database for session locking.
Run this once to update the Azure PostgreSQL database schema.
"""
import os
import psycopg2

# Database connection details
DB_HOST = os.environ.get("DB_HOST", "quotrading-db.postgres.database.azure.com")
DB_NAME = os.environ.get("DB_NAME", "quotrading")
DB_USER = os.environ.get("DB_USER", "quotradingadmin")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_PORT = os.environ.get("DB_PORT", "5432")

print("Connecting to Azure PostgreSQL...")
print(f"Host: {DB_HOST}")
print(f"Database: {DB_NAME}")
print(f"User: {DB_USER}")

try:
    # Connect to database
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT,
        sslmode='require'
    )
    
    cursor = conn.cursor()
    
    print("\n✅ Connected successfully!")
    print("\nAdding device_fingerprint columns...")
    
    # Add column to users table
    cursor.execute("""
        ALTER TABLE users 
        ADD COLUMN IF NOT EXISTS device_fingerprint VARCHAR(255)
    """)
    print("✅ Added device_fingerprint to users table")
    
    # Add column to heartbeats table  
    cursor.execute("""
        ALTER TABLE heartbeats 
        ADD COLUMN IF NOT EXISTS device_fingerprint VARCHAR(255)
    """)
    print("✅ Added device_fingerprint to heartbeats table")
    
    # Create indexes for performance
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_users_device_fingerprint 
        ON users(device_fingerprint)
    """)
    print("✅ Created index on users.device_fingerprint")
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_users_last_heartbeat 
        ON users(last_heartbeat)
    """)
    print("✅ Created index on users.last_heartbeat")
    
    # Commit changes
    conn.commit()
    
    print("\n🎉 Database schema updated successfully!")
    print("Session locking is now enabled.")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
