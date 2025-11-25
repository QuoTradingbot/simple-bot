"""
Check if Azure PostgreSQL is running and accessible
"""
import psycopg2
import os

# Azure PostgreSQL credentials
DB_HOST = "quotrading-db.postgres.database.azure.com"
DB_NAME = "quotrading"
DB_USER = "quotradingadmin"
DB_PASSWORD = "QuoTrade2024!SecureDB"
DB_PORT = "5432"

print("Checking Azure PostgreSQL connection...")
print(f"Host: {DB_HOST}")
print(f"Database: {DB_NAME}")
print(f"User: {DB_USER}\n")

try:
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT,
        connect_timeout=10,
        sslmode='require'
    )
    
    cursor = conn.cursor()
    
    # Check if we can query
    cursor.execute("SELECT COUNT(*) FROM rl_experiences")
    count = cursor.fetchone()[0]
    
    print(f"✅ PostgreSQL is RUNNING")
    print(f"✅ Connection successful")
    print(f"✅ rl_experiences table: {count:,} rows\n")
    
    # Check server version
    cursor.execute("SELECT version()")
    version = cursor.fetchone()[0]
    print(f"Server version: {version[:50]}...")
    
    cursor.close()
    conn.close()
    
except psycopg2.OperationalError as e:
    print(f"❌ PostgreSQL is DOWN or unreachable")
    print(f"Error: {e}\n")
    print("Possible causes:")
    print("1. Database is paused/stopped in Azure Portal")
    print("2. Firewall blocking your IP address")
    print("3. Server is restarting")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
