"""List all tables in PostgreSQL"""
import os
from sqlalchemy import create_engine, text

database_url = os.getenv('DATABASE_URL', 'postgresql://quotadmin:QuoTrading2025!Secure@quotrading-db.postgres.database.azure.com/quotrading?sslmode=require')

engine = create_engine(database_url)

with engine.connect() as conn:
    # List all tables
    result = conn.execute(text("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
    """))
    
    print("Tables in database:")
    for row in result:
        print(f"  - {row[0]}")
        
        # Get count for each table
        try:
            count_result = conn.execute(text(f"SELECT COUNT(*) FROM {row[0]}"))
            count = count_result.scalar()
            print(f"    Count: {count:,}")
        except:
            print(f"    (could not get count)")
