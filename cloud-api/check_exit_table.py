"""Check exit_experiences table structure and sample data"""
import os
from sqlalchemy import create_engine, text

database_url = os.getenv('DATABASE_URL', 'postgresql://quotadmin:QuoTrading2025!Secure@quotrading-db.postgres.database.azure.com/quotrading?sslmode=require')

engine = create_engine(database_url)

with engine.connect() as conn:
    # Get table structure
    result = conn.execute(text("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'exit_experiences'
        ORDER BY ordinal_position
    """))
    
    print("exit_experiences table structure:")
    for row in result:
        print(f"  {row[0]}: {row[1]}")
    
    # Get a sample exit experience
    print("\n" + "="*60)
    result = conn.execute(text("SELECT * FROM exit_experiences LIMIT 1"))
    sample = result.fetchone()
    
    if sample:
        print("\nSample exit experience:")
        columns = result.keys()
        for i, col in enumerate(columns):
            print(f"  {col}: {sample[i]}")
