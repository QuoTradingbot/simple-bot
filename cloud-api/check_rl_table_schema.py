"""Check the actual schema of rl_experiences table"""
import os
from sqlalchemy import create_engine, text

database_url = os.getenv('DATABASE_URL', 'postgresql://quotadmin:QuoTrading2025!Secure@quotrading-db.postgres.database.azure.com/quotrading?sslmode=require')

engine = create_engine(database_url)

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'rl_experiences'
        ORDER BY ordinal_position
    """))
    
    print("rl_experiences table columns:")
    for row in result:
        print(f"  {row[0]:25s} {row[1]:20s} {'NULL' if row[2] == 'YES' else 'NOT NULL'}")
