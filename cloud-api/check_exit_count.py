"""Check exit experience count in PostgreSQL"""
import os
from sqlalchemy import create_engine, text

database_url = os.getenv('DATABASE_URL', 'postgresql://quotadmin:QuoTrading2025!Secure@quotrading-db.postgres.database.azure.com/quotrading?sslmode=require')

engine = create_engine(database_url)

with engine.connect() as conn:
    # Check exit experiences
    result = conn.execute(text("SELECT COUNT(*) FROM rl_experiences WHERE experience_type = 'EXIT'"))
    exit_count = result.scalar()
    print(f"Exit experiences in database: {exit_count:,}")
    
    # Get a sample exit experience
    result = conn.execute(text("""
        SELECT rsi, volume_ratio, hour_of_day, day_of_week, streak, recent_pnl, vix, vwap_distance, atr
        FROM rl_experiences 
        WHERE experience_type = 'EXIT' 
        LIMIT 1
    """))
    sample = result.fetchone()
    
    if sample:
        print(f"\nSample exit experience:")
        print(f"  RSI: {sample[0]}")
        print(f"  Volume Ratio: {sample[1]}")
        print(f"  Hour: {sample[2]}")
        print(f"  Day: {sample[3]}")
        print(f"  Streak: {sample[4]}")
        print(f"  Recent PnL: {sample[5]}")
        print(f"  VIX: {sample[6]}")
        print(f"  VWAP Distance: {sample[7]}")
        print(f"  ATR: {sample[8]}")
    
    # Check how many have NULL values
    result = conn.execute(text("""
        SELECT COUNT(*) FROM rl_experiences 
        WHERE experience_type = 'EXIT' 
        AND (rsi IS NULL OR volume_ratio IS NULL OR streak IS NULL OR recent_pnl IS NULL OR vix IS NULL OR vwap_distance IS NULL)
    """))
    null_count = result.scalar()
    print(f"\nExit experiences with NULL values: {null_count:,}")
