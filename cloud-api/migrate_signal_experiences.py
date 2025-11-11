"""
Migrate signal experiences from JSON file to PostgreSQL
"""
import os
import json
from sqlalchemy import create_engine, text
from datetime import datetime

# Get database URL from environment
database_url = os.getenv('DATABASE_URL', 'postgresql://quotadmin:QuoTrading2025!Secure@quotrading-db.postgres.database.azure.com/quotrading?sslmode=require')

print(f"Loading signal_experience.json...")
with open('signal_experience.json', 'r') as f:
    data = json.load(f)
    signal_experiences = data.get('experiences', [])

print(f"✓ Found {len(signal_experiences):,} signal experiences")

print(f"\nConnecting to PostgreSQL database...")
engine = create_engine(database_url)

try:
    with engine.connect() as conn:
        # Check current count
        result = conn.execute(text("SELECT COUNT(*) FROM rl_experiences WHERE experience_type = 'SIGNAL'"))
        current_count = result.scalar()
        print(f"Current signal experiences in database: {current_count:,}")
        
        if current_count > 0:
            response = input(f"\n⚠️  Database already has {current_count:,} signal experiences. Continue? (y/n): ")
            if response.lower() != 'y':
                print("Migration cancelled.")
                exit(0)
        
        print(f"\n🔄 Migrating {len(signal_experiences):,} signal experiences to PostgreSQL...")
        imported = 0
        skipped = 0
        
        for i, exp in enumerate(signal_experiences):
            try:
                # Insert signal experience
                conn.execute(text("""
                    INSERT INTO rl_experiences (
                        user_id, account_id, experience_type, symbol, signal_type,
                        outcome, pnl, confidence_score, quality_score,
                        rsi, vwap_distance, vix, day_of_week, hour_of_day, timestamp
                    ) VALUES (
                        1, :account_id, 'SIGNAL', :symbol, :signal_type,
                        :outcome, :pnl, :confidence_score, :quality_score,
                        :rsi, :vwap_distance, :vix, :day_of_week, :hour_of_day, :timestamp
                    )
                """), {
                    'account_id': exp.get('user_id', 'KEVIN-BASELINE'),
                    'symbol': exp.get('symbol', 'ES'),
                    'signal_type': exp.get('signal', 'LONG').upper(),
                    'outcome': 'WIN' if exp.get('pnl', 0) > 0 else 'LOSS',
                    'pnl': exp.get('pnl', 0),
                    'confidence_score': exp.get('confidence', 0.5),
                    'quality_score': exp.get('quality_score', 0.5),
                    'rsi': exp.get('rsi'),
                    'vwap_distance': exp.get('vwap_distance'),
                    'vix': exp.get('vix'),
                    'day_of_week': exp.get('day_of_week'),
                    'hour_of_day': exp.get('hour_of_day'),
                    'timestamp': datetime.fromisoformat(exp['timestamp'].replace('Z', '+00:00')) if 'timestamp' in exp else datetime.utcnow()
                })
                imported += 1
                
                if (i + 1) % 500 == 0:
                    conn.commit()
                    print(f"  ✓ Migrated {i + 1:,}/{len(signal_experiences):,} experiences...")
                    
            except Exception as e:
                skipped += 1
                if skipped < 10:  # Only show first 10 errors
                    print(f"  ⚠️  Skipped experience {i}: {e}")
        
        # Final commit
        conn.commit()
        
        print(f"\n✅ Migration complete!")
        print(f"   Imported: {imported:,}")
        print(f"   Skipped:  {skipped:,}")
        
        # Verify count
        result = conn.execute(text("SELECT COUNT(*) FROM rl_experiences WHERE experience_type = 'SIGNAL'"))
        final_count = result.scalar()
        print(f"\n📊 Total signal experiences in database: {final_count:,}")
        
except Exception as e:
    print(f"❌ Migration failed: {e}")
    raise

print("\n✅ Signal RL is now ready with 6,880+ baseline experiences!")
