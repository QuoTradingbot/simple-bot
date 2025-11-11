"""
Migrate old exit experiences to PostgreSQL with partial feature support.

Old format has: timestamp, ATR, side, pnl, regime
Missing: RSI, volume_ratio, streak, recent_pnl, VIX, VWAP distance (will use NULL)
"""
import os
import json
from sqlalchemy import create_engine, text
from datetime import datetime

# Get database URL
database_url = os.getenv('DATABASE_URL', 'postgresql://quotadmin:QuoTrading2025!Secure@quotrading-db.postgres.database.azure.com/quotrading?sslmode=require')

print("Loading old exit experiences from git...")
with open('exit_experience_RESTORED.json', 'r') as f:
    data = json.load(f)
    exit_experiences = data.get('exit_experiences', [])

print(f"✓ Found {len(exit_experiences):,} old exit experiences")
print("\n⚠️  WARNING: Old format missing features - will use NULL for:")
print("   - RSI, volume_ratio, recent_pnl, streak, VIX, VWAP distance")
print("   - These experiences won't be as useful for pattern matching")
print("\n✓ Has: timestamp, ATR, side, pnl")

print(f"\nConnecting to PostgreSQL database...")
engine = create_engine(database_url)

try:
    with engine.connect() as conn:
        # Check current exit count
        result = conn.execute(text("SELECT COUNT(*) FROM rl_experiences WHERE experience_type = 'EXIT'"))
        current_count = result.scalar()
        print(f"Current exit experiences in database: {current_count:,}")
        
        if current_count > 0:
            response = input(f"\n⚠️  {current_count:,} exits already exist. Delete and re-import? (yes/no): ")
            if response.lower() != 'yes':
                print("Cancelled.")
                exit(0)
            print("  Deleting old exits...")
            conn.execute(text("DELETE FROM rl_experiences WHERE experience_type = 'EXIT'"))
            conn.commit()
        
        print(f"\n🔄 Importing {len(exit_experiences):,} exit experiences (partial features)...")
        imported = 0
        skipped = 0
        
        for i, exp in enumerate(exit_experiences):
            try:
                outcome = exp.get('outcome', {})
                situation = exp.get('situation', {})
                exit_params = exp.get('exit_params', {})
                
                # Extract what we have
                pnl = outcome.get('pnl', 0)
                side = outcome.get('side', 'LONG').upper()
                win = outcome.get('win', pnl > 0)
                
                # Parse timestamp
                timestamp_str = exp.get('timestamp')
                if timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    day_of_week = timestamp.weekday()
                    hour_of_day = timestamp.hour
                else:
                    timestamp = datetime.utcnow()
                    day_of_week = None
                    hour_of_day = None
                
                # Get ATR (try multiple locations)
                atr = (exit_params.get('current_volatility_atr') or 
                       situation.get('volatility_atr') or 
                       None)
                
                # Insert with partial features
                conn.execute(text("""
                    INSERT INTO rl_experiences (
                        user_id, account_id, experience_type, symbol, signal_type,
                        outcome, pnl, confidence_score, quality_score,
                        day_of_week, hour_of_day, atr, side, timestamp
                    ) VALUES (
                        1, 'KEVIN-BASELINE', 'EXIT', 'ES', :signal_type,
                        :outcome, :pnl, 0.5, 0.5,
                        :day_of_week, :hour_of_day, :atr, :side, :timestamp
                    )
                """), {
                    'signal_type': side,
                    'outcome': 'WIN' if win else 'LOSS',
                    'pnl': pnl,
                    'day_of_week': day_of_week,
                    'hour_of_day': hour_of_day,
                    'atr': atr,
                    'side': side.lower(),
                    'timestamp': timestamp
                })
                imported += 1
                
                if (i + 1) % 500 == 0:
                    conn.commit()
                    print(f"    ✓ Imported {i + 1:,}/{len(exit_experiences):,}...")
                    
            except Exception as e:
                print(f"    ⚠️  Error at experience {i}: {e}")
                skipped += 1
        
        conn.commit()
        
        print(f"\n✅ Import complete!")
        print(f"  Imported: {imported:,}")
        print(f"  Skipped: {skipped:,}")
        
        # Verify
        result = conn.execute(text("SELECT COUNT(*) FROM rl_experiences WHERE experience_type = 'EXIT'"))
        exit_count = result.scalar()
        
        result = conn.execute(text("SELECT COUNT(*) FROM rl_experiences WHERE experience_type = 'SIGNAL'"))
        signal_count = result.scalar()
        
        print(f"\n📊 CLOUD DATABASE TOTALS:")
        print(f"  SIGNAL: {signal_count:,} (with full 13 features)")
        print(f"  EXIT: {exit_count:,} (with partial features - missing RSI, volume, etc.)")
        print(f"  TOTAL: {signal_count + exit_count:,}")
        
        print("\n💡 NOTE: Run backtests to generate NEW exit experiences with all 9 features!")
        
except Exception as e:
    print(f"❌ Migration failed: {e}")
    raise

print("\n✅ Exit RL restored from git and migrated to cloud!")
