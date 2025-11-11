"""
Update signal experiences in PostgreSQL with proper context values from JSON
"""
import os
import json
from sqlalchemy import create_engine, text
from datetime import datetime

# Get database URL
database_url = os.getenv('DATABASE_URL', 'postgresql://quotadmin:QuoTrading2025!Secure@quotrading-db.postgres.database.azure.com/quotrading?sslmode=require')

print(f"Loading signal_experience.json...")
with open('signal_experience.json', 'r') as f:
    data = json.load(f)
    signal_experiences = data.get('experiences', [])

print(f"✓ Found {len(signal_experiences):,} signal experiences with context")

print(f"\nConnecting to PostgreSQL database...")
engine = create_engine(database_url)

try:
    with engine.connect() as conn:
        # Get current count
        result = conn.execute(text("SELECT COUNT(*) FROM rl_experiences WHERE experience_type = 'SIGNAL'"))
        current_count = result.scalar()
        print(f"Current signal experiences in database: {current_count:,}")
        
        # Check if they have ALL 13 features (check new columns)
        result = conn.execute(text("""
            SELECT COUNT(*) FROM rl_experiences 
            WHERE experience_type = 'SIGNAL' 
            AND (atr IS NULL OR volume_ratio IS NULL OR side IS NULL)
        """))
        missing_context = result.scalar()
        print(f"Experiences missing NEW 13-feature context: {missing_context:,}")
        
        if missing_context == 0:
            print("\n✅ All experiences already have context!")
            exit(0)
        
        print(f"\n🔄 Updating {current_count:,} experiences with context from JSON...")
        
        # Clear existing data and re-import with proper context
        print("  Clearing old data...")
        conn.execute(text("DELETE FROM rl_experiences WHERE experience_type = 'SIGNAL'"))
        conn.commit()
        
        print("  Re-importing with FULL 13-feature context...")
        imported = 0
        
        for i, exp in enumerate(signal_experiences):
            try:
                state = exp.get('state', {})
                
                # Insert with ALL 13 features from nested state object
                conn.execute(text("""
                    INSERT INTO rl_experiences (
                        user_id, account_id, experience_type, symbol, signal_type,
                        outcome, pnl, confidence_score, quality_score,
                        rsi, vwap_distance, vix, day_of_week, hour_of_day,
                        atr, volume_ratio, recent_pnl, streak,
                        entry_price, vwap, price, side,
                        timestamp
                    ) VALUES (
                        1, 'KEVIN-BASELINE', 'SIGNAL', 'ES', :signal_type,
                        :outcome, :pnl, 0.5, 0.5,
                        :rsi, :vwap_distance, :vix, :day_of_week, :hour_of_day,
                        :atr, :volume_ratio, :recent_pnl, :streak,
                        :entry_price, :vwap, :price, :side,
                        :timestamp
                    )
                """), {
                    'signal_type': state.get('side', 'LONG').upper(),
                    'outcome': 'WIN' if exp.get('reward', 0) > 0 else 'LOSS',
                    'pnl': exp.get('reward', 0),
                    # Original 5 features
                    'rsi': state.get('rsi'),
                    'vwap_distance': state.get('vwap_distance'),
                    'vix': state.get('atr', 0.0),  # Use ATR as VIX proxy (VIX not in historical data)
                    'day_of_week': state.get('day_of_week'),
                    'hour_of_day': state.get('hour'),
                    # New 8 features
                    'atr': state.get('atr'),
                    'volume_ratio': state.get('volume_ratio'),
                    'recent_pnl': state.get('recent_pnl'),
                    'streak': state.get('streak'),
                    'entry_price': state.get('price'),  # Use 'price' from state as entry_price
                    'vwap': None,  # Not stored in JSON state (calculated at runtime)
                    'price': state.get('price'),
                    'side': state.get('side', 'LONG').lower(),
                    'timestamp': datetime.fromisoformat(exp['timestamp'].replace('Z', '+00:00')) if 'timestamp' in exp else datetime.utcnow()
                })
                imported += 1
                
                if (i + 1) % 500 == 0:
                    conn.commit()
                    print(f"    ✓ Updated {i + 1:,}/{len(signal_experiences):,}...")
                    
            except Exception as e:
                print(f"    ⚠️  Error at experience {i}: {e}")
        
        conn.commit()
        
        print(f"\n✅ Update complete! Imported {imported:,} experiences")
        
        # Verify all 13 features are populated
        result = conn.execute(text("""
            SELECT COUNT(*) FROM rl_experiences 
            WHERE experience_type = 'SIGNAL' 
            AND rsi IS NOT NULL 
            AND vwap_distance IS NOT NULL
            AND atr IS NOT NULL
            AND volume_ratio IS NOT NULL
            AND side IS NOT NULL
        """))
        with_context = result.scalar()
        print(f"📊 Experiences with full 13-feature context: {with_context:,}/{imported:,}")
        
        # Show sample record
        result = conn.execute(text("""
            SELECT rsi, vwap_distance, atr, volume_ratio, recent_pnl, streak, 
                   day_of_week, hour_of_day, side, entry_price, price
            FROM rl_experiences 
            WHERE experience_type = 'SIGNAL'
            LIMIT 1
        """))
        sample = result.fetchone()
        if sample:
            print(f"\n📋 Sample record:")
            print(f"  RSI: {sample[0]}, VWAP Dist: {sample[1]}, ATR: {sample[2]}")
            print(f"  Volume: {sample[3]}, Recent PnL: {sample[4]}, Streak: {sample[5]}")
            print(f"  Day: {sample[6]}, Hour: {sample[7]}, Side: {sample[8]}")
            print(f"  Entry: {sample[9]}, Price: {sample[10]}")
        
except Exception as e:
    print(f"❌ Update failed: {e}")
    raise

print("\n✅ Signal RL now has proper context for pattern matching!")
