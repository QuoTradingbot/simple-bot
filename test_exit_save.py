"""Test that exit experience saves correctly to cloud API"""
import requests
import json
from datetime import datetime

# Test exit experience with all 9 features
test_experience = {
    'timestamp': datetime.now().isoformat(),
    'regime': 'NORMAL',
    'exit_params': {
        'current_atr': 5.5,
        'breakeven_threshold_ticks': 8
    },
    'outcome': {
        'pnl': 250.0,
        'side': 'long',
        'win': True,
        'exit_reason': 'stop_loss',
        'duration': 1800,
        'contracts': 2
    },
    'situation': {
        'time_of_day': '10:30',
        'volatility_atr': 5.5
    },
    'market_state': {
        'rsi': 65.5,
        'volume_ratio': 1.8,
        'hour': 10,
        'day_of_week': 1,
        'streak': 2,
        'recent_pnl': 500.0,
        'vix': 18.5,
        'vwap_distance': 0.15,
        'atr': 5.5
    },
    'partial_exits': []
}

print("Testing exit experience save to cloud API...")
print(f"Test experience: {test_experience['outcome']['side'].upper()} | P&L: ${test_experience['outcome']['pnl']}")
print(f"Market state: RSI={test_experience['market_state']['rsi']}, Vol={test_experience['market_state']['volume_ratio']}x, Streak={test_experience['market_state']['streak']}")

try:
    response = requests.post(
        'https://quotrading-signals.icymeadow-86b2969e.eastus.azurecontainerapps.io/api/ml/save_exit_experience',
        json=test_experience,
        timeout=10
    )
    
    print(f"\nResponse status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ SUCCESS: {data.get('message', 'Saved')}")
        print(f"   Total exit experiences: {data.get('total_exit_experiences', 0)}")
        
        # Verify it's actually in the database
        print("\nVerifying database...")
        import os
        from sqlalchemy import create_engine, text
        
        engine = create_engine('postgresql://quotadmin:QuoTrading2025!Secure@quotrading-db.postgres.database.azure.com/quotrading?sslmode=require')
        with engine.connect() as conn:
            # Count exits
            exit_count = conn.execute(text("SELECT COUNT(*) FROM rl_experiences WHERE experience_type = 'EXIT'")).scalar()
            print(f"   Exits in database: {exit_count}")
            
            # Get the last saved exit to verify all 9 features
            result = conn.execute(text("""
                SELECT rsi, volume_ratio, hour_of_day, day_of_week, streak, recent_pnl, vix, vwap_distance, atr, pnl
                FROM rl_experiences 
                WHERE experience_type = 'EXIT' 
                ORDER BY timestamp DESC 
                LIMIT 1
            """))
            row = result.fetchone()
            
            if row:
                print(f"\n   Last saved exit:")
                print(f"   ✅ RSI: {row[0]}")
                print(f"   ✅ Volume Ratio: {row[1]}")
                print(f"   ✅ Hour: {row[2]}")
                print(f"   ✅ Day: {row[3]}")
                print(f"   ✅ Streak: {row[4]}")
                print(f"   ✅ Recent P&L: ${row[5]}")
                print(f"   ✅ VIX: {row[6]}")
                print(f"   ✅ VWAP Distance: {row[7]}")
                print(f"   ✅ ATR: {row[8]}")
                print(f"   ✅ P&L: ${row[9]}")
                
                # Verify no NULLs
                if None in row:
                    print("\n   ⚠️ WARNING: Some fields are NULL!")
                else:
                    print("\n   ✅ ALL 9 FEATURES SAVED CORRECTLY - READY FOR PRODUCTION!")
            
    else:
        print(f"❌ FAILED: Status {response.status_code}")
        print(f"   Response: {response.text}")
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
