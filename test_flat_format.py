#!/usr/bin/env python3
"""
Test script to verify flat format experience saving.
Tests that record_outcome() saves experiences in the new flat format.
"""

import json
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from signal_confidence import SignalConfidenceRL

def test_flat_format():
    """Test that experiences are saved in flat format"""
    print("=" * 70)
    print("TEST: Flat Format Experience Saving")
    print("=" * 70)
    
    # Create a test experience file
    test_file = "/tmp/test_experiences.json"
    if os.path.exists(test_file):
        os.remove(test_file)
    
    # Create RL brain with test file
    rl_brain = SignalConfidenceRL(
        experience_file=test_file,
        backtest_mode=True,
        confidence_threshold=0.5
    )
    
    # Create a market state (flat format from capture_market_state)
    market_state = {
        "timestamp": datetime.now().isoformat(),
        "symbol": "ES",
        "price": 5042.75,
        "returns": -0.0003,
        "vwap_distance": 0.02,
        "vwap_slope": -0.0015,
        "atr": 2.5,
        "atr_slope": 0.02,
        "rsi": 45.2,
        "macd_hist": -1.3,
        "stoch_k": 72.4,
        "volume_ratio": 1.3,
        "volume_slope": 0.42,
        "hour": 14,
        "session": "RTH",
        "regime": "NORMAL_CHOPPY",
        "volatility_regime": "MEDIUM"
    }
    
    print(f"\n1. Created market state with {len(market_state)} fields")
    print(f"   Keys: {list(market_state.keys())}")
    
    # Record an outcome
    print("\n2. Recording outcome...")
    rl_brain.record_outcome(
        state=market_state,
        took_trade=True,
        pnl=125.50,
        duration_minutes=15.2,
        execution_data={
            "mfe": 200.0,
            "mae": 50.0
        }
    )
    
    # Force save
    rl_brain.save_experience()
    
    print("\n3. Saved experience to file")
    
    # Read the file and verify format
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    experiences = data.get('experiences', [])
    print(f"\n4. Loaded {len(experiences)} experiences from file")
    
    if len(experiences) == 0:
        print("\nERROR: No experiences saved!")
        return False
    
    exp = experiences[0]
    print(f"\n5. First experience has {len(exp)} fields")
    print(f"   Keys: {list(exp.keys())}")
    
    # Check for flat format (fields at top level, not nested)
    print("\n6. Checking for FLAT format...")
    
    # Should NOT have these nested keys
    nested_keys = ['state', 'action', 'reward']
    has_nested = any(key in exp for key in nested_keys)
    
    if has_nested:
        print(f"   ERROR: Found nested format keys: {[k for k in nested_keys if k in exp]}")
        return False
    else:
        print(f"   ✓ No nested keys found (good!)")
    
    # Should have these flat keys
    required_flat_keys = [
        'timestamp', 'symbol', 'price', 'returns', 'vwap_distance',
        'rsi', 'atr', 'volume_ratio', 'hour', 'session', 'regime',
        'pnl', 'duration', 'took_trade', 'mfe', 'mae'
    ]
    
    missing_keys = [k for k in required_flat_keys if k not in exp]
    
    if missing_keys:
        print(f"   ERROR: Missing required flat keys: {missing_keys}")
        return False
    else:
        print(f"   ✓ All required flat keys present!")
    
    # Verify values
    print("\n7. Verifying values...")
    print(f"   timestamp: {exp.get('timestamp', 'MISSING')}")
    print(f"   symbol: {exp.get('symbol', 'MISSING')}")
    print(f"   price: {exp.get('price', 'MISSING')}")
    print(f"   pnl: {exp.get('pnl', 'MISSING')}")
    print(f"   duration: {exp.get('duration', 'MISSING')}")
    print(f"   mfe: {exp.get('mfe', 'MISSING')}")
    print(f"   mae: {exp.get('mae', 'MISSING')}")
    print(f"   took_trade: {exp.get('took_trade', 'MISSING')}")
    
    # Verify correct values
    checks = [
        (exp.get('symbol') == 'ES', 'symbol should be ES'),
        (exp.get('price') == 5042.75, 'price should be 5042.75'),
        (exp.get('pnl') == 125.50, 'pnl should be 125.50'),
        (exp.get('duration') == 15.2, 'duration should be 15.2'),
        (exp.get('mfe') == 200.0, 'mfe should be 200.0'),
        (exp.get('mae') == 50.0, 'mae should be 50.0'),
        (exp.get('took_trade') == True, 'took_trade should be True'),
    ]
    
    all_passed = True
    for check, msg in checks:
        if not check:
            print(f"   ERROR: {msg}")
            all_passed = False
    
    if all_passed:
        print(f"   ✓ All values correct!")
    
    # Pretty print the full experience
    print("\n8. Full experience (formatted):")
    print(json.dumps(exp, indent=2))
    
    print("\n" + "=" * 70)
    if all_passed and not has_nested and not missing_keys:
        print("✓ TEST PASSED: Experiences are saved in FLAT format!")
        print("=" * 70)
        return True
    else:
        print("✗ TEST FAILED: Experiences are NOT in correct flat format")
        print("=" * 70)
        return False

if __name__ == "__main__":
    success = test_flat_format()
    sys.exit(0 if success else 1)
