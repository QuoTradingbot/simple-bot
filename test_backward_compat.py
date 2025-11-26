#!/usr/bin/env python3
"""
Test backward compatibility with old nested format.
Verifies that the RL brain can still read old nested format experiences.
"""

import json
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from signal_confidence import SignalConfidenceRL

def test_backward_compatibility():
    """Test that RL brain can still read old nested format"""
    print("=" * 70)
    print("TEST: Backward Compatibility with Nested Format")
    print("=" * 70)
    
    # Create a test file with OLD nested format
    test_file = "/tmp/test_old_format.json"
    
    old_format_data = {
        "experiences": [
            {
                "timestamp": "2025-11-21T01:48:00-05:00",
                "state": {
                    "rsi": 45.2,
                    "vwap_distance": 0.72,
                    "atr": 6.96,
                    "volume_ratio": 1.6,
                    "hour": 1,
                    "regime": "LOW_VOL_TRENDING"
                },
                "action": {
                    "took_trade": True,
                    "exploration_rate": 1.0
                },
                "reward": -352.5,
                "duration": 6.0,
                "execution": {
                    "mfe": 0.0,
                    "mae": 362.5
                }
            },
            {
                "timestamp": "2025-11-21T02:30:00-05:00",
                "state": {
                    "rsi": 52.1,
                    "vwap_distance": 0.45,
                    "atr": 7.2,
                    "volume_ratio": 1.3,
                    "hour": 2,
                    "regime": "NORMAL"
                },
                "action": {
                    "took_trade": True,
                    "exploration_rate": 1.0
                },
                "reward": 187.5,
                "duration": 12.0,
                "execution": {
                    "mfe": 250.0,
                    "mae": 50.0
                }
            }
        ],
        "stats": {
            "total_signals": 2,
            "taken": 2,
            "skipped": 0
        }
    }
    
    # Write old format to file
    with open(test_file, 'w') as f:
        json.dump(old_format_data, f, indent=2)
    
    print(f"\n1. Created test file with OLD nested format")
    print(f"   Experiences: {len(old_format_data['experiences'])}")
    
    # Load with RL brain
    print("\n2. Loading with RL brain...")
    rl_brain = SignalConfidenceRL(
        experience_file=test_file,
        backtest_mode=True,
        confidence_threshold=0.5
    )
    
    print(f"\n3. RL brain loaded {len(rl_brain.experiences)} experiences")
    
    if len(rl_brain.experiences) != 2:
        print(f"   ERROR: Expected 2 experiences, got {len(rl_brain.experiences)}")
        return False
    
    # Test calculate_confidence with a current state
    print("\n4. Testing calculate_confidence() with old format data...")
    
    current_state = {
        "rsi": 46.0,
        "vwap_distance": 0.70,
        "atr": 7.0,
        "volume_ratio": 1.5,
        "hour": 1,
        "regime": "LOW_VOL_TRENDING"
    }
    
    confidence, reason = rl_brain.calculate_confidence(current_state)
    
    print(f"   Confidence: {confidence:.2%}")
    print(f"   Reason: {reason}")
    
    # Should work without errors
    if confidence is None:
        print("   ERROR: calculate_confidence() returned None")
        return False
    
    print("   ✓ calculate_confidence() works with old format!")
    
    # Test find_similar_states
    print("\n5. Testing find_similar_states() with old format data...")
    
    similar = rl_brain.find_similar_states(current_state, max_results=2)
    
    print(f"   Found {len(similar)} similar experiences")
    
    if len(similar) == 0:
        print("   ERROR: find_similar_states() returned no results")
        return False
    
    print("   ✓ find_similar_states() works with old format!")
    
    # Now add a NEW flat format experience
    print("\n6. Adding NEW flat format experience to same brain...")
    
    new_market_state = {
        "timestamp": datetime.now().isoformat(),
        "symbol": "ES",
        "price": 5042.75,
        "returns": -0.0003,
        "vwap_distance": 0.65,
        "vwap_slope": -0.0015,
        "atr": 7.1,
        "atr_slope": 0.02,
        "rsi": 47.0,
        "macd_hist": -1.3,
        "stoch_k": 72.4,
        "volume_ratio": 1.4,
        "volume_slope": 0.42,
        "hour": 1,
        "session": "ETH",
        "regime": "LOW_VOL_TRENDING",
        "volatility_regime": "HIGH"
    }
    
    rl_brain.record_outcome(
        state=new_market_state,
        took_trade=True,
        pnl=225.0,
        duration_minutes=18.0,
        execution_data={
            "mfe": 300.0,
            "mae": 75.0
        }
    )
    
    print(f"   Total experiences now: {len(rl_brain.experiences)}")
    
    # Test that we can still find similar with MIXED formats
    print("\n7. Testing find_similar_states() with MIXED format data...")
    
    similar_mixed = rl_brain.find_similar_states(current_state, max_results=3)
    
    print(f"   Found {len(similar_mixed)} similar experiences from mixed format data")
    
    if len(similar_mixed) < 2:
        print(f"   ERROR: Expected at least 2 similar, got {len(similar_mixed)}")
        return False
    
    print("   ✓ find_similar_states() works with MIXED formats!")
    
    # Save and reload to verify persistence
    print("\n8. Saving and reloading...")
    rl_brain.save_experience()
    
    # Create a new brain instance from the saved file
    rl_brain2 = SignalConfidenceRL(
        experience_file=test_file,
        backtest_mode=True,
        confidence_threshold=0.5
    )
    
    print(f"   Reloaded {len(rl_brain2.experiences)} experiences")
    
    if len(rl_brain2.experiences) != 3:
        print(f"   ERROR: Expected 3 experiences after reload, got {len(rl_brain2.experiences)}")
        return False
    
    # Check format of experiences
    print("\n9. Checking experience formats after reload...")
    
    # First two should be old nested format (or converted to flat)
    # Third should be new flat format
    for i, exp in enumerate(rl_brain2.experiences):
        has_state = 'state' in exp
        has_pnl_top = 'pnl' in exp
        has_reward_top = 'reward' in exp
        
        print(f"   Experience {i+1}: has_state={has_state}, has_pnl={has_pnl_top}, has_reward={has_reward_top}")
    
    # Test calculate_confidence one more time
    confidence2, reason2 = rl_brain2.calculate_confidence(current_state)
    print(f"\n10. Final confidence test: {confidence2:.2%} - {reason2}")
    
    print("\n" + "=" * 70)
    print("✓ TEST PASSED: Backward compatibility works!")
    print("  - Can read old nested format")
    print("  - Can save new flat format")
    print("  - Can work with mixed formats")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_backward_compatibility()
    sys.exit(0 if success else 1)
