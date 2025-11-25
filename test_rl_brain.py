"""
Test RL Brain to see what it's doing with your experiences
"""
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from signal_confidence import SignalConfidenceRL

# Load RL brain
rl = SignalConfidenceRL(
    experience_file="data/signal_experience.json",
    backtest_mode=True
)

print(f"\n{'='*80}")
print(f"RL BRAIN TEST")
print(f"{'='*80}")
print(f"Total experiences loaded: {len(rl.experiences)}")
print()

# Test with a typical signal state
test_state = {
    'rsi': 45.0,
    'vwap_distance': 0.02,
    'atr': 3.5,
    'volume_ratio': 1.2,
    'hour': 10,
    'day_of_week': 2,
    'recent_pnl': 0.0,
    'streak': 0,
    'side': 'LONG',
    'regime': 'NORMAL'
}

print("Test State:")
for k, v in test_state.items():
    print(f"  {k}: {v}")
print()

# Get confidence
confidence, reason = rl.calculate_confidence(test_state)

print(f"RL Decision:")
print(f"  Confidence: {confidence:.1%}")
print(f"  Reason: {reason}")
print(f"  Threshold: 70%")
print(f"  Decision: {'✅ APPROVED' if confidence > 0.70 else '❌ REJECTED'}")
print()

# Find the 20 similar experiences manually to see what's happening
similar = rl.find_similar_states(test_state, max_results=20)

if similar:
    print(f"20 Similar Experiences Found:")
    wins = sum(1 for exp in similar if exp['reward'] > 0)
    losses = sum(1 for exp in similar if exp['reward'] < 0)
    avg_profit = sum(exp['reward'] for exp in similar) / len(similar)
    win_rate = wins / len(similar)
    
    print(f"  Wins: {wins}/{len(similar)} ({win_rate*100:.1f}%)")
    print(f"  Losses: {losses}/{len(similar)}")
    print(f"  Avg Profit: ${avg_profit:.2f}")
    print()
    
    print("Sample of similar experiences:")
    for i, exp in enumerate(similar[:5], 1):
        regime = exp['state'].get('regime', 'N/A')
        reward = exp['reward']
        side = exp['state'].get('side', 'N/A')
        print(f"  {i}. {side} {regime}: ${reward:.2f}")
    
    print()
    
    # Show regime breakdown
    regimes = {}
    for exp in similar:
        regime = exp['state'].get('regime', 'UNKNOWN')
        if regime not in regimes:
            regimes[regime] = {'count': 0, 'wins': 0, 'total_pnl': 0}
        regimes[regime]['count'] += 1
        if exp['reward'] > 0:
            regimes[regime]['wins'] += 1
        regimes[regime]['total_pnl'] += exp['reward']
    
    print("Regime Breakdown of Similar Experiences:")
    for regime, stats in sorted(regimes.items()):
        wr = stats['wins'] / stats['count'] * 100 if stats['count'] > 0 else 0
        avg = stats['total_pnl'] / stats['count'] if stats['count'] > 0 else 0
        print(f"  {regime}: {stats['count']} trades, {wr:.0f}% WR, ${avg:.2f} avg")

else:
    print("❌ NO SIMILAR EXPERIENCES FOUND")

print(f"\n{'='*80}\n")
