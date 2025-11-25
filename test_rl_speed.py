import time
import sys
sys.path.insert(0, 'src')

from signal_confidence import SignalConfidenceRL

# Load RL brain
print("Loading RL brain...")
start = time.time()
rl = SignalConfidenceRL(
    experience_file='data/signal_experience.json',
    backtest_mode=False,
    exploration_rate=0.0
)
load_time = time.time() - start
print(f"✅ Loaded {len(rl.experiences):,} experiences in {load_time:.3f}s")

# Test decision speed
print("\nTesting decision speed (100 queries)...")

test_state = {
    'rsi': 45.2,
    'vwap_distance': 0.02,
    'atr': 2.5,
    'volume_ratio': 1.3,
    'hour': 14,
    'day_of_week': 2,
    'recent_pnl': -50.0,
    'streak': -1,
    'side': 'long',
    'regime': 'NORMAL'
}

times = []
results = []
for i in range(100):
    start = time.time()
    result = rl.should_take_signal(test_state)
    elapsed = (time.time() - start) * 1000  # Convert to milliseconds
    times.append(elapsed)
    if i == 0:
        results = result

avg_time = sum(times) / len(times)
min_time = min(times)
max_time = max(times)
p95_time = sorted(times)[int(len(times) * 0.95)]

print(f"\n📊 RL Decision Performance:")
print(f"   Average: {avg_time:.2f}ms")
print(f"   Min:     {min_time:.2f}ms")
print(f"   Max:     {max_time:.2f}ms")
print(f"   P95:     {p95_time:.2f}ms")
print(f"\n   Result: {results}")
print(f"   Result length: {len(results)}")
