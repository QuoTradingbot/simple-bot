import json

with open('data/local_experiences/signal_experiences_v2.json') as f:
    data = json.load(f)

exps = data['experiences']
print(f"Total experiences: {len(exps)}")

# Count RSI values
rsi_zero = sum(1 for e in exps if e.get('rsi', -1) == 0.0)
rsi_nonzero = sum(1 for e in exps if e.get('rsi', -1) > 0.0)

print(f"RSI = 0.0: {rsi_zero} ({rsi_zero/len(exps)*100:.1f}%)")
print(f"RSI > 0.0: {rsi_nonzero} ({rsi_nonzero/len(exps)*100:.1f}%)")

# Sample some actual RSI values
sample_rsi = [e.get('rsi', 0) for e in exps[:20] if e.get('rsi', 0) > 0]
if sample_rsi:
    print(f"\nSample RSI values from first 20: {sample_rsi[:10]}")
