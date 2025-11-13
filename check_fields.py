import json

with open('data/local_experiences/signal_experiences_v2.json') as f:
    data = json.load(f)

exps = data['experiences']

# Check session field types
session_values = [e.get('session', 'missing') for e in exps[:20]]
print(f"Session field samples: {session_values[:10]}")
print(f"Session type: {type(session_values[0])}")

# Check trade_type field
trade_type_values = [e.get('trade_type', 'missing') for e in exps[:20]]
print(f"\nTrade_type samples: {trade_type_values[:10]}")
print(f"Trade_type type: {type(trade_type_values[0])}")
