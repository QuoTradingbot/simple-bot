"""
Fix data type issues in existing experiences
Convert session and trade_type from strings to integers
"""
import json

print("Fixing signal experiences...")
with open('data/local_experiences/signal_experiences_v2.json', 'r') as f:
    data = json.load(f)

fixed_count = 0
for exp in data['experiences']:
    # Convert session from string to int
    session = exp.get('session')
    if isinstance(session, str):
        exp['session'] = int(session) if session.isdigit() else 2  # Default to NY=2
        fixed_count += 1
    
    # Convert trade_type from string to int
    trade_type = exp.get('trade_type')
    if isinstance(trade_type, str):
        exp['trade_type'] = int(trade_type) if trade_type.isdigit() else 1  # Default to continuation=1

print(f"Fixed {fixed_count} signal experiences")

# Save back
with open('data/local_experiences/signal_experiences_v2.json', 'w') as f:
    json.dump(data, f, indent=2)

print("✅ Signal experiences fixed!")

# Fix exit experiences too
print("\nFixing exit experiences...")
with open('data/local_experiences/exit_experiences_v2.json', 'r') as f:
    data = json.load(f)

fixed_count = 0
for exp in data['experiences']:
    # Convert session from string to int
    session = exp.get('session')
    if isinstance(session, str):
        # Map session names to integers
        session_map = {'Asia': 0, 'London': 1, 'NY': 2}
        exp['session'] = session_map.get(session, 2)
        fixed_count += 1

print(f"Fixed {fixed_count} exit experiences")

# Save back
with open('data/local_experiences/exit_experiences_v2.json', 'w') as f:
    json.dump(data, f, indent=2)

print("✅ Exit experiences fixed!")
print("\n🎯 All data types corrected! Ready to retrain model.")
