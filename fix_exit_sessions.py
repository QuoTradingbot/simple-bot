import json

# Fix exit experiences - convert session strings to integers
with open('data/local_experiences/exit_experiences_v2.json', 'r') as f:
    exit_data = json.load(f)

session_map = {'Asia': 0, 'London': 1, 'NY': 2}
fixed = 0

for exp in exit_data['experiences']:
    session = exp['outcome'].get('session')
    if isinstance(session, str):
        exp['outcome']['session'] = session_map.get(session, 2)
        fixed += 1

# Save with proper formatting
with open('data/local_experiences/exit_experiences_v2.json', 'w') as f:
    json.dump(exit_data, f, indent=2)

print(f'✅ Fixed {fixed} exit experiences')
print(f'Total experiences: {len(exit_data["experiences"])}')
