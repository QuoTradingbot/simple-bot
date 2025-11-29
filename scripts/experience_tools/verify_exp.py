import json
from pathlib import Path

# Get project root (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
EXP_FILE = PROJECT_ROOT / 'experiences' / 'ES' / 'signal_experience.json'

data = json.load(open(EXP_FILE))
exps = data['experiences']

print(f'Total experiences: {len(exps)}')
print(f'Date range: {exps[0]["timestamp"]} to {exps[-1]["timestamp"]}')
wins = sum(1 for e in exps if e["pnl"] > 0)
print(f'Win rate: {wins / len(exps) * 100:.1f}%')
print(f'Avg PnL: ${sum(e["pnl"] for e in exps) / len(exps):.2f}')
print(f'Total PnL: ${sum(e["pnl"] for e in exps):.2f}')
