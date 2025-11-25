import json

# Current config
with open('data/config.json') as f:
    config = json.load(f)

print('=' * 80)
print('ITERATION 3 SETTINGS COMPARISON')
print('=' * 80)

print('\nCURRENT SETTINGS:')
print('-' * 80)
print('VWAP BANDS:')
print(f'  Band 1 (Warning):  {config.get("vwap_std_dev_1", "NOT SET - code default 2.5")}')
print(f'  Band 2 (Entry):    {config.get("vwap_std_dev_2", "NOT SET - code default 2.1")}')
print(f'  Band 3 (Exit/Stop): {config.get("vwap_std_dev_3", "NOT SET - code default 3.7")}')

print('\nRSI SETTINGS:')
print(f'  RSI Period:        {config.get("rsi_period", "NOT SET - code default 14")}')
print(f'  RSI Oversold:      {config.get("rsi_oversold", "NOT SET - code default 35")}')
print(f'  RSI Overbought:    {config.get("rsi_overbought", "NOT SET - code default 65")}')

print('\nFILTERS:')
print(f'  Use RSI Filter:    {config.get("use_rsi_filter", "NOT SET - code default True")}')
print(f'  Use Volume Filter: {config.get("use_volume_filter", "NOT SET - code default True")}')
print(f'  Use VWAP Direction: {config.get("use_vwap_direction_filter", "NOT SET - code default False")}')
print(f'  Use Trend Filter:  {config.get("use_trend_filter", "NOT SET - code default False")}')

print('\nRL SETTINGS:')
print(f'  Confidence Threshold: {config.get("rl_confidence_threshold")}')
print(f'  Max Contracts: {config.get("max_contracts")}')
print(f'  Max Trades/Day: {config.get("max_trades_per_day")}')

print('\n' + '=' * 80)
print('ITERATION 3 TARGET SETTINGS (from commit 12fc9c6 - Nov 8):')
print('=' * 80)
print('VWAP BANDS:')
print('  Band 1: 2.5  (Warning zone)')
print('  Band 2: 2.1  (ENTRY ZONE) ✓')
print('  Band 3: 3.7  (EXIT/STOP ZONE) ✓')
print('\nRSI SETTINGS:')
print('  Period: 10  ← NEEDS TO BE 10 (not 14)')
print('  Oversold: 35  (Long entries) ✓')
print('  Overbought: 65  (Short entries) ✓')
print('\nFILTERS:')
print('  Use RSI Filter: True ✓')
print('  Use VWAP Direction: True ← SHOULD BE True')
print('  Use Trend Filter: False ✓')
print('\n' + '=' * 80)

print('\nMISSING/DIFFERENT SETTINGS:')
print('-' * 80)

mismatches = []
if config.get("rsi_period", 14) != 10:
    mismatches.append("❌ RSI Period should be 10 (currently using code default 14)")
    
if config.get("use_vwap_direction_filter", False) != True:
    mismatches.append("❌ use_vwap_direction_filter should be True (currently False)")

if not mismatches:
    print("✅ All Iteration 3 settings match!")
else:
    for m in mismatches:
        print(m)

print('=' * 80)
