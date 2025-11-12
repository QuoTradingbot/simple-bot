import json
from datetime import datetime

print("=" * 80)
print("MIGRATING EXPERIENCE FILES TO INCLUDE NEW FIELDS")
print("=" * 80)

# MIGRATE SIGNAL EXPERIENCES
print("\n[1/2] Migrating signal_experiences_v2.json...")
with open('data/local_experiences/signal_experiences_v2.json', 'r') as f:
    signal_data = json.load(f)

migrated_count = 0
for exp in signal_data['experiences']:
    # Add new psychological fields if missing
    if 'cumulative_pnl_at_entry' not in exp:
        exp['cumulative_pnl_at_entry'] = 0.0
    if 'consecutive_wins' not in exp:
        exp['consecutive_wins'] = 0
    if 'consecutive_losses' not in exp:
        exp['consecutive_losses'] = 0
    if 'drawdown_pct_at_entry' not in exp:
        exp['drawdown_pct_at_entry'] = 0.0
    if 'time_since_last_trade_mins' not in exp:
        exp['time_since_last_trade_mins'] = 0.0
    
    # Add new market context fields if missing
    if 'session' not in exp:
        # Infer from hour if possible
        hour = exp.get('hour', 12)
        if 18 <= hour or hour < 3:
            exp['session'] = 'Asia'
        elif 3 <= hour < 8:
            exp['session'] = 'London'
        else:
            exp['session'] = 'NY'
    
    if 'entry_slippage_ticks' not in exp:
        exp['entry_slippage_ticks'] = 0.0
    if 'commission_cost' not in exp:
        exp['commission_cost'] = 0.0
    if 'bid_ask_spread_ticks' not in exp:
        exp['bid_ask_spread_ticks'] = 0.5
    
    # Add new market context fields (round 2)
    if 'trend_strength' not in exp:
        exp['trend_strength'] = 0.0
    if 'sr_proximity_ticks' not in exp:
        exp['sr_proximity_ticks'] = 0.0
    if 'trade_type' not in exp:
        # Infer from signal type if possible
        signal = exp.get('signal', 'LONG')
        exp['trade_type'] = 'reversal'  # Most VWAP bounce signals are reversals
    
    migrated_count += 1

# Save migrated signal experiences
with open('data/local_experiences/signal_experiences_v2.json', 'w') as f:
    json.dump({
        'experiences': signal_data['experiences'],
        'count': len(signal_data['experiences']),
        'version': '2.1',  # Bump version
        'last_updated': datetime.now().isoformat()
    }, f, indent=2)

print(f"✅ Migrated {migrated_count} signal experiences")

# MIGRATE EXIT EXPERIENCES
print("\n[2/2] Migrating exit_experiences_v2.json...")
with open('data/local_experiences/exit_experiences_v2.json', 'r') as f:
    exit_data = json.load(f)

migrated_count = 0
for exp in exit_data['experiences']:
    # Add in-trade tracking fields if missing
    if 'max_r_achieved' not in exp:
        exp['max_r_achieved'] = exp.get('r_multiple', 0.0)
    if 'min_r_achieved' not in exp:
        exp['min_r_achieved'] = min(0.0, exp.get('r_multiple', 0.0))
    if 'exit_param_update_count' not in exp:
        exp['exit_param_update_count'] = 0
    if 'stop_adjustment_count' not in exp:
        exp['stop_adjustment_count'] = 0
    if 'breakeven_activation_bar' not in exp:
        exp['breakeven_activation_bar'] = 0
    if 'trailing_activation_bar' not in exp:
        exp['trailing_activation_bar'] = 0
    if 'bars_until_breakeven' not in exp:
        exp['bars_until_breakeven'] = 0
    if 'bars_until_trailing' not in exp:
        exp['bars_until_trailing'] = 0
    if 'breakeven_activated' not in exp:
        exp['breakeven_activated'] = False
    if 'trailing_activated' not in exp:
        exp['trailing_activated'] = False
    if 'exit_param_updates' not in exp:
        exp['exit_param_updates'] = []
    if 'stop_adjustments' not in exp:
        exp['stop_adjustments'] = []
    
    # Add execution quality fields if missing
    if 'slippage_ticks' not in exp:
        exp['slippage_ticks'] = 0.0
    if 'commission_cost' not in exp:
        exp['commission_cost'] = 0.0
    if 'bid_ask_spread_ticks' not in exp:
        exp['bid_ask_spread_ticks'] = 0.5
    
    # Add market context fields if missing
    if 'session' not in exp:
        # Try to get from market_state
        if 'market_state' in exp and 'hour' in exp['market_state']:
            hour = exp['market_state']['hour']
            if 18 <= hour or hour < 3:
                exp['session'] = 'Asia'
            elif 3 <= hour < 8:
                exp['session'] = 'London'
            else:
                exp['session'] = 'NY'
        else:
            exp['session'] = 'NY'
    
    if 'volume_at_exit' not in exp:
        exp['volume_at_exit'] = 0.0
    if 'volatility_regime_change' not in exp:
        exp['volatility_regime_change'] = False
    
    # Add exit quality fields if missing
    if 'time_in_breakeven_bars' not in exp:
        exp['time_in_breakeven_bars'] = 0
    if 'rejected_partial_count' not in exp:
        exp['rejected_partial_count'] = 0
    if 'stop_hit' not in exp:
        # Infer from exit_reason
        exit_reason = exp.get('exit_reason', '')
        exp['stop_hit'] = 'stop' in exit_reason.lower()
    
    migrated_count += 1

# Save migrated exit experiences
with open('data/local_experiences/exit_experiences_v2.json', 'w') as f:
    json.dump({
        'experiences': exit_data['experiences'],
        'count': len(exit_data['experiences']),
        'version': '2.1',  # Bump version
        'last_updated': datetime.now().isoformat()
    }, f, indent=2)

print(f"✅ Migrated {migrated_count} exit experiences")

print("\n" + "=" * 80)
print("MIGRATION COMPLETE")
print("=" * 80)
print("\nAll existing experiences now have the new fields.")
print("New backtests will populate these fields with real data.")
print("\nRun: python check_missing_fields.py to verify")
