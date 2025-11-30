# Maintenance and Weekend Idle Mode - Implementation Summary

## Overview
The bot now properly handles maintenance windows and weekends by staying ON but going IDLE during these periods. The bot NEVER turns off automatically - only Ctrl+C stops it.

## Key Features

### 1. Continuous Operation
- **Bot stays running 24/7** - never automatically shuts down
- Only way to stop: Press Ctrl+C (user-initiated shutdown)
- During maintenance/weekends: Bot goes IDLE (disconnects broker, saves resources)
- Auto-reconnects when market reopens

### 2. Idle Periods

#### Weekday Maintenance (Mon-Thu)
- **Time:** 4:45 PM - 6:00 PM ET daily
- **Duration:** 1 hour 15 minutes
- **Behavior:**
  - Bot flattens all positions at 4:45 PM
  - Disconnects from broker
  - Shows: "🔧 MAINTENANCE - GOING IDLE"
  - Periodic status: "⏸️ MAINTENANCE IN PROGRESS - Bot idle, will resume when market reopens" (every 5 min)
  - Auto-reconnects at 6:00 PM ET

#### Weekend (Fri-Sun)
- **Time:** Friday 4:45 PM - Sunday 6:00 PM ET
- **Duration:** ~50 hours
- **Behavior:**
  - Bot flattens all positions Friday at 4:45 PM
  - Disconnects from broker
  - Shows: "🔧 WEEKEND - GOING IDLE"
  - Periodic status: "⏸️ WEEKEND IN PROGRESS - Bot idle, will resume when market reopens" (every 5 min)
  - Auto-reconnects Sunday at 6:00 PM ET

### 3. Daily Reset at 6:00 PM ET

After maintenance completes (6:00 PM ET), the bot resets:

#### Daily P&L Reset
- Daily P&L resets to $0.00
- Trade count resets to 0
- Daily loss limit flag cleared

#### VWAP Reset
- VWAP data cleared
- 1-minute bars cleared (fresh calculation starts)
- VWAP bands recalculate from live data
- 15-minute trend bars continue (trend carries overnight)

### 4. Profit-Based Trade Limits

The bot now rewards profitable trading by allowing bonus trades:

- **Base Limit:** CONFIG["max_trades_per_day"] (e.g., 10 trades)
- **Bonus Trades:** 1 additional trade per $100 profit
- **Cap:** Maximum 50% more trades (e.g., 10 → max 15 trades)

#### Examples:
```
$0 profit     → 10 trades (base limit)
$100 profit   → 11 trades (base + 1 bonus)
$250 profit   → 12 trades (base + 2 bonus)
$500 profit   → 15 trades (base + 5 bonus, capped at 50%)
$1000 profit  → 15 trades (base + 5 bonus, capped at 50%)
```

### 5. Server Time Synchronization

- **Primary:** Azure cloud API provides accurate time
- **Fallback:** Local Eastern Time if cloud unreachable
- **Timezone:** US/Eastern (handles DST automatically)
- **Time checks:** Every 30 seconds via timer manager

## Implementation Details

### Code Changes

1. **check_broker_connection()** in `quotrading_engine.py` (lines 725-806)
   - Enhanced idle mode detection
   - Separate MAINTENANCE vs WEEKEND handling
   - Periodic status messages every 5 minutes
   - Clear reconnection messages

2. **can_generate_signal()** in `quotrading_engine.py` (lines 2377-2420)
   - Profit-based trade limit calculation
   - Dynamic limit based on current P&L
   - Clear logging of base + bonus trades

3. **perform_daily_reset()** in `quotrading_engine.py` (lines 6331-6373)
   - Enhanced reset messages
   - Clear indication of what's being reset
   - Loss limit alert flag reset

4. **perform_vwap_reset()** in `quotrading_engine.py` (lines 6271-6295)
   - Clearer VWAP reset messages
   - Shows what's being cleared vs continuing

### Testing

Run the test script to verify behavior:
```bash
python3 test_idle_mode.py
```

This tests:
- Profit-based trade limit calculation
- Maintenance/weekend time window detection
- Daily reset timing (6:00 PM ET)

## User Experience

### Normal Trading Day
```
06:00 PM ET - Market opens (Sunday-Thursday)
   ✓ Bot reconnects
   ✓ Daily limits reset
   ✓ VWAP resets
   ✓ Trading enabled

04:00 PM ET - Entry cutoff (next day)
   → No new positions allowed
   → Can hold existing positions

04:45 PM ET - Forced flatten
   → All positions closed
   → Bot disconnects
   → Goes IDLE

06:00 PM ET - Market reopens
   → Cycle repeats
```

### Weekend
```
Friday 04:45 PM ET - Weekend starts
   ✓ Positions flattened
   ✓ Bot disconnects
   ✓ Shows "WEEKEND IN PROGRESS" every 5 min

Sunday 06:00 PM ET - Market reopens
   ✓ Bot reconnects
   ✓ Daily limits reset
   ✓ VWAP resets
   ✓ Trading enabled
```

## Configuration

No new configuration needed! The bot uses existing settings:
- `forced_flatten_time`: 4:45 PM ET (from config)
- `shutdown_time`: 6:00 PM ET (market reopen time)
- `vwap_reset_time`: 6:00 PM ET (daily VWAP reset)
- `max_trades_per_day`: Base trade limit (user configurable)

## Notes

- Bot uses cloud server (Azure) for accurate time
- No parallel processes needed (single event loop)
- All logging follows existing patterns
- No breaking changes to existing functionality
- Fully backward compatible with existing configs
