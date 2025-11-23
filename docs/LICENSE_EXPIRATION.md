# License Expiration Handling

## Overview

The QuoTrading bot includes comprehensive license expiration management with:

1. **Pre-Expiration Warnings** - Alerts at 7 days, 24 hours, and 2 hours before expiration
2. **Grace Period Protection** - Never abandons active positions
3. **Near-Expiry Safety** - Blocks new trades within 2 hours of expiration
4. **Intelligent Shutdown** - Stops at optimal market times

This ensures customers are well-informed and never have positions abandoned mid-trade.

## Pre-Expiration Warning System

### 7 Days Before Expiration
- **Warning**: "Your license expires in 7 days"
- **Action**: Notification sent, trading continues normally
- **Purpose**: Early awareness for renewal

### 24 Hours Before Expiration  
- **Warning**: "URGENT: License expires in 24 hours"
- **Action**: Notification sent, trading continues normally
- **Purpose**: Final reminder before near-expiry mode

### 2 Hours Before Expiration (Near-Expiry Mode)
- **Warning**: "NEAR EXPIRY MODE - New trades blocked"
- **Action**: 
  - NEW trades blocked immediately
  - Can still manage existing positions
  - Notification sent
- **Purpose**: Prevent opening trades that won't have time to develop

### At Expiration
- **With No Position**: Stop immediately
- **With Active Position**: Enter grace period (manage until close)

## How It Works

### Periodic License Validation

- The bot checks license validity **every 5 minutes** during trading hours
- Validates against the cloud API to ensure the license is still active
- Receives expiration date, days remaining, and hours remaining
- Does not stop trading on temporary network errors (continues to retry)

### Grace Period for Active Positions

**The Problem**: Immediately stopping on expiration could abandon active positions → customer loses money

**The Solution**: When a license expires during an active trade:

1. **Grace Period Activates**
   - Bot continues managing the current position
   - Uses normal exit rules (target/stop/time-based)
   - Blocks NEW trade entries
   - Shows message: "License expired. Closing position safely before stopping..."

2. **Position Closes Naturally**
   - Bot manages position until it hits target, stop, or time exit
   - Could be 5 minutes or 2 hours depending on trade
   - Safest approach for the customer

3. **Trading Stops After Position Closes**
   - Once position is flat, trading is disabled
   - Emergency stop flag is set
   - Customer receives notification with final P&L

### Graceful Shutdown Strategy

When a license expiration is detected, the bot chooses the best approach:

#### 1. **Grace Period** (Active Position)
- **When**: License expires while position is open
- **Action**: 
  - Continue managing position with normal exit rules
  - Block new trade entries
  - Send grace period notification
  - Wait for position to close naturally (target/stop/time)
  - After position closes: Disable trading and send final notification

#### 2. **Immediate Stop** (No Position)
- **When**: License expires with no active position
- **Action**:
  - Immediately disable new trade entries
  - Send notification alert
  - Log expiration reason

#### 3. **Friday Market Close** (Delayed)
- **When**: Expires on Friday before market close (before 5:00 PM ET)
- **Action**:
  - Continue trading until Friday market close (5:00 PM ET)
  - Close any positions at market close
  - Disable trading for the weekend

#### 4. **Maintenance Window** (Delayed)
- **When**: Expires during flatten mode (4:45-5:00 PM ET, Monday-Thursday)
- **Action**:
  - Wait until maintenance window starts (5:00 PM ET)
  - Close positions with other daily maintenance activities
  - Minimizes disruption during active trading

## Example Scenarios

**Scenario 1: 7 Days Before Expiration**
```
Day -7 - License check detects 7 days remaining
       - Log warning message
       - Send notification
       - Trading continues normally
```

**Scenario 2: 24 Hours Before Expiration**
```
Day -1 - License check detects 24 hours remaining
       - Log URGENT warning message
       - Send notification
       - Trading continues normally
```

**Scenario 3: 2 Hours Before Expiration (Near-Expiry Mode)**
```
-2h - License check detects 2 hours remaining
    - Enter NEAR EXPIRY MODE
    - Block NEW trades
    - Can manage existing positions
    - Log critical warning
    - Send notification
```

**Scenario 4: Expires Wednesday 2 PM with Active Position (GRACE PERIOD)**
```
14:00 - License check detects expiration
14:00 - Position is ACTIVE (LONG 1 @ $5000)
14:00 - Enter GRACE PERIOD mode
14:00 - Block new trades
14:00 - Send grace period notification
14:00-14:30 - Continue managing position
14:30 - Position hits target at $5025
14:30 - Close position (+$25 profit)
14:30 - Grace period ends
14:30 - Disable trading
14:30 - Send final notification
```

**Scenario 5: Expires Wednesday 2 PM with No Position (IMMEDIATE STOP)**
```
14:00 - License check detects expiration
14:00 - No active position
14:00 - Disable trading immediately
14:00 - Send notification
```

**Scenario 6: Expires Friday 3 PM (DELAYED STOP)**
```
15:00 - License check detects expiration
15:00 - Flag set: stop_at_market_close
15:00-17:00 - Continue trading normally (if > 2h until expiry)
17:00 - Market closes, close any positions
17:00 - Disable trading
17:00 - Send notification
```

## Customer Impact

### Benefits of Pre-Expiration Warnings
- **Early Awareness**: 7-day warning gives time to renew
- **Final Reminder**: 24-hour warning ensures customer knows
- **Safety Mode**: 2-hour near-expiry blocks risky new positions
- **No Surprises**: Multiple notifications before any action taken

### Benefits of Grace Period
- **No Abandoned Positions**: Position always closes via normal exit rules
- **No Forced Market Exits**: Uses target/stop/time-based exits (not panic market orders)
- **Protects Customer P&L**: Prevents losses from premature forced exits
- **Professional Handling**: Manages positions safely and intelligently

### Benefits of Near-Expiry Mode
- **Prevents New Losses**: Won't open trades that can't properly develop
- **Safe Position Management**: Existing positions still managed normally
- **Clear Communication**: Logs and notifications explain the restriction

### Clear Communication
- **7-Day Warning**: "License expires in 7 days - please renew"
- **24-Hour Warning**: "URGENT: License expires in 24 hours"
- **2-Hour Warning**: "NEAR EXPIRY MODE - new trades blocked"
- **Grace Period Alert**: Explains position is being managed until close
- **Final Alert**: Shows final P&L and confirms trading stopped
- **Logging**: Clear messages about each stage

## Testing

Run the test suites to validate expiration handling and pre-expiration warnings:

```bash
# Test pre-expiration warnings (7 days, 24 hours, 2 hours)
python tests/test_pre_expiration_warnings.py

# Test grace period logic
python tests/test_grace_period.py

# Test expiration timing
python tests/test_expiration_simple.py
```

All scenarios are tested for correctness.

## API Changes

The Flask API now returns expiration information:

```python
{
    "license_valid": true,
    "license_expiration": "2024-12-31T23:59:59",
    "days_until_expiration": 7,
    "hours_until_expiration": 168.5,
    "message": "Valid premium license"
}
```

This enables the bot to:
- Display countdown to expiration
- Trigger warnings at appropriate times
- Block risky trades near expiration
- Provide clear user communication
