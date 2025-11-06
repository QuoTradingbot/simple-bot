# GUI Navigation Flow Documentation

## Overview
The QuoTrading Launcher has a 2-screen streamlined onboarding flow.

## Screen Flow

```
┌─────────────────────────────────────────────────────────────┐
│                  Screen 0: Broker Setup                     │
│     Broker Credentials + QuoTrading API Key + Account       │
│                                                              │
│  • Account Type selection (Prop Firm / Live Broker)         │
│  • Broker dropdown (TopStep / Tradovate)                    │
│  • QuoTrading API Key field (hidden)                        │
│  • Account Size dropdown (50k, 100k, 150k, 200k, 250k)      │
│  • Broker API Token field (hidden)                          │
│  • Broker Username/Email field                              │
│  • [NEXT →] button                                           │
└──────────────────────────────┬──────────────────────────────┘
                               │ NEXT
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             Screen 1: Trading Controls                      │
│          Symbol Selection & Risk Parameters                  │
│                                                              │
│  • Trading symbols (checkboxes)                              │
│  • Account size                                              │
│  • Risk per trade (%)                                        │
│  • Daily loss limit                                          │
│  • Max contracts                                             │
│  • Max trades per day                                        │
│  • [← BACK] button                                           │
│  • [START BOT →] button                                      │
└─────────────────────────────────────────────────────────────┘
```

## Navigation Rules

### Forward Navigation
- Screen 0 has a "NEXT" button
- Screen 1 has a "START BOT" button
- Clicking validates the current screen's inputs
- If validation passes, the user proceeds
- If validation fails, an error message is shown

### Backward Navigation
- Screen 1 has a "BACK" button
- Clicking "BACK" returns to Screen 0
- No validation is performed when going back
- Previously entered data is preserved

## Validation Flow

### Screen 0 → Screen 1
- Validates broker selection
- Validates QuoTrading API key is not empty
- Validates account size is selected
- Validates broker API token is not empty
- Validates broker username is not empty
- Makes broker validation call
- On success, saves all credentials and proceeds to trading screen
- On failure, shows error message

### Screen 1 → Start Bot
- Validates at least one trading symbol is selected
- Validates account size is a positive number
- Validates daily loss limit is a positive number
- Creates .env file with all settings
- Launches the trading bot
- Closes the launcher GUI

## Key Features

1. **Streamlined Flow**: Only 2 screens for fastest setup
2. **All-in-One Broker Setup**: QuoTrading API key, broker credentials, and account size on first screen
3. **Limited Broker Options**: Only TopStep (Prop Firm) and Tradovate (Live Broker) supported
4. **Account Size Selection**: Pre-defined account sizes (50k, 100k, 150k, 200k, 250k)
5. **Clear Navigation**: Each screen has visible navigation buttons
6. **Data Persistence**: All entered data is saved to config.json
7. **Validation Feedback**: Clear error messages guide the user
8. **Admin Bypass**: Admin master key (QUOTRADING_ADMIN_MASTER_2025) skips validation
9. **Back Navigation**: Users can go back to correct mistakes without losing data

## Changes Made

- Simplified to 2-screen flow (from 3 screens)
- Removed username/password login screen entirely
- Made broker setup the first screen (Screen 0)
- Added QuoTrading API key to broker screen
- Added account size dropdown (50k, 100k, 150k, 200k, 250k)
- Limited brokers to TopStep and Tradovate only
- Trading controls is now Screen 1 (was Screen 2)
- Updated all navigation and validation flow
- Simplified .env file generation
