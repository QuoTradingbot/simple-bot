# Experience Files - Multi-Symbol RL Learning

This folder contains symbol-specific experience data for the RL (Reinforcement Learning) brain.

## Structure

Each symbol has its own folder containing:
- `signal_experience.json` - Historical trade experiences for pattern matching

```
experiences/
├── ES/
│   └── signal_experience.json
├── MES/
│   └── signal_experience.json
├── NQ/
│   └── signal_experience.json
└── MNQ/
    └── signal_experience.json
```

## How It Works

### Live Mode
- **Reads** from local symbol-specific folder for RL pattern matching
- **Saves** to cloud only (does not save locally)
- Example: When trading ES, reads from `experiences/ES/signal_experience.json`

### Backtest Mode
- **Reads** from local symbol-specific folder
- **Saves** to local symbol-specific folder
- Example: When backtesting NQ, reads and writes to `experiences/NQ/signal_experience.json`

## Multi-Symbol Support

The RL brain automatically detects which symbol is being traded and loads the corresponding experience file. This prevents mixing experiences from different symbols which could lead to incorrect pattern matching.

For example:
- ES experiences → `experiences/ES/`
- MES experiences → `experiences/MES/`
- NQ experiences → `experiences/NQ/`
- MNQ experiences → `experiences/MNQ/`

## Data Flow

### Backtesting (Developer)
1. Run backtest for symbol (e.g., ES)
2. RL brain reads from `experiences/ES/signal_experience.json`
3. RL brain saves results back to `experiences/ES/signal_experience.json`
4. Developer distributes updated files to users weekly

### Live Trading (Users)
1. User selects symbol in GUI (e.g., NQ)
2. RL brain reads from `experiences/NQ/signal_experience.json` for pattern matching
3. RL brain makes decision based on historical patterns
4. Trade results sent to cloud for data collection
5. Local files NOT modified (cloud handles all saves)

## Important Notes

- Each symbol must have its own folder
- Experience files are NOT interchangeable between symbols
- Users receive weekly updates to experience files from developer
- Live mode never modifies local files (cloud-only saving)
