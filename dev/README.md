# Development Environment - VWAP Bounce Bot

This directory contains the **development and backtesting environment** for the VWAP Bounce Bot. It's completely separated from the production trading bot code in `src/` for clean architecture.

## 🎯 Purpose

The dev environment allows you to:
- **Test bot performance** on historical data without risking real money
- **Load and train Signal RL** models locally
- **Test pattern matching** and signal detection algorithms
- **Validate all trading logic** including regime detection, trade management, stops, targets
- **Follow production rules** for UTC maintenance windows and flatten times
- **Everything the live bot does** but in a safe, historical replay environment

## 📁 Structure

```
dev/
├── README.md              # This file
├── __init__.py            # Module initialization
├── backtesting.py         # Core backtesting framework
└── run_backtest.py        # Main entry point for running backtests
```

## 🚀 Running Backtests

### Basic Usage

```bash
# Run backtest for last 30 days
python dev/run_backtest.py --days 30

# Run backtest with specific date range
python dev/run_backtest.py --start 2024-01-01 --end 2024-01-31

# Run with tick-by-tick replay (requires tick data)
python dev/run_backtest.py --days 7 --use-tick-data

# Save results to a report file
python dev/run_backtest.py --days 30 --report backtest_results.txt
```

### Advanced Options

```bash
# Override symbol
python dev/run_backtest.py --days 30 --symbol ES

# Custom initial equity
python dev/run_backtest.py --days 30 --initial-equity 100000

# Custom data path
python dev/run_backtest.py --days 30 --data-path /path/to/historical/data

# Debug mode
python dev/run_backtest.py --days 7 --log-level DEBUG
```

## 🧠 Features

### Signal RL (Reinforcement Learning)
- Automatically loads signal experiences from `data/signal_experience.json`
- Trains and improves signal confidence scoring during backtests
- Saves learned experiences back to disk after backtest completion

### Pattern Matching
- Detects VWAP bounce patterns
- Uses same pattern recognition as production bot
- Filters signals based on pattern quality

### Regime Detection
- Identifies market regimes: HIGH_VOL_TRENDING, HIGH_VOL_CHOPPY, LOW_VOL_TRENDING, LOW_VOL_RANGING, NORMAL
- Adapts strategy parameters based on regime
- Uses production regime detection logic

### Trade Management
- **Stop Loss**: ATR-based dynamic stops
- **Profit Targets**: Risk-reward ratio-based targets
- **Breakeven**: Moves stop to breakeven after threshold
- **Trailing Stop**: Locks in profits as trade moves favorably
- **Time Decay**: Tightens stops as trade ages
- **Partial Exits**: Takes profits at multiple levels

### Production Rules
- **UTC Maintenance**: Respects 5:00-6:00 PM Eastern maintenance window
- **Flatten Time**: Auto-closes positions at 4:45 PM Eastern
- **Entry Windows**: Only enters trades during allowed hours
- **Friday Special**: Closes positions before weekend

## 📊 Output

After running a backtest, you'll see:

```
==========================================================
BACKTEST RESULTS
==========================================================
Total Trades: 42
Total P&L: $+3,450.00
Total Return: +6.90%
Win Rate: 64.29%
Average Win: $175.50
Average Loss: $-98.25
Profit Factor: 2.15
Sharpe Ratio: 1.82
Max Drawdown: $-850.00 (-1.70%)
Time in Market: 12.50%
Final Equity: $53,450.00

RL BRAIN LEARNING SUMMARY
==========================================================
[SIGNALS] 1247 total experiences
  Wins: 802 | Losses: 445 | Win Rate: 64.3%
==========================================================
```

## 🔧 Data Requirements

### Historical Data Format

Place your historical data in `data/historical_data/`:

**1-minute bars** (`ES_1min.csv`):
```csv
timestamp,open,high,low,close,volume
2024-01-01T00:00:00,4725.50,4726.25,4724.75,4725.75,1234
```

**Tick data** (optional, `ES_ticks.csv`):
```csv
timestamp,price,size
2024-01-01T00:00:00.123,4725.50,5
```

## 🎓 How It Works

1. **Loads Historical Data**: Reads CSV files with market data
2. **Initializes Bot State**: Sets up same state as production bot
3. **Loads RL Brain**: Imports signal experiences for decision making
4. **Replays Bars**: Processes each bar through production trading logic
5. **Executes Trades**: Simulates order fills with realistic slippage
6. **Tracks Performance**: Records all trades and calculates metrics
7. **Saves Learning**: Persists new RL experiences back to disk

## 🔗 Integration with Production

The dev environment imports and uses the **actual production bot code** from `src/`:
- `src/quotrading_engine.py` - Main trading logic
- `src/config.py` - Configuration management
- `src/signal_confidence.py` - Signal RL brain
- `src/regime_detection.py` - Market regime detection
- `src/monitoring.py` - Logging and monitoring

This ensures that backtests accurately reflect what the live bot will do.

## 🛡️ Safety

- **No Real Money**: Backtests only use historical data
- **No Broker API**: Completely independent of live trading systems
- **No Risk**: Safe environment for testing and development
- **Realistic**: Includes slippage, commissions, and all trading costs

## 📝 Notes

- Backtests are **for development and testing only**
- Past performance does not guarantee future results
- Always paper trade new strategies before going live
- Keep your RL experience files backed up

## 🤝 Contributing

When adding new features to the bot:
1. Test them in this backtest environment first
2. Verify they work on historical data
3. Check RL learning progression
4. Only then deploy to production

---

**Remember**: This is a development tool. The production bot lives in `src/` and is kept clean and simple for users.
