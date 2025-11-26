# Backtesting Guide

## Overview

The backtesting system provides a clean, professional way to test your trading strategy against historical data. All settings are configured through `data/config.json` - this is the single entry point for backtest configuration.

## Running a Backtest

```bash
# Basic backtest - last 7 days
python dev/run_backtest.py --days 7

# Specific date range
python dev/run_backtest.py --start 2025-11-01 --end 2025-11-30

# Different symbol
python dev/run_backtest.py --days 30 --symbol MES

# Save report to file
python dev/run_backtest.py --days 30 --report backtest_results.txt
```

## Configuration

All trading parameters come from `data/config.json`. Key settings include:

- `account_size` - Starting balance for backtest
- `max_contracts` - Number of contracts per trade
- `risk_per_trade` - Risk percentage per trade
- `rl_exploration_rate` - RL exploration rate (0-1)
- `rl_confidence_threshold` - Minimum confidence for trades
- `use_rsi_filter`, `use_vwap_direction_filter`, etc. - Trading filters

**Note:** `max_trades_per_day` is a live-trading feature and is automatically disabled during backtesting.

## Output Format

### Header
Shows your configuration settings from config.json:
```
================================================================================
                        BACKTEST CONFIGURATION
================================================================================
Symbol:           ES
Period:           2025-11-19 to 2025-11-26
Starting Balance: $50,000.00
Max Contracts:    1

Trading Parameters (from config.json):
  Risk Per Trade:         1.2%
  Daily Loss Limit:       $10,000.00
  RL Exploration Rate:    100.0%
  RL Confidence Threshold: 0.0%
  Active Filters:         RSI(10), VWAP Direction, Volume
================================================================================
```

### Trade Log
Each trade shows:
- WIN/LOSS/B/E status (✓/✗/-)
- Side (LONG/SHORT)
- Entry date/time and price
- Exit time and price
- P&L in dollars
- Exit reason (stop_loss, target, trailing_stop, etc.)
- Duration in minutes
- RL confidence percentage
- Market regime (HIGH_VOL_CHOPPY, NORMAL_TRENDING, etc.)

Example:
```
✗ LOSS: SHORT 1x | Entry: Wed 11/19 10:23 @ $6682.50 -> Exit: 10:31 @ $6694.25 | P&L: $ -590.00 | stop_loss | 8min | Conf: 41% | HIGH_VOL_CHOPPY
✓ WIN : LONG  1x | Entry: Wed 11/19 14:30 @ $6700.00 -> Exit: 14:45 @ $6710.00 | P&L: $+500.00 | target    | 15min | Conf: 85% | NORMAL_TRENDING
```

### Summary
Comprehensive performance metrics including:
- Total trades, win/loss breakdown
- Win rate and profit factor
- P&L analysis (starting/ending balance, net P&L, avg win/loss)
- Risk metrics (max drawdown)
- Signal performance (approved/rejected signals)
- Execution time

## Tips

1. **Edit config.json** to change trading parameters
2. **Progress updates** are minimal - shown only periodically
3. **Signal spam** is suppressed - signals are tracked internally
4. **Regime information** shows market conditions during each trade
5. **No max_trades_per_day** limit in backtesting (live-only feature)

## Example Workflow

1. Edit `data/config.json` with your desired settings
2. Run backtest: `python dev/run_backtest.py --days 30`
3. Review trade log and summary
4. Adjust config.json settings as needed
5. Re-run backtest to compare results
