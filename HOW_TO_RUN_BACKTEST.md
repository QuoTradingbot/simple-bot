# How to Run Full Backtest

This guide explains how to run the full backtest with all requested configurations.

## Quick Start

```bash
# From the repository root
python3 run_full_backtest.py
```

That's it! The script will automatically:
1. ✅ Load 63 days of ES 1-minute data (63,599 bars)
2. ✅ Set confidence threshold to 50%
3. ✅ Set max contracts to 3
4. ✅ Remove daily trade cap (unlimited)
5. ✅ Enable dynamic contract sizing based on confidence
6. ✅ Activate all RL/ML learning systems
7. ✅ Generate detailed results

## Requirements

```bash
pip install pytz python-dotenv
```

## Configuration Details

The backtest is automatically configured with:

- **Data Source:** `data/historical_data/ES_1min.csv`
- **Date Range:** August 31 - November 3, 2025 (auto-detected from CSV)
- **Initial Capital:** $50,000
- **Confidence Threshold:** 50% (0.5)
- **Max Contracts:** 3
- **Daily Trade Limit:** 999 (effectively unlimited - bot decides)
- **RL Enabled:** Yes
- **Dynamic Sizing:** Yes (1/2/3 contracts based on confidence)

## Output Files

After running, you'll find:

1. **Console Output** - Real-time progress and summary
2. **logs/backtest_full.log** - Detailed execution log
3. **logs/backtest_full_report.txt** - Complete trade breakdown
4. **BACKTEST_RESULTS.md** - Comprehensive analysis document

## Expected Results

Based on the last run:
- Total P&L: +$1,635.00 (+3.27% return)
- Total Trades: 25
- Win Rate: 48.0%
- Sharpe Ratio: 2.28
- RL Learning: +69 new experiences

## Verification

To verify RL/ML is working, check the console output for:

```
RL BRAIN LEARNING SUMMARY
================================================================================
Signal Experiences: 1296 → 1339 (+43)
Exit Experiences: 1296 → 1322 (+26)
Total RL Growth: +69 new experiences
```

And in the logs, look for:
```
[RL DYNAMIC SIZING] HIGH confidence (82.0%) × Max 3 = 3 contracts
```

This confirms dynamic contract sizing based on confidence is working!

## Customization

To modify parameters, edit `run_full_backtest.py`:

```python
# Around line 57
bot_config.max_contracts = 3  # Change max contracts
bot_config.max_trades_per_day = 999  # Change trade limit
bot_config.rl_confidence_threshold = 0.5  # Change confidence threshold
```

## Troubleshooting

**Issue:** "Data file not found"
- **Solution:** Ensure `data/historical_data/ES_1min.csv` exists

**Issue:** "Configuration validation failed"
- **Solution:** The script sets `BOT_BACKTEST_MODE=true` automatically

**Issue:** "Import errors"
- **Solution:** Run from repository root, not from src/

## Full Documentation

See `BACKTEST_RESULTS.md` for:
- Complete trade-by-trade breakdown
- Daily P&L analysis
- RL/ML learning progress
- Performance metrics
- Top winning/losing trades
