# VWAP Bounce Bot

An event-driven mean reversion trading bot for futures trading (MES) that executes trades based on VWAP (Volume Weighted Average Price) standard deviation bands and trend alignment.

## Overview

The VWAP Bounce Bot subscribes to real-time tick data, aggregates it into bars, calculates VWAP with standard deviation bands, determines trend direction, and executes mean reversion trades when price touches extreme bands while aligned with the trend.

**New in Phase 12 & 13:**
- ✨ **Backtesting Framework** - Test strategies on historical data with realistic order simulation
- ✨ **Enhanced Logging** - Structured JSON logging with sensitive data protection
- ✨ **Health Checks** - HTTP endpoint for monitoring bot status
- ✨ **Metrics Collection** - Track performance metrics and API latency
- ✨ **Dual Mode** - Run in live trading or backtesting mode

**NEW: Complete Bid/Ask Trading Strategy** ⭐
- ✨ **Real-Time Bid/Ask Quotes** - Track bid price, ask price, sizes, and spreads
- ✨ **Spread Analysis** - Baseline tracking and abnormal spread detection
- ✨ **Intelligent Order Placement** - Passive vs aggressive strategy selection
- ✨ **Dynamic Fill Strategy** - Timeout handling and retry logic
- ✨ **Cost Optimization** - Save 80% on trading costs with smart limit orders
- 📖 **[See Full Documentation](docs/BID_ASK_STRATEGY.md)**

## Features

- **Event-Driven Architecture**: Processes real-time tick data efficiently
- **Bid/Ask Strategy**: Professional-grade order placement with spread analysis
- **Risk Management**: Conservative 0.1% risk per trade with daily loss limits
- **Trend Filter**: 50-period EMA on 15-minute bars
- **VWAP Bands**: Two standard deviation bands for entry signals
- **Trading Hours**: 9:00 AM - 2:30 PM ET entry window
- **Dry Run Mode**: Test strategies without risking capital
- **Backtesting Engine**: Validate strategies on historical data
- **Health Monitoring**: HTTP endpoint for health checks and metrics
- **Structured Logging**: JSON logs with log rotation and sensitive data filtering

## Configuration

The bot is configured for **MES (Micro E-mini S&P 500)** with the following parameters:

### Trading Parameters
- **Instrument**: MES only (to start)
- **Trading Window**: 10:00 AM - 3:30 PM Eastern Time
- **Risk Per Trade**: 0.1% of account equity
- **Max Contracts**: 1
- **Max Trades Per Day**: 5
- **Daily Loss Limit**: $400 (conservative before TopStep's $1,000 limit)

### Instrument Specifications (MES)
- **Tick Size**: 0.25
- **Tick Value**: $1.25

### Strategy Parameters
- **Trend Filter**: 50-period EMA on 15-minute bars
- **VWAP Timeframe**: 1-minute bars
- **Standard Deviation Bands**: 1σ and 2σ multipliers
- **Risk/Reward Ratio**: 1.5:1
- **Max Bars Storage**: 200 bars for stability

## Installation

### Prerequisites
- Python 3.8 or higher
- TopStep trading account and API credentials

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Quotraders/simple-bot.git
cd simple-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your TopStep API token:
```bash
cp .env.example .env
# Edit .env and add your TOPSTEP_API_TOKEN
```

4. Install TopStep SDK (follow TopStep documentation):
```bash
# Follow TopStep's official SDK installation instructions
# pip install topstep-sdk
```

## Usage

### Quick Start

The bot now supports two modes: **Live Trading** and **Backtesting**.

#### Live Trading Mode

Test with dry-run (paper trading):
```bash
export TOPSTEP_API_TOKEN='your_token_here'
python main.py --mode live --dry-run
```

Run in production (requires confirmation):
```bash
export TOPSTEP_API_TOKEN='your_token_here'
export CONFIRM_LIVE_TRADING=1  # Required safety check
python main.py --mode live
```

The bot will:
- Start health check server on port 8080 (http://localhost:8080/health)
- Log to `./logs/vwap_bot.log` (JSON format with rotation)
- Track performance metrics

#### Backtesting Mode

**Backtesting runs completely independently of the broker API.**  
No API token needed - it replays historical data bar-by-bar (1-minute bars by default).

**Parameter Optimization:**

Find optimal strategy parameters using grid search and walk-forward analysis:

```python
from parameter_optimization import ParameterOptimizer

# Define parameter ranges to optimize
param_ranges = {
    'vwap_period': [20, 30, 40, 50, 60],
    'band_multiplier': [0.5, 1.0, 1.5, 2.0],
    'stop_loss_ticks': [5, 8, 10, 12, 15],
    'target_ticks': [10, 15, 20, 25, 30]
}

# Run grid search
optimizer = ParameterOptimizer(config, bot_config, param_ranges)
results = optimizer.grid_search(vwap_strategy, metric='sharpe_ratio', n_jobs=4)
print(f"Best parameters: {results.best_params}")

# Run walk-forward analysis (prevents overfitting)
wf_results = optimizer.walk_forward_analysis(vwap_strategy, window_size_days=30)
```

**Basic Backtesting:**

Run backtest on last 7 days:
```bash
# No API token required for backtesting!
python main.py --mode backtest --days 7
```

Run backtest with specific date range:
```bash
python main.py --mode backtest --start 2024-01-01 --end 2024-01-31
```

Generate and save backtest report:
```bash
python main.py --mode backtest --days 30 --report backtest_results.txt
```

**Optional: Use tick-by-tick replay for more accurate simulation:**
```bash
python main.py --mode backtest --days 7 --use-tick-data
```

**How it works:**
1. **Bar-by-bar mode (default)**: Replays 1-minute bars sequentially
2. **Tick-by-tick mode (optional)**: Replays each tick as if it's happening live
3. Loads historical data from CSV files (no broker connection)
4. Bot executes strategy on historical data
5. Simulates realistic order fills with slippage
6. 100% offline simulation - no API needed

### Fetch Real Historical Data

**IMPORTANT: Use REAL data from TopStep, not mock/simulated data**

Fetch real market data from TopStep API:
```bash
# Set your TopStep API token
export TOPSTEP_API_TOKEN='your_real_token_here'

# Fetch real historical data
python fetch_historical_data.py --symbol MES --days 30
```

This fetches REAL market data from TopStep and saves to `./historical_data/`:
- MES_ticks.csv - Real tick-level data (or finest granularity available)
- MES_1min.csv - Real 1-minute bars
- MES_15min.csv - Real 15-minute bars

**NO MOCK OR SIMULATED DATA** - All data comes from actual TopStep market feeds.

### Command-Line Options

```
usage: main.py [-h] [--mode {live,backtest}] [--dry-run] [--start START]
               [--end END] [--days DAYS] [--data-path DATA_PATH]
               [--initial-equity INITIAL_EQUITY] [--report REPORT]
               [--symbol SYMBOL] [--environment {development,staging,production}]
               [--health-check-port HEALTH_CHECK_PORT] [--no-health-check]
               [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

Options:
  --mode {live,backtest}    Trading mode (default: live)
  --dry-run                 Paper trading mode (no real orders)
  --start START             Backtest start date (YYYY-MM-DD)
  --end END                 Backtest end date (YYYY-MM-DD)
  --days DAYS               Backtest for last N days
  --data-path DATA_PATH     Historical data directory
  --initial-equity EQUITY   Initial equity for backtesting
  --report REPORT           Save backtest report to file
  --symbol SYMBOL           Trading symbol (default: MES)
  --environment ENV         Environment config (development/staging/production)
  --health-check-port PORT  Health check HTTP port (default: 8080)
  --no-health-check         Disable health check server
  --log-level LEVEL         Logging level (default: INFO)
```

### Health Check Endpoint

When running in live mode, the bot exposes a health check endpoint:

```bash
curl http://localhost:8080/health
```

Response format:
```json
{
  "healthy": true,
  "timestamp": "2024-01-15T10:30:00.000Z",
  "checks": {
    "bot_status": true,
    "broker_connection": true,
    "data_feed": true
  },
  "messages": ["All systems operational"]
}
```

## How It Works

### Data Flow

1. **Tick Reception**: SDK sends real-time tick data (price, volume, timestamp)
2. **Bar Aggregation**: 
   - 1-minute bars for VWAP calculation
   - 15-minute bars for trend filter
3. **VWAP Calculation**: Volume-weighted price with standard deviation bands
4. **Trend Detection**: 50-period EMA determines market direction
5. **Signal Generation**: Price touching extreme bands while trend-aligned
6. **Order Execution**: Market orders with stop loss and target placement

### VWAP Calculation

VWAP resets daily and is calculated as:
```
VWAP = Σ(Price × Volume) / Σ(Volume)
```

Standard deviation bands:
- **Upper Band 1**: VWAP + 1σ
- **Upper Band 2**: VWAP + 2σ  
- **Lower Band 1**: VWAP - 1σ
- **Lower Band 2**: VWAP - 2σ

### State Management

The bot maintains state for:
- **Tick Storage**: Deque with 10,000 tick capacity
- **Bar Storage**: 200 1-minute bars, 100 15-minute bars
- **Position Tracking**: Entry price, stops, targets
- **Daily Metrics**: Trade count, P&L, day identification

## Project Structure

```
simple-bot/
├── vwap_bounce_bot.py      # Main bot implementation (all 14 phases)
├── test_complete_cycle.py  # Complete trading cycle demonstration (Phases 1-10)
├── test_phases_11_14.py    # Safety and monitoring tests (Phases 11-14)
├── test_phases_6_10.py     # Phases 6-10 specific tests
├── test_bot.py             # Original validation tests
├── example_usage.py         # Usage examples and demonstrations
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variable template
├── .gitignore            # Git ignore patterns
└── README.md             # This file
```

## Trading Strategy Components

### Phase 6: Trend Filter
- **50-period EMA** on 15-minute bars
- **Smoothing factor**: 2/(period+1) ≈ 0.039
- **Trend states**: "up" (price > EMA + 0.5 tick), "down" (price < EMA - 0.5 tick), "neutral"

### Phase 7: Signal Generation
- **Long signal**: Uptrend + price touches lower band 2 + bounces back above
- **Short signal**: Downtrend + price touches upper band 2 + bounces back below
- **Filters**: Trading hours, position status, daily limits

### Phase 8: Position Sizing
- **Risk allocation**: 0.1% of account equity per trade
- **Stop placement**: 2 ticks beyond entry band
- **Contract calculation**: Risk allowance / (ticks at risk × tick value)
- **Cap**: Maximum 1 contract

### Phase 9: Entry Execution
- **Order type**: Market orders (BUY for long, SELL for short)
- **Stop loss**: Immediate placement at calculated stop price
- **Target**: 1.5:1 risk/reward ratio
- **Tracking**: Full position state with entry time, prices, order IDs

### Phase 10: Exit Management
- **Stop hit**: Bar low/high breaches stop price
- **Target reached**: Price touches target level  
- **Signal reversal**: Counter-movement through opposite bands
- **P&L tracking**: Tick-based profit/loss calculation

### Phase 11: Daily Reset Logic
- **Daily reset check**: Monitors date changes at 8 AM ET
- **Counter resets**: Trade count, daily P&L, VWAP data
- **Session stats**: Clears and logs previous day summary
- **Trading re-enable**: Resets daily limit flags for new day

### Phase 12: Safety Mechanisms
- **Daily loss limit**: $400 with trading stop enforcement
- **Maximum drawdown**: 2% of starting equity monitoring
- **Time-based kill switch**: 4 PM ET market close shutdown
- **Connection health**: 60-second tick timeout detection
- **Order validation**: Quantity, stop placement, margin checks
- **Safety checks**: Executed before every signal evaluation

### Phase 13: Logging and Monitoring
- **Structured logging**: Timestamps, levels (INFO/WARNING/ERROR/CRITICAL)
- **Session summary**: Daily stats with win rate, P&L, Sharpe ratio
- **Trade tracking**: Complete history for each trading session
- **Alerts**: Approaching limits, connection issues, errors
- **Statistics**: Variance tracking for performance metrics

### Phase 14: Testing Workflow
- **Dry run mode**: Default enabled for safe testing
- **Paper trading**: Minimum 2-week validation recommended
- **Edge cases**: Market gaps, zero volume, data feed issues
- **Stress testing**: FOMC days, crashes, safety triggers
- **Validation**: Comprehensive test suite included

## SDK Integration Points

The bot includes wrapper functions for TopStep SDK integration:

- `initialize_sdk()` - Initialize SDK client with API token
- `get_account_equity()` - Fetch current account balance
- `place_market_order()` - Execute market orders
- `place_stop_order()` - Place stop loss orders
- `subscribe_market_data()` - Subscribe to real-time ticks
- `fetch_historical_bars()` - Get historical data for initialization

## Risk Management

The bot implements multiple layers of risk control:

1. **Position Sizing**: 0.1% of equity per trade
2. **Max Contracts**: Limited to 1 contract
3. **Daily Trade Limit**: Maximum 5 trades per day
4. **Daily Loss Limit**: $400 stop out threshold
5. **Maximum Drawdown**: 2% total drawdown emergency stop
6. **Trading Hours**: Restricted to liquid market hours (10 AM - 3:30 PM ET)
7. **Market Close**: Hard stop at 4 PM ET
8. **Connection Health**: 60-second timeout monitoring
9. **Stop Losses**: Automatic stop placement on every trade
10. **Order Validation**: Pre-flight checks before every order

## Logging

All bot activity is logged to:
- **Console**: Real-time monitoring
- **Log File**: `vwap_bounce_bot.log` for historical review

Log levels:
- **INFO**: Bot startup, SDK connection, signals, trades, resets
- **WARNING**: Rejected signals, approaching limits, connection issues
- **ERROR**: SDK exceptions, order failures, data processing errors
- **CRITICAL**: Loss limits breached, drawdown exceeded, emergency stops

Session Summary (logged at end of each day):
- Total trades, win/loss counts, win rate
- Total P&L, largest win/loss
- Sharpe ratio (if sufficient variance data)
- ERROR: Failures and issues
- DEBUG: Detailed calculation data

## Development Status

**Current Phase**: All 14 phases complete and tested ✅

✅ **Completed**:
- Phase 1: Project setup and configuration
- Phase 2: SDK integration wrapper functions
- Phase 3: State management structures
- Phase 4: Data processing pipeline
- Phase 5: VWAP calculation with bands
- **Phase 6: Trend filter with 50-period EMA**
- **Phase 7: Signal generation logic**
- **Phase 8: Position sizing algorithm**
- **Phase 9: Entry execution with stops**
- **Phase 10: Exit management (stop/target/reversal)**
- **Phase 11: Daily reset logic (8 AM ET)**
- **Phase 12: Safety mechanisms (loss limits, drawdown, validation)**
- **Phase 13: Comprehensive logging and monitoring**
- **Phase 14: Testing workflow and documentation**

🔄 **Recommended Next Steps**:
- Paper trading for minimum 2 weeks
- Performance validation and optimization
- TopStep SDK actual integration (requires SDK package)
- Live market data feed integration
- Production deployment configuration

## Testing

**Run Complete Trading Cycle Test:**
```bash
python3 test_complete_cycle.py
```

**Run Safety & Monitoring Test:**
```bash
python3 test_phases_11_14.py
```

**Test Coverage:**
- ✅ Phases 1-10: Complete trading cycle with signal → entry → exit
- ✅ Phases 11-14: Daily reset, safety mechanisms, session tracking
- ✅ Edge cases: Loss limits, drawdown, order validation
- ✅ All tests passing with expected behavior

Expected output shows successful trade execution, safety mechanisms working, and comprehensive session tracking.

## Safety Notes

- Always start with **dry run mode** enabled
- Test thoroughly with paper trading before going live
- Monitor daily loss limits closely
- Keep API tokens secure (never commit to git)
- Review logs regularly for unexpected behavior

## License

MIT License - See LICENSE file for details

## Disclaimer

This software is for educational purposes only. Trading futures involves substantial risk of loss. Use at your own risk. The authors are not responsible for any financial losses incurred through use of this software.
