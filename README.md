# VWAP Bounce Bot

An event-driven mean reversion trading bot for futures trading (MES) that executes trades based on VWAP (Volume Weighted Average Price) standard deviation bands and trend alignment.

## Overview

The VWAP Bounce Bot subscribes to real-time tick data, aggregates it into bars, calculates VWAP with standard deviation bands, determines trend direction, and executes mean reversion trades when price touches extreme bands while aligned with the trend.

## Architecture

### Five-Phase Implementation

1. **Project Setup** - Configuration and risk management parameters
2. **SDK Integration** - TopStep SDK wrapper functions for trading operations
3. **State Management** - Data structures for ticks, bars, positions, and P&L tracking
4. **Data Processing Pipeline** - Real-time tick handling and bar aggregation
5. **VWAP Calculation** - Volume-weighted average price with standard deviation bands

## Features

- **Event-Driven Architecture**: Processes real-time tick data efficiently
- **Risk Management**: Conservative 0.1% risk per trade with daily loss limits
- **Trend Filter**: 50-period EMA on 15-minute bars
- **VWAP Bands**: Two standard deviation bands for entry signals
- **Trading Hours**: 10:00 AM - 3:30 PM ET (avoiding open/close chaos)
- **Dry Run Mode**: Test strategies without risking capital

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

### Running in Dry Run Mode (Default)

Test the bot without executing real trades:

```bash
export TOPSTEP_API_TOKEN='your_token_here'
python vwap_bounce_bot.py
```

### Running in Live Mode

⚠️ **WARNING**: Only use live mode after thorough testing in dry run mode!

Edit `vwap_bounce_bot.py` and set:
```python
CONFIG = {
    ...
    "dry_run": False,
    ...
}
```

Then run:
```bash
python vwap_bounce_bot.py
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
├── example_simulation.py   # Basic simulation example
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
- **Daily loss limit**: $400 with critical alert and trading stop
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
