#!/usr/bin/env python3
"""
Single backtest runner for iterative parameter optimization
"""
import os
import sys
import logging
from datetime import datetime, timedelta
import pytz
from config import BotConfiguration
from backtesting import BacktestConfig, BacktestEngine, ReportGenerator

# Set backtest mode
os.environ['BOT_BACKTEST_MODE'] = 'true'

def run_backtest(params):
    """Run a single backtest with given parameters"""
    logging.basicConfig(level=logging.WARNING)
    
    # Setup ES config
    bot_config = BotConfiguration()
    bot_config.instrument = "ES"
    bot_config.tick_size = 0.25
    bot_config.tick_value = 12.50
    bot_config.backtest_mode = True
    
    # Apply test parameters
    for key, value in params.items():
        setattr(bot_config, key, value)
    
    # Backtest config - use all available data
    et = pytz.timezone('America/New_York')
    end_date = datetime.now(et)
    start_date = datetime(2025, 8, 31, tzinfo=et)  # Start of data
    
    backtest_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_equity=25000.0,
        symbols=["ES"],
        slippage_ticks=1.5,
        commission_per_contract=2.50,
        data_path="./historical_data",
        use_tick_data=False
    )
    
    # Run backtest
    bot_config_dict = bot_config.to_dict()
    engine = BacktestEngine(backtest_config, bot_config_dict)
    
    from vwap_bounce_bot import initialize_state, on_tick, check_for_signals, check_exit_conditions, check_daily_reset, state
    
    def vwap_strategy(bars_1min, bars_15min):
        symbol = "ES"
        initialize_state(symbol)
        et = pytz.timezone('US/Eastern')
        
        for bar in bars_1min:
            timestamp = bar['timestamp']
            price = bar['close']
            volume = bar['volume']
            timestamp_ms = int(timestamp.timestamp() * 1000)
            timestamp_et = timestamp.astimezone(et)
            
            check_daily_reset(symbol, timestamp_et)
            on_tick(symbol, price, volume, timestamp_ms)
            check_for_signals(symbol)
            check_exit_conditions(symbol)
    
    results = engine.run_with_strategy(vwap_strategy)
    return results

if __name__ == '__main__':
    # Parse parameters from command line
    params = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Try to convert to appropriate type
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
            params[key] = value
    
    results = run_backtest(params)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*80}")
    print(f"Parameters tested: {params}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.1f}%")
    print(f"Total P&L: ${results['total_pnl']:,.2f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Max Drawdown: ${results['max_drawdown_dollars']:,.2f} ({results['max_drawdown_percent']:.1f}%)")
    print(f"Final Equity: ${results['final_equity']:,.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"{'='*80}\n")
