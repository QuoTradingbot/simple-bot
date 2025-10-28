"""
Example: Running VWAP Bounce Bot with Simulated Data

This example demonstrates how to:
1. Initialize the bot
2. Feed it with simulated tick data
3. Monitor VWAP calculations
4. View the bot's state
"""

import os
import sys
from datetime import datetime, timedelta
import time
import pytz

# Set API token before importing bot
os.environ['TOPSTEP_API_TOKEN'] = 'your_test_token_here'

from vwap_bounce_bot import (
    initialize_sdk, initialize_state, on_tick, 
    state, CONFIG, logger, is_trading_hours
)


def simulate_market_data():
    """Simulate realistic market tick data for MES"""
    print("\n" + "="*70)
    print("VWAP Bounce Bot - Live Simulation Example")
    print("="*70)
    
    # Initialize the bot
    print("\n🚀 Initializing bot...")
    initialize_sdk()
    symbol = CONFIG["instrument"]
    initialize_state(symbol)
    
    # Setup simulation parameters
    tz = pytz.timezone(CONFIG["timezone"])
    current_time = datetime.now(tz).replace(hour=10, minute=30, second=0, microsecond=0)
    
    # Starting price for MES (Micro E-mini S&P 500)
    base_price = 4500.0
    
    print(f"📊 Symbol: {symbol}")
    print(f"⏰ Start Time: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"💰 Base Price: ${base_price:.2f}")
    print(f"🎯 Mode: {'DRY RUN' if CONFIG['dry_run'] else 'LIVE'}")
    
    # Check if in trading hours
    if not is_trading_hours(current_time):
        print(f"⚠️  Warning: {current_time.time()} is outside trading hours")
        print(f"   Trading window: {CONFIG['trading_window']['start']} - {CONFIG['trading_window']['end']} ET")
    else:
        print(f"✅ Within trading hours")
    
    print("\n" + "-"*70)
    print("Starting tick simulation...")
    print("-"*70)
    
    # Simulate 10 minutes of trading
    tick_number = 0
    price = base_price
    
    for minute in range(10):
        # Simulate price movement (random walk)
        minute_drift = (minute % 3 - 1) * 0.5  # -0.5, 0, or +0.5
        
        # Generate ticks for this minute
        for second in range(0, 60, 6):  # 10 ticks per minute
            tick_time = current_time + timedelta(minutes=minute, seconds=second)
            timestamp_ms = int(tick_time.timestamp() * 1000)
            
            # Simulate price movement within the minute
            tick_variance = ((second / 60.0) - 0.5) * 0.5  # -0.25 to +0.25
            tick_price = price + minute_drift + tick_variance
            
            # Round to tick size
            tick_price = round(tick_price / CONFIG['tick_size']) * CONFIG['tick_size']
            
            # Simulate volume (1-5 contracts per tick)
            tick_volume = 1 + (tick_number % 5)
            
            # Feed tick to bot
            on_tick(symbol, tick_price, tick_volume, timestamp_ms)
            tick_number += 1
            
            # Display periodic updates
            if tick_number % 20 == 0:
                display_status(symbol, tick_number, tick_price)
        
        # Update price for next minute
        price = tick_price
    
    print("\n" + "="*70)
    print("Simulation Complete")
    print("="*70)
    
    # Display final results
    display_final_summary(symbol, tick_number)


def display_status(symbol: str, tick_count: int, current_price: float):
    """Display current bot status"""
    s = state[symbol]
    
    print(f"\n📈 Tick #{tick_count:4d} | Price: ${current_price:8.2f} | ", end="")
    
    if s['vwap'] is not None:
        print(f"VWAP: ${s['vwap']:.2f} | ", end="")
        
        # Show position relative to bands
        if current_price >= s['vwap_bands']['upper_2']:
            print("Position: Above Upper Band 2 🔴")
        elif current_price >= s['vwap_bands']['upper_1']:
            print("Position: Above Upper Band 1 🟡")
        elif current_price <= s['vwap_bands']['lower_2']:
            print("Position: Below Lower Band 2 🔵")
        elif current_price <= s['vwap_bands']['lower_1']:
            print("Position: Below Lower Band 1 🟢")
        else:
            print("Position: Within bands ⚪")
    else:
        print("VWAP: Calculating...")


def display_final_summary(symbol: str, total_ticks: int):
    """Display final summary of the simulation"""
    s = state[symbol]
    
    print(f"\n📊 Final Statistics:")
    print(f"   Total Ticks Processed: {total_ticks}")
    print(f"   Ticks in Memory: {len(s['ticks'])}")
    print(f"   1-Minute Bars: {len(s['bars_1min'])}")
    print(f"   15-Minute Bars: {len(s['bars_15min'])}")
    
    if s['vwap'] is not None:
        print(f"\n💹 VWAP Analysis:")
        print(f"   VWAP: ${s['vwap']:.2f}")
        print(f"   Standard Deviation: ${s['vwap_std_dev']:.4f}")
        print(f"\n   📊 Bands:")
        print(f"      Upper Band 2 (+2σ): ${s['vwap_bands']['upper_2']:.2f}")
        print(f"      Upper Band 1 (+1σ): ${s['vwap_bands']['upper_1']:.2f}")
        print(f"      VWAP (center):      ${s['vwap']:.2f}")
        print(f"      Lower Band 1 (-1σ): ${s['vwap_bands']['lower_1']:.2f}")
        print(f"      Lower Band 2 (-2σ): ${s['vwap_bands']['lower_2']:.2f}")
        
        # Calculate band width
        band_width = s['vwap_bands']['upper_2'] - s['vwap_bands']['lower_2']
        print(f"\n   Band Width (4σ): ${band_width:.2f}")
    
    if len(s['bars_1min']) > 0:
        print(f"\n📊 Recent Bars (Last 3):")
        for i, bar in enumerate(list(s['bars_1min'])[-3:], 1):
            bar_time = bar['timestamp'].strftime('%H:%M')
            print(f"   {i}. {bar_time} | O:{bar['open']:.2f} H:{bar['high']:.2f} "
                  f"L:{bar['low']:.2f} C:{bar['close']:.2f} | Vol:{bar['volume']}")
    
    print(f"\n🔍 Trend Analysis:")
    if s['trend_ema'] is not None:
        print(f"   Trend EMA: ${s['trend_ema']:.2f}")
        print(f"   Direction: {s['trend_direction']}")
    else:
        needed = CONFIG['trend_filter_period'] - len(s['bars_15min'])
        print(f"   Status: Need {needed} more 15-min bars for trend calculation")
    
    print(f"\n💼 Trading Status:")
    print(f"   Position Active: {s['position']['active']}")
    print(f"   Daily Trades: {s['daily_trade_count']}/{CONFIG['max_trades_per_day']}")
    print(f"   Daily P&L: ${s['daily_pnl']:.2f}")
    
    print(f"\n⚙️  Configuration:")
    print(f"   Max Contracts: {CONFIG['max_contracts']}")
    print(f"   Risk per Trade: {CONFIG['risk_per_trade']*100}%")
    print(f"   Daily Loss Limit: ${CONFIG['daily_loss_limit']:.2f}")
    print(f"   Risk/Reward: {CONFIG['risk_reward_ratio']}:1")


def main():
    """Main execution"""
    print("\n" + "🤖 "*20)
    print("VWAP Bounce Bot - Example Simulation")
    print("🤖 "*20)
    
    print("\nThis example demonstrates:")
    print("  ✓ Bot initialization and setup")
    print("  ✓ Real-time tick data processing")
    print("  ✓ Bar aggregation (1-min and 15-min)")
    print("  ✓ VWAP calculation with bands")
    print("  ✓ State management and tracking")
    
    input("\nPress Enter to start simulation...")
    
    simulate_market_data()
    
    print("\n" + "="*70)
    print("✅ Example completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review vwap_bounce_bot.py for full implementation")
    print("  2. Set your TOPSTEP_API_TOKEN in environment")
    print("  3. Install TopStep SDK per their documentation")
    print("  4. Add strategy logic for trade signals")
    print("  5. Test in dry run mode before going live")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")
        sys.exit(0)
