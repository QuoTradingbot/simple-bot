"""
Test Time Management and VWAP Reset Enhancement
Tests for Phase One, Two, and Three implementations
"""

import sys
from datetime import datetime, time, timedelta
import pytz

# Import the bot
import vwap_bounce_bot as bot

def print_section(title):
    """Print section header"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def test_configuration():
    """Test Phase One: Enhanced Configuration"""
    print_section("Testing Phase One: Enhanced Configuration")
    
    print("\n📋 Time-Related Configuration:")
    print(f"  Entry Window Start: {bot.CONFIG['entry_window_start']}")
    print(f"  Entry Window End: {bot.CONFIG['entry_window_end']}")
    print(f"  Warning Time: {bot.CONFIG['warning_time']}")
    print(f"  Forced Flatten Time: {bot.CONFIG['forced_flatten_time']}")
    print(f"  Shutdown Time: {bot.CONFIG['shutdown_time']}")
    print(f"  VWAP Reset Time: {bot.CONFIG['vwap_reset_time']}")
    print(f"  Timezone: {bot.CONFIG['timezone']}")
    
    print("\n🔧 Additional Parameters:")
    print(f"  Flatten Buffer Ticks: {bot.CONFIG['flatten_buffer_ticks']}")
    print(f"  Stop Buffer Ticks: {bot.CONFIG['stop_buffer_ticks']}")
    
    print("\n🚩 Flatten Mode Flag:")
    print(f"  Flatten Mode: {bot.bot_status['flatten_mode']}")
    
    # Verify configuration values
    assert bot.CONFIG['entry_window_start'] == time(9, 0), "Entry window start should be 9:00 AM"
    assert bot.CONFIG['entry_window_end'] == time(14, 30), "Entry window end should be 2:30 PM"
    assert bot.CONFIG['warning_time'] == time(16, 30), "Warning time should be 4:30 PM"
    assert bot.CONFIG['forced_flatten_time'] == time(16, 45), "Forced flatten time should be 4:45 PM"
    assert bot.CONFIG['shutdown_time'] == time(17, 0), "Shutdown time should be 5:00 PM"
    assert bot.CONFIG['vwap_reset_time'] == time(9, 30), "VWAP reset time should be 9:30 AM"
    assert bot.CONFIG['timezone'] == "America/New_York", "Timezone should be America/New_York"
    assert bot.CONFIG['flatten_buffer_ticks'] == 2, "Flatten buffer should be 2 ticks"
    
    print("\n✅ Configuration tests passed!")

def test_time_check_function():
    """Test Phase Two: Time Check Function"""
    print_section("Testing Phase Two: Time Check Function")
    
    tz = pytz.timezone("America/New_York")
    
    # Test different times of day
    test_cases = [
        (time(8, 0), "before_open", "8:00 AM - before open"),
        (time(8, 59), "before_open", "8:59 AM - before open"),
        (time(9, 0), "entry_window", "9:00 AM - entry window starts"),
        (time(10, 30), "entry_window", "10:30 AM - during entry window"),
        (time(14, 29), "entry_window", "2:29 PM - still entry window"),
        (time(14, 30), "exit_only", "2:30 PM - exit only starts"),
        (time(15, 0), "exit_only", "3:00 PM - during exit only"),
        (time(16, 29), "exit_only", "4:29 PM - still exit only"),
        (time(16, 30), "flatten_mode", "4:30 PM - flatten mode starts"),
        (time(16, 40), "flatten_mode", "4:40 PM - during flatten mode"),
        (time(16, 45), "closed", "4:45 PM - closed/shutdown"),
        (time(17, 0), "closed", "5:00 PM - shutdown time"),
        (time(21, 0), "closed", "9:00 PM - after hours"),
    ]
    
    print("\n⏰ Testing Trading State Detection:")
    for test_time, expected_state, description in test_cases:
        # Create a datetime with the test time
        dt = datetime.combine(datetime.today(), test_time)
        dt = tz.localize(dt)
        
        state = bot.get_trading_state(dt)
        status = "✅" if state == expected_state else "❌"
        print(f"  {status} {description}: {state}")
        
        assert state == expected_state, f"Expected {expected_state} but got {state} for {test_time}"
    
    print("\n✅ Time check function tests passed!")

def test_vwap_reset():
    """Test Phase Three: VWAP Daily Reset Enhanced"""
    print_section("Testing Phase Three: VWAP Daily Reset Enhanced")
    
    # Initialize state
    symbol = "MES"
    bot.initialize_state(symbol)
    
    tz = pytz.timezone("America/New_York")
    
    # Simulate initial day
    day1 = datetime(2025, 10, 28, 10, 0, 0)  # Oct 28, 10:00 AM
    day1 = tz.localize(day1)
    
    print("\n📅 Day 1: October 28, 2025")
    print(f"  Time: {day1.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Add some VWAP data
    bot.state[symbol]["vwap"] = 4500.0
    bot.state[symbol]["vwap_day"] = day1.date()
    bot.state[symbol]["trading_day"] = day1.date()
    bot.state[symbol]["daily_trade_count"] = 3
    bot.state[symbol]["daily_pnl"] = 150.0
    bot.state[symbol]["bars_1min"].append({"timestamp": day1, "open": 4500, "high": 4505, "low": 4495, "close": 4500, "volume": 100})
    
    print(f"  Initial VWAP: {bot.state[symbol]['vwap']}")
    print(f"  Initial 1-min bars: {len(bot.state[symbol]['bars_1min'])}")
    print(f"  Daily trade count: {bot.state[symbol]['daily_trade_count']}")
    print(f"  Daily P&L: ${bot.state[symbol]['daily_pnl']}")
    
    # Test VWAP reset at 9:30 AM on day 2
    day2_vwap_reset = datetime(2025, 10, 29, 9, 30, 0)  # Oct 29, 9:30 AM
    day2_vwap_reset = tz.localize(day2_vwap_reset)
    
    print("\n📅 Day 2: October 29, 2025 - VWAP Reset Time (9:30 AM)")
    print(f"  Time: {day2_vwap_reset.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Check VWAP reset
    bot.check_vwap_reset(symbol, day2_vwap_reset)
    
    print(f"  VWAP after reset: {bot.state[symbol]['vwap']}")
    print(f"  1-min bars after reset: {len(bot.state[symbol]['bars_1min'])}")
    print(f"  VWAP day: {bot.state[symbol]['vwap_day']}")
    
    # Verify VWAP was reset
    assert bot.state[symbol]['vwap'] is None, "VWAP should be reset to None"
    assert len(bot.state[symbol]['bars_1min']) == 0, "1-min bars should be cleared"
    assert bot.state[symbol]['vwap_day'] == day2_vwap_reset.date(), "VWAP day should be updated"
    
    # Check daily reset (also at 9:30 AM)
    bot.check_daily_reset(symbol, day2_vwap_reset)
    
    print(f"  Daily trade count after reset: {bot.state[symbol]['daily_trade_count']}")
    print(f"  Daily P&L after reset: ${bot.state[symbol]['daily_pnl']}")
    print(f"  Trading day: {bot.state[symbol]['trading_day']}")
    
    # Verify daily counters were reset
    assert bot.state[symbol]['daily_trade_count'] == 0, "Daily trade count should be reset"
    assert bot.state[symbol]['daily_pnl'] == 0.0, "Daily P&L should be reset"
    assert bot.state[symbol]['trading_day'] == day2_vwap_reset.date(), "Trading day should be updated"
    
    print("\n✅ VWAP reset tests passed!")

def test_flatten_price_calculation():
    """Test flatten price with buffer"""
    print_section("Testing Flatten Price Calculation")
    
    symbol = "MES"
    bot.initialize_state(symbol)
    
    tick_size = bot.CONFIG['tick_size']  # 0.25
    buffer_ticks = bot.CONFIG['flatten_buffer_ticks']  # 2
    
    print(f"\n🔢 Tick Size: {tick_size}")
    print(f"🔢 Buffer Ticks: {buffer_ticks}")
    print(f"🔢 Buffer Amount: {tick_size * buffer_ticks}")
    
    # Test long position flatten
    current_price = 4500.0
    flatten_price_long = bot.get_flatten_price(symbol, "long", current_price)
    expected_long = 4500.0 - (2 * 0.25)  # 4499.50
    
    print(f"\n📉 Long Position Flatten:")
    print(f"  Current Price: {current_price}")
    print(f"  Flatten Price: {flatten_price_long}")
    print(f"  Expected: {expected_long}")
    
    assert flatten_price_long == expected_long, f"Long flatten price should be {expected_long}"
    
    # Test short position flatten
    flatten_price_short = bot.get_flatten_price(symbol, "short", current_price)
    expected_short = 4500.0 + (2 * 0.25)  # 4500.50
    
    print(f"\n📈 Short Position Flatten:")
    print(f"  Current Price: {current_price}")
    print(f"  Flatten Price: {flatten_price_short}")
    print(f"  Expected: {expected_short}")
    
    assert flatten_price_short == expected_short, f"Short flatten price should be {expected_short}"
    
    print("\n✅ Flatten price calculation tests passed!")

def test_trading_state_integration():
    """Test integration of trading states with signal generation"""
    print_section("Testing Trading State Integration")
    
    symbol = "MES"
    bot.initialize_state(symbol)
    
    tz = pytz.timezone("America/New_York")
    
    # Create a bar during entry window
    entry_window_time = datetime(2025, 10, 29, 10, 0, 0)  # 10:00 AM
    entry_window_time = tz.localize(entry_window_time)
    
    bot.state[symbol]["bars_1min"].append({
        "timestamp": entry_window_time,
        "open": 4500, "high": 4505, "low": 4495, "close": 4500, "volume": 100
    })
    
    state = bot.get_trading_state(entry_window_time)
    print(f"\n⏰ Entry Window Time (10:00 AM): State = {state}")
    assert state == "entry_window", "Should be in entry window at 10:00 AM"
    
    # Create a bar during exit only window
    exit_only_time = datetime(2025, 10, 29, 15, 0, 0)  # 3:00 PM
    exit_only_time = tz.localize(exit_only_time)
    
    state = bot.get_trading_state(exit_only_time)
    print(f"⏰ Exit Only Time (3:00 PM): State = {state}")
    assert state == "exit_only", "Should be in exit only at 3:00 PM"
    
    # Create a bar during flatten mode
    flatten_time = datetime(2025, 10, 29, 16, 35, 0)  # 4:35 PM
    flatten_time = tz.localize(flatten_time)
    
    state = bot.get_trading_state(flatten_time)
    print(f"⏰ Flatten Mode Time (4:35 PM): State = {state}")
    assert state == "flatten_mode", "Should be in flatten mode at 4:35 PM"
    
    # Create a bar after close
    closed_time = datetime(2025, 10, 29, 17, 0, 0)  # 5:00 PM
    closed_time = tz.localize(closed_time)
    
    state = bot.get_trading_state(closed_time)
    print(f"⏰ Closed Time (5:00 PM): State = {state}")
    assert state == "closed", "Should be closed at 5:00 PM"
    
    print("\n✅ Trading state integration tests passed!")

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("🤖 Time Management & VWAP Reset Test Suite")
    print("=" * 60)
    
    try:
        test_configuration()
        test_time_check_function()
        test_vwap_reset()
        test_flatten_price_calculation()
        test_trading_state_integration()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)
        print("\n✨ Phase One, Two, and Three implementations verified!")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
