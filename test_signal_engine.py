"""
Test Signal Engine Locally
Test the VWAP signal generation logic before deploying.
"""

import sys
sys.path.append('cloud-api')

from signal_engine import signal_engine
from datetime import datetime, timedelta

# Create test bars (simulating ES futures data)
def create_test_bars():
    """Create realistic 1-min bar data for testing."""
    bars = []
    base_price = 5800.0
    timestamp = datetime.utcnow() - timedelta(minutes=30)
    
    # Generate 30 bars with VWAP bounce pattern
    for i in range(30):
        # Simulate price moving below VWAP then bouncing
        if i < 15:
            # Downtrend - price dropping
            price = base_price - (i * 2)
        else:
            # Bounce back up
            price = base_price - 30 + ((i - 15) * 1.5)
        
        bar = {
            "timestamp": (timestamp + timedelta(minutes=i)).isoformat(),
            "open": price,
            "high": price + 1.0,
            "low": price - 1.5,
            "close": price - 0.5,
            "volume": 1000 + (i * 10)
        }
        bars.append(bar)
    
    return bars


def test_vwap_calculation():
    """Test VWAP band calculation."""
    print("=" * 80)
    print("TEST 1: VWAP Band Calculation")
    print("=" * 80)
    
    bars = create_test_bars()
    bands = signal_engine.calculate_vwap_bands(bars, 2.0)
    
    print(f"VWAP: {bands['vwap']:.2f}")
    print(f"Upper Band 2: {bands['upper_2']:.2f}")
    print(f"Lower Band 2: {bands['lower_2']:.2f}")
    print(f"Std Dev: {bands['std_dev']:.2f}")
    print()


def test_rsi_calculation():
    """Test RSI calculation."""
    print("=" * 80)
    print("TEST 2: RSI Calculation")
    print("=" * 80)
    
    bars = create_test_bars()
    rsi = signal_engine.calculate_rsi(bars, period=14)
    
    print(f"RSI (14): {rsi:.2f}")
    print()


def test_long_signal_generation():
    """Test LONG signal generation."""
    print("=" * 80)
    print("TEST 3: LONG Signal Generation")
    print("=" * 80)
    
    bars = create_test_bars()
    
    # User settings
    settings = {
        "account_size": 50000,
        "risk_per_trade": 0.01,
        "use_rsi_filter": True,
        "rsi_oversold": 30.0,
        "use_volume_filter": False,
        "stop_loss_ticks": 8,
        "tick_size": 0.25,
        "tick_value": 12.50,
        "max_contracts": 25
    }
    
    signal = signal_engine.generate_signal(
        user_id="test_user_1",
        symbol="ES",
        bars=bars,
        current_position=None,
        settings=settings
    )
    
    print(f"Action: {signal['action']}")
    print(f"Contracts: {signal['contracts']}")
    print(f"Entry: ${signal['entry']:.2f}")
    print(f"Stop: ${signal['stop']:.2f}")
    print(f"Target: ${signal['target']:.2f}")
    print(f"Confidence: {signal['confidence']:.1%}")
    print(f"Reason: {signal['reason']}")
    print(f"VWAP: ${signal.get('vwap', 0):.2f}")
    print(f"RSI: {signal.get('rsi', 0):.2f}")
    print()


def test_position_size_calculation():
    """Test position sizing."""
    print("=" * 80)
    print("TEST 4: Position Size Calculation")
    print("=" * 80)
    
    account_size = 50000
    risk_per_trade = 0.01  # 1%
    entry_price = 5800.0
    stop_price = 5798.0
    tick_size = 0.25
    tick_value = 12.50
    max_contracts = 25
    
    contracts, stop, target = signal_engine.calculate_position_size(
        account_size, risk_per_trade, entry_price, stop_price,
        tick_size, tick_value, max_contracts
    )
    
    print(f"Account Size: ${account_size:,}")
    print(f"Risk Per Trade: {risk_per_trade:.1%} (${account_size * risk_per_trade:,.2f})")
    print(f"Entry: ${entry_price:.2f}")
    print(f"Stop: ${stop:.2f}")
    print(f"Target: ${target:.2f}")
    print(f"Contracts: {contracts}")
    print(f"Total Risk: ${contracts * abs(entry_price - stop) / tick_size * tick_value:.2f}")
    print()


def test_with_position():
    """Test signal generation when already in position."""
    print("=" * 80)
    print("TEST 5: Signal with Active Position")
    print("=" * 80)
    
    bars = create_test_bars()
    
    settings = {
        "account_size": 50000,
        "risk_per_trade": 0.01,
        "tick_size": 0.25,
        "tick_value": 12.50,
        "max_contracts": 25
    }
    
    # Simulate active position
    current_position = {
        "active": True,
        "side": "long",
        "quantity": 5,
        "entry_price": 5785.0
    }
    
    signal = signal_engine.generate_signal(
        user_id="test_user_1",
        symbol="ES",
        bars=bars,
        current_position=current_position,
        settings=settings
    )
    
    print(f"Action: {signal['action']}")
    print(f"Reason: {signal['reason']}")
    print()


if __name__ == "__main__":
    print("\n🧪 TESTING SIGNAL ENGINE\n")
    
    test_vwap_calculation()
    test_rsi_calculation()
    test_position_size_calculation()
    test_long_signal_generation()
    test_with_position()
    
    print("=" * 80)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 80)
