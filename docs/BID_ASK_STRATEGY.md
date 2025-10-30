# Complete Bid/Ask Trading Strategy

## Overview

The bot now implements a professional-grade bid/ask trading strategy that minimizes trading costs and maximizes fill rates through intelligent order placement. This strategy is essential for profitable trading as it can save significant amounts on spreads over time.

## What Your Bot Needs (All Implemented ✓)

### 1. Real-Time Market Data Access ✓

The bot receives continuous bid/ask quotes with complete market depth information:

- **Current bid price** - Where sellers are willing to sell
- **Current ask price** - Where buyers are willing to buy  
- **Bid size** - How many contracts available at bid
- **Ask size** - How many contracts available at ask
- **Last trade price** - Most recent executed trade
- **Timestamp** - Quote timestamp in milliseconds

**Implementation:** `BidAskQuote` dataclass in `bid_ask_manager.py`

```python
@dataclass
class BidAskQuote:
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    last_trade_price: float
    timestamp: int
```

### 2. Spread Analysis Before Entry ✓

Before taking any trade signal, the bot analyzes the bid/ask spread:

- **Measure current bid/ask spread** - Real-time spread calculation
- **Compare spread to normal conditions** - Statistical baseline comparison
- **Reject entries when spread is abnormally wide** - Indicates low liquidity or high volatility
- **Track average spread over time** - Establishes baseline expectations
- **Only enter when spread is acceptable** - Configurable threshold enforcement

**Implementation:** `SpreadAnalyzer` class

```python
class SpreadAnalyzer:
    """Analyzes bid/ask spreads to determine market conditions."""
    
    def __init__(self, lookback_periods=100, abnormal_multiplier=2.0):
        # Tracks last 100 spread samples
        # Rejects spreads >2x average (configurable)
        
    def is_spread_acceptable(self, current_spread) -> (bool, str):
        # Returns True if spread <= average * abnormal_multiplier
        # Returns False with reason if spread too wide
```

**Configuration Parameters:**
- `spread_lookback_periods`: 100 samples (default)
- `abnormal_spread_multiplier`: 2.0x (default)

### 3. Intelligent Order Placement Strategy ✓

The bot implements both passive and aggressive order placement approaches:

#### For Long Entries (Buying):

**Passive Approach:**
- Place limit order **AT the bid price** (join sellers, wait for fill)
- **Advantage:** Save the spread, get better entry price
- **Risk:** May not get filled if price moves away
- **Timeout:** If not filled within acceptable timeframe, cancel and use aggressive

**Aggressive Approach:**
- Place limit order **at ask price** or use market order
- **Advantage:** Guaranteed fill
- **Cost:** Pay the spread

#### For Short Entries (Selling):

**Passive Approach:**
- Place limit order **AT the ask price** (join buyers, wait for fill)
- **Advantage:** Save the spread, get better entry price
- **Risk:** May not get filled if price moves away
- **Timeout:** If not filled within acceptable timeframe, cancel and use aggressive

**Aggressive Approach:**
- Place limit order **at bid price** or use market order
- **Advantage:** Guaranteed fill
- **Cost:** Pay the spread

**Implementation:** `OrderPlacementStrategy` class

```python
class OrderPlacementStrategy:
    """Intelligent order placement strategy."""
    
    def calculate_passive_entry_price(self, side, quote):
        if side == "long":
            return quote.bid_price  # Join sellers
        else:
            return quote.ask_price  # Join buyers
    
    def calculate_aggressive_entry_price(self, side, quote):
        if side == "long":
            return quote.ask_price  # Pay the ask
        else:
            return quote.bid_price  # Hit the bid
```

### 4. Dynamic Fill Strategy ✓

The bot intelligently decides when to be passive vs aggressive:

**Use passive orders when:**
- Market is calm (spread ≤ 1.5x average)
- Spread is tight (good liquidity)
- No urgency (normal signal strength)
- Already at good price level

**Use aggressive orders when:**
- High volatility (spread > 3.0x average)
- Spread is already wide
- Time-critical situations
- Strong signal momentum

**Mixed approach (optional):**
- Split order between passive and aggressive
- Balance fill rate vs cost
- Example: 50% passive at bid, 50% aggressive at ask

**Implementation:** `DynamicFillStrategy` class

```python
class DynamicFillStrategy:
    """Manages dynamic fill strategy."""
    
    def should_use_mixed_strategy(self, contracts):
        # Returns (use_mixed, passive_qty, aggressive_qty)
        # Only for multi-contract orders
    
    def get_retry_strategy(self, attempt, max_attempts):
        # Returns strategy parameters for retry attempts
        # Escalates from passive to aggressive
```

**Configuration Parameters:**
- `passive_order_timeout`: 10 seconds (default)
- `high_volatility_spread_mult`: 3.0x (use aggressive)
- `calm_market_spread_mult`: 1.5x (use passive)
- `use_mixed_order_strategy`: False (default, can enable)
- `mixed_passive_ratio`: 0.5 (50/50 split)

## Performance Benefits

### Cost Savings Example

**Scenario:** ES futures, 1 tick spread = $12.50

**Without Bid/Ask Strategy (Market Orders):**
- Entry: Pay the spread = -$12.50
- Exit: Pay the spread = -$12.50
- **Total cost per round-trip: -$25.00**

**With Bid/Ask Strategy (Passive Entry, 80% Fill Rate):**
- Entry (80% passive): Save $12.50 * 0.80 = $10.00
- Entry (20% aggressive): Pay $12.50 * 0.20 = -$2.50
- Exit: Pay the spread = -$12.50
- **Total cost per round-trip: -$5.00**

**Savings: $20.00 per round-trip (80% reduction)**

**Annual Impact (100 trades/year):**
- Old cost: $2,500
- New cost: $500
- **Annual savings: $2,000**

## Configuration

### Default Settings (Conservative)

```python
# In config.py
passive_order_timeout: int = 10  # Wait 10 seconds for passive fill
abnormal_spread_multiplier: float = 2.0  # Reject spreads >2x average
spread_lookback_periods: int = 100  # Track 100 spread samples
high_volatility_spread_mult: float = 3.0  # Use aggressive if spread >3x
calm_market_spread_mult: float = 1.5  # Use passive if spread ≤1.5x
use_mixed_order_strategy: bool = False  # Disabled by default
mixed_passive_ratio: float = 0.5  # 50/50 split if enabled
```

### Aggressive Settings (Maximize Fill Rate)

```python
passive_order_timeout: int = 5  # Shorter timeout
abnormal_spread_multiplier: float = 3.0  # More tolerant of wide spreads
high_volatility_spread_mult: float = 4.0  # Higher threshold for aggressive
calm_market_spread_mult: float = 2.0  # Use passive more often
```

### Cost-Optimized Settings (Maximize Savings)

```python
passive_order_timeout: int = 15  # Longer timeout, more patient
abnormal_spread_multiplier: float = 1.5  # Stricter spread filter
high_volatility_spread_mult: float = 2.0  # Lower threshold for aggressive
calm_market_spread_mult: float = 1.2  # Very tight spread required for passive
use_mixed_order_strategy: bool = True  # Use mixed approach
mixed_passive_ratio: float = 0.7  # 70% passive, 30% aggressive
```

## Testing

Comprehensive test suite with 19 unit tests:

```bash
python3 test_bid_ask_manager.py -v
```

**Test Coverage:**
- Quote updates and spread calculation ✓
- Spread analyzer baseline tracking ✓
- Abnormal spread detection ✓
- Passive/aggressive price calculation ✓
- Strategy decision logic ✓
- Mixed order splitting ✓
- Retry strategy progression ✓
- Full integration tests ✓

All 19 tests passing!
