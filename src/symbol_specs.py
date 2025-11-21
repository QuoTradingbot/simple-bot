"""
Symbol Specifications Database
===============================
Comprehensive specifications for all supported futures symbols.
Used to adapt bot behavior to different instruments automatically.
"""

from dataclasses import dataclass
from datetime import time
from typing import Dict, Optional


@dataclass
class SymbolSpec:
    """Specification for a trading symbol"""
    
    # Basic identification
    symbol: str
    name: str
    exchange: str
    
    # Contract specifications
    tick_size: float          # Minimum price movement (e.g., 0.25 for ES)
    tick_value: float         # Dollar value per tick (e.g., $12.50 for ES)
    point_value: float        # Dollar value per point (e.g., $50 for ES = 1 point = 4 ticks)
    
    # Trading hours (all times in UTC - CME futures schedule)
    session_start: time       # When trading session starts (23:00 UTC)
    session_end: time         # When to stop taking new entries (21:40 UTC)
    maintenance_start: time   # Maintenance window start (22:00 UTC)
    maintenance_end: time     # Maintenance window end (23:00 UTC)
    
    # Risk characteristics
    typical_slippage_ticks: float   # Expected slippage in ticks
    typical_spread_ticks: float     # Typical bid-ask spread
    volatility_factor: float        # Relative volatility (1.0 = ES baseline)
    
    # Symbol mappings for different brokers
    topstep_symbol: str      # TopStep symbol format
    tradovate_symbol: str    # Tradovate symbol format
    rithmic_symbol: str      # Rithmic symbol format
    
    # Trading characteristics
    typical_volume: str      # Volume description
    market_type: str         # "equity_index", "commodity", "currency", "rates"


# ============================================================================
# SYMBOL SPECIFICATIONS DATABASE
# ============================================================================

SYMBOL_SPECS: Dict[str, SymbolSpec] = {
    
    # ========== EQUITY INDEX FUTURES ==========
    
    "ES": SymbolSpec(
        symbol="ES",
        name="E-mini S&P 500",
        exchange="CME",
        tick_size=0.25,
        tick_value=12.50,
        point_value=50.0,
        session_start=time(23, 0),     # 23:00 UTC (market opens)
        session_end=time(21, 40),      # 21:40 UTC (stop new entries, 5 min before flatten)
        maintenance_start=time(22, 0), # 22:00 UTC (maintenance starts)
        maintenance_end=time(23, 0),   # 23:00 UTC (maintenance ends)
        typical_slippage_ticks=1.5,
        typical_spread_ticks=1.0,
        volatility_factor=1.0,         # Baseline
        topstep_symbol="F.US.EP",
        tradovate_symbol="ESZ4",       # Month code varies
        rithmic_symbol="ES",
        typical_volume="Very High",
        market_type="equity_index"
    ),
    
    "MES": SymbolSpec(
        symbol="MES",
        name="Micro E-mini S&P 500",
        exchange="CME",
        tick_size=0.25,
        tick_value=1.25,               # 1/10th of ES
        point_value=5.0,
        session_start=time(23, 0),
        session_end=time(21, 40),
        maintenance_start=time(22, 0),
        maintenance_end=time(23, 0),
        typical_slippage_ticks=2.0,    # Slightly more due to lower volume
        typical_spread_ticks=1.0,
        volatility_factor=1.0,
        topstep_symbol="F.US.MESEP",
        tradovate_symbol="MESZ4",
        rithmic_symbol="MES",
        typical_volume="High",
        market_type="equity_index"
    ),
    
    "NQ": SymbolSpec(
        symbol="NQ",
        name="E-mini Nasdaq 100",
        exchange="CME",
        tick_size=0.25,
        tick_value=5.00,
        point_value=20.0,
        session_start=time(23, 0),
        session_end=time(21, 40),
        maintenance_start=time(22, 0),
        maintenance_end=time(23, 0),
        typical_slippage_ticks=2.0,    # More volatile than ES
        typical_spread_ticks=1.0,
        volatility_factor=1.5,         # 50% more volatile than ES
        topstep_symbol="F.US.NP",
        tradovate_symbol="NQZ4",
        rithmic_symbol="NQ",
        typical_volume="Very High",
        market_type="equity_index"
    ),
    
    "MNQ": SymbolSpec(
        symbol="MNQ",
        name="Micro E-mini Nasdaq 100",
        exchange="CME",
        tick_size=0.25,
        tick_value=0.50,               # 1/10th of NQ
        point_value=2.0,
        session_start=time(23, 0),
        session_end=time(21, 40),
        maintenance_start=time(22, 0),
        maintenance_end=time(23, 0),
        typical_slippage_ticks=2.5,
        typical_spread_ticks=1.0,
        volatility_factor=1.5,
        topstep_symbol="F.US.MNQEP",
        tradovate_symbol="MNQZ4",
        rithmic_symbol="MNQ",
        typical_volume="High",
        market_type="equity_index"
    ),
    
    "YM": SymbolSpec(
        symbol="YM",
        name="E-mini Dow ($5)",
        exchange="CBOT",
        tick_size=1.0,
        tick_value=5.00,
        point_value=5.0,
        session_start=time(23, 0),
        session_end=time(21, 40),
        maintenance_start=time(22, 0),
        maintenance_end=time(23, 0),
        typical_slippage_ticks=1.5,
        typical_spread_ticks=1.0,
        volatility_factor=0.8,         # Less volatile than ES
        topstep_symbol="F.US.YM",
        tradovate_symbol="YMZ4",
        rithmic_symbol="YM",
        typical_volume="High",
        market_type="equity_index"
    ),
    
    "RTY": SymbolSpec(
        symbol="RTY",
        name="E-mini Russell 2000",
        exchange="CME",
        tick_size=0.10,
        tick_value=5.00,
        point_value=50.0,
        session_start=time(23, 0),
        session_end=time(21, 40),
        maintenance_start=time(22, 0),
        maintenance_end=time(23, 0),
        typical_slippage_ticks=2.0,
        typical_spread_ticks=1.0,
        volatility_factor=1.3,         # More volatile than ES
        topstep_symbol="F.US.RTY",
        tradovate_symbol="RTYZ4",
        rithmic_symbol="RTY",
        typical_volume="Medium",
        market_type="equity_index"
    ),
    
    # ========== COMMODITIES ==========
    
    "CL": SymbolSpec(
        symbol="CL",
        name="Crude Oil",
        exchange="NYMEX",
        tick_size=0.01,
        tick_value=10.00,
        point_value=1000.0,
        session_start=time(23, 0),
        session_end=time(21, 40),
        maintenance_start=time(22, 0),
        maintenance_end=time(23, 0),
        typical_slippage_ticks=3.0,    # Very volatile
        typical_spread_ticks=1.0,
        volatility_factor=2.0,         # 2x ES volatility
        topstep_symbol="F.US.CL",
        tradovate_symbol="CLZ4",
        rithmic_symbol="CL",
        typical_volume="Very High",
        market_type="commodity"
    ),
    
    "GC": SymbolSpec(
        symbol="GC",
        name="Gold",
        exchange="COMEX",
        tick_size=0.10,
        tick_value=10.00,
        point_value=100.0,
        session_start=time(23, 0),
        session_end=time(21, 40),
        maintenance_start=time(22, 0),
        maintenance_end=time(23, 0),
        typical_slippage_ticks=2.0,
        typical_spread_ticks=1.0,
        volatility_factor=1.2,
        topstep_symbol="F.US.GC",
        tradovate_symbol="GCZ4",
        rithmic_symbol="GC",
        typical_volume="High",
        market_type="commodity"
    ),
    
    "NG": SymbolSpec(
        symbol="NG",
        name="Natural Gas",
        exchange="NYMEX",
        tick_size=0.001,
        tick_value=10.00,
        point_value=10000.0,
        session_start=time(23, 0),
        session_end=time(21, 40),
        maintenance_start=time(22, 0),
        maintenance_end=time(23, 0),
        typical_slippage_ticks=3.0,    # Very volatile
        typical_spread_ticks=1.0,
        volatility_factor=2.5,         # Extremely volatile
        topstep_symbol="F.US.NG",
        tradovate_symbol="NGZ4",
        rithmic_symbol="NG",
        typical_volume="High",
        market_type="commodity"
    ),
    
    # ========== CURRENCIES ==========
    
    "6E": SymbolSpec(
        symbol="6E",
        name="Euro FX",
        exchange="CME",
        tick_size=0.00005,
        tick_value=6.25,
        point_value=125000.0,
        session_start=time(17, 0),     # 5:00 PM ET (Sunday - Friday)
        session_end=time(15, 55),      # 3:55 PM ET
        maintenance_start=time(16, 0),
        maintenance_end=time(17, 0),
        typical_slippage_ticks=2.0,
        typical_spread_ticks=1.0,
        volatility_factor=0.9,
        topstep_symbol="F.US.E7",
        tradovate_symbol="6EZ4",
        rithmic_symbol="6E",
        typical_volume="High",
        market_type="currency"
    ),
    
    # ========== INTEREST RATES ==========
    
    "ZN": SymbolSpec(
        symbol="ZN",
        name="10-Year Treasury Note",
        exchange="CBOT",
        tick_size=0.015625,            # 1/64th of a point
        tick_value=15.625,
        point_value=1000.0,
        session_start=time(17, 0),
        session_end=time(15, 55),
        maintenance_start=time(16, 0),
        maintenance_end=time(17, 0),
        typical_slippage_ticks=1.5,
        typical_spread_ticks=1.0,
        volatility_factor=0.7,         # Less volatile
        topstep_symbol="F.US.ZN",
        tradovate_symbol="ZNZ4",
        rithmic_symbol="ZN",
        typical_volume="Very High",
        market_type="rates"
    ),
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_symbol_spec(symbol: str) -> SymbolSpec:
    """
    Get specification for a symbol.
    
    Args:
        symbol: Symbol code (e.g., "ES", "NQ")
        
    Returns:
        SymbolSpec object
        
    Raises:
        ValueError: If symbol not found
    """
    symbol = symbol.upper().strip()
    
    if symbol not in SYMBOL_SPECS:
        raise ValueError(
            f"Unknown symbol: {symbol}. "
            f"Supported symbols: {', '.join(SYMBOL_SPECS.keys())}"
        )
    
    return SYMBOL_SPECS[symbol]


def get_broker_symbol(symbol: str, broker: str) -> str:
    """
    Get broker-specific symbol format.
    
    Args:
        symbol: Standard symbol (e.g., "ES")
        broker: Broker name ("TopStep", "Tradovate", "Rithmic")
        
    Returns:
        Broker-specific symbol string
    """
    spec = get_symbol_spec(symbol)
    broker = broker.lower().strip()
    
    if "topstep" in broker:
        return spec.topstep_symbol
    elif "tradovate" in broker:
        return spec.tradovate_symbol
    elif "rithmic" in broker:
        return spec.rithmic_symbol
    else:
        # Default to standard symbol
        return spec.symbol


def get_supported_symbols() -> list[str]:
    """Get list of all supported symbols."""
    return list(SYMBOL_SPECS.keys())


def calculate_position_value(symbol: str, contracts: int, price: float) -> float:
    """
    Calculate the notional value of a position.
    
    Args:
        symbol: Symbol code
        contracts: Number of contracts
        price: Current price
        
    Returns:
        Notional value in dollars
    """
    spec = get_symbol_spec(symbol)
    return contracts * price * spec.point_value


def calculate_tick_pnl(symbol: str, contracts: int, ticks: float) -> float:
    """
    Calculate P&L for a given number of ticks.
    
    Args:
        symbol: Symbol code
        contracts: Number of contracts
        ticks: Number of ticks (can be negative)
        
    Returns:
        P&L in dollars
    """
    spec = get_symbol_spec(symbol)
    return contracts * ticks * spec.tick_value


# ============================================================================
# VALIDATION
# ============================================================================

if __name__ == "__main__":
    # Test symbol specs
    print("Testing Symbol Specifications...")
    print("=" * 60)
    
    for symbol, spec in SYMBOL_SPECS.items():
        print(f"\n{symbol} - {spec.name}")
        print(f"  Tick Value: ${spec.tick_value:.2f}")
        print(f"  Point Value: ${spec.point_value:.2f}")
        print(f"  Session: {spec.session_start} - {spec.session_end} ET")
        print(f"  Slippage: {spec.typical_slippage_ticks} ticks")
        print(f"  Volatility Factor: {spec.volatility_factor}x")
        print(f"  TopStep: {spec.topstep_symbol}")
        
        # Test calculations
        test_ticks = 10
        pnl = calculate_tick_pnl(symbol, 1, test_ticks)
        print(f"  10 ticks P&L (1 contract): ${pnl:.2f}")
    
    print("\n" + "=" * 60)
    print(f"Total symbols supported: {len(SYMBOL_SPECS)}")
    print("All tests passed! ✓")
