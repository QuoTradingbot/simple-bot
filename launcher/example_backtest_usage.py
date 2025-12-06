"""
Example: Using Recorded Market Data for Backtesting
====================================================

This example demonstrates how to load and use market data
recorded by the Market Data Recorder for backtesting purposes.

Each symbol has its own CSV file (ES.csv, NQ.csv, etc.)
"""

import csv
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable

# Configuration constants
TIMESTAMP_DISPLAY_LENGTH = 19  # Length of timestamp to display (YYYY-MM-DD HH:MM:SS)


def load_market_data(csv_file: str) -> List[Dict]:
    """
    Load market data from CSV file.
    
    Args:
        csv_file: Path to CSV file (e.g., 'market_data/ES.csv')
    
    Returns:
        List of data records
    """
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    return data


def get_quotes(data: List[Dict]) -> List[Dict]:
    """Extract only quote records."""
    return [d for d in data if d['data_type'] == 'quote']


def get_trades(data: List[Dict]) -> List[Dict]:
    """Extract only trade records."""
    return [d for d in data if d['data_type'] == 'trade']


def get_depth(data: List[Dict]) -> List[Dict]:
    """Extract only market depth records."""
    return [d for d in data if d['data_type'] == 'depth']


def calculate_spread(quote: Dict) -> float:
    """Calculate bid-ask spread from quote."""
    if quote['bid_price'] and quote['ask_price']:
        return float(quote['ask_price']) - float(quote['bid_price'])
    return 0.0


def analyze_market_data(csv_file: str, symbol: str):
    """
    Example analysis of recorded market data.
    
    Args:
        csv_file: Path to CSV file for the symbol (e.g., 'market_data/ES.csv')
        symbol: Symbol name (e.g., 'ES') - for display purposes
    """
    print(f"Analyzing market data for {symbol}")
    print("=" * 60)
    
    # Load all data from the symbol's file
    data = load_market_data(csv_file)
    print(f"Total records: {len(data)}")
    
    # Get different data types
    quotes = get_quotes(data)
    trades = get_trades(data)
    depth = get_depth(data)
    
    print(f"  Quotes: {len(quotes)}")
    print(f"  Trades: {len(trades)}")
    print(f"  Depth updates: {len(depth)}")
    print()
    
    # Analyze quotes
    if quotes:
        print("Quote Analysis:")
        spreads = [calculate_spread(q) for q in quotes if q['bid_price'] and q['ask_price']]
        if spreads:
            avg_spread = sum(spreads) / len(spreads)
            min_spread = min(spreads)
            max_spread = max(spreads)
            
            print(f"  Average spread: ${avg_spread:.2f}")
            print(f"  Min spread: ${min_spread:.2f}")
            print(f"  Max spread: ${max_spread:.2f}")
        
        # Show first few quotes
        print("\n  First 3 quotes:")
        for i, q in enumerate(quotes[:3]):
            ts = q['timestamp'][:TIMESTAMP_DISPLAY_LENGTH]
            print(f"    {i+1}. {ts} - Bid: {q['bid_price']} ({q['bid_size']}) | Ask: {q['ask_price']} ({q['ask_size']})")
    
    print()
    
    # Analyze trades
    if trades:
        print("Trade Analysis:")
        
        buy_trades = [t for t in trades if t['trade_side'] == 'buy']
        sell_trades = [t for t in trades if t['trade_side'] == 'sell']
        
        print(f"  Buy trades: {len(buy_trades)}")
        print(f"  Sell trades: {len(sell_trades)}")
        
        # Calculate total volume
        total_volume = sum(int(t['trade_size']) for t in trades if t['trade_size'])
        print(f"  Total volume: {total_volume} contracts")
        
        # Show first few trades
        print("\n  First 3 trades:")
        for i, t in enumerate(trades[:3]):
            ts = t['timestamp'][:TIMESTAMP_DISPLAY_LENGTH]
            print(f"    {i+1}. {ts} - {t['trade_side'].upper()} {t['trade_size']} @ {t['trade_price']}")
    
    print()
    
    # Analyze market depth
    if depth:
        print("Market Depth Analysis:")
        
        bid_levels = [d for d in depth if d['depth_side'] == 'bid']
        ask_levels = [d for d in depth if d['depth_side'] == 'ask']
        
        print(f"  Bid levels: {len(bid_levels)}")
        print(f"  Ask levels: {len(ask_levels)}")
        
        # Show sample depth
        if bid_levels:
            print("\n  Sample bid levels:")
            for i, d in enumerate(sorted(bid_levels, key=lambda x: int(x['depth_level']))[:3]):
                print(f"    Level {d['depth_level']}: {d['depth_price']} ({d['depth_size']})")
    
    print()
    print("=" * 60)


def replay_market_data(csv_file: str, symbol: str, limit: int = 100):
    """
    Replay market data chronologically (useful for backtesting).
    
    Args:
        csv_file: Path to CSV file for the symbol (e.g., 'market_data/ES.csv')
        symbol: Symbol name (e.g., 'ES') - for display purposes
        limit: Maximum number of records to replay
    """
    print(f"Replaying market data for {symbol} (first {limit} records)")
    print("=" * 60)
    
    data = load_market_data(csv_file)
    
    # Sort by timestamp to ensure chronological order
    data_sorted = sorted(data, key=lambda x: x['timestamp'])
    
    for i, record in enumerate(data_sorted[:limit]):
        timestamp = record['timestamp']
        data_type = record['data_type']
        
        if data_type == 'quote':
            print(f"[{timestamp}] QUOTE: Bid {record['bid_price']}x{record['bid_size']} | Ask {record['ask_price']}x{record['ask_size']}")
        elif data_type == 'trade':
            print(f"[{timestamp}] TRADE: {record['trade_side'].upper()} {record['trade_size']} @ {record['trade_price']}")
        elif data_type == 'depth':
            print(f"[{timestamp}] DEPTH: Level {record['depth_level']} {record['depth_side']} {record['depth_price']}x{record['depth_size']}")
        
        # In a real backtest, you would feed this data to your strategy here
        # strategy.on_market_data(record)
    
    print("=" * 60)


def simple_backtest_example(csv_file: str, symbol: str):
    """
    Simple backtest example using recorded data.
    
    This is a very basic example showing how to use the data.
    In a real backtest, you would implement your actual trading strategy.
    
    Args:
        csv_file: Path to CSV file for the symbol (e.g., 'market_data/ES.csv')
        symbol: Symbol name (e.g., 'ES') - for display purposes
    """
    print(f"Simple Backtest Example: {symbol}")
    print("=" * 60)
    
    # Load data
    data = load_market_data(csv_file)
    quotes = get_quotes(data)
    
    if not quotes:
        print("No quote data available")
        return
    
    # Simple strategy: Track when spread widens/narrows
    print("Strategy: Monitoring spread changes")
    print()
    
    spread_threshold = 0.50  # Alert when spread > $0.50
    wide_spread_count = 0
    
    for quote in quotes[:100]:  # Process first 100 quotes
        if quote['bid_price'] and quote['ask_price']:
            spread = calculate_spread(quote)
            
            if spread > spread_threshold:
                wide_spread_count += 1
                print(f"[{quote['timestamp'][:TIMESTAMP_DISPLAY_LENGTH]}] Wide spread detected: ${spread:.2f}")
                # In real backtest, you might avoid trading during wide spreads
    
    print()
    print(f"Wide spreads detected: {wide_spread_count} out of {len(quotes[:100])} quotes")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    csv_file = "market_data/ES.csv"  # Your recorded data file for ES
    symbol = "ES"
    
    print("Market Data Backtesting Example")
    print("=" * 60)
    print()
    print("NOTE: This example requires a recorded CSV file.")
    print("      Run the DataRecorder_Launcher to create one first.")
    print()
    print("=" * 60)
    print()
    
    # Uncomment these when you have actual recorded data:
    
    # Example 1: Analyze the data
    # analyze_market_data(csv_file, symbol)
    
    # Example 2: Replay the data
    # replay_market_data(csv_file, symbol, limit=50)
    
    # Example 3: Simple backtest
    # simple_backtest_example(csv_file, symbol)
    
    print("\nTo use this example:")
    print("1. Record some data using DataRecorder_Launcher.py")
    print("2. Update the csv_file path above to point to your recorded data")
    print("3. Uncomment the example functions you want to run")
    print("4. Run this script again")
