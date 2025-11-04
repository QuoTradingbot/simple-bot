"""
Analyze Recorded Live Data
Examines bid/ask spreads, order execution quality, and slippage
"""

import json
import gzip
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import statistics


@dataclass
class SpreadAnalysis:
    """Spread statistics"""
    symbol: str
    tick_count: int
    avg_spread: float
    min_spread: float
    max_spread: float
    median_spread: float
    spread_std: float
    tight_spreads_pct: float  # % of time spread <= 0.25


@dataclass
class OrderAnalysis:
    """Order execution analysis"""
    total_orders: int
    buy_orders: int
    sell_orders: int
    filled_orders: int
    
    avg_slippage: float
    buy_slippage: float
    sell_slippage: float
    
    worst_slippage: float
    best_fill: float


def load_jsonl_file(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSON Lines file (with optional gzip)"""
    records = []
    
    if filepath.suffix == '.gz':
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    
    return records


def analyze_spreads(tick_file: Path) -> SpreadAnalysis:
    """Analyze bid/ask spreads"""
    print(f"\nAnalyzing spreads from: {tick_file.name}")
    
    ticks = load_jsonl_file(tick_file)
    
    if not ticks:
        print("No tick data found!")
        return None
    
    spreads = [t['spread'] for t in ticks]
    symbol = ticks[0]['symbol']
    
    # Calculate stats
    tight_spreads = sum(1 for s in spreads if s <= 0.25)
    
    analysis = SpreadAnalysis(
        symbol=symbol,
        tick_count=len(spreads),
        avg_spread=round(statistics.mean(spreads), 4),
        min_spread=round(min(spreads), 4),
        max_spread=round(max(spreads), 4),
        median_spread=round(statistics.median(spreads), 4),
        spread_std=round(statistics.stdev(spreads), 4) if len(spreads) > 1 else 0,
        tight_spreads_pct=round(100 * tight_spreads / len(spreads), 2)
    )
    
    # Print results
    print("\n" + "="*60)
    print(f"SPREAD ANALYSIS - {symbol}")
    print("="*60)
    print(f"Total ticks: {analysis.tick_count:,}")
    print(f"Average spread: {analysis.avg_spread} points")
    print(f"Median spread: {analysis.median_spread} points")
    print(f"Min spread: {analysis.min_spread} points")
    print(f"Max spread: {analysis.max_spread} points")
    print(f"Spread std dev: {analysis.spread_std} points")
    print(f"Tight spreads (≤0.25): {analysis.tight_spreads_pct}%")
    print("="*60)
    
    # Show some sample ticks
    print("\nSample ticks:")
    for i, tick in enumerate(ticks[:5]):
        print(f"  {tick['datetime_str']}: Bid={tick['bid']}, Ask={tick['ask']}, Spread={tick['spread']}")
    
    return analysis


def analyze_orders(order_file: Path) -> OrderAnalysis:
    """Analyze order execution quality"""
    print(f"\nAnalyzing orders from: {order_file.name}")
    
    orders = load_jsonl_file(order_file)
    
    if not orders:
        print("No order data found!")
        return None
    
    # Separate by status
    filled_orders = [o for o in orders if o['status'] == 'filled' and o['slippage'] is not None]
    buy_orders = [o for o in orders if o['side'] == 'buy']
    sell_orders = [o for o in orders if o['side'] == 'sell']
    
    if not filled_orders:
        print("No filled orders found!")
        return None
    
    # Calculate slippage stats
    all_slippage = [o['slippage'] for o in filled_orders]
    buy_slippage = [o['slippage'] for o in filled_orders if o['side'] == 'buy']
    sell_slippage = [o['slippage'] for o in filled_orders if o['side'] == 'sell']
    
    analysis = OrderAnalysis(
        total_orders=len(orders),
        buy_orders=len(buy_orders),
        sell_orders=len(sell_orders),
        filled_orders=len(filled_orders),
        avg_slippage=round(statistics.mean(all_slippage), 4),
        buy_slippage=round(statistics.mean(buy_slippage), 4) if buy_slippage else 0,
        sell_slippage=round(statistics.mean(sell_slippage), 4) if sell_slippage else 0,
        worst_slippage=round(max(all_slippage), 4),
        best_fill=round(min(all_slippage), 4)
    )
    
    # Print results
    print("\n" + "="*60)
    print("ORDER EXECUTION ANALYSIS")
    print("="*60)
    print(f"Total orders: {analysis.total_orders}")
    print(f"  Buy orders: {analysis.buy_orders}")
    print(f"  Sell orders: {analysis.sell_orders}")
    print(f"Filled orders: {analysis.filled_orders}")
    print(f"\nSLIPPAGE (positive = unfavorable):")
    print(f"  Average: {analysis.avg_slippage} points")
    print(f"  Buy avg: {analysis.buy_slippage} points")
    print(f"  Sell avg: {analysis.sell_slippage} points")
    print(f"  Worst: {analysis.worst_slippage} points")
    print(f"  Best: {analysis.best_fill} points")
    print("="*60)
    
    # Show sample orders
    print("\nSample orders:")
    for i, order in enumerate(filled_orders[:5]):
        print(f"  {order['datetime_str']}: {order['side'].upper()} @ {order['fill_price']} "
              f"(market spread: {order['market_spread']}, slippage: {order['slippage']})")
    
    return analysis


def main():
    """Analyze all recordings in the live_data_recordings folder"""
    recordings_dir = Path("live_data_recordings")
    
    if not recordings_dir.exists():
        print(f"No recordings folder found at: {recordings_dir}")
        return
    
    print("="*60)
    print("LIVE DATA ANALYSIS")
    print("="*60)
    
    # Find all tick files
    tick_files = sorted(recordings_dir.glob("*_ticks_*.jsonl*"))
    order_files = sorted(recordings_dir.glob("*_orders_*.jsonl*"))
    
    print(f"\nFound {len(tick_files)} tick file(s)")
    print(f"Found {len(order_files)} order file(s)")
    
    # Analyze most recent files
    if tick_files:
        latest_tick_file = tick_files[-1]
        analyze_spreads(latest_tick_file)
    
    if order_files:
        latest_order_file = order_files[-1]
        analyze_orders(latest_order_file)
    
    if not tick_files and not order_files:
        print("\nNo recorded data found. Run record_live_data.py first!")


if __name__ == "__main__":
    main()
