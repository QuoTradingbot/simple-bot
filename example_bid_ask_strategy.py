"""
Example usage of the Bid/Ask Trading Strategy

This example demonstrates how the bid/ask manager works and the cost savings
it provides compared to traditional market orders.
"""

from bid_ask_manager import BidAskManager, BidAskQuote


def example_passive_vs_aggressive():
    """Demonstrate passive vs aggressive order placement."""
    
    print("=" * 70)
    print("BID/ASK TRADING STRATEGY - PASSIVE VS AGGRESSIVE EXAMPLE")
    print("=" * 70)
    print()
    
    # Configuration
    config = {
        "tick_size": 0.25,
        "passive_order_timeout": 10,
        "spread_lookback_periods": 100,
        "abnormal_spread_multiplier": 2.0,
        "high_volatility_spread_mult": 3.0,
        "calm_market_spread_mult": 1.5,
        "use_mixed_order_strategy": False
    }
    
    # Initialize bid/ask manager
    manager = BidAskManager(config)
    
    # Scenario 1: Normal market conditions (tight spread)
    print("SCENARIO 1: Normal Market Conditions")
    print("-" * 70)
    print()
    
    # Build baseline with normal spreads
    print("Building spread baseline...")
    for i in range(30):
        manager.update_quote(
            symbol="ES",
            bid_price=4500.00,
            ask_price=4500.25,  # Normal 1-tick spread
            bid_size=10,
            ask_size=8,
            last_price=4500.25,
            timestamp=1000000 + i
        )
    
    stats = manager.get_spread_statistics("ES")
    print(f"  Average spread: ${stats['average_spread']:.2f}")
    print(f"  Samples: {stats['current_samples']}")
    print()
    
    # Get order parameters for long entry
    params = manager.get_entry_order_params("ES", "long", 1)
    
    print(f"Long Entry Decision:")
    print(f"  Strategy: {params['strategy'].upper()}")
    print(f"  Reason: {params['reason']}")
    print(f"  Entry Price: ${params['limit_price']:.2f}")
    if 'fallback_price' in params:
        print(f"  Fallback Price: ${params['fallback_price']:.2f}")
    print(f"  Timeout: {params['timeout']}s")
    print()
    
    # Calculate savings
    quote = params['quote']
    passive_cost = quote.bid_price - quote.bid_price  # $0 (join bid)
    aggressive_cost = quote.ask_price - quote.bid_price  # Full spread
    savings = aggressive_cost - passive_cost
    
    print(f"Cost Analysis:")
    print(f"  Market order (aggressive): Pay ${aggressive_cost:.2f} spread")
    print(f"  Limit order at bid (passive): Save ${savings:.2f}")
    print(f"  Savings per entry: ${savings:.2f}")
    print()
    print()
    
    # Scenario 2: Wide spread (high volatility)
    print("SCENARIO 2: High Volatility (Wide Spread)")
    print("-" * 70)
    print()
    
    # Update with wide spread
    manager.update_quote(
        symbol="ES",
        bid_price=4500.00,
        ask_price=4501.00,  # 4-tick spread (abnormal)
        bid_size=5,
        ask_size=3,
        last_price=4500.50,
        timestamp=2000000
    )
    
    quote = manager.get_current_quote("ES")
    print(f"Current Market:")
    print(f"  Bid: ${quote.bid_price:.2f} x {quote.bid_size}")
    print(f"  Ask: ${quote.ask_price:.2f} x {quote.ask_size}")
    print(f"  Spread: ${quote.spread:.2f} (${stats['average_spread']:.2f} avg)")
    print()
    
    # Check if spread is acceptable
    is_acceptable, reason = manager.validate_entry_spread("ES")
    print(f"Spread Validation:")
    print(f"  Acceptable: {is_acceptable}")
    print(f"  Reason: {reason}")
    print()
    
    if not is_acceptable:
        print("⚠️  Entry REJECTED - Spread too wide")
        print("   Bot will wait for better market conditions")
    
    print()
    print()
    
    # Scenario 3: Cost savings over time
    print("SCENARIO 3: Annual Cost Savings")
    print("-" * 70)
    print()
    
    trades_per_year = 100
    tick_value = 12.50  # ES futures
    spread_ticks = 1  # Normal 1-tick spread
    passive_fill_rate = 0.80  # 80% of orders fill passively
    
    # Traditional market orders
    market_order_cost = trades_per_year * 2 * spread_ticks * tick_value  # Entry + Exit
    
    # Bid/ask strategy
    passive_fills = trades_per_year * passive_fill_rate
    aggressive_fills = trades_per_year * (1 - passive_fill_rate)
    
    # Passive saves on entry, still pays on exit
    passive_cost = passive_fills * spread_ticks * tick_value  # Exit only
    aggressive_cost = aggressive_fills * 2 * spread_ticks * tick_value  # Entry + Exit
    bidask_total_cost = passive_cost + aggressive_cost
    
    annual_savings = market_order_cost - bidask_total_cost
    
    print(f"Traditional Market Orders:")
    print(f"  Trades per year: {trades_per_year}")
    print(f"  Cost per round-trip: ${spread_ticks * tick_value * 2:.2f}")
    print(f"  Annual cost: ${market_order_cost:,.2f}")
    print()
    
    print(f"Bid/Ask Strategy:")
    print(f"  Passive fill rate: {passive_fill_rate * 100:.0f}%")
    print(f"  Passive fills: {passive_fills:.0f} trades")
    print(f"  Aggressive fills: {aggressive_fills:.0f} trades")
    print(f"  Annual cost: ${bidask_total_cost:,.2f}")
    print()
    
    print(f"💰 Annual Savings: ${annual_savings:,.2f} ({annual_savings/market_order_cost*100:.0f}% reduction)")
    print()
    
    print("=" * 70)


def example_mixed_strategy():
    """Demonstrate mixed order strategy."""
    
    print()
    print("=" * 70)
    print("MIXED ORDER STRATEGY EXAMPLE")
    print("=" * 70)
    print()
    
    # Configuration with mixed strategy enabled
    config = {
        "tick_size": 0.25,
        "passive_order_timeout": 10,
        "spread_lookback_periods": 100,
        "abnormal_spread_multiplier": 2.0,
        "high_volatility_spread_mult": 3.0,
        "calm_market_spread_mult": 1.5,
        "use_mixed_order_strategy": True,  # Enabled
        "mixed_passive_ratio": 0.6  # 60% passive, 40% aggressive
    }
    
    manager = BidAskManager(config)
    
    # Build baseline
    for i in range(30):
        manager.update_quote(
            symbol="ES",
            bid_price=4500.00,
            ask_price=4500.25,
            bid_size=15,
            ask_size=12,
            last_price=4500.25,
            timestamp=1000000 + i
        )
    
    print("Trading 5 contracts with mixed strategy:")
    print()
    
    # Get order parameters for multi-contract order
    params = manager.get_entry_order_params("ES", "long", 5)
    
    print(f"Strategy: {params['strategy'].upper()}")
    print()
    
    if params['strategy'] == 'mixed':
        print(f"Order Split:")
        print(f"  Passive: {params['passive_contracts']} contracts @ ${params['passive_price']:.2f}")
        print(f"  Aggressive: {params['aggressive_contracts']} contracts @ ${params['aggressive_price']:.2f}")
        print()
        
        # Calculate weighted average fill price
        total_contracts = params['passive_contracts'] + params['aggressive_contracts']
        avg_fill = (params['passive_price'] * params['passive_contracts'] + 
                   params['aggressive_price'] * params['aggressive_contracts']) / total_contracts
        
        print(f"Weighted Average Fill: ${avg_fill:.2f}")
        print()
        
        # Calculate cost vs pure aggressive
        quote = params['quote']
        mixed_cost = avg_fill - quote.bid_price
        aggressive_cost = quote.ask_price - quote.bid_price
        savings = aggressive_cost - mixed_cost
        
        print(f"Cost per contract:")
        print(f"  Pure aggressive: ${aggressive_cost:.2f}")
        print(f"  Mixed strategy: ${mixed_cost:.2f}")
        print(f"  Savings: ${savings:.2f} per contract")
        print(f"  Total savings (5 contracts): ${savings * 5:.2f}")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    example_passive_vs_aggressive()
    example_mixed_strategy()
    
    print()
    print("✅ Examples complete!")
    print()
    print("Key Takeaways:")
    print("  1. Passive orders save the spread when filled")
    print("  2. Aggressive orders guarantee fills but cost more")
    print("  3. Bot adapts strategy based on market conditions")
    print("  4. Annual savings can be significant (thousands of dollars)")
    print("  5. Mixed strategy balances fill rate and cost")
    print()
