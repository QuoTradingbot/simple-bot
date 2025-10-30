"""
Tests for Bid/Ask Manager
Validates spread analysis and intelligent order placement logic.
"""

import unittest
from bid_ask_manager import (
    BidAskManager, BidAskQuote, SpreadAnalyzer, 
    OrderPlacementStrategy, DynamicFillStrategy
)


class TestBidAskQuote(unittest.TestCase):
    """Test BidAskQuote dataclass."""
    
    def test_spread_calculation(self):
        """Test bid/ask spread calculation."""
        quote = BidAskQuote(
            bid_price=4500.00,
            ask_price=4500.25,
            bid_size=10,
            ask_size=8,
            last_trade_price=4500.25,
            timestamp=1000000
        )
        
        self.assertEqual(quote.spread, 0.25)
        self.assertEqual(quote.mid_price, 4500.125)


class TestSpreadAnalyzer(unittest.TestCase):
    """Test SpreadAnalyzer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SpreadAnalyzer(lookback_periods=100, abnormal_multiplier=2.0)
    
    def test_initial_state(self):
        """Test initial analyzer state."""
        self.assertIsNone(self.analyzer.average_spread)
        self.assertIsNone(self.analyzer.std_dev_spread)
    
    def test_spread_update(self):
        """Test spread history update."""
        # Add some spread samples
        spreads = [0.25, 0.50, 0.25, 0.50, 0.25] * 5  # 25 samples
        for spread in spreads:
            self.analyzer.update(spread)
        
        self.assertIsNotNone(self.analyzer.average_spread)
        self.assertAlmostEqual(self.analyzer.average_spread, 0.35, places=2)
    
    def test_acceptable_spread(self):
        """Test acceptable spread detection."""
        # Build baseline
        normal_spreads = [0.25] * 30
        for spread in normal_spreads:
            self.analyzer.update(spread)
        
        # Test normal spread
        is_acceptable, reason = self.analyzer.is_spread_acceptable(0.30)
        self.assertTrue(is_acceptable)
        
        # Test abnormal spread (>2x average)
        is_acceptable, reason = self.analyzer.is_spread_acceptable(1.00)
        self.assertFalse(is_acceptable)
        self.assertIn("too wide", reason.lower())
    
    def test_spread_statistics(self):
        """Test spread statistics calculation."""
        spreads = [0.25, 0.50, 0.75, 1.00, 0.25]
        for spread in spreads:
            self.analyzer.update(spread)
        
        stats = self.analyzer.get_spread_stats()
        self.assertEqual(stats['current_samples'], 5)
        self.assertEqual(stats['min_spread'], 0.25)
        self.assertEqual(stats['max_spread'], 1.00)


class TestOrderPlacementStrategy(unittest.TestCase):
    """Test order placement strategy logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "tick_size": 0.25,
            "passive_order_timeout": 10,
            "high_volatility_spread_mult": 3.0,
            "calm_market_spread_mult": 1.5
        }
        self.strategy = OrderPlacementStrategy(self.config)
    
    def test_passive_entry_calculation_long(self):
        """Test passive entry price for long positions."""
        quote = BidAskQuote(
            bid_price=4500.00,
            ask_price=4500.25,
            bid_size=10,
            ask_size=8,
            last_trade_price=4500.25,
            timestamp=1000000
        )
        
        # For long: join sellers at bid
        passive_price = self.strategy.calculate_passive_entry_price("long", quote)
        self.assertEqual(passive_price, 4500.00)
    
    def test_passive_entry_calculation_short(self):
        """Test passive entry price for short positions."""
        quote = BidAskQuote(
            bid_price=4500.00,
            ask_price=4500.25,
            bid_size=10,
            ask_size=8,
            last_trade_price=4500.25,
            timestamp=1000000
        )
        
        # For short: join buyers at ask
        passive_price = self.strategy.calculate_passive_entry_price("short", quote)
        self.assertEqual(passive_price, 4500.25)
    
    def test_aggressive_entry_calculation_long(self):
        """Test aggressive entry price for long positions."""
        quote = BidAskQuote(
            bid_price=4500.00,
            ask_price=4500.25,
            bid_size=10,
            ask_size=8,
            last_trade_price=4500.25,
            timestamp=1000000
        )
        
        # For long: pay the ask
        aggressive_price = self.strategy.calculate_aggressive_entry_price("long", quote)
        self.assertEqual(aggressive_price, 4500.25)
    
    def test_aggressive_entry_calculation_short(self):
        """Test aggressive entry price for short positions."""
        quote = BidAskQuote(
            bid_price=4500.00,
            ask_price=4500.25,
            bid_size=10,
            ask_size=8,
            last_trade_price=4500.25,
            timestamp=1000000
        )
        
        # For short: hit the bid
        aggressive_price = self.strategy.calculate_aggressive_entry_price("short", quote)
        self.assertEqual(aggressive_price, 4500.00)
    
    def test_should_use_passive_calm_market(self):
        """Test passive entry decision in calm market."""
        quote = BidAskQuote(
            bid_price=4500.00,
            ask_price=4500.25,
            bid_size=10,
            ask_size=8,
            last_trade_price=4500.25,
            timestamp=1000000
        )
        
        # Create spread analyzer with baseline
        analyzer = SpreadAnalyzer()
        for _ in range(30):
            analyzer.update(0.25)  # Normal spread
        
        use_passive, reason = self.strategy.should_use_passive_entry(quote, analyzer)
        self.assertTrue(use_passive)
        self.assertIn("tight spread", reason.lower())
    
    def test_should_use_aggressive_wide_spread(self):
        """Test aggressive entry decision with wide spread."""
        quote = BidAskQuote(
            bid_price=4500.00,
            ask_price=4501.00,  # Wide spread
            bid_size=10,
            ask_size=8,
            last_trade_price=4500.50,
            timestamp=1000000
        )
        
        # Create spread analyzer with baseline
        analyzer = SpreadAnalyzer()
        for _ in range(30):
            analyzer.update(0.25)  # Normal spread is much tighter
        
        use_passive, reason = self.strategy.should_use_passive_entry(quote, analyzer)
        self.assertFalse(use_passive)
        self.assertIn("wide spread", reason.lower())


class TestDynamicFillStrategy(unittest.TestCase):
    """Test dynamic fill strategy logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "passive_order_timeout": 10,
            "use_mixed_order_strategy": True,
            "mixed_passive_ratio": 0.5
        }
        self.fill_strategy = DynamicFillStrategy(self.config)
    
    def test_no_mixed_for_single_contract(self):
        """Test that mixed strategy is not used for single contract."""
        use_mixed, passive_qty, aggressive_qty = self.fill_strategy.should_use_mixed_strategy(1)
        self.assertFalse(use_mixed)
    
    def test_mixed_strategy_split(self):
        """Test mixed strategy contract split."""
        use_mixed, passive_qty, aggressive_qty = self.fill_strategy.should_use_mixed_strategy(4)
        self.assertTrue(use_mixed)
        self.assertEqual(passive_qty, 2)
        self.assertEqual(aggressive_qty, 2)
    
    def test_retry_strategy_progression(self):
        """Test retry strategy progression."""
        # First attempt - passive
        retry1 = self.fill_strategy.get_retry_strategy(1, max_attempts=3)
        self.assertEqual(retry1['strategy'], 'passive')
        
        # Second attempt - still passive but shorter timeout
        retry2 = self.fill_strategy.get_retry_strategy(2, max_attempts=3)
        self.assertEqual(retry2['strategy'], 'passive')
        self.assertLess(retry2['timeout'], retry1['timeout'])
        
        # Final attempt - aggressive
        retry3 = self.fill_strategy.get_retry_strategy(3, max_attempts=3)
        self.assertEqual(retry3['strategy'], 'aggressive')


class TestBidAskManager(unittest.TestCase):
    """Test complete BidAskManager integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "tick_size": 0.25,
            "passive_order_timeout": 10,
            "spread_lookback_periods": 100,
            "abnormal_spread_multiplier": 2.0,
            "high_volatility_spread_mult": 3.0,
            "calm_market_spread_mult": 1.5,
            "use_mixed_order_strategy": False
        }
        self.manager = BidAskManager(self.config)
    
    def test_quote_update(self):
        """Test quote update and retrieval."""
        self.manager.update_quote(
            symbol="ES",
            bid_price=4500.00,
            ask_price=4500.25,
            bid_size=10,
            ask_size=8,
            last_price=4500.25,
            timestamp=1000000
        )
        
        quote = self.manager.get_current_quote("ES")
        self.assertIsNotNone(quote)
        self.assertEqual(quote.bid_price, 4500.00)
        self.assertEqual(quote.ask_price, 4500.25)
    
    def test_spread_validation(self):
        """Test spread validation for entry."""
        # Build baseline with normal spreads
        for i in range(30):
            self.manager.update_quote(
                symbol="ES",
                bid_price=4500.00,
                ask_price=4500.25,
                bid_size=10,
                ask_size=8,
                last_price=4500.25,
                timestamp=1000000 + i
            )
        
        # Test with normal spread
        is_acceptable, reason = self.manager.validate_entry_spread("ES")
        self.assertTrue(is_acceptable)
        
        # Update with abnormally wide spread
        self.manager.update_quote(
            symbol="ES",
            bid_price=4500.00,
            ask_price=4501.00,  # Very wide
            bid_size=10,
            ask_size=8,
            last_price=4500.50,
            timestamp=2000000
        )
        
        is_acceptable, reason = self.manager.validate_entry_spread("ES")
        self.assertFalse(is_acceptable)
    
    def test_entry_order_params_passive(self):
        """Test entry order parameters with passive strategy."""
        # Build baseline
        for i in range(30):
            self.manager.update_quote(
                symbol="ES",
                bid_price=4500.00,
                ask_price=4500.25,
                bid_size=10,
                ask_size=8,
                last_price=4500.25,
                timestamp=1000000 + i
            )
        
        # Get order params for long entry
        params = self.manager.get_entry_order_params("ES", "long", 1)
        
        self.assertEqual(params['strategy'], 'passive')
        self.assertEqual(params['limit_price'], 4500.00)  # Bid price for long
        self.assertIn('fallback_price', params)
        self.assertEqual(params['fallback_price'], 4500.25)  # Ask price
    
    def test_entry_order_params_aggressive(self):
        """Test entry order parameters with aggressive strategy (wide spread)."""
        # Build baseline with normal spreads
        for i in range(30):
            self.manager.update_quote(
                symbol="ES",
                bid_price=4500.00,
                ask_price=4500.25,
                bid_size=10,
                ask_size=8,
                last_price=4500.25,
                timestamp=1000000 + i
            )
        
        # Update with wide spread
        self.manager.update_quote(
            symbol="ES",
            bid_price=4500.00,
            ask_price=4501.00,  # Wide spread
            bid_size=10,
            ask_size=8,
            last_price=4500.50,
            timestamp=2000000
        )
        
        params = self.manager.get_entry_order_params("ES", "long", 1)
        
        self.assertEqual(params['strategy'], 'aggressive')
        self.assertEqual(params['limit_price'], 4501.00)  # Ask price for long
    
    def test_spread_statistics(self):
        """Test spread statistics retrieval."""
        # Add some quotes
        for i in range(30):
            self.manager.update_quote(
                symbol="ES",
                bid_price=4500.00,
                ask_price=4500.25,
                bid_size=10,
                ask_size=8,
                last_price=4500.25,
                timestamp=1000000 + i
            )
        
        stats = self.manager.get_spread_statistics("ES")
        self.assertIsNotNone(stats.get('average_spread'))
        self.assertEqual(stats['current_samples'], 30)


if __name__ == '__main__':
    unittest.main()
