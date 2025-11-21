"""
Backtesting Framework for VWAP Bounce Bot
Supports historical data loading, order simulation, and performance analysis
"""

import csv
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field
import pytz
import os

# Export parameter optimization classes
__all__ = [
    'BacktestConfig', 'BacktestEngine', 'HistoricalDataLoader', 
    'PerformanceMetrics', 'ReportGenerator', 'Trade'
]


@dataclass
class Trade:
    """Represents a completed trade"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str  # 'long' or 'short'
    quantity: int
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    exit_reason: str
    pnl: float
    ticks: float
    duration_minutes: float
    

@dataclass
class BacktestConfig:
    """
    Configuration for backtesting.
    Backtesting is completely independent of broker API and uses only historical data.
    """
    start_date: datetime
    end_date: datetime
    """Enhanced backtesting engine with realistic slippage and costs."""
    
    initial_equity: float = 50000.0
    max_drawdown: float = 0.0
    symbols: List[str] = field(default_factory=lambda: ["MES"])
    slippage_ticks: float = 0.5  # Average slippage in ticks
    commission_per_contract: float = 2.50  # Round-trip commission
    data_source: str = "csv"  # "csv" for local files (no API needed)
    data_path: str = "data/historical_data"
    use_tick_data: bool = False  # Use tick-by-tick replay (default: bar-by-bar with 1min bars)
    

class HistoricalDataLoader:
    """Loads and validates historical tick/bar data"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def _normalize_timestamp(self, timestamp: datetime) -> datetime:
        """
        Normalize timestamp to be timezone-aware in UTC.
        
        Args:
            timestamp: Datetime to normalize
            
        Returns:
            Timezone-aware datetime in UTC
        """
        if timestamp.tzinfo is None:
            return pytz.UTC.localize(timestamp)
        return timestamp.astimezone(pytz.UTC)
    
    def _normalize_date_range(self) -> Tuple[datetime, datetime]:
        """
        Normalize config date range to be timezone-aware.
        
        Returns:
            Tuple of (start_date, end_date) in UTC
        """
        start_date = self.config.start_date
        end_date = self.config.end_date
        
        if start_date.tzinfo is None:
            start_date = pytz.UTC.localize(start_date)
        else:
            start_date = start_date.astimezone(pytz.UTC)
            
        if end_date.tzinfo is None:
            end_date = pytz.UTC.localize(end_date)
        else:
            end_date = end_date.astimezone(pytz.UTC)
            
        return start_date, end_date
        
    def load_tick_data(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Load tick data from CSV file.
        Expected CSV format: timestamp,price,size (or volume)
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            List of tick dictionaries
        """
        ticks = []
        
        # Try enhanced tick data first (has realistic intra-bar price movement)
        filepath = os.path.join(self.config.data_path, f"{symbol}_ticks_enhanced.csv")
        if not os.path.exists(filepath):
            # Fall back to regular tick data
            filepath = os.path.join(self.config.data_path, f"{symbol}_ticks.csv")
            if not os.path.exists(filepath):
                self.logger.warning(f"Tick data file not found: {filepath}")
                return ticks
            else:
                self.logger.info(f"Using regular tick data (enhanced not available)")
        else:
            self.logger.info(f"Using ENHANCED tick data for realistic ATR calculation")
            
        try:
            start_date, end_date = self._normalize_date_range()
            
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    timestamp = datetime.fromisoformat(row['timestamp'])
                    timestamp = self._normalize_timestamp(timestamp)
                    
                    # Filter by date range
                    if start_date <= timestamp <= end_date:
                        # Handle both 'size' and 'volume' column names
                        volume = int(row.get('size', row.get('volume', 1)))
                        ticks.append({
                            'timestamp': timestamp,
                            'price': float(row['price']),
                            'volume': volume
                        })
                        
            self.logger.info(f"Loaded {len(ticks):,} ticks for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error loading tick data: {e}")
            
        return ticks
    
    def load_bar_data(self, symbol: str, timeframe: str = "1min") -> List[Dict[str, Any]]:
        """
        Load bar data from CSV file.
        Expected CSV format: timestamp,open,high,low,close,volume
        
        Args:
            symbol: Instrument symbol
            timeframe: Bar timeframe (e.g., "1min", "15min")
            
        Returns:
            List of bar dictionaries
        """
        bars = []
        
        filepath = os.path.join(self.config.data_path, f"{symbol}_{timeframe}.csv")
        
        if not os.path.exists(filepath):
            self.logger.warning(f"Bar data file not found: {filepath}")
            return bars
            
        try:
            start_date, end_date = self._normalize_date_range()
            
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    timestamp = datetime.fromisoformat(row['timestamp'])
                    timestamp = self._normalize_timestamp(timestamp)
                    
                    # Filter by date range
                    if start_date <= timestamp <= end_date:
                        bars.append({
                            'timestamp': timestamp,
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': int(row['volume'])
                        })
                        
            self.logger.info(f"Loaded {len(bars)} {timeframe} bars for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error loading bar data: {e}")
            
        return bars
    
    def _aggregate_ticks_to_bars(self, ticks: List[Dict[str, Any]], timeframe: str = "1min") -> List[Dict[str, Any]]:
        """
        Aggregate tick data into OHLCV bars.
        
        Args:
            ticks: List of tick dictionaries with 'timestamp', 'price', 'volume'
            timeframe: Timeframe for aggregation (e.g., "1min", "5min", "15min")
            
        Returns:
            List of bar dictionaries with OHLCV data
        """
        if not ticks:
            return []
        
        # Parse timeframe
        interval_minutes = int(timeframe.replace("min", ""))
        bars = []
        current_bar = None
        
        for tick in ticks:
            tick_time = tick['timestamp']
            
            # Round timestamp to bar interval
            bar_time = tick_time.replace(second=0, microsecond=0)
            bar_time = bar_time.replace(minute=(bar_time.minute // interval_minutes) * interval_minutes)
            
            # Start new bar if needed
            if current_bar is None or current_bar['timestamp'] != bar_time:
                if current_bar is not None:
                    bars.append(current_bar)
                
                current_bar = {
                    'timestamp': bar_time,
                    'open': tick['price'],
                    'high': tick['price'],
                    'low': tick['price'],
                    'close': tick['price'],
                    'volume': tick['volume']
                }
            else:
                # Update current bar
                current_bar['high'] = max(current_bar['high'], tick['price'])
                current_bar['low'] = min(current_bar['low'], tick['price'])
                current_bar['close'] = tick['price']
                current_bar['volume'] += tick['volume']
        
        # Add final bar
        if current_bar is not None:
            bars.append(current_bar)
        
        return bars
    
    def validate_data_quality(self, data: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate data quality and identify issues.
        
        Args:
            data: List of tick or bar data
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        if len(data) == 0:
            issues.append("No data available")
            return False, issues
            
        # Check for gaps
        if len(data) > 1:
            timestamps = [d['timestamp'] for d in data]
            for i in range(1, len(timestamps)):
                gap = (timestamps[i] - timestamps[i-1]).total_seconds()
                if gap > 3600:  # Gap > 1 hour
                    issues.append(f"Data gap detected: {gap/3600:.1f} hours at {timestamps[i]}")
        
        # Check for invalid prices
        for i, d in enumerate(data):
            price = d.get('price') or d.get('close')
            if price is None or price <= 0:
                issues.append(f"Invalid price at index {i}: {price}")
                
        # Check for invalid volumes
        for i, d in enumerate(data):
            volume = d.get('volume')
            if volume is None or volume < 0:
                issues.append(f"Invalid volume at index {i}: {volume}")
        
        is_valid = len(issues) == 0
        return is_valid, issues


class OrderFillSimulator:
    """Simulates realistic order fills in backtesting"""
    
    def __init__(self, tick_size: float, tick_value: float, slippage_ticks: float = 0.5):
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.slippage_ticks = slippage_ticks
        self.logger = logging.getLogger(__name__)
        
    def simulate_market_order(self, side: str, price: float, next_bar: Dict[str, Any]) -> float:
        """
        Simulate market order fill at next bar's open with slippage.
        
        Args:
            side: 'BUY' or 'SELL'
            price: Order price (for reference)
            next_bar: Next bar data
            
        Returns:
            Fill price
        """
        # Market orders fill at next bar open plus slippage
        fill_price = next_bar['open']
        
        if side == 'BUY':
            # Buying - pay slippage (worse price)
            fill_price += self.slippage_ticks * self.tick_size
        else:  # SELL
            # Selling - receive slippage (worse price)
            fill_price -= self.slippage_ticks * self.tick_size
            
        return fill_price
    
    def simulate_stop_order(self, side: str, stop_price: float, bar: Dict[str, Any]) -> Optional[float]:
        """
        Simulate stop order - triggers when price crosses stop level.
        
        Args:
            side: 'BUY' or 'SELL'
            stop_price: Stop trigger price
            bar: Current bar data
            
        Returns:
            Fill price if triggered, None otherwise
        """
        if side == 'BUY':
            # Buy stop triggers when price goes above stop
            if bar['high'] >= stop_price:
                # Fill at stop price plus slippage
                return stop_price + (self.slippage_ticks * self.tick_size)
        else:  # SELL
            # Sell stop triggers when price goes below stop
            if bar['low'] <= stop_price:
                # Fill at stop price minus slippage
                return stop_price - (self.slippage_ticks * self.tick_size)
                
        return None
    
    def simulate_limit_order(self, side: str, limit_price: float, bar: Dict[str, Any]) -> Optional[float]:
        """
        Simulate limit order - fills when price reaches limit.
        
        Args:
            side: 'BUY' or 'SELL'
            limit_price: Limit price
            bar: Current bar data
            
        Returns:
            Fill price if executed, None otherwise
        """
        if side == 'BUY':
            # Buy limit fills when price drops to or below limit
            if bar['low'] <= limit_price:
                return limit_price  # Fill at limit (best case)
        else:  # SELL
            # Sell limit fills when price rises to or above limit
            if bar['high'] >= limit_price:
                return limit_price  # Fill at limit (best case)
                
        return None


class PerformanceMetrics:
    """Calculates and tracks backtest performance metrics"""
    
    def __init__(self, initial_equity: float, tick_value: float, commission_per_contract: float):
        self.initial_equity = initial_equity
        self.tick_value = tick_value
        self.commission_per_contract = commission_per_contract
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = [(datetime.now(), initial_equity)]
        
    def add_trade(self, trade: Trade) -> None:
        """Add a completed trade to the metrics"""
        self.trades.append(trade)
        
        # Update equity curve
        if len(self.equity_curve) > 0:
            last_equity = self.equity_curve[-1][1]
        else:
            last_equity = self.initial_equity
            
        new_equity = last_equity + trade.pnl
        self.equity_curve.append((trade.exit_time, new_equity))
        
    def calculate_total_pnl(self) -> float:
        """Calculate total profit/loss"""
        return sum(t.pnl for t in self.trades)
    
    def calculate_win_rate(self) -> float:
        """Calculate percentage of winning trades"""
        if len(self.trades) == 0:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return (wins / len(self.trades)) * 100
    
    def calculate_average_win_loss(self) -> Tuple[float, float]:
        """Calculate average win and average loss"""
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl < 0]
        
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        
        return avg_win, avg_loss
    
    def calculate_max_drawdown(self) -> Tuple[float, float]:
        """
        Calculate maximum drawdown in dollars and percentage.
        
        Returns:
            Tuple of (max_drawdown_dollars, max_drawdown_percent)
        """
        if len(self.equity_curve) == 0:
            return 0.0, 0.0
            
        peak = self.initial_equity
        max_dd = 0.0
        
        for _, equity in self.equity_curve:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            if drawdown > max_dd:
                max_dd = drawdown
                
        max_dd_percent = (max_dd / peak * 100) if peak > 0 else 0.0
        return max_dd, max_dd_percent
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio (annualized).
        
        Args:
            risk_free_rate: Annual risk-free rate (default 0%)
            
        Returns:
            Sharpe ratio
        """
        if len(self.trades) < 2:
            return 0.0
            
        # Calculate daily returns
        returns = [t.pnl for t in self.trades]
        
        # Calculate statistics
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return 0.0
            
        # Annualize (assuming ~252 trading days)
        sharpe = (mean_return - risk_free_rate) / std_dev
        sharpe_annualized = sharpe * (252 ** 0.5)
        
        return sharpe_annualized
    
    def calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
            
        return gross_profit / gross_loss
    
    def calculate_time_in_market(self) -> float:
        """Calculate percentage of time spent in positions"""
        if len(self.trades) == 0:
            return 0.0
            
        total_time_in_position = sum(t.duration_minutes for t in self.trades)
        
        # Calculate total backtest duration
        if len(self.equity_curve) > 1:
            start_time = self.equity_curve[0][0]
            end_time = self.equity_curve[-1][0]
            
            # Normalize to timezone-naive for comparison
            if start_time.tzinfo is not None:
                start_time = start_time.replace(tzinfo=None)
            if end_time.tzinfo is not None:
                end_time = end_time.replace(tzinfo=None)
                
            total_duration = (end_time - start_time).total_seconds() / 60.0
        else:
            return 0.0
            
        if total_duration == 0:
            return 0.0
            
        return (total_time_in_position / total_duration) * 100
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        avg_win, avg_loss = self.calculate_average_win_loss()
        max_dd_dollars, max_dd_percent = self.calculate_max_drawdown()
        
        return {
            'total_trades': len(self.trades),
            'total_pnl': self.calculate_total_pnl(),
            'win_rate': self.calculate_win_rate(),
            'average_win': avg_win,
            'average_loss': avg_loss,
            'max_drawdown_dollars': max_dd_dollars,
            'max_drawdown_percent': max_dd_percent,
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'profit_factor': self.calculate_profit_factor(),
            'time_in_market_percent': self.calculate_time_in_market(),
            'final_equity': self.equity_curve[-1][1] if self.equity_curve else self.initial_equity,
            'total_return': ((self.equity_curve[-1][1] / self.initial_equity - 1) * 100) if self.equity_curve else 0.0
        }


class BacktestEngine:
    """
    Main backtesting engine that replays historical data through the bot.
    
    This engine is completely independent of the broker API and runs using
    only historical data. It simulates live trading by:
    1. Loading historical tick and bar data from CSV files
    2. Replaying each tick/bar as if it's happening in real-time
    3. Executing the trading strategy on historical data
    4. Simulating realistic order fills with slippage
    5. Tracking performance metrics
    
    No broker connection or API token is needed for backtesting.
    """
    
    def __init__(self, config: BacktestConfig, bot_config: Dict[str, Any]):
        self.config = config
        self.bot_config = bot_config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_loader = HistoricalDataLoader(config)
        self.order_simulator = OrderFillSimulator(
            tick_size=bot_config.get('tick_size', 0.25),
            tick_value=bot_config.get('tick_value', 1.25),
            slippage_ticks=config.slippage_ticks
        )
        self.metrics = PerformanceMetrics(
            initial_equity=config.initial_equity,
            tick_value=bot_config.get('tick_value', 1.25),
            commission_per_contract=config.commission_per_contract
        )
        
        # Backtest state
        self.current_equity = config.initial_equity
        self.current_position: Optional[Dict[str, Any]] = None
        self.pending_orders: List[Dict[str, Any]] = []
        
        # RL brain tracking
        self.initial_signal_count = 0
        self.bot_instance = None  # Will be set when bot is created
        
    def run(self, strategy_func: Any) -> Dict[str, Any]:
        """
        LEGACY METHOD - Kept for backward compatibility.
        Use run_with_strategy() for integrated bot strategy.
        
        Args:
            strategy_func: Function that implements trading strategy logic
            
        Returns:
            Performance metrics dictionary
        """
        return self.run_with_strategy(strategy_func)
    
    def run_with_strategy(self, strategy_func: Any) -> Dict[str, Any]:
        """
        Run the backtest with integrated bot strategy.
        
        Args:
            strategy_func: Function that receives bars and executes strategy logic.
                          Should be: func(bars_1min: List[Dict], bars_15min: List[Dict]) -> None
            
        Returns:
            Performance metrics dictionary
        """
        self.logger.info("="*60)
        self.logger.info("Starting Backtest")
        self.logger.info("="*60)
        self.logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")
        self.logger.info(f"Initial Equity: ${self.config.initial_equity:,.2f}")
        self.logger.info(f"Symbols: {', '.join(self.config.symbols)}")
        
        for symbol in self.config.symbols:
            self._run_symbol_backtest_integrated(symbol, strategy_func)
            
        # Calculate and return metrics
        results = self.metrics.get_summary()
        
        self.logger.info("="*60)
        self.logger.info("Backtest Complete")
        self.logger.info("="*60)
        self._log_results(results)
        
        return results
    
    def _run_symbol_backtest(self, symbol: str, strategy_func: Any) -> None:
        """
        LEGACY METHOD - Kept for backward compatibility.
        Use _run_symbol_backtest_integrated() for integrated bot strategy.
        """
        return self._run_symbol_backtest_integrated(symbol, strategy_func)
    
    def _run_symbol_backtest_integrated(self, symbol: str, strategy_func: Any) -> None:
        """
        Run backtest for a single symbol with integrated bot strategy.
        By default uses bar-by-bar replay with 1-minute bars.
        Can optionally use tick-by-tick replay if enabled.
        
        Args:
            symbol: Symbol to backtest
            strategy_func: Function that receives bars and executes strategy.
                          Signature: func(bars_1min: List[Dict], bars_15min: List[Dict]) -> None
        """
        self.logger.info(f"\nBacktesting {symbol}...")
        
        # Load data based on replay mode
        if self.config.use_tick_data:
            # TICK-BY-TICK MODE: Load actual tick data
            ticks = self.data_loader.load_tick_data(symbol)
            if len(ticks) == 0:
                self.logger.warning(f"No tick data available for {symbol}")
                return
            self.logger.info(f"Running TICK-BY-TICK replay with {len(ticks):,} actual ticks")
            
            # Still load bar data for higher timeframe analysis
            bars_15min = self.data_loader.load_bar_data(symbol, "15min")
            
            # Convert ticks to 1-min bars for compatibility with strategy
            bars_1min = self.data_loader._aggregate_ticks_to_bars(ticks, "1min")
            self.logger.info(f"Aggregated {len(ticks):,} ticks into {len(bars_1min)} 1-minute bars")
        else:
            # BAR-BY-BAR MODE: Load pre-aggregated bars (default)
            bars_1min = self.data_loader.load_bar_data(symbol, "1min")
            bars_15min = self.data_loader.load_bar_data(symbol, "15min")
            
            if len(bars_1min) == 0:
                self.logger.warning(f"No bar data available for {symbol}")
                return
            
        # Validate data quality
        is_valid, issues = self.data_loader.validate_data_quality(bars_1min)
        if not is_valid:
            self.logger.warning(f"Data quality issues found:")
            for issue in issues:
                self.logger.warning(f"  - {issue}")
        
        # Execute strategy with historical bars
        # The strategy function will process bars and update bot state/positions
        # The backtest engine monitors bot state to track trades
        self.logger.info(f"Running bar-by-bar replay with {len(bars_1min)} 1-minute bars")
        
        try:
            # Call the integrated strategy function with historical data
            strategy_func(bars_1min, bars_15min)
        except Exception as e:
            self.logger.error(f"Error running strategy: {e}", exc_info=True)
            raise

    
    def _close_position(self, exit_time: datetime, exit_price: float, reason: str) -> None:
        """Close the current position and record the trade"""
        if self.current_position is None:
            return
            
        pos = self.current_position
        
        # Calculate P&L
        if pos['side'] == 'long':
            price_change = exit_price - pos['entry_price']
        else:
            price_change = pos['entry_price'] - exit_price
            
        tick_size = self.bot_config.get('tick_size', 0.25)
        tick_value = self.bot_config.get('tick_value', 1.25)
        ticks = price_change / tick_size
        pnl = ticks * tick_value * pos['quantity']
        
        # Subtract commission
        pnl -= self.config.commission_per_contract * pos['quantity']
        
        # Calculate duration
        duration = (exit_time - pos['entry_time']).total_seconds() / 60.0
        
        # Create trade record
        trade = Trade(
            entry_time=pos['entry_time'],
            exit_time=exit_time,
            symbol=pos['symbol'],
            side=pos['side'],
            quantity=pos['quantity'],
            entry_price=pos['entry_price'],
            exit_price=exit_price,
            stop_price=pos.get('stop_price', 0.0),
            target_price=pos.get('target_price', 0.0),
            exit_reason=reason,
            pnl=pnl,
            ticks=ticks,
            duration_minutes=duration
        )
        
        # Record trade
        self.metrics.add_trade(trade)
        self.current_equity += pnl
        
        # Clear position
        self.current_position = None
        
        self.logger.debug(f"Trade closed: {reason}, P&L: ${pnl:+.2f}")
    
    def set_bot_instance(self, bot) -> None:
        """Set bot instance for RL tracking"""
        self.bot_instance = bot
        # Capture initial experience counts
        if hasattr(bot, 'signal_rl') and hasattr(bot.signal_rl, 'experiences'):
            self.initial_signal_count = len(bot.signal_rl.experiences)
            
    def _log_rl_brain_growth(self) -> None:
        """Log RL brain growth during backtest"""
        if self.bot_instance is None:
            self.logger.info("RL tracking: bot_instance is None")
            return
            
        # Get final counts
        final_signal_count = 0
        
        if hasattr(self.bot_instance, 'signal_rl') and hasattr(self.bot_instance.signal_rl, 'experiences'):
            final_signal_count = len(self.bot_instance.signal_rl.experiences)
            
        # Calculate growth
        signal_growth = final_signal_count - self.initial_signal_count
        
        # Log the results
        self.logger.info("")
        self.logger.info("="*60)
        self.logger.info("RL BRAIN GROWTH")
        self.logger.info("="*60)
        self.logger.info(f"Signal Experiences: {self.initial_signal_count} → {final_signal_count} (+{signal_growth})")
        self.logger.info("="*60)
    
    def _log_results(self, results: Dict[str, Any]) -> None:
        """Log backtest results"""
        self.logger.info(f"Total Trades: {results['total_trades']}")
        self.logger.info(f"Total P&L: ${results['total_pnl']:+,.2f}")
        self.logger.info(f"Total Return: {results['total_return']:+.2f}%")
        self.logger.info(f"Win Rate: {results['win_rate']:.2f}%")
        self.logger.info(f"Average Win: ${results['average_win']:+.2f}")
        self.logger.info(f"Average Loss: ${results['average_loss']:+.2f}")
        self.logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
        self.logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        self.logger.info(f"Max Drawdown: ${results['max_drawdown_dollars']:,.2f} ({results['max_drawdown_percent']:.2f}%)")
        self.logger.info(f"Time in Market: {results['time_in_market_percent']:.2f}%")
        self.logger.info(f"Final Equity: ${results['final_equity']:,.2f}")


class ReportGenerator:
    """Generates backtest reports and visualizations"""
    
    def __init__(self, metrics: PerformanceMetrics):
        self.metrics = metrics
        self.logger = logging.getLogger(__name__)
        
    def generate_trade_breakdown(self) -> str:
        """Generate trade-by-trade breakdown"""
        lines = []
        lines.append("="*80)
        lines.append("TRADE-BY-TRADE BREAKDOWN")
        lines.append("="*80)
        lines.append(f"{'#':<4} {'Entry Time':<20} {'Exit Time':<20} {'Side':<6} {'P&L':<10} {'Reason':<20}")
        lines.append("-"*80)
        
        for i, trade in enumerate(self.metrics.trades, 1):
            lines.append(
                f"{i:<4} "
                f"{trade.entry_time.strftime('%Y-%m-%d %H:%M'):<20} "
                f"{trade.exit_time.strftime('%Y-%m-%d %H:%M'):<20} "
                f"{trade.side:<6} "
                f"${trade.pnl:+8.2f} "
                f"{trade.exit_reason:<20}"
            )
            
        return "\n".join(lines)
    
    def generate_daily_pnl(self) -> Dict[str, float]:
        """Generate daily P&L summary"""
        daily_pnl = {}
        
        for trade in self.metrics.trades:
            date = trade.exit_time.date()
            if date not in daily_pnl:
                daily_pnl[date] = 0.0
            daily_pnl[date] += trade.pnl
            
        return daily_pnl
    
    def save_report(self, filepath: str) -> None:
        """Save comprehensive report to file"""
        try:
            with open(filepath, 'w') as f:
                # Write summary
                summary = self.metrics.get_summary()
                f.write("BACKTEST PERFORMANCE SUMMARY\n")
                f.write("="*60 + "\n\n")
                
                for key, value in summary.items():
                    f.write(f"{key}: {value}\n")
                    
                f.write("\n\n")
                
                # Write trade breakdown
                f.write(self.generate_trade_breakdown())
                f.write("\n\n")
                
                # Write daily P&L
                daily_pnl = self.generate_daily_pnl()
                f.write("DAILY P&L\n")
                f.write("="*60 + "\n")
                for date, pnl in sorted(daily_pnl.items()):
                    f.write(f"{date}: ${pnl:+.2f}\n")
                    
            self.logger.info(f"Report saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving report: {e}")
