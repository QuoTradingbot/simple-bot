"""
Broker Interface Abstraction Layer
Provides clean separation between trading strategy and broker execution with TopStep SDK integration.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import logging
import time

# Import TopStep SDK (Project-X)
try:
    from project_x_py import ProjectX, ProjectXConfig, TradingSuite, TradingSuiteConfig
    from project_x_py import OrderSide, OrderType
    TOPSTEP_SDK_AVAILABLE = True
except ImportError:
    TOPSTEP_SDK_AVAILABLE = False
    logging.warning("TopStep SDK (project-x-py) not installed - broker operations will not work")


logger = logging.getLogger(__name__)


class BrokerInterface(ABC):
    """
    Abstract base class for broker operations.
    Allows swapping brokers without changing strategy code.
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to broker and authenticate.
        
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker."""
        pass
    
    @abstractmethod
    def get_account_equity(self) -> float:
        """
        Get current account equity.
        
        Returns:
            Account equity in dollars
        """
        pass
    
    @abstractmethod
    def get_position_quantity(self, symbol: str) -> int:
        """
        Get current position quantity for symbol.
        
        Args:
            symbol: Instrument symbol
        
        Returns:
            Position quantity (positive for long, negative for short, 0 for flat)
        """
        pass
    
    @abstractmethod
    def place_market_order(self, symbol: str, side: str, quantity: int) -> Optional[Dict[str, Any]]:
        """
        Place a market order.
        
        Args:
            symbol: Instrument symbol
            side: Order side ("BUY" or "SELL")
            quantity: Number of contracts
        
        Returns:
            Order details if successful, None otherwise
        """
        pass
    
    @abstractmethod
    def place_limit_order(self, symbol: str, side: str, quantity: int, 
                         limit_price: float) -> Optional[Dict[str, Any]]:
        """
        Place a limit order.
        
        Args:
            symbol: Instrument symbol
            side: Order side ("BUY" or "SELL")
            quantity: Number of contracts
            limit_price: Limit price
        
        Returns:
            Order details if successful, None otherwise
        """
        pass
    
    @abstractmethod
    def place_stop_order(self, symbol: str, side: str, quantity: int, 
                        stop_price: float) -> Optional[Dict[str, Any]]:
        """
        Place a stop order.
        
        Args:
            symbol: Instrument symbol
            side: Order side ("BUY" or "SELL")
            quantity: Number of contracts
            stop_price: Stop price
        
        Returns:
            Order details if successful, None otherwise
        """
        pass
    
    @abstractmethod
    def subscribe_market_data(self, symbol: str, callback: Callable[[str, float, int, int], None]) -> None:
        """
        Subscribe to real-time market data (trades).
        
        Args:
            symbol: Instrument symbol
            callback: Function to call with tick data (symbol, price, volume, timestamp)
        """
        pass
    
    @abstractmethod
    def subscribe_quotes(self, symbol: str, callback: Callable[[str, float, float, int, int, float, int], None]) -> None:
        """
        Subscribe to real-time bid/ask quotes.
        
        Args:
            symbol: Instrument symbol
            callback: Function to call with quote data (symbol, bid_price, ask_price, bid_size, ask_size, last_price, timestamp)
        """
        pass
    
    @abstractmethod
    def fetch_historical_bars(self, symbol: str, timeframe: int, count: int) -> list:
        """
        Fetch historical bars.
        
        Args:
            symbol: Instrument symbol
            timeframe: Timeframe in minutes
            count: Number of bars to fetch
        
        Returns:
            List of bar dictionaries
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if broker connection is active.
        
        Returns:
            True if connected
        """
        pass


class TopStepBroker(BrokerInterface):
    """
    TopStep SDK broker implementation using Project-X SDK.
    Wraps TopStep API calls with error handling and retry logic.
    """
    
    def __init__(self, api_token: str, max_retries: int = 3, timeout: int = 30):
        """
        Initialize TopStep broker.
        
        Args:
            api_token: TopStep API token
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        self.api_token = api_token
        self.max_retries = max_retries
        self.timeout = timeout
        self.connected = False
        self.circuit_breaker_open = False
        self.failure_count = 0
        self.circuit_breaker_threshold = 5
        
        # TopStep SDK client (Project-X)
        self.sdk_client: Optional[ProjectX] = None
        self.trading_suite: Optional[TradingSuite] = None
        
        if not TOPSTEP_SDK_AVAILABLE:
            logger.error("TopStep SDK (project-x-py) not installed!")
            logger.error("Install with: pip install project-x-py")
            raise RuntimeError("TopStep SDK not available")
    
    def connect(self) -> bool:
        """Connect to TopStep SDK."""
        if self.circuit_breaker_open:
            logger.error("Circuit breaker is open - cannot connect")
            return False
        
        try:
            logger.info("Connecting to TopStep SDK (Project-X)...")
            
            # Initialize SDK client
            config = ProjectXConfig(api_key=self.api_token)
            self.sdk_client = ProjectX(config)
            
            # Initialize trading suite for order management
            suite_config = TradingSuiteConfig(
                api_key=self.api_token,
                environment="production"
            )
            self.trading_suite = TradingSuite(suite_config)
            
            # Test connection by getting account info
            account = self.sdk_client.get_account()
            if account:
                logger.info(f"Connected to TopStep - Account: {account.account_id}")
                self.connected = True
                self.failure_count = 0
                return True
            else:
                logger.error("Failed to retrieve account info")
                self._record_failure()
                return False
            
        except Exception as e:
            logger.error(f"Failed to connect to TopStep SDK: {e}")
            self._record_failure()
            return False
    
    def disconnect(self) -> None:
        """Disconnect from TopStep SDK."""
        try:
            if self.trading_suite:
                # Close any active connections
                self.trading_suite = None
            if self.sdk_client:
                self.sdk_client = None
            self.connected = False
            logger.info("Disconnected from TopStep SDK")
        except Exception as e:
            logger.error(f"Error disconnecting from TopStep SDK: {e}")
    
    def get_account_equity(self) -> float:
        """Get account equity from TopStep."""
        if not self.connected or not self.sdk_client:
            logger.error("Cannot get equity: not connected")
            return 0.0
        
        try:
            account = self.sdk_client.get_account()
            if account:
                equity = float(account.balance or 0.0)
                return equity
            return 0.0
        except Exception as e:
            logger.error(f"Error getting account equity: {e}")
            self._record_failure()
            return 0.0
    
    def get_position_quantity(self, symbol: str) -> int:
        """Get position quantity from TopStep."""
        if not self.connected or not self.sdk_client:
            logger.error("Cannot get position: not connected")
            return 0
        
        try:
            positions = self.sdk_client.get_positions()
            for pos in positions:
                if pos.instrument.symbol == symbol:
                    # Return signed quantity (positive for long, negative for short)
                    qty = int(pos.quantity)
                    return qty if pos.position_type.value == "LONG" else -qty
            return 0  # No position found
        except Exception as e:
            logger.error(f"Error getting position quantity: {e}")
            self._record_failure()
            return 0
    
    def place_market_order(self, symbol: str, side: str, quantity: int) -> Optional[Dict[str, Any]]:
        """Place market order using TopStep SDK."""
        if not self.connected or not self.trading_suite:
            logger.error("Cannot place order: not connected")
            return None
        
        try:
            # Convert side to SDK enum
            order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            
            # Place market order
            order_response = self.trading_suite.place_market_order(
                symbol=symbol,
                side=order_side,
                quantity=quantity
            )
            
            if order_response and order_response.order:
                order = order_response.order
                return {
                    "order_id": order.order_id,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "type": "MARKET",
                    "status": order.status.value,
                    "filled_quantity": order.filled_quantity or 0,
                    "avg_fill_price": order.avg_fill_price or 0.0
                }
            else:
                logger.error("Market order placement failed")
                self._record_failure()
                return None
                
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            self._record_failure()
            return None
    
    def place_limit_order(self, symbol: str, side: str, quantity: int, limit_price: float) -> Optional[Dict[str, Any]]:
        """Place limit order using TopStep SDK."""
        if not self.connected or not self.trading_suite:
            logger.error("Cannot place order: not connected")
            return None
        
        try:
            # Convert side to SDK enum
            order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            
            # Place limit order
            order_response = self.trading_suite.place_limit_order(
                symbol=symbol,
                side=order_side,
                quantity=quantity,
                limit_price=limit_price
            )
            
            if order_response and order_response.order:
                order = order_response.order
                return {
                    "order_id": order.order_id,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "type": "LIMIT",
                    "limit_price": limit_price,
                    "status": order.status.value,
                    "filled_quantity": order.filled_quantity or 0
                }
            else:
                logger.error("Limit order placement failed")
                self._record_failure()
                return None
                
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            self._record_failure()
            return None
    
    def place_stop_order(self, symbol: str, side: str, quantity: int, stop_price: float) -> Optional[Dict[str, Any]]:
        """Place stop order using TopStep SDK."""
        if not self.connected or not self.trading_suite:
            logger.error("Cannot place order: not connected")
            return None
        
        try:
            # Convert side to SDK enum
            order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            
            # Place stop order
            order_response = self.trading_suite.place_stop_order(
                symbol=symbol,
                side=order_side,
                quantity=quantity,
                stop_price=stop_price
            )
            
            if order_response and order_response.order:
                order = order_response.order
                return {
                    "order_id": order.order_id,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "type": "STOP",
                    "stop_price": stop_price,
                    "status": order.status.value
                }
            else:
                logger.error("Stop order placement failed")
                self._record_failure()
                return None
                
        except Exception as e:
            logger.error(f"Error placing stop order: {e}")
            self._record_failure()
            return None
    
    def subscribe_market_data(self, symbol: str, callback: Callable[[str, float, int, int], None]) -> None:
        """Subscribe to real-time market data (trades)."""
        if not self.connected or not self.sdk_client:
            logger.error("Cannot subscribe: not connected")
            return
        
        try:
            # Subscribe to realtime data
            realtime_client = self.sdk_client.get_realtime_client()
            if realtime_client:
                # Subscribe to trades/quotes for the symbol
                realtime_client.subscribe_trades(
                    symbol,
                    lambda trade: callback(
                        trade.instrument.symbol,
                        float(trade.price),
                        int(trade.size),
                        int(trade.timestamp.timestamp() * 1000)
                    )
                )
                logger.info(f"Subscribed to market data for {symbol}")
            else:
                logger.error("Failed to get realtime client")
        except Exception as e:
            logger.error(f"Error subscribing to market data: {e}")
            self._record_failure()
    
    def subscribe_quotes(self, symbol: str, callback: Callable[[str, float, float, int, int, float, int], None]) -> None:
        """Subscribe to real-time bid/ask quotes."""
        if not self.connected or not self.sdk_client:
            logger.error("Cannot subscribe to quotes: not connected")
            return
        
        try:
            # Subscribe to realtime quotes
            realtime_client = self.sdk_client.get_realtime_client()
            if realtime_client:
                # Subscribe to quotes (bid/ask) for the symbol
                realtime_client.subscribe_quotes(
                    symbol,
                    lambda quote: callback(
                        quote.instrument.symbol,
                        float(quote.bid_price),
                        float(quote.ask_price),
                        int(quote.bid_size),
                        int(quote.ask_size),
                        float(quote.last_price) if hasattr(quote, 'last_price') else float(quote.bid_price),
                        int(quote.timestamp.timestamp() * 1000)
                    )
                )
                logger.info(f"Subscribed to bid/ask quotes for {symbol}")
            else:
                logger.error("Failed to get realtime client")
        except Exception as e:
            logger.error(f"Error subscribing to quotes: {e}")
            self._record_failure()
    
    def fetch_historical_bars(self, symbol: str, timeframe: str, count: int) -> List[Dict[str, Any]]:
        """Fetch historical bars from TopStep."""
        if not self.connected or not self.sdk_client:
            logger.error("Cannot fetch bars: not connected")
            return []
        
        try:
            # Fetch historical data
            bars = self.sdk_client.get_historical_bars(
                symbol=symbol,
                interval=timeframe,
                limit=count
            )
            
            if bars:
                return [
                    {
                        "timestamp": bar.timestamp,
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": int(bar.volume)
                    }
                    for bar in bars
                ]
            return []
        except Exception as e:
            logger.error(f"Error fetching historical bars: {e}")
            self._record_failure()
            return []
    def is_connected(self) -> bool:
        """Check if connected to TopStep SDK."""
        return self.connected and not self.circuit_breaker_open
    
    def _record_failure(self) -> None:
        """Record a failure and potentially open circuit breaker."""
        self.failure_count += 1
        if self.failure_count >= self.circuit_breaker_threshold:
            self.circuit_breaker_open = True
            logger.critical(f"Circuit breaker opened after {self.failure_count} failures")
    
    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker (manual recovery)."""
        self.circuit_breaker_open = False
        self.failure_count = 0
        logger.info("Circuit breaker reset")


def create_broker(api_token: str) -> BrokerInterface:
    """
    Factory function to create TopStep broker instance.
    
    Args:
        api_token: API token for TopStep (required)
    
    Returns:
        TopStepBroker instance
    
    Raises:
        ValueError: If API token is missing
    """
    if not api_token:
        raise ValueError("API token is required for TopStep broker")
    return TopStepBroker(api_token=api_token)
