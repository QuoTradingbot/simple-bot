"""
Broker Interface Abstraction Layer
Provides clean separation between trading strategy and broker execution.
Supports multiple brokers through a common interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import logging
import time
import asyncio

# Import broker SDKs (optional dependencies)
# NOTE: Moved imports inside methods to avoid initialization errors at module import time
BROKER_SDK_AVAILABLE = False
try:
    import project_x_py
    # Only test if the module exists, don't import classes yet (they may have initialization bugs)
    BROKER_SDK_AVAILABLE = True
except ImportError:
    logging.warning("Broker SDK (project-x-py) not installed - some broker operations may not work")

# Import WebSocket streamer
try:
    from broker_websocket import BrokerWebSocketStreamer
    BROKER_WEBSOCKET_AVAILABLE = True
except ImportError:
    BROKER_WEBSOCKET_AVAILABLE = False
    logging.warning("Broker WebSocket module not found - live streaming will not work")


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


class BrokerSDKImplementation(BrokerInterface):
    """
    Broker SDK implementation using Project-X SDK.
    Wraps broker API calls with error handling and retry logic.
    Compatible with brokers that support the Project-X SDK protocol.
    """
    
    def __init__(self, api_token: str, username: str = None, max_retries: int = 3, timeout: int = 30, instrument: str = None):
        """
        Initialize broker connection.
        
        Args:
            api_token: Broker API token for authentication
            username: Broker username/email (required for SDK v3.5+)
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            instrument: Trading instrument symbol (must be configured by user)
        """
        self.api_token = api_token
        self.username = username
        self.max_retries = max_retries
        self.timeout = timeout
        self.instrument = instrument  # Store for TradingSuiteConfig
        self.connected = False
        self.circuit_breaker_open = False
        self.failure_count = 0
        self.circuit_breaker_threshold = 5
        
        # TopStep SDK client (Project-X)
        self.sdk_client: Optional[ProjectX] = None
        self.trading_suite: Optional[TradingSuite] = None
        
        # WebSocket streamer for live data
        self.websocket_streamer: Optional[BrokerWebSocketStreamer] = None
        self._contract_id_cache: Dict[str, str] = {}  # symbol -> contract_id mapping (populated during connection)
        
        # Dynamic balance tracking for auto-reconfiguration
        self._last_configured_balance: float = 0.0
        self._balance_change_threshold: float = 0.05  # Reconfigure if balance changes by 5%
        self.config: Optional[Any] = None  # Store reference to config for dynamic updates
        
        if not BROKER_SDK_AVAILABLE:
            logger.error("TopStep SDK (project-x-py) not installed!")
            logger.error("Install with: pip install project-x-py")
            raise RuntimeError("TopStep SDK not available")
    
    def connect(self, max_retries: int = None) -> bool:
        """
        Connect to TopStep SDK with retry logic.
        
        Args:
            max_retries: Override default max retries (default: 3)
        
        Returns:
            True if connected, False if all retries failed
        """
        import asyncio
        
        if self.circuit_breaker_open:
            logger.error("Circuit breaker is open - cannot connect")
            return False
        
        # Use the async version wrapped in asyncio.run
        retries = max_retries if max_retries is not None else self.max_retries
        return asyncio.run(self.connect_async(retries))
    
    async def connect_async(self, max_retries: int = 3) -> bool:
        """
        Connect to TopStep SDK asynchronously with exponential backoff retry.
        
        Args:
            max_retries: Maximum retry attempts
        
        Returns:
            True if connected, False if all retries failed
        """
        # Import SDK classes here to avoid module-level initialization errors
        from project_x_py import ProjectX, ProjectXConfig, TradingSuite, TradingSuiteConfig
        from project_x_py.realtime.core import ProjectXRealtimeClient
        
        if self.circuit_breaker_open:
            logger.error("Circuit breaker is open - cannot connect")
            return False
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    # Exponential backoff: 2^attempt seconds (2s, 4s, 8s)
                    wait_time = min(2 ** attempt, 30)  # Max 30 seconds
                    pass  # Silent retry
                    await asyncio.sleep(wait_time)
                
                pass  # Silent connection attempt
                
                # Initialize SDK client with username and API key
                self.sdk_client = ProjectX(
                    username=self.username or "",
                    api_key=self.api_token,
                    config=ProjectXConfig()
                )
                
                # Authenticate first (async method)
                pass  # Silent authentication
                try:
                    await self.sdk_client.authenticate()
                    pass  # Silent - authentication successful
                    # Give SDK a moment to establish session
                    await asyncio.sleep(0.5)
                except Exception as auth_error:
                    logger.error(f"Authentication error: {auth_error}")
                    if attempt == max_retries - 1:
                        # Last attempt failed
                        logger.error(f"[FAILED] Authentication failed after {max_retries} attempts")
                        self._record_failure()
                        return False
                    else:
                        # Will retry
                        logger.warning("Authentication failed, will retry...")
                        continue
                
                # Test connection by getting account info first
                try:
                    account = self.sdk_client.get_account_info()
                    pass  # Silent - account info retrieved
                except Exception as account_error:
                    logger.error(f"Failed to get account info: {account_error}")
                    if attempt == max_retries - 1:
                        logger.error(f"[FAILED] Account query failed after {max_retries} attempts")
                        self._record_failure()
                        return False
                    else:
                        logger.warning("Account query failed, will retry...")
                        continue
                
                # Initialize WebSocket streamer first (needed for TradingSuite)
                try:
                    session_token = self.sdk_client.get_session_token()
                    if session_token:
                        pass  # Silent - websocket initialization
                        self.websocket_streamer = BrokerWebSocketStreamer(session_token)
                        if self.websocket_streamer.connect():
                            pass  # Silent - websocket connected
                        else:
                            logger.warning("WebSocket connection failed - will use REST API polling")
                    else:
                        logger.warning("No session token available - WebSocket disabled")
                except Exception as ws_error:
                    logger.warning(f"WebSocket initialization failed: {ws_error} - will use REST API")
                    self.websocket_streamer = None
                
                # Initialize trading suite for order placement (requires realtime_client)
                try:
                    # TradingSuite needs the SDK's realtime client for live order updates
                    # Get JWT token and account info to initialize realtime client
                    jwt_token = self.sdk_client.get_session_token()
                    account_info = self.sdk_client.get_account_info()
                    account_id = str(getattr(account_info, 'id', getattr(account_info, 'account_id', '')))
                    
                    if jwt_token and account_id:
                        # Initialize ProjectX realtime client
                        realtime_client = ProjectXRealtimeClient(
                            jwt_token=jwt_token,
                            account_id=account_id
                        )
                        
                        # Now initialize TradingSuite with the realtime client
                        self.trading_suite = TradingSuite(
                            client=self.sdk_client,
                            realtime_client=realtime_client,
                            config=TradingSuiteConfig(instrument=self.instrument)
                        )
                        pass  # Silent - trading suite initialized
                    else:
                        logger.warning("Missing JWT token or account ID - order placement disabled")
                        self.trading_suite = None
                except Exception as ts_error:
                    logger.warning(f"Trading suite initialization failed: {ts_error}")
                    self.trading_suite = None
                
                # Connection successful! Setup account info
                if account:
                    account_id = getattr(account, 'account_id', getattr(account, 'id', 'N/A'))
                    account_balance = float(getattr(account, 'balance', getattr(account, 'equity', 0)))
                    
                    logger.info(f"✅ Broker Connected - Account: {account_id} | Balance: ${account_balance:,.2f}")
                    
                    # AUTO-CONFIGURE: Set risk limits based on account size
                    # This makes the bot work on ANY TopStep account automatically!
                    from config import BotConfiguration
                    config = BotConfiguration()
                    config.auto_configure_for_account(account_balance, logger)
                    
                    # Store config and balance for dynamic reconfiguration
                    self.config = config
                    self._last_configured_balance = account_balance
                    
                    # CRITICAL: Cache contract IDs while event loop is still active
                    # This MUST happen here before asyncio.run() completes and closes the loop
                    try:
                        instruments = await self.sdk_client.search_instruments(query=self.instrument)
                        if instruments and len(instruments) > 0:
                            # Use first match for caching (attribute is 'id' not 'contract_id')
                            first_contract = getattr(instruments[0], 'id', None)
                            if first_contract:
                                self._contract_id_cache[self.instrument] = first_contract
                                pass  # Silent - contract ID cached
                            else:
                                logger.warning(f"No contract ID found for {self.instrument}")
                        else:
                            logger.warning(f"No instruments found for {self.instrument}")
                    except Exception as cache_err:
                        logger.error(f"Failed to cache contract ID: {cache_err}")
                    
                    # SUCCESS! Connection established
                    self.connected = True
                    self.failure_count = 0
                    pass  # Silent - connection successful
                    return True
                else:
                    logger.error("Account info was None")
                    if attempt == max_retries - 1:
                        logger.error(f"[FAILED] Connection failed after {max_retries} attempts")
                        self._record_failure()
                        return False
                    else:
                        logger.warning("Will retry connection...")
                        continue
                
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    # Last attempt - fail permanently
                    logger.error(f"[FAILED] All {max_retries} connection attempts failed")
                    self._record_failure()
                    return False
                else:
                    # Will retry
                    logger.warning(f"Will retry connection (error: {str(e)[:100]}...)")
                    continue
        
        # Should never reach here, but handle it gracefully
        logger.error("[FAILED] Unexpected exit from connection loop")
        self._record_failure()
        return False
    
    def disconnect(self) -> None:
        """Disconnect from TopStep SDK and WebSocket."""
        try:
            # Disconnect WebSocket streamer first
            if self.websocket_streamer:
                try:
                    self.websocket_streamer.disconnect()
                    self.websocket_streamer = None
                except Exception as e:
                    pass  # Silent - websocket disconnect error
            
            # Close SDK connections
            if self.trading_suite:
                # Close any active connections
                self.trading_suite = None
            if self.sdk_client:
                self.sdk_client = None
            self.connected = False
            pass  # Silent - disconnected from broker
        except Exception as e:
            logger.error(f"Error disconnecting from TopStep SDK: {e}")
    
    def verify_connection(self) -> bool:
        """
        Verify connection is still alive by testing account access.
        This is called periodically by health monitor every 30 seconds.
        
        Returns:
            bool: True if connection is healthy, False if dead
        """
        if not self.connected or not self.sdk_client:
            return False
        
        try:
            # Quick health check - try to get account info
            # This tests the actual API connection, not just local state
            account = self.sdk_client.get_account_info()
            if account is None:
                logger.warning("[CONNECTION] Account info returned None - connection is dead")
                self.connected = False
                return False
            
            # Additional validation - check if we can get balance
            balance = getattr(account, 'balance', None)
            if balance is None:
                logger.warning("[CONNECTION] Account has no balance field - API may have changed")
                # Don't disconnect for this - might be API issue
            
            # Connection is alive and working
            return True
            
        except AttributeError as e:
            # SDK client might be None or invalid
            logger.error(f"[CONNECTION] SDK client invalid: {e}")
            self.connected = False
            return False
        except Exception as e:
            # Any other error means connection is broken
            logger.error(f"[CONNECTION] Health check failed: {e}")
            self.connected = False
            return False
    
    async def _ensure_token_fresh(self) -> bool:
        """
        Ensure JWT token is fresh and refresh if needed.
        The SDK handles this automatically, but we call it explicitly for long-running bots.
        
        Returns:
            bool: True if token is fresh/refreshed, False if refresh failed
        """
        if not self.sdk_client or not self.connected:
            return False
        
        try:
            # SDK's built-in method checks expiry and refreshes if within 5 minutes
            await self.sdk_client._refresh_authentication()
            return True
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return False
    
    def get_account_equity(self) -> float:
        """
        Get account equity from TopStep.
        Automatically reconfigures risk limits if balance changes significantly.
        Ensures 100% TopStep compliance at all times.
        """
        if not self.connected or not self.sdk_client:
            logger.error("Cannot get equity: not connected")
            return 0.0
        
        try:
            account = self.sdk_client.get_account_info()
            if account:
                current_balance = float(account.balance or 0.0)
                
                # CRITICAL: Always reconfigure if config doesn't exist (safety net)
                if not self.config:
                    logger.warning("[WARNING] Config missing - initializing auto-configuration")
                    from config import BotConfiguration
                    self.config = BotConfiguration()
                    if self.config.auto_configure_for_account(current_balance, logger):
                        self._last_configured_balance = current_balance
                    return current_balance
                
                
                # Check if balance changed significantly (5% threshold for reconfiguration)
                if self._last_configured_balance > 0:
                    balance_change_pct = abs(current_balance - self._last_configured_balance) / self._last_configured_balance
                    
                    if balance_change_pct >= self._balance_change_threshold:
                        logger.info("=" * 80)
                        logger.info("💰 BALANCE CHANGED - AUTO-RECONFIGURING RISK LIMITS")
                        logger.info("=" * 80)
                        logger.info(f"Previous Balance: ${self._last_configured_balance:,.2f}")
                        logger.info(f"Current Balance: ${current_balance:,.2f}")
                        logger.info(f"Change: {balance_change_pct * 100:.1f}%")
                        
                        # Reconfigure with new balance (with safety checks)
                        if self.config.auto_configure_for_account(current_balance, logger):
                            self._last_configured_balance = current_balance
                            logger.info("[SUCCESS] Risk limits updated successfully")
                        else:
                            logger.error("❌ Failed to reconfigure - keeping previous limits")
                            logger.error(f"Still using limits for ${self._last_configured_balance:,.2f} balance")
                
                return current_balance
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
            # Use search_open_positions() instead of deprecated get_positions()
            # This is an async function, so we need to run it in the event loop
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                # If loop is already running, we can't use run_until_complete
                # Return 0 and log debug - position reconciliation will happen async
                pass  # Silent - event loop running
                return 0
            except RuntimeError:
                # No running loop, safe to create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    positions = loop.run_until_complete(self.sdk_client.search_open_positions())
                    for pos in positions:
                        if pos.instrument.symbol == symbol:
                            # Return signed quantity (positive for long, negative for short)
                            qty = int(pos.quantity)
                            return qty if pos.position_type.value == "LONG" else -qty
                    return 0  # No position found
                finally:
                    loop.close()
        except AttributeError as e:
            # Common Windows asyncio proactor error during shutdown - ignore
            if "'NoneType' object has no attribute 'send'" in str(e):
                pass  # Silent - asyncio error during shutdown
                return 0
            logger.error(f"Error getting position quantity: {e}")
            self._record_failure()
            return 0
        except Exception as e:
            logger.error(f"Error getting position quantity: {e}")
            self._record_failure()
            return 0
    
    def place_market_order(self, symbol: str, side: str, quantity: int) -> Optional[Dict[str, Any]]:
        """Place market order using TopStep SDK."""
        if not self.connected or self.trading_suite is None:
            logger.error("Cannot place order: not connected")
            return None
        
        try:
            import asyncio
            # Import order enums here to avoid module-level import issues
            from project_x_py import OrderSide, OrderType
            
            # Get contract ID for the symbol dynamically
            contract_id = self._get_contract_id_sync(symbol)
            if not contract_id:
                logger.error(f"Failed to resolve contract ID for {symbol}")
                return None
            
            # Convert side to OrderSide enum
            order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            
            pass  # Silent - order placement is logged at higher level
            
            # Define async wrapper
            async def place_order_async():
                # Refresh token if needed (for long-running bots)
                await self._ensure_token_fresh()
                
                return await self.trading_suite.orders.place_market_order(
                    contract_id=contract_id,
                    side=order_side,
                    size=quantity
                )
            
            # Run async order placement - check for existing event loop
            try:
                loop = asyncio.get_running_loop()
                # If we get here, we're in an async context - use thread pool
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    order_response = pool.submit(
                        lambda: asyncio.run(place_order_async())
                    ).result()
            except RuntimeError:
                # No running loop - safe to use asyncio.run
                order_response = asyncio.run(place_order_async())
            
            logger.info(f"Order response: {order_response}")
            
            if order_response and order_response.success:
                pass  # Silent - order success logged at higher level
                return {
                    "order_id": order_response.orderId,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "type": "MARKET",
                    "status": "SUBMITTED",
                    "filled_quantity": 0
                }
            else:
                error_msg = order_response.errorMessage if order_response else "Unknown error"
                logger.error(f"Market order placement failed: {error_msg}")
                self._record_failure()
                return None
                
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            import traceback
            traceback.print_exc()
            self._record_failure()
            return None
    
    def place_limit_order(self, symbol: str, side: str, quantity: int, limit_price: float) -> Optional[Dict[str, Any]]:
        """Place limit order using TopStep SDK."""
        if not self.connected or self.trading_suite is None:
            logger.error("Cannot place order: not connected")
            return None
        
        try:
            import asyncio
            # Import order enums here to avoid module-level import issues
            from project_x_py import OrderSide, OrderType
            
            # Get contract ID for the symbol dynamically
            contract_id = self._get_contract_id_sync(symbol)
            if not contract_id:
                logger.error(f"Failed to resolve contract ID for {symbol}")
                return None
            
            # Convert side to OrderSide enum
            order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            
            pass  # Silent - limit order placement logged at higher level
            
            # Define async wrapper
            async def place_order_async():
                # Refresh token if needed (for long-running bots)
                await self._ensure_token_fresh()
                
                return await self.trading_suite.orders.place_limit_order(
                    contract_id=contract_id,
                    side=order_side,
                    size=quantity,
                    limit_price=limit_price
                )
            
            # Run async order placement - check for existing event loop
            try:
                loop = asyncio.get_running_loop()
                # If we get here, we're in an async context - use thread pool
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    order_response = pool.submit(
                        lambda: asyncio.run(place_order_async())
                    ).result()
            except RuntimeError:
                # No running loop - safe to use asyncio.run
                order_response = asyncio.run(place_order_async())
            
            logger.info(f"Order response: {order_response}")
            
            if order_response and order_response.success:
                pass  # Silent - order success logged at higher level
                return {
                    "order_id": order_response.orderId,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "type": "LIMIT",
                    "limit_price": limit_price,
                    "status": "SUBMITTED",
                    "filled_quantity": 0
                }
            else:
                error_msg = order_response.errorMessage if order_response else "Unknown error"
                logger.error(f"Limit order placement failed: {error_msg}")
                self._record_failure()
                return None
                
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            import traceback
            traceback.print_exc()
            self._record_failure()
            return None
            self._record_failure()
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order using TopStep SDK."""
        if not self.connected or self.trading_suite is None:
            logger.error("Cannot cancel order: not connected")
            return False
        
        try:
            import asyncio
            
            # Define async wrapper
            async def cancel_order_async():
                return await self.trading_suite.orders.cancel_order(order_id=order_id)
            
            # Run async order cancellation - check for existing event loop
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    cancel_response = pool.submit(
                        lambda: asyncio.run(cancel_order_async())
                    ).result()
            except RuntimeError:
                cancel_response = asyncio.run(cancel_order_async())
            
            if cancel_response and cancel_response.success:
                pass  # Silent - order cancelled logged at higher level
                return True
            else:
                error_msg = cancel_response.errorMessage if cancel_response else "Unknown error"
                logger.error(f"Order cancellation failed: {error_msg}")
                self._record_failure()
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            self._record_failure()
            return False
    
    def place_stop_order(self, symbol: str, side: str, quantity: int, stop_price: float) -> Optional[Dict[str, Any]]:
        """Place stop order using TopStep SDK."""
        if not self.connected or self.trading_suite is None:
            logger.error("Cannot place order: not connected")
            return None
        
        try:
            import asyncio
            # Import order enums here to avoid module-level import issues
            from project_x_py import OrderSide, OrderType
            
            # Convert side to SDK enum
            order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            
            # Define async wrapper
            async def place_order_async():
                return await self.trading_suite.orders.place_stop_order(
                    symbol=symbol,
                    side=order_side,
                    quantity=quantity,
                    stop_price=stop_price
                )
            
            # Run async order placement - check for existing event loop
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    order_response = pool.submit(
                        lambda: asyncio.run(place_order_async())
                    ).result()
            except RuntimeError:
                order_response = asyncio.run(place_order_async())
            
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
        """Subscribe to real-time market data (trades) via WebSocket."""
        if not self.connected:
            logger.error("Cannot subscribe: not connected")
            return
        
        if not self.websocket_streamer:
            logger.error("WebSocket streamer not initialized - cannot subscribe to live data")
            logger.error("Make sure WebSocket module is available and session token is valid")
            return
        
        try:
            # Get contract ID for the symbol (synchronous)
            contract_id = self._get_contract_id_sync(symbol)
            if not contract_id:
                logger.error(f"Failed to get contract ID for {symbol}")
                return
            
            # Define callback wrapper to convert WebSocket data format
            def trade_callback(data):
                """Handle trade data from WebSocket: [contract_id, [{trade1}, {trade2}, ...]]"""
                if isinstance(data, list) and len(data) >= 2:
                    trades = data[1]  # List of trade dicts
                    if isinstance(trades, list):
                        for trade in trades:
                            price = float(trade.get('price', 0))
                            volume = int(trade.get('volume', 0))
                            
                            # Parse timestamp (ISO format string to milliseconds)
                            timestamp_str = trade.get('timestamp', '')
                            try:
                                from datetime import datetime
                                if timestamp_str:
                                    dt = datetime.fromisoformat(str(timestamp_str).replace('Z', '+00:00'))
                                    timestamp = int(dt.timestamp() * 1000)
                                else:
                                    timestamp = int(datetime.now().timestamp() * 1000)
                            except (ValueError, TypeError, AttributeError) as e:
                                logger.debug(f"Timestamp parsing error: {e}, using current time")
                                timestamp = int(datetime.now().timestamp() * 1000)
                            
                            # Call bot's callback with tick data
                            callback(symbol, price, volume, timestamp)
            
            # Subscribe to trades via WebSocket
            self.websocket_streamer.subscribe_trades(contract_id, trade_callback)
            pass  # Silent - data subscription is internal
            
        except Exception as e:
            logger.error(f"Error subscribing to market data: {e}")
            self._record_failure()
    
    def subscribe_quotes(self, symbol: str, callback: Callable[[str, float, float, int, int, float, int], None]) -> None:
        """Subscribe to real-time bid/ask quotes via WebSocket."""
        if not self.connected:
            logger.error("Cannot subscribe to quotes: not connected")
            return
        
        if not self.websocket_streamer:
            logger.warning("WebSocket streamer not initialized - quote subscription unavailable")
            return
        
        try:
            # Get contract ID for the symbol (synchronous)
            contract_id = self._get_contract_id_sync(symbol)
            if not contract_id:
                logger.error(f"Failed to get contract ID for {symbol}")
                return
            
            # Sticky state - keep last valid bid/ask
            last_valid_bid = [0.0]  # Use list to maintain closure reference
            last_valid_ask = [0.0]
            last_valid_timestamp = [0]
            
            # Define callback wrapper to convert WebSocket data format
            def quote_callback(data):
                """Handle quote data from WebSocket: [contract_id, {quote_dict}]"""
                if isinstance(data, list) and len(data) >= 2:
                    quote = data[1]  # Quote dict
                    if isinstance(quote, dict):
                        # STICKY STATE PATTERN: Update only if new values are valid
                        # This prevents false signals from partial/incomplete WebSocket updates
                        
                        # Update bid if present and valid
                        if 'bestBid' in quote:
                            new_bid = float(quote.get('bestBid', 0))
                            if new_bid > 0:
                                last_valid_bid[0] = new_bid
                        
                        # Update ask if present and valid
                        if 'bestAsk' in quote:
                            new_ask = float(quote.get('bestAsk', 0))
                            if new_ask > 0:
                                last_valid_ask[0] = new_ask
                        
                        # Only process if we have BOTH valid bid and ask
                        if last_valid_bid[0] <= 0 or last_valid_ask[0] <= 0:
                            pass  # Silent - waiting for valid quote
                            return
                        
                        # Sanity check: ask must be >= bid
                        if last_valid_ask[0] < last_valid_bid[0]:
                            logger.warning(f"Inverted market: bid={last_valid_bid[0]} > ask={last_valid_ask[0]} - skipping")
                            return
                        
                        bid_price = last_valid_bid[0]
                        ask_price = last_valid_ask[0]
                        last_price = float(quote.get('lastPrice', bid_price))  # Default to bid if missing
                        bid_size = 1  # TopStep doesn't provide sizes in quote data
                        ask_size = 1
                        
                        # Parse timestamp (ISO format string to milliseconds)
                        timestamp_str = quote.get('timestamp', '')
                        try:
                            from datetime import datetime
                            import time
                            if timestamp_str:
                                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                timestamp = int(dt.timestamp() * 1000)
                            else:
                                timestamp = int(datetime.now().timestamp() * 1000)
                            
                            # Staleness detection: warn if quote is more than 2 seconds old
                            current_time = int(time.time() * 1000)
                            age_seconds = (current_time - last_valid_timestamp[0]) / 1000.0
                            if last_valid_timestamp[0] > 0 and age_seconds > 2.0:
                                logger.warning(f"Quote feed stale - {age_seconds:.1f}s since last update")
                            last_valid_timestamp[0] = current_time
                            
                        except (ValueError, TypeError, AttributeError) as e:
                            logger.debug(f"Timestamp parsing error: {e}, using current time")
                            timestamp = int(datetime.now().timestamp() * 1000)
                        
                        # Call bot's callback with quote data
                        callback(symbol, bid_price, ask_price, bid_size, ask_size, last_price, timestamp)
            
            # Subscribe to quotes via WebSocket
            self.websocket_streamer.subscribe_quotes(contract_id, quote_callback)
            pass  # Silent - quote subscription is internal
            
        except Exception as e:
            logger.error(f"Error subscribing to quotes: {e}")
            self._record_failure()
    
    def _get_contract_id_sync(self, symbol: str) -> Optional[str]:
        """
        Get TopStep contract ID for a symbol (e.g., ES -> CON.F.US.EP.Z25).
        Uses cache to avoid repeated API calls. Falls back to synchronous lookup if not cached.
        """
        # Check cache first (populated during connection)
        if symbol in self._contract_id_cache:
            pass  # Silent - using cached contract ID
            return self._contract_id_cache[symbol]
        
        # Remove leading slash if present (e.g., /ES -> ES)
        clean_symbol = symbol.lstrip('/')
        
        # Not in cache - need to look it up
        # This shouldn't happen often if connection caching works properly
        logger.warning(f"Contract ID for {symbol} not in cache - performing lookup")
        
        try:
            # Use the SDK's synchronous method if available, otherwise async
            if hasattr(self.sdk_client, 'search_instruments_sync'):
                instruments = self.sdk_client.search_instruments_sync(query=clean_symbol)
            else:
                # Run async method in a new thread with its own event loop
                import asyncio
                from concurrent.futures import ThreadPoolExecutor
                
                def run_async_search():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(
                            self.sdk_client.search_instruments(query=clean_symbol)
                        )
                    finally:
                        loop.close()
                
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_async_search)
                    instruments = future.result(timeout=10)
            
            if instruments and len(instruments) > 0:
                # Find exact match or closest match
                for instr in instruments:
                    if instr.symbol == clean_symbol or instr.symbol.startswith(clean_symbol):
                        contract_id = instr.contract_id
                        self._contract_id_cache[symbol] = contract_id
                        pass  # Silent - contract ID cached
                        return contract_id
                
                # No exact match - use first result
                contract_id = instruments[0].contract_id
                self._contract_id_cache[symbol] = contract_id
                pass  # Silent - using first match
                return contract_id
            
            logger.error(f"No contracts found for symbol: {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting contract ID for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def fetch_historical_bars(self, symbol: str, timeframe: str, count: int, 
                             start_date: datetime = None, end_date: datetime = None) -> List[Dict[str, Any]]:
        """Fetch historical bars from TopStep."""
        import asyncio
        
        if not self.connected or not self.sdk_client:
            logger.error("Cannot fetch bars: not connected")
            return []
        
        try:
            # Convert timeframe string to interval (e.g., '1m' -> 1 minute)
            if 'm' in timeframe or 'min' in timeframe:
                interval = int(timeframe.replace('m', '').replace('min', ''))
                unit = 2  # Minutes
            elif 'h' in timeframe:
                interval = int(timeframe.replace('h', '')) * 60
                unit = 2  # Minutes
            else:
                interval = 5  # Default to 5 minutes
                unit = 2
            
            # Fetch historical data using get_bars (async method)
            bars_df = asyncio.run(self.sdk_client.get_bars(
                symbol=symbol,
                interval=interval,
                unit=unit,
                limit=count,
                start_time=start_date,
                end_time=end_date
            ))
            
            if bars_df is not None and len(bars_df) > 0:
                # Convert Polars DataFrame to list of dicts
                return [
                    {
                        "timestamp": row['timestamp'],
                        "open": float(row['open']),
                        "high": float(row['high']),
                        "low": float(row['low']),
                        "close": float(row['close']),
                        "volume": int(row['volume'])
                    }
                    for row in bars_df.iter_rows(named=True)
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
        pass  # Silent - circuit breaker reset


def create_broker(api_token: str, username: str = None, instrument: str = None) -> BrokerInterface:
    """
    Factory function to create a broker instance.
    
    Args:
        api_token: Broker API token (required)
        username: Broker username/email (required for SDK v3.5+)
        instrument: Trading instrument symbol (must be configured by user)
    
    Returns:
        BrokerInterface implementation
    
    Raises:
        ValueError: If API token is missing
    """
    if not api_token:
        raise ValueError("API token is required for broker connection")
    return BrokerSDKImplementation(api_token=api_token, username=username, instrument=instrument)


# Backward compatibility alias for existing code
TopStepBroker = BrokerSDKImplementation

