"""
Broker WebSocket Streamer using SignalR
Generic WebSocket implementation for broker market data streaming
Supports any broker using SignalR protocol
"""

import logging
import time
from typing import Optional, Callable, Dict
from signalrcore.hub_connection_builder import HubConnectionBuilder

logger = logging.getLogger(__name__)


class BrokerWebSocketStreamer:
    """Real-time WebSocket streamer for broker market data via SignalR"""
    
    def __init__(self, session_token: str, hub_url: str = None, max_reconnect_attempts: int = 5):
        """
        Initialize WebSocket streamer
        
        Args:
            session_token: Broker session token for authentication
            hub_url: WebSocket hub URL (broker-specific endpoint)
            max_reconnect_attempts: Maximum reconnection attempts (default: 5)
        """
        self.session_token = session_token
        self.hub_url = hub_url or "wss://rtc.topstepx.com/hubs/market"  # Default for backward compatibility
        self.connection = None
        self.is_connected = False
        
        # Callbacks
        self.on_quote_callback: Optional[Callable] = None
        self.on_trade_callback: Optional[Callable] = None
        self.on_depth_callback: Optional[Callable] = None
        
        # Stats
        self.quotes_received = 0
        self.trades_received = 0
        self.depth_updates_received = 0
        self.last_message_time = None
        
        # Reconnection tracking
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_attempt = 0
        self.subscriptions = []  # Track active subscriptions for resubscription
    
    def connect(self) -> bool:
        """Connect to broker SignalR market hub"""
        try:
            pass  # Silent - Connecting to WebSocket
            
            auth_url = f"{self.hub_url}?access_token={self.session_token}"
            
            self.connection = (
                HubConnectionBuilder()
                .with_url(auth_url)
                .configure_logging(logging.INFO)
                .with_automatic_reconnect({"type": "interval", "intervals": [0, 2, 5, 10, 30]})
                .build()
            )
            
            self._register_handlers()
            self.connection.start()
            time.sleep(1)
            
            self.is_connected = True
            pass  # Silent - WebSocket connected
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to connect to WebSocket: {e}", exc_info=True)
            self.is_connected = False
            return False
    
    def _register_handlers(self):
        """Register SignalR event handlers"""
        self.connection.on_open(self._on_open)
        self.connection.on_close(self._on_close)
        self.connection.on_error(self._on_error)
        self.connection.on("GatewayQuote", self._on_quote)
        self.connection.on("GatewayTrade", self._on_trade)
        self.connection.on("GatewayDepth", self._on_depth)
    
    def _on_open(self):
        """Called when WebSocket connection opens"""
        pass  # Silent - WebSocket opened
        self.is_connected = True
        self.reconnect_attempt = 0  # Reset reconnect counter on successful connection
        
        # Resubscribe to previous subscriptions after reconnection
        if self.subscriptions:
            pass  # Silent - Resubscribing
            for sub_type, symbol in self.subscriptions:
                try:
                    if sub_type == "quotes":
                        self.connection.send("SubscribeContractQuotes", [symbol])
                    elif sub_type == "trades":
                        self.connection.send("SubscribeContractTrades", [symbol])
                    elif sub_type == "depth":
                        self.connection.send("Subscribe", [symbol, "Depth"])
                    pass  # Silent - Resubscribed
                except Exception as e:
                    logger.error(f"Failed to resubscribe to {sub_type} for {symbol}: {e}")
    
    def _on_close(self):
        """Called when WebSocket connection closes"""
        was_connected = self.is_connected
        self.is_connected = False
        
        # If we're intentionally disconnecting (e.g., maintenance), don't try to reconnect
        if self.reconnect_attempt >= self.max_reconnect_attempts:
            # Intentional disconnect or max attempts reached
            return
        
        # Unexpected disconnect - attempt reconnect
        if was_connected and self.reconnect_attempt < self.max_reconnect_attempts:
            self.reconnect_attempt += 1
            wait_time = min(2 ** self.reconnect_attempt, 30)  # Exponential backoff (2s, 4s, 8s...)
            logger.info(f"[WebSocket] Connection closed unexpectedly - reconnecting in {wait_time}s...")
            time.sleep(wait_time)
            
            try:
                self.connect()
                logger.info("[WebSocket] Reconnected successfully")
            except Exception as e:
                logger.error(f"Manual reconnection attempt {self.reconnect_attempt} failed: {e}")
                if self.reconnect_attempt >= self.max_reconnect_attempts:
                    logger.error(f"[WARN] All {self.max_reconnect_attempts} reconnection attempts failed")
                    logger.error("WebSocket will remain disconnected. Bot will continue with REST API polling.")
    
    def _on_error(self, error):
        """Called when WebSocket error occurs"""
        # If we're intentionally disconnected, don't log errors
        if not self.is_connected and self.reconnect_attempt >= self.max_reconnect_attempts:
            return  # Ignore errors during intentional disconnect
        
        # Extract actual error message from CompletionMessage if present
        error_msg = error
        if hasattr(error, 'error'):
            error_msg = error.error
        elif hasattr(error, 'message'):
            error_msg = error.message
        elif hasattr(error, '__dict__'):
            error_msg = str(error.__dict__)
        
        # Check if this is a connection closed error (expected during maintenance)
        error_str = str(error_msg)
        if any(x in error_str for x in ['Connection closed', 'recv_strict', 'recv_header', 'recv_frame', 'WebSocket connection is closed']):
            # This is expected during broker maintenance - log at info level, not error
            logger.info("[WebSocket] Connection closed by server (expected during maintenance)")
        else:
            logger.error(f"[ERROR] WebSocket error: {error_msg}")
    
    def _on_quote(self, data):
        """Handle incoming quote data"""
        self.quotes_received += 1
        self.last_message_time = time.time()
        if self.on_quote_callback:
            try:
                pass  # Silent - First quote structure logged
                self.on_quote_callback(data)
            except Exception as e:
                logger.error(f"Error in quote callback: {e}")
    
    def _on_trade(self, data):
        """Handle incoming trade data"""
        self.trades_received += 1
        self.last_message_time = time.time()
        if self.on_trade_callback:
            try:
                pass  # Silent - First trade structure logged
                self.on_trade_callback(data)
            except Exception as e:
                logger.error(f"Error in trade callback: {e}")
    
    def _on_depth(self, data):
        """Handle incoming market depth data"""
        self.depth_updates_received += 1
        self.last_message_time = time.time()
        if self.on_depth_callback:
            try:
                self.on_depth_callback(data)
            except Exception as e:
                logger.error(f"Error in depth callback: {e}")
    
    def subscribe_quotes(self, symbol: str, callback: Callable):
        """Subscribe to real-time quotes using contract ID"""
        self.on_quote_callback = callback
        
        try:
            # Some brokers use contract IDs, others use symbols
            # The calling code should pass the appropriate identifier
            self.connection.send("SubscribeContractQuotes", [symbol])
            pass  # Silent - Subscribed to quotes
            
            # Track subscription for reconnection
            sub = ("quotes", symbol)
            if sub not in self.subscriptions:
                self.subscriptions.append(sub)
        except Exception as e:
            logger.error(f"Failed to subscribe to quotes: {e}", exc_info=True)
    
    def subscribe_trades(self, symbol: str, callback: Callable):
        """Subscribe to real-time trades using contract ID"""
        self.on_trade_callback = callback
        
        try:
            # Some brokers use contract IDs, others use symbols
            # The calling code should pass the appropriate identifier
            self.connection.send("SubscribeContractTrades", [symbol])
            pass  # Silent - Subscribed to trades
            
            # Track subscription for reconnection
            sub = ("trades", symbol)
            if sub not in self.subscriptions:
                self.subscriptions.append(sub)
        except Exception as e:
            logger.error(f"Failed to subscribe to trades: {e}", exc_info=True)
    
    def subscribe_depth(self, symbol: str, callback: Callable):
        """Subscribe to Level 2 market depth"""
        self.on_depth_callback = callback
        try:
            # Try common SignalR method variations
            self.connection.send("Subscribe", [symbol, "Depth"])
            pass  # Silent - Subscribed to market depth
            
            # Track subscription for reconnection
            sub = ("depth", symbol)
            if sub not in self.subscriptions:
                self.subscriptions.append(sub)
        except Exception as e:
            logger.error(f"Failed to subscribe to depth: {e}", exc_info=True)
    
    def disconnect(self):
        """Disconnect from WebSocket gracefully"""
        try:
            self.is_connected = False  # Mark as disconnected first to prevent error logs
            self.reconnect_attempt = self.max_reconnect_attempts  # Prevent auto-reconnect
            
            if self.connection:
                try:
                    self.connection.stop()
                except Exception:
                    pass  # Ignore errors during disconnect - connection may already be closed
                self.connection = None
            
            logger.info("[WebSocket] Disconnected gracefully")
        except Exception as e:
            # Ignore all errors during disconnect - we're intentionally closing
            pass
    
    def get_stats(self) -> Dict:
        """Get streaming statistics"""
        return {
            'connected': self.is_connected,
            'quotes_received': self.quotes_received,
            'trades_received': self.trades_received,
            'depth_updates_received': self.depth_updates_received,
            'last_message_time': self.last_message_time
        }


# Backward compatibility alias
TopStepWebSocketStreamer = BrokerWebSocketStreamer
