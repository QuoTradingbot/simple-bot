"""
Integration Example: Event Loop and Error Recovery
Demonstrates how to integrate the event loop and error recovery mechanisms
with the VWAP Bounce Bot.
"""

import logging
import time
import pytz
from datetime import datetime
from event_loop import EventLoop, EventType, EventPriority, TimerManager
from error_recovery import (
    ErrorRecoveryManager, ConnectionMonitor, DataFeedMonitor,
    ErrorType, RecoveryAction
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_integration():
    """Demonstrate event loop and error recovery integration."""
    
    # Bot configuration
    config = {
        "vwap_reset_time": None,  # Will be set dynamically
        "warning_time": None,
        "shutdown_time": None,
        "tick_timeout_seconds": 60,
    }
    
    # Bot status
    bot_status = {
        "trading_enabled": True,
        "last_tick_time": None,
        "emergency_stop": False,
        "flatten_mode": False,
    }
    
    # Initialize timezone
    timezone = pytz.timezone("America/New_York")
    
    # Initialize error recovery
    recovery_manager = ErrorRecoveryManager(config)
    logger.info("Error recovery manager initialized")
    
    # Initialize event loop
    event_loop = EventLoop(bot_status, config)
    logger.info("Event loop initialized")
    
    # Initialize timer manager
    timer_manager = TimerManager(event_loop, config, timezone)
    
    # Register event handlers
    def handle_tick_data(event):
        """Handle market data tick events."""
        data = event.data
        logger.info(f"Processing tick: price={data.get('price')}, volume={data.get('volume')}")
        bot_status["last_tick_time"] = time.time()
    
    def handle_order_fill(event):
        """Handle order fill events."""
        data = event.data
        logger.info(f"Order filled: {data.get('order_id')}, qty={data.get('quantity')}")
    
    def handle_time_check(event):
        """Handle periodic time check events."""
        current_time = event.data.get("time")
        logger.debug(f"Time check: {current_time}")
    
    def handle_vwap_reset(event):
        """Handle VWAP reset events."""
        logger.info("VWAP reset triggered")
    
    def handle_flatten_mode(event):
        """Handle flatten mode activation."""
        logger.warning("Flatten mode activated")
        bot_status["flatten_mode"] = True
    
    def handle_shutdown(event):
        """Handle shutdown events."""
        reason = event.data.get("reason", "unknown")
        logger.warning(f"Shutdown triggered: {reason}")
        event_loop.request_shutdown()
    
    # Register handlers
    event_loop.register_handler(EventType.TICK_DATA, handle_tick_data)
    event_loop.register_handler(EventType.ORDER_FILL, handle_order_fill)
    event_loop.register_handler(EventType.TIME_CHECK, handle_time_check)
    event_loop.register_handler(EventType.VWAP_RESET, handle_vwap_reset)
    event_loop.register_handler(EventType.FLATTEN_MODE, handle_flatten_mode)
    event_loop.register_handler(EventType.SHUTDOWN, handle_shutdown)
    
    # Register shutdown handler
    def cleanup_on_shutdown():
        """Execute cleanup on shutdown."""
        logger.info("Running cleanup tasks...")
        
        # Save state
        state = {
            "timestamp": time.time(),
            "position": 0,
            "equity": 10000.0,
            "bot_status": bot_status,
        }
        recovery_manager.save_state(state)
        
        # Stop timer manager
        timer_manager.stop()
        
        logger.info("Cleanup complete")
    
    event_loop.register_shutdown_handler(cleanup_on_shutdown)
    
    # Simulate some events
    logger.info("\n" + "=" * 60)
    logger.info("SIMULATING TRADING SESSION")
    logger.info("=" * 60)
    
    # Post some tick data events
    for i in range(5):
        event_loop.post_event(
            EventType.TICK_DATA,
            EventPriority.MEDIUM,
            {"price": 5000.0 + i * 0.25, "volume": 10 + i}
        )
    
    # Simulate an order fill
    event_loop.post_event(
        EventType.ORDER_FILL,
        EventPriority.HIGH,
        {"order_id": "ORD123", "quantity": 1, "price": 5001.0}
    )
    
    # Start timer manager in background
    timer_manager.start()
    
    # Process events for a short time
    import threading
    
    def run_event_loop():
        """Run event loop in thread."""
        event_loop.run()
    
    loop_thread = threading.Thread(target=run_event_loop, daemon=True)
    loop_thread.start()
    
    # Let it run for a bit
    time.sleep(2)
    
    # Demonstrate error recovery
    logger.info("\n" + "=" * 60)
    logger.info("DEMONSTRATING ERROR RECOVERY")
    logger.info("=" * 60)
    
    # Test circuit breaker
    breaker = recovery_manager.get_circuit_breaker("order_placement")
    if breaker:
        def place_order():
            logger.info("Placing order...")
            return {"order_id": "ORD456", "status": "filled"}
        
        success, result = breaker.call(place_order)
        logger.info(f"Order placement: success={success}, result={result}")
    
    # Test error handling
    action = recovery_manager.handle_error(
        ErrorType.NETWORK_DISCONNECTION,
        {"timestamp": time.time()}
    )
    logger.info(f"Recovery action for network disconnection: {action.value}")
    
    # Test data feed monitor
    data_monitor = DataFeedMonitor(stale_data_threshold=5)
    data_monitor.update_tick()
    is_current = data_monitor.check_data_feed()
    logger.info(f"Data feed status: {'current' if is_current else 'stale'}")
    
    # Request shutdown
    logger.info("\n" + "=" * 60)
    logger.info("REQUESTING SHUTDOWN")
    logger.info("=" * 60)
    
    event_loop.request_shutdown()
    
    # Wait for shutdown
    loop_thread.join(timeout=5)
    
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    example_integration()
