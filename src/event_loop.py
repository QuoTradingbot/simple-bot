"""
Event Loop Implementation for VWAP Bounce Bot
Handles asynchronous operations, market data processing, and timer-based events.
"""

import logging
import signal
import time
import threading
from queue import Queue, PriorityQueue, Empty
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from enum import IntEnum
from dataclasses import dataclass, field
import pytz


logger = logging.getLogger(__name__)


class EventPriority(IntEnum):
    """Event priority levels (lower number = higher priority)."""
    CRITICAL = 1   # Shutdown signals, emergency flatten
    HIGH = 2       # Order fills, stop hits
    MEDIUM = 3     # Tick data, time checks
    LOW = 4        # Logging, statistics updates


class EventType(IntEnum):
    """Event types for the trading bot."""
    # Critical events
    SHUTDOWN = 1
    EMERGENCY_FLATTEN = 2
    
    # High priority events
    ORDER_FILL = 10
    ORDER_PARTIAL_FILL = 11
    ORDER_REJECT = 12
    STOP_HIT = 13
    POSITION_RECONCILIATION = 14  # New: Periodic position sync check
    CONNECTION_HEALTH = 15  # New: Periodic broker connection health check
    LICENSE_CHECK = 16  # New: Periodic license validation check
    
    # Medium priority events
    TICK_DATA = 20
    TIME_CHECK = 21
    VWAP_RESET = 22
    
    # Low priority events
    LOG_UPDATE = 30
    STATS_UPDATE = 31


@dataclass(order=True)
class Event:
    """Event object with priority-based ordering."""
    priority: int
    event_type: EventType = field(compare=False)
    timestamp: float = field(compare=False)
    data: Dict[str, Any] = field(default_factory=dict, compare=False)


class EventLoop:
    """
    Event-driven architecture for the trading bot.
    Handles market data, orders, timers, and graceful shutdown.
    """
    
    def __init__(self, bot_status: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize event loop.
        
        Args:
            bot_status: Global bot status dictionary
            config: Bot configuration dictionary
        """
        self.bot_status = bot_status
        self.config = config
        self.event_queue: PriorityQueue = PriorityQueue(maxsize=10000)
        self.running = False
        self.shutdown_requested = False
        
        # Event handlers registry
        self.handlers: Dict[EventType, Callable] = {}
        
        # Monitoring metrics
        self.metrics = {
            "loop_iterations": 0,
            "events_processed": 0,
            "max_queue_depth": 0,
            "max_processing_time_ms": 0.0,
            "total_processing_time_ms": 0.0,
            "stall_count": 0,
            "last_iteration_time": None,
        }
        
        # Shutdown handlers
        self.shutdown_handlers: list = []
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        pass  # Silent - event loop initialized
    
    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle SIGINT and SIGTERM signals for graceful shutdown."""
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.warning(f"Received {signal_name} - initiating graceful shutdown")
        self.request_shutdown()
    
    def register_handler(self, event_type: EventType, handler: Callable) -> None:
        """
        Register event handler for specific event type.
        
        Args:
            event_type: Type of event to handle
            handler: Callable that processes the event
        """
        self.handlers[event_type] = handler
        pass  # Silent - handler registered
    
    def register_shutdown_handler(self, handler: Callable) -> None:
        """
        Register shutdown handler.
        
        Args:
            handler: Callable to execute during shutdown
        """
        self.shutdown_handlers.append(handler)
        pass  # Silent - shutdown handler registered
    
    def post_event(self, event_type: EventType, priority: EventPriority, 
                   data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Post event to the queue.
        
        Args:
            event_type: Type of event
            priority: Event priority
            data: Event data dictionary
        
        Returns:
            True if event posted successfully
        """
        if data is None:
            data = {}
        
        event = Event(
            priority=priority.value,
            event_type=event_type,
            timestamp=time.time(),
            data=data
        )
        
        try:
            self.event_queue.put_nowait(event)
            
            # Update queue depth metric
            queue_depth = self.event_queue.qsize()
            if queue_depth > self.metrics["max_queue_depth"]:
                self.metrics["max_queue_depth"] = queue_depth
            
            return True
        except Exception as e:
            logger.error(f"Failed to post event {event_type.name}: {e}")
            return False
    
    def request_shutdown(self) -> None:
        """Request graceful shutdown of the event loop."""
        self.shutdown_requested = True
        self.post_event(EventType.SHUTDOWN, EventPriority.CRITICAL)
    
    def run(self) -> None:
        """
        Main event loop.
        Processes events until shutdown is requested.
        """
        self.running = True
        self.start_time = time.time()
        pass  # Silent - event loop starting (message shown in main startup)
        
        last_status_time = time.time()
        
        try:
            while self.running and self.bot_status.get("trading_enabled", True):
                iteration_start = time.time()
                
                # Periodic status update (every 60 seconds)
                if iteration_start - last_status_time > 60:
                    events_processed = self.metrics["events_processed"]
                    uptime = iteration_start - self.start_time if hasattr(self, 'start_time') else 0
                    pass  # Silent - periodic status removed (customers don't need event loop stats)
                    last_status_time = iteration_start
                
                # Check for shutdown
                if self.shutdown_requested:
                    pass  # Silent shutdown request
                    break
                
                # Process events with timeout to prevent blocking
                try:
                    event = self.event_queue.get(timeout=0.1)
                    self._process_event(event)
                    self.event_queue.task_done()
                except Empty:
                    # No events in queue - this is normal
                    pass
                
                # Update metrics
                iteration_time_ms = (time.time() - iteration_start) * 1000
                self.metrics["loop_iterations"] += 1
                self.metrics["total_processing_time_ms"] += iteration_time_ms
                
                if iteration_time_ms > self.metrics["max_processing_time_ms"]:
                    self.metrics["max_processing_time_ms"] = iteration_time_ms
                
                # Detect stalls (iterations taking too long)
                if iteration_time_ms > 1000:  # 1 second
                    self.metrics["stall_count"] += 1
                    logger.warning(f"Event loop stall detected: {iteration_time_ms:.2f}ms")
                
                self.metrics["last_iteration_time"] = time.time()
                
                # Small sleep to prevent busy-waiting
                time.sleep(0.001)  # 1ms
        
        except Exception as e:
            logger.error(f"Event loop error: {e}", exc_info=True)
        finally:
            self._shutdown()
    
    def _process_event(self, event: Event) -> None:
        """
        Process a single event.
        
        Args:
            event: Event to process
        """
        start_time = time.time()
        
        try:
            # Get handler for this event type
            handler = self.handlers.get(event.event_type)
            
            if handler:
                handler(event)
                self.metrics["events_processed"] += 1
            else:
                logger.warning(f"No handler registered for event type: {event.event_type.name}")
            
            # Log slow events
            processing_time_ms = (time.time() - start_time) * 1000
            if processing_time_ms > 100:  # 100ms
                logger.warning(
                    f"Slow event processing: {event.event_type.name} "
                    f"took {processing_time_ms:.2f}ms"
                )
        
        except Exception as e:
            logger.error(
                f"Error processing event {event.event_type.name}: {e}",
                exc_info=True
            )
    
    def _shutdown(self) -> None:
        """Execute graceful shutdown procedures."""
        logger.info("Bot Shutting Down")
        
        self.running = False
        
        # Execute shutdown handlers
        for handler in self.shutdown_handlers:
            try:
                handler()
            except Exception as e:
                logger.error(f"Error in shutdown handler: {e}", exc_info=True)
        
        # Log final metrics
        self._log_metrics()
        
        pass  # Silent - event loop stopped
    
    def _log_metrics(self) -> None:
        """Log event loop metrics."""
        pass  # Silent - metrics logged only for debugging, not for customers
    
    def get_queue_depth(self) -> int:
        """Get current event queue depth."""
        return self.event_queue.qsize()
    
    def is_running(self) -> bool:
        """Check if event loop is running."""
        return self.running


class TimerManager:
    """
    Manages timer-based events for the trading bot.
    Checks time-based conditions and posts events to the event loop.
    """
    
    def __init__(self, event_loop: EventLoop, config: Dict[str, Any], 
                 timezone: pytz.timezone):
        """
        Initialize timer manager.
        
        Args:
            event_loop: Event loop to post timer events to
            config: Bot configuration
            timezone: Trading timezone
        """
        self.event_loop = event_loop
        self.config = config
        self.timezone = timezone
        self.last_checks: Dict[str, datetime] = {}
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        logger.info("Timer manager initialized")
    
    def start(self) -> None:
        """Start timer manager thread."""
        if self.running:
            logger.warning("Timer manager already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info("Timer manager started")
    
    def stop(self) -> None:
        """Stop timer manager thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Timer manager stopped")
    
    def _run(self) -> None:
        """Main timer loop."""
        while self.running:
            try:
                current_time = datetime.now(self.timezone)
                current_time_only = current_time.time()
                
                # Check VWAP reset time
                vwap_reset_time = self.config.get("vwap_reset_time")
                if vwap_reset_time and self._should_check("vwap_reset", current_time, 60):
                    if current_time_only >= vwap_reset_time:
                        self.event_loop.post_event(
                            EventType.VWAP_RESET,
                            EventPriority.MEDIUM,
                            {"time": current_time}
                        )
                
                # Position reconciliation check (every 5 minutes)
                if self._should_check("position_reconciliation", current_time, 300):
                    self.event_loop.post_event(
                        EventType.POSITION_RECONCILIATION,
                        EventPriority.HIGH,
                        {"time": current_time}
                    )
                
                # Connection health check (every 20 seconds)
                if self._should_check("connection_health", current_time, 20):
                    self.event_loop.post_event(
                        EventType.CONNECTION_HEALTH,
                        EventPriority.HIGH,
                        {"time": current_time}
                    )
                
                # License validation check (every 5 minutes)
                if self._should_check("license_check", current_time, 300):
                    self.event_loop.post_event(
                        EventType.LICENSE_CHECK,
                        EventPriority.HIGH,
                        {"time": current_time}
                    )
                
                # Check shutdown time (only during maintenance window: 4:50 PM - 6:00 PM ET)
                shutdown_time = self.config.get("shutdown_time")
                entry_start = self.config.get("entry_start_time")  # 6 PM - when trading resumes
                if shutdown_time and entry_start and self._should_check("shutdown", current_time, 60):
                    # Only shutdown if we're in the maintenance window (shutdown_time <= now < entry_start)
                    if shutdown_time <= current_time_only < entry_start:
                        self.event_loop.post_event(
                            EventType.SHUTDOWN,
                            EventPriority.CRITICAL,
                            {"time": current_time, "reason": "maintenance_window"}
                        )
                
                # Post periodic time check event
                if self._should_check("periodic", current_time, 1):
                    self.event_loop.post_event(
                        EventType.TIME_CHECK,
                        EventPriority.MEDIUM,
                        {"time": current_time}
                    )
                
                time.sleep(0.1)  # Check every 100ms
            
            except Exception as e:
                logger.error(f"Timer manager error: {e}", exc_info=True)
                time.sleep(1)
    
    def _should_check(self, check_name: str, current_time: datetime, 
                     interval_seconds: int) -> bool:
        """
        Determine if a check should be performed based on interval.
        
        Args:
            check_name: Name of the check
            current_time: Current time
            interval_seconds: Minimum interval between checks
        
        Returns:
            True if check should be performed
        """
        last_check = self.last_checks.get(check_name)
        
        if last_check is None:
            self.last_checks[check_name] = current_time
            return True
        
        elapsed = (current_time - last_check).total_seconds()
        
        if elapsed >= interval_seconds:
            self.last_checks[check_name] = current_time
            return True
        
        return False


# Separator constant for consistency
SEPARATOR_LINE = "=" * 60
