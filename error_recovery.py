"""
Error Recovery Mechanisms for VWAP Bounce Bot
Handles all failure modes and ensures the bot can recover from errors.
"""

import logging
import time
import json
import os
from typing import Dict, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import pytz


logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors the bot can encounter."""
    NETWORK_DISCONNECTION = "network_disconnection"
    DATA_FEED_INTERRUPTION = "data_feed_interruption"
    ORDER_REJECTION = "order_rejection"
    PARTIAL_FILL = "partial_fill"
    POSITION_DISCREPANCY = "position_discrepancy"
    API_RATE_LIMIT = "api_rate_limit"
    CLOCK_SYNC_ISSUE = "clock_sync_issue"
    DISK_SPACE_LOW = "disk_space_low"
    MEMORY_LEAK = "memory_leak"
    SDK_CRASH = "sdk_crash"


class RecoveryAction(Enum):
    """Actions to take when recovering from errors."""
    RETRY = "retry"
    RECONNECT = "reconnect"
    PAUSE_TRADING = "pause_trading"
    FLATTEN_POSITION = "flatten_position"
    ALERT_AND_WAIT = "alert_and_wait"
    SHUTDOWN = "shutdown"


@dataclass
class CircuitBreakerState:
    """State of a circuit breaker."""
    operation_name: str
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    is_open: bool = False
    half_open_time: Optional[float] = None


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    Opens after consecutive failures, prevents cascading failures.
    """
    
    def __init__(self, name: str, failure_threshold: int = 5, 
                 timeout_seconds: int = 60):
        """
        Initialize circuit breaker.
        
        Args:
            name: Operation name
            failure_threshold: Number of failures before opening
            timeout_seconds: Seconds to wait before trying half-open
        """
        self.state = CircuitBreakerState(operation_name=name)
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        
        logger.info(f"Circuit breaker '{name}' initialized "
                   f"(threshold={failure_threshold}, timeout={timeout_seconds}s)")
    
    def call(self, operation: Callable, *args, **kwargs) -> Tuple[bool, Any]:
        """
        Execute operation through circuit breaker.
        
        Args:
            operation: Function to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
        
        Returns:
            Tuple of (success: bool, result: Any)
        """
        # Check if circuit is open
        if self.state.is_open:
            # Check if timeout has passed for half-open attempt
            if self.state.last_failure_time:
                elapsed = time.time() - self.state.last_failure_time
                
                if elapsed < self.timeout_seconds:
                    logger.warning(
                        f"Circuit breaker '{self.state.operation_name}' is OPEN "
                        f"({self.timeout_seconds - elapsed:.1f}s remaining)"
                    )
                    return False, None
                else:
                    # Try half-open state
                    logger.info(f"Circuit breaker '{self.state.operation_name}' "
                               f"attempting half-open")
                    self.state.half_open_time = time.time()
        
        # Attempt operation
        try:
            result = operation(*args, **kwargs)
            self._record_success()
            return True, result
        
        except Exception as e:
            self._record_failure()
            logger.error(f"Circuit breaker operation failed: {e}")
            return False, None
    
    def _record_success(self) -> None:
        """Record successful operation."""
        self.state.success_count += 1
        self.state.last_success_time = time.time()
        
        # Close circuit if it was open
        if self.state.is_open:
            logger.info(f"Circuit breaker '{self.state.operation_name}' CLOSED")
            self.state.is_open = False
            self.state.failure_count = 0
            self.state.half_open_time = None
    
    def _record_failure(self) -> None:
        """Record failed operation."""
        self.state.failure_count += 1
        self.state.last_failure_time = time.time()
        
        # Open circuit if threshold reached
        if self.state.failure_count >= self.failure_threshold:
            if not self.state.is_open:
                logger.error(
                    f"Circuit breaker '{self.state.operation_name}' OPENED "
                    f"after {self.state.failure_count} failures"
                )
                self.state.is_open = True
    
    def reset(self) -> None:
        """Manually reset circuit breaker."""
        logger.info(f"Circuit breaker '{self.state.operation_name}' manually reset")
        self.state.failure_count = 0
        self.state.is_open = False
        self.state.half_open_time = None
    
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self.state.is_open
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state as dictionary."""
        return asdict(self.state)


class RetryManager:
    """
    Manages retry logic with exponential backoff.
    """
    
    def __init__(self, max_retries: int = 3, initial_delay: float = 1.0,
                 max_delay: float = 60.0, backoff_factor: float = 2.0):
        """
        Initialize retry manager.
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            backoff_factor: Multiplier for exponential backoff
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        
        logger.info(f"Retry manager initialized (max_retries={max_retries}, "
                   f"initial_delay={initial_delay}s, backoff_factor={backoff_factor})")
    
    def execute_with_retry(self, operation: Callable, operation_name: str,
                          *args, **kwargs) -> Tuple[bool, Any]:
        """
        Execute operation with exponential backoff retry.
        
        Args:
            operation: Function to execute
            operation_name: Name for logging
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Tuple of (success: bool, result: Any)
        """
        delay = self.initial_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                result = operation(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"{operation_name} succeeded on attempt {attempt + 1}")
                
                return True, result
            
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                    )
                    logger.info(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    
                    # Exponential backoff
                    delay = min(delay * self.backoff_factor, self.max_delay)
                else:
                    logger.error(
                        f"{operation_name} failed after {self.max_retries + 1} attempts: {e}"
                    )
                    return False, None
        
        return False, None


class StatePersistence:
    """
    Handles saving and loading critical bot state to disk.
    """
    
    def __init__(self, state_file: str = "bot_state.json"):
        """
        Initialize state persistence.
        
        Args:
            state_file: Path to state file
        """
        self.state_file = state_file
        logger.info(f"State persistence initialized (file={state_file})")
    
    def save_state(self, state: Dict[str, Any]) -> bool:
        """
        Save state to disk.
        
        Args:
            state: State dictionary to save
        
        Returns:
            True if successful
        """
        try:
            # Create backup of existing state
            if os.path.exists(self.state_file):
                backup_file = f"{self.state_file}.backup"
                os.replace(self.state_file, backup_file)
            
            # Save new state
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.debug(f"State saved to {self.state_file}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """
        Load state from disk.
        
        Returns:
            State dictionary if successful, None otherwise
        """
        try:
            if not os.path.exists(self.state_file):
                logger.info("No existing state file found")
                return None
            
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Verify state integrity
            if self._verify_state(state):
                logger.info(f"State loaded from {self.state_file}")
                return state
            else:
                logger.error("State file corrupted - attempting backup")
                return self._load_backup()
        
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return self._load_backup()
    
    def _verify_state(self, state: Dict[str, Any]) -> bool:
        """
        Verify state integrity.
        
        Args:
            state: State dictionary
        
        Returns:
            True if state is valid
        """
        # Check for required fields
        required_fields = ["timestamp", "position", "equity"]
        
        for field in required_fields:
            if field not in state:
                logger.warning(f"State missing required field: {field}")
                return False
        
        return True
    
    def _load_backup(self) -> Optional[Dict[str, Any]]:
        """Load state from backup file."""
        backup_file = f"{self.state_file}.backup"
        
        try:
            if not os.path.exists(backup_file):
                logger.warning("No backup state file found")
                return None
            
            with open(backup_file, 'r') as f:
                state = json.load(f)
            
            if self._verify_state(state):
                logger.info(f"State loaded from backup: {backup_file}")
                return state
            else:
                logger.error("Backup state file also corrupted")
                return None
        
        except Exception as e:
            logger.error(f"Failed to load backup state: {e}")
            return None


class ConnectionMonitor:
    """
    Monitors network connectivity and handles reconnection.
    """
    
    def __init__(self, broker_interface: Any, retry_manager: RetryManager):
        """
        Initialize connection monitor.
        
        Args:
            broker_interface: Broker interface instance
            retry_manager: Retry manager for reconnection attempts
        """
        self.broker = broker_interface
        self.retry_manager = retry_manager
        self.is_connected = False
        self.last_check_time = time.time()
        self.reconnect_count = 0
        
        logger.info("Connection monitor initialized")
    
    def check_connection(self) -> bool:
        """
        Check if connection is active.
        
        Returns:
            True if connected
        """
        try:
            self.is_connected = self.broker.is_connected()
            self.last_check_time = time.time()
            return self.is_connected
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            self.is_connected = False
            return False
    
    def reconnect(self) -> bool:
        """
        Attempt to reconnect to broker.
        
        Returns:
            True if reconnection successful
        """
        logger.info("Attempting broker reconnection...")
        self.reconnect_count += 1
        
        success, _ = self.retry_manager.execute_with_retry(
            self.broker.connect,
            "broker_reconnect"
        )
        
        if success:
            logger.info(f"Reconnection successful (attempt #{self.reconnect_count})")
            self.is_connected = True
            return True
        else:
            logger.error(f"Reconnection failed after retries (attempt #{self.reconnect_count})")
            self.is_connected = False
            return False


class DataFeedMonitor:
    """
    Monitors market data feed for interruptions.
    """
    
    def __init__(self, stale_data_threshold: int = 60):
        """
        Initialize data feed monitor.
        
        Args:
            stale_data_threshold: Seconds without data before considered stale
        """
        self.stale_threshold = stale_data_threshold
        self.last_tick_time: Optional[float] = None
        self.is_stale = False
        
        logger.info(f"Data feed monitor initialized (threshold={stale_data_threshold}s)")
    
    def update_tick(self) -> None:
        """Record receipt of new tick data."""
        self.last_tick_time = time.time()
        
        if self.is_stale:
            logger.info("Data feed recovered - no longer stale")
            self.is_stale = False
    
    def check_data_feed(self) -> bool:
        """
        Check if data feed is current.
        
        Returns:
            True if data is current (not stale)
        """
        if self.last_tick_time is None:
            return True  # Haven't started receiving data yet
        
        elapsed = time.time() - self.last_tick_time
        
        if elapsed > self.stale_threshold:
            if not self.is_stale:
                logger.error(f"Data feed is stale (no ticks for {elapsed:.1f}s)")
                self.is_stale = True
            return False
        
        return True
    
    def is_data_stale(self) -> bool:
        """Check if data is currently stale."""
        return self.is_stale


class ErrorRecoveryManager:
    """
    Coordinates all error recovery mechanisms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize error recovery manager.
        
        Args:
            config: Bot configuration
        """
        self.config = config
        
        # Initialize components
        self.retry_manager = RetryManager(max_retries=3, initial_delay=1.0)
        self.state_persistence = StatePersistence()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Create default circuit breakers
        self._create_circuit_breakers()
        
        logger.info("Error recovery manager initialized")
    
    def _create_circuit_breakers(self) -> None:
        """Create circuit breakers for different operations."""
        operations = [
            ("order_placement", 5, 60),
            ("market_data", 10, 30),
            ("account_query", 5, 60),
        ]
        
        for name, threshold, timeout in operations:
            self.circuit_breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=threshold,
                timeout_seconds=timeout
            )
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """
        Get circuit breaker by name.
        
        Args:
            name: Circuit breaker name
        
        Returns:
            CircuitBreaker instance or None
        """
        return self.circuit_breakers.get(name)
    
    def handle_error(self, error_type: ErrorType, context: Dict[str, Any]) -> RecoveryAction:
        """
        Determine recovery action for error type.
        
        Args:
            error_type: Type of error encountered
            context: Additional context about the error
        
        Returns:
            Recommended recovery action
        """
        logger.warning(f"Handling error: {error_type.value}")
        
        # Map error types to recovery actions
        if error_type == ErrorType.NETWORK_DISCONNECTION:
            return RecoveryAction.RECONNECT
        
        elif error_type == ErrorType.DATA_FEED_INTERRUPTION:
            return RecoveryAction.PAUSE_TRADING
        
        elif error_type == ErrorType.ORDER_REJECTION:
            rejection_reason = context.get("reason", "unknown")
            logger.error(f"Order rejected: {rejection_reason}")
            
            # Specific handling based on rejection reason
            if "margin" in rejection_reason.lower():
                return RecoveryAction.FLATTEN_POSITION
            else:
                return RecoveryAction.ALERT_AND_WAIT
        
        elif error_type == ErrorType.PARTIAL_FILL:
            return RecoveryAction.RETRY  # Adjust position and continue
        
        elif error_type == ErrorType.POSITION_DISCREPANCY:
            return RecoveryAction.FLATTEN_POSITION
        
        elif error_type == ErrorType.API_RATE_LIMIT:
            return RecoveryAction.PAUSE_TRADING
        
        elif error_type == ErrorType.SDK_CRASH:
            return RecoveryAction.SHUTDOWN
        
        else:
            return RecoveryAction.ALERT_AND_WAIT
    
    def save_state(self, state: Dict[str, Any]) -> bool:
        """Save bot state to disk."""
        return self.state_persistence.save_state(state)
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """Load bot state from disk."""
        return self.state_persistence.load_state()


# Separator constant
SEPARATOR_LINE = "=" * 60
