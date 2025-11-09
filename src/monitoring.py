"""
Enhanced Logging and Monitoring for VWAP Bounce Bot
Provides structured logging, health checks, and metrics collection
"""

import json
import logging
import logging.handlers
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
import pytz
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
        
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id'):
            log_data['correlation_id'] = record.correlation_id
            
        # Add trade context if available
        if hasattr(record, 'trade_context'):
            log_data['trade_context'] = record.trade_context
            
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
            
        # Include extra fields if requested
        if self.include_extra and hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data
            
        return json.dumps(log_data)


class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive data from logs"""
    
    SENSITIVE_KEYS = ['api_token', 'api_key', 'password', 'secret', 'authorization', 'token']
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Redact sensitive data from log messages"""
        import re
        
        message = record.getMessage()
        
        # Redact sensitive patterns
        for key in self.SENSITIVE_KEYS:
            # Create pattern that matches with optional spaces (e.g., "API Token" or "api_token")
            # Split key by underscore and allow spaces
            key_parts = key.split('_')
            pattern_key = '\\s*'.join(key_parts)
            
            if key.replace('_', ' ') in message.lower() or key in message.lower():
                # Pattern matches: "key: value" or "key=value"  
                pattern = f"{pattern_key}['\"]?\\s*[:=]\\s*['\"]?([^'\"\\s,}}]+)"
                message = re.sub(pattern, f"{key.replace('_', ' ').title()}: ***", message, flags=re.IGNORECASE)
                
        record.msg = message
        return True


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup comprehensive logging with rotation and structured format.
    
    Args:
        config: Bot configuration dictionary
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('vwap_bot')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler with human-readable format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Force UTF-8 encoding for Windows console to handle emojis
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
    
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # Add sensitive data filter
    sensitive_filter = SensitiveDataFilter()
    console_handler.addFilter(sensitive_filter)
    
    logger.addHandler(console_handler)
    
    # File handler with JSON format and rotation
    log_dir = config.get('log_directory', './logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Add account ID to log filename for multi-user support
    account_id = os.getenv('SELECTED_ACCOUNT_ID', 'default')
    log_file = os.path.join(log_dir, f'vwap_bot_{account_id}.log')
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_file,
        when='midnight',
        interval=1,
        backupCount=30,  # Keep 30 days of logs
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(StructuredFormatter())
    file_handler.addFilter(sensitive_filter)
    
    logger.addHandler(file_handler)
    
    # Performance log handler (separate file for performance metrics)
    # Also add account ID to performance log for isolation
    perf_log_file = os.path.join(log_dir, f'performance_{account_id}.log')
    perf_handler = logging.handlers.TimedRotatingFileHandler(
        perf_log_file,
        when='midnight',
        interval=1,
        backupCount=30
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(StructuredFormatter())
    
    # Create performance logger
    perf_logger = logging.getLogger('vwap_bot.performance')
    perf_logger.addHandler(perf_handler)
    
    return logger


@dataclass
class HealthCheckStatus:
    """Health check status information"""
    healthy: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    

class HealthChecker:
    """Provides health check endpoints and status monitoring"""
    
    def __init__(self, bot_status: Dict[str, Any], config: Dict[str, Any]):
        self.bot_status = bot_status
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.custom_checks: List[Callable[[], Tuple[bool, str]]] = []
        
    def add_custom_check(self, check_func: Callable[[], Tuple[bool, str]]) -> None:
        """Add a custom health check function"""
        self.custom_checks.append(check_func)
        
    def check_bot_status(self) -> Tuple[bool, str]:
        """Check if bot is in healthy trading state"""
        if self.bot_status.get('emergency_stop', False):
            return False, "Emergency stop is active"
            
        if not self.bot_status.get('trading_enabled', False):
            return False, f"Trading disabled: {self.bot_status.get('stop_reason', 'Unknown')}"
            
        return True, "Bot status OK"
    
    def check_broker_connection(self) -> Tuple[bool, str]:
        """Check broker connection health"""
        # This would check actual broker connection
        # For now, we check if we've had recent data
        last_tick_time = self.bot_status.get('last_tick_time')
        
        if last_tick_time is None:
            return True, "No data received yet (startup)"
        
        # Make sure we compare timezone-aware datetimes
        now = datetime.now(pytz.UTC)
        if last_tick_time.tzinfo is None:
            # If last_tick_time is naive, assume UTC
            last_tick_time = pytz.UTC.localize(last_tick_time)
        else:
            # Convert to UTC for comparison
            last_tick_time = last_tick_time.astimezone(pytz.UTC)
            
        time_since_tick = (now - last_tick_time).total_seconds()
        timeout = self.config.get('tick_timeout_seconds', 60)
        
        if time_since_tick > timeout:
            return False, f"No data for {time_since_tick:.0f} seconds"
            
        return True, "Broker connection OK"
    
    def check_data_feed(self) -> Tuple[bool, str]:
        """Check data feed status"""
        last_tick_time = self.bot_status.get('last_tick_time')
        
        if last_tick_time is None:
            return True, "Waiting for data"
        
        # Make sure we compare timezone-aware datetimes
        now = datetime.now(pytz.UTC)
        if last_tick_time.tzinfo is None:
            # If last_tick_time is naive, assume UTC
            last_tick_time = pytz.UTC.localize(last_tick_time)
        else:
            # Convert to UTC for comparison
            last_tick_time = last_tick_time.astimezone(pytz.UTC)
            
        # Check for recent data
        time_since_tick = (now - last_tick_time).total_seconds()
        
        if time_since_tick > 60:
            return False, f"Data feed lag: {time_since_tick:.0f}s"
            
        return True, "Data feed OK"
    
    def get_status(self) -> HealthCheckStatus:
        """Get comprehensive health check status"""
        checks = {}
        messages = []
        
        # Run built-in checks
        is_ok, msg = self.check_bot_status()
        checks['bot_status'] = is_ok
        if not is_ok:
            messages.append(msg)
            
        is_ok, msg = self.check_broker_connection()
        checks['broker_connection'] = is_ok
        if not is_ok:
            messages.append(msg)
            
        is_ok, msg = self.check_data_feed()
        checks['data_feed'] = is_ok
        if not is_ok:
            messages.append(msg)
            
        # Run custom checks
        for i, check_func in enumerate(self.custom_checks):
            try:
                is_ok, msg = check_func()
                checks[f'custom_check_{i}'] = is_ok
                if not is_ok:
                    messages.append(msg)
            except Exception as e:
                checks[f'custom_check_{i}'] = False
                messages.append(f"Check failed: {str(e)}")
                
        # Overall health is OK if all checks pass
        healthy = all(checks.values())
        
        return HealthCheckStatus(
            healthy=healthy,
            checks=checks,
            messages=messages if not healthy else ["All systems operational"]
        )


class HealthCheckHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler for health check endpoint"""
    
    health_checker: Optional[HealthChecker] = None
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/health' or self.path == '/':
            status = self.health_checker.get_status()
            
            # Return 200 if healthy, 503 if not
            status_code = 200 if status.healthy else 503
            
            response = {
                'healthy': status.healthy,
                'timestamp': status.timestamp.isoformat(),
                'checks': status.checks,
                'messages': status.messages
            }
            
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response, indent=2).encode())
        else:
            self.send_response(404)
            self.end_headers()
            
    def log_message(self, format, *args):
        """Suppress default HTTP logging"""
        pass


class HealthCheckServer:
    """HTTP server for health checks"""
    
    def __init__(self, health_checker: HealthChecker, port: int = 8080):
        self.health_checker = health_checker
        self.port = port
        self.server = None
        self.thread = None
        self.logger = logging.getLogger(__name__)
        
    def start(self) -> None:
        """Start the health check server"""
        HealthCheckHTTPHandler.health_checker = self.health_checker
        
        self.server = HTTPServer(('0.0.0.0', self.port), HealthCheckHTTPHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        
        self.logger.info(f"Health check server started on port {self.port}")
        
    def stop(self) -> None:
        """Stop the health check server"""
        if self.server:
            self.server.shutdown()
            self.logger.info("Health check server stopped")


@dataclass
class Metrics:
    """Container for bot metrics"""
    total_trades_today: int = 0
    daily_pnl: float = 0.0
    current_position_qty: int = 0
    open_order_count: int = 0
    api_error_count: int = 0
    data_feed_lag_ms: float = 0.0
    event_loop_iteration_time_ms: float = 0.0
    api_call_latency_ms: float = 0.0
    order_execution_time_ms: float = 0.0
    

class MetricsCollector:
    """Collects and tracks bot performance metrics"""
    
    def __init__(self):
        self.metrics = Metrics()
        self.api_latencies = deque(maxlen=100)  # Last 100 API calls
        self.event_loop_times = deque(maxlen=1000)  # Last 1000 iterations
        self.logger = logging.getLogger('vwap_bot.performance')
        
    def record_api_call(self, latency_ms: float, success: bool) -> None:
        """Record API call latency and status"""
        self.api_latencies.append(latency_ms)
        
        if not success:
            self.metrics.api_error_count += 1
            
        # Update average latency
        if self.api_latencies:
            self.metrics.api_call_latency_ms = sum(self.api_latencies) / len(self.api_latencies)
            
    def record_event_loop_iteration(self, duration_ms: float) -> None:
        """Record event loop iteration time"""
        self.event_loop_times.append(duration_ms)
        
        # Update average
        if self.event_loop_times:
            self.metrics.event_loop_iteration_time_ms = sum(self.event_loop_times) / len(self.event_loop_times)
            
        # Log warning if iteration is slow
        if duration_ms > 100:
            self.logger.warning(f"Slow event loop iteration: {duration_ms:.2f}ms")
            
    def record_order_execution(self, duration_ms: float) -> None:
        """Record order execution time"""
        self.metrics.order_execution_time_ms = duration_ms
        
        # Log if execution is slow
        if duration_ms > 1000:
            self.logger.warning(f"Slow order execution: {duration_ms:.2f}ms")
            
    def update_position_metrics(self, position_qty: int, daily_pnl: float, daily_trades: int) -> None:
        """Update position and P&L metrics"""
        self.metrics.current_position_qty = position_qty
        self.metrics.daily_pnl = daily_pnl
        self.metrics.total_trades_today = daily_trades
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics as dictionary"""
        return asdict(self.metrics)


class AuditLogger:
    """Specialized logger for audit trail"""
    
    def __init__(self, log_dir: str = './logs'):
        self.logger = logging.getLogger('vwap_bot.audit')
        self.logger.setLevel(logging.INFO)
        
        # Create audit log handler
        os.makedirs(log_dir, exist_ok=True)
        audit_file = os.path.join(log_dir, 'audit.log')
        
        handler = logging.handlers.TimedRotatingFileHandler(
            audit_file,
            when='midnight',
            interval=1,
            backupCount=365,  # Keep 1 year
            encoding='utf-8'
        )
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
        
    def log_trade(self, trade_data: Dict[str, Any]) -> None:
        """Log trade execution with full context"""
        self.logger.info(
            'Trade executed',
            extra={
                'extra_data': {
                    'event_type': 'trade_execution',
                    **trade_data
                }
            }
        )
        
    def log_signal(self, signal_data: Dict[str, Any], accepted: bool, reason: str = None) -> None:
        """Log signal evaluation"""
        self.logger.info(
            f'Signal {"accepted" if accepted else "rejected"}',
            extra={
                'extra_data': {
                    'event_type': 'signal_evaluation',
                    'accepted': accepted,
                    'reason': reason,
                    **signal_data
                }
            }
        )
        
    def log_position_change(self, position_data: Dict[str, Any]) -> None:
        """Log position changes"""
        self.logger.info(
            'Position changed',
            extra={
                'extra_data': {
                    'event_type': 'position_change',
                    **position_data
                }
            }
        )
        
    def log_risk_check(self, check_type: str, passed: bool, details: Dict[str, Any]) -> None:
        """Log risk limit checks"""
        self.logger.info(
            f'Risk check: {check_type} {"passed" if passed else "failed"}',
            extra={
                'extra_data': {
                    'event_type': 'risk_check',
                    'check_type': check_type,
                    'passed': passed,
                    **details
                }
            }
        )
        
    def log_parameter_change(self, parameter: str, old_value: Any, new_value: Any, reason: str) -> None:
        """Log parameter changes"""
        self.logger.info(
            f'Parameter changed: {parameter}',
            extra={
                'extra_data': {
                    'event_type': 'parameter_change',
                    'parameter': parameter,
                    'old_value': old_value,
                    'new_value': new_value,
                    'reason': reason
                }
            }
        )
