"""
Capitulation Reversal Bot - Mean Reversion Trading Strategy
Event-driven bot that trades reversals after panic selling/buying flushes

THE EDGE:
Wait for panic selling or panic buying. When everyone is rushing for the exits
(or FOMO buying), step in the opposite direction and ride the snapback to VWAP.

STRATEGY FLOW:
1. DETECT THE FLUSH - Price dropped/pumped 20+ ticks in 5-10 min (2x ATR)
2. CONFIRM EXHAUSTION - Volume spike then decline, extreme RSI, momentum fading
3. ENTRY TRIGGER - Reversal candle closes, price stretched from VWAP
4. STOP LOSS - 2-4 ticks below/above flush low/high
5. PROFIT TARGET - VWAP (mean reversion destination)
6. TRADE MANAGEMENT - Breakeven at 12 ticks, trail after 15 ticks

TRADEABLE REGIMES:
- HIGH_VOL_TRENDING: Big moves happen, good for this strategy
- HIGH_VOL_CHOPPY: Still has flushes, just choppier

SKIP THESE REGIMES:
- NORMAL, NORMAL_CHOPPY, LOW_VOL: Not enough volatility for real flushes

========================================================================
24/7 MULTI-USER READY ARCHITECTURE - US EASTERN TIME (DST-AWARE)
========================================================================

This bot is designed to run continuously using US Eastern wall-clock time:

Γ£à US EASTERN TIME: Uses US/Eastern timezone (handles EST/EDT automatically via pytz)
Γ£à AUTO-FLATTEN: Automatically closes positions at 4:45 PM ET (15 min before maintenance)
Γ£à AUTO-RESUME: Automatically resumes trading when market reopens (6:00 PM ET)
Γ£à NO MANUAL SHUTDOWN: Bot runs 24/7, just pauses trading when market closed
Γ£à DST-AWARE: pytz automatically handles daylight saving time transitions

CME Futures Trading Schedule (US Eastern Wall-Clock):
- MAIN SESSION OPENS: 6:00 PM Eastern (market resumes after maintenance)
- ENTRY CUTOFF: 4:00 PM Eastern (no new positions after this time)
- FLATTEN POSITIONS: 4:45 PM Eastern (close existing positions before maintenance)
- DAILY MAINTENANCE: 4:45-6:00 PM Eastern (1hr 15min daily break)
- SUNDAY OPEN: 6:00 PM Eastern Sunday (weekly start)
- FRIDAY CLOSE: 4:45 PM Eastern (weekly close, start of weekend maintenance)
- THANKSGIVING: Last Thursday of November - flatten at 12:45 PM, market closes 1:00 PM ET

IMPORTANT ENTRY/EXIT RULES:
- Bot can OPEN new positions: 6:00 PM - 4:00 PM next day
- Bot can HOLD existing positions: Until 4:45 PM (forced flatten time)
- Gap between 4:00 PM - 4:45 PM: Can hold positions but cannot open new ones

NOTE: These times NEVER change - always same wall-clock time regardless of DST.
pytz handles EST (UTC-5) and EDT (UTC-4) conversions automatically.

Bot States:
- entry_window: Market open, can trade (6:00 PM - 4:45 PM)
- closed: During maintenance (4:45-6:00 PM ET) or weekend, auto-flatten positions

Friday Special Rules:
- Trading ends at 4:45 PM ET (start of weekend maintenance)
- No special flatten logic needed on Friday, market closes at maintenance time

For Multi-User Subscriptions:
- All users see US Eastern times (CME standard)
- Bot uses Eastern Time, can display in user's local timezone if needed
- Each user gets their own position/RL/VWAP state

"""

import os
import sys
import logging
import argparse

# CRITICAL: Suppress ALL project_x_py loggers BEFORE any other imports
# Install a filter on the root logger to block all project_x_py child loggers
class _SuppressProjectXLoggers(logging.Filter):
    def filter(self, record):
        return not record.name.startswith('project_x_py')

logging.getLogger().addFilter(_SuppressProjectXLoggers())

# Also suppress the parent logger directly with extreme prejudice
_project_x_root = logging.getLogger('project_x_py')
_project_x_root.setLevel(logging.CRITICAL + 1)  # Beyond CRITICAL to block everything
_project_x_root.propagate = False
_project_x_root.handlers = []
_project_x_root.addHandler(logging.NullHandler())  # Add null handler to absorb any logs
_project_x_root.disabled = True  # Completely disable the logger

from datetime import datetime, timedelta
from datetime import time as datetime_time  # Alias to avoid conflict with time.time()
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Callable
import pytz
import time as time_module  # Import time module with alias
import statistics  # For calculating statistics like mean, median, etc.
import asyncio
import hashlib
import signal
import atexit

# Load environment variables at module import time
from dotenv import load_dotenv
from pathlib import Path

# Determine project root and load .env
PROJECT_ROOT = Path(__file__).parent.parent
env_path = PROJECT_ROOT / '.env'
load_dotenv(dotenv_path=env_path)

# Import rainbow logo display - with fallback if not available
try:
    from rainbow_logo import display_animated_logo, Colors, get_rainbow_bot_art, get_rainbow_bot_art_with_message, display_animated_thank_you, display_static_thank_you
    RAINBOW_LOGO_AVAILABLE = True
except ImportError:
    RAINBOW_LOGO_AVAILABLE = False
    display_animated_logo = None
    Colors = None
    get_rainbow_bot_art = None
    get_rainbow_bot_art_with_message = None
    display_animated_thank_you = None
    display_static_thank_you = None

# Startup logo configuration
STARTUP_LOGO_DURATION = 8.0  # Seconds to display startup logo

# ===== EXE-COMPATIBLE FILE PATH HELPERS =====
# These ensure files are saved in the correct location whether running as:
# - Python script (development)
# - PyInstaller EXE (customer distribution)

def get_application_path() -> 'Path':
    """
    Get the application's base directory.
    Works correctly whether running as script or frozen EXE.
    
    When frozen (EXE): Returns directory containing the EXE
    When script: Returns project root directory
    
    Returns:
        Path: Application base directory where data/ and logs/ folders live
    """
    from pathlib import Path
    
    if getattr(sys, 'frozen', False):
        # Running as compiled EXE (PyInstaller)
        # Use the directory containing the EXE, not the temp _MEIPASS
        application_path = Path(sys.executable).parent
    else:
        # Running as Python script - go up one level from src/
        application_path = Path(__file__).parent.parent
    
    return application_path


def get_device_fingerprint(symbol: str = None) -> str:
    """
    Generate a unique device fingerprint for session locking.
    Supports multi-symbol mode: each symbol gets its own session on the same device.
    
    Components:
    - Machine ID (MAC address via uuid.getnode)
    - Username (from getpass)
    - Platform name (Windows/Linux/Darwin)
    - Symbol (optional, for multi-symbol support)
    
    Args:
        symbol: Optional trading symbol (e.g., 'ES', 'NQ'). When provided,
               creates a symbol-specific fingerprint allowing multiple symbols
               to run on the same device without session conflicts.
    
    Returns:
        Unique device fingerprint (hashed for privacy)
    
    Security Note: 
    - Launcher uses fingerprint without symbol (validates license once)
    - Each bot instance uses fingerprint WITH symbol (creates symbol-specific session)
    - This allows multiple symbols to run concurrently on the same device
    """
    import hashlib
    import platform
    import getpass
    import uuid
    
    # Get platform-specific machine ID
    try:
        machine_id = str(uuid.getnode())  # MAC address as unique ID
    except:
        machine_id = "unknown"
    
    # Get username
    try:
        username = getpass.getuser()
    except:
        username = "unknown"
    
    # Get platform info
    platform_name = platform.system()  # Windows, Darwin (Mac), Linux
    
    # Combine all components - include symbol if provided for multi-symbol support
    # Each symbol gets its own session to prevent conflicts when running multiple symbols
    if symbol:
        fingerprint_raw = f"{machine_id}:{username}:{platform_name}:{symbol}"
    else:
        fingerprint_raw = f"{machine_id}:{username}:{platform_name}"
    
    # Hash for privacy (don't send raw MAC address to server)
    fingerprint_hash = hashlib.sha256(fingerprint_raw.encode()).hexdigest()[:16]
    
    return fingerprint_hash


def get_data_file_path(filename: str) -> 'Path':
    """
    Get full path to a data file, creating directory if needed.
    Works for both script and EXE modes.
    
    Args:
        filename: Relative path like "data/bot_state.json" or "logs/vwap_bot.log"
    
    Returns:
        Path: Full absolute path to file
        
    Example:
        # Development: C:/Users/kevin/Downloads/simple-bot-1/data/bot_state.json
        # Customer EXE: C:/Users/customer/QuoTrading/data/bot_state.json
    """
    from pathlib import Path
    
    app_path = get_application_path()
    file_path = app_path / filename
    
    # Create parent directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    return file_path

# Import new production modules
from config import load_config, BotConfiguration, DEFAULT_MAX_STOP_LOSS_DOLLARS
from event_loop import EventLoop, EventType, EventPriority, TimerManager
from error_recovery import ErrorRecoveryManager, ErrorType as RecoveryErrorType
from bid_ask_manager import BidAskManager, BidAskQuote
from notifications import get_notifier
from signal_confidence import SignalConfidenceRL
from regime_detection import get_regime_detector, REGIME_DEFINITIONS, is_regime_tradeable
from capitulation_detector import get_capitulation_detector, CapitulationDetector, FlushEvent
from cloud_api import CloudAPIClient

# Conditionally import broker (only needed for live trading, not backtesting)
try:
    from broker_interface import create_broker, BrokerInterface
except ImportError:
    # Broker interface not available (e.g., in backtest-only mode)
    create_broker = None
    BrokerInterface = None


# ============================================================================
# BACKTEST MODE UTILITIES
# ============================================================================

def is_backtest_mode() -> bool:
    """
    Check if running in backtest mode.
    Centralized check to avoid code duplication.
    
    Returns:
        True if in backtest mode, False otherwise
    """
    return os.getenv('BOT_BACKTEST_MODE', '').lower() in ('true', '1', 'yes')


# Load configuration from environment and config module
# Check if running in backtest mode from environment variable
_is_backtest = is_backtest_mode()
_bot_config = load_config(backtest_mode=_is_backtest)
# Only validate if not in backtest mode (backtest mode skips broker requirements)
if not _is_backtest:
    _bot_config.validate()  # Validate configuration at startup

# Convert BotConfiguration to dictionary for backward compatibility with existing code
CONFIG: Dict[str, Any] = _bot_config.to_dict()

# Auto-load symbol specifications if available (for multi-symbol support)
SYMBOL_SPEC = None
try:
    from symbol_specs import get_symbol_spec
    SYMBOL_SPEC = get_symbol_spec(CONFIG["instrument"])
    
    # Override config with symbol-specific values if not explicitly set by user
    if not os.getenv("BOT_TICK_VALUE"):
        CONFIG["tick_value"] = SYMBOL_SPEC.tick_value
        _bot_config.tick_value = SYMBOL_SPEC.tick_value
    
    if not os.getenv("BOT_TICK_SIZE"):
        CONFIG["tick_size"] = SYMBOL_SPEC.tick_size
        _bot_config.tick_size = SYMBOL_SPEC.tick_size
    
    if not os.getenv("BOT_SLIPPAGE_TICKS"):
        CONFIG["slippage_ticks"] = SYMBOL_SPEC.typical_slippage_ticks
        _bot_config.slippage_ticks = SYMBOL_SPEC.typical_slippage_ticks
    
except Exception as e:
    # Symbol specs not available - will use defaults from config
    pass


def get_symbol_tick_specs(symbol: str) -> Tuple[float, float]:
    """
    Get tick_size and tick_value for a symbol.
    
    Dynamically looks up symbol specs from the symbol_specs database.
    Used by both Live Mode and AI Mode for accurate P&L calculations.
    
    Handles various symbol formats:
    - Standard: "ES", "NQ", "CL"
    - TopStep: "F.US.EP", "F.US.NP", "F.US.CL"
    - Tradovate: "ESZ24", "NQZ24"
    - Contract IDs: "CON.F.US.EP.Z25"
    
    Args:
        symbol: Trading symbol in any format
        
    Returns:
        Tuple of (tick_size, tick_value)
        Falls back to CONFIG defaults if symbol not found
    """
    try:
        from symbol_specs import SYMBOL_SPECS
        symbol_upper = symbol.upper()
        
        # Try direct lookup
        if symbol in SYMBOL_SPECS:
            spec = SYMBOL_SPECS[symbol]
            return spec.tick_size, spec.tick_value
        
        # Also try uppercase
        if symbol_upper in SYMBOL_SPECS:
            spec = SYMBOL_SPECS[symbol_upper]
            return spec.tick_size, spec.tick_value
        
        # Check broker symbol mappings (e.g., "F.US.EP" -> "ES")
        for spec_symbol, spec in SYMBOL_SPECS.items():
            if hasattr(spec, 'broker_symbols') and spec.broker_symbols:
                for broker_name, broker_symbol in spec.broker_symbols.items():
                    broker_symbol_upper = broker_symbol.upper()
                    # Exact match
                    if broker_symbol_upper == symbol_upper:
                        return spec.tick_size, spec.tick_value
                    # Symbol ends with broker symbol (e.g., "F.US.EP" ends with "EP")
                    if symbol_upper.endswith("." + broker_symbol_upper):
                        return spec.tick_size, spec.tick_value
                    # Symbol contains broker symbol (e.g., "CON.F.US.EP.Z25" contains "EP")
                    if "." + broker_symbol_upper + "." in symbol_upper or symbol_upper.endswith("." + broker_symbol_upper):
                        return spec.tick_size, spec.tick_value
        
        # Check if standard symbol is contained in the input (e.g., "ESZ24" contains "ES")
        for spec_symbol, spec in SYMBOL_SPECS.items():
            if spec_symbol in symbol_upper:
                return spec.tick_size, spec.tick_value
        
    except Exception:
        pass
    
    # Fallback to config defaults (ES-like values)
    return CONFIG.get("tick_size", 0.25), CONFIG.get("tick_value", 12.50)


def normalize_symbol_to_standard(symbol: str) -> Optional[str]:
    """
    Normalize any symbol format to the standard trading symbol.
    
    Handles various symbol formats from different brokers:
    - Standard: "ES", "NQ", "MNQ" -> returns as-is
    - TopStep broker format: "F.US.EP" -> "ES", "F.US.MNQEP" -> "MNQ"
    - Contract IDs: "CON.F.US.EP.Z25" -> "ES", "CON.F.US.MNQEP.Z25" -> "MNQ"
    
    This is critical for AI Mode which needs to match positions from the broker
    with the symbol the user configured in the GUI.
    
    Args:
        symbol: Trading symbol in any format
        
    Returns:
        Standard trading symbol (e.g., "ES", "MNQ") or None if not found
    """
    if not symbol:
        return None
    
    try:
        from symbol_specs import SYMBOL_SPECS
        symbol_upper = symbol.upper()
        
        # Direct lookup - already a standard symbol
        if symbol_upper in SYMBOL_SPECS:
            return symbol_upper
        
        # Check broker symbol mappings to find the standard symbol
        # This reverses the mapping: broker format -> standard symbol
        # Sort by TopStep symbol length (longest first) to avoid partial matches
        # e.g., "F.US.MESEP" should match before "F.US.EP"
        sorted_specs = sorted(
            SYMBOL_SPECS.items(),
            key=lambda x: len(x[1].broker_symbols.get('topstep', '')) if hasattr(x[1], 'broker_symbols') and x[1].broker_symbols else 0,
            reverse=True
        )
        
        for std_symbol, spec in sorted_specs:
            if not hasattr(spec, 'broker_symbols') or not spec.broker_symbols:
                continue
            
            topstep_symbol = spec.broker_symbols.get('topstep', '')
            if not topstep_symbol:
                continue
            
            topstep_upper = topstep_symbol.upper()
            
            # Check if the broker symbol matches or is contained in the input
            # E.g., "F.US.MNQEP" in "CON.F.US.MNQEP.Z25"
            # Using word boundary check: the broker symbol should be surrounded by
            # dots or be at the start/end of the string
            if topstep_upper in symbol_upper:
                # Verify it's a proper match (not a partial word match)
                # Find the position and check boundaries
                idx = symbol_upper.find(topstep_upper)
                if idx >= 0:
                    # Check that it's at a boundary (start or preceded by .)
                    before_ok = idx == 0 or symbol_upper[idx-1] == '.'
                    # Check that it's at a boundary (end or followed by .)
                    after_idx = idx + len(topstep_upper)
                    after_ok = after_idx == len(symbol_upper) or symbol_upper[after_idx] == '.'
                    if before_ok and after_ok:
                        return std_symbol
            
            # Also check for the key part with dot boundaries
            # E.g., ".MNQEP." in ".MNQEP." or ends with ".MNQEP"
            broker_parts = topstep_upper.split('.')
            if len(broker_parts) >= 2:
                key_part = broker_parts[-1]  # Last part like "MNQEP" or "EP"
                if f".{key_part}." in symbol_upper or symbol_upper.endswith(f".{key_part}"):
                    return std_symbol
        
        # Check if a standard symbol name is directly in the input
        # E.g., "MNQ" in "MNQZ24" or "MNQ" in some format
        # Sort by symbol length (longest first) to avoid "ES" matching before "MES"
        sorted_symbols = sorted(SYMBOL_SPECS.keys(), key=len, reverse=True)
        for std_symbol in sorted_symbols:
            # Check for word boundary match: symbol should be at start/end or surrounded by non-alpha chars
            idx = symbol_upper.find(std_symbol)
            if idx >= 0:
                # Check before boundary
                before_ok = idx == 0 or not symbol_upper[idx-1].isalpha()
                # Check after boundary (allow alphanumeric for month codes like "Z24")
                after_idx = idx + len(std_symbol)
                after_ok = after_idx == len(symbol_upper) or not symbol_upper[after_idx].isalpha() or symbol_upper[after_idx:after_idx+1].isdigit() or symbol_upper[after_idx:after_idx+1] in 'FGHJKMNQUVXZ'  # Month codes
                if before_ok and after_ok:
                    return std_symbol
        
    except Exception:
        pass
    
    return None


def symbols_match(symbol1: str, symbol2: str) -> bool:
    """
    Check if two symbols refer to the same trading instrument.
    
    Handles cases where broker returns symbol in a different format
    than what the user configured. For example:
    - User configures "MNQ" in GUI
    - Broker returns position with symbol "F.US.MNQEP" or contract_id
    
    Args:
        symbol1: First symbol (e.g., from broker position)
        symbol2: Second symbol (e.g., user's configured symbol)
        
    Returns:
        True if both symbols refer to the same instrument
    """
    if not symbol1 or not symbol2:
        return False
    
    # Direct comparison (case-insensitive)
    if symbol1.upper() == symbol2.upper():
        return True
    
    # Normalize both to standard symbols and compare
    std1 = normalize_symbol_to_standard(symbol1)
    std2 = normalize_symbol_to_standard(symbol2)
    
    if std1 and std2:
        return std1 == std2
    
    return False


# String constants
MSG_LIVE_TRADING_NOT_IMPLEMENTED = "Live trading not implemented - SDK integration required"
SEPARATOR_LINE = "=" * 60

# Daily Loss Limit Threshold
DAILY_LOSS_APPROACHING_THRESHOLD = 0.80  # Stop trading at 80% of daily loss limit

# Idle Mode Configuration
IDLE_STATUS_MESSAGE_INTERVAL = 300  # Show status message every 5 minutes (300 seconds) during idle

# Regime Detection Constants
DEFAULT_FALLBACK_ATR = 5.0  # Default ATR when calculation not possible (ES futures typical value)

# Global broker instance (replaces sdk_client)
broker: Optional[BrokerInterface] = None

# Global event loop instance
event_loop: Optional[EventLoop] = None

# Global error recovery manager
recovery_manager: Optional[ErrorRecoveryManager] = None

# Global timer manager
timer_manager: Optional[TimerManager] = None

# Global RL brain for signal confidence learning (used in both live and backtest modes)
rl_brain: Optional[SignalConfidenceRL] = None

# Global cloud API client for reporting trade outcomes (data collection only)
cloud_api_client: Optional[CloudAPIClient] = None

# Global bid/ask manager
bid_ask_manager: Optional[BidAskManager] = None

# Global trading symbol for multi-symbol session support
# Set early in main() and used by session-related functions
# Note: Each bot process runs independently with its own symbol (not thread-shared)
current_trading_symbol: Optional[str] = None


def get_current_symbol_for_session() -> str:
    """
    Get the current trading symbol for session-related operations.
    Provides consistent fallback logic for all session functions.
    
    Returns:
        Trading symbol from current_trading_symbol or CONFIG fallback
    """
    return current_trading_symbol if current_trading_symbol else CONFIG.get("instrument", "ES")


# State management dictionary
state: Dict[str, Any] = {}

# Backtest mode tracking
# When True, runs in backtest mode using historical data (no broker connections)
# Note: Cloud API may still be used for outcome reporting in live mode
# Global variable to track simulation time during backtesting.
# When None, get_current_time() uses real datetime.now()
# When set (by handle_tick_event), get_current_time() uses this historical timestamp
backtest_current_time: Optional[datetime] = None

# Global tracking for safety mechanisms (Phase 12)
bot_status: Dict[str, Any] = {
    "trading_enabled": True,
    "starting_equity": None,
    "last_tick_time": None,
    "emergency_stop": False,
    "stop_reason": None,
    # PRODUCTION: Track trading costs
    "total_slippage_cost": 0.0,  # Total slippage costs across all trades
    "total_commission": 0.0,  # Total commissions across all trades
    # Track target wait decisions
    "target_wait_wins": 0,
    "target_wait_losses": 0,
    "early_close_wins": 0,
    "early_close_losses": 0,
    "early_close_saves": 0,
    "flatten_mode": False,
    "session_start_time": None,  # Track when bot started for session runtime display
    # CRITICAL FIX: Track pending entry orders to prevent duplicate orders
    # When an entry order is inflight, position reconciliation should not clear state
    "entry_order_pending": False,  # True when an entry order is being processed
    "entry_order_pending_since": None,  # Timestamp when entry started
    "entry_order_pending_symbol": None,  # Symbol being traded
    "entry_order_pending_id": None,  # Order ID of pending order (for verification)
    # FIX: Track flatten in progress to prevent spam
    # When a flatten operation is initiated, prevent repeated attempts
    "flatten_in_progress": False,  # True when a flatten order has been placed
    "flatten_in_progress_since": None,  # Timestamp when flatten started
    "flatten_in_progress_symbol": None,  # Symbol being flattened
}

# Timeout for flatten in progress check (seconds)
FLATTEN_IN_PROGRESS_TIMEOUT = 30


def clear_flatten_flags() -> None:
    """
    Clear all flatten-related flags in bot_status.
    Call this when position flatten is confirmed complete.
    """
    bot_status["flatten_in_progress"] = False
    bot_status["flatten_in_progress_since"] = None
    bot_status["flatten_in_progress_symbol"] = None


def set_flatten_flags(symbol: str) -> None:
    """
    Set flatten flags to indicate a flatten operation is in progress.
    Call this before initiating a flatten order.
    
    Args:
        symbol: Symbol being flattened
    """
    bot_status["flatten_in_progress"] = True
    bot_status["flatten_in_progress_since"] = datetime.now()
    bot_status["flatten_in_progress_symbol"] = symbol


def verify_broker_position_for_flatten(symbol: str, expected_side: str, expected_quantity: int) -> Tuple[bool, int, str]:
    """
    Verify with broker that we have a position to flatten.
    
    This safeguard prevents over-flattening that could create opposite positions.
    
    Args:
        symbol: Symbol to check
        expected_side: Expected position side ('long' or 'short')
        expected_quantity: Expected position quantity
        
    Returns:
        Tuple of (can_flatten, actual_quantity, reason)
        - can_flatten: True if we should proceed with flatten
        - actual_quantity: The actual quantity to flatten (from broker)
        - reason: Description of any mismatch or issue
    """
    broker_position = get_position_quantity(symbol)
    
    # No position at broker
    if broker_position == 0:
        return False, 0, "Position already flat at broker"
    
    # Determine broker side and quantity
    broker_side = "long" if broker_position > 0 else "short"
    broker_quantity = abs(broker_position)
    
    # Verify side matches
    if broker_side != expected_side:
        return False, 0, f"Position side mismatch: bot={expected_side}, broker={broker_side}"
    
    # Log quantity mismatch but still proceed with broker's quantity
    if broker_quantity != expected_quantity:
        logger.warning(f"Position quantity mismatch: bot={expected_quantity}, broker={broker_quantity} - using broker quantity")
    
    return True, broker_quantity, "Position verified"


def setup_logging() -> logging.Logger:
    """Configure logging for the bot - Console only (no log files for customers)"""
    
    # CRITICAL: Suppress ALL project_x_py loggers with root-level filter
    class SuppressProjectXLoggers(logging.Filter):
        def filter(self, record):
            return not record.name.startswith('project_x_py')
    
    # Install filter on root logger FIRST
    logging.getLogger().addFilter(SuppressProjectXLoggers())
    
    # Also suppress the parent logger directly with extreme prejudice
    project_x_root = logging.getLogger('project_x_py')
    project_x_root.setLevel(logging.CRITICAL + 1)  # Beyond CRITICAL to block everything
    project_x_root.propagate = False
    project_x_root.handlers = []
    project_x_root.addHandler(logging.NullHandler())  # Add null handler to absorb any logs
    project_x_root.disabled = True  # Completely disable the logger
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',  # Clean format - no timestamps or module names for customer UI
        handlers=[
            logging.StreamHandler()  # Console output only - customers don't need log files
        ]
    )
    
    # Suppress technical noise per LOGGING_SPECIFICATION.md Section "🚫 LOGS THAT SHOULD BE SUPPRESSED"
    
    # 1. Third-party libraries (SDK internals, HTTP, WebSocket)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    
    # Suppress ALL project_x_py loggers by disabling propagation at root level
    project_x_logger = logging.getLogger('project_x_py')
    project_x_logger.setLevel(logging.CRITICAL + 1)  # Beyond CRITICAL to block everything
    project_x_logger.propagate = False  # Don't propagate to root logger
    project_x_logger.handlers = []  # Clear all handlers to prevent JSON output
    project_x_logger.addHandler(logging.NullHandler())  # Add null handler to absorb any logs
    project_x_logger.disabled = True  # Completely disable the logger
    
    # SignalR WebSocket logging - only show warnings and above (not connection close tracebacks)
    # Connection close errors during maintenance are expected and handled in broker_websocket.py
    signalr_logger = logging.getLogger('signalrcore')
    signalr_logger.setLevel(logging.WARNING)  # Only warnings and above
    
    # SignalRCoreClient is the internal logger that logs tracebacks - set to WARNING
    signalr_client_logger = logging.getLogger('SignalRCoreClient')
    signalr_client_logger.setLevel(logging.WARNING)  # Suppress DEBUG/INFO but keep warnings
    
    # Websocket library - connection errors during maintenance are expected
    websocket_logger = logging.getLogger('websocket')
    websocket_logger.setLevel(logging.WARNING)  # Only warnings and above
    
    # Suppress all nested project_x_py loggers (they use deeply nested child loggers)
    # These loggers output JSON which clutters customer UI
    for logger_name in ['project_x_py.statistics', 
                        'project_x_py.statistics.bounded_statistics',
                        'project_x_py.statistics.bounded_statistics.bounded_stats',
                        'project_x_py.order_manager', 
                        'project_x_py.order_manager.core',
                        'project_x_py.position_manager',
                        'project_x_py.position_manager.core',
                        'project_x_py.trading_suite',
                        'project_x_py.data_manager', 
                        'project_x_py.risk_manager']:
        child_logger = logging.getLogger(logger_name)
        child_logger.setLevel(logging.CRITICAL + 1)  # Beyond CRITICAL
        child_logger.propagate = False
        child_logger.handlers = []  # Clear all handlers
        child_logger.addHandler(logging.NullHandler())  # Add null handler
        child_logger.disabled = True  # Completely disable
    
    # 2. Initialization & Setup (RL brain, bid/ask manager, event loop, broker SDK details)
    logging.getLogger('signal_confidence').setLevel(logging.WARNING)  # RL brain initialization
    logging.getLogger('bid_ask_manager').setLevel(logging.WARNING)  # Bid/ask manager initialization
    logging.getLogger('event_loop').setLevel(logging.ERROR)  # Event loop initialization & stats (suppress warnings)
    logging.getLogger('broker_interface').setLevel(logging.ERROR)  # Broker SDK initialization details (suppress warnings)
    
    # 3. Order Management (order placement confirmations, IDs, internals)
    # broker_interface already handles this at WARNING level
    
    # 4. Broker Communication (heartbeats, websocket, health checks)
    logging.getLogger('broker_websocket').setLevel(logging.WARNING)  # WebSocket connection details
    
    # 5. Cloud & Data Sync (cloud API sync, heartbeats, file operations)
    logging.getLogger('cloud_api').setLevel(logging.WARNING)  # Cloud API communication
    
    # 6. State Management (file saves, serialization, session fingerprints)
    logging.getLogger('session_state').setLevel(logging.WARNING)  # State serialization details
    
    # 7. Non-Critical Errors (notification failures, alert delivery errors)
    logging.getLogger('notifications').setLevel(logging.WARNING)  # Notification send failures
    logging.getLogger('error_recovery').setLevel(logging.WARNING)  # Non-critical error handling
    
    # 8. Regime Detection Internals (algorithm details, thresholds - show only changes)
    logging.getLogger('regime_detection').setLevel(logging.WARNING)  # Regime detection internals
    
    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================================
# CLOUD API INTEGRATION - Data Collection Only
# ============================================================================
# Cloud API is used ONLY for reporting trade outcomes (data collection).
# Trading decisions (confidence/approval) are made locally using RL brain.



async def get_ml_confidence_async(rl_state: Dict[str, Any], side: str) -> Tuple[bool, float, str]:
    """
    Get RL decision from local RL brain for both live and backtest modes.
    
    LIVE MODE: Uses local rl_brain for confidence decisions (no cloud dependency)
    BACKTEST MODE: Uses local rl_brain for learning and testing
    
    Returns: (take_signal, confidence, reason)
    """
    global rl_brain
    
    # Use local RL brain for all modes (live and backtest)
    if rl_brain is not None:
        return rl_brain.should_take_signal(rl_state)
    
    # Fallback if RL brain not initialized
    logger.warning("RL brain not initialized - using default approval")
    return True, 0.65, "No RL brain initialized - default approval"


def get_ml_confidence(rl_state: Dict[str, Any], side: str) -> Tuple[bool, float, str]:
    """Synchronous wrapper for get_ml_confidence_async"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(get_ml_confidence_async(rl_state, side))
    finally:
        loop.close()



async def save_trade_experience_async(
    rl_state: Dict[str, Any],
    side: str,
    pnl: float,
    duration_minutes: float,
    execution_data: Dict[str, Any]
) -> None:
    """
    Save trade outcome based on mode.
    
    LIVE MODE: Saves to cloud ONLY (reads local for pattern matching, but doesn't save local)
    BACKTEST MODE: Saves to local RL brain only
    SHADOW MODE: Does NOT send to cloud (signal-only mode)
    AI MODE: Does NOT send to cloud (position management mode)
    """
    global cloud_api_client, rl_brain
    
    # BACKTEST MODE: Save to local RL brain only
    if is_backtest_mode() or CONFIG.get("backtest_mode", False):
        if rl_brain is not None:
            rl_brain.record_outcome(rl_state, True, pnl, duration_minutes, execution_data)
        return
    
    # SHADOW MODE: Do NOT send data to cloud RL database
    # This mode is for user experimentation and should not pollute the training data
    if CONFIG.get("shadow_mode", False):
        return
    
    # LIVE MODE: Report to cloud ONLY (don't save locally)
    if cloud_api_client is None:
        pass  # Silent - not customer-facing
        return
    
    try:
        # Add side and price to state
        state_with_context = rl_state.copy()
        state_with_context['side'] = side.lower()
        state_with_context['price'] = state.get("current_price", 0)
        
        # Convert duration to seconds
        duration_seconds = duration_minutes * 60.0
        
        # Report to cloud in background (non-blocking) with execution_data
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            cloud_api_client.report_trade_outcome,
            state_with_context,
            True,  # took_trade
            pnl,
            duration_seconds,
            execution_data  # Pass execution metrics to cloud
        )
        
        pass  # Silent - cloud outcome reporting is transparent
        
    except Exception as e:
        pass  # Silent - cloud sync is transparent to customer



def save_trade_experience(
    rl_state: Dict[str, Any],
    side: str,
    pnl: float,
    duration_minutes: float,
    execution_data: Dict[str, Any]
) -> None:
    """Synchronous wrapper for save_trade_experience_async"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(save_trade_experience_async(rl_state, side, pnl, duration_minutes, execution_data))
    finally:
        loop.close()


# ============================================================================
# PHASE TWO: SDK Integration
# ============================================================================

def validate_license_at_startup() -> None:
    """
    Validate license at bot startup BEFORE any initialization.
    This is the "login screen" - checks license and session lock.
    Called at the very beginning of main() to fail fast if license is invalid.
    
    Uses current_trading_symbol global (set in main before calling this)
    to create symbol-specific sessions for multi-symbol support.
    Falls back to CONFIG symbol if current_trading_symbol is not set.
    """
    global current_trading_symbol
    
    # Skip in backtest mode
    if is_backtest_mode():
        pass  # Silent - backtest mode initialization
        return
    
    license_key = os.getenv("QUOTRADING_LICENSE_KEY")
    
    if not license_key:
        logger.critical("=" * 70)
        logger.critical("NO LICENSE KEY FOUND")
        logger.critical("Please set QUOTRADING_LICENSE_KEY in your .env file")
        logger.critical("Contact support@quotrading.com to purchase a license")
        logger.critical("=" * 70)
        sys.exit(1)
    
    pass  # Silent - license validation in progress
    try:
        import requests
        api_url = os.getenv("QUOTRADING_API_URL", "https://quotrading-flask-api.azurewebsites.net")
        
        # Get device fingerprint WITH symbol for multi-symbol session support
        # Each symbol gets its own session to prevent conflicts
        symbol_for_session = get_current_symbol_for_session()
        device_fp = get_device_fingerprint(symbol_for_session)
        import os as os_mod
        pid = os_mod.getpid()
        pass  # Silent - device fingerprint internal
        
        # Validate with server using /api/validate-license (session locking)
        # Include device fingerprint for session locking (symbol-specific)
        # MULTI-SYMBOL FIX: Include symbol explicitly so server can manage per-symbol sessions
        response = requests.post(
            f"{api_url}/api/validate-license",
            json={
                "license_key": license_key,
                "device_fingerprint": device_fp,  # Session locking (symbol-specific)
                "symbol": symbol_for_session,  # MULTI-SYMBOL: Explicit symbol for server-side session management
                "check_only": False  # Create/claim session
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("license_valid"):
                logger.info(f"✅ License validated - {data.get('message', 'Access Granted')}")
            else:
                reason = data.get('message', 'Unknown error')
                logger.critical("=" * 70)
                logger.critical("INVALID OR EXPIRED LICENSE")
                logger.critical(f"Reason: {reason}")
                if "expired" in reason.lower():
                    logger.critical("Your license has expired. Please renew to continue trading.")
                logger.critical("Contact: support@quotrading.com")
                logger.critical("=" * 70)
                sys.exit(1)
        elif response.status_code == 403:
            # Check if it's a session conflict or license expiration
            data = response.json()
            
            # Check for license expiration first (explicit check for False)
            if "license_valid" in data and data["license_valid"] is False:
                reason = data.get('message', 'Unknown error')
                logger.critical("=" * 70)
                logger.critical("INVALID OR EXPIRED LICENSE")
                logger.critical(f"Reason: {reason}")
                if "expir" in reason.lower():
                    logger.critical("")
                    logger.critical("  ⚠️ YOUR LICENSE HAS EXPIRED")
                    logger.critical("")
                    logger.critical("  Your license key has expired and needs to be renewed.")
                    logger.critical("  Please renew your subscription to continue trading.")
                    logger.critical("")
                logger.critical("  Contact: support@quotrading.com")
                logger.critical("=" * 70)
                sys.exit(1)
            elif data.get("session_conflict"):
                # Session conflict - another device is ACTIVELY using this license
                # Server already auto-clears stale sessions, so this is a real conflict
                logger.critical("=" * 70)
                logger.critical("")
                logger.critical("  ⚠️ LICENSE ALREADY IN USE")
                logger.critical("")
                logger.critical("  Your license key is currently active on another device.")
                logger.critical("  Only one device can use a license at a time.")
                logger.critical("")
                logger.critical("  If the other device is not running, wait a moment and try again.")
                logger.critical("  Contact: support@quotrading.com")
                logger.critical("")
                logger.critical("=" * 70)
                sys.exit(1)
            else:
                # Generic 403 error - could be license expiration without explicit flag
                reason = data.get('message', 'Unknown error')
                logger.critical("=" * 70)
                logger.critical("LICENSE VALIDATION FAILED")
                logger.critical(f"Reason: {reason}")
                # Check message for expiration keywords as fallback
                if "expir" in reason.lower():
                    logger.critical("")
                    logger.critical("  ⚠️ YOUR LICENSE HAS EXPIRED")
                    logger.critical("")
                    logger.critical("  Your license key has expired and needs to be renewed.")
                    logger.critical("  Please renew your subscription to continue trading.")
                    logger.critical("")
                logger.critical("  Contact: support@quotrading.com")
                logger.critical("=" * 70)
                sys.exit(1)
        else:
            logger.critical(f"License validation failed - HTTP {response.status_code}")
            logger.critical("Please contact support@quotrading.com")
            sys.exit(1)
    except Exception as e:
        logger.critical(f"License validation error: {e}")
        logger.critical("Cannot start bot without valid license")
        sys.exit(1)


def initialize_broker() -> None:
    """
    Initialize the broker interface using configuration.
    Uses configured broker with error recovery and circuit breaker.
    SHADOW MODE: Shows trading signals without executing (manual trading mode).
    
    Note: License validation is done at startup in validate_license_at_startup().
    This function only handles broker connection.
    """
    global broker, recovery_manager
    
    # In shadow mode, show signals only (no execution)
    if CONFIG.get("shadow_mode", False):
        logger.info("≡ƒôè SIGNAL-ONLY MODE - Shows signals without executing trades")
    
    pass  # Silent - broker initialization is internal
    
    # Create error recovery manager
    recovery_manager = ErrorRecoveryManager(CONFIG)
    
    # Create broker using configuration
    # In shadow mode, broker streams data but doesn't execute actual orders
    broker = create_broker(_bot_config.api_token, _bot_config.username, CONFIG["instrument"])
    
    # Connect to broker (initial connection doesn't use circuit breaker)
    pass  # Silent - connection in progress
    if not broker.connect():
        logger.error("Failed to connect to broker")
        return False
        raise RuntimeError("Broker connection failed")
    
    logger.info("✅ Bot Ready - Connected to broker")
    
    # Send bot startup alert
    try:
        notifier = get_notifier()
        notifier.send_error_alert(
            error_message=f"Bot started successfully and connected to broker. Ready to trade {CONFIG.get('instrument', 'configured symbol')}.",
            error_type="Bot Started"
        )
    except Exception as e:
        pass  # Silent - notification failure not critical


def check_azure_time_service() -> str:
    """
    Check Azure time service for trading permission.
    Called every 20 seconds alongside kill switch check.
    
    Azure provides single source of truth for:
    - Current UTC time (timezone-accurate)
    - Market hours status
    - Maintenance windows (4:45-6:00 PM ET daily)
    - Trading permission (go/no-go flag)
    
    Returns:
        Trading state: 'entry_window' or 'closed'
    """
    # Skip in backtest mode - use local time logic instead
    if is_backtest_mode():
        return None
    
    try:
        import requests
        
        cloud_api_url = CONFIG.get("cloud_api_url", "https://quotrading-flask-api.azurewebsites.net")
        
        response = requests.get(
            f"{cloud_api_url}/api/main",
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            trading_allowed = data.get("trading_allowed", False)
            halt_reason = data.get("halt_reason", "")
            current_time_str = data.get("current_utc", data.get("current_et", ""))
            
            # Store Azure time for bot awareness
            bot_status["azure_time"] = current_time_str
            bot_status["trading_allowed"] = trading_allowed
            bot_status["halt_reason"] = halt_reason
            
            # Determine state based on Azure response
            if not trading_allowed:
                if "maintenance" in halt_reason.lower():
                    state = "closed"  # Maintenance window
                elif "weekend" in halt_reason.lower() or "closed" in halt_reason.lower():
                    state = "closed"  # Weekend or market closed
                else:
                    state = "closed"  # Unknown halt reason - be safe
            else:
                # Trading allowed - use entry_window state
                state = "entry_window"
            
            # Cache state for get_trading_state() to use
            bot_status["azure_trading_state"] = state
            return state
        else:
            # Azure unreachable - clear cached state to trigger fallback
            pass  # Silent fallback to local time
            bot_status["azure_trading_state"] = None
            return None  # Signal to use local get_trading_state()
            
    except Exception as e:
        # Non-critical - if cloud unreachable, fall back to local time
        pass  # Silent - cloud service optional
        bot_status["azure_trading_state"] = None
        return None  # Signal to use local get_trading_state()


def check_broker_connection() -> None:
    """
    Periodic health check for broker connection AND cloud services.
    Verifies connection is alive and attempts reconnection if needed.
    Called every 20 seconds by timer manager.
    Only logs when there's an issue to avoid spam.
    """
    global broker
    
    # Skip all broker/cloud checks in backtest mode
    if is_backtest_mode():
        return
    
    # CRITICAL: Check cloud time service
    try:
        check_azure_time_service()
    except Exception as e:
        pass  # Silent - time service optional
    
    # Send heartbeat to show bot is online
    try:
        send_heartbeat()
    except Exception as e:
        pass  # Silent - heartbeat failure not customer-facing
    
    # AUTO-IDLE: Disconnect broker during maintenance (no data needed)
    current_time = get_current_time()
    trading_state = get_trading_state(current_time)
    
    if trading_state == "closed" and not bot_status.get("maintenance_idle", False):
        halt_reason = bot_status.get("halt_reason", "")
        
        # Ensure time is in Eastern
        eastern_tz = pytz.timezone('US/Eastern')
        if current_time.tzinfo is None:
            current_time = eastern_tz.localize(current_time)
        eastern_time = current_time.astimezone(eastern_tz)
        
        # Determine if maintenance or weekend
        # Weekend: Friday 4:45 PM - Sunday 6:00 PM ET
        # Maintenance: Mon-Thu 4:45 PM - 6:00 PM ET
        is_friday_close = (eastern_time.weekday() == 4 and eastern_time.time() >= datetime_time(16, 45))
        is_saturday = eastern_time.weekday() == 5
        is_sunday_before_open = (eastern_time.weekday() == 6 and eastern_time.time() < datetime_time(18, 0))
        is_weekend = is_friday_close or is_saturday or is_sunday_before_open
        
        is_maintenance = (eastern_time.weekday() < 4 and  # Mon-Thu only
                         eastern_time.time() >= datetime_time(16, 45) and 
                         eastern_time.time() < datetime_time(18, 0))
        
        if is_maintenance or is_weekend or "maintenance" in halt_reason.lower() or "weekend" in halt_reason.lower():
            # Determine idle reason for clear messaging
            if is_weekend:
                idle_type = "WEEKEND"
                idle_msg = "Weekend market closure (Fri 4:45 PM - Sun 6:00 PM ET)"
                reopen_msg = "Will auto-reconnect Sunday at 6:00 PM ET"
            else:
                idle_type = "MAINTENANCE"
                idle_msg = "Daily maintenance window (4:45 PM - 6:00 PM ET)"
                reopen_msg = "Will auto-reconnect at 6:00 PM ET"
            
            # Display session summary before going idle (like Ctrl+C)
            symbol = CONFIG.get("instrument")
            if symbol and symbol in state:
                log_session_summary(symbol, logout_success=True, show_logout_status=False, show_bot_art=False)
            
            logger.critical(SEPARATOR_LINE)
            logger.critical(f"[IDLE MODE] {idle_type} - GOING IDLE")
            logger.critical(f"Time: {eastern_time.strftime('%H:%M:%S %Z')}")
            logger.critical(f"  Reason: {idle_msg}")
            logger.critical(f"  Disconnecting broker to save resources")
            logger.critical(f"  {reopen_msg}")
            logger.critical(SEPARATOR_LINE)
            
            # Disconnect broker (stops all data feeds)
            try:
                if broker is not None and broker.connected:
                    broker.disconnect()
                    logger.critical("  [OK] Broker disconnected - Bot is IDLE")
            except Exception as e:
                logger.error(f"  [ERROR] Error disconnecting: {e}")
            
            bot_status["maintenance_idle"] = True
            bot_status["idle_type"] = idle_type  # Store for status message
            # CRITICAL: Keep trading_enabled = True so event loop keeps running
            # bot_status["trading_enabled"] = False  # REMOVED - bot stays running
            bot_status["last_idle_message_time"] = eastern_time
            bot_status["idle_heartbeat_count"] = 0  # Initialize heartbeat counter
            
            # Display idle status with heartbeat
            logger.info("")
            logger.info("\033[93m⏸  Market Maintenance - Waiting for market data...\033[0m")  # Yellow
            logger.info("")
            logger.critical(f"  Bot stays ON and IDLE - checking periodically for market reopen...")
            logger.critical(f"  Press Ctrl+C to stop bot")
            return  # Skip broker health check since we just disconnected
    
    # Display idle status message every 5 minutes during maintenance/weekend
    elif trading_state == "closed" and bot_status.get("maintenance_idle", False):
        # SILENT DURING MAINTENANCE - no spam in logs
        # The initial maintenance message was already shown when entering maintenance
        # Next log will be when market reopens
        return  # Skip broker health check during idle period
    
    # AUTO-RECONNECT: Reconnect broker when market reopens at 6:00 PM ET
    elif trading_state == "entry_window" and bot_status.get("maintenance_idle", False):
        idle_type = bot_status.get("idle_type", "MAINTENANCE")
        logger.critical(SEPARATOR_LINE)
        logger.critical(f"[RECONNECT] {idle_type} COMPLETE - MARKET REOPENED - AUTO-RECONNECTING")
        logger.critical(f"Time: {current_time.strftime('%H:%M:%S %Z')}")
        logger.critical(SEPARATOR_LINE)
        
        # Reconnect to broker
        try:
            if broker is not None:
                logger.critical("  [RECONNECT] Connecting to broker...")
                success = broker.connect(max_retries=3)
                if success:
                    logger.critical("  [RECONNECT] [OK] Broker connected")
                    
                    # CRITICAL FIX: Re-subscribe to market data after reconnection
                    # Without this, the bot reconnects but no data flows because
                    # subscriptions were lost when the websocket was disconnected
                    logger.critical("  [RECONNECT] Re-subscribing to data feeds...")
                    resubscribe_market_data_after_reconnect()
                    
                    bot_status["maintenance_idle"] = False
                    bot_status["idle_type"] = None
                    bot_status["trading_enabled"] = True
                    logger.info("Trading resumed after maintenance")
                else:
                    logger.error("  [RECONNECT] [ERROR] Connection failed - Will retry periodically")
        except Exception as e:
            logger.error(f"  [RECONNECT] Error: {e}")
        
        logger.critical(SEPARATOR_LINE)
        return  # Skip normal health check since we just reconnected
    
    if broker is None:
        logger.error("[HEALTH] Broker is None - cannot check connection")
        return
    
    # Check if broker reports as connected
    if not broker.connected:
        logger.warning("[HEALTH] Broker disconnected - reconnecting immediately...")
        
        # Send connection error alert
        try:
            notifier = get_notifier()
            notifier.send_error_alert(
                error_message="Broker connection lost. Reconnecting now...",
                error_type="Connection Error"
            )
        except Exception as e:
            pass  # Silent - notification error not customer-facing
        
        try:
            # Immediate reconnect with 3 retries
            logger.critical("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.critical("⚠️  RECONNECTING TO TOPSTEP")
            success = broker.connect(max_retries=3)
            if success:
                logger.critical("✅ Reconnection successful")
                
                # CRITICAL FIX: Re-subscribe to market data after reconnection
                # Without this, the bot reconnects but no data flows because
                # subscriptions were lost when the websocket was disconnected
                resubscribe_market_data_after_reconnect()
                
                logger.critical("✅ Trading resumed")
                logger.critical("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                
                # Send success notification
                try:
                    notifier.send_error_alert(
                        error_message="✅ Reconnected to TopStep successfully. Bot is back online.",
                        error_type="Connection Restored"
                    )
                except:
                    pass
            else:
                logger.error("❌ Reconnection failed - will retry in 30s")
                logger.critical("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        except Exception as e:
            logger.error(f"[HEALTH] Reconnection error: {e}")
        return
    
    # Connection looks healthy - do a lightweight ping test
    # Only log if there's a problem (silent success to avoid spam)
    try:
        # IMPROVED: Use dedicated health check method instead of equity query
        if hasattr(broker, 'verify_connection'):
            is_healthy = broker.verify_connection()
            if not is_healthy:
                logger.warning("[HEALTH] Connection verification failed - marking as disconnected")
                broker.connected = False
                return
        else:
            # Fallback to equity check for brokers without verify_connection
            equity = broker.get_account_equity()
            if equity is None or equity <= 0:
                logger.warning("[HEALTH] Connection may be stale - got invalid equity response")
                broker.connected = False
    except Exception as e:
        logger.warning(f"[HEALTH] Connection check failed: {e}")
        # Mark as disconnected to trigger reconnect on next check
        broker.connected = False


def get_account_equity() -> float:
    """
    Fetch current account equity from broker.
    Returns account equity/balance with error handling.
    In shadow mode, returns account_size from config (actual fetched balance from launcher).
    In live mode, returns actual account balance from broker.
    """
    # Shadow mode or no broker - return account size from config
    if _bot_config.shadow_mode or broker is None:
        # Use starting_equity from bot_status if available
        if bot_status.get("starting_equity") is not None:
            return bot_status["starting_equity"]
        # Use account_size from config (fetched from broker during launcher login)
        # This is the actual account balance the user selected in the launcher
        return float(CONFIG.get("account_size", 50000.0))
    
    # Live mode - get actual balance from broker account
    try:
        # Use circuit breaker for account query
        breaker = recovery_manager.get_circuit_breaker("account_query")
        success, equity = breaker.call(broker.get_account_equity)
        
        if success:
            pass  # Silent - account equity query internal
            return equity
        else:
            logger.error("Failed to get account equity")
            return 0.0
    except Exception as e:
        logger.error(f"Error fetching account equity: {e}")
        action = recovery_manager.handle_error(
            RecoveryErrorType.SDK_CRASH,
            {"error": str(e), "function": "get_account_equity"}
        )
        return 0.0

def place_market_order(symbol: str, side: str, quantity: int) -> Optional[Dict[str, Any]]:
    """
    Place a market order through the broker interface.
    
    Args:
        symbol: Instrument symbol (e.g., 'MES')
        side: 'BUY' or 'SELL'
        quantity: Number of contracts
    
    Returns:
        Order object or None if failed
    """
    # Backtest mode: Simulate order without broker
    if is_backtest_mode():
        pass  # Silent - backtest order simulated
        return {
            "order_id": f"BACKTEST_{datetime.now().timestamp()}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": "MARKET",
            "status": "FILLED",
            "backtest": True
        }
    
    pass  # Silent - order logged at higher level (entry/exit)
    
    if broker is None:
        logger.error("Broker not initialized")
        return None
    
    try:
        # Use circuit breaker for order placement
        breaker = recovery_manager.get_circuit_breaker("order_placement")
        success, order = breaker.call(broker.place_market_order, symbol, side, quantity)
        
        if success and order:
            # Post order fill event to event loop
            if event_loop:
                event_loop.post_event(
                    EventType.ORDER_FILL,
                    EventPriority.HIGH,
                    {"order": order, "symbol": symbol}
                )
            return order
        else:
            logger.error("Market order placement failed")
            
            # Send order error alert
            try:
                notifier = get_notifier()
                notifier.send_error_alert(
                    error_message=f"Failed to place market order: {side} {quantity} {symbol}",
                    error_type="Order Error"
                )
            except Exception as e:
                pass  # Silent - alert failure not critical
            
            action = recovery_manager.handle_error(
                RecoveryErrorType.ORDER_REJECTION,
                {"symbol": symbol, "side": side, "quantity": quantity}
            )
            return None
    except Exception as e:
        logger.error(f"Error placing market order: {e}")
        
        # Send order error alert
        try:
            notifier = get_notifier()
            notifier.send_error_alert(
                error_message=f"Exception placing market order: {str(e)}",
                error_type="Order Error"
            )
        except Exception as alert_error:
            pass  # Silent - alert failure not critical
        
        action = recovery_manager.handle_error(
            RecoveryErrorType.SDK_CRASH,
            {"error": str(e), "function": "place_market_order"}
        )
        return None


def place_stop_order(symbol: str, side: str, quantity: int, stop_price: float) -> Optional[Dict[str, Any]]:
    """
    Place a stop order through the broker interface.
    
    Args:
        symbol: Instrument symbol
        side: 'BUY' or 'SELL'
        quantity: Number of contracts
        stop_price: Stop trigger price
    
    Returns:
        Order object or None if failed
    """
    # Backtest mode: Simulate order without broker
    if is_backtest_mode():
        pass  # Silent - backtest stop order simulated
        return {
            "order_id": f"BACKTEST_STOP_{datetime.now().timestamp()}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": "STOP",
            "stop_price": stop_price,
            "status": "PENDING",
            "backtest": True
        }
    
    shadow_mode = _bot_config.shadow_mode
    pass  # Silent - stop order logged at higher level
    
    if shadow_mode:
        return {
            "order_id": f"SHADOW_STOP_{datetime.now().timestamp()}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": "STOP",
            "stop_price": stop_price,
            "status": "PENDING",
            "dry_run": True
        }
    
    if broker is None:
        logger.error("Broker not initialized")
        return None
    
    try:
        # Use circuit breaker for order placement
        breaker = recovery_manager.get_circuit_breaker("order_placement")
        success, order = breaker.call(broker.place_stop_order, symbol, side, quantity, stop_price)
        
        if success and order:
            return order
        else:
            logger.error("Stop order placement failed")
            action = recovery_manager.handle_error(
                RecoveryErrorType.ORDER_REJECTION,
                {"symbol": symbol, "side": side, "quantity": quantity, "stop_price": stop_price}
            )
            return None
    except Exception as e:
        logger.error(f"Error placing stop order: {e}")
        action = recovery_manager.handle_error(
            RecoveryErrorType.SDK_CRASH,
            {"error": str(e), "function": "place_stop_order"}
        )
        return None


def place_limit_order(symbol: str, side: str, quantity: int, limit_price: float) -> Optional[Dict[str, Any]]:
    """
    Place a limit order through the broker interface.
    Phase Seven: Used for aggressive flatten orders to avoid market order slippage.
    
    Args:
        symbol: Instrument symbol
        side: 'BUY' or 'SELL'
        quantity: Number of contracts
        limit_price: Limit price
    
    Returns:
        Order object or None if failed
    """
    # Backtest mode: Simulate order without broker
    if is_backtest_mode():
        pass  # Silent - backtest limit order simulated
        return {
            "order_id": f"BACKTEST_LIMIT_{datetime.now().timestamp()}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": "LIMIT",
            "limit_price": limit_price,
            "status": "PENDING",
            "backtest": True
        }
    
    shadow_mode = _bot_config.shadow_mode
    pass  # Silent - limit order logged at higher level
    
    if shadow_mode:
        return {
            "order_id": f"SHADOW_LIMIT_{datetime.now().timestamp()}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": "LIMIT",
            "limit_price": limit_price,
            "status": "PENDING",
            "dry_run": True
        }
    
    if broker is None:
        logger.error("Broker not initialized")
        return None
    
    try:
        # Use circuit breaker for order placement
        breaker = recovery_manager.get_circuit_breaker("order_placement")
        success, order = breaker.call(broker.place_limit_order, symbol, side, quantity, limit_price)
        
        if success and order:
            return order
        else:
            logger.error("Limit order placement failed")
            action = recovery_manager.handle_error(
                RecoveryErrorType.ORDER_REJECTION,
                {"symbol": symbol, "side": side, "quantity": quantity, "limit_price": limit_price}
            )
            return None
    except Exception as e:
        logger.error(f"Error placing limit order: {e}")
        action = recovery_manager.handle_error(
            RecoveryErrorType.SDK_CRASH,
            {"error": str(e), "function": "place_limit_order"}
        )
        return None


def cancel_order(symbol: str, order_id: str) -> bool:
    """
    Cancel an open order through the broker interface.
    
    Args:
        symbol: Instrument symbol
        order_id: Order ID to cancel
    
    Returns:
        True if cancelled successfully, False otherwise
    """
    # Backtest mode: Simulate cancellation
    if is_backtest_mode():
        logger.info(f"[BACKTEST] Order {order_id} cancelled (simulated)")
        return True
    
    shadow_mode = _bot_config.shadow_mode
    pass  # Silent - order cancellation logged at higher level
    
    if shadow_mode:
        logger.info(f"[SHADOW MODE] Order {order_id} cancelled (simulated)")
        return True
    
    if broker is None:
        logger.error("Broker not initialized")
        return False
    
    try:
        # Use circuit breaker for order cancellation
        breaker = recovery_manager.get_circuit_breaker("order_placement")
        success, result = breaker.call(broker.cancel_order, order_id)
        
        if success and result:
            logger.info(f"Order {order_id} cancelled successfully")
            return True
        else:
            logger.error(f"Failed to cancel order {order_id}")
            return False
    except AttributeError as e:
        # Handle 'NoneType' object has no attribute 'send' - asyncio shutdown issue
        # This is expected during bot shutdown and not a real error
        if "'NoneType' object has no attribute 'send'" in str(e):
            logger.debug(f"Cancel order skipped - asyncio shutdown in progress (order {order_id})")
            return False
        logger.error(f"Error cancelling order {order_id}: {e}")
        return False
    except Exception as e:
        # Suppress event loop closed errors during shutdown
        if "Event loop is closed" in str(e):
            logger.debug(f"Event loop closed during cancel order {order_id}")
            return False
        logger.error(f"Error cancelling order {order_id}: {e}")
        return False


def get_position_quantity(symbol: str) -> int:
    """
    Query broker for current position quantity.
    Phase Eight: Used to check for partial fills.
    
    Args:
        symbol: Instrument symbol
    
    Returns:
        Current position quantity (positive for long, negative for short, 0 for flat)
    """
    # Backtest mode uses tracked position
    if is_backtest_mode():
        if state.get(symbol) and state[symbol]["position"]["active"]:
            qty = state[symbol]["position"]["quantity"]
            side = state[symbol]["position"]["side"]
            return qty if side == "long" else -qty
        return 0
    
    # Shadow mode uses tracked position
    if _bot_config.shadow_mode:
        if state.get(symbol) and state[symbol]["position"]["active"]:
            qty = state[symbol]["position"]["quantity"]
            side = state[symbol]["position"]["side"]
            return qty if side == "long" else -qty
        return 0
    
    if broker is None:
        logger.error("Broker not initialized")
        return 0
    
    try:
        # Use circuit breaker for position query
        breaker = recovery_manager.get_circuit_breaker("account_query")
        success, quantity = breaker.call(broker.get_position_quantity, symbol)
        
        if success:
            # Check for position discrepancy
            if state.get(symbol) and state[symbol]["position"]["active"]:
                expected_qty = state[symbol]["position"]["quantity"]
                expected_side = state[symbol]["position"]["side"]
                expected = expected_qty if expected_side == "long" else -expected_qty
                
                if quantity != expected:
                    logger.warning(f"Position discrepancy: Expected {expected}, got {quantity}")
                    action = recovery_manager.handle_error(
                        RecoveryErrorType.POSITION_DISCREPANCY,
                        {"symbol": symbol, "expected": expected, "actual": quantity}
                    )
            
            return quantity
        else:
            logger.error("Failed to get position quantity")
            return 0
    except Exception as e:
        logger.error(f"Error querying position: {e}")
        action = recovery_manager.handle_error(
            RecoveryErrorType.SDK_CRASH,
            {"error": str(e), "function": "get_position_quantity"}
        )
        return 0


def get_all_open_positions() -> List[Dict[str, Any]]:
    """
    Get all open positions from broker.
    
    Used by AI Mode to detect positions on the configured symbol.
    Returns all positions from broker, which are then filtered
    by the caller to only manage the configured symbol.
    
    Returns:
        List of position dicts with keys: symbol, quantity, side
        Empty list if no positions or broker not available
    """
    if broker is None:
        return []
    
    try:
        if hasattr(broker, 'get_all_open_positions'):
            return broker.get_all_open_positions()
        return []
    except Exception as e:
        logger.debug(f"Error getting all positions: {e}")
        return []


def subscribe_market_data(symbol: str, callback: Callable[[str, float, int, int], None]) -> None:
    """
    Subscribe to real-time market data for a symbol through broker interface.
    
    Args:
        symbol: Instrument symbol
        callback: Function to call with tick data (symbol, price, volume, timestamp)
    """
    pass  # Silent - market data subscription internal
    
    if broker is None:
        logger.error("Broker not initialized")
        return
    
    try:
        # Subscribe through broker interface
        broker.subscribe_market_data(symbol, callback)
        pass  # Silent - subscription successful
    except Exception as e:
        logger.error(f"Error subscribing to market data: {e}")
        action = recovery_manager.handle_error(
            RecoveryErrorType.DATA_FEED_INTERRUPTION,
            {"symbol": symbol, "error": str(e)}
        )


def resubscribe_market_data_after_reconnect() -> bool:
    """
    Re-subscribe to market data after broker reconnection.
    
    This is called after the broker reconnects from maintenance mode or
    after a connection loss during trading hours. It re-establishes the
    market data and quote subscriptions that were lost when the websocket
    was disconnected.
    
    Returns:
        True if all subscriptions successful, False if any failed
    """
    trading_symbol = get_current_symbol_for_session()
    success = True
    
    # Re-subscribe to market data (tick/trade data)
    try:
        subscribe_market_data(trading_symbol, on_tick)
        logger.critical(f"  [RECONNECT] ✅ Market data subscription active for {trading_symbol}")
    except Exception as e:
        logger.error(f"  [RECONNECT] ❌ Failed to subscribe to market data: {e}")
        success = False
    
    # Re-subscribe to quotes (bid/ask data)
    try:
        if broker is not None and hasattr(broker, 'subscribe_quotes'):
            broker.subscribe_quotes(trading_symbol, on_quote)
            logger.critical("  [RECONNECT] ✅ Quote data subscription active")
    except Exception as e:
        logger.error(f"  [RECONNECT] ❌ Failed to subscribe to quote data: {e}")
        success = False
    
    return success


def fetch_historical_bars(symbol: str, timeframe: int, count: int) -> List[Dict[str, Any]]:
    """
    Fetch historical bars for initial trend calculation through broker interface.
    
    Args:
        symbol: Instrument symbol
        timeframe: Bar timeframe in minutes
        count: Number of bars to fetch
    
    Returns:
        List of bar dictionaries with OHLCV data
    """
    pass  # Silent - historical data fetch internal
    
    
    if broker is None:
        logger.error("Broker not initialized")
        return []
    
    try:
        # Fetch through broker interface
        breaker = recovery_manager.get_circuit_breaker("market_data")
        success, bars = breaker.call(broker.fetch_historical_bars, symbol, f"{timeframe}m", count)
        
        if success and bars:
            pass  # Silent - bars fetched
            return bars
        else:
            logger.warning("Failed to fetch historical bars")
            return []
    except Exception as e:
        logger.error(f"Error fetching historical bars: {e}")
        action = recovery_manager.handle_error(
            RecoveryErrorType.SDK_CRASH,
            {"error": str(e), "function": "fetch_historical_bars"}
        )
        return []


# ============================================================================
# PHASE THREE: State Management
# ============================================================================

def initialize_state(symbol: str) -> None:
    """
    Initialize state tracking for an instrument.
    
    Args:
        symbol: Instrument symbol
    """
    # CRITICAL FIX: Reload config to get latest values (fixes subprocess caching issue)
    global _bot_config, CONFIG
    _is_backtest = is_backtest_mode()
    _bot_config = load_config(backtest_mode=_is_backtest)
    # Only validate if not in backtest mode
    if not _is_backtest:
        _bot_config.validate()
    CONFIG = _bot_config.to_dict()
    
    state[symbol] = {
        # Tick data storage
        "ticks": deque(maxlen=CONFIG.get("max_tick_storage", 10000)),
        
        # Bar storage
        "bars_1min": deque(maxlen=CONFIG.get("max_bars_storage", 200)),
        "bars_15min": deque(maxlen=100),
        
        # Current incomplete bars
        "current_1min_bar": None,
        "current_15min_bar": None,
        
        # VWAP calculation data
        "vwap": None,
        "vwap_bands": {
            "upper_1": None,
            "upper_2": None,
            "upper_3": None,
            "lower_1": None,
            "lower_2": None,
            "lower_3": None
        },
        "vwap_std_dev": None,
        "vwap_day": None,  # Phase Three: Track VWAP day separately
        
        # Trend filter
        "trend_ema": None,
        "trend_direction": None,  # 'up', 'down', or 'neutral'
        
        # Technical indicators
        "rsi": None,  # RSI value (0-100)
        "macd": None,  # MACD data dict with 'macd', 'signal', 'histogram'
        "avg_volume": None,  # Average volume for spike detection
        "recent_volume_history": deque(maxlen=20),  # Last 20 bars for volume surge detection
        
        # Signal tracking
        "last_signal": None,
        "signal_bar_price": None,  # Track price of bar that generated signal
        
        # Daily tracking
        "trading_day": None,
        "daily_trade_count": 0,
        "daily_pnl": 0.0,
        "warmup_complete": False,  # Track when 114 bars collected for regime detection
        
        # Session tracking (Phase 13)
        "session_stats": {
            "trades": [],
            "win_count": 0,
            "loss_count": 0,
            "total_pnl": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "pnl_variance": 0.0,
            # Phase 20: Position duration statistics
            "trade_durations": [],  # List of durations in minutes
            "force_flattened_count": 0,  # Trades closed due to time limit
            "after_noon_entries": 0,  # Entries after 12 PM
            "after_noon_force_flattened": 0  # After-noon entries force-closed
        },
        
        # Position tracking
        "position": {
            "active": False,
            "side": None,
            "quantity": 0,
            "entry_price": None,
            "stop_price": None,
            "entry_time": None,
            # Regime Information - For dynamic exit management
            "entry_regime": None,  # Regime at entry
            "current_regime": None,  # Current regime (updated on each tick)
            "regime_change_time": None,  # When regime last changed
            "regime_history": [],  # List of regime transitions with timestamps
            "entry_atr": None,  # ATR at entry
            # Advanced Exit Management - Breakeven State
            "breakeven_active": False,
            "original_stop_price": None,
            "breakeven_activated_time": None,
            # Advanced Exit Management - Trailing Stop State
            "trailing_stop_active": False,
            "trailing_stop_price": None,
            "highest_price_reached": None,  # For longs
            "lowest_price_reached": None,  # For shorts
            "trailing_activated_time": None,
            # Advanced Exit Management - Time-Decay State
            "time_decay_50_triggered": False,
            "time_decay_75_triggered": False,
            "time_decay_90_triggered": False,
            "original_stop_distance_ticks": None,
            "current_stop_distance_ticks": None,
            # Advanced Exit Management - Partial Exit State
            "partial_exit_1_completed": False,
            "partial_exit_2_completed": False,
            "partial_exit_3_completed": False,
            "original_quantity": 0,
            "remaining_quantity": 0,
            "partial_exit_history": [],  # List of {"price": float, "quantity": int, "r_multiple": float}
            # Advanced Exit Management - General
            "initial_risk_ticks": None,
        },
        
        # Volume history
        "volume_history": deque(maxlen=CONFIG["max_bars_storage"])
    }
    
    pass  # Silent - state initialization internal


# ============================================================================
# PHASE FOUR: Data Processing Pipeline
# ============================================================================

def on_quote(symbol: str, bid_price: float, ask_price: float, bid_size: int, 
             ask_size: int, last_price: float, timestamp_ms: int) -> None:
    """
    Handle incoming bid/ask quote data.
    Updates bid/ask manager with real-time quote information.
    
    Args:
        symbol: Instrument symbol
        bid_price: Current bid price
        ask_price: Current ask price
        bid_size: Bid size (contracts)
        ask_size: Ask size (contracts)
        last_price: Last trade price
        timestamp_ms: Quote timestamp in milliseconds
    """
    if bid_ask_manager is not None:
        bid_ask_manager.update_quote(
            symbol=symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            last_price=last_price,
            timestamp=timestamp_ms
        )
    
    # Also process as tick data to build bars
    # Use last_price and estimated volume of 1 (quote updates don't have volume)
    on_tick(symbol, last_price, 1, timestamp_ms)


# ============================================================================
# PHASE FOUR: Position State Persistence (NEVER FORGET!)
# ============================================================================

def save_position_state(symbol: str) -> None:
    """
    CRITICAL: Save position state to disk immediately.
    This ensures the bot NEVER forgets what position it's in, even after:
    - Crashes
    - Restarts
    - Network failures
    - Any errors
    
    EXE-COMPATIBLE: Uses get_data_file_path() to work in both:
    - Development (Python script)
    - Production (PyInstaller EXE on customer machines)
    
    Args:
        symbol: Instrument symbol
    """
    try:
        import json
        
        # EXE-COMPATIBLE: Get proper path whether script or frozen EXE
        # Add account ID for multi-user support
        account_id = os.getenv('SELECTED_ACCOUNT_ID', 'default')
        state_file = get_data_file_path(f"data/bot_state_{account_id}.json")
        
        # Extract critical position info
        position = state[symbol]["position"]
        
        # Convert datetime objects to strings for JSON serialization
        position_state = {
            "symbol": symbol,
            "active": position["active"],
            "side": position["side"],
            "quantity": position["quantity"],
            "entry_price": position["entry_price"],
            "stop_price": position["stop_price"],
            "entry_time": position["entry_time"].isoformat() if position.get("entry_time") else None,
            "order_id": position.get("order_id"),
            "stop_order_id": position.get("stop_order_id"),
            "last_updated": datetime.now().isoformat(),
        }
        
        # Write to file with backup
        backup_file = get_data_file_path(f"data/bot_state_{account_id}.json.backup")
        if state_file.exists():
            # Delete existing backup if it exists (Windows workaround)
            if backup_file.exists():
                backup_file.unlink()
            # Create backup of previous state
            state_file.rename(backup_file)
        
        with open(state_file, 'w') as f:
            json.dump(position_state, f, indent=2)
        
        # Safe logging with None checks
        if position.get('entry_price') is not None:
            pass  # Silent - position state saved internally
        else:
            pass  # Silent - position state saved
        
    except Exception as e:
        logger.error(f"CRITICAL: Failed to save position state: {e}", exc_info=True)


def load_position_state(symbol: str) -> bool:
    """
    Load position state from disk on startup.
    Returns True if a position was restored, False otherwise.
    
    EXE-COMPATIBLE: Uses get_data_file_path() to work in both:
    - Development (Python script)
    - Production (PyInstaller EXE on customer machines)
    
    Args:
        symbol: Instrument symbol
    
    Returns:
        True if position was restored from disk
    """
    try:
        import json
        
        # EXE-COMPATIBLE: Get proper path whether script or frozen EXE
        # Add account ID for multi-user support
        account_id = os.getenv('SELECTED_ACCOUNT_ID', 'default')
        state_file = get_data_file_path(f"data/bot_state_{account_id}.json")
        if not state_file.exists():
            pass  # Silent - clean start
            return False
        
        with open(state_file, 'r') as f:
            saved_state = json.load(f)
        
        # Check if saved state is for this symbol and has an active position
        if saved_state.get("symbol") != symbol:
            pass  # Silent - different symbol
            return False
        
        if not saved_state.get("active"):
            pass  # Silent - no active position
            return False
        
        # CRITICAL: Verify with broker before restoring state
        logger.warning(SEPARATOR_LINE)
        logger.warning("RESTORING POSITION FROM SAVED STATE")
        logger.warning(f"  Saved: {saved_state['side']} {saved_state['quantity']} @ ${saved_state['entry_price']:.2f}")
        logger.warning("  Verifying with broker...")
        
        broker_position = get_position_quantity(symbol)
        expected = saved_state['quantity'] if saved_state['side'] == "long" else -saved_state['quantity']
        
        if broker_position != expected:
            logger.error(f"  MISMATCH: Broker={broker_position}, Saved={expected}")
            logger.error("  Cannot restore - position state is stale or incorrect")
            logger.warning(SEPARATOR_LINE)
            return False
        
        # Broker confirms - restore the position state
        logger.warning("  Γ£ô Broker confirms position - restoring state")
        
        # Restore position to state
        state[symbol]["position"]["active"] = True
        state[symbol]["position"]["side"] = saved_state["side"]
        state[symbol]["position"]["quantity"] = saved_state["quantity"]
        state[symbol]["position"]["entry_price"] = saved_state["entry_price"]
        state[symbol]["position"]["stop_price"] = saved_state["stop_price"]
        state[symbol]["position"]["order_id"] = saved_state.get("order_id")
        state[symbol]["position"]["stop_order_id"] = saved_state.get("stop_order_id")
        
        if saved_state.get("entry_time"):
            state[symbol]["position"]["entry_time"] = datetime.fromisoformat(saved_state["entry_time"])
        
        logger.warning(f"  Position restored successfully")
        logger.warning(SEPARATOR_LINE)
        return True
        
    except Exception as e:
        logger.error(f"Error loading position state: {e}", exc_info=True)
        return False


def on_tick(symbol: str, price: float, volume: int, timestamp_ms: int) -> None:
    """
    Handle incoming tick data by posting to event loop.
    
    Args:
        symbol: Instrument symbol
        price: Tick price
        volume: Tick volume
        timestamp_ms: Timestamp in milliseconds
    """
    # Post tick data to event loop for processing
    if event_loop:
        event_loop.post_event(
            EventType.TICK_DATA,
            EventPriority.MEDIUM,
            {
                "symbol": symbol,
                "price": price,
                "volume": volume,
                "timestamp": timestamp_ms
            }
        )
    else:
        # Fallback if event loop not initialized (backtesting mode)
        # Suppress warning spam during backtests - this is expected behavior
        handle_tick_event({
            "symbol": symbol,
            "price": price,
            "volume": volume,
            "timestamp": timestamp_ms
        })


def update_1min_bar(symbol: str, price: float, volume: int, dt: datetime) -> None:
    """
    Update or create 1-minute bars for VWAP calculation.
    
    Args:
        symbol: Instrument symbol
        price: Current price
        volume: Current volume
        dt: Current datetime
    """
    # Get current minute boundary
    minute_boundary = dt.replace(second=0, microsecond=0)
    
    current_bar = state[symbol]["current_1min_bar"]
    
    if current_bar is None or current_bar["timestamp"] != minute_boundary:
        # Finalize previous bar if exists
        if current_bar is not None:
            state[symbol]["bars_1min"].append(current_bar)
            bar_count = len(state[symbol]["bars_1min"])
            
            # Calculate VWAP after new bar is added
            calculate_vwap(symbol)
            
            # Classify market condition on every bar if bid_ask_manager available
            if bid_ask_manager is not None:
                try:
                    condition, _ = bid_ask_manager.classify_market_condition(symbol)
                    state[symbol]["market_condition"] = condition
                except Exception as e:
                    logger.debug(f"Could not classify market condition: {e}")
                    state[symbol]["market_condition"] = "UNKNOWN"
            
            # Display market snapshot on every 1-minute bar close
            # Only display if bot has been running for at least 1 minute to avoid confusion
            # with rapid bar creation during startup
            # SILENCE DURING MAINTENANCE - no spam in logs
            if bot_status.get("maintenance_idle", False):
                pass  # Silent during maintenance - no market updates
            elif bot_status.get("session_start_time"):
                time_since_start = (get_current_time() - bot_status["session_start_time"]).total_seconds()
                if time_since_start < 60:
                    # Skip display for first minute of runtime
                    pass
                else:
                    vwap_data = state[symbol].get("vwap", {})
                    market_cond = state[symbol].get("market_condition", "UNKNOWN")
                    current_regime = state[symbol].get("current_regime", "NORMAL")
                    
                    # Get current bid/ask from bid_ask_manager if available
                    quote_info = ""
                    if bid_ask_manager is not None:
                        quote = bid_ask_manager.get_current_quote(symbol)
                        if quote:
                            spread = quote.ask_price - quote.bid_price
                            quote_info = f" | Bid: ${quote.bid_price:.2f} x {quote.bid_size} | Ask: ${quote.ask_price:.2f} x {quote.ask_size} | Spread: ${spread:.2f}"
                    
                    # Get latest bar volume
                    vol_info = f" | Vol: {current_bar['volume']}"
                    
                    # Get VWAP if available
                    vwap_info = ""
                    if vwap_data and isinstance(vwap_data, dict):
                        vwap_val = vwap_data.get('vwap', 0)
                        std_dev = vwap_data.get('std_dev', 0)
                        if vwap_val > 0:
                            vwap_info = f" | VWAP: ${vwap_val:.2f} ± ${std_dev:.2f}"
                    
                    logger.info(f"📊 Market: {symbol} @ ${current_bar['close']:.2f}{quote_info}{vol_info} | Bars: {bar_count}{vwap_info} | Condition: {market_cond} | Regime: {current_regime}")
            else:
                # Fallback if session_start_time not set (shouldn't happen)
                vwap_data = state[symbol].get("vwap", {})
                market_cond = state[symbol].get("market_condition", "UNKNOWN")
                current_regime = state[symbol].get("current_regime", "NORMAL")
                
                # Get current bid/ask from bid_ask_manager if available
                quote_info = ""
                if bid_ask_manager is not None:
                    quote = bid_ask_manager.get_current_quote(symbol)
                    if quote:
                        spread = quote.ask_price - quote.bid_price
                        quote_info = f" | Bid: ${quote.bid_price:.2f} x {quote.bid_size} | Ask: ${quote.ask_price:.2f} x {quote.ask_size} | Spread: ${spread:.2f}"
                
                # Get latest bar volume
                vol_info = f" | Vol: {current_bar['volume']}"
                
                # Get VWAP if available
                vwap_info = ""
                if vwap_data and isinstance(vwap_data, dict):
                    vwap_val = vwap_data.get('vwap', 0)
                    std_dev = vwap_data.get('std_dev', 0)
                    if vwap_val > 0:
                        vwap_info = f" | VWAP: ${vwap_val:.2f} ± ${std_dev:.2f}"
                
                logger.info(f"📊 Market: {symbol} @ ${current_bar['close']:.2f}{quote_info}{vol_info} | Bars: {bar_count}{vwap_info} | Condition: {market_cond} | Regime: {current_regime}")
            
            # Update current regime after bar completion
            update_current_regime(symbol)
            
            # Check for exit conditions if position is active
            check_exit_conditions(symbol)
            # Check for entry signals if no position
            check_for_signals(symbol)
        
        # Start new bar
        state[symbol]["current_1min_bar"] = {
            "timestamp": minute_boundary,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": volume
        }
    else:
        # Update current bar
        current_bar["high"] = max(current_bar["high"], price)
        current_bar["low"] = min(current_bar["low"], price)
        current_bar["close"] = price
        current_bar["volume"] += volume
        
        # CRITICAL FOR LIVE TRADING: Check exits on EVERY TICK (intrabar)
        # Don't wait for bar close - exit immediately if stop/target hit
        if state[symbol]["position"]["active"]:
            check_exit_conditions(symbol)


def inject_complete_bar(symbol: str, bar: Dict[str, Any]) -> None:
    """
    Inject a complete OHLCV bar directly (historical data replay).
    This preserves accurate high/low ranges for ATR calculation.
    
    Args:
        symbol: Instrument symbol
        bar: Complete bar dict with timestamp, open, high, low, close, volume
    """
    global backtest_current_time
    
    # BACKTEST MODE: Update simulation time so all time-based logic uses historical time
    if is_backtest_mode() and 'timestamp' in bar:
        backtest_current_time = bar['timestamp']
    
    # First bar check
    if len(state[symbol]["bars_1min"]) == 0:
        pass  # Silent - bar injection internal (backtest mode)
    
    # Finalize any pending bar first
    if state[symbol]["current_1min_bar"] is not None:
        state[symbol]["bars_1min"].append(state[symbol]["current_1min_bar"])
        state[symbol]["current_1min_bar"] = None
    
    # Add the complete bar with proper OHLC
    state[symbol]["bars_1min"].append(bar)
    
    # Update current regime after adding new bar
    update_current_regime(symbol)
    
    # Update all indicators after each 1-minute bar (all on same timeframe)
    update_macd(symbol)
    update_rsi(symbol)
    update_volume_average(symbol)
    
    # Update VWAP and check conditions
    calculate_vwap(symbol)
    check_exit_conditions(symbol)
    check_for_signals(symbol)



def update_15min_bar(symbol: str, price: float, volume: int, dt: datetime) -> None:
    """
    Update or create 15-minute bars for trend filter.
    
    Args:
        symbol: Instrument symbol
        price: Current price
        volume: Current volume
        dt: Current datetime
    """
    # Get 15-minute boundary
    minute = (dt.minute // 15) * 15
    boundary_15min = dt.replace(minute=minute, second=0, microsecond=0)
    
    current_bar = state[symbol]["current_15min_bar"]
    
    if current_bar is None or current_bar["timestamp"] != boundary_15min:
        # Finalize previous bar if exists
        if current_bar is not None:
            state[symbol]["bars_15min"].append(current_bar)
            # Update trend filter only (RSI/MACD/Volume now on 1-min, updated after 1-min bars)
            update_trend_filter(symbol)
        
        # Start new bar
        state[symbol]["current_15min_bar"] = {
            "timestamp": boundary_15min,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": volume
        }
    else:
        # Update current bar
        current_bar["high"] = max(current_bar["high"], price)
        current_bar["low"] = min(current_bar["low"], price)
        current_bar["close"] = price
        current_bar["volume"] += volume


def update_trend_filter(symbol: str) -> None:
    """
    Update the trend filter using EMA of 15-minute bars.
    
    Args:
        symbol: Instrument symbol
    """
    bars = state[symbol]["bars_15min"]
    period = CONFIG.get("trend_ema_period", 20)
    
    if len(bars) < period:
        pass  # Silent - technical detail not for customers
        return
    
    # Calculate EMA
    closes = [bar["close"] for bar in bars]
    ema = calculate_ema(closes, period)
    
    if ema is not None:
        state[symbol]["trend_ema"] = ema
        
        # Determine trend direction with neutral zone (half tick)
        current_price = closes[-1]
        half_tick = CONFIG["tick_size"] / 2.0
        
        if current_price > ema + half_tick:
            state[symbol]["trend_direction"] = "up"
        elif current_price < ema - half_tick:
            state[symbol]["trend_direction"] = "down"
        else:
            state[symbol]["trend_direction"] = "neutral"
        
        pass  # Silent - trend calculation internal


def calculate_ema(values: List[float], period: int) -> Optional[float]:
    """
    Calculate Exponential Moving Average.
    
    Args:
        values: List of values
        period: EMA period
    
    Returns:
        EMA value or None
    """
    if len(values) < period:
        return None
    
    multiplier = 2.0 / (period + 1)
    
    # Start with SMA
    ema = sum(values[:period]) / period
    
    # Calculate EMA for remaining values
    for value in values[period:]:
        ema = (value - ema) * multiplier + ema
    
    return ema


def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """
    Calculate Relative Strength Index.
    
    Args:
        prices: List of closing prices
        period: RSI period (default 14)
    
    Returns:
        RSI value (0-100) or None if insufficient data
    """
    if len(prices) < period + 1:
        return None
    
    # Calculate price changes
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    
    # Separate gains and losses
    gains = [change if change > 0 else 0 for change in changes]
    losses = [-change if change < 0 else 0 for change in changes]
    
    # Calculate initial average gain and loss (SMA)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    # Calculate smoothed averages (EMA style)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    # Calculate RSI
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def calculate_macd(prices: List[float], fast_period: int = 12, 
                   slow_period: int = 26, signal_period: int = 9) -> Optional[Dict[str, float]]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: List of closing prices
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)
    
    Returns:
        Dictionary with 'macd', 'signal', 'histogram' or None if insufficient data
    """
    if len(prices) < slow_period + signal_period:
        return None
    
    # Calculate fast and slow EMAs
    fast_ema = calculate_ema(prices, fast_period)
    slow_ema = calculate_ema(prices, slow_period)
    
    if fast_ema is None or slow_ema is None:
        return None
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate MACD values for signal line
    # We need to calculate MACD for each point to get signal line
    macd_values = []
    for i in range(slow_period, len(prices) + 1):
        fast = calculate_ema(prices[:i], fast_period)
        slow = calculate_ema(prices[:i], slow_period)
        if fast is not None and slow is not None:
            macd_values.append(fast - slow)
    
    if len(macd_values) < signal_period:
        return None
    
    # Calculate signal line (EMA of MACD)
    signal_line = calculate_ema(macd_values, signal_period)
    
    if signal_line is None:
        return None
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram
    }


def calculate_atr(symbol: str, period: int = 14) -> Optional[float]:
    """
    Calculate Average True Range (ATR) for volatility measurement using 15-minute bars.
    
    Args:
        symbol: Instrument symbol
        period: ATR period (default 14)
    
    Returns:
        ATR value in price units, or None if not enough data
    """
    bars = state[symbol]["bars_15min"]
    
    if len(bars) < 2:
        return None
    
    true_ranges = []
    for i in range(1, len(bars)):
        high = bars[i]["high"]
        low = bars[i]["low"]
        prev_close = bars[i-1]["close"]
        
        # True Range is the maximum of:
        # 1. Current High - Current Low
        # 2. abs(Current High - Previous Close)
        # 3. abs(Current Low - Previous Close)
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)
    
    if not true_ranges:
        return None
    
    # Calculate ATR (simple moving average of TR)
    if len(true_ranges) >= period:
        atr = sum(true_ranges[-period:]) / period
    else:
        atr = sum(true_ranges) / len(true_ranges)
    
    return atr


def calculate_atr_1min(symbol: str, period: int = 14) -> Optional[float]:
    """
    Calculate Average True Range (ATR) using 1-minute bars for regime detection.
    
    This function uses 1-minute bars to provide higher-resolution volatility data
    for accurate regime detection. The regime detector needs ATR calculated from
    the same timeframe as the bars it analyzes (1-minute bars).
    
    Args:
        symbol: Instrument symbol
        period: ATR period (default 14)
    
    Returns:
        ATR value in price units, or None if not enough data
    """
    bars = state[symbol]["bars_1min"]
    
    if len(bars) < 2:
        return None
    
    true_ranges = []
    for i in range(1, len(bars)):
        high = bars[i]["high"]
        low = bars[i]["low"]
        prev_close = bars[i-1]["close"]
        
        # True Range is the maximum of:
        # 1. Current High - Current Low
        # 2. abs(Current High - Previous Close)
        # 3. abs(Current Low - Previous Close)
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)
    
    if not true_ranges:
        return None
    
    # Calculate ATR (simple moving average of TR)
    if len(true_ranges) >= period:
        atr = sum(true_ranges[-period:]) / period
    else:
        atr = sum(true_ranges) / len(true_ranges)
    
    return atr


def update_rsi(symbol: str) -> None:
    """
    Update RSI indicator for the symbol using 1-minute bars.
    Changed from 15-min to 1-min for mean reversion scalping strategy.
    
    Args:
        symbol: Instrument symbol
    """
    bars = state[symbol]["bars_1min"]  # Changed from 15-min to 1-min
    rsi_period = CONFIG.get("rsi_period", 10)  # Iteration 3 - fast RSI
    
    if len(bars) < rsi_period + 1:
        pass  # Silent - RSI calculation internal
        return
    
    # Use recent bars for calculation
    closes = [bar["close"] for bar in list(bars)[-100:]]
    rsi = calculate_rsi(closes, rsi_period)
    
    if rsi is not None:
        state[symbol]["rsi"] = rsi
        pass  # Silent - RSI value internal


def update_macd(symbol: str) -> None:
    """
    Update MACD indicator for the symbol using 1-minute bars for faster initialization.
    
    Args:
        symbol: Instrument symbol
    """
    # Use 1-minute bars for faster MACD calculation (faster initialization)
    bars = state[symbol]["bars_1min"]
    
    # Get MACD parameters from config
    fast_period = CONFIG.get("macd_fast", 12)
    slow_period = CONFIG.get("macd_slow", 26)
    signal_period = CONFIG.get("macd_signal", 9)
    
    if len(bars) < slow_period + signal_period:
        pass  # Silent - MACD calculation internal
        return
    
    # Use recent bars for calculation
    closes = [bar["close"] for bar in list(bars)[-100:]]  # Use last 100 bars
    macd_data = calculate_macd(closes, fast_period, slow_period, signal_period)
    
    if macd_data is not None:
        state[symbol]["macd"] = macd_data
        pass  # Silent - MACD value internal


def update_volume_average(symbol: str) -> None:
    """
    Update average volume for spike detection using 1-minute bars.
    Changed from 15-min to 1-min for mean reversion scalping strategy.
    
    Args:
        symbol: Instrument symbol
    """
    bars = state[symbol]["bars_1min"]  # Changed from 15-min to 1-min
    lookback = CONFIG.get("volume_lookback", 20)
    
    if len(bars) < lookback:
        pass  # Silent - volume calculation internal
        return
    
    # Calculate average volume over lookback period
    recent_bars = list(bars)[-lookback:]
    volumes = [bar["volume"] for bar in recent_bars]
    avg_volume = sum(volumes) / len(volumes)
    
    state[symbol]["avg_volume"] = avg_volume
    
    # Track recent volume history for surge detection (last 20 1-min bars)
    bars_1min = state[symbol]["bars_1min"]
    if len(bars_1min) > 0:
        current_volume = bars_1min[-1]["volume"]
        state[symbol]["recent_volume_history"].append(current_volume)
    
    pass  # Silent - volume average internal


def detect_volume_surge(symbol: str) -> Tuple[bool, float]:
    """
    Detect if volume is surging (potential reversal signal).
    Uses recent volume history to identify sudden spikes.
    
    Args:
        symbol: Instrument symbol
    
    Returns:
        Tuple of (is_surging, surge_ratio)
        - is_surging: True if volume surge detected
        - surge_ratio: Current volume / average recent volume
    """
    recent_volumes = state[symbol]["recent_volume_history"]
    
    if len(recent_volumes) < 10:
        return False, 1.0  # Not enough data
    
    # Calculate average of recent volumes (excluding current bar)
    recent_avg = statistics.mean(list(recent_volumes)[:-1]) if len(recent_volumes) > 1 else recent_volumes[0]
    current_volume = recent_volumes[-1]
    
    if recent_avg == 0:
        return False, 1.0
    
    surge_ratio = current_volume / recent_avg
    
    # Consider it a surge if current volume is 2.5x recent average
    surge_threshold = CONFIG.get("volume_surge_threshold", 2.5)
    
    if surge_ratio >= surge_threshold:
        return True, surge_ratio
    
    return False, surge_ratio


def check_market_divergence(symbol: str, position_side: str) -> Tuple[bool, str]:
    """
    Check if current position is diverging from recent price momentum.
    Uses recent bar direction as proxy for market trend.
    
    Args:
        symbol: Current symbol being traded
        position_side: Current position side ('long' or 'short')
    
    Returns:
        Tuple of (is_diverging, reason)
        - is_diverging: True if position fighting against momentum
        - reason: Explanation of divergence
    """
    if not CONFIG.get("market_correlation_enabled", False):
        return False, "Market correlation disabled in config"
    
    bars = state[symbol]["bars_1min"]
    if len(bars) < 5:
        return False, "Not enough bars for momentum check"
    
    # Check last 5 bars for momentum direction
    recent_bars = list(bars)[-5:]
    price_changes = []
    
    for i in range(1, len(recent_bars)):
        change = recent_bars[i]["close"] - recent_bars[i-1]["close"]
        price_changes.append(change)
    
    # Count up/down moves
    up_moves = sum(1 for change in price_changes if change > 0)
    down_moves = sum(1 for change in price_changes if change < 0)
    
    # Strong upward momentum = 4+ up moves
    # Strong downward momentum = 4+ down moves
    has_strong_up_momentum = up_moves >= 4
    has_strong_down_momentum = down_moves >= 4
    
    # Check for divergence
    if position_side == "long" and has_strong_down_momentum:
        total_change = sum(price_changes)
        return True, f"Long position fighting downward momentum ({down_moves}/4 down bars, {total_change:.2f} total change)"
    
    if position_side == "short" and has_strong_up_momentum:
        total_change = sum(price_changes)
        return True, f"Short position fighting upward momentum ({up_moves}/4 up bars, {total_change:.2f} total change)"
    
    return False, "Position aligned with momentum"


# ============================================================================
# PHASE FIVE: VWAP Calculation
# ============================================================================

def calculate_vwap(symbol: str) -> None:
    """
    Calculate VWAP and standard deviation bands from 1-minute bars.
    VWAP is calculated per bar without logging.
    
    Args:
        symbol: Instrument symbol
    """
    bars = state[symbol]["bars_1min"]
    
    if len(bars) == 0:
        return
    
    # Calculate cumulative VWAP
    total_pv = 0.0  # price * volume
    total_volume = 0.0
    
    for bar in bars:
        typical_price = (bar["high"] + bar["low"] + bar["close"]) / 3.0
        pv = typical_price * bar["volume"]
        total_pv += pv
        total_volume += bar["volume"]
    
    if total_volume == 0:
        return
    
    # VWAP = sum(price * volume) / sum(volume)
    vwap = total_pv / total_volume
    state[symbol]["vwap"] = vwap
    
    # Calculate standard deviation (volume-weighted)
    variance_sum = 0.0
    for bar in bars:
        typical_price = (bar["high"] + bar["low"] + bar["close"]) / 3.0
        squared_diff = (typical_price - vwap) ** 2
        variance_sum += squared_diff * bar["volume"]
    
    variance = variance_sum / total_volume
    std_dev = variance ** 0.5
    state[symbol]["vwap_std_dev"] = std_dev
    
    # Calculate bands using ITERATION 3 standard deviation multipliers
    band_1_mult = CONFIG.get("vwap_std_dev_1", 2.5)  # Iteration 3
    band_2_mult = CONFIG.get("vwap_std_dev_2", 2.1)  # Iteration 3 - Entry zone
    band_3_mult = CONFIG.get("vwap_std_dev_3", 3.7)  # Iteration 3 - Exit/stop zone
    state[symbol]["vwap_bands"]["upper_1"] = vwap + (std_dev * band_1_mult)
    state[symbol]["vwap_bands"]["upper_2"] = vwap + (std_dev * band_2_mult)
    state[symbol]["vwap_bands"]["upper_3"] = vwap + (std_dev * band_3_mult)
    state[symbol]["vwap_bands"]["lower_1"] = vwap - (std_dev * band_1_mult)
    state[symbol]["vwap_bands"]["lower_2"] = vwap - (std_dev * band_2_mult)
    state[symbol]["vwap_bands"]["lower_3"] = vwap - (std_dev * band_3_mult)


# ============================================================================
# PHASE SEVEN: Signal Generation Logic
# ============================================================================

def validate_signal_requirements(symbol: str, bar_time: datetime) -> Tuple[bool, Optional[str]]:
    """
    Validate that all requirements are met for signal generation.
    24/5 trading - signals allowed anytime except maintenance/weekend/economic events.
    
    Args:
        symbol: Instrument symbol
        bar_time: Current bar timestamp
    
    Returns:
        Tuple of (is_valid, reason)
    """
    # Check trading state - block signals when market closed or in flatten mode
    # USE CURRENT TIME, NOT BAR TIME (bar timestamps can be delayed in live feeds)
    current_time = get_current_time()
    trading_state = get_trading_state(current_time)
    
    if trading_state == "closed":
        logger.debug(f"Market closed, skipping signal check")
        return False, f"Market closed"
    
    # Trading state is "entry_window" - market is open, proceed with checks
    
    # THANKSGIVING SPECIAL: No new entries after 12:00 PM on last Thursday of November
    eastern_tz = pytz.timezone('US/Eastern')
    current_time_et = current_time.astimezone(eastern_tz)
    if current_time_et.weekday() == 3 and current_time_et.month == 11:  # Thursday in November
        # Check if this is the last Thursday of November
        if current_time_et.month == 12:
            next_month = current_time_et.replace(year=current_time_et.year + 1, month=1, day=1)
        else:
            next_month = current_time_et.replace(month=current_time_et.month + 1, day=1)
        last_day = (next_month - timedelta(days=1)).day
        
        # Find last Thursday
        for day in range(last_day, 0, -1):
            check_date = current_time_et.replace(day=day)
            if check_date.weekday() == 3:  # Thursday
                last_thursday = day
                break
        
        # If today is Thanksgiving, no new entries after 12:00 PM (flatten at 12:45 PM)
        if current_time_et.day == last_thursday and current_time_et.time() >= datetime_time(12, 0):
            log_time_based_action(
                "thanksgiving_entry_blocked",
                f"Thanksgiving Day - no new entries after 12:00 PM (market closes 1:00 PM)",
                {"time": current_time_et.strftime('%H:%M:%S')}
            )
            logger.debug(f"Thanksgiving Day - no new entries after 12:00 PM (flatten at 12:45 PM, market closes at 1:00 PM)")
            return False, "Thanksgiving entry cutoff (12:00 PM ET)"
    
    # Daily entry cutoff - no new positions after 4:00 PM ET (can hold until 4:45 PM flatten)
    # CRITICAL: This only applies BEFORE the market reopens at 6:00 PM
    # Market schedule: 6:00 PM (today) → 4:00 PM (next day) with maintenance 5:00-6:00 PM
    # Entry window: 6:00 PM → 4:00 PM next day (no new entries in the 4:00-6:00 PM window)
    current_time_only = current_time.time()
    if datetime_time(16, 0) <= current_time_only < datetime_time(18, 0):  # 4:00 PM - 6:00 PM ET
        log_time_based_action(
            "daily_entry_blocked",
            f"Between 4:00-6:00 PM ET, no new trades (flatten/maintenance window)",
            {"time": current_time.strftime('%H:%M:%S')}
        )
        logger.debug(f"4:00-6:00 PM ET window - no new entries (flatten at 4:45 PM, maintenance starts at 4:45 PM, reopens at 6:00 PM)")
        return False, "Daily entry cutoff (4:00-6:00 PM ET)"
    
    # Check if already have position
    if state[symbol]["position"]["active"]:
        logger.debug("Position already active, skipping signal generation")
        return False, "Position active"
    
    # Check daily trade limit (skip in backtest mode)
    if not is_backtest_mode() and state[symbol]["daily_trade_count"] >= CONFIG["max_trades_per_day"]:
        logger.debug(f"Daily trade limit reached ({CONFIG['max_trades_per_day']}), stopping for the day")
        
        # Send max trades reached alert (only once)
        if state[symbol]["daily_trade_count"] == CONFIG["max_trades_per_day"]:
            try:
                notifier = get_notifier()
                notifier.send_error_alert(
                    error_message=f"Max trades per day reached! Count: {state[symbol]['daily_trade_count']} / Limit: {CONFIG['max_trades_per_day']}. No more trades today.",
                    error_type="Max Trades Reached"
                )
            except Exception as e:
                logger.debug(f"Failed to send max trades alert: {e}")
        
        return False, "Daily trade limit"
    
    # Daily loss limit - Adjusted by current profit
    # If user sets daily limit to $X and trader makes $Y profit, effective limit becomes $X + $Y
    # This means they can lose up to $X + $Y before hitting the limit
    current_pnl = state[symbol]["daily_pnl"]
    base_loss_limit = CONFIG["daily_loss_limit"]
    
    # Adjust loss limit by adding current profit (if positive)
    # Profit acts as a buffer - the more profit made, the more can be lost before limit is hit
    effective_loss_limit = base_loss_limit + max(0, current_pnl)
    
    if state[symbol]["daily_pnl"] <= -effective_loss_limit:
        logger.warning(f"Daily loss limit hit (${state[symbol]['daily_pnl']:.2f}), stopping for the day")
        logger.warning(f"  Base limit: ${base_loss_limit:.2f}, Profit cushion: ${max(0, current_pnl):.2f}, Effective limit: ${effective_loss_limit:.2f}")
        
        # Send alert once when limit hit
        if not state[symbol].get("loss_limit_alerted", False):
            try:
                notifier = get_notifier()
                notifier.send_error_alert(
                    error_message=f"Daily Loss Limit Reached: ${state[symbol]['daily_pnl']:.2f} / -${effective_loss_limit:.2f}. Bot stopped trading for today. Will auto-resume tomorrow.",
                    error_type="Daily Loss Limit"
                )
                state[symbol]["loss_limit_alerted"] = True
            except Exception as e:
                logger.debug(f"Failed to send loss limit alert: {e}")
        
        return False, "Daily loss limit"
    
    # Check data availability
    if len(state[symbol]["bars_1min"]) < 2:
        pass  # Silent - signal check skipped (not enough bars)
        return False, "Insufficient bars"
    
    # ========================================================================
    # WARMUP PERIOD: Block signals until enough bars for regime detection
    # ========================================================================
    # Regime detection requires 114 bars for accurate classification
    # During warmup, we use NORMAL regime as fallback but should not trade
    WARMUP_BARS_REQUIRED = 114
    current_bar_count = len(state[symbol]["bars_1min"])
    
    if current_bar_count < WARMUP_BARS_REQUIRED:
        # Log warmup progress every 10 bars (clean professional logs)
        if current_bar_count % 10 == 0 or current_bar_count == 1:
            bars_remaining = WARMUP_BARS_REQUIRED - current_bar_count
            minutes_remaining = bars_remaining  # 1-min bars = 1 minute each
            progress_pct = (current_bar_count / WARMUP_BARS_REQUIRED) * 100
            
            # Create progress bar
            filled = int(progress_pct / 5)  # 20 chars total
            bar = "█" * filled + "░" * (20 - filled)
            
            logger.info(f"⏳ WARMUP [{bar}] {progress_pct:.0f}% | Bars: {current_bar_count}/{WARMUP_BARS_REQUIRED} | ~{minutes_remaining} min remaining")
            logger.info(f"   Collecting data for accurate regime detection. Signals blocked until warmup complete.")
        
        return False, f"Warmup ({current_bar_count}/{WARMUP_BARS_REQUIRED} bars)"
    
    # Log warmup completion once
    if not state[symbol].get("warmup_complete", False):
        state[symbol]["warmup_complete"] = True
        current_regime = state[symbol].get("current_regime", "NORMAL")
        logger.info("=" * 60)
        logger.info("✅ WARMUP COMPLETE - TRADING ENABLED")
        logger.info("=" * 60)
        logger.info(f"   📊 Bars collected: {current_bar_count}")
        logger.info(f"   🎯 Regime detection: ACTIVE")
        logger.info(f"   📈 Current regime: {current_regime}")
        logger.info(f"   🚀 Signal generation: ENABLED")
        logger.info("=" * 60)
    
    # Check VWAP is available (needed for target/safety net exit)
    # Note: VWAP bands are NOT used for entry signals in Capitulation Reversal strategy
    vwap = state[symbol].get("vwap")
    if vwap is None or vwap <= 0:
        pass  # Silent - VWAP not ready
        return False, "VWAP not ready"
    
    # Trend filter - DISABLED for Capitulation Reversal strategy
    # Flush direction determines trade direction (condition #8 in capitulation_detector.py)
    # - Long: price < VWAP (buying at discount after flush down)
    # - Short: price > VWAP (selling at premium after flush up)
    
    # Check RSI is available (needed for capitulation signal detection)
    # Note: RSI thresholds (25/75) are hardcoded in capitulation_detector.py
    rsi = state[symbol]["rsi"]
    if rsi is None:
        pass  # Silent - RSI not yet calculated, allow to proceed
        # Signal-specific functions will handle RSI check with 25/75 thresholds
    
    # Volume check - handled by capitulation_detector.py (condition #5: 2x average)
    avg_volume = state[symbol].get("avg_volume")
    if avg_volume is None:
        pass  # Silent - average volume not yet calculated, allow to proceed
        # Signal-specific functions will handle volume check with 2x threshold
    
    # VWAP direction filter - DISABLED for Capitulation Reversal strategy
    # Price vs VWAP check is done in capitulation_detector.py condition #8
    # (Long: price < VWAP, Short: price > VWAP)
    
    # Check bid/ask spread and market condition (Phase: Bid/Ask Strategy)
    if bid_ask_manager is not None:
        # Validate spread (Requirement 8)
        is_acceptable, spread_reason = bid_ask_manager.validate_entry_spread(symbol)
        if not is_acceptable:
            pass  # Silent - spread check (internal filter)
            return False, spread_reason
        
        # Classify market condition (Requirement 11)
        try:
            condition, condition_reason = bid_ask_manager.classify_market_condition(symbol)
            pass  # Silent - market condition check (internal filter)
            
            # Save market condition to state for monitoring display
            state[symbol]["market_condition"] = condition
            
            # Skip trading in stressed markets
            if condition == "stressed":
                logger.warning("Market is stressed - skipping trade")
                return False, "Stressed market conditions"
            
            # Warn about illiquid markets (position size already adjusted in execute_entry)
            if condition == "illiquid":
                logger.warning("Illiquid market detected - position size will be adjusted")
        except Exception as e:
            logger.warning(f"Could not classify market: {e}")
    
    return True, None


def check_long_signal_conditions(symbol: str, prev_bar: Dict[str, Any], 
                                 current_bar: Dict[str, Any]) -> bool:
    """
    Check if long signal conditions are met - CAPITULATION REVERSAL STRATEGY.
    
    ALL 9 CONDITIONS MUST BE TRUE:
    1. Flush Happened - Range of last 10 bars >= 20 ticks
    2. Flush Was Fast - Velocity >= 4 ticks per bar
    3. We Are Near The Bottom - Within 5 ticks of flush low
    4. RSI Is Extreme Oversold - RSI < 25
    5. Volume Spiked - Current volume >= 2x 20-bar average
    6. Flush Stopped Making New Lows - Current bar low >= previous bar low
    7. Reversal Candle - Current bar closes green (close > open)
    8. Price Is Below VWAP - Current close < VWAP
    9. Regime Allows Trading - HIGH_VOL_TRENDING or HIGH_VOL_CHOPPY
    
    Args:
        symbol: Instrument symbol
        prev_bar: Previous 1-minute bar
        current_bar: Current 1-minute bar
    
    Returns:
        True if ALL 9 conditions are met
    """
    vwap = state[symbol]["vwap"]
    bars = state[symbol]["bars_1min"]
    rsi = state[symbol]["rsi"]
    current_regime = state[symbol].get("current_regime", "NORMAL")
    
    # DIAGNOSTIC: Track how many times we reach this function
    state[symbol]["signal_check_attempts"] = state[symbol].get("signal_check_attempts", 0) + 1
    
    # CRITICAL: VWAP is required for capitulation strategy (it's the target)
    if vwap is None or vwap <= 0:
        logger.debug("Long rejected - VWAP not available (required for mean reversion target)")
        state[symbol]["vwap_missing_count"] = state[symbol].get("vwap_missing_count", 0) + 1
        return False
    
    # Need at least 10 bars for flush detection
    if len(bars) < 10:
        logger.debug("Long rejected - insufficient bars for flush detection")
        state[symbol]["insufficient_bars_count"] = state[symbol].get("insufficient_bars_count", 0) + 1
        return False
    
    # Calculate 20-bar average volume
    if len(bars) >= 20:
        recent_volumes = [bar.get("volume", 0) for bar in list(bars)[-20:]]
        avg_volume_20 = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 1
    else:
        avg_volume_20 = current_bar.get("volume", 1)
    
    # Get capitulation detector
    tick_size = CONFIG.get("tick_size", 0.25)
    tick_value = CONFIG.get("tick_value", 12.50)
    cap_detector = get_capitulation_detector(tick_size, tick_value)
    
    # Check ALL 9 conditions
    all_passed, details = cap_detector.check_all_long_conditions(
        bars=bars,
        current_bar=current_bar,
        prev_bar=prev_bar,
        rsi=rsi,
        avg_volume_20=avg_volume_20,
        current_price=current_bar["close"],
        vwap=vwap,
        regime=current_regime
    )
    
    if not all_passed:
        # Log periodically which conditions are failing (for debugging)
        if details.get("reason"):
            logger.debug(f"Long rejected: {details['reason']}")
        return False
    
    # Store entry details for position management
    state[symbol]["entry_details"] = details
    state[symbol]["flush_low"] = details.get("flush_low")
    state[symbol]["flush_high"] = details.get("flush_high")
    
    return True


def check_short_signal_conditions(symbol: str, prev_bar: Dict[str, Any], 
                                  current_bar: Dict[str, Any]) -> bool:
    """
    Check if short signal conditions are met - CAPITULATION REVERSAL STRATEGY.
    
    ALL 9 CONDITIONS MUST BE TRUE:
    1. Pump Happened - Range of last 10 bars >= 20 ticks
    2. Pump Was Fast - Velocity >= 4 ticks per bar
    3. We Are Near The Top - Within 5 ticks of flush high
    4. RSI Is Extreme Overbought - RSI > 75
    5. Volume Spiked - Current volume >= 2x 20-bar average
    6. Pump Stopped Making New Highs - Current bar high <= previous bar high
    7. Reversal Candle - Current bar closes red (close < open)
    8. Price Is Above VWAP - Current close > VWAP
    9. Regime Allows Trading - HIGH_VOL_TRENDING or HIGH_VOL_CHOPPY
    
    Args:
        symbol: Instrument symbol
        prev_bar: Previous 1-minute bar
        current_bar: Current 1-minute bar
    
    Returns:
        True if ALL 9 conditions are met
    """
    vwap = state[symbol]["vwap"]
    bars = state[symbol]["bars_1min"]
    rsi = state[symbol]["rsi"]
    current_regime = state[symbol].get("current_regime", "NORMAL")
    
    # CRITICAL: VWAP is required for capitulation strategy (it's the target)
    if vwap is None or vwap <= 0:
        logger.debug("Short rejected - VWAP not available (required for mean reversion target)")
        return False
    
    # Need at least 10 bars for flush detection
    if len(bars) < 10:
        logger.debug("Short rejected - insufficient bars for flush detection")
        return False
    
    # Calculate 20-bar average volume
    if len(bars) >= 20:
        recent_volumes = [bar.get("volume", 0) for bar in list(bars)[-20:]]
        avg_volume_20 = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 1
    else:
        avg_volume_20 = current_bar.get("volume", 1)
    
    # Get capitulation detector
    tick_size = CONFIG.get("tick_size", 0.25)
    tick_value = CONFIG.get("tick_value", 12.50)
    cap_detector = get_capitulation_detector(tick_size, tick_value)
    
    # Check ALL 9 conditions
    all_passed, details = cap_detector.check_all_short_conditions(
        bars=bars,
        current_bar=current_bar,
        prev_bar=prev_bar,
        rsi=rsi,
        avg_volume_20=avg_volume_20,
        current_price=current_bar["close"],
        vwap=vwap,
        regime=current_regime
    )
    
    if not all_passed:
        # Log periodically which conditions are failing (for debugging)
        if details.get("reason"):
            logger.debug(f"Short rejected: {details['reason']}")
        return False
    
    # Store entry details for position management
    state[symbol]["entry_details"] = details
    state[symbol]["flush_low"] = details.get("flush_low")
    state[symbol]["flush_high"] = details.get("flush_high")
    
    return True


def calculate_slope(values: List[float], periods: int = 5) -> float:
    """
    Calculate the slope (rate of change) of recent values.
    
    Args:
        values: List of values (most recent last)
        periods: Number of periods to calculate slope over
    
    Returns:
        Slope as percentage change per period
    """
    if len(values) < periods:
        return 0.0
    
    recent = values[-periods:]
    if len(recent) < 2 or recent[0] == 0:
        return 0.0
    
    # Calculate percentage change from first to last
    change = (recent[-1] - recent[0]) / recent[0]
    return change


def calculate_stochastic(bars: deque, k_period: int = 14, d_period: int = 3) -> Dict[str, float]:
    """
    Calculate Stochastic oscillator (%K and %D).
    
    Args:
        bars: Deque of OHLC bars
        k_period: Period for %K calculation
        d_period: Period for %D calculation (SMA of %K)
    
    Returns:
        Dictionary with 'k' and 'd' values (0-100)
    """
    if len(bars) < k_period:
        return {"k": 50.0, "d": 50.0}
    
    recent_bars = list(bars)[-k_period:]
    
    # Get current close and highest/lowest over period
    current_close = recent_bars[-1]["close"]
    highest_high = max(bar["high"] for bar in recent_bars)
    lowest_low = min(bar["low"] for bar in recent_bars)
    
    # Calculate %K
    if highest_high == lowest_low:
        k_value = 50.0
    else:
        k_value = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
    
    # Calculate %D (SMA of %K) - simplified: just return %K for now
    # Full implementation would track recent %K values
    d_value = k_value
    
    return {"k": k_value, "d": d_value}


def get_session_type(current_time) -> str:
    """
    Determine if currently in RTH (Regular Trading Hours) or ETH (Extended Trading Hours).
    
    RTH for ES: 9:30 AM - 4:00 PM ET
    ETH: All other times
    
    Args:
        current_time: datetime object in ET timezone
    
    Returns:
        "RTH" or "ETH"
    """
    hour = current_time.hour
    minute = current_time.minute
    
    # RTH: 9:30 AM - 4:00 PM ET
    if (hour == 9 and minute >= 30) or (10 <= hour < 16):
        return "RTH"
    else:
        return "ETH"


def get_volatility_regime(atr: float, symbol: str) -> str:
    """
    Classify current volatility regime based on ATR.
    
    Args:
        atr: Current ATR value
        symbol: Instrument symbol
    
    Returns:
        "LOW", "MEDIUM", or "HIGH"
    """
    # Get historical ATR values for comparison
    bars = state[symbol]["bars_1min"]
    if len(bars) < 100:
        return "MEDIUM"
    
    # Calculate average ATR over last 100 bars
    recent_bars = list(bars)[-100:]
    atr_values = []
    
    for i in range(14, len(recent_bars)):  # Need 14 bars for ATR
        bars_slice = recent_bars[i-14:i+1]
        true_ranges = []
        
        for j in range(1, len(bars_slice)):
            high = bars_slice[j]["high"]
            low = bars_slice[j]["low"]
            prev_close = bars_slice[j-1]["close"]
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        if true_ranges:
            atr_values.append(sum(true_ranges) / len(true_ranges))
    
    if not atr_values:
        return "MEDIUM"
    
    avg_atr = sum(atr_values) / len(atr_values)
    
    # Classify based on deviation from average
    if atr < avg_atr * 0.75:
        return "LOW"
    elif atr > avg_atr * 1.25:
        return "HIGH"
    else:
        return "MEDIUM"


def capture_market_state(symbol: str, current_price: float) -> Dict[str, Any]:
    """
    Capture market state snapshot for CAPITULATION REVERSAL pattern matching.
    
    SIMPLIFIED 16-FIELD STRUCTURE:
    ================================
    The 12 Pattern Matching Fields:
    1. flush_size_ticks
    2. flush_velocity
    3. volume_climax_ratio
    4. flush_direction
    5. rsi
    6. distance_from_flush_low
    7. reversal_candle
    8. no_new_extreme
    9. vwap_distance_ticks
    10. regime
    11. session
    12. hour
    
    The 4 Metadata Fields:
    13. symbol
    14. timestamp
    15. pnl (added later by record_outcome)
    16. took_trade (added later by record_outcome)
    
    DROPPED FIELDS: price, bars_since_flush_start, atr, stop_distance_ticks, 
    target_distance_ticks, risk_reward_ratio, duration, exploration_rate, 
    mfe, mae, order_type_used, entry_slippage_ticks, exit_reason
    
    Args:
        symbol: Instrument symbol
        current_price: Current market price
    
    Returns:
        Dictionary with 14 market state features (pnl and took_trade added later)
    """
    vwap = state[symbol].get("vwap", current_price)
    rsi = state[symbol].get("rsi", 50)
    
    # Get current time and session
    current_time = get_current_time()
    hour = current_time.hour
    session = get_session_type(current_time)
    
    # Get regime
    regime = state[symbol].get("current_regime", "NORMAL")
    
    # Get tick size
    tick_size = CONFIG.get("tick_size", 0.25)
    tick_value = CONFIG.get("tick_value", 12.50)
    
    # Calculate volume_climax_ratio (current volume vs 20-bar average)
    bars_1min = state[symbol]["bars_1min"]
    if len(bars_1min) >= 20:
        recent_volumes = [bar.get("volume", 0) for bar in list(bars_1min)[-20:]]
        avg_volume_20 = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 1
        current_bar = bars_1min[-1] if bars_1min else {"volume": 0}
        volume_climax_ratio = current_bar.get("volume", 0) / avg_volume_20 if avg_volume_20 > 0 else 1.0
    else:
        volume_climax_ratio = 1.0
    
    # Distance from VWAP in ticks
    vwap_actual = state[symbol].get("vwap", current_price)
    vwap_distance_ticks = (current_price - vwap_actual) / tick_size if tick_size > 0 else 0
    
    # Get capitulation detector state for flush metrics
    cap_detector = get_capitulation_detector(tick_size, tick_value)
    
    # Initialize flush-related fields
    flush_size_ticks = 0.0
    flush_velocity = 0.0
    flush_direction = "NONE"
    distance_from_flush_low = 0.0
    
    if cap_detector.last_flush:
        flush = cap_detector.last_flush
        flush_size_ticks = flush.flush_size_ticks
        flush_velocity = flush.flush_velocity
        flush_direction = flush.direction
        distance_from_flush_low = (current_price - flush.flush_low) / tick_size if tick_size > 0 else 0
    
    # Reversal candle detection (current bar closes green for longs, red for shorts)
    reversal_candle = False
    no_new_extreme = False
    
    if len(bars_1min) >= 2:
        current_bar = bars_1min[-1]
        prev_bar = bars_1min[-2]
        
        if flush_direction == "DOWN":
            # For long: green candle and stopped making new lows
            reversal_candle = current_bar["close"] > current_bar["open"]
            no_new_extreme = current_bar["low"] >= prev_bar["low"]
        elif flush_direction == "UP":
            # For short: red candle and stopped making new highs
            reversal_candle = current_bar["close"] < current_bar["open"]
            no_new_extreme = current_bar["high"] <= prev_bar["high"]
    
    # Build the simplified 16-field experience record structure
    market_state = {
        # The 12 Pattern Matching Fields
        "flush_size_ticks": round(flush_size_ticks, 1),
        "flush_velocity": round(flush_velocity, 2),
        "volume_climax_ratio": round(volume_climax_ratio, 2),
        "flush_direction": flush_direction,
        "rsi": round(rsi, 1) if rsi is not None else 50,
        "distance_from_flush_low": round(distance_from_flush_low, 1),
        "reversal_candle": reversal_candle,
        "no_new_extreme": no_new_extreme,
        "vwap_distance_ticks": round(vwap_distance_ticks, 1),
        "regime": regime,
        "session": session,
        "hour": hour,
        
        # The 4 Metadata Fields
        "symbol": symbol,
        "timestamp": current_time.isoformat(),
        # pnl and took_trade will be added by record_outcome()
    }
    
    return market_state




def check_for_signals(symbol: str) -> None:
    """
    Check for trading signals on each completed 1-minute bar.
    Coordinates signal detection through helper functions.
    
    Args:
        symbol: Instrument symbol
    """
    # Check safety conditions first
    is_safe, reason = check_safety_conditions(symbol)
    if not is_safe:
        # SILENCE DURING MAINTENANCE - no spam in logs
        if not bot_status.get("maintenance_idle", False):
            logger.info(f"[SIGNAL CHECK] Safety check failed: {reason}")
        return
    
    # Get the latest bar
    if len(state[symbol]["bars_1min"]) == 0:
        logger.info(f"[SIGNAL CHECK] No 1-min bars yet")
        return
    
    latest_bar = state[symbol]["bars_1min"][-1]
    bar_time = latest_bar["timestamp"]
    
    # Validate signal requirements
    is_valid, reason = validate_signal_requirements(symbol, bar_time)
    if not is_valid:
        # PERIODIC STATUS: Log validation failures periodically so users know why signals aren't generating
        # This helps users understand the bot is running but conditions aren't met
        validation_fail_counter = state[symbol].get("validation_fail_counter", 0) + 1
        state[symbol]["validation_fail_counter"] = validation_fail_counter
        
        # Log every 15 minutes (15 bars) - just show the reason, not strategy details
        if validation_fail_counter % 15 == 0:
            logger.info(f"📋 Signal check: {reason} - bot is monitoring and will trade when conditions allow")
        return
    
    # Reset validation fail counter when validation passes
    state[symbol]["validation_fail_counter"] = 0
    
    # Get bars for signal check
    prev_bar = state[symbol]["bars_1min"][-2]
    current_bar = state[symbol]["bars_1min"][-1]
    vwap = state[symbol].get("vwap", 0)
    regime = state[symbol].get("current_regime", "NORMAL")
    
    logger.debug(f"Signal check: regime={regime}, prev_low={prev_bar['low']:.2f}, "
                f"current_close={current_bar['close']:.2f}, vwap={vwap:.2f}")
    
    # PERIODIC HEARTBEAT: Show bot is actively scanning for signals (every 15 minutes)
    # Does NOT reveal strategy details - just confirms bot is running
    signal_check_counter = state[symbol].get("signal_check_counter", 0) + 1
    state[symbol]["signal_check_counter"] = signal_check_counter
    
    if signal_check_counter % 15 == 0:  # Every 15 minutes
        price = current_bar["close"]
        logger.info(f"📊 Bot active | Price: ${price:.2f} | Scanning for entry signals...")
    
    # Declare global RL brain for both signal checks
    global rl_brain
    
    # Check for long signal
    if check_long_signal_conditions(symbol, prev_bar, current_bar):
        # MARKET STATE CAPTURE - Record comprehensive market conditions
        # Capture current market state (flat structure with all 16 indicators)
        market_state = capture_market_state(symbol, current_bar["close"])
        
        # Ask cloud RL API for decision (or local RL as fallback)
        # Market state has all fields needed: rsi, vwap_distance, atr, volume_ratio, etc.
        take_signal, confidence, reason = get_ml_confidence(market_state, "long")
        
        if not take_signal:
            # Show rejected signals in both live mode and shadow mode
            # Users need to see all AI decisions to understand the system's behavior
            logger.info(f"⚠️  Signal Declined: LONG at ${market_state.get('price', 0):.2f} - {reason} (confidence: {confidence:.0%})")
            # Store the rejected signal state for potential future learning
            state[symbol]["last_rejected_signal"] = {
                "time": get_current_time(),
                "state": market_state,
                "side": "long",
                "confidence": confidence,
                "reason": reason
            }
            return
        
        # RL approved - adjust position size based on confidence
        regime = market_state.get('regime', 'NORMAL')
        
        # Show approved signal with confidence
        logger.info(f"✅ LONG SIGNAL APPROVED | Price: ${market_state.get('price', 0):.2f} | AI Confidence: {confidence:.0%} | Regime: {regime}")
        
        # Store market state for outcome recording
        state[symbol]["entry_market_state"] = market_state
        state[symbol]["entry_rl_confidence"] = confidence
        
        execute_entry(symbol, "long", current_bar["close"])
        return
    
    # Check for short signal
    if check_short_signal_conditions(symbol, prev_bar, current_bar):
        # MARKET STATE CAPTURE - Record comprehensive market conditions
        # Capture current market state (flat structure with all 16 indicators)
        market_state = capture_market_state(symbol, current_bar["close"])
        
        # Ask cloud RL API for decision (or local RL as fallback)
        # Market state has all fields needed: rsi, vwap_distance, atr, volume_ratio, etc.
        take_signal, confidence, reason = get_ml_confidence(market_state, "short")
        
        if not take_signal:
            # Show rejected signals in both live mode and shadow mode
            # Users need to see all AI decisions to understand the system's behavior
            logger.info(f"⚠️  Signal Declined: SHORT at ${market_state.get('price', 0):.2f} - {reason} (confidence: {confidence:.0%})")
            # Store the rejected signal state for potential future learning
            state[symbol]["last_rejected_signal"] = {
                "time": get_current_time(),
                "state": market_state,
                "side": "short",
                "confidence": confidence,
                "reason": reason
            }
            return
        
        # RL approved - adjust position size based on confidence
        regime = market_state.get('regime', 'NORMAL')
        
        # Show approved signal with confidence
        logger.info(f"✅ SHORT SIGNAL APPROVED | Price: ${market_state.get('price', 0):.2f} | AI Confidence: {confidence:.0%} | Regime: {regime}")
        
        # Store market state for outcome recording
        state[symbol]["entry_market_state"] = market_state
        state[symbol]["entry_rl_confidence"] = confidence
        
        execute_entry(symbol, "short", current_bar["close"])
        return


# ============================================================================
# PHASE EIGHT: Position Sizing
# ============================================================================

def calculate_position_size(symbol: str, side: str, entry_price: float, rl_confidence: Optional[float] = None) -> Tuple[int, float]:
    """
    Calculate position size based on risk management rules.
    
    FIXED CONTRACTS: User's max_contracts setting determines position size.
    - User configures max_contracts (e.g., 3 contracts)
    - Position size is ALWAYS fixed at this value (no dynamic scaling)
    - Risk-based calculation ensures we don't exceed risk tolerance
    
    STOP LOSS CALCULATION:
    - Uses user's "Max Loss Per Trade" setting from GUI (in dollars)
    - This represents the TOTAL TRADE RISK, not per-contract risk
    - Stop distance is calculated to ensure total risk equals user's setting
    - Automatically adapts to different symbols (ES, NQ, CL, GC, etc.)
    - Correctly handles symbol-specific tick sizes and tick values
    
    EXAMPLES (same $700 max loss, different symbols):
    - ES (tick_value=$12.50): Stop = 56 ticks, Total Risk = $700
    - NQ (tick_value=$5.00): Stop = 140 ticks, Total Risk = $700
    - CL (tick_value=$10.00): Stop = 70 ticks, Total Risk = $700
    
    MULTI-CONTRACT HANDLING:
    - All contracts use the SAME stop distance (same number of ticks)
    - Stop is NOT multiplied by contract count
    - With 3 contracts and $700 max loss on ES:
      * Stop distance = 56 ticks for ALL contracts
      * Total risk = $700 (NOT $700 × 3 = $2100)
    - This ensures risk scales correctly regardless of contract count
    
    Args:
        symbol: Instrument symbol (ES, NQ, CL, GC, etc.)
        side: 'long' or 'short'
        entry_price: Expected entry price
        rl_confidence: Optional RL confidence (for tracking, not used for position sizing)
    
    Returns:
        Tuple of (contracts, stop_price)
    """
    # Get account equity
    equity = get_account_equity()
    
    # Get max stop loss from GUI (user sets in dollars, e.g., $300)
    # This is the "Max Loss Per Trade" setting configured by the user in the launcher
    max_stop_dollars = CONFIG.get("max_stop_loss_dollars", DEFAULT_MAX_STOP_LOSS_DOLLARS)
    logger.info(f"Account equity: ${equity:.2f}, Max stop loss per trade: ${max_stop_dollars:.2f}")
    
    # Determine stop price using user's max stop loss setting
    vwap_bands = state[symbol]["vwap_bands"]
    vwap = state[symbol]["vwap"]
    
    # CRITICAL: Get symbol-specific tick values from SymbolSpec
    # This ensures correct stop loss placement for ALL symbols (ES, NQ, CL, GC, etc.)
    # Each symbol has different tick sizes and values:
    # - ES: tick_size=0.25, tick_value=$12.50 (4 ticks per point, $50 per point)
    # - NQ: tick_size=0.25, tick_value=$5.00 (4 ticks per point, $20 per point)
    # - CL: tick_size=0.01, tick_value=$10.00 (100 ticks per point, $1000 per point)
    # - GC: tick_size=0.10, tick_value=$10.00 (10 ticks per point, $100 per point)
    from symbol_specs import SYMBOL_SPECS
    if symbol in SYMBOL_SPECS:
        spec = SYMBOL_SPECS[symbol]
        tick_size = spec.tick_size
        tick_value = spec.tick_value
    else:
        # Fallback to config defaults (usually ES)
        tick_size = CONFIG.get("tick_size", 0.25)
        tick_value = CONFIG.get("tick_value", 12.50)
    
    # STEP 1: Convert user's max loss (dollars) to ticks
    # Formula: max_loss_dollars / tick_value = number of ticks
    # Example (ES): $700 / $12.50 = 56 ticks
    # Example (NQ): $700 / $5.00 = 140 ticks
    # Example (CL): $1000 / $10.00 = 100 ticks
    max_stop_ticks = max_stop_dollars / tick_value  # Convert dollars to ticks
    
    # STEP 2: Convert ticks to price distance
    # Formula: ticks * tick_size = price distance
    # Example (ES): 56 ticks * 0.25 = 14 points
    # Example (NQ): 140 ticks * 0.25 = 35 points
    # Example (CL): 100 ticks * 0.01 = 1.00 points
    stop_distance = max_stop_ticks * tick_size  # Convert ticks to price distance
    
    # Detect current regime for entry (for logging purposes)
    regime_detector = get_regime_detector()
    bars = state[symbol]["bars_1min"]
    atr = calculate_atr_1min(symbol, CONFIG.get("atr_period", 14))
    
    if atr is not None:
        entry_regime = regime_detector.detect_regime(bars, atr, CONFIG.get("atr_period", 14))
        logger.info(f"Fixed stop: {max_stop_ticks:.0f} ticks (${max_stop_dollars:.2f}) - Regime: {entry_regime.name}")
    else:
        logger.info(f"Fixed stop: {max_stop_ticks:.0f} ticks (${max_stop_dollars:.2f})")
    
    # STEP 3: Calculate stop price based on entry side
    # For LONG positions: stop is BELOW entry (entry - stop_distance)
    # For SHORT positions: stop is ABOVE entry (entry + stop_distance)
    if side == "long":
        stop_price = entry_price - stop_distance
    else:  # short
        stop_price = entry_price + stop_distance
    
    # Round to nearest valid tick (ensures broker accepts the price)
    stop_price = round_to_tick(stop_price)
    
    # STEP 4: Verify the risk calculation
    # Recalculate actual stop distance after rounding (may differ slightly)
    stop_distance = abs(entry_price - stop_price)
    ticks_at_risk = stop_distance / tick_size
    
    # Calculate actual risk per contract in dollars
    # This should equal max_stop_dollars (or very close due to rounding)
    risk_per_contract = ticks_at_risk * tick_value
    
    # STEP 5: Use fixed contracts from GUI (no dynamic scaling)
    # User explicitly sets the contract count in the launcher
    # The bot respects this setting regardless of account size or risk
    user_max_contracts = CONFIG["max_contracts"]
    contracts = user_max_contracts
    
    logger.info(f"[FIXED CONTRACTS] Using {contracts} contract(s) as set in GUI")
    
    if contracts == 0:
        logger.warning(f"Position size is zero - check GUI settings")
        return 0, stop_price
    
    # TOTAL RISK CALCULATION EXPLANATION:
    # ====================================
    # The stop distance (in ticks) is the SAME for all contracts in the position.
    # This is NOT a per-contract stop - all contracts share the same stop price.
    # 
    # The max_loss_per_trade setting defines the TOTAL POSITION RISK.
    # 
    # HOW IT WORKS:
    # - Calculate stop distance: max_loss_per_trade / tick_value = ticks
    # - ALL contracts use this SAME tick distance
    # - Total risk = ticks × tick_value = max_loss_per_trade (original setting)
    # 
    # Example with ES (tick_value=$12.50):
    # - User sets max_loss_per_trade = $700
    # - Stop distance = $700 / $12.50 = 56 ticks
    # 
    # Scenario 1: 1 contract
    # - Stop: 56 ticks
    # - Total risk: 56 ticks × $12.50 = $700 ✓
    # 
    # Scenario 2: 3 contracts  
    # - Stop: 56 ticks (SAME as 1 contract, not 56×3!)
    # - Total risk: 56 ticks × $12.50 = $700 ✓ (NOT $700×3 = $2100)
    # 
    # This ensures the user's risk tolerance is respected exactly as configured,
    # regardless of how many contracts are traded.
    
    logger.info(f"Position sizing: {contracts} contract(s)")
    logger.info(f"  Entry: ${entry_price:.2f}, Stop: ${stop_price:.2f}")
    logger.info(f"  Risk: {ticks_at_risk:.1f} ticks (${risk_per_contract:.2f})")
    logger.info(f"  VWAP: ${vwap:.2f} (mean reversion reference)")
    
    return contracts, stop_price


# ============================================================================
# PHASE NINE: Entry Execution
# ============================================================================

def validate_entry_price_still_valid(symbol: str, signal_price: float, side: str) -> Tuple[bool, str, float]:
    """
    Validate that current market price hasn't moved too far from signal.
    SIMPLIFIED: Check once and decide immediately - no waiting.
    
    If price moved away, skip the entry and let the next bar generate a new signal
    if conditions persist. This keeps the bot responsive and lets the RL system
    learn optimal entry timing patterns.
    
    Args:
        symbol: Instrument symbol
        signal_price: Original signal price
        side: 'long' or 'short'
    
    Returns:
        Tuple of (is_valid, reason, current_market_price)
    """
    max_deterioration_ticks = CONFIG.get("max_entry_price_deterioration_ticks", 3)
    tick_size = CONFIG["tick_size"]
    max_deterioration = max_deterioration_ticks * tick_size
    
    # Get current market price from bid/ask if available
    current_price = signal_price  # Default to signal price
    if bid_ask_manager is not None:
        quote = bid_ask_manager.get_current_quote(symbol)
        if quote:
            # For longs, use ask (what we'd pay)
            # For shorts, use bid (what we'd receive)
            current_price = quote.ask_price if side == "long" else quote.bid_price
    
    # Check price movement
    price_move = current_price - signal_price
    price_move_ticks = abs(price_move) / tick_size
    
    # For longs: price moving UP is bad (paying more)
    # For shorts: price moving DOWN is bad (receiving less)
    is_acceptable = False
    
    if side == "long":
        is_acceptable = price_move <= max_deterioration
    else:  # short
        is_acceptable = price_move >= -max_deterioration
    
    # Instant decision - no waiting
    if is_acceptable:
        logger.info(f"  Γ£à Entry price valid: ${signal_price:.2f} ΓåÆ ${current_price:.2f} ({price_move_ticks:+.1f} ticks, limit: {max_deterioration_ticks})")
        return True, "Price acceptable", current_price
    else:
        if side == "long":
            reason = f"Price moved UP {price_move_ticks:.1f} ticks (limit: {max_deterioration_ticks})"
        else:
            reason = f"Price moved DOWN {price_move_ticks:.1f} ticks (limit: {max_deterioration_ticks})"
        
        logger.warning(f"  Γ¥î Entry skipped - {reason}")
        logger.warning(f"     Signal: ${signal_price:.2f} ΓåÆ Current: ${current_price:.2f}")
        logger.info(f"     Will retry on next bar if signal persists")
        return False, reason, current_price


def is_market_moving_too_fast(symbol: str) -> Tuple[bool, str]:
    """
    Detect if market is moving too fast for safe entry.
    
    CRITICAL FIX: Execution Risk #4 - Fast Market Detection
    
    Returns:
        Tuple of (too_fast, reason)
    """
    if bid_ask_manager is None:
        return False, "No bid/ask data available"
    
    try:
        # Check spread widening (sign of fast market)
        spread_analyzer = bid_ask_manager.spread_analyzer
        is_widening, widening_reason = spread_analyzer.is_spread_widening()
        if is_widening:
            return True, f"Fast market detected: {widening_reason}"
    except Exception as e:
        logger.debug(f"Could not check spread widening: {e}")
    
    # Check recent price volatility
    bars = state[symbol]["bars_1min"]
    if len(bars) < 5:
        return False, "Not enough data"
    
    # Convert deque to list for slicing
    recent_bars = list(bars)[-5:]
    price_ranges = [(b["high"] - b["low"]) for b in recent_bars]
    avg_range = statistics.mean(price_ranges)
    current_range = recent_bars[-1]["high"] - recent_bars[-1]["low"]
    
    # If current bar range is > multiplier * average, market is moving fast
    volatility_mult = CONFIG.get("fast_market_volatility_multiplier", 2.0)
    if current_range > avg_range * volatility_mult:
        return True, f"High volatility: {current_range:.2f} vs avg {avg_range:.2f} ({current_range/avg_range:.1f}x threshold)"
    
    return False, "Normal market conditions"


def execute_entry(symbol: str, side: str, entry_price: float) -> None:
    """
    Execute entry order with stop loss.
    
    SIMPLE EXECUTION:
    - All entries use MARKET ORDER for immediate execution
    - Stop loss placed immediately based on user's max_loss_per_trade GUI setting
    - Position state validation (prevents double positioning)
    - Fast market detection (skips dangerous entries)
    
    SHADOW MODE: Logs signal without placing actual orders.
    
    Args:
        symbol: Instrument symbol
        side: 'long' or 'short'
        entry_price: Approximate entry price (mid or last)
    """
    # ===== SHADOW MODE: Signal-only (manual trading mode) =====
    if _bot_config.shadow_mode:
        logger.info(SEPARATOR_LINE)
        logger.info(f"≡ƒôè SIGNAL ALERT - MANUAL TRADE OPPORTUNITY")
        logger.info(f"  Symbol: {symbol}")
        logger.info(f"  Direction: {side.upper()}")
        logger.info(f"  Entry Price: ${entry_price:.2f}")
        logger.info(f"  Time: {get_current_time().strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info(f"  VWAP: ${state[symbol]['vwap']:.2f}")
        
        # Show suggested stop and target
        vwap_bands = state[symbol]["vwap_bands"]
        tick_size = CONFIG["tick_size"]
        max_stop_ticks = 11
        
        if side == "long":
            suggested_stop = entry_price - (max_stop_ticks * tick_size)
            logger.info(f"  Suggested Stop: ${suggested_stop:.2f} ({max_stop_ticks} ticks)")
        else:
            suggested_stop = entry_price + (max_stop_ticks * tick_size)
            logger.info(f"  Suggested Stop: ${suggested_stop:.2f} ({max_stop_ticks} ticks)")
        
        logger.info(f"")
        logger.info(f"  ≡ƒÄ» SHADOW MODE: Signal shown - No automatic execution")
        logger.info(f"  ≡ƒô▒ Trade manually if you agree with this signal")
        logger.info(SEPARATOR_LINE)
        
        # Send notification if enabled
        try:
            notifier = get_notifier()
            notifier.send_trade_alert(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                stop_price=suggested_stop,
                mode="SIGNAL_ONLY"
            )
        except Exception as e:
            logger.debug(f"Notification send failed: {e}")
        
        # EXIT - Don't execute the trade, just return
        return
    
    # ===== CRITICAL FIX #1: Position State Validation =====
    # Prevent double positioning if signal fires while already in trade
    current_position = get_position_quantity(symbol)
    
    if current_position != 0:
        logger.warning(SEPARATOR_LINE)
        logger.warning("≡ƒÜ¿ ENTRY SKIPPED - Already In Position")
        logger.warning(f"  Current Position: {current_position} contracts ({'LONG' if current_position > 0 else 'SHORT'})")
        logger.warning(f"  New Signal: {side.upper()} @ ${entry_price:.2f}")
        logger.warning(f"  Reason: Cannot enter conflicting or additional position")
        logger.warning(SEPARATOR_LINE)
        return
    
    # ===== EXECUTION RISK FIX #4: Fast Market Detection =====
    too_fast, fast_reason = is_market_moving_too_fast(symbol)
    if too_fast:
        logger.warning(SEPARATOR_LINE)
        logger.warning("≡ƒÜ¿ ENTRY SKIPPED - Fast Market Detected")
        logger.warning(f"  Reason: {fast_reason}")
        logger.warning(f"  Signal: {side.upper()} @ ${entry_price:.2f}")
        logger.warning(SEPARATOR_LINE)
        return
    
    # ===== EXECUTION RISK FIX #1: Price Deterioration Protection =====
    is_valid_price, price_reason, current_market_price = validate_entry_price_still_valid(symbol, entry_price, side)
    if not is_valid_price:
        logger.warning(SEPARATOR_LINE)
        logger.warning("≡ƒÜ¿ ENTRY ABORTED - Price Deteriorated")
        logger.warning(f"  {price_reason}")
        logger.warning(f"  Signal: {side.upper()} @ ${entry_price:.2f}")
        logger.warning(SEPARATOR_LINE)
        
        # Store for potential shadow outcome tracking
        rl_state = state[symbol].get("entry_rl_state")
        if rl_state is not None:
            # Extract price movement from reason (e.g., "Price moved UP 5.0 ticks")
            import re
            match = re.search(r'(\d+\.?\d*)\s+ticks', price_reason)
            price_move_ticks = float(match.group(1)) if match else 0.0
            
            state[symbol]["last_rejected_signal"] = {
                "time": get_current_time(),
                "state": rl_state,
                "side": side,
                "reason": price_reason,
                "price_move_ticks": price_move_ticks,
                "signal_price": entry_price,
                "current_price": current_market_price
            }
        
        return
    
    # Use current market price instead of stale signal price
    logger.info(f"  [OK] Price validation passed: ${entry_price:.2f} -> ${current_market_price:.2f}")
    entry_price = current_market_price
    
    # Get RL confidence if available (for tracking)
    rl_confidence = state[symbol].get("entry_rl_confidence")
    
    # Calculate position size
    contracts, stop_price = calculate_position_size(symbol, side, entry_price, rl_confidence)
    
    if contracts == 0:
        logger.warning("Cannot enter trade - position size is zero")
        return
    
    # Phase 12: Validate order before placing
    is_valid, error_msg = validate_order(symbol, side, contracts, entry_price, stop_price)
    if not is_valid:
        logger.error(f"Order validation failed: {error_msg}")
        return
    
    # Phase Four: Final time check before placing order
    entry_time = get_current_time()
    trading_state = get_trading_state(entry_time)
    
    if trading_state not in ["entry_window"]:
        logger.warning(SEPARATOR_LINE)
        logger.warning("ENTRY ABORTED - No longer in entry window")
        logger.warning(f"  Current state: {trading_state}")
        logger.warning(f"  Time: {entry_time.strftime('%H:%M:%S %Z')}")
        logger.warning(SEPARATOR_LINE)
        return
    
    logger.info(SEPARATOR_LINE)
    logger.info(f"ENTERING {side.upper()} POSITION")
    logger.info(f"  Time: {entry_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"  Symbol: {symbol}")
    
    # Position size is fixed based on user's GUI settings
    # No spread-aware adjustments - keep it simple
    
    logger.info(f"  Contracts: {contracts}")
    logger.info(f"  Stop Loss: ${stop_price:.2f}")
    
    # Track order execution details for post-trade analysis
    fill_start_time = datetime.now()
    order_type_used = "market"  # Always market order
    
    # Prepare order parameters
    order_side = "BUY" if side == "long" else "SELL"
    actual_fill_price = entry_price
    order = None
    
    # CRITICAL FIX: Set entry order pending flag BEFORE placing order
    # This prevents position reconciliation from clearing state while order is inflight
    bot_status["entry_order_pending"] = True
    bot_status["entry_order_pending_since"] = datetime.now()
    bot_status["entry_order_pending_symbol"] = symbol
    bot_status["entry_order_pending_id"] = None  # Will be set after order is placed
    
    try:
        # SIMPLE ENTRY: Always use market order for immediate execution
        # Stop loss is placed immediately based on user's max_loss_per_trade setting
        logger.info(f"  Entry: MARKET ORDER")
        order = place_market_order(symbol, order_side, contracts)
        order_type_used = "market"
        
        if order is not None:
            bot_status["entry_order_pending_id"] = order.get("order_id")
            # CRITICAL: IMMEDIATELY save minimal position state to prevent loss on crash
            state[symbol]["position"]["active"] = True
            state[symbol]["position"]["side"] = side
            state[symbol]["position"]["quantity"] = contracts
            state[symbol]["position"]["entry_price"] = entry_price
            state[symbol]["position"]["entry_time"] = entry_time
            state[symbol]["position"]["order_id"] = order.get("order_id")
            save_position_state(symbol)
            logger.info(f"  [OK] Market order placed successfully")
        
        if order is None:
            logger.error("Failed to place entry order")
            return
    
    finally:
        # CRITICAL FIX: Clear entry order pending flag when entry completes (success or failure)
        # This allows position reconciliation to resume
        bot_status["entry_order_pending"] = False
        bot_status["entry_order_pending_since"] = None
        bot_status["entry_order_pending_symbol"] = None
        bot_status["entry_order_pending_id"] = None
    
    # ===== CRITICAL FIX #7: Entry Fill Validation (Live Trading) =====
    # Validate actual entry fill price vs expected (critical for live trading)
    # In live trading, get actual fill price from broker
    try:
        actual_fill_from_broker = get_last_fill_price(symbol)
        if actual_fill_from_broker and actual_fill_from_broker != actual_fill_price:
            # Calculate entry slippage using symbol-specific tick values
            tick_size, tick_value = get_symbol_tick_specs(symbol)
            entry_slippage = abs(actual_fill_from_broker - actual_fill_price)
            entry_slippage_ticks = entry_slippage / tick_size
            entry_slippage_cost = entry_slippage_ticks * tick_value * contracts
            
            # Get alert threshold from config
            entry_slippage_alert_threshold = CONFIG.get("entry_slippage_alert_ticks", 2)
            
            if entry_slippage_ticks > entry_slippage_alert_threshold:
                # HIGH ENTRY SLIPPAGE DETECTED
                logger.warning("=" * 80)
                logger.warning("[WARN] CRITICAL: HIGH ENTRY SLIPPAGE DETECTED!")
                logger.warning("=" * 80)
                logger.warning(f"  Expected Entry: ${actual_fill_price:.2f}")
                logger.warning(f"  Actual Fill: ${actual_fill_from_broker:.2f}")
                logger.warning(f"  Slippage: {entry_slippage_ticks:.1f} ticks (${entry_slippage_cost:.2f})")
                logger.warning(f"  Side: {side.upper()}, Contracts: {contracts}")
                logger.warning(f"  [WARN] Entry slippage >{entry_slippage_alert_threshold} ticks - consider tighter price validation or avoid volatile periods")
                logger.warning("=" * 80)
                
                # Track for session statistics
                if "high_entry_slippage_count" not in bot_status:
                    bot_status["high_entry_slippage_count"] = 0
                bot_status["high_entry_slippage_count"] += 1
            elif entry_slippage_ticks > 0:
                # Normal slippage logging
                logger.info(f"  Entry Slippage: {entry_slippage_ticks:.1f} ticks (${entry_slippage_cost:.2f})")
            
            # Use actual fill price for position tracking
            actual_fill_price = actual_fill_from_broker
            logger.info(f"  Validated Fill Price: ${actual_fill_price:.2f}")
    except Exception as e:
        logger.debug(f"Could not validate entry fill price: {e}")
    
    logger.info(f"  Final Entry Price: ${actual_fill_price:.2f}")
    
    # Send trade entry alert
    try:
        notifier = get_notifier()
        notifier.send_trade_alert(
            trade_type="ENTRY",
            symbol=symbol,
            price=actual_fill_price,
            contracts=contracts,
            side="LONG" if side == 'long' else "SHORT"
        )
    except Exception as e:
        logger.debug(f"Failed to send entry alert: {e}")
    
    # Record trade execution for cost tracking (Requirement 5)
    if bid_ask_manager is not None:
        try:
            fill_time_seconds = (datetime.now() - fill_start_time).total_seconds()
            bid_ask_manager.record_trade_execution(
                symbol=symbol,
                side=side,
                signal_price=entry_price,
                fill_price=actual_fill_price,
                quantity=contracts,
                order_type=order_type_used
            )
            
            # Record for post-trade analysis (Requirement 13)
            quote = bid_ask_manager.get_current_quote(symbol)
            if quote:
                estimated_costs = {"total": quote.spread}
                actual_costs = {"total": abs(actual_fill_price - entry_price)}
                bid_ask_manager.record_post_trade_analysis(
                    signal_price=entry_price,
                    fill_price=actual_fill_price,
                    side=side,
                    order_type=order_type_used,
                    spread_at_order=quote.spread,
                    fill_time_seconds=fill_time_seconds,
                    estimated_costs=estimated_costs,
                    actual_costs=actual_costs
                )
        except Exception as e:
            logger.warning(f"  Could not record trade execution: {e}")
    
    # Calculate initial risk in ticks
    stop_distance_ticks = abs(actual_fill_price - stop_price) / CONFIG["tick_size"]
    
    # Detect entry regime
    regime_detector = get_regime_detector()
    bars = state[symbol]["bars_1min"]
    atr = calculate_atr_1min(symbol, CONFIG.get("atr_period", 14))
    if atr is None:
        atr = DEFAULT_FALLBACK_ATR  # Use constant instead of magic number
        logger.warning(f"ATR not calculable, using fallback value: {DEFAULT_FALLBACK_ATR}")
    
    entry_regime = regime_detector.detect_regime(bars, atr, CONFIG.get("atr_period", 14))
    
    # CAPITULATION REVERSAL: Fixed trade management rules (no regime adjustments)
    breakeven_trigger = CONFIG.get("breakeven_trigger_ticks", 12)
    breakeven_offset = CONFIG.get("breakeven_offset_ticks", 1)
    trailing_trigger = CONFIG.get("trailing_trigger_ticks", 15)
    trailing_distance = CONFIG.get("trailing_distance_ticks", 8)
    time_stop_bars = CONFIG.get("time_stop_bars", 20)
    time_stop_enabled = CONFIG.get("time_stop_enabled", False)
    
    logger.info(f"")
    logger.info(f"  📊 CAPITULATION REVERSAL - FIXED RULES")
    logger.info(f"  Entry Regime: {entry_regime.name}")
    logger.info(f"")
    logger.info(f"  Stop Loss:")
    logger.info(f"    Primary: 2 ticks beyond flush extreme")
    logger.info(f"    Stop Distance: {stop_distance_ticks:.1f} ticks (${abs(actual_fill_price - stop_price):.2f})")
    logger.info(f"    Stop Price: ${stop_price:.2f}")
    logger.info(f"")
    logger.info(f"  Breakeven Protection:")
    logger.info(f"    Trigger: {breakeven_trigger} ticks profit")
    logger.info(f"    Action: Move stop to entry + {breakeven_offset} tick")
    logger.info(f"")
    logger.info(f"  Trailing Stop:")
    logger.info(f"    Trigger: {trailing_trigger} ticks profit")
    logger.info(f"    Trail Distance: {trailing_distance} ticks behind peak")
    logger.info(f"")
    if time_stop_enabled:
        logger.info(f"  Time Stop: Exit after {time_stop_bars} bars")
    else:
        logger.info(f"  Time Stop: Disabled (trade to target or stop)")
    
    # Update position tracking
    state[symbol]["position"] = {
        "active": True,
        "side": side,
        "quantity": contracts,
        "entry_price": actual_fill_price,
        "stop_price": stop_price,
        "entry_time": entry_time,
        "order_id": order.get("order_id"),
        "order_type_used": order_type_used,  # Track for exit optimization
        # Signal & RL Information - Preserved for partial fills and exits
        "entry_rl_confidence": rl_confidence,  # RL confidence at entry (for tracking)
        "entry_rl_state": state[symbol].get("entry_rl_state"),  # RL market state
        "original_entry_price": entry_price,  # Original signal price (before validation)
        "actual_entry_price": actual_fill_price,  # Actual fill price
        # Regime Information - For dynamic exit management
        "entry_regime": entry_regime.name,  # Regime at entry
        "current_regime": entry_regime.name,  # Current regime (updated on each tick)
        "regime_change_time": None,  # When regime last changed
        "regime_history": [],  # List of regime transitions with timestamps
        "entry_atr": atr,  # ATR at entry
        # Advanced Exit Management - Breakeven State
        "breakeven_active": False,
        "original_stop_price": stop_price,
        "breakeven_activated_time": None,
        # Advanced Exit Management - Trailing Stop State
        "trailing_stop_active": False,
        "trailing_stop_price": None,
        "highest_price_reached": actual_fill_price if side == "long" else None,
        "lowest_price_reached": actual_fill_price if side == "short" else None,
        "trailing_activated_time": None,
        # Advanced Exit Management - Time-Decay State
        "time_decay_50_triggered": False,
        "time_decay_75_triggered": False,
        "time_decay_90_triggered": False,
        "original_stop_distance_ticks": stop_distance_ticks,
        "current_stop_distance_ticks": stop_distance_ticks,
        # Advanced Exit Management - Partial Exit State
        "partial_exit_1_completed": False,
        "partial_exit_2_completed": False,
        "partial_exit_3_completed": False,
        "original_quantity": contracts,
        "remaining_quantity": contracts,
        "partial_exit_history": [],
        # Advanced Exit Management - General
        "initial_risk_ticks": stop_distance_ticks,
    }
    
    # CRITICAL: IMMEDIATELY save position state to disk - NEVER forget we're in a trade!
    save_position_state(symbol)
    logger.info("  Γ£ô Position state saved to disk")
    
    # ===== CRITICAL FIX #2: Stop Loss Execution Validation =====
    # Verify stop order accepted by broker - critical for capital protection
    stop_side = "SELL" if side == "long" else "BUY"
    stop_order = place_stop_order(symbol, stop_side, contracts, stop_price)
    
    if stop_order:
        state[symbol]["position"]["stop_order_id"] = stop_order.get("order_id")
        logger.info(f"  [OK] Stop loss placed and validated: ${stop_price:.2f}")
        logger.info(f"     Stop Order ID: {stop_order.get('order_id')}")
    else:
        # CRITICAL: Stop order rejected - this is DANGEROUS!
        logger.error(SEPARATOR_LINE)
        logger.error("≡ƒÜ¿ CRITICAL: STOP ORDER REJECTED BY BROKER!")
        logger.error(f"  Entry filled at ${actual_fill_price:.2f} with {contracts} contracts")
        logger.error(f"  Stop order at ${stop_price:.2f} FAILED to place")
        logger.error(f"  Position is NOW UNPROTECTED - emergency exit required!")
        logger.error(SEPARATOR_LINE)
        
        # EMERGENCY: Close position immediately with market order
        logger.error("  ≡ƒåÿ Executing emergency market close to protect capital...")
        emergency_close_order = place_market_order(symbol, stop_side, contracts)
        
        if emergency_close_order:
            logger.error(f"  Γ£ô Emergency close executed - Position closed")
            logger.error(f"  This trade is abandoned due to stop order failure")
            
            # CRITICAL FIX: Clear position state since we closed it
            state[symbol]["position"]["active"] = False
            state[symbol]["position"]["quantity"] = 0
            state[symbol]["position"]["side"] = None
            state[symbol]["position"]["entry_price"] = None
            state[symbol]["position"]["stop_price"] = None
            
            # CRITICAL: Save state to disk immediately
            save_position_state(symbol)
            logger.error("  Γ£ô Position state cleared and saved to disk")
        else:
            logger.error(f"  [FAIL] EMERGENCY CLOSE ALSO FAILED - MANUAL INTERVENTION REQUIRED!")
            logger.error(f"  Symbol: {symbol}, Side: {side}, Contracts: {contracts}")
            
            # CRITICAL FIX: Even if close failed, mark position as needing manual intervention
            # Don't let bot think it can trade normally with this broken state
            state[symbol]["position"]["active"] = False  # Prevent bot from managing this
            bot_status["emergency_stop"] = True
            bot_status["stop_reason"] = "stop_order_placement_failed"
            
            # CRITICAL: Save state to disk immediately
            save_position_state(symbol)
            logger.error("  Γ£ô Emergency stop activated and saved to disk")
        
        # Don't track this position - it's closed or needs manual handling
        logger.error(SEPARATOR_LINE)
        return
    
    # Increment daily trade counter
    state[symbol]["daily_trade_count"] += 1
    
    logger.info(f"Position opened successfully")
    logger.info(SEPARATOR_LINE)
# ============================================================================
# PHASE TEN: Exit Management
# ============================================================================

def check_stop_hit(symbol: str, current_bar: Dict[str, Any], position: Dict[str, Any]) -> Tuple[bool, Optional[float]]:
    """
    Check if stop loss has been hit.
    
    Args:
        symbol: Instrument symbol
        current_bar: Current 1-minute bar
        position: Position dictionary
    
    Returns:
        Tuple of (stop_hit, stop_price)
    """
    side = position["side"]
    stop_price = position["stop_price"]
    
    if side == "long":
        if current_bar["low"] <= stop_price:
            return True, stop_price
    else:  # short
        if current_bar["high"] >= stop_price:
            return True, stop_price
    
    return False, None


def check_reversal_signal(symbol: str, current_bar: Dict[str, Any], position: Dict[str, Any]) -> Tuple[bool, Optional[float]]:
    """
    Check if price has reached VWAP BEFORE trailing stop activates.
    
    CAPITULATION REVERSAL EXIT STRATEGY:
    - Trailing stop handles all profit-taking (activates at 15+ ticks profit)
    - VWAP is a SAFETY NET only used if price reaches VWAP before trailing activates
    - Once trailing is active (15+ ticks), ignore VWAP and let trailing ride
    
    Logic:
    - If price reaches VWAP before trailing activates (before 15 ticks profit): exit at VWAP
    - If trailing activates first (15+ ticks profit): let it ride, ignore VWAP target
    - Trailing stop eventually exits you, either at VWAP or beyond
    
    This captures more profit on big reversals while still protecting gains on weak ones.
    
    Args:
        symbol: Instrument symbol
        current_bar: Current 1-minute bar
        position: Position dictionary
    
    Returns:
        Tuple of (should_exit_at_vwap, exit_price)
    """
    # If trailing stop is already active, DO NOT use VWAP target
    # Let trailing stop manage the exit for maximum profit capture
    if position.get("trailing_stop_active", False):
        return False, None
    
    vwap = state[symbol].get("vwap")
    side = position["side"]
    entry_price = position.get("entry_price", 0)
    
    # VWAP not available - cannot check target
    if vwap is None or vwap <= 0:
        return False, None
    
    # Calculate current profit in ticks
    tick_size = CONFIG.get("tick_size", 0.25)
    current_price = current_bar["close"]
    
    if side == "long":
        profit_ticks = (current_price - entry_price) / tick_size
    else:
        profit_ticks = (entry_price - current_price) / tick_size
    
    # Check if trailing stop would be active at this profit level
    trailing_trigger = CONFIG.get("trailing_trigger_ticks", 15)
    if profit_ticks >= trailing_trigger:
        # Trailing stop should be active - don't use VWAP target
        return False, None
    
    # Trailing not active yet - check if we've reached VWAP (safety net)
    if side == "long":
        if current_price >= vwap:
            logger.info(f"📊 VWAP TARGET HIT (before trailing): Price ${current_price:.2f} >= VWAP ${vwap:.2f}")
            return True, current_price
    
    if side == "short":
        if current_price <= vwap:
            logger.info(f"📊 VWAP TARGET HIT (before trailing): Price ${current_price:.2f} <= VWAP ${vwap:.2f}")
            return True, current_price
    
    return False, None


def check_time_based_exits(symbol: str, current_bar: Dict[str, Any], position: Dict[str, Any], 
                           bar_time: datetime) -> Tuple[Optional[str], Optional[float]]:
    """
    Check time-based exit conditions.
    
    ONLY checks for:
    1. Emergency forced flatten at 4:45 PM ET (market close before maintenance)
    
    All other exits use fixed rules (stops, breakeven, trailing) - NOT regime-based.
    Trade management parameters are set at entry and never change.
    
    Args:
        symbol: Instrument symbol
        current_bar: Current 1-minute bar
        position: Position dictionary
        bar_time: Current bar timestamp
    
    Returns:
        Tuple of (exit_reason, exit_price) or (None, None)
    """
    side = position["side"]
    
    # Force close at forced_flatten_time (4:45 PM ET - maintenance starts)
    trading_state = get_trading_state(bar_time)
    if trading_state == "closed":
        return "emergency_forced_flatten", get_flatten_price(symbol, side, current_bar["close"])
    
    return None, None


# ============================================================================
# PHASE THREE: Breakeven Protection Logic
# ============================================================================

def check_breakeven_protection(symbol: str, current_price: float) -> None:
    """
    Check if breakeven protection should be activated and move stop to breakeven.
    
    CAPITULATION REVERSAL STRATEGY - FIXED RULES (no regime adjustments):
    - Trigger: After 12 ticks profit
    - Action: Move stop to entry + 1 tick
    - Same rule every time, no exceptions
    
    Args:
        symbol: Instrument symbol
        current_price: Current market price
    """
    
    position = state[symbol]["position"]
    
    # Step 1 - Check eligibility: Only process positions that haven't activated breakeven yet
    if not position["active"] or position["breakeven_active"]:
        return
    
    side = position["side"]
    entry_price = position["entry_price"]
    # Use symbol-specific tick values for accurate P&L calculation across different instruments
    tick_size, tick_value = get_symbol_tick_specs(symbol)
    
    # CAPITULATION REVERSAL: Fixed breakeven threshold at 12 ticks (no regime adjustment)
    breakeven_threshold_ticks = CONFIG.get("breakeven_trigger_ticks", 12)
    
    # Stop at entry + 1 tick (locks in 1 tick profit)
    breakeven_offset_ticks = CONFIG.get("breakeven_offset_ticks", 1)
    
    # Step 2 - Calculate current profit in ticks
    if side == "long":
        profit_ticks = (current_price - entry_price) / tick_size
    else:  # short
        profit_ticks = (entry_price - current_price) / tick_size
    
    # Step 3 - Compare to threshold
    if profit_ticks < breakeven_threshold_ticks:
        return  # Not enough profit yet
    
    # Step 4 - Calculate new breakeven stop price
    if side == "long":
        new_stop_price = entry_price + (breakeven_offset_ticks * tick_size)
    else:  # short
        new_stop_price = entry_price - (breakeven_offset_ticks * tick_size)
    
    new_stop_price = round_to_tick(new_stop_price)
    
    # Step 5 - Update stop loss with continuous protection
    # PROFESSIONAL APPROACH: Place new stop FIRST, then cancel old
    # This ensures continuous protection (brief dual coverage is safer than no coverage)
    stop_side = "SELL" if side == "long" else "BUY"
    contracts = position["quantity"]
    new_stop_order = place_stop_order(symbol, stop_side, contracts, new_stop_price)
    
    if new_stop_order:
        # New stop confirmed - now safe to cancel old stop
        old_stop_order_id = position.get("stop_order_id")
        if old_stop_order_id:
            cancel_success = cancel_order(symbol, old_stop_order_id)
            if cancel_success:
                logger.debug(f"✔ Replaced stop order: {old_stop_order_id} → {new_stop_order.get('order_id')}")
            else:
                logger.warning(f"⚠ New stop active but failed to cancel old stop {old_stop_order_id}")
    
    if new_stop_order:
        # Update position tracking
        position["breakeven_active"] = True
        position["breakeven_activated_time"] = get_current_time()
        original_stop = position["original_stop_price"]
        position["stop_price"] = new_stop_price
        if new_stop_order.get("order_id"):
            position["stop_order_id"] = new_stop_order.get("order_id")
        
        # Calculate profit locked in
        profit_locked_ticks = (new_stop_price - entry_price) / tick_size if side == "long" else (entry_price - new_stop_price) / tick_size
        profit_locked_dollars = profit_locked_ticks * tick_value * contracts
        
        # Step 6 - Log activation
        logger.info("=" * 60)
        logger.info("🛡️ BREAKEVEN PROTECTION ACTIVATED")
        logger.info("=" * 60)
        logger.info(f"  Trigger: {breakeven_threshold_ticks} ticks profit reached")
        logger.info(f"  Current Profit: {profit_ticks:.1f} ticks")
        logger.info(f"  Original Stop: ${original_stop:.2f}")
        logger.info(f"  New Stop: ${new_stop_price:.2f} (entry + {breakeven_offset_ticks} tick)")
        logger.info(f"  Profit Locked: ${profit_locked_dollars:.2f}")
        logger.info("=" * 60)
    else:
        logger.error("Failed to place breakeven stop order")


# ============================================================================
# PHASE FOUR: Trailing Stop Logic
# ============================================================================

def check_trailing_stop(symbol: str, current_price: float) -> None:
    """
    Check and update trailing stop based on price movement.
    
    CAPITULATION REVERSAL STRATEGY - FIXED RULES (no regime adjustments):
    - Trigger: After 15 ticks profit
    - Trail: 8 ticks behind the peak profit
    - Same rule every time, no exceptions
    
    Runs AFTER breakeven check. Only processes positions where breakeven is already active.
    Continuously updates stop to follow profitable price movement while protecting gains.
    
    Args:
        symbol: Instrument symbol
        current_price: Current market price
    """
    
    position = state[symbol]["position"]
    
    # Step 1 - Check eligibility: Position must have breakeven active
    if not position["active"] or not position["breakeven_active"]:
        return
    
    side = position["side"]
    entry_price = position["entry_price"]
    # Use symbol-specific tick values for accurate P&L calculation across different instruments
    tick_size, tick_value = get_symbol_tick_specs(symbol)
    
    # CAPITULATION REVERSAL: Fixed trailing parameters (no regime adjustment)
    trailing_distance_ticks = CONFIG.get("trailing_distance_ticks", 8)
    min_profit_ticks = CONFIG.get("trailing_trigger_ticks", 15)
    
    # Calculate current profit
    if side == "long":
        profit_ticks = (current_price - entry_price) / tick_size
    else:  # short
        profit_ticks = (entry_price - current_price) / tick_size
    
    # Must exceed minimum profit threshold before activating trailing
    if profit_ticks < min_profit_ticks:
        return
    
    # Step 2 - Track price extremes
    if side == "long":
        # Update highest price reached
        if position["highest_price_reached"] is None:
            position["highest_price_reached"] = current_price
        else:
            position["highest_price_reached"] = max(position["highest_price_reached"], current_price)
        
        price_extreme = position["highest_price_reached"]
    else:  # short
        # Update lowest price reached
        if position["lowest_price_reached"] is None:
            position["lowest_price_reached"] = current_price
        else:
            position["lowest_price_reached"] = min(position["lowest_price_reached"], current_price)
        
        price_extreme = position["lowest_price_reached"]
    
    # Step 3 - Calculate trailing stop
    if side == "long":
        new_trailing_stop = price_extreme - (trailing_distance_ticks * tick_size)
    else:  # short
        new_trailing_stop = price_extreme + (trailing_distance_ticks * tick_size)
    
    new_trailing_stop = round_to_tick(new_trailing_stop)
    
    # Step 4 - Compare to current stop (never move stop backwards)
    current_stop = position["stop_price"]
    
    should_update = False
    if side == "long":
        # For longs, only update if new stop is HIGHER
        if new_trailing_stop > current_stop:
            should_update = True
    else:  # short
        # For shorts, only update if new stop is LOWER
        if new_trailing_stop < current_stop:
            should_update = True
    
    if not should_update:
        return  # No improvement, don't update
    
    # Step 5 - Update stop loss with continuous protection
    # PROFESSIONAL APPROACH: Place new stop FIRST, then cancel old
    stop_side = "SELL" if side == "long" else "BUY"
    contracts = position["quantity"]
    new_stop_order = place_stop_order(symbol, stop_side, contracts, new_trailing_stop)
    
    if new_stop_order:
        # New trailing stop confirmed - now safe to cancel old stop
        old_stop_order_id = position.get("stop_order_id")
        if old_stop_order_id:
            cancel_success = cancel_order(symbol, old_stop_order_id)
            if cancel_success:
                logger.debug(f"✔ Replaced stop order: {old_stop_order_id} → {new_stop_order.get('order_id')}")
            else:
                logger.warning(f"⚠ New trailing stop active but failed to cancel old stop {old_stop_order_id}")
    
    if new_stop_order:
        # Activate trailing stop flag if not already active
        if not position["trailing_stop_active"]:
            position["trailing_stop_active"] = True
            position["trailing_activated_time"] = get_current_time()
            logger.info(f"📈 TRAILING STOP ACTIVATED - Trail distance: {trailing_distance_ticks} ticks")
        
        # Update position tracking
        old_stop = position["stop_price"]
        position["stop_price"] = new_trailing_stop
        position["trailing_stop_price"] = new_trailing_stop
        if new_stop_order.get("order_id"):
            position["stop_order_id"] = new_stop_order.get("order_id")
        
        # Calculate profit now locked in
        profit_locked_ticks = (new_trailing_stop - entry_price) / tick_size if side == "long" else (entry_price - new_trailing_stop) / tick_size
        profit_locked_dollars = profit_locked_ticks * tick_value * contracts
        
        # Step 6 - Log updates
        logger.info("=" * 60)
        logger.info("📈 TRAILING STOP UPDATED")
        logger.info("=" * 60)
        logger.info(f"  Side: {side.upper()}")
        logger.info(f"  Price Extreme: ${price_extreme:.2f}")
        logger.info(f"  Old Stop: ${old_stop:.2f}")
        logger.info(f"  New Stop: ${new_trailing_stop:.2f}")
        logger.info(f"  Profit Locked: {profit_locked_ticks:.1f} ticks (${profit_locked_dollars:+.2f})")
        logger.info(f"  Current Price: ${current_price:.2f}")
        logger.info("=" * 60)
    else:
        # ===== CRITICAL FIX #3: Trailing Stop Validation =====
        logger.error(SEPARATOR_LINE)
        logger.error("≡ƒÜ¿ CRITICAL: TRAILING STOP UPDATE FAILED!")
        logger.error(f"  Tried to update stop from ${position['stop_price']:.2f} to ${new_trailing_stop:.2f}")
        logger.error(f"  Current profit: ${profit_locked_dollars:+.2f} (UNPROTECTED)")
        logger.error("  Position now at risk - emergency exit required!")
        logger.error(SEPARATOR_LINE)
        
        # EMERGENCY: Close position immediately to lock in profit
        logger.error("  ≡ƒåÿ Executing emergency market close to protect profit...")
        emergency_close_order = place_market_order(symbol, stop_side, contracts)
        
        if emergency_close_order:
            logger.error("  Γ£ô Emergency close executed - profit protected")
            # Execute full exit with tracking
            execute_exit(symbol, current_price, "trailing_stop_failure_emergency")
        else:
            logger.error("  [FAIL] EMERGENCY CLOSE ALSO FAILED - MANUAL INTERVENTION REQUIRED!")
            logger.error(f"  Position: {side.upper()} {contracts} contracts at ${entry_price:.2f}")
            logger.error(f"  Current Price: ${current_price:.2f}, Profit at Risk: ${profit_locked_dollars:+.2f}")


# ============================================================================
# PHASE FIVE: Time-Decay Tightening Logic
# ============================================================================

def check_time_decay_tightening(symbol: str, current_time: datetime) -> None:
    """
    Tighten stop loss as position ages to reduce risk over time.
    
    Applies progressive tightening at 50%, 75%, and 90% of max holding period.
    Only tightens stops on profitable positions.
    
    Args:
        symbol: Instrument symbol
        current_time: Current datetime
    """
    position = state[symbol]["position"]
    
    # Only process active positions
    if not position["active"]:
        return
    
    # Get entry time
    entry_time = position["entry_time"]
    if entry_time is None:
        return
    
    side = position["side"]
    entry_price = position["entry_price"]
    # Use symbol-specific tick values for accurate calculations across different instruments
    tick_size, tick_value = get_symbol_tick_specs(symbol)
    
    # Step 1 - Calculate time percentage
    # Max holding period: use time until flatten mode (conservative)
    # From entry window end (2:30 PM) to flatten deadline (4:45 PM) = 135 minutes
    max_holding_minutes = 60  # Conservative 60 minute max hold as mentioned in config
    
    time_held = (current_time - entry_time).total_seconds() / 60.0  # minutes
    time_percentage = (time_held / max_holding_minutes) * 100.0
    
    # Alert if position stuck (held > 120 minutes = 2 hours)
    if time_held > 120 and not position.get("stuck_alert_sent", False):
        try:
            unrealized_pnl = position.get("unrealized_pnl", 0.0)
            notifier = get_notifier()
            notifier.send_error_alert(
                error_message=f"Position stuck! Held for {time_held:.0f} minutes ({time_held/60:.1f} hours). Side: {side}. P&L: ${unrealized_pnl:.2f}",
                error_type="Position Stuck"
            )
            position["stuck_alert_sent"] = True
        except Exception as e:
            logger.debug(f"Failed to send position stuck alert: {e}")
    
    # Step 2 - Determine tightening level
    tightening_pct = None
    threshold_flag = None
    
    if time_percentage >= 90 and not position["time_decay_90_triggered"]:
        tightening_pct = CONFIG.get("time_decay_90_percent_tightening", 0.30)
        threshold_flag = "time_decay_90_triggered"
    elif time_percentage >= 75 and not position["time_decay_75_triggered"]:
        tightening_pct = CONFIG.get("time_decay_75_percent_tightening", 0.20)
        threshold_flag = "time_decay_75_triggered"
    elif time_percentage >= 50 and not position["time_decay_50_triggered"]:
        tightening_pct = CONFIG.get("time_decay_50_percent_tightening", 0.10)
        threshold_flag = "time_decay_50_triggered"
    
    # Step 3 - Check if already tightened (handled above with flags)
    if tightening_pct is None or threshold_flag is None:
        return  # No tightening needed at this time
    
    # Step 6 - Only tighten if profitable
    current_price = state[symbol]["bars_1min"][-1]["close"] if len(state[symbol]["bars_1min"]) > 0 else entry_price
    if side == "long":
        unrealized_profit_ticks = (current_price - entry_price) / tick_size
    else:  # short
        unrealized_profit_ticks = (entry_price - current_price) / tick_size
    
    if unrealized_profit_ticks <= 0:
        logger.debug(f"Time-decay skipped: position not profitable ({unrealized_profit_ticks:.1f} ticks)")
        return
    
    # Step 4 - Calculate new stop distance
    original_stop_distance_ticks = position["original_stop_distance_ticks"]
    if original_stop_distance_ticks is None:
        # Calculate from current position if not set
        original_stop = position["original_stop_price"]
        original_stop_distance_ticks = abs(entry_price - original_stop) / tick_size
        position["original_stop_distance_ticks"] = original_stop_distance_ticks
    
    new_stop_distance_ticks = original_stop_distance_ticks * (1.0 - tightening_pct)
    
    # Step 5 - Calculate new stop price
    if side == "long":
        new_stop_price = entry_price - (new_stop_distance_ticks * tick_size)
    else:  # short
        new_stop_price = entry_price + (new_stop_distance_ticks * tick_size)
    
    new_stop_price = round_to_tick(new_stop_price)
    
    # Only update if new stop is better than current stop
    current_stop = position["stop_price"]
    should_update = False
    
    if side == "long":
        if new_stop_price > current_stop:
            should_update = True
    else:  # short
        if new_stop_price < current_stop:
            should_update = True
    
    if not should_update:
        logger.debug(f"Time-decay skipped: new stop ${new_stop_price:.2f} not better than current ${current_stop:.2f}")
        return
    
    # Step 7 - Update stop loss with continuous protection
    # PROFESSIONAL APPROACH: Place new stop FIRST, then cancel old
    stop_side = "SELL" if side == "long" else "BUY"
    contracts = position["quantity"]
    new_stop_order = place_stop_order(symbol, stop_side, contracts, new_stop_price)
    
    if new_stop_order:
        # New tightened stop confirmed - now safe to cancel old stop
        old_stop_order_id = position.get("stop_order_id")
        if old_stop_order_id:
            cancel_success = cancel_order(symbol, old_stop_order_id)
            if cancel_success:
                logger.debug(f"Γ£ô Replaced stop order: {old_stop_order_id} ΓåÆ {new_stop_order.get('order_id')}")
            else:
                logger.warning(f"ΓÜá New tightened stop active but failed to cancel old stop {old_stop_order_id}")
    
    if new_stop_order:
        # Update position tracking
        old_stop = position["stop_price"]
        position["stop_price"] = new_stop_price
        position["current_stop_distance_ticks"] = new_stop_distance_ticks
        position[threshold_flag] = True  # Mark this tightening level as complete
        if new_stop_order.get("order_id"):
            position["stop_order_id"] = new_stop_order.get("order_id")
        
        # Step 8 - Log tightening
        logger.info("=" * 60)
        logger.info("TIME-DECAY TIGHTENING ACTIVATED")
        logger.info("=" * 60)
        logger.info(f"  Time Held: {time_held:.1f} minutes ({time_percentage:.1f}% of max)")
        logger.info(f"  Tightening Applied: {tightening_pct * 100:.0f}%")
        logger.info(f"  Original Stop Distance: {original_stop_distance_ticks:.1f} ticks")
        logger.info(f"  New Stop Distance: {new_stop_distance_ticks:.1f} ticks")
        logger.info(f"  Old Stop: ${old_stop:.2f}")
        logger.info(f"  New Stop: ${new_stop_price:.2f}")
        logger.info("=" * 60)
    else:
        logger.error("Failed to place time-decay tightened stop order")


# ============================================================================
# PHASE SIX: Partial Exit Logic
# ============================================================================

def check_partial_exits(symbol: str, current_price: float) -> None:
    """
    Execute partial exits at predefined R-multiple thresholds.
    
    Scales out of position at 2R, 3R, and 5R to lock in profits while
    maintaining exposure to further gains.
    
    NOTE: This feature is DISABLED by default. Set partial_exits_enabled=True in config to enable.
    
    Args:
        symbol: Instrument symbol
        current_price: Current market price
    """
    # Check if partial exits are enabled in configuration
    if not CONFIG.get("partial_exits_enabled", False):
        return  # Partial exits disabled - exit entire position at target instead
    
    position = state[symbol]["position"]
    
    # Only process active positions
    if not position["active"]:
        return
    
    side = position["side"]
    entry_price = position["entry_price"]
    # Use symbol-specific tick values for accurate P&L calculation across different instruments
    tick_size, tick_value = get_symbol_tick_specs(symbol)
    
    # Get initial risk
    initial_risk_ticks = position["initial_risk_ticks"]
    if initial_risk_ticks is None or initial_risk_ticks <= 0:
        logger.warning("Cannot calculate R-multiple: initial risk not set")
        return
    
    # Step 1 - Calculate R-multiple
    if side == "long":
        profit_ticks = (current_price - entry_price) / tick_size
    else:  # short
        profit_ticks = (entry_price - current_price) / tick_size
    
    r_multiple = profit_ticks / initial_risk_ticks
    
    # Get original quantity (for calculating partial sizes)
    original_quantity = position["original_quantity"]
    if original_quantity <= 1:
        # Step 10 - Handle edge case: skip partials for single contract
        logger.debug("Skipping partial exits: only 1 contract")
        return
    
    # Check each partial exit threshold in order
    
    # Step 2 & 3 & 4 - First partial (50% at 2.0R)
    if (r_multiple >= CONFIG.get("partial_exit_1_r_multiple", 2.0) and 
        not position["partial_exit_1_completed"]):
        
        partial_pct = CONFIG.get("partial_exit_1_percentage", 0.50)
        contracts_to_close = int(original_quantity * partial_pct)
        
        if contracts_to_close >= 1:
            execute_partial_exit(symbol, contracts_to_close, current_price, r_multiple, 
                                "partial_exit_1_completed", 1, partial_pct)
            return  # Exit one partial per bar to avoid race conditions
    
    # Step 5 & 6 & 7 - Second partial (30% at 3.0R)
    if (r_multiple >= CONFIG.get("partial_exit_2_r_multiple", 3.0) and 
        not position["partial_exit_2_completed"]):
        
        partial_pct = CONFIG.get("partial_exit_2_percentage", 0.30)
        contracts_to_close = int(original_quantity * partial_pct)
        
        if contracts_to_close >= 1:
            execute_partial_exit(symbol, contracts_to_close, current_price, r_multiple,
                                "partial_exit_2_completed", 2, partial_pct)
            return
    
    # Step 8 & 9 - Third partial (remaining 20% at 5.0R)
    if (r_multiple >= CONFIG.get("partial_exit_3_r_multiple", 5.0) and 
        not position["partial_exit_3_completed"]):
        
        # Close all remaining contracts (final exit at 5.0R)
        remaining_quantity = position["remaining_quantity"]
        
        if remaining_quantity >= 1:
            execute_partial_exit(symbol, remaining_quantity, current_price, r_multiple,
                                "partial_exit_3_completed", 3, 1.0, is_final=True)
            return


def execute_partial_exit(symbol: str, contracts: int, exit_price: float, r_multiple: float,
                        completion_flag: str, level: int, percentage: float, is_final: bool = False) -> None:
    """
    Execute a partial exit and update position tracking.
    
    Args:
        symbol: Instrument symbol
        contracts: Number of contracts to close
        exit_price: Exit price
        r_multiple: Current R-multiple
        completion_flag: Position flag to mark this partial as complete
        level: Partial exit level (1, 2, or 3)
        percentage: Percentage of original position being closed
        is_final: Whether this is the final exit closing entire position
    """
    position = state[symbol]["position"]
    side = position["side"]
    # Use symbol-specific tick values for accurate P&L calculation across different instruments
    tick_size, tick_value = get_symbol_tick_specs(symbol)
    entry_price = position["entry_price"]
    
    # Calculate profit for this partial
    if side == "long":
        profit_ticks = (exit_price - entry_price) / tick_size
    else:
        profit_ticks = (entry_price - exit_price) / tick_size
    
    profit_dollars = profit_ticks * tick_value * contracts
    
    logger.info("=" * 60)
    logger.info(f"PARTIAL EXIT #{level} - {percentage * 100:.0f}% @ {r_multiple:.1f}R")
    logger.info("=" * 60)
    logger.info(f"  Closing: {contracts} of {position['original_quantity']} contracts")
    logger.info(f"  Exit Price: ${exit_price:.2f}")
    logger.info(f"  Profit: {profit_ticks:.1f} ticks (${profit_dollars:+.2f})")
    logger.info(f"  R-Multiple: {r_multiple:.2f}")
    
    # Execute the partial exit
    order_side = "SELL" if side == "long" else "BUY"
    order = place_market_order(symbol, order_side, contracts)
    
    if order:
        # Verify actual fill in live mode (not backtest)
        if not is_backtest_mode():
            time_module.sleep(1)  # Brief wait for fill confirmation
            
            # Check actual position to verify fill
            current_position = abs(get_position_quantity(symbol))
            expected_remaining = position["remaining_quantity"] - contracts
            
            # Check if fill doesn't match expected (allow for small broker reporting delays)
            actual_remaining_diff = abs(current_position - expected_remaining)
            if actual_remaining_diff > 0:
                # Partial fill detected
                actual_filled = position["remaining_quantity"] - current_position
                logger.warning(f"  [PARTIAL FILL] Only {actual_filled} of {contracts} contracts filled")
                contracts = actual_filled  # Adjust to actual filled amount
                
                # Recalculate profit based on actual fill
                profit_dollars = profit_ticks * tick_value * contracts
        
        # Update position tracking with actual filled amount
        position["remaining_quantity"] -= contracts
        position["quantity"] = position["remaining_quantity"]
        position[completion_flag] = True
        
        # Add to partial exit history
        position["partial_exit_history"].append({
            "price": exit_price,
            "quantity": contracts,
            "r_multiple": r_multiple,
            "level": level
        })
        
        # Step 10 - Handle edge case: check if position should be fully closed
        if position["remaining_quantity"] < 1 or is_final:
            logger.info("  Position FULLY CLOSED via partial exits")
            logger.info("=" * 60)
            
            # Mark position as inactive
            position["active"] = False
            
            # Save position state to ensure persistence across restarts
            save_position_state(symbol)
            
            # Update daily P&L
            state[symbol]["daily_pnl"] += profit_dollars
            
            # Update session stats
            update_session_stats(symbol, profit_dollars)
        else:
            logger.info(f"  Remaining: {position['remaining_quantity']} contracts")
            logger.info("=" * 60)
            
            # Update daily P&L for this partial
            state[symbol]["daily_pnl"] += profit_dollars
            
            # Save position state after partial exit
            save_position_state(symbol)
    else:
        logger.error(f"Failed to execute partial exit #{level}")


def update_current_regime(symbol: str) -> None:
    """
    Update the current regime for the symbol based on latest bars.
    This is called after each bar completion to keep regime detection current.
    
    Args:
        symbol: Instrument symbol
    """
    regime_detector = get_regime_detector()
    bars = state[symbol]["bars_1min"]
    
    # Need enough bars for regime detection (114 = 100 baseline + 14 current)
    if len(bars) < 114:
        state[symbol]["current_regime"] = "NORMAL"
        return
    
    current_atr = calculate_atr_1min(symbol, CONFIG.get("atr_period", 14))
    if current_atr is None:
        state[symbol]["current_regime"] = "NORMAL"
        return
    
    # Store previous regime to detect changes
    prev_regime = state[symbol].get("current_regime", "NORMAL")
    
    # Detect and store current regime
    detected_regime = regime_detector.detect_regime(bars, current_atr, CONFIG.get("atr_period", 14))
    state[symbol]["current_regime"] = detected_regime.name
    
    # Log regime changes for customers (not just backtest)
    # SILENCE DURING MAINTENANCE - no spam in logs
    if prev_regime != detected_regime.name and not bot_status.get("maintenance_idle", False):
        logger.info(f"📊 Market Regime Changed: {prev_regime} → {detected_regime.name}")
    else:
        pass  # Silent - no regime change or in maintenance


def check_regime_change(symbol: str, current_price: float) -> None:
    """
    Check if market regime has changed during an active trade (informational only).
    
    This function:
    1. Detects current regime from last 20 bars
    2. Compares to entry regime
    3. Logs regime change for awareness (DOES NOT adjust trade parameters)
    
    IMPORTANT: Trade management uses FIXED rules - regime changes do NOT affect:
    - Stop loss (set at entry, only moved by trailing)
    - Breakeven threshold (fixed at 12 ticks)
    - Trailing distance (fixed at 8 ticks)
    
    Regime only affects trade ENTRY decisions, never trade MANAGEMENT.
    
    Args:
        symbol: Instrument symbol
        current_price: Current market price
    """
    position = state[symbol]["position"]
    
    # Only check for active positions
    if not position["active"]:
        return
    
    # Get entry regime
    entry_regime_name = position.get("entry_regime", "NORMAL")
    
    # Detect current regime
    regime_detector = get_regime_detector()
    bars = state[symbol]["bars_1min"]
    current_atr = calculate_atr_1min(symbol, CONFIG.get("atr_period", 14))
    
    if current_atr is None:
        logger.debug("ATR not calculable, skipping regime change check")
        return  # Can't detect regime without ATR
    
    current_regime = regime_detector.detect_regime(bars, current_atr, CONFIG.get("atr_period", 14))
    
    # Check if regime has changed
    has_changed, new_regime = regime_detector.check_regime_change(
        entry_regime_name, current_regime
    )
    
    if not has_changed:
        return  # No regime change
    
    # Update position tracking with new regime
    change_time = get_current_time()
    position["current_regime"] = current_regime.name
    position["regime_change_time"] = change_time
    
    # Record regime transition in history
    if "regime_history" not in position:
        position["regime_history"] = []
    
    position["regime_history"].append({
        "from_regime": entry_regime_name,
        "to_regime": current_regime.name,
        "timestamp": change_time
    })
    
    # Log the regime change for awareness (informational only - no parameter changes)
    logger.info("=" * 60)
    logger.info(f"📊 REGIME CHANGE DETECTED: {entry_regime_name} → {current_regime.name}")
    logger.info("=" * 60)
    logger.info(f"  Time: {get_current_time().strftime('%H:%M:%S')}")
    logger.info(f"  Trade management UNCHANGED (fixed rules):")
    logger.info(f"  - Breakeven: 12 ticks | Trailing: 8 ticks | Stop: Set at entry")
    logger.info("=" * 60)


def check_exit_conditions(symbol: str) -> None:
    """
    Check exit conditions for open position on each bar.
    Coordinates various exit checks through helper functions.
    
    Args:
        symbol: Instrument symbol
    """
    if not state[symbol]["position"]["active"]:
        return
    
    position = state[symbol]["position"]
    
    if len(state[symbol]["bars_1min"]) == 0:
        return
    
    current_bar = state[symbol]["bars_1min"][-1]
    bar_time = current_bar["timestamp"]
    side = position["side"]
    entry_price = position["entry_price"]
    stop_price = position["stop_price"]
    current_price = current_bar["close"]
    
    # CRITICAL: Update price extremes on EVERY bar for accurate MFE/MAE tracking
    # This tracks the best and worst prices reached during the trade
    if side == "long":
        # Track highest price for longs (MFE)
        if position["highest_price_reached"] is None:
            position["highest_price_reached"] = current_price
        else:
            position["highest_price_reached"] = max(position["highest_price_reached"], current_price)
        
        # Track lowest price for longs (MAE)
        if position["lowest_price_reached"] is None:
            position["lowest_price_reached"] = current_price
        else:
            position["lowest_price_reached"] = min(position["lowest_price_reached"], current_price)
    else:  # short
        # Track highest price for shorts (MAE)
        if position["highest_price_reached"] is None:
            position["highest_price_reached"] = current_price
        else:
            position["highest_price_reached"] = max(position["highest_price_reached"], current_price)
        
        # Track lowest price for shorts (MFE)
        if position["lowest_price_reached"] is None:
            position["lowest_price_reached"] = current_price
        else:
            position["lowest_price_reached"] = min(position["lowest_price_reached"], current_price)
    
    # Phase Two: Check trading state and handle market close/open
    trading_state = get_trading_state(bar_time)
    
    # AUTO-FLATTEN: Market closing - ALWAYS flatten (no config option)
    if trading_state == "closed" and position["active"]:
        # FIX: Check if flatten is already in progress to prevent spam
        # This prevents repeated flatten attempts when order is already placed
        if bot_status.get("flatten_in_progress", False):
            flatten_since = bot_status.get("flatten_in_progress_since")
            if flatten_since:
                elapsed = (datetime.now() - flatten_since).total_seconds()
                # Allow up to FLATTEN_IN_PROGRESS_TIMEOUT seconds for flatten to complete before retrying
                if elapsed < FLATTEN_IN_PROGRESS_TIMEOUT:
                    logger.debug(f"Flatten already in progress for {elapsed:.1f}s - skipping duplicate attempt")
                    return
                else:
                    # Flatten has been pending too long - clear and retry
                    logger.warning(f"Flatten pending for {elapsed:.1f}s - clearing flag and retrying")
                    clear_flatten_flags()
        
        logger.critical(SEPARATOR_LINE)
        logger.critical("MARKET CLOSING - AUTO-FLATTENING POSITION")
        logger.critical(f"Time: {bar_time.strftime('%H:%M:%S %Z')}")
        logger.critical(f"Position: {side.upper()} {position['quantity']} @ ${entry_price:.2f}")
        logger.critical(SEPARATOR_LINE)
        
        # FIX: Set flatten in progress BEFORE placing order to prevent spam
        set_flatten_flags(symbol)
        
        # SAFEGUARD: Verify with broker that we actually have this position before flattening
        # This prevents over-flattening if multiple flatten attempts are made
        can_flatten, actual_quantity, verify_reason = verify_broker_position_for_flatten(
            symbol, side, position["quantity"]
        )
        
        if not can_flatten:
            logger.warning(f"Skipping flatten: {verify_reason}")
            position["active"] = False
            position["flatten_pending"] = False
            clear_flatten_flags()
            return
        
        # Update position quantity if it differs from broker
        if actual_quantity != position["quantity"]:
            position["quantity"] = actual_quantity
        
        # Force close immediately
        close_side = "sell" if side == "long" else "buy"
        flatten_price = get_flatten_price(symbol, side, current_bar["close"])
        
        log_time_based_action(
            "market_close_flatten",
            "Market closed - auto-flattening position for 24/7 operation",
            {
                "side": side,
                "quantity": actual_quantity,  # Use verified broker quantity
                "entry_price": f"${entry_price:.2f}",
                "exit_price": f"${flatten_price:.2f}",
                "time": bar_time.strftime('%H:%M:%S %Z')
            }
        )
        
        # Execute close order
        handle_exit_orders(symbol, position, flatten_price, "market_close")
        
        # FIX: Mark position as inactive AFTER placing the flatten order
        # This prevents repeated flatten attempts while order is being processed
        # Position reconciliation will verify the actual state with broker
        position["active"] = False
        position["flatten_pending"] = True  # Track that we're waiting for flatten confirmation
        
        logger.info("Position flattened - bot will continue running and auto-resume when market opens")
        return
    
    # Market closed - Force close all positions
    # NOTE: This block should not trigger because the first block handles market close
    # and sets position["active"] = False. This is a safety fallback.
    if trading_state == "closed":
        if position["active"] and not position.get("flatten_pending", False):
            # FIX: Check if flatten is already in progress
            if bot_status.get("flatten_in_progress", False):
                logger.debug("Flatten already in progress - skipping emergency flatten")
                return
            
            # SAFEGUARD: Verify with broker that we actually have this position
            can_flatten, actual_quantity, verify_reason = verify_broker_position_for_flatten(
                symbol, side, position["quantity"]
            )
            if not can_flatten:
                logger.warning(f"Skipping emergency flatten: {verify_reason}")
                position["active"] = False
                position["flatten_pending"] = False
                return
            
            # Update position quantity if it differs from broker
            if actual_quantity != position["quantity"]:
                position["quantity"] = actual_quantity
            
            logger.warning(SEPARATOR_LINE)
            logger.warning("MARKET CLOSED - EMERGENCY POSITION FLATTEN")
            logger.warning(f"Time: {bar_time.strftime('%H:%M:%S %Z')}")
            logger.warning(SEPARATOR_LINE)
            
            # Calculate unrealized PnL for logging
            tick_size = CONFIG["tick_size"]
            tick_value = CONFIG["tick_value"]
            if side == "long":
                price_change = current_bar["close"] - entry_price
            else:
                price_change = entry_price - current_bar["close"]
            ticks = price_change / tick_size
            unrealized_pnl = ticks * tick_value * actual_quantity
            
            flatten_details = {
                "reason": "Market closed - maintenance/weekend",
                "side": position["side"],
                "quantity": actual_quantity,
                "entry_price": f"${position['entry_price']:.2f}",
                "current_price": f"${current_bar['close']:.2f}",
                "unrealized_pnl": f"${unrealized_pnl:+.2f}",
                "time": bar_time.strftime('%H:%M:%S %Z')
            }
            
            log_time_based_action(
                "emergency_market_closure",
                "Position closed due to market closure (maintenance or weekend)",
                flatten_details
            )
            
            # Send emergency flatten alert
            try:
                notifier = get_notifier()
                notifier.send_error_alert(
                    error_message=f"Emergency position close - market closed at {bar_time.strftime('%I:%M %p %Z')}. Position: {side.upper()} {position['quantity']} @ ${entry_price:.2f}, closed @ ${current_bar['close']:.2f}, P/L: ${unrealized_pnl:+.2f}",
                    error_type="Emergency Flatten - Market Closed"
                )
            except Exception as e:
                logger.debug(f"Failed to send emergency flatten alert: {e}")
            
            # FIX: Mark position as inactive to prevent spam
            position["active"] = False
            position["flatten_pending"] = True
        
        # Don't process bars when market is closed
        return
        logger.critical("FLATTEN MODE ACTIVATED - POSITION MUST CLOSE IN 15 MINUTES")
        logger.critical(SEPARATOR_LINE)
        
        # Send pre-maintenance daily recap alert
        try:
            notifier = get_notifier()
            
            # Calculate daily stats for recap
            total_trades = state[symbol]["daily_trade_count"]
            daily_pnl = state[symbol]["daily_pnl"]
            stats = state[symbol]["session_stats"]
            winning_trades = stats["win_count"]
            losing_trades = stats["loss_count"]
            total_completed = winning_trades + losing_trades
            win_rate = (winning_trades / total_completed * 100) if total_completed > 0 else 0.0
            
            # Build recap message
            recap_msg = f"≡ƒôè PRE-MAINTENANCE DAILY RECAP (4:45 PM ET)\n\n"
            recap_msg += f"Trades Today: {total_trades}\n"
            recap_msg += f"Daily P&L: ${daily_pnl:+.2f}\n"
            recap_msg += f"Win Rate: {win_rate:.1f}% ({winning_trades}W/{losing_trades}L)\n"
            
            if position["quantity"] > 0:
                recap_msg += f"\nOpen Position: {position['side'].upper()} {position['quantity']} @ ${position['entry_price']:.2f}\n"
                recap_msg += f"Unrealized P&L: ${unrealized_pnl:+.2f}\n"
            
            recap_msg += f"\nBot entering flatten mode. All positions will close before 4:45 PM forced flatten."
            
            notifier.send_error_alert(
                error_message=recap_msg,
                error_type="Daily Recap"
            )
        except Exception as e:
            logger.debug(f"Failed to send daily recap alert: {e}")
    
    # ========================================================================
    # PHASE SEVEN: Integration Priority and Execution Order
    # ========================================================================
    # Critical execution sequence - order matters!
    
    # FIRST - Time-based exit check (only emergency flatten at 4:45 PM)
    reason, price = check_time_based_exits(symbol, current_bar, position, bar_time)
    if reason:
        # Log specific messages for certain exit types
        if reason == "emergency_forced_flatten":
            logger.critical(SEPARATOR_LINE)
            logger.critical("EMERGENCY FORCED FLATTEN - 4:45 PM MARKET CLOSURE")
            logger.critical(SEPARATOR_LINE)
        
        execute_exit(symbol, price, reason)
        return
    
    # SECOND - VWAP stop hit check
    stop_hit, price = check_stop_hit(symbol, current_bar, position)
    if stop_hit:
        execute_exit(symbol, price, "stop_loss")
        return
    
    # REGIME CHANGE CHECK - Detect and log regime changes (informational only)
    # Trade management uses FIXED rules and is NOT affected by regime changes
    check_regime_change(symbol, current_bar["close"])
    
    # Get current time for any time-based checks
    current_time = get_current_time()
    
    # FOURTH - Partial exits (happens before breakeven/trailing because it reduces position size)
    check_partial_exits(symbol, current_bar["close"])
    
    # FIFTH - Breakeven protection (must activate before trailing)
    check_breakeven_protection(symbol, current_bar["close"])
    
    # SIXTH - Trailing stop (only runs if breakeven already active)
    check_trailing_stop(symbol, current_bar["close"])
    
    # SEVENTH - Market divergence check (tighten stops if fighting momentum)
    if position["active"]:
        is_diverging, divergence_reason = check_market_divergence(symbol, position["side"])
        if is_diverging:
            logger.warning(f"ΓÜá∩╕Å DIVERGENCE DETECTED: {divergence_reason}")
            
            # Tighten stop by 30% when fighting momentum
            current_stop = position["stop_price"]
            entry_price = position["entry_price"]
            tick_size = CONFIG["tick_size"]
            
            # Calculate tightened stop (30% closer to entry)
            stop_distance = abs(entry_price - current_stop)
            new_stop_distance = stop_distance * 0.70  # Tighten by 30%
            
            if side == "long":
                new_stop = entry_price - new_stop_distance
            else:
                new_stop = entry_price + new_stop_distance
            
            new_stop = round_to_tick(new_stop)
            
            # Only tighten (never loosen)
            should_tighten = False
            if side == "long" and new_stop > current_stop:
                should_tighten = True
            elif side == "short" and new_stop < current_stop:
                should_tighten = True
            
            if should_tighten:
                logger.warning(f"≡ƒöÆ Tightening stop due to divergence: ${current_stop:.2f} ΓåÆ ${new_stop:.2f}")
                
                # PROFESSIONAL APPROACH: Place new stop FIRST, then cancel old
                stop_side = "SELL" if side == "long" else "BUY"
                contracts = position["quantity"]
                new_stop_order = place_stop_order(symbol, stop_side, contracts, new_stop)
                
                if new_stop_order:
                    # New divergence stop confirmed - now safe to cancel old stop
                    old_stop_order_id = position.get("stop_order_id")
                    if old_stop_order_id:
                        cancel_success = cancel_order(symbol, old_stop_order_id)
                        if cancel_success:
                            logger.debug(f"Γ£ô Replaced stop order: {old_stop_order_id} ΓåÆ {new_stop_order.get('order_id')}")
                        else:
                            logger.warning(f"ΓÜá New divergence stop active but failed to cancel old stop {old_stop_order_id}")
                
                if new_stop_order:
                    position["stop_price"] = new_stop
                    if new_stop_order.get("order_id"):
                        position["stop_order_id"] = new_stop_order.get("order_id")
    
    # EIGHTH - Time-decay tightening (last priority, gradual adjustment)
    check_time_decay_tightening(symbol, bar_time)
    
    # Check for VWAP target hit (mean reversion complete)
    target_hit, price = check_reversal_signal(symbol, current_bar, position)
    if target_hit:
        execute_exit(symbol, price, "target_hit")
        return


def get_flatten_price(symbol: str, side: str, current_price: float) -> float:
    """
    Calculate flatten price with buffer to avoid worst price.
    Places limit order N ticks worse than current bid/offer.
    
    Args:
        symbol: Instrument symbol
        side: Position side ('long' or 'short')
        current_price: Current market price
    
    Returns:
        Adjusted price for flatten order
    """
    tick_size = CONFIG["tick_size"]
    buffer_ticks = CONFIG["flatten_buffer_ticks"]
    
    if side == "long":
        # Selling, so go buffer ticks below current price (worse than bid)
        flatten_price = current_price - (buffer_ticks * tick_size)
    else:  # short
        # Buying, so go buffer ticks above current price (worse than offer)
        flatten_price = current_price + (buffer_ticks * tick_size)
    
    return round_to_tick(flatten_price)


def calculate_pnl(position: Dict[str, Any], exit_price: float, symbol: str = None) -> Tuple[float, float]:
    """
    Calculate profit/loss for the exit - PRODUCTION READY with slippage and commissions.
    
    Args:
        position: Position dictionary
        exit_price: Exit price (before slippage)
        symbol: Instrument symbol for accurate tick calculations (optional, falls back to CONFIG)
    
    Returns:
        Tuple of (ticks, pnl_dollars after all costs)
    """
    entry_price = position["entry_price"]
    contracts = position["quantity"]
    # Use symbol-specific tick values for accurate P&L calculation across different instruments
    if symbol:
        tick_size, tick_value = get_symbol_tick_specs(symbol)
    else:
        # Fallback to CONFIG for backward compatibility
        tick_size = CONFIG["tick_size"]
        tick_value = CONFIG["tick_value"]
    
    actual_exit_price = exit_price
    
    # ===== CRITICAL FIX #6: Exit Slippage Tracking and Alerts =====
    # Track slippage impact separately for critical exits (stops)
    if exit_price != actual_exit_price:
        slippage_amount = abs(exit_price - actual_exit_price)
        slippage_ticks_actual = slippage_amount / tick_size
        slippage_cost_dollars = slippage_ticks_actual * tick_value * contracts
        
        # Check if this is a stop loss exit (higher slippage risk)
        is_stop_exit = position.get("stop_price") and abs(exit_price - position["stop_price"]) < (2 * tick_size)
        
        # Get alert threshold from config
        slippage_alert_threshold = CONFIG.get("exit_slippage_alert_ticks", 2)
        
        if is_stop_exit and slippage_ticks_actual > slippage_alert_threshold:
            # CRITICAL: Stop loss slippage exceeds threshold!
            logger.warning("=" * 80)
            logger.warning("[WARN] CRITICAL: HIGH STOP LOSS SLIPPAGE DETECTED!")
            logger.warning("=" * 80)
            logger.warning(f"  Expected Exit: ${exit_price:.2f}")
            logger.warning(f"  Actual Fill: ${actual_exit_price:.2f}")
            logger.warning(f"  Slippage: {slippage_ticks_actual:.1f} ticks (${slippage_cost_dollars:.2f})")
            logger.warning(f"  Risk Taken: 4 ticks, Actual Loss: {slippage_ticks_actual + 4:.1f} ticks")
            logger.warning(f"  [WARN] Stop losses experiencing >{slippage_alert_threshold} tick slippage - consider tighter stops or avoid fast markets")
            logger.warning("=" * 80)
            
            # Send high slippage alert
            try:
                notifier = get_notifier()
                notifier.send_error_alert(
                    error_message=f"HIGH SLIPPAGE on stop loss! {slippage_ticks_actual:.1f} ticks (${slippage_cost_dollars:.2f}). Expected: ${exit_price:.2f}, Actual: ${actual_exit_price:.2f}. Fast market conditions detected.",
                    error_type="High Slippage Warning"
                )
            except Exception as e:
                logger.debug(f"Failed to send high slippage alert: {e}")
        elif slippage_ticks_actual > 0:
            # Normal slippage logging
            logger.info(f"  Exit Slippage: {slippage_ticks_actual:.1f} ticks (${slippage_cost_dollars:.2f})")
            logger.info(f"  Expected: ${exit_price:.2f}, Actual: ${actual_exit_price:.2f}")
    
    # Calculate gross P&L
    if position["side"] == "long":
        price_change = actual_exit_price - entry_price
    else:
        price_change = entry_price - actual_exit_price
    
    ticks = price_change / tick_size
    gross_pnl = ticks * tick_value * contracts
    
    # CONFIGURABLE CAP: Maximum loss (protects against slippage/gaps)
    max_stop_loss = CONFIG.get("max_stop_loss_dollars", DEFAULT_MAX_STOP_LOSS_DOLLARS)
    if gross_pnl < -max_stop_loss:
        logger.warning(f"⚠️ Loss capped: ${gross_pnl:.2f} -> $-{max_stop_loss:.2f} (max loss protection)")
        gross_pnl = -max_stop_loss
    
    # Deduct commissions
    commission = CONFIG.get("commission_per_contract", 0.0) * contracts
    net_pnl = gross_pnl - commission
    
    return ticks, net_pnl


def update_position_statistics(symbol: str, position: Dict[str, Any], exit_time: datetime, 
                               reason: str, time_based_reasons: List[str]) -> None:
    """
    Update position duration and statistics.
    
    Args:
        symbol: Instrument symbol
        position: Position dictionary
        exit_time: Exit timestamp
        reason: Exit reason
        time_based_reasons: List of time-based exit reasons
    """
    if position["entry_time"] is None:
        return
    
    # Ensure both timestamps are timezone-aware for accurate duration calculation
    entry_time = position["entry_time"]
    
    # Calculate duration
    try:
        duration_seconds = (exit_time - entry_time).total_seconds()
        duration_minutes = duration_seconds / 60.0
        
        # Sanity check - if duration is negative or absurdly large, log error
        if duration_minutes < 0:
            logger.error(f"Invalid duration: {duration_minutes:.1f} min (negative)")
            logger.error(f"  Entry: {entry_time}")
            logger.error(f"  Exit: {exit_time}")
            return
        elif duration_minutes > 10000:  # More than ~7 days
            logger.error(f"Invalid duration: {duration_minutes:.1f} min (too large - over 7 days)")
            logger.error(f"  Entry time: {entry_time}")
            logger.error(f"  Exit time: {exit_time}")
            logger.error(f"  Entry TZ: {entry_time.tzinfo}, Exit TZ: {exit_time.tzinfo}")
            return
            
        state[symbol]["session_stats"]["trade_durations"].append(duration_minutes)
        logger.info(f"  Position Duration: {duration_minutes:.1f} minutes")
        
    except Exception as e:
        logger.error(f"Error calculating trade duration: {e}")
        logger.error(f"  Entry: {entry_time} (type: {type(entry_time)})")
        logger.error(f"  Exit: {exit_time} (type: {type(exit_time)})")
        return
    
    # Track if this was a forced flatten due to time
    if reason in time_based_reasons:
        state[symbol]["session_stats"]["force_flattened_count"] += 1
    
    # Track after-noon entries
    entry_hour = entry_time.hour
    if entry_hour >= 12:
        state[symbol]["session_stats"]["after_noon_entries"] += 1
        if reason in time_based_reasons:
            state[symbol]["session_stats"]["after_noon_force_flattened"] += 1


def handle_exit_orders(symbol: str, position: Dict[str, Any], exit_price: float, reason: str) -> None:
    """
    Handle exit order placement using intelligent bid/ask optimization.
    Requirement 9: Exit Order Optimization
    
    Args:
        symbol: Instrument symbol
        position: Position dictionary
        exit_price: Exit price
        reason: Exit reason
    """
    order_side = "SELL" if position["side"] == "long" else "BUY"
    contracts = position["quantity"]
    
    # ===== CRITICAL FIX #5: Forced Flatten with Aggressive Retries =====
    # For emergency forced flatten, use aggressive retry logic to ensure position closes
    if reason == "emergency_forced_flatten":
        logger.critical("=" * 80)
        logger.critical("FORCED FLATTEN EXECUTION - AGGRESSIVE RETRY MODE")
        logger.critical("=" * 80)
        
        # Get retry configuration
        max_attempts = CONFIG.get("forced_flatten_max_retries", 5)
        retry_backoff_base = CONFIG.get("forced_flatten_retry_backoff_base", 1)
        
        for attempt in range(1, max_attempts + 1):
            logger.critical(f"≡ƒåÿ Forced flatten attempt {attempt}/{max_attempts}")
            logger.critical(f"  Position: {position['side'].upper()} {contracts} contracts")
            logger.critical(f"  Exit Price: ${exit_price:.2f}")
            
            # Use market order for maximum urgency
            order = place_market_order(symbol, order_side, contracts)
            
            if order:
                logger.critical(f"  ✓ Order placed - Order ID: {order.get('order_id', 'N/A')}")
                
                # In backtesting, position closes immediately
                # In live trading, wait briefly and verify
                time_module.sleep(1)
                
                # Verify position actually closed or partially filled
                current_position = get_position_quantity(symbol)
                
                if current_position == 0:
                    logger.critical("=" * 80)
                    logger.critical(f"[SUCCESS] FORCED FLATTEN SUCCESSFUL (Attempt {attempt})")
                    logger.critical("=" * 80)
                    return  # SUCCESS - position fully closed
                else:
                    # Partial fill - update contracts remaining and retry
                    contracts_filled = contracts - abs(current_position)
                    if contracts_filled > 0:
                        logger.warning(f"  [PARTIAL FILL] {contracts_filled} of {contracts} contracts filled")
                        logger.warning(f"  [REMAINING] {abs(current_position)} contracts still open - retrying...")
                        contracts = abs(current_position)  # Update quantity for next attempt
                    else:
                        logger.error(f"  [WARN] Position still shows {current_position} contracts - no fill detected")
            else:
                logger.error(f"  [FAIL] Order placement FAILED on attempt {attempt}")
            
            # Failed - retry with increasing urgency
            if attempt < max_attempts:
                wait_time = attempt * retry_backoff_base  # 1s, 2s, 3s, 4s delays
                logger.error(f"  Retrying in {wait_time} seconds with increased urgency...")
                time_module.sleep(wait_time)
        
        # ALL RETRIES FAILED - CRITICAL ALERT!
        logger.critical("=" * 80)
        logger.critical(f"[!!!] FORCED FLATTEN FAILED AFTER {max_attempts} ATTEMPTS! [!!!]")
        logger.critical("=" * 80)
        logger.critical(f"  Position: {position['side'].upper()} {contracts} contracts")
        logger.critical(f"  Entry: ${position['entry_price']:.2f}, Current: ${exit_price:.2f}")
        logger.critical(f"  Symbol: {symbol}")
        logger.critical("  [WARN] POSITION STILL OPEN - MANUAL INTERVENTION REQUIRED IMMEDIATELY!")
        logger.critical("  [WARN] OVERNIGHT RISK - CONTACT BROKER TO FORCE CLOSE!")
        logger.critical("=" * 80)
        
        # Send CRITICAL flatten failure alert
        try:
            unrealized_pnl = (exit_price - position['entry_price']) * contracts * CONFIG['tick_value'] / CONFIG['tick_size']
            if position['side'] == 'short':
                unrealized_pnl = -unrealized_pnl
            notifier = get_notifier()
            notifier.send_error_alert(
                error_message=f"≡ƒåÿ CRITICAL: FLATTEN FAILED after {max_attempts} attempts! Position: {position['side'].upper()} {contracts} {symbol}. Entry: ${position['entry_price']:.2f}. P&L: ${unrealized_pnl:.2f}. MANUAL INTERVENTION REQUIRED!",
                error_type="FLATTEN FAILED - URGENT"
            )
        except Exception as e:
            logger.debug(f"Failed to send flatten failure alert: {e}")
        
        return  # Cannot continue - manual intervention needed
    
    # Normal exit handling (non-forced-flatten)
    # Determine exit type based on reason
    exit_type_map = {
        "stop_loss": "stop",
        "time_based_profit_take": "time_flatten",
        "time_based_loss_cut": "time_flatten",
        "signal_reversal": "partial",
        "early_profit_lock": "partial",
        "trailing_stop_failure_emergency": "emergency"  # Added for trailing stop fix
    }
    exit_type = exit_type_map.get(reason, "stop")  # Default to stop for safety
    
    # Use bid/ask manager for intelligent exit routing
    if bid_ask_manager is not None:
        try:
            # Determine urgency based on volume surge and exit reason
            urgency = "normal"
            
            # Check for volume surge (potential reversal)
            is_surging, surge_ratio = detect_volume_surge(symbol)
            if is_surging:
                urgency = "high"
                logger.warning(f"ΓÜí Volume surge detected ({surge_ratio:.1f}x) - using high urgency exit")
            
            # Override urgency for critical exits
            if reason in ["stop_loss", "proactive_stop", "signal_reversal"]:
                urgency = "high"
                logger.info(f"Critical exit ({reason}) - using high urgency")
            elif reason in ["emergency_forced_flatten", "time_based_loss_cut"]:
                urgency = "high"
                logger.info(f"Time-critical exit ({reason}) - using high urgency")
            
            strategy = bid_ask_manager.get_exit_order_strategy(
                exit_type=exit_type,
                symbol=symbol,
                side=position["side"],
                urgency=urgency
            )
            
            logger.info(f"Exit Strategy: {strategy['order_type']} - {strategy['reason']}")
            
            if strategy['order_type'] == 'passive':
                # Try passive exit to collect spread
                limit_price = strategy['limit_price']
                logger.info(f"Passive exit at ${limit_price:.2f} (collecting spread)")
                order = place_limit_order(symbol, order_side, contracts, limit_price)
                
                if order and strategy.get('timeout', 0) > 0:
                    # Wait for fill with timeout
                    time_module.sleep(strategy['timeout'])
                    
                    # Check if filled (handle partial fills)
                    current_position = abs(get_position_quantity(symbol))
                    if current_position == 0:
                        logger.info("✓ Passive exit filled completely")
                        return
                    elif current_position < abs(position["quantity"]):
                        # Partial fill detected
                        filled = abs(position["quantity"]) - current_position
                        logger.warning(f"  [PARTIAL FILL] {filled} of {contracts} contracts filled")
                        logger.warning(f"  [REMAINING] {current_position} contracts - using aggressive for remainder")
                        remaining_contracts = current_position
                        # Place aggressive order for remaining
                        if 'fallback_price' in strategy:
                            order = place_limit_order(symbol, order_side, remaining_contracts, strategy['fallback_price'])
                        else:
                            order = place_market_order(symbol, order_side, remaining_contracts)
                    else:
                        # Not filled at all, use fallback
                        logger.warning("✗ Passive exit not filled, using aggressive")
                        if 'fallback_price' in strategy:
                            order = place_limit_order(symbol, order_side, contracts, strategy['fallback_price'])
                        else:
                            order = place_market_order(symbol, order_side, contracts)
                else:
                    # No timeout or order failed, go aggressive
                    order = place_market_order(symbol, order_side, contracts)
            else:
                # Aggressive exit
                if 'limit_price' in strategy:
                    logger.info(f"Aggressive exit at ${strategy['limit_price']:.2f}")
                    order = place_limit_order(symbol, order_side, contracts, strategy['limit_price'])
                else:
                    order = place_market_order(symbol, order_side, contracts)
            
            # Verify final fill for aggressive orders
            if order:
                logger.info(f"Exit order placed: {order.get('order_id')}")
                
                # For aggressive/market orders, verify fill in live mode
                if not is_backtest_mode() and strategy.get('order_type') == 'aggressive':
                    time_module.sleep(1)  # Brief wait for fill
                    final_position = abs(get_position_quantity(symbol))
                    if final_position > 0:
                        logger.warning(f"  [WARN] Aggressive exit left {final_position} contracts unfilled")
                        logger.warning(f"  Retrying with remaining {final_position} contracts")
                        # Retry once more with market order
                        retry_order = place_market_order(symbol, order_side, final_position)
                        if retry_order:
                            logger.info(f"  Retry order placed: {retry_order.get('order_id')}")
                return
                
        except Exception as e:
            logger.error(f"Error using bid/ask exit optimization: {e}")
            logger.info("Falling back to traditional exit")
    
    # Fallback to traditional exit logic
    is_emergency_exit = reason in [
        "time_based_profit_take", 
        "time_based_loss_cut", "emergency_forced_flatten"
    ]
    
    if is_emergency_exit:
        logger.info("Using aggressive limit order strategy for emergency exit")
        execute_flatten_with_limit_orders(symbol, order_side, contracts, exit_price, reason)
    else:
        # Normal exit - use market order with partial fill retry
        logger.info(f"Placing market exit order for {contracts} contracts")
        order = place_market_order(symbol, order_side, contracts)
        if order:
            logger.info(f"Exit order placed: {order.get('order_id')}")
            
            # Verify fill in live mode and retry if needed
            if not is_backtest_mode():
                time_module.sleep(1)  # Wait for fill
                remaining = abs(get_position_quantity(symbol))
                if remaining > 0:
                    logger.warning(f"  [PARTIAL FILL] {remaining} contracts still open")
                    logger.info(f"  Retrying with market order for {remaining} contracts")
                    retry_order = place_market_order(symbol, order_side, remaining)
                    if retry_order:
                        logger.info(f"  Retry order placed: {retry_order.get('order_id')}")


def execute_exit(symbol: str, exit_price: float, reason: str) -> None:
    """
    Execute exit order and update P&L.
    Coordinates exit handling through helper functions.
    
    Args:
        symbol: Instrument symbol
        exit_price: Exit price
        reason: Reason for exit (stop_loss, target_reached, signal_reversal, etc.)
    """
    
    position = state[symbol]["position"]
    
    if not position["active"]:
        return
    
    # Cancel stop order BEFORE exiting position (prevents orphaned stop orders)
    stop_order_id = position.get("stop_order_id")
    if stop_order_id:
        cancel_success = cancel_order(symbol, stop_order_id)
        if cancel_success:
            logger.debug(f"Γ£ô Cancelled stop order {stop_order_id} before exit")
        else:
            logger.warning(f"ΓÜá Failed to cancel stop order {stop_order_id} - may remain active!")
    
    exit_time = get_current_time()  # Use get_current_time() for backtest compatibility
    
    logger.info(SEPARATOR_LINE)
    logger.info(f"EXITING {position['side'].upper()} POSITION")
    logger.info(f"  Reason: {reason.replace('_', ' ').title()}")
    logger.info(f"  Time: {exit_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Calculate P&L with symbol-specific tick values
    ticks, pnl = calculate_pnl(position, exit_price, symbol)
    
    logger.info(f"  Entry: ${position['entry_price']:.2f}, Exit: ${exit_price:.2f}")
    logger.info(f"  Ticks: {ticks:+.1f}, P&L: ${pnl:+.2f}")
    
    # Send trade exit alert
    try:
        notifier = get_notifier()
        notifier.send_trade_alert(
            trade_type="EXIT",
            symbol=symbol,
            price=exit_price,
            contracts=position['quantity'],
            side="LONG" if position['side'] == 'long' else "SHORT"
        )
    except Exception as e:
        logger.debug(f"Failed to send exit alert: {e}")
    
    # REINFORCEMENT LEARNING - Record outcome to cloud API (shared learning pool)
    try:
        # Check if we have the entry market state stored (NEW FORMAT)
        if "entry_market_state" in state[symbol]:
            market_state = state[symbol]["entry_market_state"]
            
            # Calculate trade duration in minutes
            entry_time = position.get("entry_time")
            duration_minutes = 0
            if entry_time:
                duration = exit_time - entry_time
                duration_minutes = duration.total_seconds() / 60
            
            # Calculate MFE (Max Favorable Excursion) and MAE (Max Adverse Excursion)
            entry_price = position.get("entry_price", exit_price)
            # Use symbol-specific tick values for accurate P&L calculation across different instruments
            tick_size, tick_value = get_symbol_tick_specs(symbol)
            
            # Get from position tracking (if available)
            if position["side"] == "long":
                highest_price = position.get("highest_price_reached", exit_price)
                lowest_price = position.get("lowest_price_reached", exit_price)
                mfe_ticks = (highest_price - entry_price) / tick_size
                mae_ticks = (entry_price - lowest_price) / tick_size
            else:  # short
                highest_price = position.get("highest_price_reached", exit_price)
                lowest_price = position.get("lowest_price_reached", exit_price)
                mfe_ticks = (entry_price - lowest_price) / tick_size
                mae_ticks = (highest_price - entry_price) / tick_size
            
            mfe = mfe_ticks * tick_value
            mae = mae_ticks * tick_value
            
            # Get the side from position
            trade_side = position.get("side", "unknown")
            
            # Get execution quality metrics for RL learning
            order_type_used = position.get("order_type_used", "unknown")
            
            # Calculate entry slippage if we have the data (tick_size from get_symbol_tick_specs() on line ~6718)
            entry_slippage_ticks = 0
            if position.get("actual_entry_price") and position.get("original_entry_price"):
                entry_slippage = abs(position.get("actual_entry_price", 0) - position.get("original_entry_price", 0))
                entry_slippage_ticks = entry_slippage / tick_size if tick_size > 0 else 0
            
            # Record market state + outcomes to local RL brain (backtest) or cloud (live)
            save_trade_experience(
                rl_state=market_state,  # Market state (flat structure)
                side=trade_side,  # Trade direction (for cloud API)
                pnl=pnl,
                duration_minutes=duration_minutes,
                execution_data={
                    "mfe": mfe,
                    "mae": mae,
                    "exit_reason": reason,  # CRITICAL: How trade closed
                    "order_type_used": order_type_used,  # IMPORTANT: Order type for execution optimization
                    "entry_slippage_ticks": entry_slippage_ticks  # IMPORTANT: Slippage tracking
                }
            )
            
            logger.info(f"πΎ [EXPERIENCE] Recorded: ${pnl:+.2f} in {duration_minutes:.1f}min | MFE: ${mfe:.2f}, MAE: ${mae:.2f}")
            
            # Clean up state
            if "entry_market_state" in state[symbol]:
                del state[symbol]["entry_market_state"]
            if "entry_rl_confidence" in state[symbol]:
                del state[symbol]["entry_rl_confidence"]
        
    except Exception as e:
        logger.debug(f"RL outcome recording failed: {e}")
    
    # Store exit reason in state for backtest tracking (position gets reset)
    state[symbol]["last_exit_reason"] = reason
    position["exit_reason"] = reason
    
    # Log time-based exits with detailed audit trail
    time_based_reasons = [
        "emergency_forced_flatten"
    ]
    
    if reason in time_based_reasons:
        exit_details = {
            "exit_price": f"${exit_price:.2f}",
            "pnl": f"${pnl:+.2f}",
            "side": position["side"],
            "quantity": position["quantity"],
            "entry_price": f"${position['entry_price']:.2f}"
        }
        
        reason_descriptions = {
            "emergency_forced_flatten": "4:45 PM flatten before maintenance"
        }
        
        log_time_based_action(
            "position_closed",
            reason_descriptions.get(reason, reason),
            exit_details
        )
    
    # Handle exit orders
    handle_exit_orders(symbol, position, exit_price, reason)
    
    # Update daily P&L
    state[symbol]["daily_pnl"] += pnl
    
    # Update position statistics
    update_position_statistics(symbol, position, exit_time, reason, time_based_reasons)
    
    # Update session statistics
    update_session_stats(symbol, pnl)
    
    logger.info(f"Daily P&L: ${state[symbol]['daily_pnl']:+.2f}")
    logger.info(SEPARATOR_LINE)
    
    # Update session state for cross-session awareness
    
    # Write trade summary to file for GUI display
    try:
        import json
        pnl_percent = (pnl / (position['entry_price'] * position['quantity'] * CONFIG.get('tick_value', 12.50))) * 100 if position['entry_price'] and position['quantity'] else 0
        
        trade_summary = {
            'symbol': symbol,
            'direction': position['side'].upper(),
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'contracts': position['quantity'],
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'timestamp': exit_time.isoformat(),
            'reason': reason
        }
        
        # Write to file
        summary_file = get_data_file_path('data/trade_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(trade_summary, f, indent=2)
        
        # Also write daily stats
        wins = len([t for t in state[symbol].get('trade_history', []) if t.get('pnl', 0) > 0])
        losses = len([t for t in state[symbol].get('trade_history', []) if t.get('pnl', 0) < 0])
        
        daily_summary = {
            'total_pnl': state[symbol]['daily_pnl'],
            'wins': wins,
            'losses': losses,
            'account_balance': CONFIG.get('account_size', 50000) + state[symbol]['daily_pnl'],
            'timestamp': exit_time.isoformat()
        }
        
        daily_file = get_data_file_path('daily_summary.json')
        with open(daily_file, 'w') as f:
            json.dump(daily_summary, f, indent=2)
            
        logger.debug(f"[GUI] Trade summary written to {summary_file}")
    except Exception as e:
        logger.debug(f"[GUI] Failed to write trade summary: {e}")
    
    # Reset position tracking
    state[symbol]["position"] = {
        "active": False,
        "side": None,
        "quantity": 0,
        "entry_price": None,
        "stop_price": None,
        "entry_time": None,
        # Advanced Exit Management - Breakeven State
        "breakeven_active": False,
        "original_stop_price": None,
        "breakeven_activated_time": None,
        # Advanced Exit Management - Trailing Stop State
        "trailing_stop_active": False,
        "trailing_stop_price": None,
        "highest_price_reached": None,
        "lowest_price_reached": None,
        "trailing_activated_time": None,
        # Advanced Exit Management - Time-Decay State
        "time_decay_50_triggered": False,
        "time_decay_75_triggered": False,
        "time_decay_90_triggered": False,
        "original_stop_distance_ticks": None,
        "current_stop_distance_ticks": None,
        # Advanced Exit Management - Partial Exit State
        "partial_exit_1_completed": False,
        "partial_exit_2_completed": False,
        "partial_exit_3_completed": False,
        "original_quantity": 0,
        "remaining_quantity": 0,
        "partial_exit_history": [],
        # Advanced Exit Management - General
        "initial_risk_ticks": None,
    }
    
    # CRITICAL: IMMEDIATELY save state to disk - position is now FLAT
    save_position_state(symbol)
    
    # Check if we're in license grace period and position just closed
    if bot_status.get("license_grace_period", False):
        logger.critical("=" * 70)
        logger.critical("")
        logger.critical("  LICENSE EXPIRED - Position Safely Closed")
        logger.critical("")
        logger.critical(f"  Final P&L: ${pnl:+.2f}")
        logger.critical(f"  Exit: {reason}")
        logger.critical("")
        logger.critical("  Your license has expired.")
        logger.critical("  Please renew your license to continue trading.")
        logger.critical("")
        logger.critical("  Contact: support@quotrading.com")
        logger.critical("")
        logger.critical("=" * 70)
        
        # End grace period
        bot_status["license_grace_period"] = False
        bot_status["trading_enabled"] = False
        bot_status["emergency_stop"] = True
        bot_status["stop_reason"] = "License expired - grace period ended after position close"
        
        # Send notification
        try:
            notifier = get_notifier()
            notifier.send_error_alert(
                error_message=f"🔒 TRADING STOPPED - Grace Period Ended\n\n"
                             f"Your license expired and the active position has now closed safely.\n"
                             f"Final P&L: ${pnl:+.2f}\n"
                             f"Exit Reason: {reason}\n\n"
                             f"Trading is now stopped. Please renew your license to continue.",
                error_type="License Expired - Grace Period Ended"
            )
        except Exception as e:
            pass  # Silent
        
        # Disconnect broker cleanly
        logger.critical("Disconnecting from broker...")
        logger.critical("LICENSE EXPIRED - Stopping all trading and market data")
        try:
            global broker
            if broker is not None:
                broker.disconnect()
                logger.critical("Websocket disconnected - No data streaming")
        except Exception as e:
            pass  # Silent disconnect
        
        # Bot stays ON but IDLE - never exits unless user presses Ctrl+C
        logger.critical("Bot will remain ON but IDLE (no trading)")
        logger.critical("LICENSE EXPIRED - Please renew your license")
        logger.critical("Press Ctrl+C to stop bot")
    logger.info("  Γ£ô Position state saved to disk (FLAT)")


def calculate_aggressive_price(base_price: float, order_side: str, attempt: int) -> float:
    """
    Calculate increasingly aggressive limit price based on attempt number.
    
    Args:
        base_price: Base price for limit orders
        order_side: 'BUY' or 'SELL'
        attempt: Current attempt number (1-indexed)
    
    Returns:
        Aggressive limit price
    """
    tick_size = CONFIG["tick_size"]
    ticks_aggressive = attempt
    
    if order_side == "SELL":
        # Selling - go below bid
        limit_price = base_price - (ticks_aggressive * tick_size)
    else:  # BUY
        # Buying - go above offer
        limit_price = base_price + (ticks_aggressive * tick_size)
    
    return round_to_tick(limit_price)


def wait_for_fill(symbol: str, attempt: int, max_attempts: int) -> int:
    """
    Wait for order to fill and return current position quantity.
    
    Args:
        symbol: Instrument symbol
        attempt: Current attempt number
        max_attempts: Maximum number of attempts
    
    Returns:
        Current position quantity (0 if fully closed)
    """
    import time as time_module
    
    if attempt < max_attempts:
        wait_seconds = 5 if attempt < 5 else 2  # Shorter waits as we get more urgent
        logger.debug(f"Waiting {wait_seconds} seconds for fill...")
        time_module.sleep(wait_seconds)
    
    return get_position_quantity(symbol)


def handle_partial_fill(current_qty: int, contracts: int, attempt: int) -> int:
    """
    Handle partial fill and return remaining contracts.
    
    Args:
        current_qty: Current position quantity from broker
        contracts: Original number of contracts
        attempt: Current attempt number
    
    Returns:
        Number of contracts still remaining
    """
    if current_qty == 0:
        logger.info("Position fully closed")
        return 0
    else:
        filled_contracts = contracts - abs(current_qty)
        if filled_contracts > 0:
            logger.warning(f"Partial fill: {filled_contracts} of {contracts} filled, {abs(current_qty)} remaining")
            return abs(current_qty)
        else:
            logger.warning(f"No fill on attempt {attempt}, retrying with more aggressive price")
            return abs(current_qty)


def execute_flatten_with_limit_orders(symbol: str, order_side: str, contracts: int, 
                                       base_price: float, reason: str) -> None:
    """
    Execute flatten using aggressive limit orders with partial fill handling.
    Main orchestration function that coordinates helpers.
    
    Args:
        symbol: Instrument symbol
        order_side: 'BUY' or 'SELL'
        contracts: Number of contracts to close
        base_price: Base price for limit orders
        reason: Exit reason
    """
    remaining_contracts = contracts
    attempt = 0
    max_attempts = 10
    
    while remaining_contracts > 0 and attempt < max_attempts:
        attempt += 1
        
        # Calculate aggressive limit price
        limit_price = calculate_aggressive_price(base_price, order_side, attempt)
        
        logger.info(f"Flatten attempt {attempt}/{max_attempts}: {order_side} {remaining_contracts} @ {limit_price:.2f}")
        
        # Place aggressive limit order
        order = place_limit_order(symbol, order_side, remaining_contracts, limit_price)
        
        if order:
            logger.info(f"Flatten limit order placed: {order.get('order_id')}")
        
        # Wait and check for fills
        if attempt < max_attempts:
            current_qty = wait_for_fill(symbol, attempt, max_attempts)
            remaining_contracts = handle_partial_fill(current_qty, contracts, attempt)
            
            if remaining_contracts == 0:
                break
        else:
            # Final attempt - at market price
            logger.critical(f"Final attempt - placing market order for remaining {remaining_contracts}")
            order = place_market_order(symbol, order_side, remaining_contracts)
            if order:
                logger.info(f"Emergency market order placed: {order.get('order_id')}")
            break
    
    if remaining_contracts > 0:
        logger.error(f"Failed to fully flatten position - {remaining_contracts} contracts may remain")
    else:
        logger.info(f"Successfully flattened {contracts} contracts using aggressive limit orders")


# ============================================================================
# PHASE ELEVEN: Daily Reset Logic
# ============================================================================

def check_vwap_reset(symbol: str, current_time: datetime) -> None:
    """
    Check if VWAP should reset at 6:00 PM ET (futures market day start).
    For 24/5 trading: VWAP resets at 6:00 PM ET when futures trading day begins.
    
    Args:
        symbol: Instrument symbol
        current_time: Current datetime in Eastern Time
    """
    current_date = current_time.date()
    vwap_reset_time = datetime_time(18, 0)  # 6:00 PM ET - futures trading day starts
    
    # Check if we've crossed 6:00 PM ET on a new day
    if state[symbol]["vwap_day"] is None:
        # First run - initialize VWAP day
        state[symbol]["vwap_day"] = current_date
        pass  # Silent - VWAP day initialization
        return
    
    # If it's a new day and we're past 6:00 PM ET, reset VWAP
    # OR if it's the same calendar day but we just crossed 6:00 PM ET
    last_reset_date = state[symbol]["vwap_day"]
    crossed_reset_time = current_time.time() >= vwap_reset_time
    
    # New trading day starts at 6:00 PM ET, so check if we've moved to a new VWAP session
    if crossed_reset_time and last_reset_date != current_date:
        perform_vwap_reset(symbol, current_date, current_time)


def perform_vwap_reset(symbol: str, new_date: Any, reset_time: datetime) -> None:
    """
    DISABLED: VWAP reset function - not called per strategy requirements.
    Strategy only needs accurate VWAP value for current bar.
    """
    pass  # Function disabled - VWAP continues calculating without reset


def check_daily_reset(symbol: str, current_time: datetime) -> None:
    """
    Check if we've crossed into a new trading day and reset daily counters.
    For 24/5 trading: Resets at 6:00 PM ET (futures trading day start).
    
    Args:
        symbol: Instrument symbol
        current_time: Current datetime in Eastern Time
    """
    current_date = current_time.date()
    vwap_reset_time = datetime_time(18, 0)  # 6:00 PM ET - futures trading day starts
    
    # If we have a trading day stored and it's different from current date
    if state[symbol]["trading_day"] is not None:
        if state[symbol]["trading_day"] != current_date:
            # Reset daily counters at 6:00 PM ET (same as VWAP reset)
            if current_time.time() >= vwap_reset_time:
                perform_daily_reset(symbol, current_date)
    else:
        # First run - initialize trading day
        state[symbol]["trading_day"] = current_date
        pass  # Silent - trading day initialization


def perform_daily_reset(symbol: str, new_date: Any) -> None:
    """
    Perform the actual daily reset operations.
    Resets daily counters and session stats.
    VWAP reset is handled separately by perform_vwap_reset.
    
    Args:
        symbol: Instrument symbol
        new_date: The new trading date
    """
    logger.info(SEPARATOR_LINE)
    logger.info(f"DAILY RESET - New Trading Day: {new_date}")
    logger.info(SEPARATOR_LINE)
    
    # Log session summary before reset
    log_session_summary(symbol)
    
    # Reset daily counters
    state[symbol]["daily_trade_count"] = 0
    state[symbol]["daily_pnl"] = 0.0
    state[symbol]["trading_day"] = new_date
    state[symbol]["loss_limit_alerted"] = False  # Reset alert flag
    
    # Reset session stats
    state[symbol]["session_stats"] = {
        "trades": [],
        "win_count": 0,
        "loss_count": 0,
        "total_pnl": 0.0,
        "largest_win": 0.0,
        "largest_loss": 0.0,
        "pnl_variance": 0.0,
        # Phase 20: Position duration statistics
        "trade_durations": [],  # List of durations in minutes
        "force_flattened_count": 0,  # Trades closed due to time limit
        "after_noon_entries": 0,  # Entries after 12 PM
        "after_noon_force_flattened": 0  # After-noon entries force-closed
    }
    
    # Re-enable trading if it was stopped for any daily limit reason
    # "daily_loss_limit" = specific daily loss limit breached
    # "daily_limits_reached" = approaching failure without recovery mode
    if bot_status["stop_reason"] in ["daily_loss_limit", "daily_limits_reached"]:
        bot_status["trading_enabled"] = True
        bot_status["stop_reason"] = None
        logger.info("[OK] Trading re-enabled for new day - daily limits reset")
    
    logger.info("Daily reset complete - Ready for trading")
    logger.info("  [OK] Daily P&L reset to $0.00")
    logger.info("  [OK] Trade count reset to 0")
    logger.info("  [OK] VWAP bands will recalculate from live data")
    logger.info(SEPARATOR_LINE)


# ============================================================================
# PHASE TWELVE: Safety Mechanisms
# ============================================================================

def check_daily_loss_limit(symbol: str) -> Tuple[bool, Optional[str]]:
    """
    Check if daily loss limit has been exceeded.
    Loss limit is adjusted by current profit - if trader made profit and limit is set by user,
    they can lose [user's limit + profit] before hitting the limit.
    
    Args:
        symbol: Instrument symbol
    
    Returns:
        Tuple of (is_safe, reason)
    """
    current_pnl = state[symbol]["daily_pnl"]
    base_loss_limit = CONFIG["daily_loss_limit"]
    
    # Adjust loss limit by adding current profit (if positive)
    effective_loss_limit = base_loss_limit + max(0, current_pnl)
    
    if state[symbol]["daily_pnl"] <= -effective_loss_limit:
        if bot_status["trading_enabled"]:
            logger.critical(f"DAILY LOSS LIMIT BREACHED: ${state[symbol]['daily_pnl']:.2f}")
            logger.critical(f"  Base limit: ${base_loss_limit:.2f}, Profit cushion: ${max(0, current_pnl):.2f}, Effective limit: ${effective_loss_limit:.2f}")
            logger.critical("Trading STOPPED for the day")
            bot_status["trading_enabled"] = False
            bot_status["stop_reason"] = "daily_loss_limit"
            
            # Send daily loss limit breach alert
            try:
                notifier = get_notifier()
                notifier.send_error_alert(
                    error_message=f"DAILY LOSS LIMIT HIT! Trading stopped. Loss: ${state[symbol]['daily_pnl']:.2f} / Limit: ${-effective_loss_limit:.2f} (base ${base_loss_limit:.2f} + ${max(0, current_pnl):.2f} profit cushion)",
                    error_type="Daily Loss Limit Breached"
                )
            except Exception as e:
                logger.debug(f"Failed to send loss limit breach alert: {e}")
        return False, "Daily loss limit exceeded"
    return True, None




def check_approaching_failure(symbol: str) -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Check if bot is approaching daily loss limit.
    
    Args:
        symbol: Instrument symbol
    
    Returns:
        Tuple of (is_approaching, reason, severity_level)
        - is_approaching: True if at DAILY_LOSS_APPROACHING_THRESHOLD (80%) or more of daily loss limit
        - reason: Description of what limit is being approached
        - severity_level: 0.0-1.0 indicating how close to failure (0.8 = at 80%, 1.0 = at 100%)
    """
    daily_loss_limit = CONFIG.get("daily_loss_limit")
    if daily_loss_limit is None or daily_loss_limit <= 0:
        return False, None, 0.0
    
    if state[symbol]["daily_pnl"] <= -daily_loss_limit * DAILY_LOSS_APPROACHING_THRESHOLD:
        daily_loss_severity = abs(state[symbol]["daily_pnl"]) / daily_loss_limit
        reason = f"Daily loss at {daily_loss_severity*100:.1f}% of limit (${state[symbol]['daily_pnl']:.2f}/${-daily_loss_limit:.2f})"
        
        # Send warning alert when approaching limit (80%)
        try:
            notifier = get_notifier()
            notifier.send_daily_limit_warning(
                current_loss=state[symbol]["daily_pnl"],
                limit=daily_loss_limit
            )
        except Exception as e:
            logger.debug(f"Failed to send daily limit warning: {e}")
        
        return True, reason, daily_loss_severity
    
    return False, None, 0.0


def check_tick_timeout(current_time: datetime) -> Tuple[bool, Optional[str]]:
    """
    Check if data feed has timed out.
    
    Args:
        current_time: Current datetime in Eastern Time
    
    Returns:
        Tuple of (is_safe, reason)
    """
    if bot_status["last_tick_time"] is not None:
        trading_state = get_trading_state(current_time)
        # Check for tick timeout during any active trading state (not before_open or closed)
        if trading_state not in ["before_open", "closed"]:
            time_since_tick = (current_time - bot_status["last_tick_time"]).total_seconds()
            if time_since_tick > CONFIG["tick_timeout_seconds"]:
                logger.error(f"DATA FEED ISSUE: No tick in {time_since_tick:.0f} seconds")
                logger.error("Trading paused - connection health check failed")
                bot_status["trading_enabled"] = False
                bot_status["stop_reason"] = "data_feed_timeout"
                return False, f"No tick data for {time_since_tick:.0f} seconds"
    return True, None


def check_trade_limits(current_time: datetime) -> Tuple[bool, Optional[str]]:
    """
    Check emergency stop and trading enabled status.
    CME Futures trading - uses US Eastern wall-clock time (DST-aware).
    
    Args:
        current_time: Current datetime in Eastern Time
    
    Returns:
        Tuple of (is_safe, reason)
    """
    # Check if emergency stop is active
    if bot_status["emergency_stop"]:
        return False, f"Emergency stop active: {bot_status['stop_reason']}"
    
    # Ensure current_time is in Eastern
    eastern_tz = pytz.timezone('US/Eastern')
    if current_time.tzinfo is None:
        current_time = eastern_tz.localize(current_time)
    eastern_time = current_time.astimezone(eastern_tz)
    
    # Check for weekend (Saturday + Sunday before 6:00 PM ET)
    if eastern_time.weekday() == 5:  # Saturday - always closed
        if bot_status["trading_enabled"]:
            logger.debug(f"Saturday detected - market closed")
            bot_status["trading_enabled"] = False
            bot_status["stop_reason"] = "weekend"
        return False, "Weekend - market closed"
    
    if eastern_time.weekday() == 6:  # Sunday
        if eastern_time.time() < datetime_time(18, 0):  # Before 6:00 PM ET Sunday
            if bot_status["trading_enabled"]:
                logger.debug(f"Sunday before 6:00 PM ET - market closed")
                bot_status["trading_enabled"] = False
                bot_status["stop_reason"] = "weekend"
            return False, "Weekend - market closed (opens 6:00 PM ET)"
    
    # Check for futures maintenance window (5:00-6:00 PM ET Monday-Friday)
    if eastern_time.weekday() < 5:  # Monday through Friday only
        maintenance_start = datetime_time(16, 45)  # 4:45 PM ET - Futures maintenance
        maintenance_end = datetime_time(18, 0)    # 6:00 PM ET
        if maintenance_start <= eastern_time.time() < maintenance_end:
            if bot_status["trading_enabled"]:
                logger.debug(f"Maintenance window - disabling trading")
                bot_status["trading_enabled"] = False
                bot_status["stop_reason"] = "maintenance"
            return False, "Maintenance window (5:00-6:00 PM ET)"
    
    # Re-enable trading after maintenance/weekend
    if not bot_status["trading_enabled"]:
        logger.debug(f"Re-enabling trading - market open at {eastern_time}")
        bot_status["trading_enabled"] = True
        bot_status["stop_reason"] = None
    
    return True, None


def check_safety_conditions(symbol: str) -> Tuple[bool, Optional[str]]:
    """
    Check all safety conditions before allowing trading.
    Coordinates various safety checks through helper functions.
    
    Args:
        symbol: Instrument symbol
    
    Returns:
        Tuple of (is_safe, reason) where is_safe is True if safe to trade
    """
    current_time = get_current_time()
    
    # Check if license has expired
    if bot_status.get("license_expired", False):
        # SAFETY: Grace period for active positions
        # If position is active, allow bot to manage it until closed
        # Only block NEW trades, not position management
        if symbol in state and state[symbol]["position"]["active"]:
            # Position active - allow management during grace period
            logger.debug(f"License expired but position active - managing until close")
            # Don't block - let position management continue
            return True, None
        else:
            # No position - block new trades
            reason = bot_status.get("license_expiry_reason", "License expired")
            return False, f"Trading disabled: {reason}"
    
    # Check if near expiry mode (within 2 hours of expiration)
    if bot_status.get("near_expiry_mode", False):
        # Block NEW trades but allow managing existing positions
        if symbol in state and state[symbol]["position"]["active"]:
            # Position active - allow management
            logger.debug(f"Near expiry mode but position active - managing until close")
            return True, None
        else:
            # No position - block new trades
            hours_left = bot_status.get("hours_until_expiration", 0)
            return False, f"License expires in {hours_left:.1f} hours - new trades blocked"
    
    # Check trade limits and emergency stops
    is_safe, reason = check_trade_limits(current_time)
    if not is_safe:
        return False, reason
    
    # Daily loss limit DISABLED for backtesting
    # is_safe, reason = check_daily_loss_limit(symbol)
    # if not is_safe:
    #     return False, reason
    
    # Check if approaching daily loss limit (SIMPLIFIED - no recovery mode or dynamic scaling)
    is_approaching, approach_reason, severity = check_approaching_failure(symbol)
    if is_approaching:
        # SIMPLIFIED: Just stop trading when approaching limit
        # No dynamic scaling, no recovery mode
        
        if bot_status.get("stop_reason") != "daily_limits_reached":
            logger.warning("=" * 80)
            logger.warning("ΓÜá∩╕Å APPROACHING DAILY LOSS LIMIT - STOPPING TRADING")
            logger.warning(f"Reason: {approach_reason}")
            logger.warning(f"Severity: {severity*100:.1f}%")
            logger.warning("Bot will STOP making new trades until daily reset at 6 PM ET")
            logger.warning("Bot continues running and monitoring - will resume after maintenance hour")
            logger.warning("=" * 80)
            bot_status["stop_reason"] = "daily_limits_reached"
            
            # Send limit warning alert
            try:
                notifier = get_notifier()
                notifier.send_error_alert(
                    error_message=f"Daily loss limit reached. Severity: {severity*100:.1f}%. {approach_reason}. Trading stopped.",
                    error_type="DAILY_LIMIT_REACHED"
                )
            except Exception as e:
                logger.debug(f"Failed to send daily limit alert: {e}")
        
        bot_status["trading_enabled"] = False
        
        # Close any active position when limit reached
        if state[symbol]["position"]["active"]:
            position = state[symbol]["position"]
            entry_price = position.get("entry_price", 0)
            side = position.get("side", "long")
            current_price = state[symbol]["bars"][-1]["close"] if state[symbol]["bars"] else entry_price
            
            # Calculate current P&L for the position
            if side == "long":
                position_pnl = (current_price - entry_price) * position["quantity"]
            else:
                position_pnl = (entry_price - current_price) * position["quantity"]
            
            logger.warning("=" * 80)
            logger.warning("POSITION MANAGEMENT: Closing active position due to daily limit")
            logger.warning(f"Position: {side.upper()} {position['quantity']} @ ${entry_price:.2f}")
            logger.warning(f"Current Price: ${current_price:.2f}")
            logger.warning(f"Position P&L: ${position_pnl:.2f}")
            logger.warning("Executing exit to protect account")
            logger.warning("=" * 80)
            
            # Get smart flatten price (includes buffer to ensure fill)
            flatten_price = get_flatten_price(symbol, side, current_price)
            
            # Close the position
            handle_exit_orders(symbol, position, flatten_price, "daily_limit_protection")
            
            logger.info(f"Position closed at ${flatten_price:.2f} to protect from further losses")
        
        return False, "Daily limits reached - trading stopped until next session (6 PM ET reset)"
    else:
        # Not approaching failure - clear any safety mode that was set
        if bot_status.get("stop_reason") == "daily_limits_reached":
            logger.info("=" * 80)
            logger.info("SAFE ZONE: Back to normal operation")
            logger.info("Bot has moved away from failure thresholds")
            logger.info("Resuming normal trading")
            logger.info("=" * 80)
            bot_status["trading_enabled"] = True
            bot_status["stop_reason"] = None
    
    # Check tick timeout
    is_safe, reason = check_tick_timeout(current_time)
    if not is_safe:
        return False, reason
    
    return True, None


def check_no_overnight_positions(symbol: str) -> None:
    """
    Critical safety check - ensure NO positions past 4:45 PM ET (forced flatten time).
    This prevents gap risk and prop firm evaluation issues.
    
    Args:
        symbol: Instrument symbol
    """
    if not state[symbol]["position"]["active"]:
        return  # No position, all good
    
    eastern_tz = pytz.timezone('US/Eastern')
    current_time = datetime.now(eastern_tz)
    
    # Critical: If it's past 4:45 PM ET and we still have a position, this is a SERIOUS ERROR
    if current_time.time() >= CONFIG["shutdown_time"]:
        logger.critical("=" * 70)
        logger.critical("CRITICAL ERROR: POSITION DETECTED PAST 4:45 PM ET")
        logger.critical("OVERNIGHT POSITION RISK - IMMEDIATE EMERGENCY CLOSE REQUIRED")
        logger.critical("=" * 70)
        logger.critical(f"Position: {state[symbol]['position']['side']} "
                       f"{state[symbol]['position']['quantity']} contracts")
        logger.critical(f"Entry: ${state[symbol]['position']['entry_price']:.2f}")
        logger.critical("This should NEVER happen - flatten logic failed")
        logger.critical("Manual intervention required")
        logger.critical("=" * 70)
        
        # Emergency flatten at market
        position = state[symbol]["position"]
        order_side = "SELL" if position["side"] == "long" else "BUY"
        place_market_order(symbol, order_side, position["quantity"])
        
        # Force close the position in tracking
        state[symbol]["position"]["active"] = False
        bot_status["emergency_stop"] = True
        bot_status["stop_reason"] = "overnight_position_detected"


def validate_order(symbol: str, side: str, quantity: int, entry_price: float, 
                   stop_price: float) -> Tuple[bool, Optional[str]]:
    """
    Validate order parameters before placing.
    
    Args:
        symbol: Instrument symbol
        side: 'long' or 'short'
        quantity: Number of contracts
        entry_price: Entry price
        stop_price: Stop loss price
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check quantity is positive
    if quantity <= 0:
        return False, f"Invalid quantity: {quantity}"
    
    # Check stop price is on correct side of entry
    if side == "long":
        if stop_price >= entry_price:
            return False, f"Stop price ${stop_price:.2f} must be below entry ${entry_price:.2f} for long"
    else:  # short
        if stop_price <= entry_price:
            return False, f"Stop price ${stop_price:.2f} must be above entry ${entry_price:.2f} for short"
    
    # Check we have sufficient account equity for margin
    equity = get_account_equity()
    if equity <= 0:
        return False, "Invalid account equity"
    
    # Basic margin check (simplified - actual margin requirements vary)
    # For MES, approximate initial margin is ~$1,200 per contract
    estimated_margin = quantity * 1200
    if estimated_margin > equity * 0.5:  # Don't use more than 50% of equity for margin
        return False, f"Insufficient margin: need ~${estimated_margin:.0f}, have ${equity:.2f}"
    
    return True, None


# ============================================================================
# PHASE THIRTEEN: Logging and Monitoring
# ============================================================================

def format_trade_statistics(stats: Dict[str, Any]) -> None:
    """
    Format and log basic trade statistics.
    
    Args:
        stats: Session statistics dictionary
    """
    logger.info(f"Total Trades: {len(stats['trades'])}")
    logger.info(f"Wins: {stats['win_count']}")
    logger.info(f"Losses: {stats['loss_count']}")
    
    if len(stats['trades']) > 0:
        win_rate = stats['win_count'] / len(stats['trades']) * 100
        logger.info(f"Win Rate: {win_rate:.1f}%")
    else:
        logger.info("Win Rate: N/A (no trades)")


def format_pnl_summary(stats: Dict[str, Any]) -> None:
    """
    Format and log P&L summary including Sharpe ratio.
    
    Args:
        stats: Session statistics dictionary
    """
    logger.info(f"Total P&L: ${stats['total_pnl']:+.2f}")
    logger.info(f"Largest Win: ${stats['largest_win']:+.2f}")
    logger.info(f"Largest Loss: ${stats['largest_loss']:+.2f}")
    
    # Calculate Sharpe ratio if we have variance data
    if stats['pnl_variance'] > 0 and len(stats['trades']) > 1:
        avg_pnl = stats['total_pnl'] / len(stats['trades'])
        std_dev = (stats['pnl_variance'] / (len(stats['trades']) - 1)) ** 0.5
        if std_dev > 0:
            sharpe_ratio = avg_pnl / std_dev
            logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")


def format_time_statistics(stats: Dict[str, Any]) -> None:
    """
    Format and log position duration statistics.
    
    Args:
        stats: Session statistics dictionary
    """
    if len(stats['trade_durations']) == 0:
        return
    
    logger.info(SEPARATOR_LINE)
    logger.info("POSITION DURATION ANALYSIS (Phase 20)")
    
    avg_duration = sum(stats['trade_durations']) / len(stats['trade_durations'])
    min_duration = min(stats['trade_durations'])
    max_duration = max(stats['trade_durations'])
    
    logger.info(f"Average Position Duration: {avg_duration:.1f} minutes")
    logger.info(f"Shortest Trade: {min_duration:.1f} minutes")
    logger.info(f"Longest Trade: {max_duration:.1f} minutes")
    
    # Calculate force flatten statistics
    total_trades = len(stats['trades'])
    force_flatten_pct = (stats['force_flattened_count'] / total_trades * 100) if total_trades > 0 else 0
    logger.info(f"Force Flattened: {stats['force_flattened_count']}/{total_trades} ({force_flatten_pct:.1f}%)")
    
    if force_flatten_pct > 30:
        logger.warning("  >30% force-flattened - trade duration too long for time window")
        logger.warning("   Consider: earlier entry cutoff or tighter trailing stops")
    else:
        logger.info(" <30% force-flattened - acceptable duration")
    
    # After-noon entry analysis
    if stats['after_noon_entries'] > 0:
        after_noon_flatten_pct = (stats['after_noon_force_flattened'] / 
                                  stats['after_noon_entries'] * 100)
        logger.info(f"After-Noon Entries: {stats['after_noon_entries']}")
        logger.info(f"After-Noon Force Flattened: {stats['after_noon_force_flattened']} "
                   f"({after_noon_flatten_pct:.1f}%)")
        
        if after_noon_flatten_pct > 50:
            logger.warning("  >50% of after-noon entries force-flattened")
            logger.warning("   Entry window may be too late - avg duration {:.1f} min vs time remaining"
                          .format(avg_duration))
    
    # Time compatibility analysis
    time_to_flatten_at_2pm = 165  # minutes from 2 PM to 4:45 PM
    if avg_duration > time_to_flatten_at_2pm * 0.8:
        logger.warning("  Average duration uses >80% of available time window")
        logger.warning(f"   Avg duration {avg_duration:.1f} min vs {time_to_flatten_at_2pm} min available at 2 PM")


def format_risk_metrics() -> None:
    """
    Format and log flatten mode exit analysis.
    """
    total_decisions = (bot_status["target_wait_wins"] + bot_status["target_wait_losses"] + 
                       bot_status["early_close_saves"])
    if total_decisions == 0:
        return
    
    logger.info(SEPARATOR_LINE)
    logger.info("FLATTEN MODE EXIT ANALYSIS (Phase 10)")
    logger.info(f"Trailing Stop Wins: {bot_status['target_wait_wins']}")
    logger.info(f"Trailing Stop Losses: {bot_status['target_wait_losses']}")
    logger.info(f"Early Close Saves: {bot_status['early_close_saves']}")
    if bot_status["target_wait_wins"] > 0:
        trailing_success_rate = (bot_status["target_wait_wins"] / 
                               (bot_status["target_wait_wins"] + bot_status["target_wait_losses"]) * 100)
        logger.info(f"Trailing Stop Success Rate: {trailing_success_rate:.1f}%")


def log_session_summary(symbol: str, logout_success: bool = True, show_logout_status: bool = True, show_bot_art: bool = True) -> None:
    """
    Log comprehensive session summary at end of trading day.
    Coordinates summary formatting through helper functions.
    Displays with rainbow thank you message on the right side.
    
    Args:
        symbol: Instrument symbol
        logout_success: Whether cleanup/logout was successful
        show_logout_status: Whether to show the logout status message (False for maintenance mode)
        show_bot_art: Whether to show rainbow bot art (False for maintenance mode)
    """
    stats = state[symbol]["session_stats"]
    
    # Display session summary (without bot art on the right - it's shown animated at the bottom)
    logger.info(SEPARATOR_LINE)
    logger.info("SESSION SUMMARY")
    logger.info(SEPARATOR_LINE)
    logger.info(f"Trading Day: {state[symbol]['trading_day']}")
    
    # Calculate and display session runtime
    if bot_status.get("session_start_time"):
        tz = pytz.timezone(CONFIG.get("timezone", "US/Eastern"))
        session_end = datetime.now(tz)
        runtime = session_end - bot_status["session_start_time"]
        hours, remainder = divmod(int(runtime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"Session Runtime: {hours}h {minutes}m {seconds}s")
    
    # Format trade statistics
    format_trade_statistics(stats)
    
    # Format P&L summary
    format_pnl_summary(stats)
    
    # Format risk metrics (flatten mode analysis)
    format_risk_metrics()
    
    # Format time statistics (position duration)
    format_time_statistics(stats)
    
    logger.info(SEPARATOR_LINE)
    
    # Log logout status right after session summary (only if show_logout_status is True)
    # Then immediately show the animated thank you on the same visual area
    if show_logout_status:
        logger.info("")
        if logout_success:
            logger.info("\033[92m✓ Logged out successfully\033[0m")  # Green
        else:
            logger.info("\033[91m✗ Logout completed with errors\033[0m")  # Red
    
    # Display animated rainbow "Thanks for using QuoTrading AI" message
    # Only show when bot art is enabled (not during maintenance mode)
    # SKIP in backtest mode to prevent spam
    # Starts right after logout message (no extra blank lines)
    if show_bot_art and display_animated_thank_you and not is_backtest_mode():
        try:
            display_animated_thank_you(duration=60.0, fps=15)
        except Exception as e:
            # Fallback to static display if animation fails
            logger.debug(f"Animation failed, using static display: {e}")
            if display_static_thank_you:
                display_static_thank_you()
    
    # Send daily summary alert
    try:
        notifier = get_notifier()
        total_trades = len(stats["trades"])
        total_pnl = stats.get("total_pnl", 0.0)
        win_count = stats.get("win_count", 0)
        loss_count = stats.get("loss_count", 0)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0
        
        # Calculate max drawdown from trade history
        max_drawdown = 0.0
        running_total = 0.0
        peak = 0.0
        for pnl in stats["trades"]:
            running_total += pnl
            peak = max(peak, running_total)
            drawdown = peak - running_total
            max_drawdown = max(max_drawdown, drawdown)
        
        notifier.send_daily_summary(
            trades=total_trades,
            profit_loss=total_pnl,
            win_rate=win_rate,
            max_drawdown=max_drawdown
        )
    except Exception as e:
        logger.debug(f"Failed to send daily summary alert: {e}")


def update_session_stats(symbol: str, pnl: float) -> None:
    """
    Update session statistics after a trade.
    
    Args:
        symbol: Instrument symbol
        pnl: Profit/Loss from the trade
    """
    stats = state[symbol]["session_stats"]
    
    # Add trade to history
    stats["trades"].append(pnl)
    
    # Update win/loss counts
    if pnl > 0:
        stats["win_count"] += 1
        stats["largest_win"] = max(stats["largest_win"], pnl)
    elif pnl < 0:
        stats["loss_count"] += 1
        stats["largest_loss"] = min(stats["largest_loss"], pnl)
    
    # Update total P&L
    stats["total_pnl"] += pnl
    
    # Update variance for Sharpe ratio calculation
    # Using Welford's online algorithm for variance
    n = len(stats["trades"])
    if n == 1:
        stats["mean_pnl"] = pnl
        stats["pnl_variance"] = 0.0
    else:
        old_mean = stats.get("mean_pnl", 0.0)
        new_mean = old_mean + (pnl - old_mean) / n
        stats["pnl_variance"] += (pnl - old_mean) * (pnl - new_mean)
        stats["mean_pnl"] = new_mean


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================




def round_to_tick(price: float) -> float:
    """
    Round price to nearest tick size.
    
    Args:
        price: Price to round
    
    Returns:
        Rounded price
    """
    tick_size = CONFIG["tick_size"]
    return round(price / tick_size) * tick_size


# ============================================================================
# PHASE TWO: Time Management - Support Both Live and Backtest
# ============================================================================

def get_current_time() -> datetime:
    """
    Get current time - either real time (live) or backtest simulation time.
    
    Returns:
        Current datetime with timezone
    """
    global backtest_current_time
    
    if backtest_current_time is not None:
        # Backtest mode: use simulated time
        return backtest_current_time
    else:
        # Live mode: use real time
        tz = pytz.timezone(CONFIG["timezone"])
        return datetime.now(tz)


def get_trading_state(dt: datetime = None) -> str:
    """
    Centralized time checking function that returns current trading state.
    CME Futures trading - uses US Eastern wall-clock time (DST-aware).
    
    **AZURE-FIRST DESIGN**: Checks Azure time service first for:
    - Maintenance windows (4:45-6:00 PM ET daily)
    - Single source of truth for all time-based decisions
    
    Falls back to local Eastern time logic if Azure unreachable.
    
    Args:
        dt: Datetime to check (defaults to current time - live or backtest)
            Should be timezone-aware.
    
    Returns:
        Trading state:
        - 'entry_window': Market open, ready to trade (6:00 PM - 4:45 PM next day)
        - 'closed': Market closed (flatten all positions immediately)
    
    CME Futures Schedule (US Eastern Wall-Clock - NEVER changes with DST):
    - Market opens: 6:00 PM Eastern (Sunday-Thursday)
    - Forced flatten: 4:45 PM Eastern daily (Mon-Fri) - close all positions
    - Maintenance: 4:45-6:00 PM Eastern (1hr 15min daily break)
    - Friday close: 4:45 PM Eastern (market closes for weekend maintenance)
    - Sunday open: 6:00 PM Eastern Sunday (weekly start)
    """
    # AZURE-FIRST: Try cloud time service (unless in backtest mode)
    if backtest_current_time is None:  # Live mode only
        azure_state = bot_status.get("azure_trading_state")
        if azure_state:
            # FIX: Sanity check - verify Azure "closed" state against local time
            # This prevents false positives where Azure incorrectly returns trading_allowed=False
            if azure_state == "closed":
                # Get local Eastern time to verify
                eastern_tz = pytz.timezone('US/Eastern')
                local_dt = datetime.now(eastern_tz) if dt is None else dt
                if local_dt.tzinfo is None:
                    local_dt = eastern_tz.localize(local_dt)
                eastern_time = local_dt.astimezone(eastern_tz)
                weekday = eastern_time.weekday()
                current_time_local = eastern_time.time()
                
                # Quick local validation of market hours
                # Market should only be "closed" during these times:
                # - Saturday (all day)
                # - Sunday before 6:00 PM ET
                # - Mon-Thu 4:45 PM - 6:00 PM ET (maintenance)
                # - Friday after 4:45 PM ET
                is_actually_closed = False
                
                if weekday == 5:  # Saturday
                    is_actually_closed = True
                elif weekday == 6 and current_time_local < datetime_time(18, 0):  # Sunday before 6 PM
                    is_actually_closed = True
                elif weekday == 4 and current_time_local >= datetime_time(16, 45):  # Friday after 4:45 PM
                    is_actually_closed = True
                elif weekday < 4 and datetime_time(16, 45) <= current_time_local < datetime_time(18, 0):  # Mon-Thu maintenance
                    is_actually_closed = True
                
                if not is_actually_closed:
                    # Azure says closed but local time says market should be open
                    # Don't trust Azure - fall through to local logic
                    logger.warning(f"Azure reports 'closed' but local time ({eastern_time.strftime('%H:%M %Z')}) indicates market should be open - using local time")
                    # Clear the bad Azure state
                    bot_status["azure_trading_state"] = None
                else:
                    # Azure and local agree - market is closed
                    return azure_state
            else:
                # Azure says entry_window - trust it
                return azure_state
    
    # FALLBACK: Local Eastern time logic (backtest mode or Azure unreachable)
    # Get current time in Eastern
    if dt is None:
        dt = get_current_time()
    
    # Ensure we're working in Eastern Time
    eastern_tz = pytz.timezone('US/Eastern')
    if dt.tzinfo is None:
        # If naive datetime, localize to Eastern
        dt = eastern_tz.localize(dt)
    
    # Convert to Eastern if not already
    eastern_time = dt.astimezone(eastern_tz)
    
    weekday = eastern_time.weekday()  # 0=Monday, 6=Sunday
    current_time = eastern_time.time()
    
    # CME Futures Hours (US Eastern - wall-clock time):
    # Sunday 6:00 PM - Friday 5:00 PM (with daily 5:00-6:00 PM maintenance Mon-Thu)
    
    # THANKSGIVING SPECIAL: Last Thursday of November closes at 1:00 PM ET
    if weekday == 3:  # Thursday
        # Check if this is the last Thursday of November
        if eastern_time.month == 11:
            # Get last day of November
            if eastern_time.month == 12:
                next_month = eastern_time.replace(year=eastern_time.year + 1, month=1, day=1)
            else:
                next_month = eastern_time.replace(month=eastern_time.month + 1, day=1)
            last_day = (next_month - timedelta(days=1)).day
            
            # Find last Thursday (iterate backwards from last day)
            for day in range(last_day, 0, -1):
                check_date = eastern_time.replace(day=day)
                if check_date.weekday() == 3:  # Thursday
                    last_thursday = day
                    break
            
            # If today is Thanksgiving (last Thursday of November)
            if eastern_time.day == last_thursday:
                # Market closes at 1:00 PM ET (flatten at 12:45 PM)
                if current_time >= datetime_time(12, 45):
                    return 'closed'
    
    # CLOSED: Saturday (all day) - Market is closed
    if weekday == 5:  # Saturday
        return 'closed'
    
    # CLOSED: Sunday before 6:00 PM ET (opens AT 6:00 PM - weekly open)
    if weekday == 6 and current_time < datetime_time(18, 0):
        return 'closed'
    
    # FRIDAY SPECIAL: Market closes at 4:45 PM ET (futures maintenance)
    if weekday == 4 and current_time >= datetime_time(16, 45):
        return 'closed'
    
    # Get configured forced flatten time from CONFIG (CME Eastern schedule)
    # Futures market: Trading 6:00 PM - 4:45 PM daily
    # Maintenance: 4:45 PM - 6:00 PM daily
    forced_flatten_time = CONFIG.get("forced_flatten_time", datetime_time(16, 45))  # 4:45 PM ET - force close
    
    # CLOSED: Daily maintenance (4:45-6:00 PM ET, Monday-Thursday)
    if weekday < 4:  # Monday-Thursday
        if forced_flatten_time <= current_time < datetime_time(18, 0):
            return 'closed'  # Daily maintenance period
    
    # ENTRY WINDOW: Market open, ready to trade
    # We're in entry window if:
    # - Between 6:00 PM and 4:45 PM next day (Mon-Thu)
    # - Between 6:00 PM Sunday and 4:45 PM Friday
    # - NOT in closed periods above
    return 'entry_window'


# ============================================================================
# PHASE FIFTEEN & SIXTEEN: Timezone Handling and Time-Based Logging
# ============================================================================

def validate_timezone_configuration() -> None:
    """
    Validate timezone configuration on bot startup.
    CME Futures use US Eastern wall-clock time - pytz handles DST automatically.
    """
    tz = pytz.timezone('US/Eastern')
    current_time = datetime.now(tz)
    
    # Show timezone info per LOGGING_SPECIFICATION.md startup section
    logger.info(f"Timezone: US/Eastern | Current: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")


def log_time_based_action(action: str, reason: str, details: Optional[Dict[str, Any]] = None) -> None:
    """
    Log all time-based actions with timestamp and reason.
    Creates audit trail for reviewing time-based rule performance.
    
    Args:
        action: Type of action (e.g., 'entry_blocked', 'flatten_activated', 'position_closed')
        reason: Human-readable reason for the action
        details: Optional dictionary of additional details
    """
    eastern_tz = pytz.timezone('US/Eastern')
    timestamp = datetime.now(eastern_tz)
    
    log_msg = f"TIME-BASED ACTION: {action}"
    log_msg += f" | Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    log_msg += f" | Reason: {reason}"
    
    if details:
        for key, value in details.items():
            log_msg += f" | {key}: {value}"
    
    logger.info(log_msg)


# ============================================================================
# PHASE SEVENTEEN & EIGHTEEN: Backtesting and Monitoring Guidelines
# ============================================================================

"""
Phase Seventeen: Backtesting Time Logic

When backtesting this strategy on historical data, you must implement time-based
flatten rules accurately:

1. For every historical trade, check entry time
   - If entered after 2 PM, calculate time remaining until 4:45 PM flatten
   - Execute forced flatten at whatever price existed at 4:45 PM if position
     hadn't hit target or stop

2. Track forced flatten statistics:
   - Count how many trades were force-flattened before hitting target/stop
   - Calculate how many times forced flatten saved from overnight gap losses
   - Calculate how many times it cost profit by closing a winner early

3. Analyze trade duration:
   - If 30%+ of trades get force-flattened, average trade duration is too long
   - Either extend trading hours or accept lower targets to close positions faster

4. DST transition testing:
   - Test bot behavior on DST change days (March and November)
   - Ensure time checks still work correctly during "spring forward" and "fall back"

Phase Eighteen: Monitoring During Market Close Window (4:45 PM ET)

This is the highest-risk period requiring active monitoring if possible:

1. Manual intervention scenarios:
   - Bot tries to close but gets no fills even with aggressive limits
   - Manual close at market through broker platform may be needed
   
   - Technical glitch: bot thinks it's flat but broker shows open position
   - Immediate manual intervention required
   
   - Order system failure during critical flatten window
   - Manual close through broker as backup

2. Monitoring checklist:
   - Verify position quantity matches between bot and broker
   - Check that flatten orders are actually being placed
   - Monitor fill confirmations
   - Verify position is actually closed before 5 PM

3. Contingency plan:
   - Have broker platform open and ready
   - Know how to manually close position
   - Have broker support number available
   - Test manual close procedure in paper trading

4. Post-flatten validation:
   - At 5 PM, verify zero positions in both bot and broker
   - Check overnight position safety function logged correctly
   - Review flatten window logs for any issues

This 15-minute window is when most bot failures happen due to:
- Racing against hard deadline
- Deteriorating market conditions
- Widening spreads
- Lower liquidity

Active monitoring provides safety net for automation failures.
"""


# ============================================================================
# PHASE TWENTY: Position Duration Statistics & Complete Summary
# ============================================================================

"""
Phase Twenty: Position Duration Statistics & Time-Window Compatibility

Track how long positions stay open on average to ensure compatibility with
time-based flatten requirements:

1. Position Duration Tracking:
   - Record duration (minutes) for every closed position
   - Calculate average, min, max duration
   - Compare against available time window

2. Force Flatten Analysis:
   - Count trades force-flattened due to time limits
   - Calculate percentage: force_flattened / total_trades
   - RED FLAG if >30% are force-flattened
   
3. After-Noon Entry Analysis:
   - Track entries after 12 PM (noon)
   - Calculate force-flatten rate for after-noon entries
   - If entering at 2 PM with 3-hour avg duration, you'll be force-flattened
   
4. Time Window Compatibility:
   - Entry at 2 PM  165 minutes until 4:45 PM deadline
   - If avg duration >132 min (80% of 165), trades run out of time
   - Recommend: move entry cutoff earlier OR use faster targets

5. Strategic Adjustments Based on Data:
   - If most trades close in 30 min  plenty of buffer time
   - If most trades take 2-3 hours  cutting it close, risk force-flatten
   - Solution A: Earlier entry cutoff (12 PM instead of 2:30 PM)
   - Solution B: Faster targets (1:1 R/R instead of 1.5:1)
   - The data tells you which adjustment fits your strategy

Complete Time-Based Logic Summary
==================================

Your bot operates in distinct time-based modes controlling all actions:

TIME WINDOWS (All times Eastern Time - 24/5 Futures Trading):
- Saturday: CLOSED - Market closed for weekend
- Sunday before 6:00 PM: CLOSED - Waiting for futures open
- Sunday 6:00 PM: MARKET OPEN - Trading resumes for the week
- 6:00 PM - 4:45 PM (next day): ENTRY WINDOW - Full trading allowed 24 hours (Mon-Thu)
- 21:45 - 5:00 PM ET: FLATTEN MODE - Close positions (15 min before maintenance)
- 22:00 - 6:00 PM ET: MAINTENANCE - Daily settlement (Mon-Thu), market closed
- Friday 5:00 PM ET onwards: WEEKEND - Market closes early before weekend
- Saturday: WEEKEND - Market closed
- Sunday before 5:00 PM ET: WEEKEND - Market closed until Sunday 5:00 PM ET

FLATTEN SCHEDULE (UTC - CME Futures):
- Monday-Thursday: Flatten 21:45-5:00 PM ET (15 min before daily maintenance)
- Friday: Market closes 21:00 UTC (no flatten mode needed, market just closes)
- During flatten mode: Aggressive closing, no new entries
- After 5:00 PM ET: Maintenance window (Mon-Thu) or weekend (Fri-Sun)

DAILY RESETS:
- 6:00 PM ET: Daily session opens (after maintenance window) - VWAP resets here at market open (6 PM EST)
- Daily counters reset at 6:00 PM ET when new session starts

CRITICAL SAFETY RULES (24/5 FUTURES - UTC):
1. FLATTEN BEFORE MAINTENANCE - Close by 4:45 PM ET daily (15 min buffer before 5:00 PM ET)
2. NO WEEKEND POSITIONS - Market closes 5:00 PM ET Friday (weekend begins)
3. MAINTENANCE WINDOW - Market closed 22:00-6:00 PM ET Mon-Thu for settlement
4. TIMEZONE ENFORCEMENT - All decisions use UTC (CME futures standard)
5. NO DST ISSUES - UTC never changes, no daylight saving complications
6. AUDIT TRAIL - Every time-based action logged with timestamp and reason

WHY THIS MATTERS FOR PROP FIRMS:
Prop firm rules are designed to fail traders who don't respect:
- Daily settlement (5:00 PM ET reset Mon-Thu, maintenance window)
- Overnight gap exposure
- Weekend event risk
- Daily loss limits (restart at 5:00 PM ET, not midnight)

By building time constraints into core logic, you protect against:
- Gap risk from overnight news (Asia/Europe markets, economic data)
- Weekend geopolitical events (can't control, can't trade out)
- Settlement skew manipulation (institutional games in final seconds)
- Starting day already halfway to loss limit (overnight position losses carry forward)

This time-based framework is NOT OPTIONAL for prop firm trading.
It's the difference between controlled risk and catastrophic account blowups.
Being in a position when you shouldn't be is the #1 futures trading killer.

VALIDATION:
- Phase 20 statistics tell you if your strategy fits the time windows
- If >30% force-flattened: strategy incompatible with time constraints
- Adjust entry cutoff earlier OR use tighter trailing stops
- After-noon entries especially risky - limited time to work
"""


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(symbol_override: str = None) -> None:
    """Main bot execution with event loop integration
    
    Args:
        symbol_override: Optional symbol to trade (overrides CONFIG["instrument"])
                        Used for multi-symbol bot instances
    """
    global event_loop, timer_manager, bid_ask_manager, cloud_api_client, rl_brain, current_trading_symbol
    
    # CRITICAL: Determine trading symbol FIRST, before license validation
    # This enables symbol-specific sessions for multi-symbol support
    current_trading_symbol = symbol_override if symbol_override else CONFIG["instrument"]
    trading_symbol = current_trading_symbol  # Keep local variable for compatibility
    
    # Track session start time for runtime display
    bot_status["session_start_time"] = datetime.now(pytz.timezone(CONFIG.get("timezone", "US/Eastern")))
    
    # CRITICAL: Validate license FIRST, before any initialization
    # This is the "login screen" - fail fast if license invalid or session conflict
    # Uses current_trading_symbol for symbol-specific session (multi-symbol support)
    validate_license_at_startup()
    
    # Professional startup header with GUI settings
    logger.info("=" * 80)
    logger.info("QuoTrading AI Professional Trading System")
    logger.info("=" * 80)
    
    # Display mode and connection
    if _bot_config.shadow_mode:
        mode_str = "SIGNAL-ONLY MODE (Manual Trading)"
    else:
        mode_str = "LIVE TRADING"
    logger.info(f"Mode: {mode_str}")
    
    # Show the configured symbol
    logger.info(f"Symbol: {trading_symbol}")
    
    # Show broker connection status (will be updated after broker connects)
    logger.info("Broker: Connecting...")
    
    # Display GUI Settings in professional format
    logger.info("")
    logger.info("📋 Trading Configuration:")
    logger.info(f"  • Max Contracts: {CONFIG['max_contracts']}")
    logger.info(f"  • Max Trades/Day: {CONFIG['max_trades_per_day']}")
    logger.info(f"  • Max Loss Per Trade: ${CONFIG.get('max_stop_loss_dollars', DEFAULT_MAX_STOP_LOSS_DOLLARS):.0f}")
    logger.info(f"  • Daily Loss Limit: ${CONFIG['daily_loss_limit']}")
    logger.info(f"  • Entry Window: {CONFIG['entry_start_time']} - {CONFIG['entry_end_time']} ET")
    logger.info(f"  • Force Close: {CONFIG['forced_flatten_time']} ET")
    logger.info("=" * 80)
    logger.info("")
    
    # Initialize local RL brain for LIVE and BACKTEST modes
    # LIVE MODE: Reads from local symbol-specific folder for pattern matching, saves to cloud only
    # BACKTEST MODE: Reads and saves to local symbol-specific folder
    if is_backtest_mode() or CONFIG.get("backtest_mode", False):
        pass  # Silent - backtest mode initialization
    else:
        pass  # Silent - live mode initialization
        # Use symbol-specific folder for experiences
        signal_exp_file = str(get_data_file_path(f"experiences/{trading_symbol}/signal_experience.json"))
        rl_brain = SignalConfidenceRL(
            experience_file=signal_exp_file,
            backtest_mode=False,  # Live mode
            confidence_threshold=CONFIG.get("rl_confidence_threshold"),
            exploration_rate=0.0,  # No exploration in live mode (pure exploitation)
            min_exploration=0.0,
            exploration_decay=0.995,
            save_local=False  # Live mode: read local but save to cloud only
        )
        pass  # Silent - RL brain initialized
        pass  # Silent - live mode save configuration
        
        # Initialize Cloud API Client for reporting trade outcomes to cloud
        license_key = os.getenv("QUOTRADING_LICENSE_KEY")
        if license_key:
            cloud_api_url = "https://quotrading-flask-api.azurewebsites.net"
            cloud_api_client = CloudAPIClient(
                api_url=cloud_api_url,
                license_key=license_key,
                timeout=10
            )
            pass  # Silent - cloud API initialized
        else:
            logger.warning(f"No license key - cloud outcome reporting disabled")
    
    # Symbol specifications - suppress detailed info, just essentials
    # (Tick value, slippage, etc. are technical details customers don't need)
    
    # Operating mode and settings already shown in header - no need to repeat
    pass  # Silent - configuration already displayed in header
    
    # Phase Fifteen: Validate timezone configuration
    validate_timezone_configuration()
    
    # Initialize bid/ask manager
    pass  # Silent - bid/ask manager initialization
    bid_ask_manager = BidAskManager(CONFIG)
    
    # Initialize broker (replaces initialize_sdk)
    initialize_broker()
    
    # Phase 12: Record starting equity for drawdown monitoring
    bot_status["starting_equity"] = get_account_equity()
    
    # Update header with broker connection status and starting equity
    logger.info(f"✅ Broker Connected | Starting Equity: ${bot_status['starting_equity']:.2f}")
    logger.info("")
    
    # AI MODE: Clean startup - no "waiting for data" message
    # LIVE MODE: Show waiting for data message
    logger.info("📊 Waiting for market data... (Bars and quotes will appear once data flows)")
    logger.info("")
    
    # Initialize state for instrument (use override symbol if provided)
    initialize_state(trading_symbol)
    
    # CRITICAL: Try to restore position state from disk if bot was restarted
    pass  # Silent - checking for saved position state
    position_restored = load_position_state(trading_symbol)
    if position_restored:
        logger.warning(f"[{trading_symbol}] ⚠️  BOT RESTARTED WITH ACTIVE POSITION - Managing existing trade")
    else:
        pass  # Silent - no saved position
    
    # Skip historical bars fetching in live mode - not needed for real-time trading
    # The bot will build bars from live tick data
    pass  # Silent - skipping historical bars fetch
    
    # Initialize event loop
    pass  # Silent - event loop initialization
    event_loop = EventLoop(bot_status, CONFIG)
    
    # Register event handlers
    event_loop.register_handler(EventType.TICK_DATA, handle_tick_event)
    event_loop.register_handler(EventType.TIME_CHECK, handle_time_check_event)
    event_loop.register_handler(EventType.VWAP_RESET, handle_vwap_reset_event)
    event_loop.register_handler(EventType.POSITION_RECONCILIATION, handle_position_reconciliation_event)
    event_loop.register_handler(EventType.CONNECTION_HEALTH, handle_connection_health_event)
    event_loop.register_handler(EventType.LICENSE_CHECK, handle_license_check_event)
    event_loop.register_handler(EventType.SHUTDOWN, handle_shutdown_event)
    
    # Register shutdown handlers for cleanup
    event_loop.register_shutdown_handler(cleanup_on_shutdown)
    
    # Register atexit handler to ensure session is ALWAYS released
    atexit.register(release_session)
    
    # Register signal handlers for Ctrl+C, SIGTERM, etc.
    def signal_handler(signum, frame):
        release_session()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)  # Windows Ctrl+Break
    
    # Initialize timer manager for periodic events
    tz = pytz.timezone(CONFIG["timezone"])
    timer_manager = TimerManager(event_loop, CONFIG, tz)
    timer_manager.start()
    
    # LIVE MODE: Subscribe to market data (trades) - use trading_symbol
    subscribe_market_data(trading_symbol, on_tick)
    
    # Subscribe to bid/ask quotes if broker supports it
    if broker is not None and hasattr(broker, 'subscribe_quotes'):
        pass  # Silent - quote subscription
        try:
            broker.subscribe_quotes(trading_symbol, on_quote)
        except Exception as e:
            logger.warning(f"Failed to subscribe to quotes: {e}")
            logger.warning("Continuing without bid/ask quote data")
    
    # RL is CLOUD-ONLY - no local RL components
    # Users get confidence from cloud, contribute to cloud hive mind
    # Only the dev (Kevin) gets the experience data saved to cloud
    pass  # Silent - RL cloud mode (not customer-facing)
    
    # Show current date/time before starting
    current_time = datetime.now(tz)
    logger.info(f"📅 {current_time.strftime('%A, %B %d, %Y at %I:%M %p %Z')}")
    logger.info("")
    logger.info("🚀 Bot Ready - Monitoring for Signals")
    logger.info("")
    
    # Run event loop (blocks until shutdown signal)
    try:
        event_loop.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Event loop error: {e}")
    finally:
        pass  # Silent - event loop cleanup
        
        # CRITICAL: Release session immediately on ANY exit
        release_session()
        
        # Metrics are already logged by event loop's _log_metrics()
        # No need to call get_metrics() here


# ============================================================================
# EVENT HANDLERS
# ============================================================================

def handle_tick_event(event) -> None:
    """Handle tick data event from event loop"""
    global backtest_current_time
    
    # Extract data from Event object
    data = event.data if hasattr(event, 'data') else event
    
    symbol = data["symbol"]
    price = data["price"]
    volume = data["volume"]
    timestamp_ms = data["timestamp"]
    
    if symbol not in state:
        initialize_state(symbol)
    
    # Phase 12: Update last tick time for connection health check
    tz = pytz.timezone(CONFIG["timezone"])
    dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=tz)
    bot_status["last_tick_time"] = dt
    
    # BACKTEST MODE: Update simulation time so all time-based logic uses historical time
    if is_backtest_mode():
        backtest_current_time = dt
    
    # Increment total tick counter (separate from deque storage which caps at 10k)
    if "total_ticks_received" not in state[symbol]:
        state[symbol]["total_ticks_received"] = 0
    state[symbol]["total_ticks_received"] += 1
    
    # Market monitoring is now done on every 1-minute bar close (see update_bars_1min function)
    
    # Create tick object
    tick = {
        "price": price,
        "volume": volume,
        "timestamp": timestamp_ms
    }
    
    # Append to tick storage
    state[symbol]["ticks"].append(tick)
    
    # Update 1-minute bars
    update_1min_bar(symbol, price, volume, dt)
    
    # Update 15-minute bars
    update_15min_bar(symbol, price, volume, dt)


def handle_time_check_event(data: Dict[str, Any]) -> None:
    """Handle time-based checks event"""
    symbol = CONFIG["instrument"]
    if symbol in state:
        tz = pytz.timezone(CONFIG["timezone"])
        current_time = datetime.now(tz)
        current_time_only = current_time.time()
        
        # Check for daily reset
        check_daily_reset(symbol, current_time)
        
        # Check for delayed license expiration stop conditions
        # If license expired and we're waiting for market close (Friday)
        if bot_status.get("stop_at_market_close", False):
            maintenance_start = CONFIG.get("forced_flatten_time", datetime_time(16, 45))  # 4:45 PM ET
            if current_time_only >= maintenance_start:
                logger.critical("≡ƒ¢æ Market closed - stopping trading due to expired license")
                
                # Flatten any open positions
                if state[symbol]["position"]["active"]:
                    logger.critical(f"≡ƒöÆ Closing position at market close")
                    position = state[symbol]["position"]
                    current_price = state[symbol]["bars"][-1]["close"] if state[symbol]["bars"] else None
                    
                    if current_price:
                        side = position["side"]
                        exit_side = "SELL" if side == "long" else "BUY"
                        quantity = position["quantity"]
                        
                        try:
                            order = broker.place_market_order(symbol, exit_side, quantity)
                            if order:
                                logger.info(f"Γ£à Position closed at market close")
                        except Exception as e:
                            logger.error(f"Failed to close position: {e}")
                
                # Disable trading
                bot_status["trading_enabled"] = False
                bot_status["emergency_stop"] = True
                bot_status["stop_reason"] = "License expired - stopped at Friday market close"
                bot_status["stop_at_market_close"] = False
        
        # If license expired and we're waiting for maintenance window
        elif bot_status.get("stop_at_maintenance", False):
            maintenance_start = CONFIG.get("forced_flatten_time", datetime_time(16, 45))  # 4:45 PM ET
            if current_time_only >= maintenance_start:
                logger.critical("≡ƒ¢æ Maintenance window reached - stopping trading due to expired license")
                
                # Flatten any open positions
                if state[symbol]["position"]["active"]:
                    logger.critical(f"≡ƒöÆ Closing position at maintenance window")
                    position = state[symbol]["position"]
                    current_price = state[symbol]["bars"][-1]["close"] if state[symbol]["bars"] else None
                    
                    if current_price:
                        side = position["side"]
                        exit_side = "SELL" if side == "long" else "BUY"
                        quantity = position["quantity"]
                        
                        try:
                            order = broker.place_market_order(symbol, exit_side, quantity)
                            if order:
                                logger.info(f"Γ£à Position closed at maintenance window")
                        except Exception as e:
                            logger.error(f"Failed to close position: {e}")
                
                # Disable trading
                bot_status["trading_enabled"] = False
                bot_status["emergency_stop"] = True
                bot_status["stop_reason"] = "License expired - stopped at maintenance window"
                bot_status["stop_at_maintenance"] = False
        
        # Critical safety check - NO positions past 5 PM
        check_no_overnight_positions(symbol)




def handle_vwap_reset_event(data: Dict[str, Any]) -> None:
    """Handle VWAP reset event - DISABLED per strategy requirements"""
    # VWAP reset disabled: Strategy only needs accurate VWAP for current bar (condition 8)
    # No need to reset VWAP at maintenance open - just keep calculating as usual
    # The capitulation reversal strategy doesn't depend on VWAP bands or mean reversion
    pass


def handle_position_reconciliation_event(data: Dict[str, Any]) -> None:
    """
    Handle periodic position reconciliation check.
    Verifies bot's position state matches broker's actual position.
    Runs every 5 seconds to detect and correct any desyncs.
    
    LIVE MODE: Checks configured symbol for position mismatch and auto-corrects.
    
    CRITICAL FIX: Skip reconciliation when an entry order is pending to prevent
    clearing position state while order is still being processed. This prevents
    duplicate orders and state corruption.
    """
    # CRITICAL FIX: Skip reconciliation when entry order is pending
    # This prevents the race condition where:
    # 1. Bot places order
    # 2. Reconciliation runs before fill is confirmed
    # 3. Reconciliation sees broker=0, bot=1 (order not yet filled)
    # 4. Reconciliation clears bot state
    # 5. Bot then places another order (duplicate!)
    if bot_status.get("entry_order_pending", False):
        pending_since = bot_status.get("entry_order_pending_since")
        if pending_since:
            elapsed = (datetime.now() - pending_since).total_seconds()
            # Give orders up to 60 seconds to complete before forcing reconciliation
            # Note: This timeout could be made configurable via CONFIG if needed
            max_pending_seconds = 60
            if elapsed < max_pending_seconds:
                logger.debug(f"Skipping reconciliation - entry order pending for {elapsed:.1f}s")
                return
            else:
                # Order has been pending too long - something is wrong
                logger.warning(f"Entry order pending for {elapsed:.1f}s - forcing reconciliation")
                bot_status["entry_order_pending"] = False
                bot_status["entry_order_pending_since"] = None
                bot_status["entry_order_pending_symbol"] = None
                bot_status["entry_order_pending_id"] = None
    
    # ============================================================
    # LIVE MODE: Position reconciliation for configured symbol
    # ============================================================
    symbol = CONFIG["instrument"]
    
    if symbol not in state:
        return
    
    try:
        # Get broker's actual position
        broker_position = get_position_quantity(symbol)
        
        # Get bot's tracked position
        bot_active = state[symbol]["position"]["active"]
        flatten_pending = state[symbol]["position"].get("flatten_pending", False)
        
        if bot_active:
            bot_qty = state[symbol]["position"]["quantity"]
            bot_side = state[symbol]["position"]["side"]
            bot_position = bot_qty if bot_side == "long" else -bot_qty
        else:
            bot_position = 0
        
        # FIX: If broker is flat and we had a flatten pending, confirm the flatten completed
        if broker_position == 0 and flatten_pending:
            logger.info("Position flatten confirmed - broker position is now flat")
            state[symbol]["position"]["flatten_pending"] = False
            clear_flatten_flags()
            return  # No mismatch to handle
        
        # Check for mismatch
        if broker_position != bot_position:
            logger.error("=" * 60)
            logger.error("POSITION RECONCILIATION MISMATCH DETECTED!")
            logger.error("=" * 60)
            logger.error(f"  Broker Position: {broker_position} contracts")
            logger.error(f"  Bot Position:    {bot_position} contracts")
            logger.error(f"  Discrepancy:     {abs(broker_position - bot_position)} contracts")
            
            # Determine corrective action
            if broker_position == 0 and bot_position != 0:
                # Broker is flat but bot thinks it has a position
                # CRITICAL FIX: Check if we recently placed an order - if so, wait for fill confirmation
                # This prevents clearing state when broker API just hasn't caught up yet
                position = state[symbol].get("position", {})
                entry_time = position.get("entry_time")
                
                if entry_time:
                    # Check how long ago we entered
                    if isinstance(entry_time, datetime):
                        time_since_entry = (get_current_time() - entry_time).total_seconds()
                    else:
                        time_since_entry = 999  # Unknown, don't skip
                    
                    # If we entered less than 10 seconds ago, don't clear - wait for broker to catch up
                    # Note: This grace period could be made configurable via CONFIG if needed
                    reconciliation_grace_period = 10  # seconds
                    if time_since_entry < reconciliation_grace_period:
                        logger.warning(f"  [WAIT] Position entered {time_since_entry:.1f}s ago - waiting for broker confirmation")
                        logger.warning(f"  [WAIT] Not clearing state yet - broker may not have reported fill")
                        return
                
                logger.error("  Cause: Position was closed externally or bot missed exit fill")
                logger.error("  Action: Clearing bot's position state")
                state[symbol]["position"]["active"] = False
                state[symbol]["position"]["quantity"] = 0
                state[symbol]["position"]["side"] = None
                state[symbol]["position"]["entry_price"] = None
                state[symbol]["position"]["flatten_pending"] = False
                
                # FIX: Clear flatten in progress flag now that position is confirmed closed
                clear_flatten_flags()
                
            elif broker_position != 0 and bot_position == 0:
                # Broker has position but bot thinks it's flat - close the unexpected position
                logger.error("  Cause: Position opened externally or bot missed entry fill")
                logger.error("  Action: CLOSING UNEXPECTED POSITION at market")
                
                # Emergency flatten the unexpected position
                side = "sell" if broker_position > 0 else "buy"
                quantity = abs(broker_position)
                
                logger.warning(f"Placing emergency market order: {side} {quantity} {symbol}")
                broker.place_market_order(symbol, side, quantity)
            
            else:
                # Both have positions but quantities don't match
                logger.error("  Cause: Partial fill or quantity mismatch")
                logger.error("  Action: Syncing bot state to match broker")
                
                # Update bot state to match broker
                state[symbol]["position"]["active"] = True if broker_position != 0 else False
                state[symbol]["position"]["quantity"] = abs(broker_position)
                state[symbol]["position"]["side"] = "long" if broker_position > 0 else "short"
            
            # Save corrected state
            if recovery_manager:
                recovery_manager.save_state(state)
                logger.info("Corrected position state saved to disk")
            
            logger.error("=" * 60)
            
            # TODO: Send alert notification when implemented
            # send_telegram_alert(f"Position mismatch: Broker={broker_position}, Bot={bot_position}")
            
        else:
            # Positions match - silent per LOGGING_SPECIFICATION.md
            pass  # Silent - position reconciliation
    
    except Exception as e:
        logger.error(f"Error during position reconciliation: {e}", exc_info=True)


def handle_connection_health_event(data: Dict[str, Any]) -> None:
    """
    Handle periodic connection health check event.
    Verifies broker connection is alive and reconnects if needed.
    Runs every 20 seconds.
    """
    check_broker_connection()


def handle_license_check_event(data: Dict[str, Any]) -> None:
    """
    Handle periodic license validation check event.
    Validates license with cloud API and gracefully stops trading if expired.
    Runs every 5 minutes.
    
    Expiration Handling Strategy:
    - If expires during maintenance window (5:00-6:00 PM ET): Stop when maintenance begins
    - If expires on Friday after hours: Stop when market closes (5:00 PM ET)  
    - If expires on weekend: Stop on Friday at close
    - Otherwise: Stop immediately and flatten positions
    """
    global cloud_api_client, broker
    
    # Skip if in backtest mode or no cloud client
    if is_backtest_mode() or cloud_api_client is None:
        return
    
    # Skip if already stopped for expiration
    if bot_status.get("license_expired", False):
        return
    
    try:
        # Get license key from config
        license_key = os.getenv("QUOTRADING_LICENSE_KEY")
        if not license_key:
            logger.warning("No license key configured - cannot validate")
            return
        
        # Send heartbeat to maintain session (don't use validate-license as it creates new sessions)
        # Use symbol-specific fingerprint for multi-symbol session support
        # MULTI-SYMBOL FIX: Include symbol explicitly so server can manage per-symbol sessions
        import requests
        api_url = os.getenv("QUOTRADING_API_URL", "https://quotrading-flask-api.azurewebsites.net")
        symbol_for_session = get_current_symbol_for_session()
        
        response = requests.post(
            f"{api_url}/api/heartbeat",
            json={
                "license_key": license_key,
                "device_fingerprint": get_device_fingerprint(symbol_for_session),
                "symbol": symbol_for_session  # MULTI-SYMBOL: Explicit symbol for server-side session management
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract expiration information
            expiration_iso = data.get("license_expiration")
            days_until_expiration = data.get("days_until_expiration")
            hours_until_expiration = data.get("hours_until_expiration")
            
            # Check if license is valid
            if not data.get("license_valid", False):
                # License has expired or been revoked
                reason = data.get("message", "License invalid")
                logger.critical(f"🛡️ LICENSE EXPIRED: {reason}")
                bot_status["license_expired"] = True
                bot_status["license_expiry_reason"] = reason
                
                # LICENSE EXPIRED - Stop trading immediately
                # Exception: If position is active, enter grace period to safely close it
                symbol = CONFIG["instrument"]
                has_active_position = symbol in state and state[symbol]["position"]["active"]
                
                if has_active_position:
                    # GRACE PERIOD: License expired but position is active
                    # Allow bot to manage position until it closes naturally
                    logger.warning("=" * 70)
                    logger.warning("⏳ LICENSE GRACE PERIOD ACTIVATED")
                    logger.warning("License expired but position is active")
                    logger.warning("Bot will continue managing position until it closes")
                    logger.warning("Position will close via normal exit rules (trailing stop/time)")
                    logger.warning("No new trades will be allowed")
                    logger.warning("=" * 70)
                    
                    # Set grace period flag
                    bot_status["license_grace_period"] = True
                    bot_status["grace_period_reason"] = "Active position - managing until close"
                    
                    # Block new trades but allow position management
                    # Note: check_safety_conditions will allow position management
                    # but block new entries when license_expired=True and position is active
                    
                    # Send notification about grace period
                    try:
                        notifier = get_notifier()
                        position = state[symbol]["position"]
                        notifier.send_error_alert(
                            error_message=f"🛡️ LICENSE EXPIRED (Grace Period Active)\n\n"
                                         f"Your license has expired but you have an active {position['side']} position.\n"
                                         f"Bot will continue managing the position until it closes.\n"
                                         f"Position: {position['quantity']} contracts @ ${position['entry_price']:.2f}\n\n"
                                         f"No new trades will be allowed.\n"
                                         f"Please renew your license.",
                            error_type="License Expired - Grace Period"
                        )
                    except Exception as e:
                        logger.debug(f"Failed to send grace period notification: {e}")
                    
                else:
                    # No active position - stop immediately
                    logger.critical("=" * 70)
                    logger.critical("")
                    logger.critical("  LICENSE EXPIRED")
                    logger.critical("")
                    logger.critical("  Your license has expired and is no longer valid.")
                    logger.critical("  Please renew your license to continue trading.")
                    logger.critical("")
                    logger.critical("  Contact: support@quotrading.com")
                    logger.critical("")
                    logger.critical("=" * 70)
                    
                    # Disable trading
                    bot_status["trading_enabled"] = False
                    bot_status["emergency_stop"] = True
                    bot_status["stop_reason"] = "License expired - trading stopped"
                    
                    # Send notification
                    try:
                        notifier = get_notifier()
                        notifier.send_error_alert(
                            error_message=f"🛡️ TRADING STOPPED: {reason}\n\nPlease renew your license to continue trading.",
                            error_type="License Expired"
                        )
                    except Exception as e:
                        pass  # Silent - no logs after expiration
                    
                    # Disconnect broker cleanly
                    logger.critical("Disconnecting from broker...")
                    logger.critical("LICENSE EXPIRED - Stopping all trading and market data")
                    try:
                        if broker is not None:
                            broker.disconnect()
                            logger.critical("Websocket disconnected - No data streaming")
                    except Exception as e:
                        pass  # Silent disconnect
                    
                    # Bot stays ON but IDLE - never exits unless user presses Ctrl+C
                    logger.critical("Bot will remain ON but IDLE (no trading)")
                    logger.critical("LICENSE EXPIRED - Please renew your license")
                    logger.critical("Press Ctrl+C to stop bot")
            else:
                # License is still valid
                logger.debug("Γ£à License validation successful")
                
                # PRE-EXPIRATION WARNINGS
                # Store expiration info in bot_status
                if expiration_iso:
                    bot_status["license_expiration"] = expiration_iso
                    bot_status["days_until_expiration"] = days_until_expiration
                    bot_status["hours_until_expiration"] = hours_until_expiration
                
                # WARNING: License expiring within 24 hours
                if hours_until_expiration is not None and hours_until_expiration <= 24 and hours_until_expiration > 0:
                    # Only warn once per session
                    if not bot_status.get("expiry_warning_24h_sent", False):
                        logger.warning("=" * 70)
                        logger.warning("ΓÜá∩╕Å LICENSE EXPIRATION WARNING - 24 HOURS")
                        logger.warning(f"Your license will expire in {hours_until_expiration:.1f} hours")
                        logger.warning("Please renew your license to avoid interruption")
                        logger.warning("Any open trades will be safely closed before expiration")
                        logger.warning("=" * 70)
                        
                        bot_status["expiry_warning_24h_sent"] = True
                        
                        # Send notification
                        try:
                            notifier = get_notifier()
                            notifier.send_error_alert(
                                error_message=f"ΓÜá∩╕Å LICENSE EXPIRING SOON\n\n"
                                             f"Your license will expire in {hours_until_expiration:.1f} hours.\n"
                                             f"Expiration: {expiration_iso}\n\n"
                                             f"Please renew to avoid interruption.\n"
                                             f"Any open trades will be safely closed.",
                                error_type="License Expiration Warning - 24 Hours"
                            )
                        except Exception as e:
                            logger.debug(f"Failed to send 24h expiry warning: {e}")
                
                # WARNING: License expiring within 7 days
                elif days_until_expiration is not None and days_until_expiration <= 7 and days_until_expiration > 1:
                    # Only warn once per session
                    if not bot_status.get("expiry_warning_7d_sent", False):
                        logger.warning("=" * 70)
                        logger.warning(f"ΓÜá∩╕Å LICENSE EXPIRATION WARNING - {days_until_expiration} DAYS")
                        logger.warning(f"Your license will expire in {days_until_expiration} days")
                        logger.warning("Please renew your license to avoid interruption")
                        logger.warning("=" * 70)
                        
                        bot_status["expiry_warning_7d_sent"] = True
                        
                        # Send notification (only for 7 days, not every check)
                        try:
                            notifier = get_notifier()
                            notifier.send_error_alert(
                                error_message=f"ΓÜá∩╕Å LICENSE EXPIRING IN {days_until_expiration} DAYS\n\n"
                                             f"Your license will expire on {expiration_iso}.\n\n"
                                             f"Please renew to continue trading without interruption.",
                                error_type=f"License Expiration Warning - {days_until_expiration} Days"
                            )
                        except Exception as e:
                            logger.debug(f"Failed to send 7d expiry warning: {e}")
                
                # CRITICAL: Don't enter new trades if expiring within 2 hours
                if hours_until_expiration is not None and hours_until_expiration <= 2 and hours_until_expiration > 0:
                    if not bot_status.get("near_expiry_mode", False):
                        logger.critical("=" * 70)
                        logger.critical("≡ƒÜ¿ NEAR EXPIRY MODE ACTIVATED")
                        logger.critical(f"License expires in {hours_until_expiration:.1f} hours")
                        logger.critical("NEW TRADES BLOCKED - Will only manage existing positions")
                        logger.critical("=" * 70)
                        
                        bot_status["near_expiry_mode"] = True
                        
                        # Send notification
                        try:
                            notifier = get_notifier()
                            notifier.send_error_alert(
                                error_message=f"≡ƒÜ¿ NEAR EXPIRY MODE\n\n"
                                             f"License expires in {hours_until_expiration:.1f} hours.\n"
                                             f"New trades are blocked.\n"
                                             f"Bot will only manage existing positions.\n\n"
                                             f"Please renew immediately.",
                                error_type="License Near Expiry - 2 Hours"
                            )
                        except Exception as e:
                            logger.debug(f"Failed to send near expiry notification: {e}")
                else:
                    # Clear near expiry mode if we have more than 2 hours
                    bot_status["near_expiry_mode"] = False
        
        elif response.status_code == 401:
            # Unauthorized - invalid license key  
            logger.critical("LICENSE VALIDATION FAILED - Invalid License Key")
            bot_status["license_expired"] = True
            bot_status["trading_enabled"] = False
            bot_status["emergency_stop"] = True
            bot_status["stop_reason"] = "License validation failed - invalid key"
        
        elif response.status_code == 403:
            # Session conflict - another device is using this license
            try:
                data = response.json()
                if data.get("session_conflict"):
                    logger.critical("=" * 70)
                    logger.critical("")
                    logger.critical("  ⚠️ LICENSE ALREADY IN USE - AUTO SHUTDOWN")
                    logger.critical("")
                    logger.critical("  Your license key is currently active on another device/instance.")
                    logger.critical("  Only one instance can use a license at a time.")
                    logger.critical("  This bot instance will now shut down automatically.")
                    logger.critical("")
                    logger.critical("  Contact: support@quotrading.com")
                    logger.critical("")
                    logger.critical("=" * 70)
                    bot_status["license_expired"] = True
                    bot_status["trading_enabled"] = False
                    bot_status["emergency_stop"] = True
                    bot_status["stop_reason"] = "License in use on another device"
                    
                    # Disconnect broker
                    if broker is not None:
                        try:
                            broker.disconnect()
                        except:
                            pass
                    
                    # Exit the bot completely
                    logger.critical("Shutting down bot in 5 seconds...")
                    time_module.sleep(5)
                    logger.critical("BOT SHUTDOWN - Session conflict detected")
                    sys.exit(1)  # Exit with error code
                else:
                    logger.critical("LICENSE VALIDATION FAILED - Forbidden")
                    bot_status["license_expired"] = True
                    bot_status["trading_enabled"] = False
                    bot_status["emergency_stop"] = True
                    bot_status["stop_reason"] = "License validation failed - forbidden"
            except:
                logger.critical("LICENSE VALIDATION FAILED - Forbidden")
                bot_status["license_expired"] = True
                bot_status["trading_enabled"] = False
                bot_status["emergency_stop"] = True
                bot_status["stop_reason"] = "License validation failed - forbidden"
        
        else:
            # Other error - log but don't stop trading (could be temporary API issue)
            logger.warning(f"ΓÜá∩╕Å License validation returned HTTP {response.status_code} - continuing for now")
    
    except requests.Timeout:
        # Timeout - don't stop trading, could be temporary network issue
        logger.warning("ΓÅ▒∩╕Å License validation timeout - will retry in 5 minutes")
    
    except Exception as e:
        # Other error - log but continue trading
        logger.warning(f"License validation error: {e} - will retry in 5 minutes")


def send_heartbeat() -> None:
    """
    Send bot heartbeat to cloud API for online status tracking.
    Called every 20 seconds to show bot is alive.
    Admin dashboard uses this to show online users and performance.
    
    Uses current_trading_symbol for symbol-specific session (multi-symbol support).
    """
    global broker, current_trading_symbol
    
    # Skip in backtest mode
    if is_backtest_mode():
        return
    
    try:
        import requests
        
        api_url = os.getenv("QUOTRADING_API_URL", "https://quotrading-flask-api.azurewebsites.net")
        license_key = os.getenv("QUOTRADING_LICENSE_KEY")
        
        if not license_key:
            return
        
        # Use current_trading_symbol if set, otherwise fall back to CONFIG
        symbol = current_trading_symbol if current_trading_symbol else CONFIG.get("instrument", "ES")
        
        # Get session stats if symbol state exists
        session_pnl = 0.0
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        current_position = 0
        position_pnl = 0.0
        
        if symbol in state:
            session_stats = state[symbol].get("session_stats", {})
            session_pnl = session_stats.get("total_pnl", 0.0)
            total_trades = len(session_stats.get("trades", []))
            winning_trades = session_stats.get("win_count", 0)
            losing_trades = session_stats.get("loss_count", 0)
            
            # Get current position info
            position = state[symbol].get("position", {})
            if position.get("active", False):
                qty = position.get("quantity", 0)
                current_position = qty if position.get("side") == "long" else -qty
                
                # Calculate unrealized P&L if we have current price
                entry_price = position.get("entry_price", 0)
                current_price = state[symbol].get("last_price", 0)
                if entry_price and current_price:
                    tick_value = CONFIG.get("tick_value", 12.50)
                    if position.get("side") == "long":
                        position_pnl = (current_price - entry_price) * qty * tick_value
                    else:
                        position_pnl = (entry_price - current_price) * qty * tick_value
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Send heartbeat with bot status and performance
        # Use symbol-specific fingerprint for multi-symbol session support
        # MULTI-SYMBOL FIX: Include symbol at top level so server can manage per-symbol sessions
        payload = {
            "license_key": license_key,
            "device_fingerprint": get_device_fingerprint(symbol),  # For session locking (symbol-specific)
            "symbol": symbol,  # MULTI-SYMBOL: Explicit symbol for server-side session management
            "bot_version": "2.0.0",
            "status": "online" if bot_status.get("trading_enabled", False) else "idle",
            "metadata": {
                "symbol": symbol,
                "shadow_mode": _bot_config.shadow_mode,
                # Real-time performance metrics
                "session_pnl": round(session_pnl, 2),
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": round(win_rate, 1),
                "current_position": current_position,
                "position_pnl": round(position_pnl, 2),
                # License status indicators
                "license_expired": bot_status.get("license_expired", False),
                "license_grace_period": bot_status.get("license_grace_period", False),
                "near_expiry_mode": bot_status.get("near_expiry_mode", False),
                "days_until_expiration": bot_status.get("days_until_expiration"),
                "hours_until_expiration": bot_status.get("hours_until_expiration")
            }
        }
        
        pass  # Silent - heartbeat is internal health check
        
        response = requests.post(
            f"{api_url}/api/heartbeat",
            json=payload,
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Check for session conflict (license in use on another device)
            if data.get("session_conflict", False):
                logger.critical("=" * 70)
                logger.critical("")
                logger.critical("  ⚠️ LICENSE ALREADY IN USE - AUTO SHUTDOWN")
                logger.critical("")
                logger.critical("  Your license key is currently active on another device/instance.")
                logger.critical(f"  Active device: {data.get('active_device', 'Unknown')}")
                logger.critical("")
                logger.critical("  Only one instance can use a license at a time.")
                logger.critical("  This bot instance will now shut down automatically.")
                logger.critical("")
                logger.critical("  Contact: support@quotrading.com")
                logger.critical("")
                logger.critical("="  * 70)
                
                # Disable trading and shut down
                bot_status["trading_enabled"] = False
                bot_status["emergency_stop"] = True
                bot_status["stop_reason"] = "session_conflict"
                
                # Disconnect broker
                if broker is not None:
                    try:
                        broker.disconnect()
                    except:
                        pass
                
                # Exit the bot completely
                logger.critical("Shutting down bot in 5 seconds...")
                time_module.sleep(5)
                logger.critical("BOT SHUTDOWN - Session conflict detected")
                sys.exit(1)  # Exit with error code
            
            logger.debug("Heartbeat sent successfully")
        elif response.status_code == 403:
            # Session conflict detected by server
            logger.critical("=" * 70)
            logger.critical("")
            logger.critical("  ⚠️ LICENSE ALREADY IN USE - AUTO SHUTDOWN")
            logger.critical("")
            logger.critical("  Your license key is currently active on another device/instance.")
            logger.critical("  Only one instance can use a license at a time.")
            logger.critical("  This bot instance will now shut down automatically.")
            logger.critical("")
            logger.critical("  Contact: support@quotrading.com")
            logger.critical("")
            logger.critical("=" * 70)
            
            # Disable trading and shut down
            bot_status["trading_enabled"] = False
            bot_status["emergency_stop"] = True
            bot_status["stop_reason"] = "license_conflict"
            
            # Disconnect broker
            if broker is not None:
                try:
                    broker.disconnect()
                except:
                    pass
            
            # Exit the bot completely
            logger.critical("Shutting down bot in 5 seconds...")
            time_module.sleep(5)
            logger.critical("BOT SHUTDOWN - Session conflict detected")
            sys.exit(1)  # Exit with error code
        else:
            logger.debug(f"Heartbeat returned HTTP {response.status_code}")
    
    except Exception as e:
        logger.debug(f"Heartbeat error: {e}")



def handle_shutdown_event(data: Dict[str, Any]) -> None:
    """Handle shutdown event"""
    logger.info("Shutdown event received")
    bot_status["trading_enabled"] = False


def release_session() -> None:
    """Release session lock - called on ANY exit.
    
    Uses current_trading_symbol for symbol-specific session (multi-symbol support).
    Falls back to CONFIG symbol if current_trading_symbol is not set.
    """
    global current_trading_symbol
    
    try:
        import requests
        license_key = os.getenv("QUOTRADING_LICENSE_KEY")
        if license_key:
            api_url = os.getenv("QUOTRADING_API_URL", "https://quotrading-flask-api.azurewebsites.net")
            
            # Use symbol-specific fingerprint for multi-symbol session support
            # MULTI-SYMBOL FIX: Include symbol explicitly so server releases the correct session
            symbol_for_session = get_current_symbol_for_session()
            response = requests.post(
                f"{api_url}/api/session/release",
                json={
                    "license_key": license_key,
                    "device_fingerprint": get_device_fingerprint(symbol_for_session),
                    "symbol": symbol_for_session  # MULTI-SYMBOL: Explicit symbol for server-side session management
                },
                timeout=5
            )
            if response.status_code == 200:
                pass  # Silent - session released
            else:
                pass  # Silent - release failed
    except Exception as e:
        logger.warning(f"⚠️ Error releasing session lock: {e}")


def cleanup_on_shutdown() -> None:
    """Cleanup tasks on shutdown"""
    # Track cleanup success
    cleanup_success = True
    
    # Release session lock
    try:
        release_session()
    except Exception as e:
        cleanup_success = False
        logger.debug(f"Failed to release session: {e}")
    
    # Send bot shutdown alert
    try:
        notifier = get_notifier()
        notifier.send_error_alert(
            error_message="Bot is shutting down. All positions should be closed before shutdown.",
            error_type="Bot Shutdown"
        )
    except Exception as e:
        logger.debug(f"Failed to send shutdown alert: {e}")
    
    # Save state to disk
    if recovery_manager:
        try:
            recovery_manager.save_state(state)
        except Exception as e:
            cleanup_success = False
            logger.debug(f"Failed to save state: {e}")
    
    # Disconnect broker
    if broker and broker.is_connected():
        try:
            broker.disconnect()
        except Exception as e:
            cleanup_success = False
            logger.debug(f"Failed to disconnect broker: {e}")
    
    # Stop timer manager
    if timer_manager:
        try:
            timer_manager.stop()
        except Exception as e:
            cleanup_success = False
            logger.debug(f"Failed to stop timer: {e}")
    
    # Log session summary with logout status
    symbol = CONFIG["instrument"]
    if symbol in state:
        log_session_summary(symbol, cleanup_success)
    else:
        # No session to summarize, just log logout status
        logger.info("")
        if cleanup_success:
            logger.info("\033[92m✓ Logged out successfully\033[0m")  # Green
        else:
            logger.info("\033[91m✗ Logout completed with errors\033[0m")  # Red
        logger.info("")


if __name__ == "__main__":
    # Parse command-line arguments for multi-symbol support
    # Usage: python src/quotrading_engine.py [SYMBOL]
    # Example: python src/quotrading_engine.py ES
    #          python src/quotrading_engine.py NQ
    parser = argparse.ArgumentParser(
        description='QuoTrading AI - Professional Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/quotrading_engine.py           # Uses symbol from config/env
  python src/quotrading_engine.py ES        # Trade ES (E-mini S&P 500)
  python src/quotrading_engine.py NQ        # Trade NQ (E-mini Nasdaq)
  python src/quotrading_engine.py MES       # Trade MES (Micro E-mini S&P)

Multi-Symbol Mode:
  Launch separate terminals for each symbol:
  - Window 1: python src/quotrading_engine.py ES
  - Window 2: python src/quotrading_engine.py NQ
  
  Each window uses symbol-specific RL data from experiences/{symbol}/
        """
    )
    parser.add_argument(
        'symbol',
        nargs='?',
        default=None,
        help='Trading symbol (e.g., ES, NQ, MES, MNQ). If not provided, uses symbol from config.'
    )
    
    args = parser.parse_args()
    
    # Display rainbow logo IMMEDIATELY when PowerShell opens (no initial clear screen)
    # This ensures the logo appears instantly instead of showing a black screen first
    # SKIP in backtest mode to prevent spam
    if RAINBOW_LOGO_AVAILABLE and not is_backtest_mode():
        try:
            # Show logo immediately without clearing first - instant display
            # This creates a professional loading screen effect
            display_animated_logo(duration=STARTUP_LOGO_DURATION, fps=20, with_headers=False)
            
            # Clear screen after logo to make room for logs
            if os.name == 'nt':
                os.system('cls')
            else:
                os.system('clear')
        except Exception as e:
            # Logo display failed - log and continue (not critical)
            # Use logger if available, otherwise print
            try:
                logger.debug(f"Could not display startup logo: {e}")
            except:
                pass  # Logger not initialized yet, silently continue
    
    # Pass symbol from command line to main function
    main(symbol_override=args.symbol)
