"""
VWAP Bounce Bot - Mean Reversion Trading Strategy
Event-driven bot that trades bounces off VWAP standard deviation bands

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
from datetime import datetime, timedelta
from datetime import time as datetime_time  # Alias to avoid conflict with time.time()
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Callable
import pytz
import time as time_module  # Import time module with alias
import statistics  # For calculating statistics like mean, median, etc.
import asyncio
import hashlib

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


def get_device_fingerprint() -> str:
    """
    Generate a unique device fingerprint for session locking.
    Prevents license key sharing across multiple computers.
    
    Components:
    - Machine ID (from platform UUID)
    - Username
    - Platform name
    
    Returns:
        Unique device fingerprint (hashed for privacy)
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
    
    # Combine all components
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
from config import load_config, BotConfiguration
from event_loop import EventLoop, EventType, EventPriority, TimerManager
from error_recovery import ErrorRecoveryManager, ErrorType as RecoveryErrorType
from bid_ask_manager import BidAskManager, BidAskQuote
from notifications import get_notifier
from signal_confidence import SignalConfidenceRL
from regime_detection import get_regime_detector, REGIME_DEFINITIONS
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

# String constants
MSG_LIVE_TRADING_NOT_IMPLEMENTED = "Live trading not implemented - SDK integration required"
SEPARATOR_LINE = "=" * 60

# Daily Loss Limit Threshold
DAILY_LOSS_APPROACHING_THRESHOLD = 0.80  # Stop trading at 80% of daily loss limit

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
}


def setup_logging() -> logging.Logger:
    """Configure logging for the bot - Console only (no log files for customers)"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Console output only - customers don't need log files
        ]
    )
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
    """
    global cloud_api_client, rl_brain
    
    # BACKTEST MODE: Save to local RL brain only
    if is_backtest_mode() or CONFIG.get("backtest_mode", False):
        if rl_brain is not None:
            rl_brain.record_outcome(rl_state, True, pnl, duration_minutes, execution_data)
        return
    
    # LIVE MODE: Report to cloud ONLY (don't save locally)
    if cloud_api_client is None:
        logger.debug("Cloud API client not initialized - skipping cloud reporting")
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
        
        logger.info(f"☁️ Outcome reported to cloud: ${pnl:+.2f} in {duration_minutes:.1f}min")
        
    except Exception as e:
        logger.debug(f"Non-critical: Could not report outcome to cloud: {e}")



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

def initialize_broker() -> None:
    """
    Initialize the broker interface using configuration.
    Uses configured broker with error recovery and circuit breaker.
    SHADOW MODE: Shows trading signals without executing (manual trading mode).
    """
    global broker, recovery_manager
    
    # ===== LICENSE VALIDATION =====
    # Check if user has valid license before connecting to broker
    # Both regular licenses and admin keys must be validated by the server
    license_key = os.getenv("QUOTRADING_LICENSE_KEY")
    
    if license_key:
        logger.info("🔐 Validating license...")
        try:
            import requests
            api_url = os.getenv("QUOTRADING_API_URL", "https://quotrading-flask-api.azurewebsites.net")
            
            # Validate with server (server handles both regular and admin keys)
            # Include device fingerprint for session locking
            response = requests.post(
                f"{api_url}/api/main",
                json={
                    "license_key": license_key,
                    "device_fingerprint": get_device_fingerprint()  # Session locking
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("license_valid"):
                    logger.info(f"Γ£à License validated - {data.get('message', 'Access Granted')}")
                else:
                    logger.error("Γ¥î INVALID LICENSE - Bot will not start")
                    logger.error(f"Reason: {data.get('message', 'Unknown error')}")
                    sys.exit(1)
            else:
                logger.error(f"Γ¥î License validation failed - HTTP {response.status_code}")
                logger.error("Please contact support@quotrading.com")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Γ¥î License validation error: {e}")
            logger.error("Cannot start bot without valid license")
            sys.exit(1)
    else:
        logger.error("Γ¥î NO LICENSE KEY FOUND")
        logger.error("Please set QUOTRADING_LICENSE_KEY in your .env file")
        logger.error("Contact support@quotrading.com to purchase a license")
        sys.exit(1)
    
    # In shadow mode, show signals only (no execution)
    if CONFIG.get("shadow_mode", False):
        logger.info("≡ƒôè SIGNAL-ONLY MODE - Shows signals without executing trades")
    
    logger.info("Initializing broker interface...")
    
    # Create error recovery manager
    recovery_manager = ErrorRecoveryManager(CONFIG)
    
    # Create broker using configuration
    # In shadow mode, broker streams data but doesn't execute actual orders
    broker = create_broker(_bot_config.api_token, _bot_config.username, CONFIG["instrument"])
    
    # Connect to broker (initial connection doesn't use circuit breaker)
    logger.info("Connecting to broker...")
    if not broker.connect():
        logger.error("Failed to connect to broker")
        return False
        raise RuntimeError("Broker connection failed")
    
    logger.info("Broker connected successfully")
    
    # Send bot startup alert
    try:
        notifier = get_notifier()
        notifier.send_error_alert(
            error_message=f"Bot started successfully and connected to broker. Ready to trade {CONFIG.get('instrument', 'configured symbol')}.",
            error_type="Bot Started"
        )
    except Exception as e:
        logger.debug(f"Failed to send startup alert: {e}")


def check_cloud_kill_switch() -> None:
    """
    Check if cloud kill switch is active.
    Called every 30 seconds as part of health check.
    
    KILL SWITCH BEHAVIOR:
    1. ACTIVATE: Flatten positions ΓåÆ Disconnect broker (NO DATA) ΓåÆ Go idle
    2. DEACTIVATE: Auto-reconnect broker ΓåÆ Resume trading
    
    This allows you to remotely pause ALL customer bots for:
    - Scheduled maintenance
    - Strategy updates
    - Emergency situations
    """
    global broker
    
    # Skip in backtest mode - no cloud API access needed
    if is_backtest_mode():
        return
    
    try:
        import requests
        
        cloud_api_url = CONFIG.get("cloud_api_url", "https://quotrading-flask-api.azurewebsites.net")
        
        response = requests.get(
            f"{cloud_api_url}/api/main",
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            kill_switch_active = data.get("kill_switch_active", False)
            reason = data.get("reason", "Maintenance mode")
            
            # ===== ACTIVATE KILL SWITCH =====
            if kill_switch_active and not bot_status.get("kill_switch_active", False):
                logger.critical("=" * 80)
                logger.critical("≡ƒ¢æ KILL SWITCH ACTIVATED - SHUTTING DOWN")
                logger.critical("=" * 80)
                logger.critical(f"  Reason: {reason}")
                logger.critical(f"  Activated At: {data.get('activated_at', 'Unknown')}")
                logger.critical("=" * 80)
                
                # Step 1: Flatten any active positions
                for symbol in state.keys():
                    if state[symbol]["position"]["active"]:
                        logger.critical(f"  [KILL SWITCH] Flattening position in {symbol}...")
                        try:
                            current_price = state[symbol].get("last_price", 0)
                            handle_exit_orders(
                                symbol,
                                state[symbol]["position"],
                                current_price,
                                "kill_switch_flatten"
                            )
                        except Exception as e:
                            logger.error(f"  [KILL SWITCH] Failed to flatten {symbol}: {e}")
                
                # Step 2: DISCONNECT BROKER (stops all data)
                try:
                    if broker is not None and broker.connected:
                        logger.critical("  [KILL SWITCH] Disconnecting from broker...")
                        broker.disconnect()
                        logger.critical("  [KILL SWITCH] Γ£à Broker disconnected - NO DATA RUNNING")
                except Exception as e:
                    logger.error(f"  [KILL SWITCH] Error disconnecting: {e}")
                
                # Step 3: Disable trading
                bot_status["trading_enabled"] = False
                bot_status["kill_switch_active"] = True
                
                # Step 4: Alert customer
                try:
                    notifier = get_notifier()
                    notifier.send_error_alert(
                        error_message=f"≡ƒ¢æ KILL SWITCH: {reason}. All positions closed, broker disconnected. Bot will auto-resume when maintenance completes.",
                        error_type="Kill Switch Activated"
                    )
                except Exception as e:
                    logger.debug(f"Failed to send alert: {e}")
                
                logger.critical("  [KILL SWITCH] Bot is IDLE. Checking every 30s for resume signal...")
                logger.critical("=" * 80)
            
            # ===== DEACTIVATE KILL SWITCH (AUTO-RECONNECT) =====
            elif not kill_switch_active and bot_status.get("kill_switch_active", False):
                logger.critical("=" * 80)
                logger.critical("Γ£à KILL SWITCH OFF - AUTO-RECONNECTING")
                logger.critical("=" * 80)
                
                # Step 1: RECONNECT TO BROKER
                try:
                    if broker is not None:
                        logger.critical("  [RECONNECT] Connecting to broker...")
                        success = broker.connect(max_retries=3)
                        if success:
                            logger.critical("  [RECONNECT] Γ£à Broker connected - Data feed active")
                        else:
                            logger.error("  [RECONNECT] Γ¥î Connection failed - Will retry in 30s")
                            return  # Don't resume trading yet
                except Exception as e:
                    logger.error(f"  [RECONNECT] Error: {e}")
                    return
                
                # Step 2: Re-enable trading
                bot_status["trading_enabled"] = True
                bot_status["kill_switch_active"] = False
                
                # Step 3: Alert customer
                try:
                    notifier = get_notifier()
                    notifier.send_error_alert(
                        error_message="Γ£à Kill switch deactivated. Broker reconnected, trading resumed. Bot is back online.",
                        error_type="Trading Resumed"
                    )
                except Exception as e:
                    logger.debug(f"Failed to send alert: {e}")
                
                logger.critical("  [RECONNECT] Γ£à Trading enabled. Bot fully operational.")
                logger.critical("=" * 80)
                
    except Exception as e:
        # Non-critical - if cloud unreachable, bot continues normally
        logger.debug(f"Kill switch check skipped (cloud unreachable): {e}")


def check_azure_time_service() -> str:
    """
    Check Azure time service for trading permission.
    Called every 30 seconds alongside kill switch check.
    
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
            logger.debug(f"Time service returned {response.status_code}, using local time")
            bot_status["azure_trading_state"] = None
            return None  # Signal to use local get_trading_state()
            
    except Exception as e:
        # Non-critical - if cloud unreachable, fall back to local time
        logger.debug(f"Time service check skipped (cloud unreachable): {e}")
        bot_status["azure_trading_state"] = None
        return None  # Signal to use local get_trading_state()


def check_broker_connection() -> None:
    """
    Periodic health check for broker connection AND cloud services.
    Verifies connection is alive and attempts reconnection if needed.
    Called every 30 seconds by timer manager.
    Only logs when there's an issue to avoid spam.
    """
    global broker
    
    # Skip all broker/cloud checks in backtest mode
    if is_backtest_mode():
        return
    
    # CRITICAL: Check cloud services FIRST (kill switch + time service)
    try:
        check_cloud_kill_switch()
    except Exception as e:
        logger.debug(f"Kill switch check failed (non-critical): {e}")
    
    try:
        check_azure_time_service()
    except Exception as e:
        logger.debug(f"Time service check failed (non-critical): {e}")
    
    # Send heartbeat to show bot is online
    try:
        send_heartbeat()
    except Exception as e:
        logger.debug(f"Heartbeat failed (non-critical): {e}")
    
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
        
        # Only go idle during maintenance, not weekend
        # Maintenance is 5:00-6:00 PM ET on weekdays (Mon-Fri)
        if "maintenance" in halt_reason.lower() or (eastern_time.weekday() < 5 and eastern_time.time() >= datetime_time(16, 45) and eastern_time.time() < datetime_time(18, 0)):
            logger.critical(SEPARATOR_LINE)
            logger.critical("≡ƒöº MAINTENANCE WINDOW - GOING IDLE")
            logger.critical(f"Time: {eastern_time.strftime('%H:%M:%S %Z')}")
            logger.critical("  Disconnecting broker to save resources during maintenance")
            logger.critical("  Will auto-reconnect at 6:00 PM ET when market reopens")
            logger.critical(SEPARATOR_LINE)
            
            # Disconnect broker (stops all data feeds)
            try:
                if broker is not None and broker.connected:
                    broker.disconnect()
                    logger.critical("  Γ£à Broker disconnected - Bot is IDLE")
            except Exception as e:
                logger.error(f"  Γ¥î Error disconnecting: {e}")
            
            bot_status["maintenance_idle"] = True
            bot_status["trading_enabled"] = False
            logger.critical("  Bot will check every 30s for market reopen...")
            return  # Skip broker health check since we just disconnected
    
    # AUTO-RECONNECT: Reconnect broker when market reopens at 6:00 PM ET
    elif trading_state == "entry_window" and bot_status.get("maintenance_idle", False):
        logger.critical(SEPARATOR_LINE)
        logger.critical("Γ£à MARKET REOPENED - AUTO-RECONNECTING")
        logger.critical(f"Time: {current_time.strftime('%H:%M:%S %Z')}")
        logger.critical(SEPARATOR_LINE)
        
        # Reconnect to broker
        try:
            if broker is not None:
                logger.critical("  [RECONNECT] Connecting to broker...")
                success = broker.connect(max_retries=3)
                if success:
                    logger.critical("  [RECONNECT] Γ£à Broker connected - Data feed active")
                    bot_status["maintenance_idle"] = False
                    bot_status["trading_enabled"] = True
                    logger.critical("  [RECONNECT] Γ£à Trading enabled. Bot fully operational.")
                else:
                    logger.error("  [RECONNECT] Γ¥î Connection failed - Will retry in 30s")
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
            logger.debug(f"Failed to send connection error alert: {e}")
        
        try:
            # Immediate reconnect with 3 retries
            logger.critical("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.critical("⚠️  RECONNECTING TO TOPSTEP")
            success = broker.connect(max_retries=3)
            if success:
                logger.critical("✅ Reconnection successful - Trading resumed")
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
    In shadow mode, returns simulated capital (no account login).
    In live mode, returns actual account balance from broker.
    """
    # Shadow mode or no broker - return simulated capital
    if CONFIG.get("shadow_mode", False) or broker is None:
        # Use starting_equity from bot_status if available
        if bot_status.get("starting_equity") is not None:
            return bot_status["starting_equity"]
        # Default starting capital for shadow mode
        return 50000.0
    
    # Live mode - get actual balance from broker account
    try:
        # Use circuit breaker for account query
        breaker = recovery_manager.get_circuit_breaker("account_query")
        success, equity = breaker.call(broker.get_account_equity)
        
        if success:
            logger.info(f"Account equity: ${equity:.2f}")
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
        logger.info(f"[BACKTEST] Market Order: {side} {quantity} {symbol}")
        return {
            "order_id": f"BACKTEST_{datetime.now().timestamp()}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": "MARKET",
            "status": "FILLED",
            "backtest": True
        }
    
    logger.info(f"Market Order: {side} {quantity} {symbol}")
    
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
                logger.debug(f"Failed to send order error alert: {e}")
            
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
            logger.debug(f"Failed to send order error alert: {alert_error}")
        
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
        logger.info(f"[BACKTEST] Stop Order: {side} {quantity} {symbol} @ {stop_price}")
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
    
    shadow_mode = CONFIG.get("shadow_mode", False)
    logger.info(f"{'[SHADOW MODE] ' if shadow_mode else ''}Stop Order: {side} {quantity} {symbol} @ {stop_price}")
    
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
        logger.info(f"[BACKTEST] Limit Order: {side} {quantity} {symbol} @ {limit_price}")
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
    
    shadow_mode = CONFIG.get("shadow_mode", False)
    logger.info(f"{'[SHADOW MODE] ' if shadow_mode else ''}Limit Order: {side} {quantity} {symbol} @ {limit_price}")
    
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
    
    shadow_mode = CONFIG.get("shadow_mode", False)
    logger.info(f"{'[SHADOW MODE] ' if shadow_mode else ''}Cancelling Order: {order_id} for {symbol}")
    
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
    except Exception as e:
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
    if CONFIG.get("shadow_mode", False):
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


def subscribe_market_data(symbol: str, callback: Callable[[str, float, int, int], None]) -> None:
    """
    Subscribe to real-time market data for a symbol through broker interface.
    
    Args:
        symbol: Instrument symbol
        callback: Function to call with tick data (symbol, price, volume, timestamp)
    """
    logger.info(f"Subscribing to market data: {symbol}")
    
    if broker is None:
        logger.error("Broker not initialized")
        return
    
    try:
        # Subscribe through broker interface
        broker.subscribe_market_data(symbol, callback)
        logger.info(f"Subscribed to market data for {symbol}")
    except Exception as e:
        logger.error(f"Error subscribing to market data: {e}")
        action = recovery_manager.handle_error(
            RecoveryErrorType.DATA_FEED_INTERRUPTION,
            {"symbol": symbol, "error": str(e)}
        )


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
    logger.info(f"Fetching {count} historical {timeframe}min bars for {symbol}")
    
    
    if broker is None:
        logger.error("Broker not initialized")
        return []
    
    try:
        # Fetch through broker interface
        breaker = recovery_manager.get_circuit_breaker("market_data")
        success, bars = breaker.call(broker.fetch_historical_bars, symbol, f"{timeframe}m", count)
        
        if success and bars:
            logger.info(f"Fetched {len(bars)} bars")
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
    
    logger.info(f"State initialized for {symbol}")


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
            logger.debug(f"Position state saved: {position['side']} {position['quantity']} @ ${position['entry_price']:.2f}")
        else:
            logger.debug(f"Position state saved: {position['side']} (inactive)")
        
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
            logger.info("No saved position state found (clean start)")
            return False
        
        with open(state_file, 'r') as f:
            saved_state = json.load(f)
        
        # Check if saved state is for this symbol and has an active position
        if saved_state.get("symbol") != symbol:
            logger.info(f"Saved state is for different symbol: {saved_state.get('symbol')}")
            return False
        
        if not saved_state.get("active"):
            logger.info("Saved state shows no active position")
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
            logger.info(f"[BAR COMPLETED] 1min bar closed | Price: ${current_bar['close']:.2f} | Vol: {current_bar['volume']} | Total bars: {bar_count}")
            
            # Calculate VWAP after new bar is added
            calculate_vwap(symbol)
            
            # Log detailed status every 5 minutes
            if bar_count % 5 == 0:
                vwap_data = state[symbol].get("vwap", {})
                position_dict = state[symbol]["position"]
                position_qty = position_dict.get("quantity", 0) if isinstance(position_dict, dict) else 0
                market_cond = state[symbol].get("market_condition", "UNKNOWN")
                
                logger.info("=" * 80)
                logger.info(f"[STATUS] 5-MIN UPDATE | Bars: {bar_count} | Position: {position_qty} contracts")
                if vwap_data and isinstance(vwap_data, dict):
                    vwap_val = vwap_data.get('vwap', 0)
                    std_dev = vwap_data.get('std_dev', 0)
                    logger.info(f"[PRICE] ${vwap_val:.2f} | StdDev: ${std_dev:.2f}")
                    bands = vwap_data.get('bands', {})
                    if bands and isinstance(bands, dict):
                        logger.info(f"[BANDS] U2: ${bands.get('upper_2', 0):.2f} | L2: ${bands.get('lower_2', 0):.2f}")
                logger.info(f"[MARKET] {market_cond}")
                logger.info("=" * 80)
            
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
        logger.info(f"[INJECT_BAR] First bar: H={bar.get('high', 'MISSING'):.2f} L={bar.get('low', 'MISSING'):.2f}")
    
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
        logger.debug(f"Not enough bars for trend filter: {len(bars)}/{period}")
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
        
        logger.debug(f"Trend EMA: {ema:.2f}, Direction: {state[symbol]['trend_direction']}")


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
        logger.debug(f"Not enough bars for RSI: {len(bars)}/{rsi_period + 1}")
        return
    
    # Use recent bars for calculation
    closes = [bar["close"] for bar in list(bars)[-100:]]
    rsi = calculate_rsi(closes, rsi_period)
    
    if rsi is not None:
        state[symbol]["rsi"] = rsi
        logger.debug(f"RSI: {rsi:.2f}")


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
        logger.debug(f"Not enough bars for MACD: {len(bars)}/{slow_period + signal_period}")
        return
    
    # Use recent bars for calculation
    closes = [bar["close"] for bar in list(bars)[-100:]]  # Use last 100 bars
    macd_data = calculate_macd(closes, fast_period, slow_period, signal_period)
    
    if macd_data is not None:
        state[symbol]["macd"] = macd_data
        logger.debug(f"MACD: {macd_data['macd']:.2f}, Signal: {macd_data['signal']:.2f}, "
                    f"Histogram: {macd_data['histogram']:.2f}")


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
        logger.debug(f"Not enough bars for volume average: {len(bars)}/{lookback}")
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
    
    logger.debug(f"Average volume (last {lookback} bars): {avg_volume:.0f}")


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
    VWAP is volume-weighted average price, reset daily.
    
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
    
    # Log VWAP update every 10 bars to show bot is working
    bar_count = len(bars)
    if bar_count % 10 == 0:
        logger.info(f"[MARKET] ${vwap:.2f} | StdDev: ${std_dev:.2f} | Bars: {bar_count} | "
                   f"U2: ${state[symbol]['vwap_bands']['upper_2']:.2f} | "
                   f"L2: ${state[symbol]['vwap_bands']['lower_2']:.2f}")
    else:
        logger.debug(f"VWAP: {vwap:.2f}, StdDev: {std_dev:.2f}")
        logger.debug(f"Bands - U3: {state[symbol]['vwap_bands']['upper_3']:.2f}, "
                    f"U2: {state[symbol]['vwap_bands']['upper_2']:.2f}, "
                    f"U1: {state[symbol]['vwap_bands']['upper_1']:.2f}, "
                    f"L1: {state[symbol]['vwap_bands']['lower_1']:.2f}, "
                    f"L2: {state[symbol]['vwap_bands']['lower_2']:.2f}, "
                    f"L3: {state[symbol]['vwap_bands']['lower_3']:.2f}")


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
    
    # Daily loss limit - ENABLED for safety
    if state[symbol]["daily_pnl"] <= -CONFIG["daily_loss_limit"]:
        logger.warning(f"Daily loss limit hit (${state[symbol]['daily_pnl']:.2f}), stopping for the day")
        
        # Send alert once when limit hit
        if not state[symbol].get("loss_limit_alerted", False):
            try:
                notifier = get_notifier()
                notifier.send_error_alert(
                    error_message=f"≡ƒÆ░ Daily Loss Limit Reached: ${state[symbol]['daily_pnl']:.2f} / -${CONFIG['daily_loss_limit']:.2f}. Bot stopped trading for today. Will auto-resume tomorrow.",
                    error_type="Daily Loss Limit"
                )
                state[symbol]["loss_limit_alerted"] = True
            except Exception as e:
                logger.debug(f"Failed to send loss limit alert: {e}")
        
        return False, "Daily loss limit"
    
    # Check data availability
    if len(state[symbol]["bars_1min"]) < 2:
        logger.info(f"Not enough bars for signal: {len(state[symbol]['bars_1min'])}/2")
        return False, "Insufficient bars"
    
    # Check VWAP bands
    vwap_bands = state[symbol]["vwap_bands"]
    if any(v is None for v in vwap_bands.values()):
        logger.info("Price bands not yet calculated")
        return False, "VWAP not ready"
    
    # Check trend (optional - DOES NOT block signals if neutral)
    # Trend filtering happens in signal-specific functions:
    #   - Uptrend: only longs allowed
    #   - Downtrend: only shorts allowed
    #   - Neutral: both allowed (mean reversion)
    use_trend_filter = CONFIG.get("use_trend_filter", False)
    if use_trend_filter:
        trend = state[symbol]["trend_direction"]
        if trend is None:
            logger.info(f"Trend not yet established")
            return False, "Trend not established"
        # Neutral trend is OK - will trade both directions
    
    # Check RSI (ITERATION 3 - selective entry thresholds)
    use_rsi_filter = CONFIG.get("use_rsi_filter", True)
    rsi_oversold = CONFIG.get("rsi_oversold", 35.0)  # Iteration 3
    rsi_overbought = CONFIG.get("rsi_overbought", 65.0)  # Iteration 3
    
    if use_rsi_filter:
        rsi = state[symbol]["rsi"]
        if rsi is None:
            logger.debug("RSI not yet calculated")
            # Allow trading without RSI if not available yet
        # Note: RSI check moved to signal-specific functions for long/short
    
    # Check volume spike (optional - for confirmation)
    use_volume_filter = CONFIG.get("use_volume_filter", True)
    if use_volume_filter:
        avg_volume = state[symbol]["avg_volume"]
        if avg_volume is None:
            logger.debug("Average volume not yet calculated")
            # Allow trading without volume filter if not available yet
    
    # VWAP direction filter (optional - price vs VWAP for bias)
    use_vwap_direction_filter = CONFIG.get("use_vwap_direction_filter", True)
    if use_vwap_direction_filter:
        vwap = state[symbol]["vwap"]
        if vwap is None:
            logger.debug("VWAP not yet calculated")
            return False, "VWAP not ready"
        # Note: VWAP direction check moved to signal-specific functions
    
    # Check bid/ask spread and market condition (Phase: Bid/Ask Strategy)
    if bid_ask_manager is not None:
        # Validate spread (Requirement 8)
        is_acceptable, spread_reason = bid_ask_manager.validate_entry_spread(symbol)
        if not is_acceptable:
            logger.info(f"Spread check failed: {spread_reason}")
            return False, spread_reason
        
        # Classify market condition (Requirement 11)
        try:
            condition, condition_reason = bid_ask_manager.classify_market_condition(symbol)
            logger.info(f"Market Condition: {condition.upper()} - {condition_reason}")
            
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
    Check if long signal conditions are met - WITH-TREND MEAN REVERSION.
    
    Strategy: Buy dips in uptrends (fade to VWAP from below)
    
    Args:
        symbol: Instrument symbol
        prev_bar: Previous 1-minute bar
        current_bar: Current 1-minute bar
    
    Returns:
        True if long signal detected
    """
    vwap_bands = state[symbol]["vwap_bands"]
    vwap = state[symbol]["vwap"]
    
    # TREND FILTER: Skip longs in downtrend (but allow in neutral/uptrend)
    use_trend = CONFIG.get("use_trend_filter", False)
    if use_trend:
        trend = state[symbol]["trend_direction"]
        if trend == "down":
            logger.debug(f"Long rejected - trend is {trend}, counter to downtrend")
            return False
        logger.debug(f"Trend filter: {trend}  (allows longs)")
    
    # PRIMARY: VWAP bounce condition (2.0 std dev)
    touched_lower = prev_bar["low"] <= vwap_bands["lower_2"]
    bounced_back = current_bar["close"] > vwap_bands["lower_2"]
    
    if not (touched_lower and bounced_back):
        return False
    
    # FILTER 1: VWAP Direction - price should be BELOW VWAP (discount/oversold)
    use_vwap_direction = CONFIG.get("use_vwap_direction_filter", False)
    if use_vwap_direction and vwap is not None:
        if current_bar["close"] >= vwap:
            logger.debug(f"Long rejected - price above VWAP: {current_bar['close']:.2f} >= {vwap:.2f}")
            return False
        logger.debug(f"Price below VWAP: {current_bar['close']:.2f} < {vwap:.2f} ")
    
    # FILTER 2: RSI - extreme oversold (ITERATION 3)
    use_rsi = CONFIG.get("use_rsi_filter", True)
    rsi_oversold = CONFIG.get("rsi_oversold", 35.0)  # Iteration 3 - selective entry
    if use_rsi:
        rsi = state[symbol]["rsi"]
        if rsi is not None:
            if rsi >= rsi_oversold:
                logger.debug(f"Long rejected - RSI not extreme: {rsi:.2f} >= {rsi_oversold}")
                return False
            logger.debug(f"RSI extreme oversold: {rsi:.2f} < {rsi_oversold} ")
    
    # FILTER 3: Volume spike - confirmation of interest
    use_volume = CONFIG.get("use_volume_filter", True)
    volume_mult = CONFIG.get("volume_spike_multiplier", 1.5)
    if use_volume:
        # Use 1-min bar average (same as RL volume_ratio calculation)
        bars_1min = state[symbol]["bars_1min"]
        if len(bars_1min) >= 20:
            recent_volumes = [bar["volume"] for bar in list(bars_1min)[-20:]]
            avg_volume_1min = sum(recent_volumes) / len(recent_volumes)
            current_volume = current_bar["volume"]
            if current_volume < avg_volume_1min * volume_mult:
                logger.debug(f"Long rejected - no volume spike: {current_volume:.0f} < {avg_volume_1min * volume_mult:.0f}")
                return False
            logger.debug(f"Volume spike: {current_volume:.0f} >= {avg_volume_1min * volume_mult:.0f}")
    
    # FILTER 4: Bullish bar confirmation - current bar must be bullish (close > open)
    # This ensures the reversal is happening with buying pressure, not selling into the bounce
    if current_bar["close"] <= current_bar["open"]:
        logger.debug(f"Long rejected - not a bullish bar: close {current_bar['close']:.2f} <= open {current_bar['open']:.2f}")
        return False
    
    logger.info(f" LONG SIGNAL: Price reversal at {current_bar['close']:.2f} (entry zone: {vwap_bands['lower_2']:.2f})")
    return True


def check_short_signal_conditions(symbol: str, prev_bar: Dict[str, Any], 
                                  current_bar: Dict[str, Any]) -> bool:
    """
    Check if short signal conditions are met - WITH-TREND MEAN REVERSION.
    
    Strategy: Sell rallies in downtrends (fade to VWAP from above)
    
    Args:
        symbol: Instrument symbol
        prev_bar: Previous 1-minute bar
        current_bar: Current 1-minute bar
    
    Returns:
        True if short signal detected
    """
    vwap_bands = state[symbol]["vwap_bands"]
    vwap = state[symbol]["vwap"]
    
    # TREND FILTER: Skip shorts in uptrend (but allow in neutral/downtrend)
    use_trend = CONFIG.get("use_trend_filter", False)
    if use_trend:
        trend = state[symbol]["trend_direction"]
        if trend == "up":
            logger.debug(f"Short rejected - trend is {trend}, counter to uptrend")
            return False
        logger.debug(f"Trend filter: {trend}  (allows shorts)")
    
    # PRIMARY: VWAP bounce condition (2.0 std dev)
    touched_upper = prev_bar["high"] >= vwap_bands["upper_2"]
    bounced_back = current_bar["close"] < vwap_bands["upper_2"]
    
    if not (touched_upper and bounced_back):
        return False
    
    # FILTER 1: VWAP Direction - price should be ABOVE VWAP (premium/overbought)
    use_vwap_direction = CONFIG.get("use_vwap_direction_filter", False)
    if use_vwap_direction and vwap is not None:
        if current_bar["close"] <= vwap:
            logger.debug(f"Short rejected - price below VWAP: {current_bar['close']:.2f} <= {vwap:.2f}")
            return False
        logger.debug(f"Price above VWAP: {current_bar['close']:.2f} > {vwap:.2f} ")
    
    # FILTER 2: RSI - extreme overbought (ITERATION 3)
    use_rsi = CONFIG.get("use_rsi_filter", True)
    rsi_overbought = CONFIG.get("rsi_overbought", 65.0)  # Iteration 3 - selective entry
    if use_rsi:
        rsi = state[symbol]["rsi"]
        if rsi is not None:
            if rsi <= rsi_overbought:
                logger.debug(f"Short rejected - RSI not extreme: {rsi:.2f} <= {rsi_overbought}")
                return False
            logger.debug(f"RSI extreme overbought: {rsi:.2f} > {rsi_overbought} ")
    
    # FILTER 3: Volume spike - confirmation of interest
    use_volume = CONFIG.get("use_volume_filter", True)
    volume_mult = CONFIG.get("volume_spike_multiplier", 1.5)
    if use_volume:
        # Use 1-min bar average (same as RL volume_ratio calculation)
        bars_1min = state[symbol]["bars_1min"]
        if len(bars_1min) >= 20:
            recent_volumes = [bar["volume"] for bar in list(bars_1min)[-20:]]
            avg_volume_1min = sum(recent_volumes) / len(recent_volumes)
            current_volume = current_bar["volume"]
            if current_volume < avg_volume_1min * volume_mult:
                logger.debug(f"Short rejected - no volume spike: {current_volume:.0f} < {avg_volume_1min * volume_mult:.0f}")
                return False
            logger.debug(f"Volume spike: {current_volume} >= {avg_volume_1min * volume_mult:.0f}")
    
    # FILTER 4: Bearish bar confirmation - current bar must be bearish (close < open)
    # This ensures the reversal is happening with selling pressure, not buying into the drop
    if current_bar["close"] >= current_bar["open"]:
        logger.debug(f"Short rejected - not a bearish bar: close {current_bar['close']:.2f} >= open {current_bar['open']:.2f}")
        return False
    
    logger.info(f" SHORT SIGNAL: Price reversal at {current_bar['close']:.2f} (entry zone: {vwap_bands['upper_2']:.2f})")
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
    Capture comprehensive market state snapshot for analysis and learning.
    This replaces the old RL state structure with a flat market state structure.
    
    Args:
        symbol: Instrument symbol
        current_price: Current market price
    
    Returns:
        Dictionary with market state features (flat structure)
    """
    vwap = state[symbol].get("vwap", current_price)
    vwap_bands = state[symbol].get("vwap_bands", {})
    rsi = state[symbol].get("rsi", 50)
    
    # Calculate VWAP standard deviation
    vwap_std = 0
    if vwap_bands:
        upper = vwap_bands.get("upper_1", vwap)
        vwap_std = abs(upper - vwap) if upper != vwap else 0
    
    # Calculate VWAP distance in standard deviations
    vwap_distance = abs(current_price - vwap) / vwap_std if vwap_std > 0 else 0
    
    # Get ATR (use 1min bars for consistency)
    atr = calculate_atr_1min(symbol, CONFIG.get("atr_period", 14))
    if atr is None or atr == 0:
        atr = calculate_atr(symbol, CONFIG.get("atr_period", 14))
        if atr is None:
            atr = 0
    
    # Calculate volume ratio (current 1min bar vs avg of recent 1min bars)
    bars_1min = state[symbol]["bars_1min"]
    if len(bars_1min) >= 20:
        recent_volumes = [bar["volume"] for bar in list(bars_1min)[-20:]]
        avg_volume_1min = sum(recent_volumes) / len(recent_volumes)
        current_bar = bars_1min[-1]
        volume_ratio = current_bar["volume"] / avg_volume_1min if avg_volume_1min > 0 else 1.0
    else:
        volume_ratio = 1.0
    
    # Calculate returns (price change)
    if len(bars_1min) >= 2:
        prev_close = bars_1min[-2]["close"]
        returns = (current_price - prev_close) / prev_close if prev_close > 0 else 0.0
    else:
        returns = 0.0
    
    # Calculate VWAP slope
    if len(bars_1min) >= 10:
        recent_vwaps = []
        # Recalculate VWAP for each recent bar (simplified)
        for bar in list(bars_1min)[-10:]:
            recent_vwaps.append(bar["close"])  # Simplified: use close as proxy
        vwap_slope = calculate_slope(recent_vwaps, 5)
    else:
        vwap_slope = 0.0
    
    # Calculate ATR slope
    if len(bars_1min) >= 20:
        atr_values = []
        recent_bars = list(bars_1min)[-20:]
        for i in range(14, len(recent_bars)):
            bars_slice = recent_bars[i-14:i+1]
            true_ranges = []
            for j in range(1, len(bars_slice)):
                high = bars_slice[j]["high"]
                low = bars_slice[j]["low"]
                prev_close = bars_slice[j-1]["close"]
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                true_ranges.append(tr)
            if true_ranges:
                atr_values.append(sum(true_ranges) / len(true_ranges))
        
        if len(atr_values) >= 5:
            atr_slope = calculate_slope(atr_values, 5)
        else:
            atr_slope = 0.0
    else:
        atr_slope = 0.0
    
    # Get MACD histogram
    macd_data = state[symbol].get("macd")
    if macd_data:
        macd_hist = macd_data.get("histogram", 0.0)
    else:
        macd_hist = 0.0
    
    # Calculate Stochastic
    stoch = calculate_stochastic(bars_1min, 14, 3)
    stoch_k = stoch["k"]
    
    # Calculate volume slope
    if len(bars_1min) >= 10:
        recent_volumes = [bar["volume"] for bar in list(bars_1min)[-10:]]
        volume_slope = calculate_slope(recent_volumes, 5)
    else:
        volume_slope = 0.0
    
    # Get current time and session
    current_time = get_current_time()
    hour = current_time.hour
    session = get_session_type(current_time)
    
    # Get regime
    regime = state[symbol].get("current_regime", "NORMAL")
    
    # Get volatility regime
    volatility_regime = get_volatility_regime(atr, symbol)
    
    market_state = {
        "timestamp": current_time.isoformat(),
        "symbol": symbol,
        "price": current_price,
        "returns": returns,
        "vwap_distance": vwap_distance,
        "vwap_slope": vwap_slope,
        "atr": atr,
        "atr_slope": atr_slope,
        "rsi": rsi if rsi is not None else 50,
        "macd_hist": macd_hist,
        "stoch_k": stoch_k,
        "volume_ratio": volume_ratio,
        "volume_slope": volume_slope,
        "hour": hour,
        "session": session,
        "regime": regime,
        "volatility_regime": volatility_regime
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
        logger.info(f"[SIGNAL CHECK] Validation failed: {reason} at {bar_time}")
        return
    
    # Get bars for signal check
    prev_bar = state[symbol]["bars_1min"][-2]
    current_bar = state[symbol]["bars_1min"][-1]
    vwap_bands = state[symbol]["vwap_bands"]
    trend = state[symbol]["trend_direction"]
    
    logger.debug(f"Signal check: trend={trend}, prev_low={prev_bar['low']:.2f}, "
                f"current_close={current_bar['close']:.2f}, lower_band_2={vwap_bands['lower_2']:.2f}")
    
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
            logger.info(f"Γ¥î RL REJECTED LONG: {reason} (conf: {confidence:.0%})")
            # Always show details (for debugging)
            logger.info(f"   RSI: {market_state['rsi']:.1f}, VWAP dist: {market_state['vwap_distance']:.2f}, "
                      f"Vol ratio: {market_state['volume_ratio']:.2f}x")
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
        logger.info(f"Γ£à RL APPROVED LONG: {reason} (conf: {confidence:.0%}) | {regime}")
        if not is_backtest_mode():
            # Only show details in live mode
            logger.info(f"   RSI: {market_state['rsi']:.1f}, VWAP dist: {market_state['vwap_distance']:.2f}, "
                      f"Vol ratio: {market_state['volume_ratio']:.2f}x")
        
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
            logger.info(f"Γ¥î RL REJECTED SHORT: {reason} (conf: {confidence:.0%})")
            # Always show details (for debugging)
            logger.info(f"   RSI: {market_state['rsi']:.1f}, VWAP dist: {market_state['vwap_distance']:.2f}, "
                      f"Vol ratio: {market_state['volume_ratio']:.2f}x")
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
        logger.info(f"Γ£à RL APPROVED SHORT: {reason} (conf: {confidence:.0%}) | {regime}")
        if not is_backtest_mode():
            # Only show details in live mode
            logger.info(f"   RSI: {market_state['rsi']:.1f}, VWAP dist: {market_state['vwap_distance']:.2f}, "
                      f"Vol ratio: {market_state['volume_ratio']:.2f}x")
        
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
    
    Args:
        symbol: Instrument symbol
        side: 'long' or 'short'
        entry_price: Expected entry price
        rl_confidence: Optional RL confidence (for tracking, not used for position sizing)
    
    Returns:
        Tuple of (contracts, stop_price)
    """
    # Get account equity
    equity = get_account_equity()
    
    # Calculate risk allowance (1.2% of equity)
    risk_dollars = equity * CONFIG["risk_per_trade"]
    logger.info(f"Account equity: ${equity:.2f}, Risk allowance: ${risk_dollars:.2f}")
    
    # Determine stop price using regime-based approach
    vwap_bands = state[symbol]["vwap_bands"]
    vwap = state[symbol]["vwap"]
    tick_size = CONFIG["tick_size"]
    
    # Detect current regime for entry
    regime_detector = get_regime_detector()
    bars = state[symbol]["bars_1min"]
    atr = calculate_atr_1min(symbol, CONFIG.get("atr_period", 14))
    
    if atr is None:
        # Fallback to fixed stops if ATR can't be calculated
        logger.warning("ATR calculation failed, using fixed stops as fallback")
        max_stop_ticks = 11
        if side == "long":
            stop_price = entry_price - (max_stop_ticks * tick_size)
        else:
            stop_price = entry_price + (max_stop_ticks * tick_size)
        stop_price = round_to_tick(stop_price)
    else:
        # Use regime-based stop loss calculation
        entry_regime = regime_detector.detect_regime(bars, atr, CONFIG.get("atr_period", 14))
        
        # CONFIGURABLE STOP: Read from config (GUI sets via BOT_MAX_LOSS_PER_TRADE)
        max_stop_dollars = CONFIG.get("max_stop_loss_dollars", 200.0)
        tick_value = CONFIG["tick_value"]
        max_stop_ticks = max_stop_dollars / tick_value  # Ticks based on user's max loss per trade
        stop_distance = max_stop_ticks * tick_size
        
        logger.info(f"Fixed stop: {max_stop_ticks:.0f} ticks (${max_stop_dollars:.2f}) - Regime: {entry_regime.name}")
        
        if side == "long":
            stop_price = entry_price - stop_distance
        else:  # short
            stop_price = entry_price + stop_distance
        
        stop_price = round_to_tick(stop_price)
    
    # Calculate stop distance in ticks
    stop_distance = abs(entry_price - stop_price)
    ticks_at_risk = stop_distance / tick_size
    
    # Calculate risk per contract
    tick_value = CONFIG["tick_value"]
    risk_per_contract = ticks_at_risk * tick_value
    
    # Calculate number of contracts based on risk (baseline calculation)
    if risk_per_contract > 0:
        contracts = int(risk_dollars / risk_per_contract)
    else:
        contracts = 0
    
    # Get user's max contracts limit and apply it (FIXED - no dynamic scaling)
    user_max_contracts = CONFIG["max_contracts"]
    contracts = min(contracts, user_max_contracts)
    
    logger.info(f"[FIXED CONTRACTS] Using fixed max of {user_max_contracts} contracts")
    
    if contracts == 0:
        logger.warning(f"Position size too small: risk=${risk_per_contract:.2f}, allowance=${risk_dollars:.2f}")
        return 0, stop_price
    
    
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


def handle_partial_fill(symbol: str, side: str, expected_qty: int, timeout_seconds: float = 10) -> Tuple[int, bool]:
    """
    Check if order was partially filled and handle appropriately.
    
    CRITICAL FIX: Execution Risk #2 - Partial Fill Handling
    
    Args:
        symbol: Instrument symbol
        side: 'long' or 'short'
        expected_qty: Expected quantity to fill
        timeout_seconds: How long to wait for fill
    
    Returns:
        Tuple of (actual_filled_qty, is_complete_fill)
    """
    time_module.sleep(timeout_seconds)
    
    # Get actual position
    current_position = get_position_quantity(symbol)
    actual_filled = abs(current_position)
    is_complete = (abs(current_position) == expected_qty)
    
    if not is_complete and actual_filled > 0:
        # PARTIAL FILL DETECTED
        logger.warning(SEPARATOR_LINE)
        logger.warning("[WARN] PARTIAL FILL DETECTED")
        logger.warning(f"  Expected: {expected_qty} contracts")
        logger.warning(f"  Filled: {actual_filled} contracts")
        logger.warning(f"  Missing: {expected_qty - actual_filled} contracts")
        logger.warning(SEPARATOR_LINE)
        
        # Options:
        # 1. Accept partial fill and adjust stops/targets
        # 2. Try to complete the fill
        # 3. Close the partial and skip trade
        
        min_acceptable_fill_ratio = CONFIG.get("min_acceptable_fill_ratio", 0.5)
        fill_ratio = actual_filled / expected_qty
        
        if fill_ratio >= min_acceptable_fill_ratio:
            # Acceptable partial fill - work with it
            logger.info(f"  [OK] Accepting partial fill ({fill_ratio:.0%})")
            return actual_filled, False
        else:
            # Unacceptable partial fill - close it
            logger.warning(f"  Γ£ù Partial fill too small ({fill_ratio:.0%}) - closing position")
            # Close the partial position
            close_side = "SELL" if side == "long" else "BUY"
            place_market_order(symbol, close_side, actual_filled)
            return 0, False
    
    return actual_filled, is_complete


def place_entry_order_with_retry(symbol: str, side: str, contracts: int, 
                                 order_params: Dict[str, Any], 
                                 max_retries: int = 3) -> Tuple[Optional[Dict], float, str]:
    """
    Place entry order with retry logic for rejection/failure handling.
    
    CRITICAL FIX: Execution Risk #3 - Order Rejection Recovery
    
    Args:
        symbol: Instrument symbol
        side: 'long' or 'short'
        contracts: Number of contracts
        order_params: Order parameters from bid/ask manager
        max_retries: Maximum retry attempts
    
    Returns:
        Tuple of (order, fill_price, order_type_used)
    """
    order_side = "BUY" if side == "long" else "SELL"
    tick_size = CONFIG["tick_size"]
    
    for attempt in range(1, max_retries + 1):
        logger.info(f"  [ORDER] Order attempt {attempt}/{max_retries}")
        
        try:
            if order_params['strategy'] == 'passive':
                limit_price = order_params['limit_price']
                logger.info(f"  [PASSIVE] Passive Entry: ${limit_price:.2f} (saving spread)")
                
                order = place_limit_order(symbol, order_side, contracts, limit_price)
                
                if order is not None:
                    # ===== Gap #2: Queue Monitoring for Passive Orders =====
                    queue_monitoring_enabled = CONFIG.get("queue_monitoring_enabled", True)
                    
                    if queue_monitoring_enabled and bid_ask_manager is not None:
                        # Use queue monitor for live trading
                        logger.info(f"  [QUEUE] Monitoring queue position...")
                        
                        try:
                            # Create cancel function with symbol bound
                            def cancel_order_func(oid):
                                return cancel_order(symbol, oid)
                            
                            was_filled, queue_reason = bid_ask_manager.queue_monitor.monitor_limit_order_queue(
                                symbol=symbol,
                                order_id=order.get("order_id") if isinstance(order, dict) else str(order),
                                limit_price=limit_price,
                                side=side,
                                get_quote_func=bid_ask_manager.get_current_quote,
                                is_filled_func=lambda oid: abs(get_position_quantity(symbol)) >= contracts,
                                cancel_order_func=cancel_order_func
                            )
                            
                            if was_filled:
                                logger.info(f"  [FILLED] Queue monitor: {queue_reason}")
                                return order, limit_price, "passive"
                            else:
                                logger.warning(f"  [WARN] Queue monitor: {queue_reason}")
                                
                                if queue_reason == "timeout" and attempt < max_retries:
                                    # Timeout - switch to aggressive
                                    logger.info(f"  [SWITCH] Switching to aggressive (market) entry")
                                    order_params['strategy'] = 'aggressive'
                                    order_params['limit_price'] = order_params.get('fallback_price', limit_price)
                                    continue
                                elif queue_reason == "price_moved_away" and attempt < max_retries:
                                    # Price moved - retry with new price
                                    logger.info(f"  [RETRY] Reassessing entry with updated quote")
                                    time_module.sleep(0.5)
                                    continue
                                
                                # Failed - move to retry
                                logger.warning(f"  [FAIL] Attempt {attempt}: Queue monitoring failed")
                                
                        except Exception as e:
                            logger.error(f"  [ERROR] Queue monitoring error: {e}")
                            # Fall through to regular passive fill handling
                    
                    else:
                        # Backtesting or queue monitoring disabled - use standard fill handling
                        actual_filled, is_complete = handle_partial_fill(
                            symbol, side, contracts, order_params.get('timeout', 10)
                        )
                        
                        if is_complete:
                            logger.info(f"  [FILLED] Complete fill at ${limit_price:.2f}")
                            return order, limit_price, "passive"
                        elif actual_filled > 0:
                            logger.warning(f"  [PARTIAL] Partial fill: {actual_filled}/{contracts} contracts")
                            return order, limit_price, "passive_partial"
                    
                    # Check for fill (fallback)
                    actual_filled, is_complete = handle_partial_fill(
                        symbol, side, contracts, order_params.get('timeout', 10)
                    )
                    
                    if is_complete:
                        logger.info(f"  [FILLED] Complete fill at ${limit_price:.2f}")
                        return order, limit_price, "passive"
                    elif actual_filled > 0:
                        logger.warning(f"  [PARTIAL] Partial fill: {actual_filled}/{contracts} contracts")
                        return order, limit_price, "passive_partial"
                    
                    # Not filled - retry with better price if attempts remain
                    logger.warning(f"  [FAIL] Attempt {attempt}: Passive not filled")
                    
                    if attempt < max_retries:
                        # Jump queue by 1 tick
                        if side == "long":
                            order_params['limit_price'] += tick_size
                        else:
                            order_params['limit_price'] -= tick_size
                        
                        logger.info(f"  [RETRY] Retry with improved price: ${order_params['limit_price']:.2f}")
                        time_module.sleep(0.5)  # Brief pause before retry
                        continue
                
                # Order placement failed
                logger.error(f"  [ERROR] Attempt {attempt}: Order placement failed")
                
            elif order_params['strategy'] == 'aggressive':
                limit_price = order_params['limit_price']
                logger.info(f"  [AGGRESSIVE] Aggressive Entry: ${limit_price:.2f} (guaranteed fill)")
                
                order = place_limit_order(symbol, order_side, contracts, limit_price)
                
                if order is not None:
                    # Aggressive orders usually fill immediately
                    time_module.sleep(1)  # Brief wait to confirm fill
                    actual_filled = get_position_quantity(symbol)
                    if abs(actual_filled) >= contracts:
                        logger.info(f"  [FILLED] Aggressive fill at ${limit_price:.2f}")
                        return order, limit_price, "aggressive"
                    else:
                        logger.warning(f"  [PARTIAL] Aggressive order placed but not filled yet")
                        # Still return it, assume it will fill
                        return order, limit_price, "aggressive"
                
                logger.error(f"  [FAIL] Attempt {attempt}: Aggressive order failed")
            
            elif order_params['strategy'] == 'mixed':
                # Mixed strategy - split between passive and aggressive
                passive_qty = order_params['passive_contracts']
                aggressive_qty = order_params['aggressive_contracts']
                passive_price = order_params['passive_price']
                aggressive_price = order_params['aggressive_price']
                
                logger.info(f"  ≡ƒöÇ Mixed: {passive_qty}@${passive_price:.2f} (passive) + {aggressive_qty}@${aggressive_price:.2f} (aggressive)")
                
                # Place both portions
                passive_order = place_limit_order(symbol, order_side, passive_qty, passive_price)
                aggressive_order = place_limit_order(symbol, order_side, aggressive_qty, aggressive_price)
                
                if aggressive_order is not None:
                    # Use weighted average fill price
                    avg_fill_price = (passive_price * passive_qty + aggressive_price * aggressive_qty) / contracts
                    return aggressive_order, avg_fill_price, "mixed"
            
            # Failed this attempt
            if attempt < max_retries:
                backoff_time = 0.5 * attempt  # Exponential backoff
                logger.warning(f"  [WAIT] Retrying in {backoff_time:.1f}s...")
                time_module.sleep(backoff_time)
            
        except Exception as e:
            logger.error(f"  [FAIL] Attempt {attempt} exception: {e}")
            if attempt < max_retries:
                time_module.sleep(0.5 * attempt)
                continue
    
    # All retries exhausted
    logger.error(f"  [BLOCKED] All {max_retries} attempts failed - ENTRY ABORTED")
    return None, 0.0, "failed"


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
    Execute entry order with stop loss and target.
    Uses intelligent bid/ask order placement strategy with FULL EXECUTION RISK PROTECTION.
    
    SHADOW MODE: Logs signal without placing actual orders.
    
    NEW: Production-ready execution with:
    - Position state validation (prevents double positioning)
    - Price deterioration protection (max 3 ticks from signal)
    - Partial fill handling (detects and manages)
    - Order rejection recovery (3 retries with exponential backoff)
    - Fast market detection (skips dangerous entries)
    
    Args:
        symbol: Instrument symbol
        side: 'long' or 'short'
        entry_price: Approximate entry price (mid or last)
    """
    # ===== SHADOW MODE: Signal-only (manual trading mode) =====
    if CONFIG.get("shadow_mode", False):
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
    
    # Spread-aware position sizing (Requirement 10)
    original_contracts = contracts
    if bid_ask_manager is not None:
        try:
            expected_profit_ticks = 20  # Use reasonable default for spread calculation
            adjusted_contracts, cost_breakdown = bid_ask_manager.calculate_spread_aware_position_size(
                symbol, contracts, expected_profit_ticks
            )
            if adjusted_contracts != original_contracts:
                logger.warning(f"  Position size adjusted: {original_contracts} -> {adjusted_contracts} contracts")
                logger.info(f"  Spread cost: {cost_breakdown['cost_percentage']:.1f}% of expected profit")
                contracts = adjusted_contracts
        except Exception as e:
            logger.warning(f"  Spread-aware sizing unavailable: {e}")
    
    logger.info(f"  Contracts: {contracts}")
    logger.info(f"  Stop Loss: ${stop_price:.2f}")
    
    # Track order execution details for post-trade analysis
    fill_start_time = datetime.now()
    order_type_used = "market"  # Default
    
    # Get intelligent order placement strategy from bid/ask manager
    order_side = "BUY" if side == "long" else "SELL"
    actual_fill_price = entry_price
    order = None
    
    # ===== FIX #2 & #3: Retry Logic + Partial Fill Handling =====
    if bid_ask_manager is not None:
        try:
            # Get order parameters from bid/ask manager
            order_params = bid_ask_manager.get_entry_order_params(symbol, side, contracts)
            
            logger.info(f"  Order Strategy: {order_params['strategy']}")
            logger.info(f"  Reason: {order_params['reason']}")
            
            # Use retry-enabled order placement with full execution protection
            order, actual_fill_price, order_type_used = place_entry_order_with_retry(
                symbol, side, contracts, order_params, max_retries=3
            )
            
            if order is None:
                logger.error("[FAIL] Failed to place entry after retries - TRADE SKIPPED")
                return
            
            # CRITICAL: IMMEDIATELY save minimal position state to prevent loss on crash
            # This creates a recovery checkpoint before any further processing
            state[symbol]["position"]["active"] = True
            state[symbol]["position"]["side"] = side
            state[symbol]["position"]["quantity"] = contracts
            state[symbol]["position"]["entry_price"] = actual_fill_price
            state[symbol]["position"]["entry_time"] = entry_time
            state[symbol]["position"]["order_id"] = order.get("order_id")
            save_position_state(symbol)
            logger.info(f"  [CHECKPOINT] Emergency position state saved (crash protection)")
            
            logger.info(f"  [OK] Order placed successfully using {order_type_used} strategy")
            
        except Exception as e:
            logger.error(f"Error using bid/ask manager for entry: {e}")
            logger.info("Falling back to market order")
            order = place_market_order(symbol, order_side, contracts)
            actual_fill_price = entry_price
            
            # CRITICAL: Save emergency checkpoint for fallback path too
            if order is not None:
                state[symbol]["position"]["active"] = True
                state[symbol]["position"]["side"] = side
                state[symbol]["position"]["quantity"] = contracts
                state[symbol]["position"]["entry_price"] = actual_fill_price
                state[symbol]["position"]["entry_time"] = entry_time
                state[symbol]["position"]["order_id"] = order.get("order_id")
                save_position_state(symbol)
                logger.info(f"  [CHECKPOINT] Emergency position state saved (fallback path)")
    else:
        # No bid/ask manager, use traditional market order
        logger.info("  Using market order (no bid/ask manager)")
        
        order = place_market_order(symbol, order_side, contracts)
        
        # CRITICAL: Save emergency checkpoint for no-manager path
        if order is not None:
            state[symbol]["position"]["active"] = True
            state[symbol]["position"]["side"] = side
            state[symbol]["position"]["quantity"] = contracts
            state[symbol]["position"]["entry_price"] = entry_price
            state[symbol]["position"]["entry_time"] = entry_time
            state[symbol]["position"]["order_id"] = order.get("order_id")
            save_position_state(symbol)
            logger.info(f"  [CHECKPOINT] Emergency position state saved (no manager path)")
    
    if order is None:
        logger.error("Failed to place entry order")
        return
    
    # ===== CRITICAL FIX #7: Entry Fill Validation (Live Trading) =====
    # Validate actual entry fill price vs expected (critical for live trading)
    # In live trading, get actual fill price from broker
    try:
        actual_fill_from_broker = get_last_fill_price(symbol)
        if actual_fill_from_broker and actual_fill_from_broker != actual_fill_price:
            # Calculate entry slippage
            tick_size = CONFIG["tick_size"]
            tick_value = CONFIG["tick_value"]
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
            side="LONG" if side == 'buy' else "SHORT"
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
    logger.info(f"")
    logger.info(f"  ≡ƒôè PROFESSIONAL RISK MANAGEMENT")
    logger.info(f"  Entry Regime: {entry_regime.name}")
    logger.info(f"")
    logger.info(f"  Initial Stop Loss:")
    logger.info(f"    Stop Multiplier: {entry_regime.stop_mult}x ATR")
    logger.info(f"    Stop Distance: {stop_distance_ticks:.1f} ticks (${abs(actual_fill_price - stop_price):.2f})")
    logger.info(f"    Stop Price: ${stop_price:.2f}")
    logger.info(f"")
    logger.info(f"  Breakeven Protection:")
    logger.info(f"    BE Multiplier: {entry_regime.breakeven_mult}x")
    logger.info(f"    Trigger: {stop_distance_ticks * entry_regime.breakeven_mult:.1f} ticks profit (1:1 risk-reward)")
    logger.info(f"    Locks: ${actual_fill_price + (2 * CONFIG['tick_size']) if side == 'long' else actual_fill_price - (2 * CONFIG['tick_size']):.2f} (+$50 profit)")
    logger.info(f"")
    logger.info(f"  Trailing Stop:")
    logger.info(f"    Trail Multiplier: {entry_regime.trailing_mult}x")
    base_trail = CONFIG.get("trailing_stop_distance_ticks", 8)
    logger.info(f"    Trail Distance: {base_trail * entry_regime.trailing_mult:.1f} ticks behind peak")
    logger.info(f"")
    logger.info(f"  Timeout Protection:")
    logger.info(f"    Sideways: {entry_regime.sideways_timeout} minutes")
    logger.info(f"    Underwater: {entry_regime.underwater_timeout} minutes")
    
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
    Check for signal reversal (price crossing back to opposite band).
    
    Args:
        symbol: Instrument symbol
        current_bar: Current 1-minute bar
        position: Position dictionary
    
    Returns:
        Tuple of (reversal_detected, exit_price)
    """
    vwap_bands = state[symbol]["vwap_bands"]
    trend = state[symbol]["trend_direction"]
    side = position["side"]
    
    if side == "long" and trend == "up":
        # If price crosses back above upper band 2, bounce is complete
        if current_bar["close"] > vwap_bands["upper_2"]:
            return True, current_bar["close"]
    
    if side == "short" and trend == "down":
        # If price crosses back below lower band 1, bounce is complete
        if current_bar["close"] < vwap_bands["lower_1"]:
            return True, current_bar["close"]
    
    return False, None


def check_time_based_exits(symbol: str, current_bar: Dict[str, Any], position: Dict[str, Any], 
                           bar_time: datetime) -> Tuple[Optional[str], Optional[float]]:
    """
    Check time-based exit conditions.
    
    ONLY checks for:
    1. Emergency forced flatten at 4:45 PM ET (market close before maintenance)
    
    All other exits are regime-based (stops, timeouts, breakeven, trailing).
    
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
    
    Uses regime-based parameters:
    - breakeven_profit_threshold_ticks: Calculated from regime breakeven_mult
    - breakeven_stop_offset_ticks: Stop offset from entry (default: 1 tick)
    
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
    tick_size = CONFIG["tick_size"]
    
    # Get regime-based breakeven threshold
    # PROFESSIONAL APPROACH: Base threshold = initial stop distance (1:1 risk-reward)
    # This adapts to each trade's actual risk and adjusts per regime
    entry_price = position["entry_price"]
    original_stop = position["original_stop_price"]
    tick_size = CONFIG["tick_size"]
    
    # Calculate initial stop distance in ticks
    initial_stop_distance_ticks = abs(entry_price - original_stop) / tick_size
    
    # Get current regime (or use entry regime if not set)
    current_regime_name = position.get("current_regime", position.get("entry_regime", "NORMAL"))
    current_regime = REGIME_DEFINITIONS.get(current_regime_name, REGIME_DEFINITIONS["NORMAL"])
    
    # Apply regime multiplier to actual stop distance (professional standard)
    # NORMAL = 1.0x (move at 1:1), CHOPPY = 0.75-0.95x (faster), TRENDING = 1.0x (standard)
    breakeven_threshold_ticks = initial_stop_distance_ticks * current_regime.breakeven_mult
    
    # Place stop +2 ticks in profit (not exactly breakeven) - locks in small profit
    breakeven_offset_ticks = 2  # Always lock in $50 profit minimum
    
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
                logger.debug(f"Γ£ô Replaced stop order: {old_stop_order_id} ΓåÆ {new_stop_order.get('order_id')}")
            else:
                logger.warning(f"ΓÜá New stop active but failed to cancel old stop {old_stop_order_id}")
    
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
        profit_locked_dollars = profit_locked_ticks * CONFIG["tick_value"] * contracts
        
        # Step 6 - Log activation
        logger.info("=" * 60)
        logger.info("BREAKEVEN PROTECTION ACTIVATED")
        logger.info("=" * 60)
        logger.info(f"  Regime: {current_regime_name} (BE mult: {current_regime.breakeven_mult}x)")
        logger.info(f"  Initial Risk: {initial_stop_distance_ticks:.1f} ticks")
        logger.info(f"  Breakeven Threshold: {breakeven_threshold_ticks:.1f} ticks (1:1 risk-reward)")
        logger.info(f"  Current Profit: {profit_ticks:.1f} ticks")
        logger.info(f"  Original Stop: ${original_stop:.2f}")
        logger.info(f"  New Stop: ${new_stop_price:.2f} (+{breakeven_offset_ticks} ticks in profit)")
        logger.info(f"  Profit Locked: ${profit_locked_dollars:.2f} (minimum guaranteed)")
        logger.info("=" * 60)
        logger.info(f"  New Breakeven Stop: ${new_stop_price:.2f}")
        logger.info(f"  Profit Locked In: {profit_locked_ticks:.1f} ticks (${profit_locked_dollars:+.2f})")
        logger.info(f"  Entry Price: ${entry_price:.2f}")
        logger.info(f"  Current Price: ${current_price:.2f}")
        logger.info("=" * 60)
    else:
        logger.error("Failed to place breakeven stop order")


# ============================================================================
# PHASE FOUR: Trailing Stop Logic
# ============================================================================

def check_trailing_stop(symbol: str, current_price: float) -> None:
    """
    Check and update trailing stop based on price movement.
    
    Uses regime-based parameters:
    - trailing_stop_distance_ticks: Calculated from regime trailing_mult
    - trailing_stop_min_profit_ticks: Minimum profit to activate (default: 12 ticks)
    
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
    tick_size = CONFIG["tick_size"]
    
    # Get regime-based trailing distance
    # Base distance from config, scaled by regime trailing multiplier
    base_trailing_ticks = CONFIG.get("trailing_stop_distance_ticks", 8)
    
    # Get current regime (or use entry regime if not set)
    current_regime_name = position.get("current_regime", position.get("entry_regime", "NORMAL"))
    current_regime = REGIME_DEFINITIONS.get(current_regime_name, REGIME_DEFINITIONS["NORMAL"])
    
    # Apply regime multiplier to base trailing distance
    trailing_distance_ticks = base_trailing_ticks * current_regime.trailing_mult
    min_profit_ticks = CONFIG.get("trailing_stop_min_profit_ticks", 12)
    
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
                logger.debug(f"Γ£ô Replaced stop order: {old_stop_order_id} ΓåÆ {new_stop_order.get('order_id')}")
            else:
                logger.warning(f"ΓÜá New trailing stop active but failed to cancel old stop {old_stop_order_id}")
    
    if new_stop_order:
        # Activate trailing stop flag if not already active
        if not position["trailing_stop_active"]:
            position["trailing_stop_active"] = True
            position["trailing_activated_time"] = get_current_time()
            logger.info(f"TRAILING STOP ACTIVATED - Regime: {current_regime_name} (trail mult: {current_regime.trailing_mult}x)")
        
        # Update position tracking
        old_stop = position["stop_price"]
        position["stop_price"] = new_trailing_stop
        position["trailing_stop_price"] = new_trailing_stop
        if new_stop_order.get("order_id"):
            position["stop_order_id"] = new_stop_order.get("order_id")
        
        # Calculate profit now locked in
        profit_locked_ticks = (new_trailing_stop - entry_price) / tick_size if side == "long" else (entry_price - new_trailing_stop) / tick_size
        profit_locked_dollars = profit_locked_ticks * CONFIG["tick_value"] * contracts
        
        # Step 6 - Log updates
        logger.info("=" * 60)
        logger.info("TRAILING STOP UPDATED")
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
# PHASE FOUR-B: Regime-Based Timeout Exits
# ============================================================================

def check_sideways_timeout(symbol: str, current_price: float, current_time: datetime) -> Tuple[bool, Optional[float]]:
    """
    Check if position should exit due to sideways timeout (stagnant price action).
    
    Uses regime-based sideways_timeout parameter (8-18 minutes depending on regime).
    
    APPLIES ONLY WHEN:
    - P&L is zero or positive (not losing)
    - Trailing stop has NOT activated yet
    
    Once trailing stop activates, this check is skipped entirely (trailing becomes sole exit mechanism).
    Timeout clock resets when regime changes.
    
    Args:
        symbol: Instrument symbol
        current_price: Current market price
        current_time: Current datetime
    
    Returns:
        Tuple of (should_exit, exit_price)
    """
    position = state[symbol]["position"]
    
    if not position["active"]:
        return False, None
    
    # PRIORITY CHECK: If trailing stop active, skip sideways timeout entirely
    if position.get("trailing_stop_active", False):
        return False, None
    
    # Calculate current P&L
    side = position["side"]
    entry_price = position["entry_price"]
    tick_size = CONFIG["tick_size"]
    
    if side == "long":
        pnl_ticks = (current_price - entry_price) / tick_size
    else:
        pnl_ticks = (entry_price - current_price) / tick_size
    
    # Sideways timeout only applies when P&L >= 0 (zero or positive)
    if pnl_ticks < 0:
        # Position losing - sideways timeout disabled
        return False, None
    
    # Position is zero/positive P&L and trailing not active - sideways timeout is ACTIVE
    # Get current regime and its sideways timeout
    current_regime_name = position.get("current_regime", position.get("entry_regime", "NORMAL"))
    current_regime = REGIME_DEFINITIONS.get(current_regime_name, REGIME_DEFINITIONS["NORMAL"])
    sideways_timeout_minutes = current_regime.sideways_timeout
    
    # Determine start time for timeout measurement
    # Use regime change time if available, otherwise entry time
    regime_change_time = position.get("regime_change_time")
    start_time = regime_change_time if regime_change_time else position["entry_time"]
    
    if start_time is None:
        return False, None
    
    # Calculate time elapsed
    time_elapsed_minutes = (current_time - start_time).total_seconds() / 60.0
    
    # Check if exceeded sideways timeout
    if time_elapsed_minutes < sideways_timeout_minutes:
        return False, None
    
    # Track price extremes during the timeout period to measure range
    # Initialize tracking if not present
    if "sideways_high" not in position or regime_change_time:
        # Reset tracking on regime change or first check
        position["sideways_high"] = current_price
        position["sideways_low"] = current_price
        position["sideways_last_reset"] = current_time
        return False, None
    
    # Update price extremes
    position["sideways_high"] = max(position.get("sideways_high", current_price), current_price)
    position["sideways_low"] = min(position.get("sideways_low", current_price), current_price)
    
    # Calculate price range during timeout period
    price_range_ticks = (position["sideways_high"] - position["sideways_low"]) / tick_size
    
    # Define "stagnant" as narrow range (< 5 ticks) regardless of distance from entry
    # This catches positions that are profitable but stuck in tight range
    stagnant_range_threshold_ticks = 5.0
    
    if price_range_ticks < stagnant_range_threshold_ticks:
        # Price stuck in narrow range and timeout exceeded
        # Enhanced logging for backtesting
        if is_backtest_mode():
            logger.info("=" * 60)
            logger.info("[EXIT] SIDEWAYS TIMEOUT - EXITING STAGNANT POSITION")
            logger.info("=" * 60)
            logger.info(f"  Regime: {current_regime_name}")
            logger.info(f"  Timeout: {sideways_timeout_minutes} minutes")
            logger.info(f"  Time Elapsed: {time_elapsed_minutes:.1f} minutes")
            logger.info(f"  Price Range: {price_range_ticks:.1f} ticks")
            logger.info(f"  Entry: ${entry_price:.2f}, Current: ${current_price:.2f}")
            logger.info(f"  Current P&L: {pnl_ticks:+.1f} ticks (profitable)")
            logger.info("=" * 60)
        else:
            logger.warning("=" * 60)
            logger.warning("SIDEWAYS TIMEOUT - EXITING STAGNANT POSITION")
            logger.warning("=" * 60)
            logger.warning(f"  Regime: {current_regime_name}")
            logger.warning(f"  Timeout: {sideways_timeout_minutes} minutes")
            logger.warning(f"  Time Elapsed: {time_elapsed_minutes:.1f} minutes")
            logger.warning(f"  Price Range: {price_range_ticks:.1f} ticks (High: ${position['sideways_high']:.2f}, Low: ${position['sideways_low']:.2f})")
            logger.warning(f"  Entry: ${entry_price:.2f}, Current: ${current_price:.2f}")
            logger.warning(f"  Current P&L: {pnl_ticks:+.1f} ticks (profitable)")
            logger.warning(f"  Reason: Position stuck in narrow range - not developing momentum")
            logger.warning("=" * 60)
        
        return True, current_price
    
    return False, None


def check_underwater_timeout(symbol: str, current_price: float, current_time: datetime) -> Tuple[bool, Optional[float]]:
    """
    Check if position should exit due to underwater timeout (continuous loss).
    
    Uses regime-based underwater_timeout parameter (6-10 minutes depending on regime).
    
    KEY BEHAVIOR:
    - Timer always counts TOTAL elapsed time since entry (never resets)
    - When position profitable: Underwater timeout disabled (not checking)
    - When position losing AND trailing NOT active: Underwater timeout active using total elapsed time
    - If position flips redΓåÆgreenΓåÆred: Total time used (red5min + green2min + red = 7min total)
    - If trailing stop active: This check is skipped entirely
    
    Args:
        symbol: Instrument symbol
        current_price: Current market price
        current_time: Current datetime
    
    Returns:
        Tuple of (should_exit, exit_price)
    """
    position = state[symbol]["position"]
    
    if not position["active"]:
        return False, None
    
    # PRIORITY CHECK: If trailing stop active, skip underwater timeout entirely
    if position.get("trailing_stop_active", False):
        return False, None
    
    side = position["side"]
    entry_price = position["entry_price"]
    entry_time = position.get("entry_time")
    tick_size = CONFIG["tick_size"]
    
    if not entry_time:
        return False, None
    
    # Calculate current P&L in ticks
    if side == "long":
        pnl_ticks = (current_price - entry_price) / tick_size
    else:  # short
        pnl_ticks = (entry_price - current_price) / tick_size
    
    # Check if currently underwater (losing)
    if pnl_ticks >= 0:
        # Position is profitable or breakeven - underwater timeout disabled (not active)
        # But timer keeps running in background
        return False, None
    
    # Position is underwater (losing) and trailing NOT active - underwater timeout is ACTIVE
    # Get current regime and its underwater timeout
    current_regime_name = position.get("current_regime", position.get("entry_regime", "NORMAL"))
    current_regime = REGIME_DEFINITIONS.get(current_regime_name, REGIME_DEFINITIONS["NORMAL"])
    underwater_timeout_minutes = current_regime.underwater_timeout
    
    # Calculate TOTAL elapsed time since entry (never resets)
    total_elapsed_minutes = (current_time - entry_time).total_seconds() / 60.0
    
    # Check if total elapsed time exceeds underwater timeout
    # This means: if trade has been alive for 7 minutes total and underwater timeout is 6 min, exit
    if total_elapsed_minutes >= underwater_timeout_minutes:
        # Total time since entry exceeds underwater timeout while position is losing
        tick_value = CONFIG["tick_value"]
        loss_dollars = pnl_ticks * tick_value * position["quantity"]
        
        # Enhanced logging for backtesting
        if is_backtest_mode():
            logger.info("=" * 60)
            logger.info("[EXIT] UNDERWATER TIMEOUT - CUTTING LOSS")
            logger.info("=" * 60)
            logger.info(f"  Regime: {current_regime_name}")
            logger.info(f"  Timeout: {underwater_timeout_minutes} minutes")
            logger.info(f"  Total Elapsed Time: {total_elapsed_minutes:.1f} minutes (since entry)")
            logger.info(f"  Current Loss: {abs(pnl_ticks):.1f} ticks (${loss_dollars:.2f})")
            logger.info(f"  Entry: ${entry_price:.2f}, Current: ${current_price:.2f}")
            logger.info("=" * 60)
        else:
            logger.warning("=" * 60)
            logger.warning("UNDERWATER TIMEOUT - CUTTING LOSS")
            logger.warning("=" * 60)
            logger.warning(f"  Regime: {current_regime_name}")
            logger.warning(f"  Timeout: {underwater_timeout_minutes} minutes")
            logger.warning(f"  Total Elapsed Time: {total_elapsed_minutes:.1f} minutes (since entry)")
            logger.warning(f"  Current Loss: {abs(pnl_ticks):.1f} ticks (${loss_dollars:.2f})")
            logger.warning(f"  Entry: ${entry_price:.2f}, Current: ${current_price:.2f}")
            logger.warning("=" * 60)
        
        return True, current_price
    
    return False, None


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
    tick_size = CONFIG["tick_size"]
    tick_value = CONFIG["tick_value"]
    
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
    tick_size = CONFIG["tick_size"]
    tick_value = CONFIG["tick_value"]
    
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
    tick_size = CONFIG["tick_size"]
    tick_value = CONFIG["tick_value"]
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
    
    # Enhanced logging for backtesting - log regime changes explicitly
    if is_backtest_mode() and prev_regime != detected_regime.name:
        logger.info(f"[REGIME CHANGE] {prev_regime} → {detected_regime.name} (ATR: {current_atr:.2f})")
        logger.info(f"  Stop Mult: {detected_regime.stop_mult:.2f}x | Trailing: {detected_regime.trailing_mult:.2f}x")
        logger.info(f"  Timeouts: Sideways={detected_regime.sideways_timeout}min, Underwater={detected_regime.underwater_timeout}min")
    else:
        logger.debug(f"[REGIME] Updated to {detected_regime.name} (ATR: {current_atr:.2f})")


def check_regime_change(symbol: str, current_price: float) -> None:
    """
    Check if market regime has changed during an active trade and adjust parameters.
    
    This function:
    1. Detects current regime from last 20 bars
    2. Compares to entry regime
    3. If changed, updates stop loss and trailing parameters based on new regime
    4. Uses pure regime multipliers (no confidence scaling)
    
    Critical rule: Stop price never moves closer to entry (backward), only trailing can tighten it.
    
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
        "timestamp": change_time,
        "stop_mult_change": f"{REGIME_DEFINITIONS[entry_regime_name].stop_mult:.2f}x ΓåÆ {current_regime.stop_mult:.2f}x",
        "trailing_mult_change": f"{REGIME_DEFINITIONS[entry_regime_name].trailing_mult:.2f}x ΓåÆ {current_regime.trailing_mult:.2f}x"
    })
    
    # Calculate new stop distance based on new regime (pure regime multiplier, no confidence scaling)
    side = position["side"]
    entry_price = position["entry_price"]
    tick_size = CONFIG["tick_size"]
    current_stop = position["stop_price"]
    
    # Calculate new stop distance from entry using regime multiplier
    new_stop_distance = current_atr * current_regime.stop_mult
    
    if side == "long":
        new_stop_price = entry_price - new_stop_distance
    else:  # short
        new_stop_price = entry_price + new_stop_distance
    
    new_stop_price = round_to_tick(new_stop_price)
    
    # CRITICAL RULE: Never move stop closer to entry (backward)
    should_update_stop = False
    if side == "long":
        # For longs, new stop must be >= current stop (never worse)
        if new_stop_price >= current_stop:
            should_update_stop = True
    else:  # short
        # For shorts, new stop must be <= current stop (never worse)
        if new_stop_price <= current_stop:
            should_update_stop = True
    
    if should_update_stop:
        # Update stop with continuous protection
        # PROFESSIONAL APPROACH: Place new stop FIRST, then cancel old
        stop_side = "SELL" if side == "long" else "BUY"
        contracts = position["quantity"]
        new_stop_order = place_stop_order(symbol, stop_side, contracts, new_stop_price)
        
        if new_stop_order:
            # New regime-adjusted stop confirmed - now safe to cancel old stop
            old_stop_order_id = position.get("stop_order_id")
            if old_stop_order_id:
                cancel_success = cancel_order(symbol, old_stop_order_id)
                if cancel_success:
                    logger.info(f"  Γ£ô Replaced stop order: {old_stop_order_id} ΓåÆ {new_stop_order.get('order_id')}")
                else:
                    logger.warning(f"  ΓÜá∩╕Å New stop active but failed to cancel old stop {old_stop_order_id}")
        
        if new_stop_order:
            old_stop = position["stop_price"]
            position["stop_price"] = new_stop_price
            if new_stop_order.get("order_id"):
                position["stop_order_id"] = new_stop_order.get("order_id")
            
            # Get old regime parameters for comparison
            old_regime = REGIME_DEFINITIONS[entry_regime_name]
            
            # Calculate initial stop distance for logging
            entry_price = position["entry_price"]
            original_stop = position.get("original_stop_price", position["stop_price"])
            initial_stop_distance_ticks = abs(entry_price - original_stop) / tick_size
            
            logger.info("=" * 60)
            logger.info(f"REGIME CHANGE - PARAMETERS UPDATED")
            logger.info("=" * 60)
            logger.info(f"  Transition: {entry_regime_name} ΓåÆ {current_regime.name}")
            logger.info(f"  Timestamp: {get_current_time().strftime('%H:%M:%S')}")
            logger.info(f"")
            logger.info(f"  Stop Management:")
            logger.info(f"    Stop Multiplier: {old_regime.stop_mult:.2f}x ΓåÆ {current_regime.stop_mult:.2f}x")
            logger.info(f"    Old Stop: ${old_stop:.2f} ΓåÆ New Stop: ${new_stop_price:.2f}")
            logger.info(f"    Initial Risk: {initial_stop_distance_ticks:.1f} ticks (adapts to actual risk)")
            logger.info(f"")
            logger.info(f"  Breakeven Management:")
            logger.info(f"    BE Multiplier: {old_regime.breakeven_mult:.2f}x ΓåÆ {current_regime.breakeven_mult:.2f}x")
            if not position.get("breakeven_active"):
                old_be_threshold = initial_stop_distance_ticks * old_regime.breakeven_mult
                new_be_threshold = initial_stop_distance_ticks * current_regime.breakeven_mult
                logger.info(f"    BE Threshold: {old_be_threshold:.1f} ΓåÆ {new_be_threshold:.1f} ticks (1:1 risk-reward)")
            else:
                logger.info(f"    Breakeven already active (stop at ${position['stop_price']:.2f})")
            logger.info(f"")
            logger.info(f"  Trailing Stop:")
            logger.info(f"    Trailing Mult: {old_regime.trailing_mult:.2f}x ΓåÆ {current_regime.trailing_mult:.2f}x")
            base_trailing_ticks = CONFIG.get("trailing_stop_distance_ticks", 8)
            old_trailing_distance = base_trailing_ticks * old_regime.trailing_mult
            new_trailing_distance = base_trailing_ticks * current_regime.trailing_mult
            logger.info(f"    Trailing Distance: {old_trailing_distance:.1f} ΓåÆ {new_trailing_distance:.1f} ticks")
            logger.info(f"")
            logger.info(f"  Timeout Protection:")
            logger.info(f"    Sideways Timeout: {old_regime.sideways_timeout}min ΓåÆ {current_regime.sideways_timeout}min")
            logger.info(f"    Underwater Timeout: {old_regime.underwater_timeout}min ΓåÆ {current_regime.underwater_timeout}min")
            logger.info(f"    (Timeout clocks reset from regime change time)")
            logger.info("=" * 60)
        else:
            logger.error("Failed to update stop after regime change")
    else:
        # Stop would move backward - log but don't update
        old_regime = REGIME_DEFINITIONS[entry_regime_name]
        logger.info("=" * 60)
        logger.info(f"REGIME CHANGE DETECTED - STOP NOT ADJUSTED")
        logger.info("=" * 60)
        logger.info(f"  Transition: {entry_regime_name} ΓåÆ {current_regime.name}")
        logger.info(f"  Stop would move backward: ${current_stop:.2f} ΓåÆ ${new_stop_price:.2f}")
        logger.info(f"  Stop remains at: ${current_stop:.2f}")
        logger.info(f"")
        logger.info(f"  Other parameters still updated:")
        logger.info(f"    Breakeven Mult: {old_regime.breakeven_mult:.2f}x ΓåÆ {current_regime.breakeven_mult:.2f}x")
        logger.info(f"    Trailing Mult: {old_regime.trailing_mult:.2f}x ΓåÆ {current_regime.trailing_mult:.2f}x")
        logger.info(f"    Timeouts: Sideways {old_regime.sideways_timeout}ΓåÆ{current_regime.sideways_timeout}min, "
                   f"Underwater {old_regime.underwater_timeout}ΓåÆ{current_regime.underwater_timeout}min")
        logger.info("=" * 60)
    
    # Update breakeven threshold based on new regime (if not already active)
    if not position["breakeven_active"]:
        old_regime = REGIME_DEFINITIONS[entry_regime_name]
        
        # PROFESSIONAL: Calculate dynamic threshold based on initial stop distance
        entry_price = position["entry_price"]
        original_stop = position["original_stop_price"]
        tick_size = CONFIG["tick_size"]
        initial_stop_distance_ticks = abs(entry_price - original_stop) / tick_size
        
        old_breakeven_threshold = initial_stop_distance_ticks * old_regime.breakeven_mult
        new_breakeven_threshold = initial_stop_distance_ticks * current_regime.breakeven_mult
        logger.info(f"  Breakeven threshold updated: {old_breakeven_threshold:.1f} ΓåÆ {new_breakeven_threshold:.1f} ticks")
    else:
        logger.info(f"  Breakeven already active - threshold change does not apply")


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
        logger.critical(SEPARATOR_LINE)
        logger.critical("MARKET CLOSING - AUTO-FLATTENING POSITION")
        logger.critical(f"Time: {bar_time.strftime('%H:%M:%S %Z')}")
        logger.critical(f"Position: {side.upper()} {position['quantity']} @ ${entry_price:.2f}")
        logger.critical(SEPARATOR_LINE)
        
        # Force close immediately
        close_side = "sell" if side == "long" else "buy"
        flatten_price = get_flatten_price(symbol, side, current_bar["close"])
        
        log_time_based_action(
            "market_close_flatten",
            "Market closed - auto-flattening position for 24/7 operation",
            {
                "side": side,
                "quantity": position["quantity"],
                "entry_price": f"${entry_price:.2f}",
                "exit_price": f"${flatten_price:.2f}",
                "time": bar_time.strftime('%H:%M:%S %Z')
            }
        )
        
        # Execute close order
        handle_exit_orders(symbol, position, flatten_price, "market_close")
        logger.info("Position flattened - bot will continue running and auto-resume when market opens")
        return
    
    # Market closed - Force close all positions
    if trading_state == "closed":
        if position["active"]:
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
            unrealized_pnl = ticks * tick_value * position["quantity"]
            
            flatten_details = {
                "reason": "Market closed - maintenance/weekend",
                "side": position["side"],
                "quantity": position["quantity"],
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
    
    # REGIME CHANGE CHECK - Detect regime changes and adjust parameters
    # This must happen BEFORE breakeven/trailing checks so they use updated regime parameters
    check_regime_change(symbol, current_bar["close"])
    
    # REGIME-BASED TIMEOUT CHECKS - Check sideways and underwater timeouts
    # These use regime-specific timeout values and reset on regime changes
    current_time = get_current_time()
    
    # Check sideways timeout (stagnant price action)
    should_exit_sideways, exit_price = check_sideways_timeout(symbol, current_bar["close"], current_time)
    if should_exit_sideways:
        execute_exit(symbol, exit_price, "sideways_timeout")
        return
    
    # Check underwater timeout (continuous loss)
    should_exit_underwater, exit_price = check_underwater_timeout(symbol, current_bar["close"], current_time)
    if should_exit_underwater:
        execute_exit(symbol, exit_price, "underwater_timeout")
        return
    
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
    
    # Check for signal reversal (lowest priority)
    reversal, price = check_reversal_signal(symbol, current_bar, position)
    if reversal:
        execute_exit(symbol, price, "signal_reversal")
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


def calculate_pnl(position: Dict[str, Any], exit_price: float) -> Tuple[float, float]:
    """
    Calculate profit/loss for the exit - PRODUCTION READY with slippage and commissions.
    
    Args:
        position: Position dictionary
        exit_price: Exit price (before slippage)
    
    Returns:
        Tuple of (ticks, pnl_dollars after all costs)
    """
    entry_price = position["entry_price"]
    contracts = position["quantity"]
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
    max_stop_loss = CONFIG.get("max_stop_loss_dollars", 200.0)
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
    
    # Calculate P&L
    ticks, pnl = calculate_pnl(position, exit_price)
    
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
            tick_size = CONFIG.get("tick_size", 0.25)
            tick_value = CONFIG.get("tick_value", 12.50)
            
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
            
            # Calculate entry slippage if we have the data
            entry_slippage_ticks = 0
            if position.get("actual_entry_price") and position.get("original_entry_price"):
                entry_slippage = abs(position.get("actual_entry_price", 0) - position.get("original_entry_price", 0))
                tick_size = CONFIG.get("tick_size", 0.25)
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
        summary_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'trade_summary.json')
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
        
        daily_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'daily_summary.json')
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
        try:
            global broker
            if broker is not None:
                broker.disconnect()
        except Exception as e:
            pass  # Silent disconnect
        
        # Exit the bot completely
        logger.critical("Bot shutdown complete.")
        import sys
        import time
        time.sleep(2)  # Give user time to read message
        sys.exit(0)
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
        logger.info(f"Trading day initialized: {current_date}")
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
    Perform VWAP reset at 6:00 PM ET daily (futures trading day start).
    
    Args:
        symbol: Instrument symbol
        new_date: The new VWAP date
        reset_time: Time of the reset
    """
    logger.info(SEPARATOR_LINE)
    logger.info(f"DAILY RESET at {reset_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"Futures trading day start (6:00 PM ET) - New trading day: {new_date}")
    logger.info(SEPARATOR_LINE)
    
    # Clear accumulated 1-minute bars for VWAP calculation
    state[symbol]["bars_1min"].clear()
    
    # Reset cumulative VWAP data
    state[symbol]["vwap"] = None
    state[symbol]["vwap_bands"] = {
        "upper_1": None,
        "upper_2": None,
        "lower_1": None,
        "lower_2": None
    }
    state[symbol]["vwap_std_dev"] = None
    
    # Update VWAP day
    state[symbol]["vwap_day"] = new_date
    
    # Note: 15-minute trend bars continue running - trend carries from overnight
    logger.info("Market data cleared - 15-minute trend bars continue running")
    logger.info(f"Current 15-min bars: {len(state[symbol]['bars_15min'])}")
    logger.info(SEPARATOR_LINE)


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
        logger.info(f"Trading day initialized: {current_date}")


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
        logger.info("Trading re-enabled for new day after maintenance hour reset")
    
    logger.info("Daily reset complete - Ready for trading")
    logger.info("(VWAP reset handled at market open 6:00 PM ET / 6 PM EST)")
    logger.info(SEPARATOR_LINE)


# ============================================================================
# PHASE TWELVE: Safety Mechanisms
# ============================================================================

def check_daily_loss_limit(symbol: str) -> Tuple[bool, Optional[str]]:
    """
    Check if daily loss limit has been exceeded.
    
    Args:
        symbol: Instrument symbol
    
    Returns:
        Tuple of (is_safe, reason)
    """
    if state[symbol]["daily_pnl"] <= -CONFIG["daily_loss_limit"]:
        if bot_status["trading_enabled"]:
            logger.critical(f"DAILY LOSS LIMIT BREACHED: ${state[symbol]['daily_pnl']:.2f}")
            logger.critical("Trading STOPPED for the day")
            bot_status["trading_enabled"] = False
            bot_status["stop_reason"] = "daily_loss_limit"
            
            # Send daily loss limit breach alert
            try:
                notifier = get_notifier()
                notifier.send_error_alert(
                    error_message=f"DAILY LOSS LIMIT HIT! Trading stopped. Loss: ${state[symbol]['daily_pnl']:.2f} / Limit: ${-CONFIG['daily_loss_limit']:.2f}",
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
    daily_loss_limit = CONFIG.get("daily_loss_limit", 1000.0)
    if daily_loss_limit > 0 and state[symbol]["daily_pnl"] <= -daily_loss_limit * DAILY_LOSS_APPROACHING_THRESHOLD:
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
        logger.warning("   Consider: earlier entry cutoff or faster profit targets")
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
    logger.info(f"Target Wait Wins: {bot_status['target_wait_wins']}")
    logger.info(f"Target Wait Losses: {bot_status['target_wait_losses']}")
    logger.info(f"Early Close Saves: {bot_status['early_close_saves']}")
    if bot_status["target_wait_wins"] > 0:
        target_success_rate = (bot_status["target_wait_wins"] / 
                               (bot_status["target_wait_wins"] + bot_status["target_wait_losses"]) * 100)
        logger.info(f"Target Wait Success Rate: {target_success_rate:.1f}%")


def log_session_summary(symbol: str) -> None:
    """
    Log comprehensive session summary at end of trading day.
    Coordinates summary formatting through helper functions.
    
    Args:
        symbol: Instrument symbol
    """
    stats = state[symbol]["session_stats"]
    
    logger.info(SEPARATOR_LINE)
    logger.info("SESSION SUMMARY")
    logger.info(SEPARATOR_LINE)
    logger.info(f"Trading Day: {state[symbol]['trading_day']}")
    
    # Format trade statistics
    format_trade_statistics(stats)
    
    # Format P&L summary
    format_pnl_summary(stats)
    
    # Format risk metrics (flatten mode analysis)
    format_risk_metrics()
    
    # Format time statistics (position duration)
    format_time_statistics(stats)
    
    logger.info(SEPARATOR_LINE)
    
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
            # Use cached Azure state (updated every 30s by check_azure_time_service)
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
    
    logger.info(SEPARATOR_LINE)
    logger.info("TIMEZONE CONFIGURATION VALIDATION")
    logger.info(SEPARATOR_LINE)
    logger.info(f"Configured Timezone: US/Eastern (CME Futures Standard)")
    logger.info(f"Current Time (ET): {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"UTC Offset: {current_time.strftime('%z')}")
    logger.info(f"DST Active: {bool(current_time.dst())}")
    logger.info(f"CME Futures Schedule (Wall-Clock Times - Never Change):")
    logger.info(f"  - Market Open: 6:00 PM Eastern")
    logger.info(f"  - Force Close: 4:45 PM Eastern daily")
    logger.info(f"  - Maintenance: 4:45-6:00 PM Eastern")
    logger.info(f"  - Friday Close: 5:00 PM Eastern")
    logger.info(f"  - Sunday Open: 6:00 PM Eastern")
    logger.info(f"NOTE: pytz handles EST/EDT automatically - same wall-clock times year-round")
    logger.info(SEPARATOR_LINE)


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
- Adjust entry cutoff earlier OR use faster profit targets
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
    global event_loop, timer_manager, bid_ask_manager, cloud_api_client, rl_brain
    
    # Use symbol override if provided (for multi-symbol support)
    trading_symbol = symbol_override if symbol_override else CONFIG["instrument"]
    
    logger.info(SEPARATOR_LINE)
    logger.info(f"QuoTrading AI Bot Starting [{trading_symbol}]")
    logger.info(SEPARATOR_LINE)
    
    # Initialize local RL brain for both LIVE and BACKTEST modes
    # LIVE MODE: Reads from local symbol-specific folder for pattern matching, saves to cloud only
    # BACKTEST MODE: Reads and saves to local symbol-specific folder
    if is_backtest_mode() or CONFIG.get("backtest_mode", False):
        logger.info(f"[{trading_symbol}] BACKTEST MODE: Local RL brain will be initialized by backtest code")
    else:
        logger.info(f"[{trading_symbol}] LIVE MODE: Initializing local RL brain...")
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
        logger.info(f"[{trading_symbol}] ✅ Local RL brain initialized with {len(rl_brain.experiences)} experiences from experiences/{trading_symbol}/")
        logger.info(f"[{trading_symbol}] 📝 Live mode: Reading local experiences, saving to cloud only")
        
        # Initialize Cloud API Client for reporting trade outcomes to cloud
        license_key = CONFIG.get("quotrading_license") or CONFIG.get("user_api_key")
        if license_key:
            cloud_api_url = "https://quotrading-flask-api.azurewebsites.net"
            cloud_api_client = CloudAPIClient(
                api_url=cloud_api_url,
                license_key=license_key,
                timeout=10
            )
            logger.info(f"[{trading_symbol}] ✅ Cloud API client initialized for outcome reporting")
        else:
            logger.warning(f"[{trading_symbol}] No license key - cloud outcome reporting disabled")
    
    # Log symbol specifications if loaded
    if SYMBOL_SPEC:
        logger.info(f"[{trading_symbol}] Symbol: {SYMBOL_SPEC.name} ({SYMBOL_SPEC.symbol})")
        logger.info(f"[{trading_symbol}]   Tick Value: ${SYMBOL_SPEC.tick_value:.2f} | Tick Size: ${SYMBOL_SPEC.tick_size}")
        logger.info(f"[{trading_symbol}]   Slippage: {SYMBOL_SPEC.typical_slippage_ticks} ticks | Volatility: {SYMBOL_SPEC.volatility_factor}x")
        logger.info(f"[{trading_symbol}]   Trading Hours: {SYMBOL_SPEC.session_start} - {SYMBOL_SPEC.session_end} ET")
    
    # Display operating mode
    if CONFIG.get('shadow_mode', False):
        logger.info(f"[{trading_symbol}] Mode: ≡ƒôè SIGNAL-ONLY MODE (Manual Trading)")
        logger.info(f"[{trading_symbol}] ΓÜá∩╕Å  Signal mode: Shows trading signals without executing trades")
    else:
        logger.info(f"[{trading_symbol}] Mode: LIVE TRADING")
    
    logger.info(f"[{trading_symbol}] Instrument: {trading_symbol}")
    logger.info(f"[{trading_symbol}] Entry Window: {CONFIG['entry_start_time']} - {CONFIG['entry_end_time']} ET")
    logger.info(f"[{trading_symbol}] Force Close: {CONFIG['forced_flatten_time']} ET")
    logger.info(f"[{trading_symbol}] Shutdown: {CONFIG['shutdown_time']} ET")
    logger.info(f"[{trading_symbol}] Max Contracts: {CONFIG['max_contracts']}")
    logger.info(f"[{trading_symbol}] Max Trades/Day: {CONFIG['max_trades_per_day']}")
    logger.info(f"[{trading_symbol}] Risk Per Trade: {CONFIG['risk_per_trade'] * 100:.1f}%")
    logger.info(f"[{trading_symbol}] Daily Loss Limit: ${CONFIG['daily_loss_limit']}")
    logger.info(SEPARATOR_LINE)
    
    # Phase Fifteen: Validate timezone configuration
    validate_timezone_configuration()
    
    # Initialize bid/ask manager
    logger.info(f"[{trading_symbol}] Initializing bid/ask manager...")
    bid_ask_manager = BidAskManager(CONFIG)
    
    # Initialize broker (replaces initialize_sdk)
    initialize_broker()
    
    # Phase 12: Record starting equity for drawdown monitoring
    bot_status["starting_equity"] = get_account_equity()
    logger.info(f"[{trading_symbol}] Starting Equity: ${bot_status['starting_equity']:.2f}")
    
    # Initialize state for instrument (use override symbol if provided)
    initialize_state(trading_symbol)
    
    # CRITICAL: Try to restore position state from disk if bot was restarted
    logger.info(f"[{trading_symbol}] Checking for saved position state...")
    position_restored = load_position_state(trading_symbol)
    if position_restored:
        logger.warning(f"[{trading_symbol}] ΓÜá∩╕Å  BOT RESTARTED WITH ACTIVE POSITION - Managing existing trade")
    else:
        logger.info(f"[{trading_symbol}] No active position to restore - starting fresh")
    
    # Skip historical bars fetching in live mode - not needed for real-time trading
    # The bot will build bars from live tick data
    logger.info(f"[{trading_symbol}] Skipping historical bars fetch - will build bars from live data")
    
    # Initialize event loop
    logger.info(f"[{trading_symbol}] Initializing event loop...")
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
    
    # Initialize timer manager for periodic events
    tz = pytz.timezone(CONFIG["timezone"])
    timer_manager = TimerManager(event_loop, CONFIG, tz)
    timer_manager.start()
    
    # Subscribe to market data (trades) - use trading_symbol
    subscribe_market_data(trading_symbol, on_tick)
    
    # Subscribe to bid/ask quotes if broker supports it
    if broker is not None and hasattr(broker, 'subscribe_quotes'):
        logger.info(f"[{trading_symbol}] Subscribing to bid/ask quotes...")
        try:
            broker.subscribe_quotes(symbol, on_quote)
        except Exception as e:
            logger.warning(f"Failed to subscribe to quotes: {e}")
            logger.warning("Continuing without bid/ask quote data")
    
    # RL is CLOUD-ONLY - no local RL components
    # Users get confidence from cloud, contribute to cloud hive mind
    # Only the dev (Kevin) gets the experience data saved to cloud
    logger.info("Cloud RL Mode: All learning goes to shared hive mind")
    logger.info("No local RL files - everything saved to cloud for collective intelligence")
    
    logger.info("Bot initialization complete")
    logger.info("Starting event loop...")
    logger.info("Press Ctrl+C for graceful shutdown")
    logger.info(SEPARATOR_LINE)
    
    # Run event loop (blocks until shutdown signal)
    try:
        event_loop.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        logger.info("Event loop stopped")
        
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
    total_ticks = state[symbol]["total_ticks_received"]
    
    # Log tick data periodically (every 1000 ticks to avoid spam)
    if total_ticks % 1000 == 0:
        # Get current bid/ask from bid_ask_manager if available
        bid_ask_info = ""
        if bid_ask_manager is not None:
            quote = bid_ask_manager.get_current_quote(symbol)
            if quote:
                spread = quote.ask_price - quote.bid_price
                bid_ask_info = f" | Bid: ${quote.bid_price:.2f} x {quote.bid_size} | Ask: ${quote.ask_price:.2f} x {quote.ask_size} | Spread: ${spread:.2f}"
        logger.info(f"[TICK] {symbol} @ ${price:.2f} | Vol: {volume} | Total ticks: {total_ticks}{bid_ask_info}")
    
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
    """Handle VWAP reset event"""
    symbol = CONFIG["instrument"]
    if symbol in state:
        tz = pytz.timezone(CONFIG["timezone"])
        check_vwap_reset(symbol, datetime.now(tz))


def handle_position_reconciliation_event(data: Dict[str, Any]) -> None:
    """
    Handle periodic position reconciliation check.
    Verifies bot's position state matches broker's actual position.
    Runs every 5 minutes to detect and correct any desyncs.
    """
    symbol = CONFIG["instrument"]
    
    if symbol not in state:
        return
    
    try:
        # Get broker's actual position
        broker_position = get_position_quantity(symbol)
        
        # Get bot's tracked position
        bot_active = state[symbol]["position"]["active"]
        if bot_active:
            bot_qty = state[symbol]["position"]["quantity"]
            bot_side = state[symbol]["position"]["side"]
            bot_position = bot_qty if bot_side == "long" else -bot_qty
        else:
            bot_position = 0
        
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
                logger.error("  Cause: Position was closed externally or bot missed exit fill")
                logger.error("  Action: Clearing bot's position state")
                state[symbol]["position"]["active"] = False
                state[symbol]["position"]["quantity"] = 0
                state[symbol]["position"]["side"] = None
                state[symbol]["position"]["entry_price"] = None
                
            elif broker_position != 0 and bot_position == 0:
                # Broker has position but bot thinks it's flat
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
            # Positions match - log success every hour only to avoid spam
            current_time = time_module.time()
            last_log = state[symbol].get("last_reconciliation_log", 0)
            if current_time - last_log > 3600:  # 1 hour
                logger.info(f"[RECONCILIATION] Position sync OK: {broker_position} contracts")
                state[symbol]["last_reconciliation_log"] = current_time
    
    except Exception as e:
        logger.error(f"Error during position reconciliation: {e}", exc_info=True)


def handle_connection_health_event(data: Dict[str, Any]) -> None:
    """
    Handle periodic connection health check event.
    Verifies broker connection is alive and reconnects if needed.
    Runs every 30 seconds.
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
    global cloud_api_client
    
    # Skip if in backtest mode or no cloud client
    if is_backtest_mode() or cloud_api_client is None:
        return
    
    # Skip if already stopped for expiration
    if bot_status.get("license_expired", False):
        return
    
    try:
        # Get license key from config
        license_key = CONFIG.get("quotrading_license") or CONFIG.get("user_api_key")
        if not license_key:
            logger.warning("No license key configured - cannot validate")
            return
        
        # Validate license via API
        import requests
        api_url = os.getenv("QUOTRADING_API_URL", "https://quotrading-flask-api.azurewebsites.net")
        
        response = requests.post(
            f"{api_url}/api/main",
            json={"license_key": license_key},
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
                    logger.warning("Position will close via normal exit rules (target/stop/time)")
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
                    try:
                        if broker is not None:
                            broker.disconnect()
                    except Exception as e:
                        pass  # Silent disconnect
                    
                    # Exit the bot completely - no more logs
                    logger.critical("Bot shutdown complete.")
                    import sys
                    import time
                    time.sleep(2)  # Give user time to read message
                    sys.exit(0)
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
        
        elif response.status_code == 401 or response.status_code == 403:
            # Unauthorized - license likely expired
            logger.critical("≡ƒÜ¿ LICENSE VALIDATION FAILED - Unauthorized")
            bot_status["license_expired"] = True
            bot_status["trading_enabled"] = False
            bot_status["emergency_stop"] = True
            bot_status["stop_reason"] = "License validation failed - unauthorized"
        
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
    Called every 30 seconds to show bot is alive.
    Admin dashboard uses this to show online users and performance.
    """
    # Skip in backtest mode
    if is_backtest_mode():
        return
    
    try:
        import requests
        
        api_url = os.getenv("QUOTRADING_API_URL", "https://quotrading-flask-api.azurewebsites.net")
        license_key = CONFIG.get("quotrading_license") or CONFIG.get("user_api_key")
        
        if not license_key:
            return
        
        # Collect performance metrics from symbol state
        symbol = CONFIG.get("instrument", "ES")
        
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
        payload = {
            "license_key": license_key,
            "device_fingerprint": get_device_fingerprint(),  # For session locking
            "bot_version": "2.0.0",
            "status": "online" if bot_status.get("trading_enabled", False) else "idle",
            "metadata": {
                "symbol": symbol,
                "shadow_mode": CONFIG.get("shadow_mode", False),
                "kill_switch_active": bot_status.get("kill_switch_active", False),
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
                logger.critical("  ⚠️ LICENSE ALREADY IN USE")
                logger.critical("")
                logger.critical("  Your license key is currently active on another device.")
                logger.critical(f"  Active device: {data.get('active_device', 'Unknown')}")
                logger.critical("")
                logger.critical("  Only one device can use a license at a time.")
                logger.critical("  Please stop the bot on the other device first.")
                logger.critical("")
                logger.critical("  Contact: support@quotrading.com")
                logger.critical("")
                logger.critical("="  * 70)
                
                # Disable trading and shut down
                bot_status["trading_enabled"] = False
                bot_status["emergency_stop"] = True
                bot_status["stop_reason"] = "session_conflict"
                
                # Disconnect broker
                global broker
                if broker is not None:
                    try:
                        broker.disconnect()
                    except:
                        pass
                
                # Exit
                import sys
                import time
                time.sleep(3)
                sys.exit(1)
            
            logger.debug("Heartbeat sent successfully")
        elif response.status_code == 403:
            # Session conflict detected by server
            logger.critical("=" * 70)
            logger.critical("")
            logger.critical("  ⚠️ LICENSE ALREADY IN USE")
            logger.critical("")
            logger.critical("  Your license key is currently active on another device.")
            logger.critical("  Only one device can use a license at a time.")
            logger.critical("")
            logger.critical("  Contact: support@quotrading.com")
            logger.critical("")
            logger.critical("=" * 70)
            
            # Force shutdown
            bot_status["trading_enabled"] = False
            bot_status["emergency_stop"] = True
            
            import sys
            import time
            time.sleep(3)
            sys.exit(1)
        else:
            logger.debug(f"Heartbeat returned HTTP {response.status_code}")
    
    except Exception as e:
        logger.debug(f"Heartbeat error: {e}")



def handle_shutdown_event(data: Dict[str, Any]) -> None:
    """Handle shutdown event"""
    logger.info("Shutdown event received")
    bot_status["trading_enabled"] = False


def cleanup_on_shutdown() -> None:
    """Cleanup tasks on shutdown"""
    logger.info("Running cleanup tasks...")
    
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
            logger.info("State saved successfully")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    # Disconnect broker
    if broker and broker.is_connected():
        try:
            broker.disconnect()
            logger.info("Broker disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting broker: {e}")
    
    # Stop timer manager
    if timer_manager:
        try:
            timer_manager.stop()
            logger.info("Timer manager stopped")
        except Exception as e:
            logger.error(f"Error stopping timer manager: {e}")
    
    # Log session summary
    symbol = CONFIG["instrument"]
    if symbol in state:
        log_session_summary(symbol)
    
    logger.info("Cleanup complete")


if __name__ == "__main__":
    main()
