"""
VWAP Bounce Bot - Mean Reversion Trading Strategy
Event-driven bot that trades bounces off VWAP standard deviation bands

========================================================================
24/7 MULTI-USER READY ARCHITECTURE
========================================================================

This bot is designed to run continuously and support global users:

✅ UTC-ONLY DESIGN: All times in UTC (matches ES futures schedule and historical data)
✅ AUTO-FLATTEN: Automatically closes positions at 20:45 UTC (15 min before maintenance)
✅ AUTO-RESUME: Automatically resumes trading when market reopens (23:00 UTC Sunday)
✅ NO MANUAL SHUTDOWN: Bot runs 24/7, just pauses trading when market closed
✅ TIMEZONE SAFE: Works for users in any timezone (displays in user's local time)

Trading Hours (ES Futures - UTC):
- OPEN: Sunday 23:00 → Friday 21:00 UTC
- MAINTENANCE: 21:00-22:00 UTC daily (Monday-Thursday)
- FLATTEN: 20:45 UTC (15 min before maintenance)
- WEEKEND: Friday 21:00 → Sunday 23:00 UTC

Bot States:
- entry_window: Market open, trading allowed (Sunday 23:00 - 20:45 UTC daily)
- flatten_mode: 20:45-21:00 UTC, aggressively close positions (15 min before maintenance)
- closed: During maintenance (21:00-22:00 Mon-Thu) or weekend, auto-flatten positions

For Multi-User Subscriptions:
- Add user_id to state dictionary for data isolation
- Each user gets their own position/RL/VWAP state
- Display times in user's local timezone (UTC → User TZ conversion)
- Bot continues running for all users regardless of individual timezone

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
import aiohttp
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
from dashboard import get_dashboard, Dashboard

# Conditionally import broker (only needed for live trading, not backtesting)
try:
    from broker_interface import create_broker, BrokerInterface
except ImportError:
    # Broker interface not available (e.g., in backtest-only mode)
    create_broker = None
    BrokerInterface = None

# ===== CRITICAL: SUPPRESS PROJECT_X_PY LOGGING IMMEDIATELY =====
# The TopStep SDK (project_x_py) uses structured JSON logging that corrupts dashboard
# Must suppress BEFORE any broker/SDK operations to prevent log spam
logging.getLogger("project_x_py").setLevel(logging.CRITICAL)
logging.getLogger("project_x_py").propagate = False
for handler in logging.getLogger("project_x_py").handlers[:]:
    logging.getLogger("project_x_py").removeHandler(handler)

# Suppress all sub-loggers
for logger_name in ["statistics", "statistics.bounded_statistics", "statistics.bounded_statistics.bounded_stats",
                     "order_manager", "order_manager.core", "position_manager", "position_manager.core",
                     "trading_suite", "data_manager", "websocket"]:
    full_name = f"project_x_py.{logger_name}"
    logging.getLogger(full_name).setLevel(logging.CRITICAL)
    logging.getLogger(full_name).propagate = False
    for handler in logging.getLogger(full_name).handlers[:]:
        logging.getLogger(full_name).removeHandler(handler)

# Load configuration from environment and config module
_bot_config = load_config()
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
    
    # Symbol specs loaded - dashboard will show this information
except Exception as e:
    # Symbol specs not available - will use defaults from config
    pass  # Symbol specs not loaded - will use defaults
    pass

# String constants
MSG_LIVE_TRADING_NOT_IMPLEMENTED = "Live trading not implemented - SDK integration required"
SEPARATOR_LINE = "=" * 60

# Recovery Mode Constants - Dynamic Risk Management
RECOVERY_APPROACHING_THRESHOLD = 0.80  # Trigger recovery mode at 80% of limits
RECOVERY_DEFAULT_SEVERITY = 0.80  # Default severity if not set
RECOVERY_SIZE_CRITICAL = 0.95  # At 95%+ of limits
RECOVERY_SIZE_HIGH = 0.90  # At 90-95% of limits
RECOVERY_SIZE_MODERATE = 0.80  # At 80-90% of limits
RECOVERY_MULTIPLIER_CRITICAL = 0.20  # Reduce to 20% of position at critical (was 33%)
RECOVERY_MULTIPLIER_HIGH = 0.40  # Reduce to 40% of position at high severity (was 50%)
RECOVERY_MULTIPLIER_MODERATE = 0.65  # Reduce to 65% of position at moderate severity (was 75%)

# Global broker instance (replaces sdk_client)
broker: Optional[BrokerInterface] = None

# Global event loop instance
event_loop: Optional[EventLoop] = None

# Global error recovery manager
recovery_manager: Optional[ErrorRecoveryManager] = None

# Global timer manager
timer_manager: Optional[TimerManager] = None

# Global RL brain for signal confidence learning
rl_brain: Optional[SignalConfidenceRL] = None

# Global bid/ask manager
bid_ask_manager: Optional[BidAskManager] = None

# Global adaptive exit manager (for streak tracking persistence)
adaptive_manager: Optional[Any] = None

# Global dashboard display
dashboard: Optional[Dashboard] = None

# Dashboard refresh throttle - prevent constant flickering
# Use list for mutable reference in closures
last_dashboard_refresh = [0.0]
dashboard_tick_counter = [0]  # Counter for dashboard updates (updates every 1000 ticks)

# State management dictionary
state: Dict[str, Any] = {}

# Backtest mode: Track current simulation time (for backtesting)
# When None, uses real datetime.now(). When set, uses this timestamp.
backtest_current_time: Optional[datetime] = None

# Global tracking for safety mechanisms (Phase 12)
bot_status: Dict[str, Any] = {
    "trading_enabled": True,
    "starting_equity": None,
    "last_tick_time": None,
    "emergency_stop": False,
    "stop_reason": None,
    "flatten_mode": False,  # Phase One: Aggressive exit mode flag
    # Phase 10: Track target vs early close decisions
    "target_wait_wins": 0,  # Times waiting for target paid off
    "target_wait_losses": 0,  # Times waiting for target caused reversal
    "early_close_saves": 0,  # Times early close prevented loss
    # PRODUCTION: Track trading costs
    "total_slippage_cost": 0.0,  # Total slippage costs across all trades
    "total_commission": 0.0,  # Total commissions across all trades
    # Recovery Mode: Dynamic risk management when approaching limits
    "recovery_confidence_threshold": None,  # Set when in recovery mode (higher confidence required)
    "recovery_severity": None,  # Severity level (0.8-1.0) indicating proximity to failure
    "aggressive_exit_mode": False,  # Set when at critical severity to aggressively manage positions
}


def setup_logging() -> logging.Logger:
    """Configure logging for the bot - Dashboard only display (logs suppressed except CRITICAL)"""
    # Suppress ALL logs - only show dashboard
    logging.basicConfig(
        level=logging.CRITICAL,  # Only critical errors
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Console output
        ]
    )
    
    # Suppress all module logs
    logging.getLogger("project_x_py").setLevel(logging.CRITICAL)
    logging.getLogger("project_x_py").propagate = False
    
    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================================
# CLOUD RL API INTEGRATION - Shared Learning Pool
# ============================================================================

# Cloud ML API configuration
CLOUD_ML_API_URL = os.getenv("CLOUD_ML_API_URL", "https://quotrading-signals.icymeadow-86b2969e.eastus.azurecontainerapps.io")
USE_CLOUD_SIGNALS = os.getenv("USE_CLOUD_SIGNALS", "true").lower() == "true"

# Generate privacy-preserving user ID
def get_user_id() -> str:
    """Generate consistent user ID from broker username"""
    username = CONFIG.get("broker_username", "default_user")
    return hashlib.md5(username.encode()).hexdigest()[:12]

USER_ID = get_user_id()


async def get_ml_confidence_async(rl_state: Dict[str, Any], side: str, current_spread_ticks: float = 1.0) -> Tuple[bool, float, float, str]:
    """
    Get ML confidence from cloud API (shared RL learning pool) with automatic retry.
    Returns: (take_signal, confidence, size_mult, reason)
    """
    if not USE_CLOUD_SIGNALS:
        # Fallback to local RL brain if cloud disabled
        if rl_brain is not None:
            return rl_brain.should_take_signal(rl_state, current_spread_ticks)
        return True, 0.5, 1.0, "Cloud API disabled, no local RL"
    
    # Retry up to 3 times with exponential backoff
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Prepare payload for cloud API (FIXED: using correct field names)
            payload = {
                "user_id": USER_ID,
                "symbol": CONFIG["instrument"],
                "signal": side.upper(),  # FIXED: was 'side', now 'signal'
                "rsi": rl_state.get("rsi", 50),
                "price": rl_state.get("price", 0),  # ADDED: need price for VWAP calc
                "vwap": rl_state.get("vwap", 0),    # ADDED: need VWAP reference
                "vix": rl_state.get("vix", 15.0)    # ADDED: volatility context
            }
            
            # Increase timeout slightly on retries
            timeout_seconds = 3.0 + (attempt * 1.0)  # 3s, 4s, 5s
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{CLOUD_ML_API_URL}/api/ml/get_confidence",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        confidence = data.get("confidence", 0.5)
                        take_signal = data.get("should_trade", False)
                        reason = data.get("reason", "Cloud RL evaluation")
                        size_mult = data.get("size_multiplier", 1.0)  # Get size mult from cloud
                        
                        if attempt > 0:
                            logger.info(f"[CLOUD RL] ✅ Success on retry {attempt + 1}")
                        
                        logger.debug(f"[CLOUD RL] {side.upper()} signal: {take_signal} (confidence: {confidence:.1%}, size: {size_mult:.2f}x) - {reason}")
                        logger.debug(f"   Shared pool: {data.get('total_trades', 0)} trades, WR: {data.get('win_rate', 0):.1%}")
                        
                        return take_signal, confidence, size_mult, reason
                    else:
                        logger.warning(f"Cloud ML API returned status {response.status} (attempt {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(0.5 * (attempt + 1))  # 0.5s, 1s, 1.5s backoff
                            continue
                        logger.error("Cloud API failed after all retries - REJECTING TRADE for safety")
                        return False, 0.0, 1.0, "Cloud API error - rejecting trade for safety"
                        
        except asyncio.TimeoutError:
            logger.warning(f"Cloud ML API timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            logger.error("Cloud API timeout after all retries - REJECTING TRADE for safety")
            return False, 0.0, 1.0, "API timeout - rejecting trade for safety"
        except Exception as e:
            logger.warning(f"Cloud ML API error: {e} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            logger.error(f"Cloud API error after all retries - REJECTING TRADE for safety: {e}")
            return False, 0.0, 1.0, f"API error - rejecting trade for safety"
    
    # Should never reach here, but just in case
    logger.error("Unknown error in RL API call - REJECTING TRADE for safety")
    return False, 0.0, 1.0, "Unknown error - rejecting trade for safety"


def get_ml_confidence(rl_state: Dict[str, Any], side: str, current_spread_ticks: float = 1.0) -> Tuple[bool, float, float, str]:
    """Synchronous wrapper for get_ml_confidence_async"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(get_ml_confidence_async(rl_state, side, current_spread_ticks))
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
    Save trade result to cloud API (shared RL learning pool) with automatic retry.
    All users contribute to and learn from the same experience pool.
    """
    if not USE_CLOUD_SIGNALS:
        # Save to local RL brain if cloud disabled
        if rl_brain is not None:
            rl_brain.record_outcome(rl_state, True, pnl, duration_minutes, execution_data)
        return
    
    # Retry up to 2 times for saves (less critical than getting confidence)
    max_retries = 2
    for attempt in range(max_retries):
        try:
            # Prepare trade experience payload
            payload = {
                "user_id": USER_ID,
                "symbol": CONFIG["instrument"],
                "side": side,
                "signal": f"{side}_bounce",  # "long_bounce" or "short_bounce"
                "rsi": rl_state.get("rsi", 50),
                "vwap_distance": rl_state.get("vwap_distance", 0),
                "volume_ratio": rl_state.get("volume_ratio", 1.0),
                "streak": rl_state.get("streak", 0),
                "time_of_day_score": rl_state.get("time_of_day_score", 0.5),
                "volatility": rl_state.get("volatility", 1.0),
                "pnl": pnl,
                "duration_minutes": duration_minutes,
                "exit_reason": execution_data.get("exit_reason", "unknown")
            }
            
            # Increase timeout on retries
            timeout_seconds = 5.0 + (attempt * 2.0)  # 5s, 7s
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{CLOUD_ML_API_URL}/api/ml/save_trade",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if attempt > 0:
                            logger.info(f"[CLOUD RL] ✅ Trade saved on retry {attempt + 1}")
                        else:
                            logger.info(f"[CLOUD RL] ✅ Trade saved to hive mind: ${pnl:+.2f}")
                        logger.info(f"   🧠 Shared pool: {data.get('total_shared_trades', 0):,} total trades, WR: {data.get('shared_win_rate', 0):.1%}")
                        return  # Success!
                    else:
                        logger.warning(f"Failed to save trade (status {response.status}, attempt {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(1.0 * (attempt + 1))  # 1s, 2s backoff
                            continue
                        logger.error(f"❌ Trade NOT saved to cloud after {max_retries} attempts - data point lost")
                        
        except Exception as e:
            logger.warning(f"Error saving trade: {e} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(1.0 * (attempt + 1))
                continue
            logger.error(f"❌ Trade NOT saved to cloud after {max_retries} attempts - data point lost")


async def save_rejected_signal_async(
    rl_state: Dict[str, Any],
    side: str,
    rejection_reason: str,
    price_move_ticks: float
) -> None:
    """
    Save rejected signal to cloud API so RL can learn from missed opportunities.
    Helps RL understand: "Should I have taken this trade anyway?"
    """
    if not USE_CLOUD_SIGNALS:
        return
    
    # Only retry once for rejected signals (less critical than actual trades)
    max_retries = 1
    for attempt in range(max_retries):
        try:
            payload = {
                "user_id": USER_ID,
                "symbol": CONFIG["instrument"],
                "side": side,
                "signal": f"{side}_bounce",
                "rsi": rl_state.get("rsi", 50),
                "vwap_distance": rl_state.get("vwap_distance", 0),
                "volume_ratio": rl_state.get("volume_ratio", 1.0),
                "streak": rl_state.get("streak", 0),
                "time_of_day_score": rl_state.get("time_of_day_score", 0.5),
                "volatility": rl_state.get("volatility", 1.0),
                "took_trade": False,
                "rejection_reason": rejection_reason,
                "price_move_ticks": price_move_ticks
            }
            
            timeout_seconds = 3.0
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{CLOUD_ML_API_URL}/api/ml/save_rejected_signal",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds)
                ) as response:
                    if response.status == 200:
                        logger.debug(f"[CLOUD RL] Rejected signal saved: {rejection_reason}")
                        return  # Success!
                    else:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(0.5)
                            continue
                        
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5)
                continue


def save_rejected_signal(
    rl_state: Dict[str, Any],
    side: str,
    rejection_reason: str,
    price_move_ticks: float
) -> None:
    """Synchronous wrapper for save_rejected_signal_async"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(save_rejected_signal_async(rl_state, side, rejection_reason, price_move_ticks))
        else:
            loop.run_until_complete(save_rejected_signal_async(rl_state, side, rejection_reason, price_move_ticks))
    except Exception as e:
        logger.debug(f"Failed to save rejected signal: {e}")


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
    Uses TopStep broker with error recovery and circuit breaker.
    SHADOW MODE: Shows trading signals without executing (manual trading mode).
    """
    global broker, recovery_manager
    
    # ===== LICENSE VALIDATION =====
    # Check if user has valid license before connecting to broker
    license_key = os.getenv("QUOTRADING_LICENSE_KEY")
    
    # Skip validation if admin/dev mode
    admin_key = os.getenv("QUOTRADING_API_KEY")
    if admin_key:
        logger.info("✅ Admin mode detected - skipping license check")
    elif license_key:
        logger.info("🔐 Validating license...")
        try:
            import requests
            api_url = os.getenv("QUOTRADING_API_URL", "https://quotrading-signals.icymeadow-86b2969e.eastus.azurecontainerapps.io")
            response = requests.post(
                f"{api_url}/api/license/validate",
                json={"license_key": license_key},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("valid"):
                    logger.info(f"✅ License validated - Welcome {data.get('customer_name', 'Trader')}!")
                else:
                    logger.error("❌ INVALID LICENSE - Bot will not start")
                    logger.error(f"Reason: {data.get('message', 'Unknown error')}")
                    sys.exit(1)
            else:
                logger.error(f"❌ License validation failed - HTTP {response.status_code}")
                logger.error("Please contact support@quotrading.com")
                sys.exit(1)
        except Exception as e:
            logger.error(f"❌ License validation error: {e}")
            logger.error("Cannot start bot without valid license")
            sys.exit(1)
    else:
        logger.error("❌ NO LICENSE KEY FOUND")
        logger.error("Please set QUOTRADING_LICENSE_KEY in your .env file")
        logger.error("Contact support@quotrading.com to purchase a license")
        sys.exit(1)
    
    # In shadow mode, show signals only (no execution)
    if CONFIG.get("shadow_mode", False):
        logger.info("📊 SIGNAL-ONLY MODE - Shows signals without executing trades")
    
    logger.info("Initializing broker interface...")
    
    # Create error recovery manager
    recovery_manager = ErrorRecoveryManager(CONFIG)
    
    # Create broker using configuration
    # In shadow mode, broker streams data but doesn't execute actual orders
    broker = create_broker(_bot_config.api_token, _bot_config.username)
    
    # CRITICAL: Suppress project_x_py logging immediately after broker creation
    # The TopStep SDK uses structured JSON logging that corrupts dashboard display
    import logging as log_module
    log_module.getLogger("project_x_py").setLevel(log_module.CRITICAL)
    log_module.getLogger("project_x_py.statistics").setLevel(log_module.CRITICAL)
    log_module.getLogger("project_x_py.statistics.bounded_statistics").setLevel(log_module.CRITICAL)
    log_module.getLogger("project_x_py.order_manager").setLevel(log_module.CRITICAL)
    log_module.getLogger("project_x_py.position_manager").setLevel(log_module.CRITICAL)
    log_module.getLogger("project_x_py.trading_suite").setLevel(log_module.CRITICAL)
    
    # Connect to broker (initial connection doesn't use circuit breaker)
    logger.info("=" * 80)
    logger.info("🔌 CONNECTING TO TOPSTEP...")
    logger.info("=" * 80)
    if not broker.connect():
        logger.critical("❌ FAILED TO CONNECT TO TOPSTEP")
        logger.error("Failed to connect to broker")
        return False
        raise RuntimeError("Broker connection failed")
    
    logger.info("=" * 80)
    logger.info("✅ TOPSTEP CONNECTION SUCCESSFUL")
    logger.info("=" * 80)
    logger.info("Broker connected successfully")
    
    # Get account balance - retry up to 3 times
    equity = 0.0
    for attempt in range(3):
        try:
            equity = broker.get_account_equity()
            if equity > 0:
                logger.info(f"💰 Account Balance: ${equity:,.2f}")
                logger.info("=" * 80)
                bot_status["starting_equity"] = equity
                bot_status["account_balance"] = equity
                break
            else:
                logger.warning(f"Account balance returned {equity}, retrying... (attempt {attempt + 1}/3)")
                time_module.sleep(1)
        except Exception as e:
            logger.warning(f"Could not fetch account balance (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                time_module.sleep(1)
    
    # If equity fetch failed, use config value as fallback
    if equity == 0:
        config_balance = CONFIG.get("fetched_account_balance") or CONFIG.get("topstep_starting_balance") or 50000.0
        logger.warning(f"⚠️  Using fallback balance from config: ${config_balance:,.2f}")
        bot_status["starting_equity"] = config_balance
        bot_status["account_balance"] = config_balance
        equity = config_balance
    
    
    # Send bot startup alert
    try:
        notifier = get_notifier()
        notifier.send_error_alert(
            error_message=f"Bot started successfully and connected to broker. Ready to trade {CONFIG.get('instrument', 'ES')}.",
            error_type="Bot Started"
        )
    except Exception as e:
        logger.debug(f"Failed to send startup alert: {e}")


def check_cloud_kill_switch() -> None:
    """
    Check if cloud kill switch is active.
    Called every 30 seconds as part of health check.
    
    KILL SWITCH BEHAVIOR:
    1. ACTIVATE: Flatten positions → Disconnect broker (NO DATA) → Go idle
    2. DEACTIVATE: Auto-reconnect broker → Resume trading
    
    This allows you to remotely pause ALL customer bots for:
    - Scheduled maintenance
    - Strategy updates
    - Emergency situations
    """
    global broker
    
    try:
        import requests
        
        cloud_api_url = CONFIG.get("cloud_api_url", "https://quotrading-signals.icymeadow-86b2969e.eastus.azurecontainerapps.io")
        
        response = requests.get(
            f"{cloud_api_url}/api/kill_switch/status",
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            kill_switch_active = data.get("kill_switch_active", False)
            reason = data.get("reason", "Maintenance mode")
            
            # ===== ACTIVATE KILL SWITCH =====
            if kill_switch_active and not bot_status.get("kill_switch_active", False):
                logger.critical("=" * 80)
                logger.critical("🛑 KILL SWITCH ACTIVATED - SHUTTING DOWN")
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
                        logger.critical("  [KILL SWITCH] ✅ Broker disconnected - NO DATA RUNNING")
                except Exception as e:
                    logger.error(f"  [KILL SWITCH] Error disconnecting: {e}")
                
                # Step 3: Disable trading
                bot_status["trading_enabled"] = False
                bot_status["kill_switch_active"] = True
                
                # Step 4: Alert customer
                try:
                    notifier = get_notifier()
                    notifier.send_error_alert(
                        error_message=f"🛑 KILL SWITCH: {reason}. All positions closed, broker disconnected. Bot will auto-resume when maintenance completes.",
                        error_type="Kill Switch Activated"
                    )
                except Exception as e:
                    logger.debug(f"Failed to send alert: {e}")
                
                logger.critical("  [KILL SWITCH] Bot is IDLE. Checking every 30s for resume signal...")
                logger.critical("=" * 80)
            
            # ===== DEACTIVATE KILL SWITCH (AUTO-RECONNECT) =====
            elif not kill_switch_active and bot_status.get("kill_switch_active", False):
                logger.critical("=" * 80)
                logger.critical("✅ KILL SWITCH OFF - AUTO-RECONNECTING")
                logger.critical("=" * 80)
                
                # Step 1: RECONNECT TO BROKER
                try:
                    if broker is not None:
                        logger.critical("  [RECONNECT] Connecting to broker...")
                        success = broker.connect(max_retries=3)
                        if success:
                            logger.critical("  [RECONNECT] ✅ Broker connected - Data feed active")
                        else:
                            logger.error("  [RECONNECT] ❌ Connection failed - Will retry in 30s")
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
                        error_message="✅ Kill switch deactivated. Broker reconnected, trading resumed. Bot is back online.",
                        error_type="Trading Resumed"
                    )
                except Exception as e:
                    logger.debug(f"Failed to send alert: {e}")
                
                logger.critical("  [RECONNECT] ✅ Trading enabled. Bot fully operational.")
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
    - Maintenance windows (Mon-Thu 21:00-22:00 UTC, Fri 21:00 - Sun 23:00 UTC)
    - Economic events (FOMC/NFP/CPI blocking)
    - Trading permission (go/no-go flag)
    
    Returns:
        Trading state: 'entry_window', 'flatten_mode', 'closed', or 'event_block'
    """
    global dashboard
    
    try:
        import requests
        import time
        
        cloud_api_url = CONFIG.get("cloud_api_url", "https://quotrading-signals.icymeadow-86b2969e.eastus.azurecontainerapps.io")
        
        # Measure latency
        start_time = time.time()
        response = requests.get(
            f"{cloud_api_url}/api/time/simple",
            timeout=5
        )
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Update dashboard with server latency
        if dashboard:
            latency_str = f"{latency_ms}ms"
            if latency_ms >= 200:
                latency_str += " (high latency)"
            
            dashboard.update_bot_data({
                "server_latency": latency_str,
                "server_latency_ms": latency_ms
            })
        
        if response.status_code == 200:
            data = response.json()
            trading_allowed = data.get("trading_allowed", False)
            halt_reason = data.get("halt_reason", "")
            current_et = data.get("current_et", "")
            
            # Store Azure time for bot awareness
            bot_status["azure_time"] = current_et
            bot_status["trading_allowed"] = trading_allowed
            bot_status["halt_reason"] = halt_reason
            
            # Determine state based on Azure response
            if not trading_allowed:
                if "maintenance" in halt_reason.lower():
                    state = "closed"  # Maintenance window
                elif "event" in halt_reason.lower() or "fomc" in halt_reason.lower() or "nfp" in halt_reason.lower():
                    # Check if FOMC blocking is enabled in config
                    fomc_enabled = CONFIG.get("fomc_block_enabled", True)
                    if fomc_enabled:
                        state = "event_block"  # Economic event blocking
                        # Log FOMC block (only once when it activates)
                        if not bot_status.get("event_block_active", False):
                            logger.warning("=" * 80)
                            logger.warning(f"📅 ECONOMIC EVENT BLOCK ACTIVATED: {halt_reason}")
                            logger.warning("  Trading halted 30 min before to 1 hour after event")
                            logger.warning("  Will auto-resume when event window closes")
                            logger.warning("=" * 80)
                            bot_status["event_block_active"] = True
                            
                            # Update dashboard with FOMC block
                            if dashboard:
                                dashboard.update_bot_data({
                                    "fomc_active": True,
                                    "fomc_message": halt_reason
                                })
                                dashboard.display()
                    else:
                        logger.info(f"[TIME SERVICE] Economic event active but FOMC blocking disabled: {halt_reason}")
                        state = "entry_window"  # User disabled event blocking
                elif "weekend" in halt_reason.lower() or "closed" in halt_reason.lower():
                    state = "closed"  # Weekend or market closed
                else:
                    state = "closed"  # Unknown halt reason - be safe
            else:
                # Clear event block flag when trading resumes
                if bot_status.get("event_block_active", False):
                    logger.info("=" * 80)
                    logger.info("✅ ECONOMIC EVENT ENDED - Trading resumed")
                    logger.info("=" * 80)
                    bot_status["event_block_active"] = False
                    
                    # Clear FOMC notification from dashboard
                    if dashboard:
                        dashboard.update_bot_data({
                            "fomc_active": False,
                            "fomc_message": ""
                        })
                        dashboard.display()
                
                # Trading allowed - check if we're approaching maintenance (flatten mode)
                # Check if current UTC time is 20:45-21:00 (flatten mode)
                try:
                    # Parse current_et which is ISO format string from Azure
                    utc_now = datetime.now(pytz.UTC)
                    utc_hour = utc_now.hour
                    utc_minute = utc_now.minute
                    
                    # Flatten mode: 20:45-21:00 UTC (before maintenance)
                    if (utc_hour == 20 and utc_minute >= 45) or (utc_hour == 21 and utc_minute == 0):
                        state = "flatten_mode"
                    else:
                        state = "entry_window"
                except Exception as e:
                    logger.debug(f"Time parsing error: {e}")
                    state = "entry_window"
            
            # Cache state for get_trading_state() to use
            bot_status["azure_trading_state"] = state
            return state
        else:
            # Azure unreachable - clear cached state to trigger fallback
            logger.debug(f"Time service returned {response.status_code}, using local time")
            bot_status["azure_trading_state"] = None
            
            # Update dashboard to show server error
            if dashboard:
                dashboard.update_bot_data({
                    "server_latency": "ERROR",
                    "server_latency_ms": 999
                })
            
            return None  # Signal to use local get_trading_state()
            
    except Exception as e:
        # Non-critical - if cloud unreachable, fall back to local time
        logger.debug(f"Time service check skipped (cloud unreachable): {e}")
        bot_status["azure_trading_state"] = None
        
        # Update dashboard to show server error
        if dashboard:
            dashboard.update_bot_data({
                "server_latency": "OFFLINE",
                "server_latency_ms": 999
            })
        
        return None  # Signal to use local get_trading_state()


def check_broker_connection() -> None:
    """
    Periodic health check for broker connection AND cloud services.
    Verifies connection is alive and attempts reconnection if needed.
    Called every 30 seconds by timer manager.
    Only logs when there's an issue to avoid spam.
    """
    global broker
    
    # CRITICAL: Check cloud services FIRST (kill switch + time service)
    try:
        check_cloud_kill_switch()
    except Exception as e:
        logger.debug(f"Kill switch check failed (non-critical): {e}")
    
    try:
        check_azure_time_service()
    except Exception as e:
        logger.debug(f"Time service check failed (non-critical): {e}")
    
    # AUTO-IDLE: Disconnect broker during maintenance (no data needed)
    current_time = get_current_time()
    trading_state = get_trading_state(current_time)
    
    if trading_state == "closed" and not bot_status.get("maintenance_idle", False):
        halt_reason = bot_status.get("halt_reason", "")
        
        # Only go idle during maintenance, not weekend
        if "maintenance" in halt_reason.lower() or (current_time.weekday() < 5 and current_time.time() >= datetime_time(21, 0) and current_time.time() < datetime_time(22, 0)):
            logger.critical(SEPARATOR_LINE)
            logger.critical("🔧 MAINTENANCE WINDOW - GOING IDLE")
            logger.critical(f"Time: {current_time.strftime('%H:%M:%S %Z')}")
            logger.critical("  Disconnecting broker to save resources during maintenance")
            logger.critical("  Will auto-reconnect at 22:00 UTC when market reopens")
            logger.critical(SEPARATOR_LINE)
            
            # Disconnect broker (stops all data feeds)
            try:
                if broker is not None and broker.connected:
                    broker.disconnect()
                    logger.critical("  ✅ Broker disconnected - Bot is IDLE")
            except Exception as e:
                logger.error(f"  ❌ Error disconnecting: {e}")
            
            bot_status["maintenance_idle"] = True
            bot_status["trading_enabled"] = False
            logger.critical(f"  [DEBUG] Set trading_enabled=False due to maintenance window")
            logger.critical("  Bot will check every 30s for market reopen...")
            return  # Skip broker health check since we just disconnected
    
    # AUTO-RECONNECT: Reconnect broker when market reopens at 11 PM UTC
    elif trading_state == "entry_window" and bot_status.get("maintenance_idle", False):
        logger.critical(SEPARATOR_LINE)
        logger.critical("✅ MARKET REOPENED - AUTO-RECONNECTING")
        logger.critical(f"Time: {current_time.strftime('%H:%M:%S %Z')}")
        logger.critical(SEPARATOR_LINE)
        
        # Reconnect to broker
        try:
            if broker is not None:
                logger.critical("  [RECONNECT] Connecting to broker...")
                success = broker.connect(max_retries=3)
                if success:
                    logger.critical("  [RECONNECT] ✅ Broker connected - Data feed active")
                    bot_status["maintenance_idle"] = False
                    bot_status["trading_enabled"] = True
                    logger.critical("  [RECONNECT] ✅ Trading enabled. Bot fully operational.")
                else:
                    logger.error("  [RECONNECT] ❌ Connection failed - Will retry in 30s")
        except Exception as e:
            logger.error(f"  [RECONNECT] Error: {e}")
        
        logger.critical(SEPARATOR_LINE)
        return  # Skip normal health check since we just reconnected
    
    if broker is None:
        logger.error("[HEALTH] Broker is None - cannot check connection")
        return
    
    # Check if broker reports as connected
    if not broker.connected:
        logger.warning("[HEALTH] Broker connection lost - attempting reconnection...")
        
        # Send connection error alert
        try:
            notifier = get_notifier()
            notifier.send_error_alert(
                error_message="Broker connection lost. Bot is attempting to reconnect automatically.",
                error_type="Connection Error"
            )
        except Exception as e:
            logger.debug(f"Failed to send connection error alert: {e}")
        
        try:
            # Attempt to reconnect
            success = broker.connect(max_retries=2)
            if success:
                logger.info("[HEALTH] Reconnection successful!")
            else:
                logger.error("[HEALTH] Reconnection failed - will retry in 30s")
        except Exception as e:
            logger.error(f"[HEALTH] Reconnection error: {e}")
        return
    
    # Connection looks healthy - do a lightweight ping test
    # Only log if there's a problem (silent success to avoid spam)
    try:
        # Try to get account equity as a connection health check
        equity = broker.get_account_equity()
        if equity is None or equity <= 0:
            logger.warning("[HEALTH] Connection may be stale - got invalid equity response")
            # Mark as disconnected to trigger reconnect on next check
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
    shadow_mode = CONFIG.get("shadow_mode", False)
    logger.info(f"{'[SHADOW MODE] ' if shadow_mode else ''}Stop Order: {side} {quantity} {symbol} @ {stop_price}")
    
    if shadow_mode:
        return {
            "order_id": f"SHADOW_STOP_{datetime.now(pytz.UTC).timestamp()}",
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
    shadow_mode = CONFIG.get("shadow_mode", False)
    logger.info(f"{'[SHADOW MODE] ' if shadow_mode else ''}Limit Order: {side} {quantity} {symbol} @ {limit_price}")
    
    if shadow_mode:
        return {
            "order_id": f"SHADOW_LIMIT_{datetime.now(pytz.UTC).timestamp()}",
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
# SHADOW TRADE TRACKING (Ghost Signals for RL Learning)
# ============================================================================

class ShadowTrade:
    """
    Represents a rejected signal being tracked in the background.
    Simulates what WOULD happen if we took the trade, providing learning data.
    
    FULLY SIMULATES A REAL TRADE:
    - Uses adaptive neural exit model (45 features → dynamic params)
    - Updates stop/breakeven/trailing EVERY bar (just like real trades)
    - Tracks MAE/MFE, partials, all metrics
    - Outcome reflects what would ACTUALLY happen with current exit logic
    """
    def __init__(self, symbol: str, rl_state: Dict, side: str, entry_price: float, 
                 entry_time: datetime, confidence: float, rejection_reason: str):
        self.symbol = symbol
        self.rl_state = rl_state.copy()  # Snapshot of RL state at rejection
        self.side = side  # 'long' or 'short'
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.confidence = confidence  # AI confidence when rejected
        self.rejection_reason = rejection_reason
        
        # Shadow position state (pretend we're IN the trade)
        self.quantity = 1  # Start with 1 contract (matches ghost simulation)
        self.remaining_quantity = 1
        self.stop_price = None  # Will be set by adaptive exit model
        self.breakeven_active = False
        self.trailing_active = False
        self.highest_price_reached = entry_price if side == 'long' else None
        self.lowest_price_reached = entry_price if side == 'short' else None
        
        # Performance tracking (like real trades)
        self.mae = 0.0  # Maximum Adverse Excursion
        self.mfe = 0.0  # Maximum Favorable Excursion
        self.partial_exits_completed = []  # Track partial exits
        
        # Exit params will be calculated dynamically each bar
        self.current_exit_params = None
        
        self.bars_elapsed = 0
        self.max_bars = 200  # Timeout after 200 bars (~3 hours)
        self.completed = False
        self.outcome = None  # Will be set when shadow completes
    
    def update(self, bar: Dict, bars_1min: deque, adaptive_manager, config: Dict) -> Optional[Dict]:
        """
        Update shadow with new bar data - FULLY SIMULATES REAL TRADE.
        
        Every bar:
        1. Calculate adaptive exit params (45-feature neural network)
        2. Update stop loss (breakeven, trailing, time-decay)
        3. Check if stopped out or target hit
        4. Track MAE/MFE
        5. Handle partial exits
        
        Args:
            bar: Latest 1-minute bar
            bars_1min: Full bar history (for adaptive exit calculation)
            adaptive_manager: AdaptiveExitManager instance
            config: Bot config
            
        Returns:
            Dict with outcome if shadow completed, None if still tracking
        """
        if self.completed:
            return None
        
        self.bars_elapsed += 1
        current_price = bar['close']
        
        # ========== STEP 1: Get Adaptive Exit Params (Neural Network) ==========
        try:
            from adaptive_exits import get_adaptive_exit_params
            
            # Create fake position dict (pretend we're in trade)
            fake_position = {
                'entry_price': self.entry_price,
                'side': self.side,
                'quantity': self.quantity,
                'remaining_quantity': self.remaining_quantity,
                'entry_time': self.entry_time,
                'entry_confidence': self.confidence,
                'breakeven_active': self.breakeven_active,
                'trailing_stop_active': self.trailing_active,
                'highest_price_reached': self.highest_price_reached,
                'lowest_price_reached': self.lowest_price_reached,
                'mae': self.mae,
                'mfe': self.mfe,
                'partial_exit_1_completed': len(self.partial_exits_completed) >= 1,
                'partial_exit_2_completed': len(self.partial_exits_completed) >= 2,
                'partial_exit_3_completed': len(self.partial_exits_completed) >= 3,
            }
            
            # Get dynamic exit params from neural network
            adaptive_params = get_adaptive_exit_params(
                bars=bars_1min,
                position=fake_position,
                current_price=current_price,
                config=config,
                adaptive_manager=adaptive_manager,
                entry_confidence=self.confidence
            )
            
            self.current_exit_params = adaptive_params
            
        except Exception as e:
            # Fallback to simple stop if adaptive fails
            atr = self.rl_state.get('atr', 2.0)
            stop_distance = atr * 3.6
            if self.side == 'long':
                self.stop_price = self.entry_price - stop_distance
            else:
                self.stop_price = self.entry_price + stop_distance
        
        # ========== STEP 2: Update Stop Loss (Breakeven/Trailing/Time-Decay) ==========
        if self.current_exit_params:
            # Calculate current R-multiple
            if self.side == 'long':
                initial_risk = self.entry_price - self.stop_price if self.stop_price else 0
                current_r = (current_price - self.entry_price) / initial_risk if initial_risk > 0 else 0
            else:
                initial_risk = self.stop_price - self.entry_price if self.stop_price else 0
                current_r = (self.entry_price - current_price) / initial_risk if initial_risk > 0 else 0
            
            # Breakeven activation
            if not self.breakeven_active:
                breakeven_threshold = self.current_exit_params.get('breakeven_threshold_ticks', 0) / 4.0  # ticks to R
                if current_r >= breakeven_threshold:
                    self.breakeven_active = True
                    breakeven_offset = self.current_exit_params.get('breakeven_offset_ticks', 0.5) / 4.0
                    if self.side == 'long':
                        self.stop_price = self.entry_price + (initial_risk * breakeven_offset)
                    else:
                        self.stop_price = self.entry_price - (initial_risk * breakeven_offset)
            
            # Trailing stop activation
            if self.breakeven_active and not self.trailing_active:
                trailing_threshold = 1.5  # Activate trailing at 1.5R
                if current_r >= trailing_threshold:
                    self.trailing_active = True
            
            # Update trailing stop
            if self.trailing_active:
                trailing_distance_ticks = self.current_exit_params.get('trailing_distance_ticks', 2.0)
                trailing_distance = trailing_distance_ticks / 4.0  # ticks to points
                
                if self.side == 'long':
                    if self.highest_price_reached is None or current_price > self.highest_price_reached:
                        self.highest_price_reached = current_price
                    new_stop = self.highest_price_reached - trailing_distance
                    if new_stop > self.stop_price:
                        self.stop_price = new_stop
                else:
                    if self.lowest_price_reached is None or current_price < self.lowest_price_reached:
                        self.lowest_price_reached = current_price
                    new_stop = self.lowest_price_reached + trailing_distance
                    if new_stop < self.stop_price:
                        self.stop_price = new_stop
        
        # ========== STEP 3: Track MAE/MFE ==========
        if self.side == 'long':
            unrealized_pnl = (current_price - self.entry_price) * 50 * self.remaining_quantity
            if unrealized_pnl < 0 and abs(unrealized_pnl) > self.mae:
                self.mae = abs(unrealized_pnl)
            if unrealized_pnl > 0 and unrealized_pnl > self.mfe:
                self.mfe = unrealized_pnl
        else:
            unrealized_pnl = (self.entry_price - current_price) * 50 * self.remaining_quantity
            if unrealized_pnl < 0 and abs(unrealized_pnl) > self.mae:
                self.mae = abs(unrealized_pnl)
            if unrealized_pnl > 0 and unrealized_pnl > self.mfe:
                self.mfe = unrealized_pnl
        
        # ========== STEP 4: Check for Stop Loss Hit ==========
        if self.stop_price:
            if self.side == 'long':
                if bar['low'] <= self.stop_price:
                    # Stopped out
                    pnl = (self.stop_price - self.entry_price) * 50 * self.remaining_quantity
                    self.outcome = {
                        'pnl': pnl,
                        'exit_price': self.stop_price,
                        'duration_min': self.bars_elapsed,
                        'exit_reason': 'ghost_stop_loss',
                        'took_trade': False,
                        'confidence': self.confidence,
                        'rejection_reason': self.rejection_reason,
                        'mae': self.mae,
                        'mfe': self.mfe,
                        'breakeven_activated': self.breakeven_active,
                        'trailing_activated': self.trailing_active,
                        'partials_taken': len(self.partial_exits_completed)
                    }
                    self.completed = True
                    return self.outcome
            else:  # short
                if bar['high'] >= self.stop_price:
                    pnl = (self.entry_price - self.stop_price) * 50 * self.remaining_quantity
                    self.outcome = {
                        'pnl': pnl,
                        'exit_price': self.stop_price,
                        'duration_min': self.bars_elapsed,
                        'exit_reason': 'ghost_stop_loss',
                        'took_trade': False,
                        'confidence': self.confidence,
                        'rejection_reason': self.rejection_reason,
                        'mae': self.mae,
                        'mfe': self.mfe,
                        'breakeven_activated': self.breakeven_active,
                        'trailing_activated': self.trailing_active,
                        'partials_taken': len(self.partial_exits_completed)
                    }
                    self.completed = True
                    return self.outcome
        
        # ========== STEP 5: Check for Partial Exits ==========
        if self.current_exit_params and self.remaining_quantity > 0:
            partial_1_r = self.current_exit_params.get('partial_1_r', 999)
            partial_2_r = self.current_exit_params.get('partial_2_r', 999)
            partial_3_r = self.current_exit_params.get('partial_3_r', 999)
            
            # Calculate current R
            if self.side == 'long':
                initial_risk = abs(self.entry_price - (self.stop_price or self.entry_price - 5))
                current_r = (current_price - self.entry_price) / initial_risk if initial_risk > 0 else 0
                
                # Check if hit partial targets
                if len(self.partial_exits_completed) == 0 and current_r >= partial_1_r:
                    if bar['high'] >= self.entry_price + (initial_risk * partial_1_r):
                        partial_pnl = initial_risk * partial_1_r * 50 * 0.33  # 1/3 position
                        self.partial_exits_completed.append({'r': partial_1_r, 'pnl': partial_pnl})
                        self.remaining_quantity *= 0.67
                
                elif len(self.partial_exits_completed) == 1 and current_r >= partial_2_r:
                    if bar['high'] >= self.entry_price + (initial_risk * partial_2_r):
                        partial_pnl = initial_risk * partial_2_r * 50 * 0.5  # 1/2 of remaining
                        self.partial_exits_completed.append({'r': partial_2_r, 'pnl': partial_pnl})
                        self.remaining_quantity *= 0.5
                
                elif len(self.partial_exits_completed) == 2 and current_r >= partial_3_r:
                    if bar['high'] >= self.entry_price + (initial_risk * partial_3_r):
                        # Full exit at R-target
                        total_pnl = sum(p['pnl'] for p in self.partial_exits_completed)
                        final_pnl = (current_price - self.entry_price) * 50 * self.remaining_quantity
                        self.outcome = {
                            'pnl': total_pnl + final_pnl,
                            'exit_price': current_price,
                            'duration_min': self.bars_elapsed,
                            'exit_reason': 'ghost_target',
                            'took_trade': False,
                            'confidence': self.confidence,
                            'rejection_reason': self.rejection_reason,
                            'mae': self.mae,
                            'mfe': self.mfe,
                            'breakeven_activated': self.breakeven_active,
                            'trailing_activated': self.trailing_active,
                            'partials_taken': len(self.partial_exits_completed)
                        }
                        self.completed = True
                        return self.outcome
        
        # ========== STEP 6: Check for Timeout ==========
        if self.bars_elapsed >= self.max_bars:
            # Exit at current close price
            total_partial_pnl = sum(p['pnl'] for p in self.partial_exits_completed)
            
            if self.side == 'long':
                final_pnl = (bar['close'] - self.entry_price) * 50 * self.remaining_quantity
            else:
                final_pnl = (self.entry_price - bar['close']) * 50 * self.remaining_quantity
            
            self.outcome = {
                'pnl': total_partial_pnl + final_pnl,
                'exit_price': bar['close'],
                'duration_min': self.bars_elapsed,
                'exit_reason': 'ghost_timeout',
                'took_trade': False,
                'confidence': self.confidence,
                'rejection_reason': self.rejection_reason,
                'mae': self.mae,
                'mfe': self.mfe,
                'breakeven_activated': self.breakeven_active,
                'trailing_activated': self.trailing_active,
                'partials_taken': len(self.partial_exits_completed)
            }
            self.completed = True
            return self.outcome
        
        return None  # Still tracking


def create_shadow_trade(symbol: str, rl_state: Dict, side: str, entry_price: float, 
                       confidence: float, rejection_reason: str) -> None:
    """
    Create a shadow trade to track a rejected signal.
    Shadow FULLY SIMULATES a real trade with adaptive neural exit model.
    Limit to max 5 active shadows to prevent memory bloat.
    
    Args:
        symbol: Instrument symbol
        rl_state: RL state snapshot at rejection
        side: 'long' or 'short'
        entry_price: Price where signal occurred
        confidence: AI confidence when rejected
        rejection_reason: Why signal was rejected
    """
    MAX_SHADOWS = 5
    
    # Remove any completed shadows first
    state[symbol]["shadow_trades"] = [s for s in state[symbol]["shadow_trades"] if not s.completed]
    
    # Only create new shadow if under limit
    if len(state[symbol]["shadow_trades"]) < MAX_SHADOWS:
        shadow = ShadowTrade(
            symbol=symbol,
            rl_state=rl_state,
            side=side,
            entry_price=entry_price,
            entry_time=get_current_time(),
            confidence=confidence,
            rejection_reason=rejection_reason
        )
        state[symbol]["shadow_trades"].append(shadow)
        logger.debug(f"[SHADOW] Created {side} shadow @ ${entry_price:.2f} (will use adaptive exit model)")
    else:
        logger.debug(f"[SHADOW] Skipped creating shadow (max {MAX_SHADOWS} active)")


def update_shadow_trades(symbol: str, bar: Dict) -> None:
    """
    Update all active shadow trades with the latest bar.
    Shadows FULLY SIMULATE real trades with adaptive neural exit model.
    Send completed shadow outcomes to cloud for RL learning.
    
    Args:
        symbol: Instrument symbol
        bar: Latest 1-minute bar
    """
    if not state[symbol]["shadow_trades"]:
        return
    
    completed_shadows = []
    
    # Get required data for adaptive exit model
    bars_1min = state[symbol]["bars_1min"]
    
    for shadow in state[symbol]["shadow_trades"]:
        # Update shadow with full simulation (adaptive exits, breakeven, trailing, partials)
        outcome = shadow.update(bar, bars_1min, adaptive_manager, CONFIG)
        if outcome:
            completed_shadows.append((shadow, outcome))
    
    # Send completed shadows to cloud
    for shadow, outcome in completed_shadows:
        # Send to cloud (async, fire and forget)
        send_shadow_outcome_to_cloud(shadow.rl_state, outcome)
        
        # Log for debugging
        pnl_sign = "+" if outcome['pnl'] > 0 else ""
        partials_info = f", {outcome['partials_taken']} partials" if outcome['partials_taken'] > 0 else ""
        breakeven_info = ", BE" if outcome.get('breakeven_activated') else ""
        trailing_info = ", TRAIL" if outcome.get('trailing_activated') else ""
        logger.info(f"[SHADOW] Completed {shadow.side} shadow: {pnl_sign}${outcome['pnl']:.2f} ({outcome['exit_reason']}) after {outcome['duration_min']} min{partials_info}{breakeven_info}{trailing_info}")
    
    # Remove completed shadows
    state[symbol]["shadow_trades"] = [s for s in state[symbol]["shadow_trades"] if not s.completed]


async def send_shadow_outcome_async(rl_state: Dict, outcome: Dict) -> None:
    """
    Send completed shadow trade outcome to cloud (async).
    
    Args:
        rl_state: RL state snapshot at rejection
        outcome: Shadow trade outcome with pnl, exit_reason, etc.
    """
    if not CLOUD_ML_API_URL:
        return  # Cloud not configured
    
    try:
        payload = {
            "user_id": USER_ID,
            "symbol": CONFIG["instrument"],
            "rl_state": rl_state,
            "outcome": outcome,
            "experience_type": "ghost_trade"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{CLOUD_ML_API_URL}/api/ml/save_experience",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=3.0)
            ) as response:
                if response.status == 200:
                    logger.debug(f"[CLOUD RL] Shadow outcome saved: {outcome['exit_reason']}")
                else:
                    logger.debug(f"[CLOUD RL] Shadow save failed: HTTP {response.status}")
    except Exception as e:
        logger.debug(f"[CLOUD RL] Shadow save error: {e}")


def send_shadow_outcome_to_cloud(rl_state: Dict, outcome: Dict) -> None:
    """
    Synchronous wrapper for sending shadow outcome to cloud.
    Fire and forget - doesn't block trading logic.
    
    Args:
        rl_state: RL state snapshot at rejection
        outcome: Shadow trade outcome
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(send_shadow_outcome_async(rl_state, outcome))
        else:
            loop.run_until_complete(send_shadow_outcome_async(rl_state, outcome))
    except Exception as e:
        logger.debug(f"[CLOUD RL] Error sending shadow outcome: {e}")


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
    _bot_config = load_config()
    _bot_config.validate()
    CONFIG = _bot_config.to_dict()
    
    state[symbol] = {
        # Tick data storage
        "ticks": deque(maxlen=CONFIG.get("max_tick_storage", 10000)),
        
        # Bar counters (no storage, just count)
        "bars_1min_count": 0,  # Total bars created since start
        "bars_1min": deque(maxlen=200),  # Keep last 200 for VWAP calculations
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
            "target_price": None,
            "entry_time": None,
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
            
            # ========================================================================
            # COMPLETE RL FEATURE TRACKING (matches backtest)
            # ========================================================================
            # Performance Metrics
            "mae": 0.0,  # Maximum Adverse Excursion (max drawdown)
            "mfe": 0.0,  # Maximum Favorable Excursion (max profit)
            "max_r_achieved": 0.0,  # Maximum R-multiple reached
            "min_r_achieved": 0.0,  # Minimum R-multiple reached (most negative)
            "unrealized_pnl": 0.0,  # Current unrealized P&L
            "peak_unrealized_pnl": 0.0,  # Highest unrealized P&L reached
            "peak_r_multiple": 0.0,  # Peak R-multiple
            
            # Exit Strategy State Tracking
            "breakeven_activation_bar": None,  # Bar number when breakeven activated
            "trailing_activation_bar": None,  # Bar number when trailing activated
            "bars_until_breakeven": 0,  # How many bars until breakeven activated
            "bars_until_trailing": 0,  # How many bars until trailing activated
            "time_in_breakeven_bars": 0,  # How many bars spent in breakeven mode
            "exit_param_update_count": 0,  # How many times exit params were updated
            "stop_adjustment_count": 0,  # How many times stop was adjusted
            "rejected_partial_count": 0,  # How many partial exits were rejected
            "stop_hit": False,  # Whether stop loss was hit
            
            # Decision History Tracking
            "exit_param_updates": [],  # List of {bar, vix, atr, reason, r_at_update}
            "stop_adjustments": [],  # List of {bar, old_stop, new_stop, reason, r_at_adjustment}
            "decision_history": [],  # Every bar's decision
            "unrealized_pnl_history": [],  # P&L tracking over time
            
            # Drawdown Tracking
            "profit_drawdown_from_peak": 0.0,  # $ drawdown from peak profit
            "max_drawdown_percent": 0.0,  # Max % drawdown
            "drawdown_bars": 0,  # Bars in drawdown
            
            # Volatility Tracking
            "entry_atr": None,  # ATR at entry
            "atr_samples": [],  # ATR values during trade
            "high_volatility_bars": 0,  # Count of high volatility bars
            "volatility_regime_at_entry": None,  # Regime when entered
            
            # Session/Context Tracking
            "entry_session": None,  # Asia/London/NY
            "entry_confidence": 0.0,  # Signal confidence at entry
            "duration_bars": 0,  # How many bars trade has been open
            "trade_number_in_session": 0,  # Nth trade of the session
            "wins_in_last_5_trades": 0,  # Recent performance
            "losses_in_last_5_trades": 0,  # Recent performance
            
            # Execution Quality
            "exit_slippage_ticks": 0.0,  # Slippage on exit
            "commission_cost": 0.0,  # Total commission paid
            "exit_bid_ask_spread_ticks": 0.0,  # Spread at exit
            
            # Exit Parameters Used (for learning)
            "exit_params_used": {},  # The adaptive exit params used for this trade
        },
        
        # ========================================================================
        # SHADOW TRADE TRACKING (Ghost signals for RL learning)
        # ========================================================================
        # Tracks rejected signals in real-time to see what WOULD have happened
        # Max 5 active shadows to prevent memory bloat
        "shadow_trades": [],  # List of active shadow trades being tracked
        
        # Volume history
        "volume_history": deque(maxlen=CONFIG["max_bars_storage"]),
        
        # Bot status tracking
        "bot_status_message": "Monitoring...",
        "skip_reason": "",  # Why bot is not trading (for dashboard display)
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
        
        # Update dashboard with live bid/ask prices
        if dashboard is not None:
            spread = ask_price - bid_price
            
            # Determine market condition based on spread
            if spread <= 0.50:
                condition = "NORMAL - Tight spread, good liquidity"
            elif spread <= 1.00:
                condition = "NORMAL - Good liquidity"
            elif spread <= 2.00:
                condition = "NORMAL - Acceptable spread"
            elif spread <= 5.00:
                condition = "CAUTION - Wider spread"
            else:
                condition = "VOLATILE - Wide spreads detected"
            
            # Calculate bars and current volume if symbol initialized
            bars_count = 0
            current_volume = 0
            if symbol in state:
                bars_count = state[symbol].get("bars_1min_count", 0)
                # Get current bar volume
                current_bar = state[symbol].get("current_1min_bar")
                if current_bar:
                    current_volume = current_bar.get("volume", 0)
            
            dashboard.update_symbol_data(symbol, {
                'bid': bid_price,
                'ask': ask_price,
                'bid_size': bid_size,
                'ask_size': ask_size,
                'spread': spread,
                'market_status': 'LIVE',
                'condition': condition,
                'bars_1min_count': bars_count,
                'current_volume': current_volume
            })
            
            # Refresh dashboard every 1000 ticks to reduce flicker
            dashboard_tick_counter[0] += 1
            if dashboard_tick_counter[0] >= 1000:
                dashboard.display()
                dashboard_tick_counter[0] = 0  # Reset counter
    
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
            "target_price": position["target_price"],
            "entry_time": position["entry_time"].isoformat() if position.get("entry_time") else None,
            "order_id": position.get("order_id"),
            "stop_order_id": position.get("stop_order_id"),
            "last_updated": datetime.now(pytz.UTC).isoformat(),
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
        logger.warning("  ✓ Broker confirms position - restoring state")
        
        # Restore position to state
        state[symbol]["position"]["active"] = True
        state[symbol]["position"]["side"] = saved_state["side"]
        state[symbol]["position"]["quantity"] = saved_state["quantity"]
        state[symbol]["position"]["entry_price"] = saved_state["entry_price"]
        state[symbol]["position"]["stop_price"] = saved_state["stop_price"]
        state[symbol]["position"]["target_price"] = saved_state["target_price"]
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
    
    # DEBUG: Log when we detect a minute change
    if current_bar is not None and current_bar["timestamp"] != minute_boundary:
        logger.info(f"[BAR_DEBUG] Minute changed: {current_bar['timestamp'].strftime('%H:%M:%S')} -> {minute_boundary.strftime('%H:%M:%S')}")
    
    if current_bar is None or current_bar["timestamp"] != minute_boundary:
        # Finalize previous bar if exists
        if current_bar is not None:
            # ONLY increment when closing a bar (minute boundary changed)
            state[symbol]["bars_1min"].append(current_bar)
            state[symbol]["bars_1min_count"] += 1  # Increment total bar counter
            bar_count = state[symbol]["bars_1min_count"]
            # Log at CRITICAL level so it shows on dashboard
            logger.critical(f"📊 BAR #{bar_count} CLOSED | {current_bar['timestamp'].strftime('%H:%M')} | ${current_bar['close']:.2f} | Vol: {current_bar['volume']}")
            
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
            
            # Check for exit conditions if position is active
            check_exit_conditions(symbol)
            # Check for entry signals if no position
            logger.critical(f"🔍 CHECKING FOR SIGNALS | Bar #{bar_count} | Bars available: {len(state[symbol]['bars_1min'])}")
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
    # DEBUG: Check what we're getting
    if len(state[symbol]["bars_1min"]) == 0:  # First bar only
        logger.info(f"[INJECT_BAR] First bar: H={bar.get('high', 'MISSING'):.2f} L={bar.get('low', 'MISSING'):.2f}")
    
    # Finalize any pending bar first
    if state[symbol]["current_1min_bar"] is not None:
        state[symbol]["bars_1min"].append(state[symbol]["current_1min_bar"])
        state[symbol]["current_1min_bar"] = None
    
    # Add the complete bar with proper OHLC
    state[symbol]["bars_1min"].append(bar)
    
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
            # Update all indicators after new bar is added
            update_trend_filter(symbol)
            update_rsi(symbol)
            update_macd(symbol)
            update_volume_average(symbol)
        
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


def calculate_atr(symbol: str, period: int = 14) -> float:
    """
    Calculate Average True Range (ATR) for volatility measurement.
    
    Args:
        symbol: Instrument symbol
        period: ATR period (default 14)
    
    Returns:
        ATR value in price units
    """
    bars = state[symbol]["bars_15min"]
    
    if len(bars) < 2:
        return 0.0
    
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
        return 0.0
    
    # Calculate ATR (simple moving average of TR)
    if len(true_ranges) >= period:
        atr = sum(true_ranges[-period:]) / period
    else:
        atr = sum(true_ranges) / len(true_ranges)
    
    return atr


def update_rsi(symbol: str) -> None:
    """
    Update RSI indicator for the symbol.
    
    Args:
        symbol: Instrument symbol
    """
    bars = state[symbol]["bars_15min"]
    rsi_period = CONFIG.get("rsi_period", 10)  # Fast RSI - feature for neural networks
    
    if len(bars) < rsi_period + 1:
        logger.debug(f"Not enough bars for RSI: {len(bars)}/{rsi_period + 1}")
        return
    
    closes = [bar["close"] for bar in bars]
    rsi = calculate_rsi(closes, rsi_period)
    
    if rsi is not None:
        state[symbol]["rsi"] = rsi
        logger.debug(f"RSI: {rsi:.2f}")


def update_macd(symbol: str) -> None:
    """
    Update MACD indicator for the symbol.
    
    Args:
        symbol: Instrument symbol
    """
    bars = state[symbol]["bars_15min"]
    
    # Get MACD parameters from config
    fast_period = CONFIG.get("macd_fast", 12)
    slow_period = CONFIG.get("macd_slow", 26)
    signal_period = CONFIG.get("macd_signal", 9)
    
    if len(bars) < slow_period + signal_period:
        logger.debug(f"Not enough bars for MACD: {len(bars)}/{slow_period + signal_period}")
        return
    
    closes = [bar["close"] for bar in bars]
    macd_data = calculate_macd(closes, fast_period, slow_period, signal_period)
    
    if macd_data is not None:
        state[symbol]["macd"] = macd_data
        logger.debug(f"MACD: {macd_data['macd']:.2f}, Signal: {macd_data['signal']:.2f}, "
                    f"Histogram: {macd_data['histogram']:.2f}")


def update_volume_average(symbol: str) -> None:
    """
    Update average volume for spike detection.
    Also tracks recent volume history for surge detection.
    
    Args:
        symbol: Instrument symbol
    """
    bars = state[symbol]["bars_15min"]
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
    
    # Calculate bands using neural network baseline standard deviation multipliers
    band_1_mult = CONFIG.get("vwap_std_dev_1", 2.5)  # Warning zone
    band_2_mult = CONFIG.get("vwap_std_dev_2", 2.1)  # Entry zone - baseline for signal detection
    band_3_mult = CONFIG.get("vwap_std_dev_3", 3.7)  # Exit/stop zone - baseline for exits
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
    
    if trading_state == "event_block":
        # FOMC/NFP/CPI event blocking (if enabled in config)
        halt_reason = bot_status.get("halt_reason", "Economic event")
        logger.debug(f"Event block active: {halt_reason}")
        return False, f"Economic event block: {halt_reason}"
    
    if trading_state == "flatten_mode":
        logger.debug(f"Flatten mode active (20:45-21:00 UTC), no new entries")
        return False, f"Flatten mode - close positions only"
    
    # Trading state is "entry_window" - market is open, proceed with checks
    
    # Daily entry cutoff - no new positions after 20:00 UTC (can still hold existing)
    if current_time.time() >= CONFIG["daily_entry_cutoff"]:
        logger.debug(f"After 20:00 UTC entry cutoff, no new trades (can hold existing)")
        return False, "After 20:00 UTC - no new entries"
    
    # Friday restriction - close before weekend (use current time, not bar time)
    if current_time.weekday() == 4 and current_time.time() >= CONFIG["friday_entry_cutoff"]:
        log_time_based_action(
            "friday_entry_blocked",
            f"Friday after {CONFIG['friday_entry_cutoff']} UTC, no new trades to avoid weekend gap risk",
            {"day": "Friday", "time": bar_time.strftime('%H:%M:%S')}
        )
        logger.info(f"Friday after {CONFIG['friday_entry_cutoff']} UTC - no new trades (weekend gap risk)")
        return False, "Friday entry cutoff"
    
    # Check if already have position
    if state[symbol]["position"]["active"]:
        logger.debug("Position already active, skipping signal generation")
        return False, "Position active"
    
    # Check daily trade limit
    if state[symbol]["daily_trade_count"] >= CONFIG["max_trades_per_day"]:
        logger.warning(f"Daily trade limit reached ({CONFIG['max_trades_per_day']}), stopping for the day")
        
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
                    error_message=f"💰 Daily Loss Limit Reached: ${state[symbol]['daily_pnl']:.2f} / -${CONFIG['daily_loss_limit']:.2f}. Bot stopped trading for today. Will auto-resume tomorrow.",
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
    
    # Check RSI (neural network baseline - selective entry thresholds)
    use_rsi_filter = CONFIG.get("use_rsi_filter", True)
    rsi_oversold = CONFIG.get("rsi_oversold", 35.0)  # Baseline for LONG signals
    rsi_overbought = CONFIG.get("rsi_overbought", 65.0)  # Baseline for SHORT signals
    
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
    
    # FILTER 2: RSI - extreme oversold (neural network baseline)
    use_rsi = CONFIG.get("use_rsi_filter", True)
    rsi_oversold = CONFIG.get("rsi_oversold", 35.0)  # Baseline - neural network validates
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
        avg_volume = state[symbol]["avg_volume"]
        if avg_volume is not None and avg_volume > 0:
            current_volume = current_bar["volume"]
            if current_volume < avg_volume * volume_mult:
                logger.debug(f"Long rejected - no volume spike: {current_volume} < {avg_volume * volume_mult:.0f}")
                return False
            logger.debug(f"Volume spike: {current_volume} >= {avg_volume * volume_mult:.0f} ")
    
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
    
    # FILTER 2: RSI - extreme overbought (neural network baseline)
    use_rsi = CONFIG.get("use_rsi_filter", True)
    rsi_overbought = CONFIG.get("rsi_overbought", 65.0)  # Baseline - neural network validates
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
        avg_volume = state[symbol]["avg_volume"]
        if avg_volume is not None and avg_volume > 0:
            current_volume = current_bar["volume"]
            if current_volume < avg_volume * volume_mult:
                logger.debug(f"Short rejected - no volume spike: {current_volume} < {avg_volume * volume_mult:.0f}")
                return False
            logger.debug(f"Volume spike: {current_volume} >= {avg_volume * volume_mult:.0f} ")
    
    logger.info(f" SHORT SIGNAL: Price reversal at {current_bar['close']:.2f} (entry zone: {vwap_bands['upper_2']:.2f})")
    return True


def capture_rl_state(symbol: str, side: str, current_price: float) -> Dict[str, Any]:
    """
    Capture market state for RL decision making.
    
    Args:
        symbol: Instrument symbol
        side: 'long' or 'short'
        current_price: Current market price
    
    Returns:
        Dictionary with state features for RL brain
    """
    vwap = state[symbol].get("vwap", current_price)
    vwap_bands = state[symbol].get("vwap_bands", {})
    rsi = state[symbol].get("rsi", 50)
    
    # Calculate VWAP standard deviation
    vwap_std = 0
    if vwap_bands:
        # Calculate std from bands
        upper = vwap_bands.get("upper_1", vwap)
        vwap_std = abs(upper - vwap) if upper != vwap else 0
    
    # Calculate VWAP distance in standard deviations
    vwap_distance = abs(current_price - vwap) / vwap_std if vwap_std > 0 else 0
    
    # Get ATR
    atr = calculate_atr(symbol, CONFIG.get("atr_period", 14))
    if atr is None:
        atr = 0
    
    # Calculate volume ratio
    avg_volume = state[symbol].get("avg_volume")
    current_bar = state[symbol]["bars_1min"][-1]
    volume_ratio = current_bar["volume"] / avg_volume if avg_volume and avg_volume > 0 else 1.0
    
    # Get current time
    current_time = get_current_time()
    hour = current_time.hour
    day_of_week = current_time.weekday()
    
    # Calculate recent P&L from trade history
    recent_pnl = 0
    if "trade_history" in state[symbol] and len(state[symbol]["trade_history"]) > 0:
        recent_trades = state[symbol]["trade_history"][-3:]  # Last 3 trades
        recent_pnl = sum(t.get("pnl", 0) for t in recent_trades)
    
    # Calculate win/loss streak
    streak = 0
    if "trade_history" in state[symbol] and len(state[symbol]["trade_history"]) > 0:
        for trade in reversed(state[symbol]["trade_history"]):
            pnl = trade.get("pnl", 0)
            if pnl > 0:
                streak += 1
            elif pnl < 0:
                streak -= 1
            else:
                break  # Stop at breakeven
    
    # ========================================================================
    # REWARD SHAPING FEATURES (matches backtest for RL training)
    # ========================================================================
    
    # 1. Volatility Trend - % change in volatility (recent 10 vs previous 10)
    bars_1min = state[symbol]["bars_1min"]
    if len(bars_1min) >= 20:
        # Calculate recent volatility (last 10 bars)
        recent_prices = [b["close"] for b in list(bars_1min)[-10:]]
        recent_vol = statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0.0
        
        # Calculate previous volatility (bars 11-20)
        previous_prices = [b["close"] for b in list(bars_1min)[-20:-10]]
        previous_vol = statistics.stdev(previous_prices) if len(previous_prices) > 1 else 0.0
        
        # % change in volatility
        volatility_trend = ((recent_vol - previous_vol) / previous_vol * 100) if previous_vol > 0 else 0.0
    else:
        volatility_trend = 0.0
    
    # 2. Drawdown % at Entry - Current account drawdown from peak
    starting_balance = CONFIG.get("account_size", 50000)
    current_balance = starting_balance + state[symbol].get("daily_pnl", 0.0)
    
    # Track peak balance (initialize if not exists)
    if "peak_balance" not in state[symbol]:
        state[symbol]["peak_balance"] = starting_balance
    
    # Update peak balance
    if current_balance > state[symbol]["peak_balance"]:
        state[symbol]["peak_balance"] = current_balance
    
    # Calculate drawdown %
    peak_balance = state[symbol]["peak_balance"]
    drawdown_pct_at_entry = ((current_balance - peak_balance) / peak_balance * 100) if peak_balance > 0 else 0.0
    
    # 3. Recent Volatility (20-bar) - Rolling standard deviation
    if len(bars_1min) >= 20:
        recent_20_prices = [b["close"] for b in list(bars_1min)[-20:]]
        recent_volatility_20bar = statistics.stdev(recent_20_prices)
    else:
        recent_volatility_20bar = 0.0
    
    rl_state = {
        "rsi": rsi if rsi is not None else 50,
        "vwap_distance": vwap_distance,
        "atr": atr,
        "volume_ratio": volume_ratio,
        "hour": hour,
        "day_of_week": day_of_week,
        "recent_pnl": recent_pnl,
        "streak": streak,
        "side": side,
        "price": current_price,
        "vwap": vwap,  # ADDED: need for cloud RL API
        "vix": 15.0,   # ADDED: default VIX (could fetch real VIX later)
        # Reward shaping features (for RL training)
        "volatility_trend": volatility_trend,
        "drawdown_pct_at_entry": drawdown_pct_at_entry,
        "recent_volatility_20bar": recent_volatility_20bar
    }
    
    return rl_state


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
        # Store reason for dashboard display
        state[symbol]["skip_reason"] = reason
        if dashboard:
            dashboard.update_symbol_data(symbol, {
                "skip_reason": reason,
                "status": "Waiting - " + reason
            })
            dashboard.display()
        return
    
    # Get the latest bar
    if len(state[symbol]["bars_1min"]) == 0:
        logger.info(f"[SIGNAL CHECK] No 1-min bars yet")
        state[symbol]["skip_reason"] = "No bars yet"
        if dashboard:
            dashboard.update_symbol_data(symbol, {
                "skip_reason": "Building bars...",
                "status": "Initializing..."
            })
            dashboard.display()
        return
    
    latest_bar = state[symbol]["bars_1min"][-1]
    bar_time = latest_bar["timestamp"]
    
    # Validate signal requirements
    is_valid, reason = validate_signal_requirements(symbol, bar_time)
    if not is_valid:
        logger.info(f"[SIGNAL CHECK] Validation failed: {reason} at {bar_time}")
        # Store reason for dashboard display
        state[symbol]["skip_reason"] = reason
        if dashboard:
            dashboard.update_symbol_data(symbol, {
                "skip_reason": reason,
                "status": "Waiting - " + reason
            })
            dashboard.display()
        return
    
    # Clear skip reason - we're actively looking for signals
    state[symbol]["skip_reason"] = ""
    
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
        logger.critical(f"🎯 LONG SIGNAL DETECTED! Prev Low: ${prev_bar['low']:.2f} | Lower Band: ${vwap_bands['lower_2']:.2f}")
        
        # === SIGNAL RL FEATURE 1: SPREAD FILTER ===
        # Check if spread is too wide (high slippage risk)
        if rl_brain is not None:
            current_spread_ticks = 1.0  # Default
            if bid_ask_manager is not None:
                quote = bid_ask_manager.get_current_quote(symbol)
                if quote and quote.bid > 0 and quote.ask > 0:
                    tick_size = CONFIG.get("tick_size", 0.25)
                    current_spread_ticks = (quote.ask - quote.bid) / tick_size
            
            spread_ok, spread_reason = rl_brain.check_spread_acceptable(current_spread_ticks)
            if not spread_ok:
                logger.info(f"❌ SPREAD FILTER REJECTED LONG: {spread_reason}")
                state[symbol]["last_rejected_signal"] = {
                    "time": get_current_time(),
                    "side": "long",
                    "price": current_bar["close"],
                    "reason": spread_reason
                }
                # Track as skipped signal for learning
                rl_state = capture_rl_state(symbol, "long", current_bar["close"])
                rl_brain.track_skipped_signal(rl_state, 0.0, spread_reason)
                
                if dashboard:
                    sig_time = get_current_time().strftime("%H:%M")
                    dashboard.update_symbol_data(symbol, {
                        "last_rejected_signal": f"{sig_time} LONG @ ${current_bar['close']:.2f} ({spread_reason})",
                        "status": "Signal rejected - spread too wide"
                    })
                    dashboard.display()
                return
        
        # === SIGNAL RL FEATURE 2: LIQUIDITY FILTER ===
        # Check if volume is too low (fill risk)
        if rl_brain is not None:
            avg_volume = state[symbol].get("avg_volume", 1)
            current_volume = current_bar.get("volume", 0)
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            liquidity_ok, liquidity_reason = rl_brain.check_liquidity_acceptable(volume_ratio)
            if not liquidity_ok:
                logger.info(f"❌ LIQUIDITY FILTER REJECTED LONG: {liquidity_reason}")
                state[symbol]["last_rejected_signal"] = {
                    "time": get_current_time(),
                    "side": "long",
                    "price": current_bar["close"],
                    "reason": liquidity_reason
                }
                # Track as skipped signal for learning
                rl_state = capture_rl_state(symbol, "long", current_bar["close"])
                rl_brain.track_skipped_signal(rl_state, 0.0, liquidity_reason)
                
                if dashboard:
                    sig_time = get_current_time().strftime("%H:%M")
                    dashboard.update_symbol_data(symbol, {
                        "last_rejected_signal": f"{sig_time} LONG @ ${current_bar['close']:.2f} ({liquidity_reason})",
                        "status": "Signal rejected - low liquidity"
                    })
                    dashboard.display()
                return
        
        # ADAPTIVE LEARNING - Check if we should skip this trade based on learned patterns
        should_skip, skip_reason = adaptive_manager.should_skip_trade(symbol, state[symbol]) if adaptive_manager else (False, "")
        if should_skip:
            logger.info(f"❌ ADAPTIVE FILTER REJECTED LONG: {skip_reason}")
            state[symbol]["last_rejected_signal"] = {
                "time": get_current_time(),
                "side": "long",
                "price": current_bar["close"],
                "reason": skip_reason
            }
            # === SIGNAL RL FEATURE 3: ADVERSE SELECTION TRACKING ===
            # Track rejected signals to learn from adverse selection
            if rl_brain is not None:
                rl_state = capture_rl_state(symbol, "long", current_bar["close"])
                rl_brain.track_skipped_signal(rl_state, 0.0, skip_reason)
            
            if dashboard:
                sig_time = get_current_time().strftime("%H:%M")
                dashboard.update_symbol_data(symbol, {
                    "last_rejected_signal": f"{sig_time} LONG @ ${current_bar['close']:.2f} (Adaptive filter)",
                    "status": "Signal rejected - learned filter"
                })
                dashboard.display()
            return
        
        # REINFORCEMENT LEARNING - Get confidence from cloud API (shared learning pool)
        # Capture market state
        rl_state = capture_rl_state(symbol, "long", current_bar["close"])
        
        # Calculate current spread in ticks for RL spread filter
        current_spread_ticks = 1.0  # Default
        if bid_ask_manager is not None:
            quote = bid_ask_manager.get_current_quote(symbol)
            if quote and quote.bid > 0 and quote.ask > 0:
                tick_size = CONFIG.get("tick_size", 0.25)
                current_spread_ticks = (quote.ask - quote.bid) / tick_size
        
        # Ask cloud RL API for decision (or local RL as fallback)
        take_signal, confidence, size_mult, reason = get_ml_confidence(rl_state, "long", current_spread_ticks)
        
        logger.critical(f"🤖 ML DECISION: {'✅ TAKE' if take_signal else '❌ REJECT'} | Confidence: {confidence:.1%} | Reason: {reason}")
        
        # ENFORCE GUI CONFIDENCE THRESHOLD - User's setting is the minimum
        user_threshold = CONFIG.get("rl_confidence_threshold", 0.5)
        if take_signal and confidence < user_threshold:
            logger.info(f"❌ GUI THRESHOLD REJECTED LONG signal: confidence {confidence:.1%} below user threshold {user_threshold:.1%}")
            logger.info(f"   User configured minimum: {user_threshold:.1%}")
            state[symbol]["last_rejected_signal"] = {
                "time": get_current_time(),
                "state": rl_state,
                "side": "long",
                "confidence": confidence,
                "reason": f"Below user threshold: {confidence:.1%} < {user_threshold:.1%}"
            }
            
            # Update dashboard
            if dashboard:
                sig_time = get_current_time().strftime("%H:%M")
                conf_str = f"{int(confidence * 100)}%"
                dashboard.update_symbol_data(symbol, {
                    "last_rejected_signal": f"{sig_time} LONG @ ${current_bar['close']:.2f} (User Min: {int(user_threshold * 100)}% > {conf_str})",
                    "status": "Below user confidence threshold"
                })
                dashboard.display()
            
            return
        
        # Check if in recovery mode and need higher confidence
        if take_signal and bot_status.get("recovery_confidence_threshold"):
            recovery_threshold = bot_status["recovery_confidence_threshold"]
            if confidence < recovery_threshold:
                logger.info(f"❌ RECOVERY MODE REJECTED LONG signal: confidence {confidence:.1%} below recovery threshold {recovery_threshold:.1%}")
                logger.info(f"   Bot is in recovery mode - only taking high-confidence signals")
                state[symbol]["last_rejected_signal"] = {
                    "time": get_current_time(),
                    "state": rl_state,
                    "side": "long",
                    "confidence": confidence,
                    "reason": f"Recovery mode: {confidence:.1%} < {recovery_threshold:.1%}"
                }
                
                # Update dashboard to show rejected signal
                if dashboard:
                    sig_time = get_current_time().strftime("%H:%M")
                    conf_str = f"{int(confidence * 100)}%"
                    dashboard.update_symbol_data(symbol, {
                        "last_rejected_signal": f"{sig_time} LONG @ ${current_bar['close']:.2f} (Recovery: {conf_str} < {int(recovery_threshold * 100)}%)",
                        "status": "Recovery mode - rejected"
                    })
                    dashboard.display()
                
                return
        
        if not take_signal:
            logger.info(f"❌ RL REJECTED LONG signal: {reason} (confidence: {confidence:.1%})")
            logger.info(f"   RSI: {rl_state['rsi']:.1f}, VWAP dist: {rl_state['vwap_distance']:.2f}, "
                      f"Vol ratio: {rl_state['volume_ratio']:.2f}x")
            # Store the rejected signal state for potential future learning
            state[symbol]["last_rejected_signal"] = {
                "time": get_current_time(),
                "state": rl_state,
                "side": "long",
                "confidence": confidence,
                "reason": reason
            }
            
            # Update dashboard to show rejected signal
            if dashboard:
                sig_time = get_current_time().strftime("%H:%M")
                conf_str = f"{int(confidence * 100)}%"
                dashboard.update_symbol_data(symbol, {
                    "last_rejected_signal": f"{sig_time} LONG @ ${current_bar['close']:.2f} (Conf: {conf_str})",
                    "status": "Signal rejected - low confidence"
                })
                dashboard.display()
            
            return
        
        # RL approved - adjust position size based on confidence and learned size multiplier
        logger.info(f"✅ RL APPROVED LONG signal: {reason} (confidence: {confidence:.1%}, size: {size_mult:.2f}x)")
        logger.info(f"   RSI: {rl_state['rsi']:.1f}, VWAP dist: {rl_state['vwap_distance']:.2f}, "
                  f"Vol ratio: {rl_state['volume_ratio']:.2f}x, Streak: {rl_state['streak']:+d}")
        
        # Store the state for outcome recording after trade
        state[symbol]["entry_rl_state"] = rl_state
        state[symbol]["entry_rl_confidence"] = confidence
        state[symbol]["entry_rl_size_mult"] = size_mult  # Store learned size multiplier
        
        # Store signal info for dashboard display
        state[symbol]["last_signal_type"] = f"LONG @ ${current_bar['close']:.2f}"
        state[symbol]["last_signal_timestamp"] = get_current_time().timestamp()
        state[symbol]["last_signal_confidence"] = confidence
        state[symbol]["last_signal_approved"] = True  # Mark as approved
        state[symbol]["bot_status_message"] = "Signal detected!"
        
        # Update dashboard immediately
        if dashboard:
            update_dashboard_for_symbol(symbol)
            dashboard.display()
        
        execute_entry(symbol, "long", current_bar["close"])
        return
    
    # Check for short signal
    if check_short_signal_conditions(symbol, prev_bar, current_bar):
        logger.critical(f"🎯 SHORT SIGNAL DETECTED! Prev High: ${prev_bar['high']:.2f} | Upper Band: ${vwap_bands['upper_2']:.2f}")
        
        # === SIGNAL RL FEATURE 1: SPREAD FILTER ===
        # Check if spread is too wide (high slippage risk)
        if rl_brain is not None:
            current_spread_ticks = 1.0  # Default
            if bid_ask_manager is not None:
                quote = bid_ask_manager.get_current_quote(symbol)
                if quote and quote.bid > 0 and quote.ask > 0:
                    tick_size = CONFIG.get("tick_size", 0.25)
                    current_spread_ticks = (quote.ask - quote.bid) / tick_size
            
            spread_ok, spread_reason = rl_brain.check_spread_acceptable(current_spread_ticks)
            if not spread_ok:
                logger.info(f"❌ SPREAD FILTER REJECTED SHORT: {spread_reason}")
                state[symbol]["last_rejected_signal"] = {
                    "time": get_current_time(),
                    "side": "short",
                    "price": current_bar["close"],
                    "reason": spread_reason
                }
                # Track as skipped signal for learning
                rl_state = capture_rl_state(symbol, "short", current_bar["close"])
                rl_brain.track_skipped_signal(rl_state, 0.0, spread_reason)
                
                if dashboard:
                    sig_time = get_current_time().strftime("%H:%M")
                    dashboard.update_symbol_data(symbol, {
                        "last_rejected_signal": f"{sig_time} SHORT @ ${current_bar['close']:.2f} ({spread_reason})",
                        "status": "Signal rejected - spread too wide"
                    })
                    dashboard.display()
                return
        
        # === SIGNAL RL FEATURE 2: LIQUIDITY FILTER ===
        # Check if volume is too low (fill risk)
        if rl_brain is not None:
            avg_volume = state[symbol].get("avg_volume", 1)
            current_volume = current_bar.get("volume", 0)
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            liquidity_ok, liquidity_reason = rl_brain.check_liquidity_acceptable(volume_ratio)
            if not liquidity_ok:
                logger.info(f"❌ LIQUIDITY FILTER REJECTED SHORT: {liquidity_reason}")
                state[symbol]["last_rejected_signal"] = {
                    "time": get_current_time(),
                    "side": "short",
                    "price": current_bar["close"],
                    "reason": liquidity_reason
                }
                # Track as skipped signal for learning
                rl_state = capture_rl_state(symbol, "short", current_bar["close"])
                rl_brain.track_skipped_signal(rl_state, 0.0, liquidity_reason)
                
                if dashboard:
                    sig_time = get_current_time().strftime("%H:%M")
                    dashboard.update_symbol_data(symbol, {
                        "last_rejected_signal": f"{sig_time} SHORT @ ${current_bar['close']:.2f} ({liquidity_reason})",
                        "status": "Signal rejected - low liquidity"
                    })
                    dashboard.display()
                return
        
        # ADAPTIVE LEARNING - Check if we should skip this trade based on learned patterns
        should_skip, skip_reason = adaptive_manager.should_skip_trade(symbol, state[symbol]) if adaptive_manager else (False, "")
        if should_skip:
            logger.info(f"❌ ADAPTIVE FILTER REJECTED SHORT: {skip_reason}")
            state[symbol]["last_rejected_signal"] = {
                "time": get_current_time(),
                "side": "short",
                "price": current_bar["close"],
                "reason": skip_reason
            }
            # === SIGNAL RL FEATURE 3: ADVERSE SELECTION TRACKING ===
            # Track rejected signals to learn from adverse selection
            if rl_brain is not None:
                rl_state = capture_rl_state(symbol, "short", current_bar["close"])
                rl_brain.track_skipped_signal(rl_state, 0.0, skip_reason)
            
            if dashboard:
                sig_time = get_current_time().strftime("%H:%M")
                dashboard.update_symbol_data(symbol, {
                    "last_rejected_signal": f"{sig_time} SHORT @ ${current_bar['close']:.2f} (Adaptive filter)",
                    "status": "Signal rejected - learned filter"
                })
                dashboard.display()
            return
        
        # REINFORCEMENT LEARNING - Get confidence from cloud API (shared learning pool)
        # Capture market state
        rl_state = capture_rl_state(symbol, "short", current_bar["close"])
        
        # Calculate current spread in ticks for RL spread filter
        current_spread_ticks = 1.0  # Default
        if bid_ask_manager is not None:
            quote = bid_ask_manager.get_current_quote(symbol)
            if quote and quote.bid > 0 and quote.ask > 0:
                tick_size = CONFIG.get("tick_size", 0.25)
                current_spread_ticks = (quote.ask - quote.bid) / tick_size
        
        # Ask cloud RL API for decision (or local RL as fallback)
        take_signal, confidence, size_mult, reason = get_ml_confidence(rl_state, "short", current_spread_ticks)
        
        logger.critical(f"🤖 ML DECISION: {'✅ TAKE' if take_signal else '❌ REJECT'} | Confidence: {confidence:.1%} | Reason: {reason}")
        
        # ENFORCE GUI CONFIDENCE THRESHOLD - User's setting is the minimum
        user_threshold = CONFIG.get("rl_confidence_threshold", 0.5)
        if take_signal and confidence < user_threshold:
            logger.info(f"❌ GUI THRESHOLD REJECTED SHORT signal: confidence {confidence:.1%} below user threshold {user_threshold:.1%}")
            logger.info(f"   User configured minimum: {user_threshold:.1%}")
            state[symbol]["last_rejected_signal"] = {
                "time": get_current_time(),
                "state": rl_state,
                "side": "short",
                "confidence": confidence,
                "reason": f"Below user threshold: {confidence:.1%} < {user_threshold:.1%}"
            }
            
            # Update dashboard
            if dashboard:
                sig_time = get_current_time().strftime("%H:%M")
                conf_str = f"{int(confidence * 100)}%"
                dashboard.update_symbol_data(symbol, {
                    "last_rejected_signal": f"{sig_time} SHORT @ ${current_bar['close']:.2f} (User Min: {int(user_threshold * 100)}% > {conf_str})",
                    "status": "Below user confidence threshold"
                })
                dashboard.display()
            
            return
        
        # Check if in recovery mode and need higher confidence
        if take_signal and bot_status.get("recovery_confidence_threshold"):
            recovery_threshold = bot_status["recovery_confidence_threshold"]
            if confidence < recovery_threshold:
                logger.info(f"❌ RECOVERY MODE REJECTED SHORT signal: confidence {confidence:.1%} below recovery threshold {recovery_threshold:.1%}")
                logger.info(f"   Bot is in recovery mode - only taking high-confidence signals")
                state[symbol]["last_rejected_signal"] = {
                    "time": get_current_time(),
                    "state": rl_state,
                    "side": "short",
                    "confidence": confidence,
                    "reason": f"Recovery mode: {confidence:.1%} < {recovery_threshold:.1%}"
                }
                
                # Update dashboard to show rejected signal
                if dashboard:
                    sig_time = get_current_time().strftime("%H:%M")
                    conf_str = f"{int(confidence * 100)}%"
                    dashboard.update_symbol_data(symbol, {
                        "last_rejected_signal": f"{sig_time} SHORT @ ${current_bar['close']:.2f} (Recovery: {conf_str} < {int(recovery_threshold * 100)}%)",
                        "status": "Recovery mode - rejected"
                    })
                    dashboard.display()
                
                return
        
        if not take_signal:
            logger.info(f"❌ RL REJECTED SHORT signal: {reason} (confidence: {confidence:.1%})")
            logger.info(f"   RSI: {rl_state['rsi']:.1f}, VWAP dist: {rl_state['vwap_distance']:.2f}, "
                      f"Vol ratio: {rl_state['volume_ratio']:.2f}x")
            # Store the rejected signal state for potential future learning
            state[symbol]["last_rejected_signal"] = {
                "time": get_current_time(),
                "state": rl_state,
                "side": "short",
                "confidence": confidence,
                "reason": reason
            }
            
            # Update dashboard to show rejected signal
            if dashboard:
                sig_time = get_current_time().strftime("%H:%M")
                conf_str = f"{int(confidence * 100)}%"
                dashboard.update_symbol_data(symbol, {
                    "last_rejected_signal": f"{sig_time} SHORT @ ${current_bar['close']:.2f} (Conf: {conf_str})",
                    "status": "Signal rejected - low confidence"
                })
                dashboard.display()
            
            return
        
        # RL approved - adjust position size based on confidence and learned size multiplier
        logger.info(f"✅ RL APPROVED SHORT signal: {reason} (confidence: {confidence:.1%}, size: {size_mult:.2f}x)")
        logger.info(f"   RSI: {rl_state['rsi']:.1f}, VWAP dist: {rl_state['vwap_distance']:.2f}, "
                  f"Vol ratio: {rl_state['volume_ratio']:.2f}x, Streak: {rl_state['streak']:+d}")
        
        # Store the state for outcome recording after trade
        state[symbol]["entry_rl_state"] = rl_state
        state[symbol]["entry_rl_confidence"] = confidence
        state[symbol]["entry_rl_size_mult"] = size_mult  # Store learned size multiplier
        
        # Store signal info for dashboard display
        state[symbol]["last_signal_type"] = f"SHORT @ ${current_bar['close']:.2f}"
        state[symbol]["last_signal_timestamp"] = get_current_time().timestamp()
        state[symbol]["last_signal_confidence"] = confidence
        state[symbol]["last_signal_approved"] = True  # Mark as approved
        state[symbol]["bot_status_message"] = "Signal detected!"
        
        # Update dashboard immediately
        if dashboard:
            update_dashboard_for_symbol(symbol)
            dashboard.display()
        
        execute_entry(symbol, "short", current_bar["close"])
        return


# ============================================================================
# PHASE EIGHT: Position Sizing
# ============================================================================

def calculate_position_size(symbol: str, side: str, entry_price: float, rl_confidence: Optional[float] = None, size_mult: Optional[float] = None) -> Tuple[int, float, float]:
    """
    Calculate position size based on risk management rules.
    
    USER SETS MAX LIMIT → RL Confidence dynamically chooses contracts within that limit!
    - User configures max_contracts (e.g., 3 contracts)
    - RL confidence scales: LOW = 1 contract, MEDIUM = 2 contracts, HIGH = 3 contracts
    - User with max_contracts=10 gets: LOW = 3, MEDIUM = 6, HIGH = 10
    
    Args:
        symbol: Instrument symbol
        side: 'long' or 'short'
        entry_price: Expected entry price
        rl_confidence: Optional RL confidence (0-1) - legacy parameter
        size_mult: Optional size multiplier from RL system (0.25-2.0x)
    
    Returns:
        Tuple of (contracts, stop_price, target_price)
    """
    # Get account equity
    equity = get_account_equity()
    
    # Use fixed contract sizing based on max_contracts
    # No risk percentage calculation - just use max contracts
    logger.info(f"Account equity: ${equity:.2f}, Using fixed contract sizing")
    
    # CRITICAL: If confidence is below threshold, return 0 contracts (reject trade)
    if rl_confidence is not None:
        user_threshold = CONFIG.get("rl_confidence_threshold", 0.5)
        if rl_confidence < user_threshold:
            logger.info(f"[POSITION SIZING] Confidence {rl_confidence:.1%} below threshold {user_threshold:.1%} - returning 0 contracts")
            return 0, 0.0, None  # 0 contracts = no trade
    
    # Determine stop price
    vwap_bands = state[symbol]["vwap_bands"]
    vwap = state[symbol]["vwap"]
    tick_size = CONFIG["tick_size"]
    
    # Check if using ATR-based stops
    if CONFIG.get("use_atr_stops", False):
        # Calculate ATR
        atr = calculate_atr(symbol, CONFIG.get("atr_period", 14))
        
        if atr > 0:
            # LEARNED STOP MULTIPLIER: Get from adaptive exit manager based on regime
            global adaptive_manager
            stop_multiplier = 3.6  # Fallback default
            
            if adaptive_manager is not None:
                try:
                    # Detect current regime
                    from adaptive_exits import detect_market_regime
                    bars = state[symbol]["bars_1min"]
                    atr_value = calculate_atr(symbol, CONFIG.get("atr_period", 14))
                    regime = detect_market_regime(bars, atr_value if atr_value else 2.0)
                    
                    # Get learned stop multiplier for this regime
                    stop_multiplier = adaptive_manager.get_stop_multiplier(regime)
                    logger.info(f"[EXIT RL] Using learned stop: {stop_multiplier:.1f}x ATR for {regime}")
                except Exception as e:
                    logger.warning(f"[EXIT RL] Failed to get learned stop, using default 3.6x: {e}")
                    stop_multiplier = 3.6
            else:
                logger.info(f"[EXIT RL] No adaptive manager, using default stop: 3.6x ATR")
            
            # NO HARDCODED TARGET - Learned partials control ALL profit exits
            if side == "long":
                stop_price = entry_price - (atr * stop_multiplier)
                target_price = None  # No fixed target - partials handle exits
            else:  # short
                stop_price = entry_price + (atr * stop_multiplier)
                target_price = None  # No fixed target - partials handle exits
            
            stop_price = round_to_tick(stop_price)
            # target_price = None (no rounding needed)
        else:
            # Fallback to fixed stops if ATR can't be calculated
            max_stop_ticks = 11
            if side == "long":
                stop_price = entry_price - (max_stop_ticks * tick_size)
                target_price = None  # No fixed target - learned partials handle exits
            else:
                stop_price = entry_price + (max_stop_ticks * tick_size)
                target_price = None  # No fixed target - learned partials handle exits
            stop_price = round_to_tick(stop_price)
            # target_price = None (no rounding needed)
    else:
        # Use fixed stops (original logic)
        max_stop_ticks = 11  # Optimized to 11 ticks
        
        if side == "long":
            # Stop 11 ticks below entry (or at lower band 3, whichever is tighter)
            band_stop = vwap_bands["lower_3"] - (2 * tick_size)  # 2 tick buffer
            tight_stop = entry_price - (max_stop_ticks * tick_size)
            stop_price = max(tight_stop, band_stop)  # Use tighter of the two
            # No fixed target - learned partials handle exits
            target_price = None
        else:  # short
            # Stop 11 ticks above entry (or at upper band 3, whichever is tighter)
            band_stop = vwap_bands["upper_3"] + (2 * tick_size)  # 2 tick buffer
            tight_stop = entry_price + (max_stop_ticks * tick_size)
            stop_price = min(tight_stop, band_stop)  # Use tighter of the two
            # No fixed target - learned partials handle exits
            target_price = None
        
        stop_price = round_to_tick(stop_price)
        # target_price = None (no rounding needed)
    
    # Calculate stop distance in ticks
    stop_distance = abs(entry_price - stop_price)
    ticks_at_risk = stop_distance / tick_size
    
    # Calculate risk per contract (for logging purposes only)
    tick_value = CONFIG["tick_value"]
    risk_per_contract = ticks_at_risk * tick_value
    
    # Use fixed max_contracts from config (no risk-based sizing)
    contracts = CONFIG["max_contracts"]
    
    # Get user's max contracts limit - THIS IS YOUR HARD CEILING
    user_max_contracts = CONFIG["max_contracts"]
    
    # Two modes:
    # FIXED MODE: Use exact max_contracts, no adjustments
    # DYNAMIC MODE: Apply confidence/streak/session multipliers WITHIN your max
    dynamic_contracts_enabled = CONFIG.get("dynamic_contracts", False)
    
    # Use size_mult from Signal RL if provided (preferred), otherwise fall back to legacy method
    if size_mult is not None and dynamic_contracts_enabled:
        # NEW: Use size multiplier directly from Signal RL (includes confidence, streak, volatility, time)
        # CRITICAL: RL scales WITHIN user's max - never exceeds it
        # Size mult 0.25-2.0 but CAPPED at user_max_contracts
        # 
        # Examples with user max = 3:
        #   size=0.5x  -> 2 contracts (0.5 * 3 = 1.5 → 2)
        #   size=1.0x  -> 3 contracts (1.0 * 3 = 3)
        #   size=2.0x  -> 3 contracts (2.0 * 3 = 6 → CAPPED at 3)
        # 
        # Examples with user max = 10:
        #   size=0.5x  -> 5 contracts
        #   size=1.0x  -> 10 contracts
        #   size=2.0x  -> 10 contracts (CAPPED, won't go to 20)
        rl_scaled_max = max(1, int(round(user_max_contracts * size_mult)))
        
        # HARD CAP: Never exceed user's max_contracts setting
        rl_scaled_max = min(rl_scaled_max, user_max_contracts)
        
        # Use MINIMUM of risk-based calculation and RL-scaled max
        # This ensures we respect BOTH the risk management AND user's max limit
        contracts = min(contracts, rl_scaled_max)
        
        # Detailed size multiplier logging
        logger.info(f"[POSITION SIZING] Size mult: {size_mult:.2f}x | "
                   f"Risk-based: {int(risk_dollars / risk_per_contract)} → "
                   f"RL-scaled max: {rl_scaled_max} (capped at user max {user_max_contracts}) → "
                   f"Final: {contracts} contracts")
        
    elif rl_confidence is not None and dynamic_contracts_enabled:
        # LEGACY: Calculate size multiplier from confidence only (deprecated)
        global rl_brain
        if rl_brain is not None:
            size_multiplier = rl_brain.get_position_size_multiplier(rl_confidence)
        else:
            size_multiplier = 1.0
            
        # Calculate RL-scaled max contracts
        # DYNAMIC SCALING: RL learns to scale WITHIN user's max
        # Examples:
        #   max=3, conf=50%  -> 2 contracts
        #   max=10, conf=50% -> 5 contracts  
        #   max=25, conf=50% -> 13 contracts
        #   max=25, conf=90% -> 23 contracts
        rl_scaled_max = max(1, int(round(user_max_contracts * size_multiplier)))
        
        # HARD CAP: Never exceed user's max_contracts setting
        rl_scaled_max = min(rl_scaled_max, user_max_contracts)
        
        # Use MINIMUM of risk-based calculation and RL-scaled max
        # This ensures we respect BOTH the risk management AND user's max limit
        contracts = min(contracts, rl_scaled_max)
        
        # Detailed confidence level logging
        if rl_confidence < 0.3:
            confidence_level = "VERY LOW"
        elif rl_confidence < 0.5:
            confidence_level = "LOW"
        elif rl_confidence < 0.7:
            confidence_level = "MEDIUM"
        elif rl_confidence < 0.85:
            confidence_level = "HIGH"
        else:
            confidence_level = "VERY HIGH"
            
        logger.info(f"[DYNAMIC CONTRACTS] {confidence_level} confidence ({rl_confidence:.1%}) × Max {user_max_contracts} = {rl_scaled_max} contracts (capped at {contracts} by risk)")
        
        # Update dashboard with confidence-based adjustment
        if dashboard:
            adjustment_msg = f"Confidence {confidence_level} ({int(rl_confidence*100)}%) → {contracts} contracts (from max {user_max_contracts})"
            dashboard.update_bot_data({"contract_adjustment": adjustment_msg})
            dashboard.display()
    else:
        # FIXED MODE - Use exact max_contracts, no learning adjustments
        contracts = min(contracts, user_max_contracts)
        if not hasattr(calculate_position_size, '_logged_fixed_mode'):
            logger.info(f"[FIXED MODE] Using exact max of {user_max_contracts} contracts - no dynamic adjustments")
            calculate_position_size._logged_fixed_mode = True
    
    # RECOVERY MODE: Further reduce position size when approaching limits
    if bot_status.get("recovery_confidence_threshold") is not None:
        severity = bot_status.get("recovery_severity", RECOVERY_DEFAULT_SEVERITY)
        # Dynamically scale position size based on proximity to failure
        # CONSERVATIVE SCALING - more aggressive reductions at higher severity
        # As bot gets FURTHER from failure, INCREASE contracts back to original
        # At 95%+ severity: 20% of normal size (was 33%)
        # At 90% severity: 40% of normal size (was 50%)
        # At 80% severity: 65% of normal size (was 75%)
        # At 70% severity: 80% of normal size (scaling back up)
        # At 60% severity: 90% of normal size (almost back to normal)
        # Below 50% severity: 100% of normal size (fully restored)
        if severity >= RECOVERY_SIZE_CRITICAL:
            recovery_multiplier = RECOVERY_MULTIPLIER_CRITICAL  # 20%
        elif severity >= RECOVERY_SIZE_HIGH:
            recovery_multiplier = RECOVERY_MULTIPLIER_HIGH  # 40%
        elif severity >= RECOVERY_SIZE_MODERATE:
            recovery_multiplier = RECOVERY_MULTIPLIER_MODERATE  # 65%
        elif severity >= 0.70:
            recovery_multiplier = 0.80  # Scaling back up - 80%
        elif severity >= 0.60:
            recovery_multiplier = 0.90  # Almost back to normal - 90%
        else:
            recovery_multiplier = 1.0  # Fully restored - 100%
        
        original_contracts = contracts
        contracts = max(1, int(round(contracts * recovery_multiplier)))
        
        if contracts != original_contracts:
            if recovery_multiplier < 1.0:
                logger.warning(f"[RECOVERY MODE] Position size adjusted: {original_contracts} → {contracts} contracts (severity: {severity*100:.0f}%, multiplier: {recovery_multiplier*100:.0f}%)")
                
                # Update dashboard with recovery mode adjustment
                if dashboard:
                    adjustment_msg = f"RECOVERY MODE: {original_contracts} → {contracts} contracts (severity {int(severity*100)}%, reduced to {int(recovery_multiplier*100)}%)"
                    dashboard.update_bot_data({"contract_adjustment": adjustment_msg})
                    dashboard.display()
            else:
                logger.info(f"[RECOVERY MODE] Position size restored: {original_contracts} → {contracts} contracts (severity: {severity*100:.0f}%, safe zone)")
                
                # Update dashboard - recovery mode restored
                if dashboard:
                    adjustment_msg = f"RECOVERY MODE CLEARED: Restored to {contracts} contracts (severity {int(severity*100)}%, safe zone)"
                    dashboard.update_bot_data({"contract_adjustment": adjustment_msg})
                    dashboard.display()
    
    if contracts == 0:
        logger.warning(f"Position size too small: risk=${risk_per_contract:.2f}, allowance=${risk_dollars:.2f}")
        return 0, stop_price, None
    
    # Calculate target distance for logging
    target_distance = abs(target_price - entry_price)
    
    logger.info(f"Position sizing: {contracts} contract(s)")
    logger.info(f"  Entry: ${entry_price:.2f}, Stop: ${stop_price:.2f}, Target: ${target_price:.2f}")
    logger.info(f"  Risk: {ticks_at_risk:.1f} ticks (${risk_per_contract:.2f})")
    logger.info(f"  Reward: {target_distance/tick_size:.1f} ticks ({target_distance/stop_distance:.1f}:1 R/R)")
    logger.info(f"  VWAP: ${vwap:.2f} (mean reversion target)")
    
    return contracts, stop_price, target_price


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
        logger.info(f"  ✅ Entry price valid: ${signal_price:.2f} → ${current_price:.2f} ({price_move_ticks:+.1f} ticks, limit: {max_deterioration_ticks})")
        return True, "Price acceptable", current_price
    else:
        if side == "long":
            reason = f"Price moved UP {price_move_ticks:.1f} ticks (limit: {max_deterioration_ticks})"
        else:
            reason = f"Price moved DOWN {price_move_ticks:.1f} ticks (limit: {max_deterioration_ticks})"
        
        logger.warning(f"  ❌ Entry skipped - {reason}")
        logger.warning(f"     Signal: ${signal_price:.2f} → Current: ${current_price:.2f}")
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
    import time
    time.sleep(timeout_seconds)
    
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
            logger.warning(f"  ✗ Partial fill too small ({fill_ratio:.0%}) - closing position")
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
    import time
    
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
                                    time.sleep(0.5)
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
                        time.sleep(0.5)  # Brief pause before retry
                        continue
                
                # Order placement failed
                logger.error(f"  [ERROR] Attempt {attempt}: Order placement failed")
                
            elif order_params['strategy'] == 'aggressive':
                limit_price = order_params['limit_price']
                logger.info(f"  [AGGRESSIVE] Aggressive Entry: ${limit_price:.2f} (guaranteed fill)")
                
                order = place_limit_order(symbol, order_side, contracts, limit_price)
                
                if order is not None:
                    # Aggressive orders usually fill immediately
                    time.sleep(1)  # Brief wait to confirm fill
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
                
                logger.info(f"  🔀 Mixed: {passive_qty}@${passive_price:.2f} (passive) + {aggressive_qty}@${aggressive_price:.2f} (aggressive)")
                
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
                time.sleep(backoff_time)
            
        except Exception as e:
            logger.error(f"  [FAIL] Attempt {attempt} exception: {e}")
            if attempt < max_retries:
                time.sleep(0.5 * attempt)
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
        logger.info(f"📊 SIGNAL ALERT - MANUAL TRADE OPPORTUNITY")
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
            suggested_target = vwap_bands["upper_3"]
            logger.info(f"  Suggested Stop: ${suggested_stop:.2f} ({max_stop_ticks} ticks)")
            logger.info(f"  Suggested Target: ${suggested_target:.2f} (Upper Band 3)")
        else:
            suggested_stop = entry_price + (max_stop_ticks * tick_size)
            suggested_target = vwap_bands["lower_3"]
            logger.info(f"  Suggested Stop: ${suggested_stop:.2f} ({max_stop_ticks} ticks)")
            logger.info(f"  Suggested Target: ${suggested_target:.2f} (Lower Band 3)")
        
        logger.info(f"")
        logger.info(f"  🎯 SHADOW MODE: Signal shown - No automatic execution")
        logger.info(f"  📱 Trade manually if you agree with this signal")
        logger.info(SEPARATOR_LINE)
        
        # Send notification if enabled
        try:
            notifier = get_notifier()
            notifier.send_trade_alert(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                stop_price=suggested_stop,
                target_price=suggested_target,
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
        logger.warning("🚨 ENTRY SKIPPED - Already In Position")
        logger.warning(f"  Current Position: {current_position} contracts ({'LONG' if current_position > 0 else 'SHORT'})")
        logger.warning(f"  New Signal: {side.upper()} @ ${entry_price:.2f}")
        logger.warning(f"  Reason: Cannot enter conflicting or additional position")
        logger.warning(SEPARATOR_LINE)
        return
    
    # ===== EXECUTION RISK FIX #4: Fast Market Detection =====
    too_fast, fast_reason = is_market_moving_too_fast(symbol)
    if too_fast:
        logger.warning(SEPARATOR_LINE)
        logger.warning("🚨 ENTRY SKIPPED - Fast Market Detected")
        logger.warning(f"  Reason: {fast_reason}")
        logger.warning(f"  Signal: {side.upper()} @ ${entry_price:.2f}")
        logger.warning(SEPARATOR_LINE)
        return
    
    # ===== EXECUTION RISK FIX #1: Price Deterioration Protection =====
    is_valid_price, price_reason, current_market_price = validate_entry_price_still_valid(symbol, entry_price, side)
    if not is_valid_price:
        logger.warning(SEPARATOR_LINE)
        logger.warning("🚨 ENTRY ABORTED - Price Deteriorated")
        logger.warning(f"  {price_reason}")
        logger.warning(f"  Signal: {side.upper()} @ ${entry_price:.2f}")
        logger.warning(SEPARATOR_LINE)
        
        # Track rejected signal for RL learning
        rl_state = state[symbol].get("entry_rl_state")
        if rl_state is not None:
            # Extract price movement from reason (e.g., "Price moved UP 5.0 ticks")
            import re
            match = re.search(r'(\d+\.?\d*)\s+ticks', price_reason)
            price_move_ticks = float(match.group(1)) if match else 0.0
            
            # Save to cloud so RL learns: "Should I take trades when price moved X ticks?"
            save_rejected_signal(
                rl_state=rl_state,
                side=side,
                rejection_reason=f"price_moved_{price_move_ticks:.0f}_ticks",
                price_move_ticks=price_move_ticks
            )
            
            # ========== SHADOW TRADE: Track what WOULD happen ==========
            # Create shadow trade to track rejected signal in real-time
            # Shadow will use FULL adaptive exit model (same as real trades)
            confidence = state[symbol].get("last_signal_confidence", 0.0)
            
            create_shadow_trade(
                symbol=symbol,
                rl_state=rl_state,
                side=side,
                entry_price=entry_price,  # Original signal price
                confidence=confidence,
                rejection_reason=f"price_moved_{price_move_ticks:.0f}_ticks"
            )
        
        return
    
    # Use current market price instead of stale signal price
    logger.info(f"  [OK] Price validation passed: ${entry_price:.2f} -> ${current_market_price:.2f}")
    entry_price = current_market_price
    
    # Get RL confidence and size multiplier if available
    rl_confidence = state[symbol].get("entry_rl_confidence")
    size_mult = state[symbol].get("entry_rl_size_mult")
    
    # Calculate position size (with RL adjustment if size_mult available)
    contracts, stop_price, target_price = calculate_position_size(symbol, side, entry_price, rl_confidence, size_mult)
    
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
            expected_profit_ticks = abs(target_price - entry_price) / CONFIG["tick_size"]
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
    logger.info(f"  Target: ${target_price:.2f}")
    
    # Track order execution details for post-trade analysis
    fill_start_time = datetime.now(pytz.UTC)
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
            state[symbol]["position"]["entry_confidence"] = state[symbol].get("entry_rl_confidence", 0.75)  # NEW: Save confidence for Exit RL correlation
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
                state[symbol]["position"]["entry_confidence"] = state[symbol].get("entry_rl_confidence", 0.75)  # NEW: Save confidence for Exit RL correlation
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
            state[symbol]["position"]["entry_confidence"] = state[symbol].get("entry_rl_confidence", 0.75)  # NEW: Save confidence for Exit RL correlation
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
            fill_time_seconds = (datetime.now(pytz.UTC) - fill_start_time).total_seconds()
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
    
    # Update position tracking
    state[symbol]["position"] = {
        "active": True,
        "side": side,
        "quantity": contracts,
        "entry_price": actual_fill_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "entry_time": entry_time,
        "order_id": order.get("order_id"),
        "order_type_used": order_type_used,  # Track for exit optimization
        # Signal & RL Information - Preserved for partial fills and exits
        "entry_rl_confidence": rl_confidence,  # RL confidence at entry
        "entry_rl_state": state[symbol].get("entry_rl_state"),  # RL market state
        "original_entry_price": entry_price,  # Original signal price (before validation)
        "actual_entry_price": actual_fill_price,  # Actual fill price
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
    logger.info("  ✓ Position state saved to disk")
    
    # Initialize all RL tracking fields for complete feature capture
    initialize_position_tracking(symbol)
    
    # Update state status message
    state[symbol]["bot_status_message"] = "In trade..."
    
    # Update dashboard with new position
    if dashboard:
        dashboard.update_symbol_data(symbol, {
            "position": f"{side.upper()} {contracts} @ ${actual_fill_price:.2f}",
            "position_qty": contracts,
            "position_side": side,
            "status": f"In trade - Target: ${target_price:.2f}"
        })
        dashboard.display()
    
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
        logger.error("🚨 CRITICAL: STOP ORDER REJECTED BY BROKER!")
        logger.error(f"  Entry filled at ${actual_fill_price:.2f} with {contracts} contracts")
        logger.error(f"  Stop order at ${stop_price:.2f} FAILED to place")
        logger.error(f"  Position is NOW UNPROTECTED - emergency exit required!")
        logger.error(SEPARATOR_LINE)
        
        # EMERGENCY: Close position immediately with market order
        logger.error("  🆘 Executing emergency market close to protect capital...")
        emergency_close_order = place_market_order(symbol, stop_side, contracts)
        
        if emergency_close_order:
            logger.error(f"  ✓ Emergency close executed - Position closed")
            logger.error(f"  This trade is abandoned due to stop order failure")
            
            # CRITICAL FIX: Clear position state since we closed it
            state[symbol]["position"]["active"] = False
            state[symbol]["position"]["quantity"] = 0
            state[symbol]["position"]["side"] = None
            state[symbol]["position"]["entry_price"] = None
            state[symbol]["position"]["stop_price"] = None
            state[symbol]["position"]["target_price"] = None
            
            # CRITICAL: Save state to disk immediately
            save_position_state(symbol)
            logger.error("  ✓ Position state cleared and saved to disk")
        else:
            critical_msg = f"EMERGENCY CLOSE FAILED - Manual intervention needed for {symbol}"
            log_critical_error(critical_msg)
            
            logger.error(f"  [FAIL] EMERGENCY CLOSE ALSO FAILED - MANUAL INTERVENTION REQUIRED!")
            logger.error(f"  Symbol: {symbol}, Side: {side}, Contracts: {contracts}")
            
            # CRITICAL FIX: Even if close failed, mark position as needing manual intervention
            # Don't let bot think it can trade normally with this broken state
            state[symbol]["position"]["active"] = False  # Prevent bot from managing this
            bot_status["emergency_stop"] = True
            bot_status["stop_reason"] = "stop_order_placement_failed"
            
            # CRITICAL: Save state to disk immediately
            save_position_state(symbol)
            logger.error("  ✓ Emergency stop activated and saved to disk")
        
        # Don't track this position - it's closed or needs manual handling
        logger.error(SEPARATOR_LINE)
        return
    
    # Increment daily trade counter
    state[symbol]["daily_trade_count"] += 1
    
    logger.info(f"Position opened successfully (Trade {state[symbol]['daily_trade_count']}/{CONFIG['max_trades_per_day']})")
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


def check_target_reached(symbol: str, current_bar: Dict[str, Any], position: Dict[str, Any], 
                        bar_time: datetime) -> Tuple[bool, Optional[float]]:
    """
    Check if profit target has been reached, including time-based adjustments.
    
    Args:
        symbol: Instrument symbol
        current_bar: Current 1-minute bar
        position: Position dictionary
        bar_time: Current bar timestamp
    
    Returns:
        Tuple of (target_reached, target_price)
    """
    side = position["side"]
    target_price = position["target_price"]
    entry_price = position["entry_price"]
    stop_price = position["stop_price"]
    
    # Check regular target
    if side == "long":
        if current_bar["high"] >= target_price:
            # ===== CRITICAL FIX #4: Target Order Validation =====
            # Price reached target - verify we can actually fill at this price
            # In backtesting, assume fill. In live, would check if limit order filled.
            
            # Check if price is still near target (within CONFIG threshold)
            tick_size = CONFIG["tick_size"]
            target_validation_ticks = CONFIG.get("target_fill_validation_ticks", 2)
            price_distance = abs(current_bar["close"] - target_price) / tick_size
            
            if price_distance <= target_validation_ticks:
                # Price still near target - good fill likely
                return True, target_price
            else:
                # Price ran past target and reversed - might not fill at target
                logger.warning(f"[WARN] Target Validation: Price hit ${target_price:.2f} but reversed to ${current_bar['close']:.2f}")
                logger.warning(f"  Distance: {price_distance:.1f} ticks (>{target_validation_ticks} tick threshold)")
                logger.warning(f"  Using current price for guaranteed fill instead")
                # Use current price (more conservative, guaranteed fill)
                return True, current_bar["close"]
    else:  # short
        if current_bar["low"] <= target_price:
            # ===== CRITICAL FIX #4: Target Order Validation =====
            tick_size = CONFIG["tick_size"]
            target_validation_ticks = CONFIG.get("target_fill_validation_ticks", 2)
            price_distance = abs(current_bar["close"] - target_price) / tick_size
            
            if price_distance <= target_validation_ticks:
                return True, target_price
            else:
                logger.warning(f"[WARN] Target Validation: Price hit ${target_price:.2f} but reversed to ${current_bar['close']:.2f}")
                logger.warning(f"  Distance: {price_distance:.1f} ticks (>{target_validation_ticks} tick threshold)")
                logger.warning(f"  Using current price for guaranteed fill instead")
                return True, current_bar["close"]
    
    # Phase Five: Time-based exit tightening after 19:00 UTC (7 PM UTC = late in session)
    if bar_time.time() >= datetime_time(19, 0) and not bot_status["flatten_mode"]:
        # After 19:00 UTC - tighten profit taking to 1:1 R/R
        stop_distance = abs(entry_price - stop_price)
        tightened_target_distance = stop_distance  # 1:1 instead of 1.5:1
        
        if side == "long":
            tightened_target = entry_price + tightened_target_distance
            if current_bar["high"] >= tightened_target:
                return True, tightened_target
        else:  # short
            tightened_target = entry_price - tightened_target_distance
            if current_bar["low"] <= tightened_target:
                return True, tightened_target
    
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
    if bot_status["flatten_mode"]:
        return False, None
    
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


def check_proactive_stop(symbol: str, current_bar: Dict[str, Any], position: Dict[str, Any]) -> Tuple[bool, Optional[float]]:
    """
    Check if position is within 2 ticks of stop during flatten mode.
    Prevents last-moment stop hunting.
    
    Args:
        symbol: Instrument symbol
        current_bar: Current 1-minute bar
        position: Position dictionary
    
    Returns:
        Tuple of (should_close, flatten_price)
    """
    if not bot_status["flatten_mode"]:
        return False, None
    
    side = position["side"]
    stop_price = position["stop_price"]
    tick_size = CONFIG["tick_size"]
    proactive_buffer = CONFIG["proactive_stop_buffer_ticks"] * tick_size
    
    if side == "long":
        # Check if within 2 ticks of stop price
        if current_bar["close"] <= stop_price + proactive_buffer:
            return True, get_flatten_price(symbol, side, current_bar["close"])
    else:  # short
        # Check if within 2 ticks of stop price
        if current_bar["close"] >= stop_price - proactive_buffer:
            return True, get_flatten_price(symbol, side, current_bar["close"])
    
    return False, None


def check_time_based_exits(symbol: str, current_bar: Dict[str, Any], position: Dict[str, Any], 
                           bar_time: datetime) -> Tuple[Optional[str], Optional[float]]:
    """
    Check all time-based exit conditions.
    
    Args:
        symbol: Instrument symbol
        current_bar: Current 1-minute bar
        position: Position dictionary
        bar_time: Current bar timestamp
    
    Returns:
        Tuple of (exit_reason, exit_price) or (None, None)
    """
    side = position["side"]
    entry_price = position["entry_price"]
    stop_price = position["stop_price"]
    tick_size = CONFIG["tick_size"]
    tick_value = CONFIG["tick_value"]
    
    # Calculate unrealized P&L (used in multiple checks)
    if side == "long":
        price_change = current_bar["close"] - entry_price
    else:
        price_change = entry_price - current_bar["close"]
    ticks = price_change / tick_size
    unrealized_pnl = ticks * tick_value * position["quantity"]
    
    # Force close at forced_flatten_time (21:00 UTC - maintenance starts)
    trading_state = get_trading_state(bar_time)
    if trading_state == "closed":
        return "emergency_forced_flatten", get_flatten_price(symbol, side, current_bar["close"])
    
    # Flatten mode: specific time-based exits (UTC times)
    if bot_status["flatten_mode"]:
        tz = pytz.timezone(CONFIG["timezone"])
        current_time = datetime.now(tz)
        
        # 20:45 UTC - close profitable positions immediately (flatten mode start)
        if current_time.time() >= datetime_time(20, 45) and unrealized_pnl > 0:
            return "time_based_profit_take", get_flatten_price(symbol, side, current_bar["close"])
        
        # 20:47 UTC - close small losses (allow 2 min for profits to exit first)
        if current_time.time() >= datetime_time(20, 47):
            stop_distance = abs(entry_price - stop_price)
            if unrealized_pnl < 0 and abs(unrealized_pnl) < (stop_distance * tick_value * position["quantity"] / 2):
                return "time_based_loss_cut", get_flatten_price(symbol, side, current_bar["close"])
        
        # Flatten mode: aggressive profit/loss management during 20:45-21:00 window
        if side == "long":
            profit_ticks = (current_bar["close"] - entry_price) / tick_size
            midpoint = entry_price - (entry_price - stop_price) / 2
            if profit_ticks > 1 or current_bar["close"] < midpoint:
                return "flatten_mode_exit", get_flatten_price(symbol, side, current_bar["close"])
        else:  # short
            profit_ticks = (entry_price - current_bar["close"]) / tick_size
            midpoint = entry_price + (stop_price - entry_price) / 2
            if profit_ticks > 1 or current_bar["close"] > midpoint:
                return "flatten_mode_exit", get_flatten_price(symbol, side, current_bar["close"])
    
    # Friday-specific exits (UTC times)
    if bar_time.weekday() == 4:  # Friday
        # Target close by 20:45 UTC to avoid weekend gap risk
        if bar_time.time() >= CONFIG["friday_close_target"]:
            return "friday_weekend_protection", get_flatten_price(symbol, side, current_bar["close"])
        
        # After 18:00 UTC on Friday, take any profit (conservative weekend protection)
        if bar_time.time() >= datetime_time(18, 0) and unrealized_pnl > 0:
            return "friday_profit_protection", get_flatten_price(symbol, side, current_bar["close"])
    
    # After 19:30 UTC - cut losses early if less than 75% of stop distance
    if bar_time.time() >= datetime_time(19, 30) and not bot_status["flatten_mode"]:
        if side == "long":
            current_loss_distance = entry_price - current_bar["close"]
            stop_distance = entry_price - stop_price
            if current_loss_distance > 0:  # In a loss
                loss_percent = current_loss_distance / stop_distance
                if loss_percent < 0.75:  # Less than 75% of stop distance
                    return "early_loss_cut", current_bar["close"]
        else:  # short
            current_loss_distance = current_bar["close"] - entry_price
            stop_distance = stop_price - entry_price
            if current_loss_distance > 0:  # In a loss
                loss_percent = current_loss_distance / stop_distance
                if loss_percent < 0.75:  # Less than 75% of stop distance
                    return "early_loss_cut", current_bar["close"]
    
    return None, None


# ============================================================================
# PHASE THREE: Breakeven Protection Logic
# ============================================================================

def check_breakeven_protection(symbol: str, current_price: float) -> None:
    """
    Check if breakeven protection should be activated and move stop to breakeven.
    
    Uses ADAPTIVE parameters that adjust based on:
    - Current market volatility (ATR)
    - Market regime (trending vs choppy)
    - Trade performance
    
    Args:
        symbol: Instrument symbol
        current_price: Current market price
    """
    global adaptive_manager
    
    position = state[symbol]["position"]
    
    # Step 1 - Check eligibility: Only process positions that haven't activated breakeven yet
    if not position["active"] or position["breakeven_active"]:
        return
    
    side = position["side"]
    entry_price = position["entry_price"]
    tick_size = CONFIG["tick_size"]
    
    # ========================================================================
    # ADAPTIVE EXIT MANAGEMENT - Calculate dynamic thresholds
    # ========================================================================
    
    if adaptive_manager is not None:
        try:
            from adaptive_exits import get_adaptive_exit_params
            
            # CONTINUOUS ADAPTATION: Recalculate exit params on EVERY bar
            # This allows the bot to adapt as market conditions change mid-trade
            
            # Get entry confidence from position (saved during entry)
            entry_confidence = position.get("entry_confidence", 0.75)  # Default to 0.75 if not saved
            
            adaptive_params = get_adaptive_exit_params(
                bars=state[symbol]["bars_1min"],
                position=position,
                current_price=current_price,
                config=CONFIG,
                adaptive_manager=adaptive_manager,  # Pass global instance for persistence
                entry_confidence=entry_confidence   # NEW: Pass confidence for exit correlation
            )
            
            breakeven_threshold_ticks = adaptive_params["breakeven_threshold_ticks"]
            breakeven_offset_ticks = adaptive_params["breakeven_offset_ticks"]
            
            # UPDATE exit params continuously (market conditions change!)
            # Compare with previous params to detect regime shifts
            prev_regime = position.get("exit_params_used", {}).get("market_regime", "UNKNOWN")
            new_regime = adaptive_params["market_regime"]
            
            if prev_regime != new_regime and prev_regime != "UNKNOWN":
                logger.info(f" [REGIME SHIFT] {prev_regime} → {new_regime}")
                logger.info(f"   Adjusting exit parameters mid-trade for new conditions")
            
            # Store updated exit params for learning when trade closes
            position["exit_params_used"] = adaptive_params
            
        except Exception as e:
            logger.error(f"[FAIL] Adaptive exits ERROR: {e}", exc_info=True)
            logger.warning("[WARN] Falling back to static params")
            breakeven_threshold_ticks = CONFIG.get("breakeven_profit_threshold_ticks", 8)
            breakeven_offset_ticks = CONFIG.get("breakeven_stop_offset_ticks", 1)
    else:
        # Static parameters
        breakeven_threshold_ticks = CONFIG.get("breakeven_profit_threshold_ticks", 8)
        breakeven_offset_ticks = CONFIG.get("breakeven_stop_offset_ticks", 1)
    
    # Step 2 - Calculate current profit in ticks
    if side == "long":
        profit_ticks = (current_price - entry_price) / tick_size
    else:  # short
        profit_ticks = (entry_price - current_price) / tick_size
    
    # === EXIT RL FEATURE 5: PROFIT LOCK ZONES ===
    # Check if profit is retreating from peak (lock in gains)
    if adaptive_manager is not None and position.get("breakeven_active"):
        try:
            # Calculate R-multiple
            initial_risk_ticks = position.get("initial_risk_ticks", 10)
            current_r_multiple = profit_ticks / initial_risk_ticks if initial_risk_ticks > 0 else 0
            
            # Track peak R-multiple
            peak_r_multiple = position.get("peak_r_multiple", 0)
            if current_r_multiple > peak_r_multiple:
                position["peak_r_multiple"] = current_r_multiple
                peak_r_multiple = current_r_multiple
            
            # Check for profit lock
            if peak_r_multiple >= 3.0:  # Only check if we've hit significant profit
                lock_result = adaptive_manager.check_profit_lock(
                    current_r_multiple=current_r_multiple,
                    peak_r_multiple=peak_r_multiple,
                    current_profit_ticks=profit_ticks,
                    direction=side
                )
                
                if lock_result.get('should_lock'):
                    lock_reason = lock_result.get('reason', 'Profit lock triggered')
                    logger.warning(f"[EXIT RL] {lock_reason}")
                    logger.warning(f"  Peak: {peak_r_multiple:.1f}R → Current: {current_r_multiple:.1f}R")
                    
                    # Execute immediate exit to lock profit
                    from quotrading_engine import execute_exit
                    execute_exit(symbol, current_price, "profit_lock")
                    return  # Exit function immediately
        except Exception as e:
            logger.debug(f"Profit lock check failed: {e}")
    
    # Step 3 - Compare to threshold
    if profit_ticks < breakeven_threshold_ticks:
        return  # Not enough profit yet
    
    # Step 4 - Calculate new breakeven stop price
    if side == "long":
        new_stop_price = entry_price + (breakeven_offset_ticks * tick_size)
    else:  # short
        new_stop_price = entry_price - (breakeven_offset_ticks * tick_size)
    
    new_stop_price = round_to_tick(new_stop_price)
    
    # Step 5 - Update stop loss
    # Place new stop at breakeven level (broker will replace existing stop)
    stop_side = "SELL" if side == "long" else "BUY"
    contracts = position["quantity"]
    new_stop_order = place_stop_order(symbol, stop_side, contracts, new_stop_price)
    
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
        logger.info(f"  Current Profit: {profit_ticks:.1f} ticks (threshold: {breakeven_threshold_ticks} ticks)")
        logger.info(f"  Original Stop: ${original_stop:.2f}")
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
    
    Uses ADAPTIVE parameters that adjust based on:
    - Current market volatility (ATR)
    - Market regime (trending vs choppy)  
    - Position holding duration
    
    Runs AFTER breakeven check. Only processes positions where breakeven is already active.
    Continuously updates stop to follow profitable price movement while protecting gains.
    
    Args:
        symbol: Instrument symbol
        current_price: Current market price
    """
    global adaptive_manager
    
    position = state[symbol]["position"]
    
    # Step 1 - Check eligibility: Position must have breakeven active
    if not position["active"] or not position["breakeven_active"]:
        return
    
    side = position["side"]
    entry_price = position["entry_price"]
    tick_size = CONFIG["tick_size"]
    
    # ========================================================================
    # ADAPTIVE EXIT MANAGEMENT - Calculate dynamic trailing parameters
    # ========================================================================
    if adaptive_manager is not None:
        try:
            from adaptive_exits import get_adaptive_exit_params
            
            # CONTINUOUS ADAPTATION: Recalculate trailing params on EVERY bar
            # Market conditions change while holding → trailing should adapt too!
            
            # Get entry confidence from position (saved during entry)
            entry_confidence = position.get("entry_confidence", 0.75)  # Default to 0.75 if not saved
            
            adaptive_params = get_adaptive_exit_params(
                bars=state[symbol]["bars_1min"],
                position=position,
                current_price=current_price,
                config=CONFIG,
                adaptive_manager=adaptive_manager,  # Pass global instance for persistence
                entry_confidence=entry_confidence   # NEW: Pass confidence for exit correlation
            )
            
            trailing_distance_ticks = adaptive_params["trailing_distance_ticks"]
            min_profit_ticks = adaptive_params["trailing_min_profit_ticks"]
            
            # UPDATE exit params continuously (already done in breakeven, but update here too)
            # This ensures trailing stop reflects latest market conditions
            prev_regime = position.get("exit_params_used", {}).get("market_regime", "UNKNOWN")
            new_regime = adaptive_params["market_regime"]
            
            # Log significant changes in trailing distance (regime shift mid-trade)
            prev_trail = position.get("exit_params_used", {}).get("trailing_distance_ticks", 0)
            if prev_trail > 0 and abs(trailing_distance_ticks - prev_trail) >= 2:
                change_pct = ((trailing_distance_ticks - prev_trail) / prev_trail * 100)
                logger.info(f" [TRAILING ADAPT] {prev_trail}t → {trailing_distance_ticks}t ({change_pct:+.0f}%)")
                if new_regime != prev_regime:
                    logger.info(f"   Reason: Regime shift ({prev_regime} → {new_regime})")
            
            # Store updated exit params for learning
            position["exit_params_used"] = adaptive_params
            
        except Exception as e:
            logger.error(f"[FAIL] Adaptive trailing ERROR: {e}", exc_info=True)
            logger.warning("[WARN] Falling back to static trailing params")
            trailing_distance_ticks = CONFIG.get("trailing_stop_distance_ticks", 8)
            min_profit_ticks = CONFIG.get("trailing_stop_min_profit_ticks", 12)
    else:
        # Static parameters
        trailing_distance_ticks = CONFIG.get("trailing_stop_distance_ticks", 8)
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
    
    # Step 5 - Update stop loss
    stop_side = "SELL" if side == "long" else "BUY"
    contracts = position["quantity"]
    new_stop_order = place_stop_order(symbol, stop_side, contracts, new_trailing_stop)
    
    if new_stop_order:
        # Activate trailing stop flag if not already active
        if not position["trailing_stop_active"]:
            position["trailing_stop_active"] = True
            position["trailing_activated_time"] = get_current_time()
        
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
        logger.error("🚨 CRITICAL: TRAILING STOP UPDATE FAILED!")
        logger.error(f"  Tried to update stop from ${position['stop_price']:.2f} to ${new_trailing_stop:.2f}")
        logger.error(f"  Current profit: ${profit_locked_dollars:+.2f} (UNPROTECTED)")
        logger.error("  Position now at risk - emergency exit required!")
        logger.error(SEPARATOR_LINE)
        
        # EMERGENCY: Close position immediately to lock in profit
        logger.error("  🆘 Executing emergency market close to protect profit...")
        emergency_close_order = place_market_order(symbol, stop_side, contracts)
        
        if emergency_close_order:
            logger.error("  ✓ Emergency close executed - profit protected")
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
    tick_size = CONFIG["tick_size"]
    tick_value = CONFIG["tick_value"]
    
    # Step 1 - Calculate time percentage
    # Max holding period: use time until flatten mode (conservative)
    # From entry window end (6:30 PM UTC) to flatten deadline (8:45 PM UTC) = 135 minutes
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
    
    # Step 7 - Update stop loss
    stop_side = "SELL" if side == "long" else "BUY"
    contracts = position["quantity"]
    new_stop_order = place_stop_order(symbol, stop_side, contracts, new_stop_price)
    
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

def get_current_hour() -> int:
    """Get current hour in EST/EDT trading timezone."""
    from datetime import datetime
    import pytz
    try:
        tz = pytz.timezone('UTC')
        now = datetime.now(tz)
        return now.hour
    except:
        # Fallback to UTC if timezone fails
        return datetime.now(pytz.UTC).hour


def check_partial_exits(symbol: str, current_price: float) -> None:
    """
    Execute partial exits using LEARNED scaling strategies.
    
    Bot learns WHEN and HOW MUCH to scale based on market context.
    
    Args:
        symbol: Instrument symbol
        current_price: Current market price
    """
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
    
    # Get learned scaling strategy from adaptive exit manager
    from adaptive_exits import get_recommended_scaling_strategy, detect_market_regime
    
    # Get bars for market state calculation
    bars = state[symbol]["bars_1min"]
    
    # Build market state for scaling decision
    market_state = {
        'rsi': position.get('entry_rsi', calculate_rsi([bar["close"] for bar in bars[-50:]]) if len(bars) >= 50 else 50.0),
        'volume_ratio': bars[-1]['volume'] / (sum(b['volume'] for b in bars[-20:]) / 20) if len(bars) >= 20 else 1.0,
        'hour': get_current_hour(),
        'streak': position.get('signal_streak', 0)
    }
    
    # Detect regime
    atr_value = calculate_atr(symbol, CONFIG.get("atr_period", 14))
    regime = detect_market_regime(bars, atr_value if atr_value else 2.0)
    
    # Get adaptive scaling strategy
    adaptive_manager = position.get('adaptive_exit_manager')
    scaling = get_recommended_scaling_strategy(market_state, regime, adaptive_manager)
    
    # Log strategy with learning mode
    if scaling.get('learning_mode') == 'EXPERIENCE_BASED':
        logger.info(f"[SCALING STRATEGY] {scaling['strategy']} | "
                   f"Found {scaling.get('similar_count', 0)} similar past exits | "
                   f"Avg P&L: ${scaling.get('avg_pnl', 0):.0f} | Avg R: {scaling.get('avg_r', 0):.1f}")
    else:
        logger.info(f"[SCALING STRATEGY] {scaling['strategy']} (rule-based fallback)")
    
    # Store scaling strategy in position for exit learning
    if "exit_params_used" not in position:
        position["exit_params_used"] = {}
    position["exit_params_used"]["scaling_strategy"] = scaling['strategy']
    position["exit_params_used"]["scaling_r_multiples"] = [scaling['partial_1_r'], scaling['partial_2_r'], scaling['partial_3_r']]
    position["exit_params_used"]["scaling_percentages"] = [scaling['partial_1_pct'], scaling['partial_2_pct'], scaling['partial_3_pct']]
    
    # Check each partial exit threshold using LEARNED strategy
    
    # First partial
    if (r_multiple >= scaling['partial_1_r'] and 
        not position["partial_exit_1_completed"]):
        
        contracts_to_close = int(original_quantity * scaling['partial_1_pct'])
        
        if contracts_to_close >= 1:
            execute_partial_exit(symbol, contracts_to_close, current_price, r_multiple, 
                                "partial_exit_1_completed", 1, scaling['partial_1_pct'])
            return  # Exit one partial per bar to avoid race conditions
    
    # Second partial
    if (r_multiple >= scaling['partial_2_r'] and 
        not position["partial_exit_2_completed"]):
        
        contracts_to_close = int(original_quantity * scaling['partial_2_pct'])
        
        if contracts_to_close >= 1:
            execute_partial_exit(symbol, contracts_to_close, current_price, r_multiple,
                                "partial_exit_2_completed", 2, scaling['partial_2_pct'])
            return
    
    # Third partial (final runner) - CHECK IF WE SHOULD HOLD LONGER
    if (r_multiple >= scaling['partial_3_r'] and 
        not position["partial_exit_3_completed"]):
        
        # RUNNER HOLD LOGIC: Check if similar big winners went much higher
        should_hold = False
        if adaptive_manager:
            try:
                # Build market state for runner analysis
                bars = state[symbol]["bars_1min"]
                current_bar = bars[-1] if len(bars) > 0 else {}
                market_state = {
                    "regime": detect_market_regime(bars, calculate_atr(symbol, CONFIG.get("atr_period", 14)) or 2.0),
                    "rsi": calculate_rsi([bar["close"] for bar in bars[-50:]]) if len(bars) >= 50 else 50.0,
                    "volume_ratio": bars[-1]["volume"] / sum(b["volume"] for b in bars[-20:]) * 20 if len(bars) >= 20 else 1.0,
                    "hour": get_current_hour(),
                    "vwap_distance": abs(current_price - current_bar.get("vwap", current_price))
                }
                
                should_hold = adaptive_manager.should_hold_runner(position, market_state, r_multiple)
                if should_hold:
                    logger.info(f"[EXIT RL] Holding runner at {r_multiple:.1f}R - similar trades averaged 5R+")
                    logger.info(f"  Regime: {market_state['regime']}, Trend strength high")
                    return  # Skip closing runner this bar
            except Exception as e:
                logger.debug(f"Runner hold check failed: {e}")
        
        # If not holding, close the runner
        if not should_hold:
            # Close all remaining contracts
            remaining_quantity = position["remaining_quantity"]
            
            if remaining_quantity >= 1:
                execute_partial_exit(symbol, remaining_quantity, current_price, r_multiple,
                                    "partial_exit_3_completed", 3, scaling['partial_3_pct'], is_final=True)
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
        # Update position tracking
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
        
        # ADAPTIVE EXIT LEARNING - Record partial exit for pattern learning
        global adaptive_manager
        try:
            if adaptive_manager is not None and hasattr(adaptive_manager, 'record_exit_outcome'):
                # Get exit parameters that were used
                if "exit_params_used" in position:
                    exit_params = position["exit_params_used"]
                    
                    # Calculate market regime at exit (don't rely on stored value)
                    from adaptive_exits import detect_market_regime
                    bars = state[symbol]["bars_1min"]
                    atr_value = calculate_atr(symbol, CONFIG.get("atr_period", 14))
                    regime = detect_market_regime(bars, atr_value if atr_value else 2.0)
                    
                    # Capture current market state for context-aware learning
                    try:
                        current_rl_state = capture_rl_state(symbol, side, exit_price)
                        market_state = {
                            'rsi': current_rl_state.get('rsi', 50.0),
                            'volume_ratio': current_rl_state.get('volume_ratio', 1.0),
                            'hour': current_rl_state.get('hour', 12),
                            'day_of_week': current_rl_state.get('day_of_week', 0),
                            'streak': current_rl_state.get('streak', 0),
                            'recent_pnl': current_rl_state.get('recent_pnl', 0.0),
                            'vix': current_rl_state.get('vix', 15.0),
                            'vwap_distance': current_rl_state.get('vwap_distance', 0.0),
                            'atr': current_rl_state.get('atr', exit_params.get('current_atr', 0))
                        }
                    except Exception as e:
                        logger.debug(f"Failed to capture market state for partial exit: {e}")
                        market_state = None
                    
                    # Record partial exit outcome WITH SCALING STRATEGY
                    adaptive_manager.record_exit_outcome(
                        regime=regime,
                        exit_params=exit_params,
                        trade_outcome={
                            'pnl': profit_dollars,
                            'duration': 0,  # Don't have duration for partials
                            'exit_reason': f'partial_exit_{level}_{r_multiple:.1f}R',
                            'side': side,
                            'contracts': contracts,
                            'win': profit_dollars > 0,
                            'partial': True,
                            'r_multiple': r_multiple
                        },
                        market_state=market_state,
                        partial_exits=[{
                            'level': level,
                            'r_multiple': r_multiple,
                            'contracts': contracts,
                            'percentage': percentage
                        }]
                    )
                    
                    logger.info(f"  [EXIT RL] Learned partial exit @ {r_multiple:.1f}R -> ${profit_dollars:+.2f} ({exit_params.get('scaling_strategy', 'UNKNOWN')})")
        except Exception as e:
            logger.debug(f"Partial exit learning failed: {e}")
        
        # Step 10 - Handle edge case: check if position should be fully closed
        if position["remaining_quantity"] < 1 or is_final:
            logger.info("  Position FULLY CLOSED via partial exits")
            logger.info("=" * 60)
            
            # Mark position as inactive
            position["active"] = False
            
            # Update daily P&L
            state[symbol]["daily_pnl"] += profit_dollars
            
            # Update session stats
            update_session_stats(symbol, profit_dollars)
        else:
            logger.info(f"  Remaining: {position['remaining_quantity']} contracts")
            logger.info("=" * 60)
            
            # Update daily P&L for this partial
            state[symbol]["daily_pnl"] += profit_dollars
    else:
        logger.error(f"Failed to execute partial exit #{level}")


def initialize_position_tracking(symbol: str) -> None:
    """
    Initialize all RL tracking fields when position is opened.
    Call this AFTER position basics (entry_price, side, etc.) are set.
    """
    position = state[symbol]["position"]
    
    # Get session info
    current_time = get_current_time()
    hour = current_time.hour
    if 0 <= hour < 8:
        session = "Asia"
    elif 8 <= hour < 14:
        session = "London"
    else:
        session = "NY"
    
    position["entry_session"] = session
    
    # Initialize ATR tracking
    current_atr = calculate_atr(symbol, CONFIG.get("atr_period", 14))
    if current_atr:
        position["entry_atr"] = current_atr
        position["atr_samples"] = [current_atr]
    
    # Get volatility regime
    if len(state[symbol]["bars_1min"]) > 0:
        latest_bar = state[symbol]["bars_1min"][-1]
        vix = latest_bar.get("vix", 15.0)
        if vix < 12:
            vol_regime = "LOW_VOL"
        elif vix > 20:
            vol_regime = "HIGH_VOL"
        else:
            vol_regime = "NORMAL"
        position["volatility_regime_at_entry"] = vol_regime
    
    # Initialize highest/lowest tracking
    position["highest_price_reached"] = position["entry_price"]
    position["lowest_price_reached"] = position["entry_price"]
    
    # Count recent wins/losses
    if "trade_history" in state[symbol] and len(state[symbol]["trade_history"]) > 0:
        recent_5 = state[symbol]["trade_history"][-5:]
        position["wins_in_last_5_trades"] = sum(1 for t in recent_5 if t.get("pnl", 0) > 0)
        position["losses_in_last_5_trades"] = sum(1 for t in recent_5 if t.get("pnl", 0) < 0)
    
    # Session trade number
    position["trade_number_in_session"] = state[symbol].get("total_trades", 0) + 1
    
    logger.debug(f"[RL TRACKING] Initialized position metrics: session={session}, entry_atr={position.get('entry_atr', 0):.2f}")


def update_position_metrics(symbol: str, current_price: float) -> None:
    """
    Update all position metrics on every bar for complete RL feature tracking.
    Matches backtest tracking exactly.
    
    Args:
        symbol: Instrument symbol
        current_price: Current market price
    """
    position = state[symbol]["position"]
    if not position["active"]:
        return
    
    entry_price = position["entry_price"]
    side = position["side"]
    quantity = position["quantity"]
    tick_size = CONFIG.get("tick_size", 0.25)
    tick_value = CONFIG.get("tick_value", 12.50)
    
    # Increment duration
    position["duration_bars"] += 1
    
    # Calculate current P&L
    if side == "long":
        pnl_ticks = (current_price - entry_price) / tick_size
    else:
        pnl_ticks = (entry_price - current_price) / tick_size
    
    unrealized_pnl = pnl_ticks * tick_value * quantity
    position["unrealized_pnl"] = unrealized_pnl
    position["unrealized_pnl_history"].append(unrealized_pnl)
    
    # Update MAE (Maximum Adverse Excursion) - worst drawdown
    if side == "long":
        current_mae = (position.get("lowest_price_reached", current_price) - entry_price) / tick_size * tick_value * quantity
    else:
        current_mae = (entry_price - position.get("highest_price_reached", current_price)) / tick_size * tick_value * quantity
    
    if current_mae < position["mae"]:
        position["mae"] = current_mae
    
    # Update MFE (Maximum Favorable Excursion) - best profit
    if side == "long":
        current_mfe = (position.get("highest_price_reached", current_price) - entry_price) / tick_size * tick_value * quantity
    else:
        current_mfe = (entry_price - position.get("lowest_price_reached", current_price)) / tick_size * tick_value * quantity
    
    if current_mfe > position["mfe"]:
        position["mfe"] = current_mfe
    
    # Update peak unrealized P&L
    if unrealized_pnl > position["peak_unrealized_pnl"]:
        position["peak_unrealized_pnl"] = unrealized_pnl
        position["profit_drawdown_from_peak"] = 0.0
    else:
        # Calculate drawdown from peak
        position["profit_drawdown_from_peak"] = position["peak_unrealized_pnl"] - unrealized_pnl
        if position["peak_unrealized_pnl"] > 0:
            position["drawdown_bars"] += 1
    
    # Calculate R-multiples
    risk_dollars = abs(entry_price - position["stop_price"]) * quantity * tick_value / tick_size
    if risk_dollars > 0:
        r_multiple = unrealized_pnl / risk_dollars
        position["max_r_achieved"] = max(position["max_r_achieved"], r_multiple)
        position["min_r_achieved"] = min(position["min_r_achieved"], r_multiple)
        position["peak_r_multiple"] = max(position["peak_r_multiple"], r_multiple)
    
    # Track ATR samples
    current_atr = calculate_atr(symbol, CONFIG.get("atr_period", 14))
    if current_atr:
        position["atr_samples"].append(current_atr)
        if position["entry_atr"] is None:
            position["entry_atr"] = current_atr
        
        # Count high volatility bars (ATR > 1.5x entry ATR)
        if current_atr > position["entry_atr"] * 1.5:
            position["high_volatility_bars"] += 1
    
    # Track time in breakeven
    if position.get("breakeven_active", False):
        position["time_in_breakeven_bars"] += 1
    
    # Update bars until milestones
    if not position.get("breakeven_active", False):
        position["bars_until_breakeven"] = position["duration_bars"]
    
    if not position.get("trailing_stop_active", False):
        position["bars_until_trailing"] = position["duration_bars"]


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
    current_price = current_bar["close"]
    
    # UPDATE ALL POSITION METRICS FIRST (for complete RL tracking)
    update_position_metrics(symbol, current_price)
    
    # UPDATE SHADOW TRADES (ghost signal tracking)
    update_shadow_trades(symbol, current_bar)
    
    current_bar = state[symbol]["bars_1min"][-1]
    bar_time = current_bar["timestamp"]
    side = position["side"]
    entry_price = position["entry_price"]
    stop_price = position["stop_price"]
    
    # Track MAE/MFE on EVERY bar for learning
    if adaptive_manager:
        try:
            trade_id = position.get("entry_time", bar_time).isoformat()
            adaptive_manager.track_mae_mfe(
                trade_id=trade_id,
                entry_price=entry_price,
                current_price=current_bar["close"],
                direction=side,
                position_size=position["quantity"]
            )
        except Exception as e:
            logger.debug(f"MAE/MFE tracking failed: {e}")
    
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
    
    # ECONOMIC EVENT: Force close ONLY if user enabled it
    if trading_state == "event_block" and position["active"]:
        fomc_force_close = CONFIG.get("fomc_force_close", False)
        
        if fomc_force_close:
            # User wants to force close during economic events
            event_reason = bot_status.get('halt_reason', 'Economic event active')
            
            logger.critical(SEPARATOR_LINE)
            logger.critical(f"ECONOMIC EVENT - AUTO-FLATTENING POSITION")
            logger.critical(f"Event: {event_reason}")
            logger.critical(f"Time: {bar_time.strftime('%H:%M:%S %Z')}")
            logger.critical(f"Position: {side.upper()} {position['quantity']} @ ${entry_price:.2f}")
            logger.critical("(User enabled 'Force Close During FOMC')")
            logger.critical(SEPARATOR_LINE)
            
            # Force close immediately
            flatten_price = get_flatten_price(symbol, side, current_bar["close"])
            
            log_time_based_action(
                "event_flatten",
                f"{event_reason} - auto-flattening position per user preference",
                {
                    "side": side,
                    "quantity": position["quantity"],
                    "entry_price": f"${entry_price:.2f}",
                    "exit_price": f"${flatten_price:.2f}",
                    "time": bar_time.strftime('%H:%M:%S %Z'),
                    "reason": event_reason
                }
            )
            
            # Execute close order
            handle_exit_orders(symbol, position, flatten_price, "event_flatten")
            logger.info("Position flattened for economic event - will auto-resume after event window")
            return
        else:
            # User wants to keep position during events, just block new entries
            logger.info(f"📅 Economic event active: {bot_status.get('halt_reason', 'Event')}")
            logger.info("  Keeping existing position (Force Close disabled)")
            logger.info("  New entries blocked until event window closes")
    
    # AUTO-RESUME: Reset flatten mode when market reopens
    if trading_state == "entry_window" and bot_status["flatten_mode"]:
        bot_status["flatten_mode"] = False
        logger.info(SEPARATOR_LINE)
        logger.info("MARKET REOPENED - AUTO-RESUMING TRADING")
        logger.info(f"Time: {bar_time.strftime('%H:%M:%S %Z')}")
        logger.info("Flatten mode deactivated - ready for new entries")
        logger.info(SEPARATOR_LINE)
        
        # Send trading resumed alert
        try:
            notifier = get_notifier()
            notifier.send_error_alert(
                error_message=f"Market reopened at {bar_time.strftime('%I:%M %p %Z')}. Bot has resumed trading and is ready for new entries.",
                error_type="Trading Resumed"
            )
        except Exception as e:
            logger.debug(f"Failed to send trading resumed alert: {e}")
    
    # Phase Six: Enhanced flatten mode activation
    if trading_state == "flatten_mode" and not bot_status["flatten_mode"]:
        bot_status["flatten_mode"] = True
        
        # Log flatten mode activation with position details
        tick_size = CONFIG["tick_size"]
        tick_value = CONFIG["tick_value"]
        if side == "long":
            price_change = current_bar["close"] - entry_price
        else:
            price_change = entry_price - current_bar["close"]
        ticks = price_change / tick_size
        unrealized_pnl = ticks * tick_value * position["quantity"]
        
        position_details = {
            "side": position["side"],
            "quantity": position["quantity"],
            "entry_price": f"${position['entry_price']:.2f}",
            "current_time": bar_time.strftime('%H:%M:%S %Z'),
            "unrealized_pnl": f"${unrealized_pnl:+.2f}"
        }
        
        log_time_based_action(
            "flatten_mode_activated",
            "All positions must be closed by 8:45 PM UTC (15 min before maintenance)",
            position_details
        )
        
        logger.critical(SEPARATOR_LINE)
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
            recap_msg = f"📊 PRE-MAINTENANCE DAILY RECAP (8:45 PM UTC)\n\n"
            recap_msg += f"Trades Today: {total_trades}\n"
            recap_msg += f"Daily P&L: ${daily_pnl:+.2f}\n"
            recap_msg += f"Win Rate: {win_rate:.1f}% ({winning_trades}W/{losing_trades}L)\n"
            
            if position["quantity"] > 0:
                recap_msg += f"\nOpen Position: {position['side'].upper()} {position['quantity']} @ ${position['entry_price']:.2f}\n"
                recap_msg += f"Unrealized P&L: ${unrealized_pnl:+.2f}\n"
            
            recap_msg += f"\nBot entering flatten mode. All positions will close before 9:00 PM UTC maintenance window."
            
            notifier.send_error_alert(
                error_message=recap_msg,
                error_type="Daily Recap"
            )
        except Exception as e:
            logger.debug(f"Failed to send daily recap alert: {e}")
    
    # Minute-by-minute status logging during flatten mode
    if bot_status["flatten_mode"]:
        tz = pytz.timezone(CONFIG["timezone"])
        current_time = datetime.now(tz)
        forced_flatten_time = datetime.combine(current_time.date(), CONFIG["forced_flatten_time"])
        forced_flatten_time = tz.localize(forced_flatten_time)
        minutes_remaining = (forced_flatten_time - current_time).total_seconds() / 60.0
        
        tick_size = CONFIG["tick_size"]
        tick_value = CONFIG["tick_value"]
        if side == "long":
            price_change = current_bar["close"] - entry_price
        else:
            price_change = entry_price - current_bar["close"]
        ticks = price_change / tick_size
        unrealized_pnl = ticks * tick_value * position["quantity"]
        
        logger.warning(f"Flatten Mode Status: {minutes_remaining:.1f} min remaining, "
                      f"P&L: ${unrealized_pnl:+.2f}, Side: {side}, Qty: {position['quantity']}")
    
    # ========================================================================
    # PHASE SEVEN: Integration Priority and Execution Order
    # ========================================================================
    # Critical execution sequence - order matters!
    
    # FIRST - Time-based exit check (highest priority - hard deadline)
    reason, price = check_time_based_exits(symbol, current_bar, position, bar_time)
    if reason:
        # Log specific messages for certain exit types
        if reason == "emergency_forced_flatten":
            logger.critical(SEPARATOR_LINE)
            logger.critical("EMERGENCY FORCED FLATTEN - 8:45 PM UTC FLATTEN WINDOW")
            logger.critical(SEPARATOR_LINE)
        elif reason == "time_based_profit_take":
            logger.critical("4:40 PM - Closing profitable position immediately")
        elif reason == "time_based_loss_cut":
            logger.critical("4:42 PM - Cutting small loss before settlement")
        elif reason == "early_profit_lock":
            logger.warning(f"Phase 10: Closing early profit instead of waiting for target")
        elif reason == "friday_weekend_protection":
            logger.critical(SEPARATOR_LINE)
            logger.critical("FRIDAY 7 PM UTC - CLOSING POSITION TO AVOID WEEKEND GAP RISK")
            logger.critical(SEPARATOR_LINE)
        elif reason == "friday_profit_protection":
            tick_size = CONFIG["tick_size"]
            tick_value = CONFIG["tick_value"]
            if side == "long":
                price_change = current_bar["close"] - entry_price
            else:
                price_change = entry_price - current_bar["close"]
            ticks = price_change / tick_size
            unrealized_pnl = ticks * tick_value * position["quantity"]
            logger.warning(f"Friday 2 PM+ - Taking ${unrealized_pnl:+.2f} profit to avoid weekend risk")
        elif reason == "early_loss_cut":
            tick_size = CONFIG["tick_size"]
            if side == "long":
                current_loss_distance = entry_price - current_bar["close"]
                stop_distance = entry_price - stop_price
                loss_percent = current_loss_distance / stop_distance
            else:
                current_loss_distance = current_bar["close"] - entry_price
                stop_distance = stop_price - entry_price
                loss_percent = current_loss_distance / stop_distance
            logger.warning(f"Time-based early loss cut (3:30 PM+): {loss_percent*100:.1f}% of stop distance")
        elif reason == "flatten_mode_exit":
            profit_ticks = abs((current_bar["close"] - entry_price) / CONFIG["tick_size"]) if side == "long" else abs((entry_price - current_bar["close"]) / CONFIG["tick_size"])
            logger.info(f"Flatten mode exit: profit_ticks={profit_ticks:.1f}")
        
        execute_exit(symbol, price, reason)
        return
    
    # SECOND - Early exit detection (learned from similar losing patterns)
    # Check if current position matches historical deterioration patterns
    if position["active"] and adaptive_manager:
        try:
            # Get current market state for similarity matching
            bars = state[symbol]["bars_1min"]
            market_state = {
                "regime": detect_market_regime(bars, calculate_atr(symbol, CONFIG.get("atr_period", 14)) or 2.0),
                "rsi": calculate_rsi([bar["close"] for bar in bars[-50:]]) if len(bars) >= 50 else 50.0,
                "volume_ratio": bars[-1]["volume"] / sum(b["volume"] for b in bars[-20:]) * 20 if len(bars) >= 20 else 1.0,
                "hour": bar_time.hour,
                "vwap_distance": abs(current_bar["close"] - current_bar.get("vwap", current_bar["close"])),
                "entry_time": position.get("entry_time", bar_time),
                "current_time": bar_time
            }
            
            should_exit = adaptive_manager.should_exit_early(position, market_state, current_bar["close"])
            if should_exit:
                logger.warning("[EXIT RL] Early exit triggered - similar trades deteriorated from here")
                logger.warning(f"  Regime: {market_state['regime']}, RSI: {market_state['rsi']:.1f}")
                execute_exit(symbol, current_bar["close"], "learned_early_exit")
                return
        except Exception as e:
            logger.debug(f"Early exit check failed: {e}")
    
    # THIRD - VWAP target hit check
    target_hit, price = check_target_reached(symbol, current_bar, position, bar_time)
    if target_hit:
        if price == position["target_price"]:
            execute_exit(symbol, price, "target_reached")
            # Track successful target wait
            if bot_status["flatten_mode"]:
                bot_status["target_wait_wins"] += 1
        else:
            # Tightened target
            logger.info("Time-based tightened profit target reached (1:1 R/R after 7 PM UTC)")
            execute_exit(symbol, price, "tightened_target")
        return
    
    # FOURTH - VWAP stop hit check
    stop_hit, price = check_stop_hit(symbol, current_bar, position)
    if stop_hit:
        execute_exit(symbol, price, "stop_loss")
        return
    
    # Check proactive stop (during flatten mode)
    should_close, price = check_proactive_stop(symbol, current_bar, position)
    if should_close:
        logger.warning(f"Proactive stop close: within {CONFIG['proactive_stop_buffer_ticks']} ticks of stop")
        execute_exit(symbol, price, "proactive_stop")
        return
    
    # FIFTH - Advanced Exit RL: Adverse Momentum Detection
    # Check if price momentum has shifted strongly against position
    if position["active"] and adaptive_manager:
        try:
            bars = state[symbol]["bars_1min"]
            recent_bars = [
                {
                    'timestamp': b['timestamp'],
                    'open': b['open'],
                    'high': b['high'],
                    'low': b['low'],
                    'close': b['close'],
                    'volume': b['volume']
                }
                for b in bars[-5:]  # Last 5 bars for momentum analysis
            ]
            
            if len(recent_bars) >= 3:
                adverse_result = adaptive_manager.detect_adverse_momentum(
                    recent_bars=recent_bars,
                    direction=side,
                    entry_price=entry_price,
                    current_price=current_bar["close"],
                    position_size=position["quantity"]
                )
                
                if adverse_result.get('adverse_detected'):
                    severity = adverse_result.get('severity', 'medium')
                    reason = adverse_result.get('reason', 'Adverse momentum detected')
                    logger.warning(f"[EXIT RL] {reason}")
                    logger.warning(f"  Severity: {severity.upper()} - Exiting FAST")
                    execute_exit(symbol, current_bar["close"], "adverse_momentum")
                    return
        except Exception as e:
            logger.debug(f"Adverse momentum check failed: {e}")
    
    # SIXTH - Advanced Exit RL: Volume Exhaustion Detection
    # Check if volume is drying up during profit run (trend exhaustion)
    if position["active"] and adaptive_manager and position.get("breakeven_active"):
        try:
            bars = state[symbol]["bars_1min"]
            recent_bars = [
                {
                    'timestamp': b['timestamp'],
                    'open': b['open'],
                    'high': b['high'],
                    'low': b['low'],
                    'close': b['close'],
                    'volume': b['volume']
                }
                for b in bars[-10:]  # Last 10 bars for volume analysis
            ]
            
            # Calculate current R-multiple
            tick_size = CONFIG["tick_size"]
            initial_risk_ticks = position.get("initial_risk_ticks", 10)
            if side == "long":
                profit_ticks = (current_bar["close"] - entry_price) / tick_size
            else:
                profit_ticks = (entry_price - current_bar["close"]) / tick_size
            r_multiple = profit_ticks / initial_risk_ticks if initial_risk_ticks > 0 else 0
            
            # Calculate average volume for context
            avg_volume = sum(b['volume'] for b in recent_bars) / len(recent_bars) if recent_bars else 1
            
            if len(recent_bars) >= 5 and r_multiple > 2.0:
                exhaustion_result = adaptive_manager.check_volume_exhaustion(
                    recent_bars=recent_bars,
                    r_multiple=r_multiple,
                    avg_volume=avg_volume
                )
                
                if exhaustion_result.get('exhaustion_detected'):
                    reason = exhaustion_result.get('reason', 'Volume exhaustion detected')
                    logger.warning(f"[EXIT RL] {reason}")
                    logger.warning(f"  Volume dropped {exhaustion_result.get('volume_drop_pct', 0):.1f}% during profit run")
                    execute_exit(symbol, current_bar["close"], "volume_exhaustion")
                    return
        except Exception as e:
            logger.debug(f"Volume exhaustion check failed: {e}")
    
    # SEVENTH - Advanced Exit RL: Failed Breakout Detection
    # Check if price hit target band but failed to sustain (weak breakout)
    if position["active"] and adaptive_manager:
        try:
            bars = state[symbol]["bars_1min"]
            recent_bars = [
                {
                    'timestamp': b['timestamp'],
                    'open': b['open'],
                    'high': b['high'],
                    'low': b['low'],
                    'close': b['close'],
                    'volume': b['volume']
                }
                for b in bars[-5:]  # Last 5 bars for breakout analysis
            ]
            
            # Check if target was hit (touched the band)
            target_price = position.get("target_price", 0)
            target_hit = False
            if target_price > 0:
                for bar in recent_bars:
                    if side == "long" and bar['high'] >= target_price:
                        target_hit = True
                    elif side == "short" and bar['low'] <= target_price:
                        target_hit = True
            
            if len(recent_bars) >= 3 and target_hit:
                breakout_result = adaptive_manager.detect_failed_breakout(
                    recent_bars=recent_bars,
                    direction=side,
                    target_hit=target_hit,
                    entry_price=entry_price,
                    current_price=current_bar["close"]
                )
                
                if breakout_result.get('failed_breakout'):
                    reason = breakout_result.get('reason', 'Failed breakout detected')
                    logger.warning(f"[EXIT RL] {reason}")
                    logger.warning(f"  Target hit but price rejected - exit before reversal")
                    execute_exit(symbol, current_bar["close"], "failed_breakout")
                    return
        except Exception as e:
            logger.debug(f"Failed breakout check failed: {e}")
    
    # EIGHTH - Partial exits (happens before breakeven/trailing because it reduces position size)
    check_partial_exits(symbol, current_bar["close"])
    
    # NINTH - Breakeven protection (must activate before trailing)
    check_breakeven_protection(symbol, current_bar["close"])
    
    # TENTH - Trailing stop (only runs if breakeven already active)
    check_trailing_stop(symbol, current_bar["close"])
    
    # ELEVENTH - Market divergence check (tighten stops if fighting momentum)
    if position["active"]:
        is_diverging, divergence_reason = check_market_divergence(symbol, position["side"])
        if is_diverging:
            logger.warning(f"⚠️ DIVERGENCE DETECTED: {divergence_reason}")
            
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
                logger.warning(f"🔒 Tightening stop due to divergence: ${current_stop:.2f} → ${new_stop:.2f}")
                stop_side = "SELL" if side == "long" else "BUY"
                contracts = position["quantity"]
                new_stop_order = place_stop_order(symbol, stop_side, contracts, new_stop)
                
                if new_stop_order:
                    position["stop_price"] = new_stop
                    if new_stop_order.get("order_id"):
                        position["stop_order_id"] = new_stop_order.get("order_id")
    
    # TWELFTH - Time-decay tightening (last priority, gradual adjustment)
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
    
    duration_seconds = (exit_time - position["entry_time"]).total_seconds()
    duration_minutes = duration_seconds / 60.0
    state[symbol]["session_stats"]["trade_durations"].append(duration_minutes)
    
    # Track if this was a forced flatten due to time
    if reason in time_based_reasons:
        state[symbol]["session_stats"]["force_flattened_count"] += 1
    
    # Track after-noon entries
    entry_hour = position["entry_time"].hour
    if entry_hour >= 12:
        state[symbol]["session_stats"]["after_noon_entries"] += 1
        if reason in time_based_reasons:
            state[symbol]["session_stats"]["after_noon_force_flattened"] += 1
    
    logger.info(f"  Position Duration: {duration_minutes:.1f} minutes")


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
            logger.critical(f"🆘 Forced flatten attempt {attempt}/{max_attempts}")
            logger.critical(f"  Position: {position['side'].upper()} {contracts} contracts")
            logger.critical(f"  Exit Price: ${exit_price:.2f}")
            
            # Use market order for maximum urgency
            order = place_market_order(symbol, order_side, contracts)
            
            if order:
                logger.critical(f"  ✓ Order placed - Order ID: {order.get('order_id', 'N/A')}")
                
                # In backtesting, position closes immediately
                # In live trading, wait briefly and verify
                import time
                time.sleep(1)
                
                # Verify position actually closed
                current_position = get_position_quantity(symbol)
                
                if current_position == 0:
                    logger.critical("=" * 80)
                    logger.critical(f"[SUCCESS] FORCED FLATTEN SUCCESSFUL (Attempt {attempt})")
                    logger.critical("=" * 80)
                    return  # SUCCESS - position closed
                else:
                    logger.error(f"  [WARN] Position still shows {current_position} contracts - verifying...")
                    # Might be a delay in reporting, continue to next check
            else:
                logger.error(f"  [FAIL] Order placement FAILED on attempt {attempt}")
            
            # Failed - retry with increasing urgency
            if attempt < max_attempts:
                wait_time = attempt * retry_backoff_base  # 1s, 2s, 3s, 4s delays
                logger.error(f"  Retrying in {wait_time} seconds with increased urgency...")
                time.sleep(wait_time)
        
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
                error_message=f"🆘 CRITICAL: FLATTEN FAILED after {max_attempts} attempts! Position: {position['side'].upper()} {contracts} {symbol}. Entry: ${position['entry_price']:.2f}. P&L: ${unrealized_pnl:.2f}. MANUAL INTERVENTION REQUIRED!",
                error_type="FLATTEN FAILED - URGENT"
            )
        except Exception as e:
            logger.debug(f"Failed to send flatten failure alert: {e}")
        
        return  # Cannot continue - manual intervention needed
    
    # Normal exit handling (non-forced-flatten)
    # Determine exit type based on reason
    exit_type_map = {
        "target_reached": "target",
        "stop_loss": "stop",
        "flatten_mode_exit": "time_flatten",
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
                logger.warning(f"⚡ Volume surge detected ({surge_ratio:.1f}x) - using high urgency exit")
            
            # Override urgency for critical exits
            if reason in ["stop_loss", "proactive_stop", "signal_reversal"]:
                urgency = "high"
                logger.info(f"Critical exit ({reason}) - using high urgency")
            elif reason in ["flatten_mode_exit", "emergency_forced_flatten", "time_based_loss_cut"]:
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
                    import time
                    time.sleep(strategy['timeout'])
                    
                    # Check if filled
                    current_position = get_position_quantity(symbol)
                    if current_position == 0:
                        logger.info(" Passive exit filled")
                        return
                    else:
                        # Not filled, use fallback
                        logger.warning(" Passive exit not filled, using aggressive")
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
            
            if order:
                logger.info(f"Exit order placed: {order.get('order_id')}")
                return
                
        except Exception as e:
            logger.error(f"Error using bid/ask exit optimization: {e}")
            logger.info("Falling back to traditional exit")
    
    # Fallback to traditional exit logic
    is_flatten_mode = bot_status["flatten_mode"] or reason in [
        "flatten_mode_exit", "time_based_profit_take", 
        "time_based_loss_cut", "emergency_forced_flatten"
    ]
    
    if is_flatten_mode:
        logger.info("Using aggressive limit order strategy for flatten")
        execute_flatten_with_limit_orders(symbol, order_side, contracts, exit_price, reason)
    else:
        # Normal exit - use market order
        order = place_market_order(symbol, order_side, contracts)
        if order:
            logger.info(f"Exit order placed: {order.get('order_id')}")


def execute_exit(symbol: str, exit_price: float, reason: str) -> None:
    """
    Execute exit order and update P&L.
    Coordinates exit handling through helper functions.
    
    Args:
        symbol: Instrument symbol
        exit_price: Exit price
        reason: Reason for exit (stop_loss, target_reached, signal_reversal, etc.)
    """
    global adaptive_manager  # Declare at top of function
    
    position = state[symbol]["position"]
    
    if not position["active"]:
        return
    
    exit_time = datetime.now(pytz.timezone(CONFIG["timezone"]))
    
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
    
    # ADAPTIVE EXIT MANAGEMENT - Record trade result for streak tracking
    if adaptive_manager is not None:
        try:
            adaptive_manager.record_trade_result(pnl)
            logger.info(f" STREAK TRACKING: Recorded P&L ${pnl:+.2f} (Recent: {len(adaptive_manager.recent_trades)} trades)")
        except Exception as e:
            logger.debug(f"Streak tracking update skipped: {e}")
    
    # REINFORCEMENT LEARNING - Record outcome to cloud API (shared learning pool)
    try:
        # Check if we have the entry state stored
        if "entry_rl_state" in state[symbol]:
            entry_state = state[symbol]["entry_rl_state"]
            entry_side = state[symbol]["position"]["side"]  # Get the trade side
            
            # Calculate trade duration in minutes
            entry_time = position.get("entry_time")
            duration_minutes = 0
            if entry_time:
                duration = exit_time - entry_time
                duration_minutes = duration.total_seconds() / 60
            
            # Record the outcome to cloud API for shared learning
            save_trade_experience(
                rl_state=entry_state,
                side=entry_side,
                pnl=pnl,
                duration_minutes=duration_minutes,
                execution_data={
                    # Execution quality metrics for RL learning
                    "order_type_used": position.get("order_type_used", "unknown"),
                    "entry_slippage_ticks": abs(position.get("actual_entry_price", 0) - position.get("original_entry_price", 0)) / CONFIG.get("tick_size", 0.25) if position.get("actual_entry_price") and position.get("original_entry_price") else 0,
                    "partial_fill": position.get("quantity", 0) < position.get("original_quantity", 0),
                    "fill_ratio": position.get("quantity", 0) / position.get("original_quantity", 1) if position.get("original_quantity") else 1.0,
                    "exit_reason": reason,
                    "held_full_duration": reason in ["target_hit", "stop_hit"]
                }
            )
            
            logger.info(f" [CLOUD RL] Recorded outcome ${pnl:+.2f} in {duration_minutes:.1f}min to shared learning pool")
            
            # Clean up state
            del state[symbol]["entry_rl_state"]
            if "entry_rl_confidence" in state[symbol]:
                del state[symbol]["entry_rl_confidence"]
        
    except Exception as e:
        logger.debug(f"RL outcome recording failed: {e}")
    
    # ADAPTIVE EXIT LEARNING - Record exit parameters and outcome
    try:
        if adaptive_manager is not None and hasattr(adaptive_manager, 'record_exit_outcome'):
            # Get exit parameters that were used
            if "exit_params_used" in position:
                exit_params = position["exit_params_used"]
                
                # Calculate market regime at exit (don't rely on stored value)
                from adaptive_exits import detect_market_regime
                bars = state[symbol]["bars_1min"]
                atr_value = calculate_atr(symbol, CONFIG.get("atr_period", 14))
                regime = detect_market_regime(bars, atr_value if atr_value else 2.0)
                
                # Calculate trade duration
                entry_time = position.get("entry_time")
                duration_minutes = 0
                if entry_time:
                    duration = exit_time - entry_time
                    duration_minutes = duration.total_seconds() / 60
                
                # Capture current market state for context-aware learning
                try:
                    current_rl_state = capture_rl_state(symbol, position["side"], exit_price)
                    market_state = {
                        'rsi': current_rl_state.get('rsi', 50.0),
                        'volume_ratio': current_rl_state.get('volume_ratio', 1.0),
                        'hour': current_rl_state.get('hour', 12),
                        'day_of_week': current_rl_state.get('day_of_week', 0),
                        'streak': current_rl_state.get('streak', 0),
                        'recent_pnl': current_rl_state.get('recent_pnl', 0.0),
                        'vix': current_rl_state.get('vix', 15.0),
                        'vwap_distance': current_rl_state.get('vwap_distance', 0.0),
                        'atr': current_rl_state.get('atr', exit_params.get('current_atr', 0))
                    }
                except Exception as e:
                    logger.debug(f"Failed to capture market state for exit: {e}")
                    market_state = None
                
                # Record for learning with market context AND ALL TRACKED FEATURES
                adaptive_manager.record_exit_outcome(
                    regime=regime,
                    exit_params=exit_params,
                    trade_outcome={
                        # Basic outcome
                        'pnl': pnl,
                        'duration': duration_minutes,
                        'exit_reason': reason,
                        'side': position["side"],
                        'contracts': position["quantity"],
                        'win': pnl > 0,
                        
                        # === COMPLETE 45-FEATURE TRACKING (matches backtest) ===
                        # Entry context
                        'entry_confidence': position.get("entry_confidence", 0.0),
                        'entry_price': position.get("entry_price", 0.0),
                        
                        # Performance Metrics (from tracking)
                        'mae': position.get("mae", 0.0),
                        'mfe': position.get("mfe", 0.0),
                        'max_r_achieved': position.get("max_r_achieved", 0.0),
                        'min_r_achieved': position.get("min_r_achieved", 0.0),
                        'r_multiple': pnl / (abs(position["entry_price"] - position["stop_price"]) * position["quantity"] * CONFIG.get("tick_value", 12.50) / CONFIG.get("tick_size", 0.25)) if position.get("stop_price") else 0.0,
                        
                        # Exit Strategy State
                        'breakeven_activated': position.get("breakeven_active", False),
                        'trailing_activated': position.get("trailing_stop_active", False),
                        'breakeven_activation_bar': position.get("breakeven_activation_bar", 0),
                        'trailing_activation_bar': position.get("trailing_activation_bar", 0),
                        'bars_until_breakeven': position.get("bars_until_breakeven", 0),
                        'bars_until_trailing': position.get("bars_until_trailing", 0),
                        'time_in_breakeven_bars': position.get("time_in_breakeven_bars", 0),
                        'exit_param_update_count': position.get("exit_param_update_count", 0),
                        'stop_adjustment_count': position.get("stop_adjustment_count", 0),
                        'rejected_partial_count': position.get("rejected_partial_count", 0),
                        'stop_hit': reason == "stop_loss",
                        
                        # Decision History
                        'decision_history': position.get("decision_history", []),
                        'unrealized_pnl_history': position.get("unrealized_pnl_history", []),
                        'peak_unrealized_pnl': position.get("peak_unrealized_pnl", 0.0),
                        'peak_r_multiple': position.get("peak_r_multiple", 0.0),
                        'opportunity_cost': max(0.0, position.get("peak_unrealized_pnl", 0.0) - pnl),
                        
                        # Advanced tracking arrays
                        'exit_param_updates': position.get("exit_param_updates", []),
                        'stop_adjustments': position.get("stop_adjustments", []),
                        
                        # Execution quality
                        'slippage_ticks': position.get("exit_slippage_ticks", 0.0),
                        'commission_cost': position.get("commission_cost", 0.0),
                        'bid_ask_spread_ticks': position.get("exit_bid_ask_spread_ticks", 0.0),
                        
                        # Market context
                        'session': position.get("entry_session", "NY"),
                        'volume_at_exit': state[symbol]["bars_1min"][-1].get("volume", 0) if len(state[symbol]["bars_1min"]) > 0 else 0,
                        'volatility_regime_change': position.get("volatility_regime_at_entry", "NORMAL") != market_state.get("volatility_regime", "NORMAL") if market_state else False,
                        
                        # Drawdown tracking
                        'profit_drawdown_from_peak': position.get("profit_drawdown_from_peak", 0.0),
                        'max_drawdown_percent': position.get("max_drawdown_percent", 0.0),
                        'drawdown_bars': position.get("drawdown_bars", 0),
                        
                        # Volatility during trade
                        'avg_atr_during_trade': sum(position.get("atr_samples", [])) / len(position.get("atr_samples", [1])),
                        'atr_change_percent': ((position.get("atr_samples", [0])[-1] - position.get("entry_atr", 1)) / position.get("entry_atr", 1) * 100.0) if position.get("atr_samples") and position.get("entry_atr", 0) > 0 else 0.0,
                        'high_volatility_bars': position.get("high_volatility_bars", 0),
                        
                        # Trade context
                        'duration_bars': position.get("duration_bars", 0),
                        'trade_number_in_session': position.get("trade_number_in_session", 0),
                        'wins_in_last_5_trades': position.get("wins_in_last_5_trades", 0),
                        'losses_in_last_5_trades': position.get("losses_in_last_5_trades", 0),
                        
                        # Market conditions (for backtest compatibility)
                        'vix': market_state.get('vix', 15.0) if market_state else 15.0,
                        'cumulative_pnl_before_trade': state[symbol].get("daily_pnl", 0.0) - pnl,  # P&L before this trade
                        
                        # NEW: Temporal features for pattern learning
                        'entry_hour': entry_time.hour if entry_time else 12,
                        'entry_minute': entry_time.minute if entry_time else 0,
                        'exit_hour': exit_time.hour,
                        'exit_minute': exit_time.minute,
                        'day_of_week': exit_time.weekday(),  # 0=Monday, 6=Sunday
                        'bars_held': position.get("duration_bars", 0),  # Already tracked
                        
                        # Time-of-day exit tracking
                        'exit_hour': exit_time.hour,
                        'exit_minute': exit_time.minute,
                        'held_through_sessions': (exit_time - position.get("entry_time", exit_time)).total_seconds() > 14400 if position.get("entry_time") else False,  # >4 hours
                        'minutes_until_close': max(0.0, (21 * 60) - (exit_time.hour * 60 + exit_time.minute)) if exit_time.hour < 21 else 0.0,
                    },
                    market_state=market_state
                )
                
                logger.info(f"[EXIT RL] Learned {regime} exit -> ${pnl:+.2f} in {duration_minutes:.1f}min with market context")
                
                # === EXIT RL FEATURE 10: EXIT EFFICIENCY ANALYSIS ===
                # Analyze MAE/MFE patterns periodically (every 5 trades)
                exit_count = len(adaptive_manager.exit_experiences) if hasattr(adaptive_manager, 'exit_experiences') else 0
                if exit_count > 0 and exit_count % 5 == 0:
                    try:
                        efficiency_results = adaptive_manager.analyze_mae_mfe_patterns()
                        if efficiency_results:
                            logger.info(f"[EXIT EFFICIENCY] Analysis complete:")
                            logger.info(f"   Analyzed {efficiency_results.get('total_analyzed', 0)} trades")
                            logger.info(f"   Avg MAE: ${efficiency_results.get('avg_mae', 0):.2f}, Avg MFE: ${efficiency_results.get('avg_mfe', 0):.2f}")
                            capture_rate = efficiency_results.get('capture_rate', 0)
                            if capture_rate > 0:
                                logger.info(f"   Capture Rate: {capture_rate:.1%} (how much of MFE was captured)")
                    except Exception as e:
                        logger.debug(f"Exit efficiency analysis failed: {e}")
    except Exception as e:
        logger.debug(f"Exit learning failed: {e}")
    
    # Log time-based exits with detailed audit trail
    time_based_reasons = [
        "flatten_mode_exit", "time_based_profit_take", "time_based_loss_cut",
        "emergency_forced_flatten", "tightened_target", "early_loss_cut",
        "proactive_stop", "early_profit_lock", "friday_weekend_protection",
        "friday_profit_protection", "trailing_stop_failure_emergency"
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
            "flatten_mode_exit": "Flatten mode aggressive exit",
            "time_based_profit_take": "8:40 PM UTC profit lock before settlement",
            "time_based_loss_cut": "8:42 PM UTC small loss cut before settlement",
            "emergency_forced_flatten": "8:45 PM UTC flatten before maintenance",
            "tightened_target": "7 PM UTC tightened target (1:1 R/R)",
            "early_loss_cut": "7:30 PM UTC early loss cut (<75% stop)",
            "proactive_stop": "Proactive stop (within 2 ticks)",
            "early_profit_lock": "Early profit lock in flatten mode",
            "friday_weekend_protection": "Friday 7 PM UTC weekend protection",
            "friday_profit_protection": "Friday 6 PM UTC profit protection",
            "trailing_stop_failure_emergency": "Trailing stop update failed - emergency exit"
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
    logger.info(f"Trades today: {state[symbol]['daily_trade_count']}/{CONFIG['max_trades_per_day']}")
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
        summary_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trade_summary.json')
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
        "target_price": None,
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
    logger.info("  ✓ Position state saved to disk (FLAT)")
    
    # Update state status
    state[symbol]["bot_status_message"] = "Monitoring..."
    
    # Store exit signal info for dashboard
    exit_pnl = state[symbol].get("daily_pnl", 0.0)
    pnl_sign = "+" if exit_pnl >= 0 else ""
    state[symbol]["last_signal_type"] = f"EXIT @ ${exit_price:.2f} ({pnl_sign}${pnl:.2f})"
    state[symbol]["last_signal_timestamp"] = get_current_time().timestamp()
    
    # Update dashboard to show flat position
    if dashboard:
        dashboard.update_symbol_data(symbol, {
            "position": "FLAT",
            "position_qty": 0,
            "position_side": None,
            "status": "Monitoring..."
        })
        # Update P&L
        dashboard.update_symbol_data(symbol, {
            "pnl_today": state[symbol].get("daily_pnl", 0.0)
        })
        dashboard.display()


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
    Check if VWAP should reset at 11 PM UTC (futures market day start).
    For 24/5 trading: VWAP resets at 11 PM when futures trading day begins (Sunday opening).
    
    Args:
        symbol: Instrument symbol
        current_time: Current datetime in UTC
    """
    current_date = current_time.date()
    vwap_reset_time = datetime_time(23, 0)  # 11 PM UTC - futures trading day starts
    
    # Check if we've crossed 11 PM UTC on a new day
    if state[symbol]["vwap_day"] is None:
        # First run - initialize VWAP day
        state[symbol]["vwap_day"] = current_date
        logger.info(f"Trading day initialized: {current_date}")
        return
    
    # If it's a new day and we're past 11 PM UTC, reset VWAP
    # OR if it's the same calendar day but we just crossed 11 PM UTC
    last_reset_date = state[symbol]["vwap_day"]
    crossed_reset_time = current_time.time() >= vwap_reset_time
    
    # New trading day starts at 11 PM UTC, so check if we've moved to a new VWAP session
    if crossed_reset_time and last_reset_date != current_date:
        perform_vwap_reset(symbol, current_date, current_time)


def perform_vwap_reset(symbol: str, new_date: Any, reset_time: datetime) -> None:
    """
    Perform VWAP reset at 11 PM UTC daily (futures trading day start).
    
    Args:
        symbol: Instrument symbol
        new_date: The new VWAP date
        reset_time: Time of the reset
    """
    logger.info(SEPARATOR_LINE)
    logger.info(f"DAILY RESET at {reset_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"Futures trading day start (11 PM UTC) - New trading day: {new_date}")
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
    For 24/5 trading: Resets at 11 PM UTC (futures trading day start).
    
    Args:
        symbol: Instrument symbol
        current_time: Current datetime in UTC
    """
    current_date = current_time.date()
    vwap_reset_time = datetime_time(23, 0)  # 11 PM UTC - futures trading day starts
    
    # If we have a trading day stored and it's different from current date
    if state[symbol]["trading_day"] is not None:
        if state[symbol]["trading_day"] != current_date:
            # Reset daily counters at 11 PM UTC (same as VWAP reset)
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
    
    # Reset flatten mode flag for new day
    bot_status["flatten_mode"] = False
    
    logger.info("Daily reset complete - Ready for trading")
    logger.info("(VWAP reset handled separately at 9:30 AM)")
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
    Used for Recovery Mode and Confidence Trading.
    
    Args:
        symbol: Instrument symbol
    
    Returns:
        Tuple of (is_approaching, reason, severity_level)
        - is_approaching: True if at RECOVERY_APPROACHING_THRESHOLD (80%) or more of daily loss limit
        - reason: Description of what limit is being approached
        - severity_level: 0.0-1.0 indicating how close to failure (0.8 = at 80%, 1.0 = at 100%)
    """
    daily_loss_limit = CONFIG.get("daily_loss_limit", 1000.0)
    if daily_loss_limit > 0 and state[symbol]["daily_pnl"] <= -daily_loss_limit * RECOVERY_APPROACHING_THRESHOLD:
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


def get_recovery_confidence_threshold(severity_level: float) -> float:
    """
    Calculate required confidence threshold for recovery mode.
    Higher severity = higher confidence requirement.
    Confidence NEVER goes below user's initial threshold - it only increases.
    
    Uses percentage-based scaling to work with any user setting (50%-90%).
    CONSERVATIVE: At critical levels (95%+), uses nearly all headroom for maximum selectivity.
    
    Args:
        severity_level: How close to failure (0.8 = at 80%, 1.0 = at 100%)
    
    Returns:
        Required confidence threshold (0.0-1.0), never below user's initial setting
    """
    # Base threshold is user's setting - this is the MINIMUM
    base_threshold = CONFIG.get("rl_confidence_threshold", 0.65)
    
    # Calculate how much headroom we have to increase (distance to 100%)
    headroom = 1.0 - base_threshold
    
    # Scale confidence based on severity using percentage of headroom
    # CONSERVATIVE SCALING - uses more headroom as danger increases
    # At 80% severity: use 30% of headroom (moderate increase)
    # At 90% severity: use 60% of headroom (significant increase)
    # At 95% severity: use 85% of headroom (very selective)
    # At 98%+ severity: use 95% of headroom (maximum selectivity - only absolute best signals)
    if severity_level >= 0.98:
        increase_multiplier = 0.95  # Use 95% of headroom - extremely selective
    elif severity_level >= 0.95:
        increase_multiplier = 0.85  # Use 85% of headroom - very selective
    elif severity_level >= 0.90:
        increase_multiplier = 0.60  # Use 60% of headroom - selective
    elif severity_level >= 0.80:
        increase_multiplier = 0.30  # Use 30% of headroom - moderately selective
    else:
        increase_multiplier = 0.0  # Normal operation - no increase
    
    # Calculate new threshold: base + (percentage of available headroom)
    dynamic_threshold = base_threshold + (headroom * increase_multiplier)
    
    # Cap at 0.98 (never require 100% - impossible to achieve)
    return min(0.98, dynamic_threshold)


def check_tick_timeout(current_time: datetime) -> Tuple[bool, Optional[str]]:
    """
    Check if data feed has timed out.
    
    Args:
        current_time: Current datetime in UTC
    
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
    24/5 trading - only stop for maintenance window and weekends.
    
    Args:
        current_time: Current datetime in UTC
    
    Returns:
        Tuple of (is_safe, reason)
    """
    # Check if emergency stop is active
    if bot_status["emergency_stop"]:
        return False, f"Emergency stop active: {bot_status['stop_reason']}"
    
    # Check for weekend (Saturday + Sunday before 23:00 UTC)
    if current_time.weekday() == 5:  # Saturday - always closed
        if bot_status["trading_enabled"]:
            logger.debug(f"Saturday detected - market closed")
            bot_status["trading_enabled"] = False
            bot_status["stop_reason"] = "weekend"
        return False, "Weekend - market closed"
    
    if current_time.weekday() == 6:  # Sunday
        if current_time.time() < datetime_time(23, 0):  # Before 23:00 UTC Sunday
            if bot_status["trading_enabled"]:
                logger.debug(f"Sunday before 23:00 UTC - market closed")
                bot_status["trading_enabled"] = False
                bot_status["stop_reason"] = "weekend"
            return False, "Weekend - market closed (opens 23:00 UTC)"
    
    # Check for futures maintenance window (21:00-22:00 UTC Monday-Thursday)
    if current_time.weekday() < 4:  # Monday through Thursday only
        maintenance_start = datetime_time(21, 0)  # 21:00 UTC
        maintenance_end = datetime_time(22, 0)    # 22:00 UTC
        if maintenance_start <= current_time.time() < maintenance_end:
            if bot_status["trading_enabled"]:
                logger.debug(f"Maintenance window - disabling trading")
                bot_status["trading_enabled"] = False
                bot_status["stop_reason"] = "maintenance"
            return False, "Maintenance window"
    
    # Re-enable trading after maintenance/weekend
    if not bot_status["trading_enabled"]:
        logger.debug(f"Re-enabling trading - market open at {current_time}")
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
    
    # Check trade limits and emergency stops
    is_safe, reason = check_trade_limits(current_time)
    if not is_safe:
        return False, reason
    
    # Daily loss limit DISABLED for backtesting
    # is_safe, reason = check_daily_loss_limit(symbol)
    # if not is_safe:
    #     return False, reason
    
    # Check if approaching failure (Confidence Trading & Recovery Mode)
    is_approaching, approach_reason, severity = check_approaching_failure(symbol)
    if is_approaching:
        # Check which safety mode is enabled
        confidence_trading_enabled = CONFIG.get("confidence_trading", False)
        recovery_mode_enabled = CONFIG.get("recovery_mode", False)
        
        # BOTH MODES USE SAME SCALING LOGIC (higher confidence + fewer contracts as approaching limit)
        # DIFFERENCE: Confidence Trading STOPS at limit, Recovery Mode CONTINUES
        
        if confidence_trading_enabled or recovery_mode_enabled:
            # AUTO-SCALE confidence based on severity (same for both modes)
            required_confidence = get_recovery_confidence_threshold(severity)
            
            # Determine which mode is active for logging
            mode_name = "RECOVERY MODE" if recovery_mode_enabled else "CONFIDENCE TRADING"
            stop_reason = "recovery_mode" if recovery_mode_enabled else "confidence_trading"
            
            if bot_status.get("stop_reason") != stop_reason:
                logger.warning("=" * 80)
                logger.warning(f"{mode_name}: APPROACHING LIMITS - SCALING DOWN")
                logger.warning(f"Reason: {approach_reason}")
                logger.warning(f"Severity: {severity*100:.1f}%")
                logger.warning(f"Required confidence DYNAMICALLY increased to {required_confidence*100:.1f}%")
                logger.warning("Bot will ONLY take highest-confidence signals")
                logger.warning("Position size will be dynamically reduced")
                if recovery_mode_enabled:
                    logger.warning("⚠️ RECOVERY MODE: Will continue trading even if limit hit")
                else:
                    logger.warning("⚠️ CONFIDENCE TRADING: Will STOP trading if limit hit")
                logger.warning("=" * 80)
                bot_status["stop_reason"] = stop_reason
                
                # Send recovery/confidence mode activation alert
                try:
                    notifier = get_notifier()
                    notifier.send_error_alert(
                        error_message=f"{mode_name} ACTIVATED. Severity: {severity*100:.1f}%. {approach_reason}. Trading scaled down to highest-confidence signals only.",
                        error_type=mode_name
                    )
                except Exception as e:
                    logger.debug(f"Failed to send {mode_name} alert: {e}")
            
            # Store threshold and severity for use in signal evaluation and position sizing
            bot_status["recovery_confidence_threshold"] = required_confidence
            bot_status["recovery_severity"] = severity
            
            # CRITICAL: Check if daily limit actually HIT (severity >= 1.0)
            # Confidence Trading STOPS, Recovery Mode CONTINUES
            if severity >= 1.0 and confidence_trading_enabled and not recovery_mode_enabled:
                logger.error("=" * 80)
                logger.error("🛑 CONFIDENCE TRADING: DAILY LIMIT HIT - STOPPING TRADING")
                logger.error(f"Daily Loss: ${state[symbol]['daily_pnl']:.2f}")
                logger.error(f"Daily Limit: ${-CONFIG.get('daily_loss_limit', 1000):.2f}")
                logger.error("Bot will STOP making new trades until daily reset at 11 PM UTC")
                logger.error("Bot continues running and monitoring - will resume after maintenance hour")
                logger.error("=" * 80)
                bot_status["trading_enabled"] = False
                bot_status["stop_reason"] = "confidence_trading_limit_hit"
                
                # Close any active position when limit hit
                if state[symbol]["position"]["active"]:
                    position = state[symbol]["position"]
                    entry_price = position.get("entry_price", 0)
                    side = position.get("side", "long")
                    current_price = state[symbol]["bars"][-1]["close"] if state[symbol]["bars"] else entry_price
                    
                    logger.error("=" * 80)
                    logger.error("CONFIDENCE TRADING: Closing active position due to daily limit")
                    logger.error(f"Position: {side.upper()} {position['quantity']} @ ${entry_price:.2f}")
                    logger.error(f"Current Price: ${current_price:.2f}")
                    logger.error("=" * 80)
                    
                    # Get smart flatten price
                    flatten_price = get_flatten_price(symbol, side, current_price)
                    
                    # Close the position
                    handle_exit_orders(symbol, position, flatten_price, "confidence_trading_limit_protection")
                    
                    logger.info(f"Position closed at ${flatten_price:.2f} - Confidence Trading limit hit")
                
                return False, "Confidence Trading: Daily limit hit - trading stopped until next session (11 PM UTC reset)"
            
            # SMART POSITION MANAGEMENT: If severity is critical (95%+), consider closing losing positions
            if severity >= 0.95 and state[symbol]["position"]["active"]:
                position = state[symbol]["position"]
                entry_price = position.get("entry_price", 0)
                current_price = state[symbol]["bars"][-1]["close"] if state[symbol]["bars"] else entry_price
                
                # Check if position is losing
                is_losing = (
                    (position["side"] == "long" and current_price < entry_price) or
                    (position["side"] == "short" and current_price > entry_price)
                )
                
                if is_losing:
                    # Calculate how much the position is losing
                    side = position.get("side", "long")
                    if side == "long":
                        position_pnl = (current_price - entry_price) * position["quantity"]
                    else:
                        position_pnl = (entry_price - current_price) * position["quantity"]
                    
                    logger.warning("=" * 80)
                    logger.warning("SMART POSITION MANAGEMENT: Critical severity (95%+) with losing position")
                    logger.warning(f"Position: {side.upper()} {position['quantity']} @ ${entry_price:.2f}")
                    logger.warning(f"Current Price: ${current_price:.2f}")
                    logger.warning(f"Position Loss: ${position_pnl:.2f}")
                    logger.warning("Closing position to prevent account failure")
                    logger.warning("=" * 80)
                    
                    # Get smart flatten price
                    flatten_price = get_flatten_price(symbol, side, current_price)
                    
                    # Close the losing position immediately
                    handle_exit_orders(symbol, position, flatten_price, "critical_loss_protection")
                    
                    logger.info(f"Losing position closed at ${flatten_price:.2f} in safety mode")
                else:
                    # Position is winning, just flag for aggressive management
                    logger.info("Position is profitable - maintaining with aggressive exit management")
                    bot_status["aggressive_exit_mode"] = True
            
            # Don't stop trading - both modes continue with scaled-down logic
        else:
            # NEITHER MODE ENABLED: STOP trading until next session (after maintenance)
            # Bot stays running but doesn't execute new trades until daily reset at 11 PM UTC
            logger.warning("=" * 80)
            logger.warning("⚠️ LIMITS REACHED - STOPPING TRADING UNTIL NEXT SESSION")
            logger.warning(f"Reason: {approach_reason}")
            logger.warning(f"Severity: {severity*100:.1f}%")
            logger.warning("Bot will STOP making new trades until daily reset at 11 PM UTC")
            logger.warning("Bot continues running and monitoring - will resume after maintenance hour")
            logger.warning("To scale down when approaching limits, enable Confidence Trading or Recovery Mode")
            logger.warning("=" * 80)
            bot_status["trading_enabled"] = False
            bot_status["stop_reason"] = "daily_limits_reached"
            
            # SMART POSITION CLOSING: Close any active position when daily limit reached
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
                logger.warning("SMART POSITION MANAGEMENT: Closing active position due to daily limit")
                logger.warning(f"Position: {side.upper()} {position['quantity']} @ ${entry_price:.2f}")
                logger.warning(f"Current Price: ${current_price:.2f}")
                logger.warning(f"Position P&L: ${position_pnl:.2f}")
                logger.warning("Executing smart exit to minimize additional losses")
                logger.warning("=" * 80)
                
                # Get smart flatten price (includes buffer to ensure fill)
                flatten_price = get_flatten_price(symbol, side, current_price)
                
                # Close the position
                handle_exit_orders(symbol, position, flatten_price, "daily_limit_protection")
                
                logger.info(f"Position closed at ${flatten_price:.2f} to protect from further losses")
            
            return False, "Daily limits reached - trading stopped until next session (11 PM UTC reset)"
    else:
        # Not approaching failure - clear any safety mode that was set
        if bot_status.get("stop_reason") in ["daily_limits_reached", "recovery_mode", "confidence_trading", "confidence_trading_limit_hit"]:
            logger.info("=" * 80)
            logger.info("SAFE ZONE: Back to normal operation")
            logger.info("Bot has moved away from failure thresholds")
            logger.info("Returning to user's initial settings")
            logger.info("=" * 80)
            bot_status["trading_enabled"] = True
            bot_status["stop_reason"] = None
            bot_status["recovery_confidence_threshold"] = None
            bot_status["recovery_severity"] = None
    
    # Check tick timeout
    is_safe, reason = check_tick_timeout(current_time)
    if not is_safe:
        return False, reason
    
    return True, None


def check_no_overnight_positions(symbol: str) -> None:
    """
    Phase Eleven: Critical safety check - ensure NO positions past 5 PM.
    This prevents gap risk and TopStep evaluation issues.
    
    Args:
        symbol: Instrument symbol
    """
    if not state[symbol]["position"]["active"]:
        return  # No position, all good
    
    tz = pytz.timezone(CONFIG["timezone"])
    current_time = datetime.now(tz)
    
    # Critical: If it's past 5 PM and we still have a position, this is a SERIOUS ERROR
    if current_time.time() >= CONFIG["shutdown_time"]:
        critical_msg = f"POSITION PAST 9 PM UTC - Emergency close required for {symbol}"
        log_critical_error(critical_msg)
        
        logger.critical("=" * 70)
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
    24/5 trading - all logic uses UTC (matches ES futures schedule and data).
    
    **AZURE-FIRST DESIGN**: Checks Azure time service first for:
    - Maintenance windows (Mon-Thu 21:00-22:00 UTC, Fri 21:00 - Sun 23:00 UTC)
    - Economic events (FOMC/NFP/CPI blocking)
    - Single source of truth for all time-based decisions
    
    Falls back to local time logic if Azure unreachable.
    
    Args:
        dt: Datetime to check (defaults to current time - live or backtest)
            Can be naive (assumed UTC) or timezone-aware (converted to UTC).
    
    Returns:
        Trading state:
        - 'entry_window': Market open, ready to trade
        - 'flatten_mode': 20:45-21:00 UTC, close positions before maintenance
        - 'closed': Market closed (flatten all positions immediately)
        - 'event_block': Economic event active (FOMC/NFP/CPI)
    """
    # AZURE-FIRST: Try cloud time service (unless in backtest mode)
    if backtest_current_time is None:  # Live mode only
        azure_state = bot_status.get("azure_trading_state")
        if azure_state:
            # Use cached Azure state (updated every 30s by check_azure_time_service)
            return azure_state
    
    # FALLBACK: Local time logic (backtest mode or Azure unreachable)
    # Get current time in UTC
    if dt is None:
        dt = get_current_time()
    
    # Convert to UTC (standardize)
    if dt.tzinfo is None:
        # If naive datetime, assume it's UTC
        dt = pytz.UTC.localize(dt)
    
    # Convert to UTC
    utc_time = dt.astimezone(pytz.UTC)
    
    weekday = utc_time.weekday()  # 0=Monday, 6=Sunday
    current_time = utc_time.time()
    
    # ES Futures Hours (UTC):
    # Sunday 23:00 - Friday 21:00 (with daily 21:00-22:00 maintenance Mon-Thu)
    
    # CLOSED: Saturday (all day)
    if weekday == 5:  # Saturday
        return 'closed'
    
    # CLOSED: Sunday before 23:00 UTC (opens AT 23:00 UTC exactly)
    if weekday == 6 and current_time < datetime_time(23, 0):
        return 'closed'
    
    # CLOSED: Friday at/after 21:00 UTC (closes AT 21:00 UTC exactly - weekend starts)
    if weekday == 4 and current_time >= datetime_time(21, 0):
        return 'closed'
    
    # Get configured trading times from CONFIG (supports 24/5 futures)
    flatten_time = CONFIG.get("flatten_time", datetime_time(20, 45))
    forced_flatten_time = CONFIG.get("forced_flatten_time", datetime_time(21, 0))
    
    # CLOSED: Daily maintenance (21:00-22:00 UTC, Monday-Thursday)
    if weekday < 4:  # Monday-Thursday
        if forced_flatten_time <= current_time < datetime_time(22, 0):
            return 'closed'  # Daily settlement period
    
    # FLATTEN MODE: 15 minutes before daily maintenance
    # Only flatten between 20:45-21:00 UTC (15 min before maintenance)
    # Not after 22:00 UTC (that's when the next session starts!)
    if flatten_time <= current_time < forced_flatten_time:
        return 'flatten_mode'
    
    # ENTRY WINDOW: Market open, ready to trade
    # For 24/5 futures, we're in entry window if:
    # - Between 23:00 UTC Sunday and 20:45 UTC next day (Mon-Thu)
    # - Between 23:00 UTC and 21:00 UTC Friday
    # - NOT in closed/flatten periods above
    return 'entry_window'


# ============================================================================
# PHASE FIFTEEN & SIXTEEN: Timezone Handling and Time-Based Logging
# ============================================================================

def validate_timezone_configuration() -> None:
    """
    Phase Fifteen: Validate timezone configuration on bot startup.
    Ensures timezone is set to UTC (matches ES futures schedule and historical data).
    """
    tz = pytz.timezone(CONFIG["timezone"])
    current_time = datetime.now(tz)
    
    logger.info(SEPARATOR_LINE)
    logger.info("TIMEZONE CONFIGURATION VALIDATION")
    logger.info(SEPARATOR_LINE)
    logger.info(f"Configured Timezone: {CONFIG['timezone']} (UTC)")
    logger.info(f"Current Time (UTC): {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"UTC Offset: {current_time.strftime('%z')}")
    
    # Verify it's actually UTC
    if CONFIG["timezone"] != "UTC":
        logger.warning("⚠️  WARNING: Timezone is not UTC!")
        logger.warning("Bot expects UTC to match ES futures schedule and historical data")
        logger.warning("Please set timezone='UTC' in config.py")
    
    # System time comparison (use UTC for consistency)
    system_time = datetime.now(pytz.UTC)
    utc_time_now = datetime.now(pytz.UTC)
    time_diff = abs((current_time.replace(tzinfo=None) - utc_time_now.replace(tzinfo=None)).total_seconds())
    
    if time_diff > 60:
        logger.warning(f"System time differs from UTC by {time_diff:.0f} seconds")
        logger.warning("All trading decisions use UTC - system time is informational only")
    
    logger.info(SEPARATOR_LINE)


def log_time_based_action(action: str, reason: str, details: Optional[Dict[str, Any]] = None) -> None:
    """
    Phase Sixteen: Log all time-based actions with timestamp and reason.
    Creates audit trail for reviewing time-based rule performance.
    
    Args:
        action: Type of action (e.g., 'entry_blocked', 'flatten_activated', 'position_closed')
        reason: Human-readable reason for the action
        details: Optional dictionary of additional details
    """
    tz = pytz.timezone(CONFIG["timezone"])
    timestamp = datetime.now(tz)
    
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

4. Friday-specific backtesting:
   - Enforce no new trades after 1 PM Friday
   - Execute forced close at 3 PM Friday
   - Measure weekend gap impact on positions that would have been held

5. DST transition testing:
   - Test bot behavior on DST change days (March and November)
   - Ensure time checks still work correctly during "spring forward" and "fall back"

Phase Eighteen: Monitoring During Flatten Window (4:45 PM - 5:00 PM)

This is the highest-risk 15-minute window requiring active monitoring if possible:

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

TIME WINDOWS (All times UTC - 24/5 Futures Trading):
- Saturday: CLOSED - Market closed for weekend
- Sunday before 11:00 PM: CLOSED - Waiting for futures open
- Sunday 11:00 PM: MARKET OPEN - Trading resumes for the week
- 11:00 PM - 8:45 PM (next day): ENTRY WINDOW - Full trading allowed 24 hours (Mon-Thu)
- 8:45 PM - 9:00 PM: FLATTEN MODE - Close positions (15 min before maintenance)
- 9:00 PM - 10:00 PM: MAINTENANCE - Daily settlement (Mon-Thu), market closed
- Friday 8:45 PM - 9:00 PM: FLATTEN MODE - Close before weekend
- Friday 9:00 PM onwards: WEEKEND - Market closed until Sunday 11:00 PM

FLATTEN SCHEDULE (preserves 24-hour trading):
- Monday-Thursday: Flatten 4:45-5:00 PM (15 min before daily maintenance)
- Friday: Flatten 4:45-5:00 PM (before weekend close)
- During flatten mode: Aggressive closing, no new entries
- After 9:00 PM UTC: Maintenance window (Mon-Thu) or weekend (Fri-Sun)

DAILY RESETS:
- 11:00 PM UTC: Daily session opens Sunday (after weekend), continues 24/5
- 1:30 PM UTC: VWAP reset (stock market alignment for equity indexes)
- Daily counters reset at 11 PM UTC when new session starts

CRITICAL SAFETY RULES (24/5 FUTURES):
1. FLATTEN BEFORE MAINTENANCE - Close by 8:45 PM UTC daily (15 min buffer before 9 PM UTC)
2. NO WEEKEND POSITIONS - Force close by 8:45 PM UTC Friday (before 9 PM UTC weekend close)
3. MAINTENANCE WINDOW - Market closed 9-10 PM UTC Mon-Thu for settlement
4. TIMEZONE ENFORCEMENT - All decisions use UTC
5. DST AWARENESS - Not needed - UTC is timezone-agnostic
6. AUDIT TRAIL - Every time-based action logged with timestamp and reason

WHY THIS MATTERS FOR PROP FIRMS:
TopStep's rules are designed to fail traders who don't respect:
- Daily settlement (9 PM UTC reset Mon-Thu, maintenance window)
- Overnight gap exposure
- Weekend event risk
- Daily loss limits (restart at 9 PM UTC, not midnight)

By building time constraints into core logic, you protect against:
- Gap risk from overnight news (Asia/Europe markets, economic data)
- Weekend geopolitical events (can't control, can't trade out)
- Settlement skew manipulation (institutional games in final 30 seconds)
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
    global event_loop, timer_manager, bid_ask_manager, dashboard
    
    # Get list of trading symbols (multi-symbol support)
    if "instruments" in CONFIG and CONFIG["instruments"]:
        trading_symbols = CONFIG["instruments"]
    else:
        trading_symbol = symbol_override if symbol_override else CONFIG["instrument"]
        trading_symbols = [trading_symbol]
    
    # Initialize dashboard FIRST (before any logging)
    dashboard = get_dashboard(trading_symbols, CONFIG)
    dashboard.start()  # This clears screen and initializes dashboard display
    
    # Update dashboard with initial bot data
    initial_account = CONFIG.get("account_size", 50000)
    dashboard.update_bot_data({
        "account_balance": initial_account,
        "max_contracts": CONFIG.get("max_contracts", 3),
        "daily_loss_limit": CONFIG.get("daily_loss_limit", 1000),
        "confidence_threshold": CONFIG.get("confidence_threshold", 65),
        "recovery_mode": CONFIG.get("recovery_mode", False),
        "confidence_trading": CONFIG.get("confidence_trading", False),
    })
    
    # Initial display
    dashboard.display()
    
    # Dashboard is now active - all status shown via dashboard only
    logger.info(SEPARATOR_LINE)
    
    # Use first symbol as primary
    primary_symbol = trading_symbols[0]
    
    # Log symbol specifications if loaded
    if SYMBOL_SPEC:
        logger.info(f"[{primary_symbol}] Symbol: {SYMBOL_SPEC.name} ({SYMBOL_SPEC.symbol})")
        logger.info(f"[{primary_symbol}]   Tick Value: ${SYMBOL_SPEC.tick_value:.2f} | Tick Size: ${SYMBOL_SPEC.tick_size}")
        logger.info(f"[{primary_symbol}]   Slippage: {SYMBOL_SPEC.typical_slippage_ticks} ticks | Volatility: {SYMBOL_SPEC.volatility_factor}x")
        logger.info(f"[{primary_symbol}]   Trading Hours: {SYMBOL_SPEC.session_start} - {SYMBOL_SPEC.session_end} UTC")
    
    # Display operating mode
    if CONFIG.get('shadow_mode', False):
        logger.info(f"[{primary_symbol}] Mode: 📊 SIGNAL-ONLY MODE (Manual Trading)")
        logger.info(f"[{primary_symbol}] ⚠️  Signal mode: Shows trading signals without executing trades")
    else:
        logger.info(f"[{primary_symbol}] Mode: LIVE TRADING")
    
    logger.info(f"[{primary_symbol}] Instruments: {', '.join(trading_symbols)}")
    logger.info(f"[{primary_symbol}] Entry Window: {CONFIG['entry_start_time']} - {CONFIG['entry_end_time']} UTC")
    logger.info(f"[{primary_symbol}] Flatten Mode: {CONFIG['flatten_time']} UTC")
    logger.info(f"[{primary_symbol}] Force Close: {CONFIG['forced_flatten_time']} UTC")
    logger.info(f"[{primary_symbol}] Shutdown: {CONFIG['shutdown_time']} UTC")
    logger.info(f"[{primary_symbol}] Max Contracts: {CONFIG['max_contracts']}")
    logger.info(f"[{primary_symbol}] Max Trades/Day: {CONFIG['max_trades_per_day']}")
    logger.info(f"[{primary_symbol}] Position Sizing: Fixed at {CONFIG['max_contracts']} contracts")
    logger.info(f"[{primary_symbol}] Daily Loss Limit: ${CONFIG['daily_loss_limit']}")
    logger.info(SEPARATOR_LINE)
    
    # Phase Fifteen: Validate timezone configuration
    validate_timezone_configuration()
    
    # Initialize bid/ask manager
    logger.info(f"[{primary_symbol}] Initializing bid/ask manager...")
    bid_ask_manager = BidAskManager(CONFIG)
    
    # Initialize adaptive exit manager for learned exit patterns
    logger.info(f"[{primary_symbol}] Initializing adaptive exit manager with CLOUD RL...")
    from adaptive_exits import AdaptiveExitManager
    adaptive_manager = AdaptiveExitManager(
        config=CONFIG,
        experience_file='data/local_experiences/exit_experiences_v2.json',  # Local v2 format
        cloud_api_url=CLOUD_ML_API_URL  # Use cloud API for shared experiences
    )
    logger.info(f"[{primary_symbol}]   ✅ Loaded {len(adaptive_manager.exit_experiences):,} exit experiences from {'CLOUD' if adaptive_manager.use_cloud else 'LOCAL'}")
    
    # Initialize broker (replaces initialize_sdk)
    initialize_broker()
    
    # Initialize Azure time service for maintenance countdown
    try:
        check_azure_time_service()
    except Exception as e:
        logger.warning(f"Could not fetch Azure time on startup: {e}")
        # Use local time as fallback
        utc_tz = pytz.timezone('UTC')
        local_utc = datetime.now(utc_tz).isoformat()
        bot_status["azure_time"] = local_utc
        logger.info(f"Using local UTC time as fallback: {local_utc}")
    
    # Phase 12: Record starting equity for drawdown monitoring
    # Use the equity already fetched during broker initialization
    if bot_status.get("starting_equity", 0) == 0:
        # If not set yet, try to fetch it now
        equity = get_account_equity()
        bot_status["starting_equity"] = equity if equity is not None else 0.0
    
    starting_equity = bot_status.get("starting_equity", 0.0) or 0.0
    logger.info(f"[{primary_symbol}] Starting Equity: ${starting_equity:.2f}")
    
    # Calculate initial maintenance countdown - always calculate, even if Azure fails
    maintenance_countdown = "-- h --m"
    if bot_status.get("azure_time"):
        try:
            maintenance_countdown = dashboard.calculate_maintenance_time(bot_status["azure_time"])
            logger.info(f"Maintenance countdown: {maintenance_countdown}")
        except Exception as e:
            logger.warning(f"Could not calculate maintenance time: {e}")
            # Fallback to simple local calculation
            maintenance_countdown = dashboard.calculate_maintenance_time(None)
    else:
        # Use local time
        maintenance_countdown = dashboard.calculate_maintenance_time(None)
        logger.info(f"Maintenance countdown (local): {maintenance_countdown}")
    
    # Update dashboard with actual account balance and maintenance
    dashboard.update_bot_data({
        "account_balance": starting_equity,
        "maintenance_countdown": maintenance_countdown
    })
    dashboard.display()
    
    # Initialize state for all trading symbols
    for symbol in trading_symbols:
        initialize_state(symbol)
        
        # CRITICAL: Try to restore position state from disk if bot was restarted
        logger.info(f"[{symbol}] Checking for saved position state...")
        position_restored = load_position_state(symbol)
        if position_restored:
            logger.warning(f"[{symbol}] ⚠️  BOT RESTARTED WITH ACTIVE POSITION - Managing existing trade")
            # Update dashboard with restored position
            pos = state[symbol]["position"]
            if pos["active"]:
                dashboard.update_symbol_data(symbol, {
                    "position": f"{pos['side'].upper()} {pos['quantity']}",
                    "status": "Restored position"
                })
        else:
            logger.info(f"[{symbol}] No active position to restore - starting fresh")
            dashboard.update_symbol_data(symbol, {
                "status": "Initializing..."
            })
        dashboard.display()
    
    # Skip historical bars fetching in live mode - not needed for real-time trading
    # The bot will build bars from live tick data
    logger.info(f"[{primary_symbol}] Skipping historical bars fetch - will build bars from live data")
    
    # Initialize event loop
    logger.info(f"[{primary_symbol}] Initializing event loop...")
    event_loop = EventLoop(bot_status, CONFIG)
    
    # Register event handlers
    event_loop.register_handler(EventType.TICK_DATA, handle_tick_event)
    event_loop.register_handler(EventType.TIME_CHECK, handle_time_check_event)
    event_loop.register_handler(EventType.VWAP_RESET, handle_vwap_reset_event)
    event_loop.register_handler(EventType.FLATTEN_MODE, handle_flatten_mode_event)
    event_loop.register_handler(EventType.POSITION_RECONCILIATION, handle_position_reconciliation_event)
    event_loop.register_handler(EventType.CONNECTION_HEALTH, handle_connection_health_event)
    event_loop.register_handler(EventType.SHUTDOWN, handle_shutdown_event)
    
    # Register shutdown handlers for cleanup
    event_loop.register_shutdown_handler(cleanup_on_shutdown)
    
    # Initialize timer manager for periodic events
    tz = pytz.timezone(CONFIG["timezone"])
    timer_manager = TimerManager(event_loop, CONFIG, tz)
    timer_manager.start()
    
    # Subscribe to market data for all symbols
    for symbol in trading_symbols:
        subscribe_market_data(symbol, on_tick)
        
        # Subscribe to bid/ask quotes if broker supports it
        if broker is not None and hasattr(broker, 'subscribe_quotes'):
            logger.info("=" * 80)
            logger.info(f"[{symbol}] Subscribing to bid/ask quotes...")
            logger.info(f"Broker type: {type(broker)}")
            logger.info(f"Has subscribe_quotes: {hasattr(broker, 'subscribe_quotes')}")
            logger.info("=" * 80)
            try:
                broker.subscribe_quotes(symbol, on_quote)
                logger.info(f"[{symbol}] ✅ Successfully called broker.subscribe_quotes()")
                dashboard.update_symbol_data(symbol, {"status": "Subscribed to quotes"})
            except Exception as e:
                logger.error(f"❌ Failed to subscribe to quotes for {symbol}: {e}")
                logger.warning("Continuing without bid/ask quote data")
                import traceback
                traceback.print_exc()
                dashboard.update_symbol_data(symbol, {"status": "Quote subscription failed"})
        else:
            logger.warning(f"[{symbol}] Broker does NOT support quote subscriptions")
            logger.warning(f"broker={broker}, has_subscribe_quotes={hasattr(broker, 'subscribe_quotes') if broker else False}")
        dashboard.display()
    
    # RL and Adaptive Exits are CLOUD-ONLY - no local RL components
    # Users get confidence from cloud, contribute to cloud hive mind
    # Only the dev (Kevin) gets the experience data saved to cloud
    logger.info("Cloud RL Mode: All learning goes to shared hive mind")
    logger.info("No local RL files - everything saved to cloud for collective intelligence")
    
    logger.info("Bot initialization complete")
    logger.info("Starting event loop...")
    logger.info("Press Ctrl+C for graceful shutdown")
    logger.info(SEPARATOR_LINE)
    
    # Final dashboard update before entering event loop
    for symbol in trading_symbols:
        dashboard.update_symbol_data(symbol, {"status": "Monitoring..."})
    dashboard.display()
    
    # Run event loop (blocks until shutdown signal)
    try:
        logger.critical("=" * 80)
        logger.critical("STARTING EVENT LOOP - Bot should now run continuously")
        logger.critical("=" * 80)
        event_loop.run()
        logger.critical("=" * 80)
        logger.critical("EVENT LOOP EXITED (this should only happen on Ctrl+C or error)")
        logger.critical("=" * 80)
    except KeyboardInterrupt:
        logger.critical("\n" + "=" * 80)
        logger.critical("🛑 SHUTDOWN REQUESTED (Ctrl+C)")
        logger.critical("=" * 80)
    except Exception as e:
        logger.critical("\n" + "=" * 80)
        logger.critical(f"❌ EVENT LOOP CRASHED: {e}")
        logger.critical("=" * 80)
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Event loop stopped")
        
        # Generate session summary for primary symbol
        if CONFIG.get("instrument") and CONFIG["instrument"] in state:
            print_session_summary(CONFIG["instrument"])
        
        # Cleanup dashboard
        if dashboard:
            dashboard.stop()
        
        # Metrics are already logged by event loop's _log_metrics()
        # No need to call get_metrics() here


# ============================================================================
# DASHBOARD UPDATE HELPERS
# ============================================================================

def log_critical_error(error_message: str):
    """
    Log a critical error to both logger and dashboard.
    Only critical errors should use this to avoid dashboard spam.
    
    Args:
        error_message: The critical error message
    """
    global dashboard
    
    # Always log to logger
    logger.error(f"🚨 CRITICAL: {error_message}")
    
    # Also display on dashboard
    if dashboard:
        dashboard.add_critical_error(error_message)
        dashboard.display()


def update_dashboard_for_symbol(symbol: str) -> None:
    """
    Update dashboard display for a specific symbol with current market data.
    
    Args:
        symbol: Symbol to update
    """
    global dashboard
    
    if dashboard is None or symbol not in state:
        return
    
    symbol_state = state[symbol]
    
    # Get current market data
    current_price = symbol_state.get("last_price", 0.0)
    
    # Get bid/ask if available
    quote_data = {}
    if bid_ask_manager:
        quote = bid_ask_manager.get_current_quote(symbol)
        if quote:
            spread = quote.ask_price - quote.bid_price
            quote_data = {
                "bid": quote.bid_price,
                "bid_size": quote.bid_size,
                "ask": quote.ask_price,
                "ask_size": quote.ask_size,
                "spread": spread,
            }
            
            # Determine market condition based on spread
            if spread <= 0.50:
                condition = "NORMAL - Tight spread, good liquidity"
            elif spread <= 1.00:
                condition = "NORMAL - Good liquidity"
            elif spread <= 2.00:
                condition = "NORMAL - Acceptable spread"
            elif spread <= 5.00:
                condition = "CAUTION - Wider spread"
            else:
                condition = "VOLATILE - Wide spreads detected"
            quote_data["condition"] = condition
    
    # Get position data with P&L percentage
    position_data = {}
    pnl_percent = 0.0
    if symbol_state["position"]["active"]:
        pos = symbol_state["position"]
        position_data = {
            "position": f"{pos['side'].upper()} {pos['quantity']} @ ${pos['entry_price']:.2f}",
            "position_qty": pos["quantity"],
            "position_side": pos["side"],
        }
        
        # Calculate P&L percentage
        entry_price = pos.get("entry_price", current_price)
        if entry_price > 0:
            if pos["side"] == "long":
                pnl_ticks = (current_price - entry_price) / CONFIG["tick_size"]
            else:
                pnl_ticks = (entry_price - current_price) / CONFIG["tick_size"]
            
            pnl_value = pnl_ticks * CONFIG["tick_value"] * pos["quantity"]
            pnl_percent = (pnl_value / (entry_price * pos["quantity"])) * 100
    else:
        position_data = {
            "position": "FLAT",
            "position_qty": 0,
            "position_side": None,
        }
    
    # Get P&L
    pnl_data = {
        "pnl_today": symbol_state.get("daily_pnl", 0.0),
        "pnl_percent": pnl_percent
    }
    
    # Get market status from Azure or local time
    current_time = get_current_time()
    trading_state = get_trading_state(current_time)
    
    market_status = "OPEN" if trading_state == "entry_window" else "CLOSED"
    if trading_state == "flatten_mode":
        market_status = "FLATTEN"
    
    # Calculate maintenance countdown from Azure server time
    maintenance_time = "-- h --m"
    if dashboard:
        server_time = bot_status.get("azure_time")
        maintenance_time = dashboard.calculate_maintenance_time(server_time)
    
    # Get last signal info with time, confidence, and approval status
    last_signal = symbol_state.get("last_signal_type", "--")
    last_signal_time = ""
    last_signal_confidence = ""
    last_signal_approved = symbol_state.get("last_signal_approved", None)
    
    if "last_signal_timestamp" in symbol_state and symbol_state["last_signal_timestamp"]:
        try:
            sig_time = datetime.fromtimestamp(symbol_state["last_signal_timestamp"])
            last_signal_time = sig_time.strftime("%H:%M")
        except:
            pass
    
    if "last_signal_confidence" in symbol_state and symbol_state["last_signal_confidence"]:
        conf_val = symbol_state["last_signal_confidence"]
        if isinstance(conf_val, float):
            last_signal_confidence = f"{int(conf_val * 100)}%"
    
    # Get last rejected signal info
    last_rejected_signal = ""
    if "last_rejected_signal" in symbol_state and symbol_state["last_rejected_signal"]:
        rejected = symbol_state["last_rejected_signal"]
        if isinstance(rejected, dict):
            # Format the rejected signal for display
            rej_time = rejected.get("time")
            if rej_time:
                rej_time_str = rej_time.strftime("%H:%M")
                rej_side = rejected.get("side", "").upper()
                rej_conf = rejected.get("confidence", 0)
                rej_conf_str = f"{int(rej_conf * 100)}%"
                # Get price from state (approximate)
                rej_reason = rejected.get("reason", "")
                last_rejected_signal = f"{rej_time_str} {rej_side} (Conf: {rej_conf_str}) - {rej_reason}"
    
    # Get current status - update with live P&L if in trade
    status = symbol_state.get("bot_status_message", "Monitoring...")
    
    # If in trade, enhance status with current P&L
    if symbol_state["position"]["active"]:
        pos = symbol_state["position"]
        entry_price = pos.get("entry_price", current_price)
        
        # Calculate current P&L
        if entry_price > 0:
            if pos["side"] == "long":
                pnl_ticks = (current_price - entry_price) / CONFIG["tick_size"]
            else:
                pnl_ticks = (entry_price - current_price) / CONFIG["tick_size"]
            
            current_pnl = pnl_ticks * CONFIG["tick_value"] * pos["quantity"]
            pnl_sign = "+" if current_pnl > 0 else ""
            
            # Show target price if available
            target = pos.get("target_price")
            if target:
                status = f"In trade - P&L: ${pnl_sign}{current_pnl:.2f} | Target: ${target:.2f}"
            else:
                status = f"In trade - P&L: ${pnl_sign}{current_pnl:.2f}"
    
    # Get skip reason if available
    skip_reason = symbol_state.get("skip_reason", "")
    
    # Combine all data
    update_data = {
        "market_status": market_status,
        "maintenance_in": maintenance_time,
        "last_signal": last_signal,
        "last_signal_time": last_signal_time,
        "last_signal_confidence": last_signal_confidence,
        "last_signal_approved": last_signal_approved,
        "last_rejected_signal": last_rejected_signal,
        "skip_reason": skip_reason,
        "status": status,
        **quote_data,
        **position_data,
        **pnl_data,
    }
    
    # Update dashboard
    dashboard.update_symbol_data(symbol, update_data)
    
    # Also update bot-level data with maintenance and TopStep status
    # Check broker connection status properly
    topstep_status = False
    if broker is not None:
        if hasattr(broker, 'is_connected'):
            topstep_status = broker.is_connected()
        elif hasattr(broker, 'connected'):
            topstep_status = broker.connected
    
    dashboard.update_bot_data({
        "maintenance_countdown": maintenance_time,
        "topstep_connected": topstep_status
    })


# ============================================================================
# EVENT HANDLERS
# ============================================================================

def handle_tick_event(event) -> None:
    """Handle tick data event from event loop"""
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
    
    # Increment total tick counter (separate from deque storage which caps at 10k)
    if "total_ticks_received" not in state[symbol]:
        state[symbol]["total_ticks_received"] = 0
    state[symbol]["total_ticks_received"] += 1
    total_ticks = state[symbol]["total_ticks_received"]
    
    # Update dashboard every 50 ticks (about every 1 second in active market)
    # More frequent updates for real-time monitoring
    if total_ticks % 50 == 0:
        update_dashboard_for_symbol(symbol)
        if dashboard:
            dashboard.display()
    
    # Suppress normal logging - only log every 1000 ticks to reduce spam
    if total_ticks % 1000 == 0:
        # Get current bid/ask from bid_ask_manager if available
        bid_ask_info = ""
        if bid_ask_manager is not None:
            quote = bid_ask_manager.get_current_quote(symbol)
            if quote:
                spread = quote.ask_price - quote.bid_price
                bid_ask_info = f" | Bid: ${quote.bid_price:.2f} x {quote.bid_size} | Ask: ${quote.ask_price:.2f} x {quote.ask_size} | Spread: ${spread:.2f}"
        logger.critical(f"[TICK] {symbol} @ ${price:.2f} | Vol: {volume} | Total ticks: {total_ticks}{bid_ask_info}")
    
    # Store last price in state
    state[symbol]["last_price"] = price
    
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
        current_time = datetime.now(tz).time()
        
        # Check for daily reset
        check_daily_reset(symbol, datetime.now(tz))
        
        # Critical safety check - NO positions past 5 PM
        check_no_overnight_positions(symbol)


def handle_vwap_reset_event(data: Dict[str, Any]) -> None:
    """Handle VWAP reset event"""
    symbol = CONFIG["instrument"]
    if symbol in state:
        tz = pytz.timezone(CONFIG["timezone"])
        check_vwap_reset(symbol, datetime.now(tz))


def handle_flatten_mode_event(data: Dict[str, Any]) -> None:
    """Handle flatten mode activation event"""
    logger.warning("Flatten mode activated - initiating position closure")
    bot_status["flatten_mode"] = True
    
    # If position is active, start flatten process
    symbol = CONFIG["instrument"]
    if symbol in state and state[symbol]["position"]["active"]:
        logger.warning(f"Active position detected - executing flatten")
        # The exit conditions check will handle the flatten


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


def handle_shutdown_event(data: Dict[str, Any]) -> None:
    """Handle shutdown event"""
    logger.info("Shutdown event received")
    bot_status["trading_enabled"] = False


def print_session_summary(symbol: str) -> None:
    """Print detailed session summary on exit"""
    print("\n" + "=" * 80)
    print("SESSION SUMMARY - QuoTrading AI Bot")
    print("=" * 80)
    
    # Session duration
    if bot_status.get("session_start_time"):
        start_time = bot_status["session_start_time"]
        end_time = datetime.now(pytz.UTC)
        duration = end_time - start_time
        hours = int(duration.total_seconds() // 3600)
        minutes = int((duration.total_seconds() % 3600) // 60)
        print(f"Session Duration: {hours}h {minutes}m")
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "-" * 80)
    print(f"SYMBOL: {symbol}")
    print("-" * 80)
    
    # Trading statistics
    symbol_state = state[symbol]
    bars_count = symbol_state.get("bars_1min_count", 0)
    ticks_received = symbol_state.get("total_ticks_received", 0)
    
    print(f"Bars Processed: {bars_count} (1-minute bars)")
    print(f"Ticks Received: {ticks_received:,}")
    
    # P&L Summary
    pnl_today = symbol_state.get("pnl_today", 0.0)
    starting_equity = bot_status.get("starting_equity", 0.0) or 0.0
    current_equity = get_account_equity()
    current_equity = current_equity if current_equity is not None else 0.0
    
    print(f"\nStarting Equity: ${starting_equity:,.2f}")
    print(f"Ending Equity: ${current_equity:,.2f}")
    print(f"P&L Today: ${pnl_today:+,.2f}")
    
    if starting_equity > 0:
        pnl_percent = (pnl_today / starting_equity) * 100
        print(f"Return: {pnl_percent:+.2f}%")
    
    # Position info
    position = symbol_state["position"]
    if position.get("active"):
        print(f"\n⚠️  WARNING: Active position still open!")
        print(f"   Side: {position['side'].upper()}")
        print(f"   Quantity: {position['quantity']}")
        print(f"   Entry: ${position.get('entry_price', 0):.2f}")
    else:
        print(f"\n✓ No open positions")
    
    # Signal statistics
    signals_generated = symbol_state.get("signals_generated", 0)
    signals_approved = symbol_state.get("signals_approved", 0)
    signals_rejected = symbol_state.get("signals_rejected", 0)
    
    if signals_generated > 0:
        print(f"\nSignals Generated: {signals_generated}")
        print(f"  Approved: {signals_approved}")
        print(f"  Rejected: {signals_rejected}")
        if signals_generated > 0:
            approval_rate = (signals_approved / signals_generated) * 100
            print(f"  Approval Rate: {approval_rate:.1f}%")
    
    # Trade statistics
    trades_count = symbol_state.get("trades_today", 0)
    winning_trades = symbol_state.get("winning_trades", 0)
    losing_trades = symbol_state.get("losing_trades", 0)
    
    if trades_count > 0:
        print(f"\nTrades Executed: {trades_count}")
        print(f"  Wins: {winning_trades}")
        print(f"  Losses: {losing_trades}")
        if trades_count > 0:
            win_rate = (winning_trades / trades_count) * 100
            print(f"  Win Rate: {win_rate:.1f}%")
    
    print("\n" + "=" * 80)
    print("Thank you for using QuoTrading AI Bot!")
    print("=" * 80 + "\n")


def cleanup_on_shutdown() -> None:
    """Cleanup tasks on shutdown"""
    logger.info("Running cleanup tasks...")
    
    # Stop dashboard display
    if dashboard:
        try:
            dashboard.stop()
            logger.info("Dashboard stopped")
        except Exception as e:
            logger.error(f"Error stopping dashboard: {e}")
    
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
    
    # Print session summary
    symbol = CONFIG["instrument"]
    if symbol in state:
        print_session_summary(symbol)
    
    logger.info("Cleanup complete")


if __name__ == "__main__":
    main()
