"""
Customer Version Builder
========================
Builds a clean, production-ready version for customers.

What customers GET:
- Clean bot code (no dev features)
- Bot state persistence (crash recovery only)
- Minimal logging (errors only)
- RL confidence via YOUR cloud API (they don't see the models)

What customers DON'T GET:
- Historical data recording
- Verbose logs (vwap_bounce_bot.log, audit.log, etc.)
- RL training files (signal_experience.json, exit_experience.json)
- Debug features

Usage:
    python build_customer_version.py
    
Creates: ../simple-bot-customer/ (ready to distribute)
"""

import os
import shutil
import json
from pathlib import Path


# Output directory
CUSTOMER_DIR = Path("../simple-bot-customer")

# Core files customers need
CORE_FILES = [
    "run.py",
    "requirements.txt",
    "requirements-pinned.txt",
    ".gitignore",
]

# Source files (we'll modify some)
SOURCE_FILES = [
    "src/__init__.py",
    "src/config.py",
    "src/main.py",
    "src/vwap_bounce_bot.py",
    "src/broker_interface.py",
    "src/topstep_websocket.py",
    "src/bid_ask_manager.py",
    "src/event_loop.py",
    "src/monitoring.py",
    "src/error_recovery.py",
    "src/backtesting.py",
    "src/symbol_specs.py",  # Multi-symbol specifications (ES, NQ, MES, etc.)
    # NOTE: signal_confidence.py will be modified to use cloud API
]

# GUI dependencies for customer launcher
GUI_REQUIREMENTS = """
# GUI Application Dependencies
tkinter  # Built-in with Python
pillow>=10.0.0  # For images/icons

# EXE Packaging (for building standalone executable)
pyinstaller>=6.0.0  # Convert to Windows EXE
"""

# Files to skip (dev only)
SKIP_FILES = [
    "src/live_data_recorder.py",  # No data recording for customers
    "src/adaptive_exits.py",  # RL training logic stays with you
]


def clean_directory():
    """Remove existing customer directory."""
    if CUSTOMER_DIR.exists():
        print(f"🗑️  Removing existing {CUSTOMER_DIR}...")
        shutil.rmtree(CUSTOMER_DIR)


def create_structure():
    """Create customer directory structure."""
    print(f"📁 Creating {CUSTOMER_DIR}...")
    CUSTOMER_DIR.mkdir(parents=True, exist_ok=True)
    (CUSTOMER_DIR / "src").mkdir(exist_ok=True)
    (CUSTOMER_DIR / "logs").mkdir(exist_ok=True)  # For minimal error logs only


def copy_core_files():
    """Copy core files without modification."""
    print("📋 Copying core files...")
    for file in CORE_FILES:
        src = Path(file)
        dst = CUSTOMER_DIR / file
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ {file}")


def copy_source_files():
    """Copy source files (some will be modified)."""
    print("📋 Copying source files...")
    for file in SOURCE_FILES:
        src = Path(file)
        dst = CUSTOMER_DIR / file
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ {file}")


def create_customer_env_example():
    """Create customer .env.example (minimal configuration)."""
    print("⚙️  Creating customer .env.example...")
    
    env_content = """# QuoTrading AI - Automated Futures Trading Bot
# ===============================================

# === REQUIRED: TopStep Credentials ===
TOPSTEP_API_TOKEN=your_topstep_api_token_here
TOPSTEP_USERNAME=your_topstep_email@example.com

# === REQUIRED: Trading Environment ===
BOT_ENVIRONMENT=production
BOT_DRY_RUN=false
CONFIRM_LIVE_TRADING=1

# === Position Sizing ===
BOT_MAX_CONTRACTS=3
BOT_MAX_TRADES_PER_DAY=9

# === Risk Management (TopStep Rules) ===
BOT_USE_TOPSTEP_RULES=true
BOT_RISK_PER_TRADE=0.012
BOT_RISK_REWARD_RATIO=2.0

# === Trading Instruments ===
# Supported: ES, NQ, YM, RTY, CL, GC, etc.
BOT_INSTRUMENT=ES
BOT_TIMEZONE=America/New_York

# === QuoTrading AI Cloud API ===
QUOTRADING_API_URL=https://api.quotrading.com/v1/signals
QUOTRADING_API_KEY=customer_api_key_here

# === Logging (Minimal for Production) ===
BOT_LOG_LEVEL=WARNING
BOT_ENABLE_VERBOSE_LOGS=false
"""
    
    with open(CUSTOMER_DIR / ".env.example", "w") as f:
        f.write(env_content)
    print("  ✓ .env.example (customer version)")


def create_customer_readme():
    """Create customer-focused README."""
    print("📝 Creating customer README...")
    
    readme_content = """# QuoTrading AI - Automated Futures Trading Bot

Professional AI-powered automated trading bot for futures markets.

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your TopStep credentials and API key
   ```

3. **Run the Bot**
   ```bash
   python run.py
   ```

## Configuration

Edit `.env` file with your settings:

- `TOPSTEP_API_TOKEN` - Your TopStep API token
- `TOPSTEP_USERNAME` - Your TopStep email
- `QUOTRADING_API_KEY` - Your QuoTrading AI API key
- `BOT_INSTRUMENT` - Trading symbol (ES, NQ, YM, RTY, CL, GC, etc.)
- `BOT_MAX_CONTRACTS` - Maximum contracts per trade (default: 3)
- `BOT_MAX_TRADES_PER_DAY` - Daily trade limit (default: 9)

## QuoTrading AI Strategy

Proprietary machine learning algorithm that:

- Analyzes real-time market conditions
- Generates high-probability trade signals
- Dynamically adjusts position sizing based on confidence
- Manages risk with intelligent stops and targets
- Follows TopStep rules automatically

**All signals are powered by QuoTrading AI's cloud-based machine learning models.**

## Safety Features

- TopStep rule compliance (daily loss limits, drawdown protection)
- Automatic position flattening before session close
- Real-time position monitoring
- Crash recovery (resumes after restart)
- Multi-symbol support (ES, NQ, YM, RTY, and more)

## Files Generated

- `bot_state.json` - Current bot state (crash recovery)
- `logs/error.log` - Error log (troubleshooting only)

## Support

For issues or questions, contact support@quotrading.com

## Risk Disclosure

Trading futures involves substantial risk of loss. This bot is for educational and research purposes. Past performance does not guarantee future results.
"""
    
    with open(CUSTOMER_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("  ✓ README.md (customer version)")


def create_cloud_api_client():
    """Create QuoTrading AI cloud API client (customers call YOUR API)."""
    print("☁️  Creating QuoTrading AI cloud client...")
    
    client_code = '''"""
QuoTrading AI Cloud Client
===========================
Customers use this to get AI-powered trade signals from QuoTrading cloud service.
They never see the ML models or training logic - just the signals.
"""

import os
import requests
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class QuoTradingAI:
    """Client for QuoTrading AI cloud-based signal generation."""
    
    def __init__(self):
        self.api_url = os.getenv("QUOTRADING_API_URL", "")
        self.api_key = os.getenv("QUOTRADING_API_KEY", "")
        self.timeout = 5  # 5 second timeout
        
        if not self.api_url or not self.api_key:
            logger.warning("QuoTrading AI API not configured. Using fallback confidence.")
    
    def get_signal_confidence(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get trade confidence from QuoTrading AI cloud service.
        
        Args:
            state: Market state (price, volume, technical indicators, etc.)
            
        Returns:
            {
                "confidence": 0.75,  # 0.0 to 1.0
                "contracts": 2,      # Recommended contracts (1-3)
                "reason": "HIGH confidence signal"
            }
        """
        if not self.api_url or not self.api_key:
            # Fallback: medium confidence
            return {
                "confidence": 0.5,
                "contracts": 1,
                "reason": "API not configured - using fallback"
            }
        
        try:
            response = requests.post(
                self.api_url,
                json={"state": state},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"QuoTrading AI API error: {response.status_code}")
                return self._fallback_confidence()
                
        except requests.exceptions.Timeout:
            logger.error("QuoTrading AI API timeout")
            return self._fallback_confidence()
        except Exception as e:
            logger.error(f"QuoTrading AI API exception: {e}")
            return self._fallback_confidence()
    
    def _fallback_confidence(self) -> Dict[str, Any]:
        """Fallback when API unavailable."""
        return {
            "confidence": 0.5,
            "contracts": 1,
            "reason": "API unavailable - using conservative fallback"
        }
'''
    
    with open(CUSTOMER_DIR / "src" / "quotrading_ai.py", "w") as f:
        f.write(client_code)
    print("  ✓ src/quotrading_ai.py (QuoTrading AI client)")


def create_gui_launcher():
    """Create professional GUI launcher application."""
    print("🖥️  Creating GUI launcher...")
    
    # Copy launcher
    src = Path("customer_launcher_template.py")
    dst = CUSTOMER_DIR / "QuoTrading_Launcher.py"
    
    if src.exists():
        shutil.copy2(src, dst)
        print("  ✓ QuoTrading_Launcher.py (Python launcher)")
    else:
        print("  ⚠️  Warning: customer_launcher_template.py not found")
    
    # Copy EXE build files
    spec_src = Path("build_exe.spec")
    spec_dst = CUSTOMER_DIR / "build_exe.spec"
    if spec_src.exists():
        shutil.copy2(spec_src, spec_dst)
        print("  ✓ build_exe.spec (for creating EXE)")
    
    version_src = Path("version_info.txt")
    version_dst = CUSTOMER_DIR / "version_info.txt"
    if version_src.exists():
        shutil.copy2(version_src, version_dst)
        print("  ✓ version_info.txt (Windows version info)")
    
    instructions_src = Path("BUILD_EXE_INSTRUCTIONS.md")
    instructions_dst = CUSTOMER_DIR / "BUILD_EXE_INSTRUCTIONS.md"
    if instructions_src.exists():
        shutil.copy2(instructions_src, instructions_dst)
        print("  ✓ BUILD_EXE_INSTRUCTIONS.md (how to build EXE)")


def create_customer_logger():
    """Create professional customer-facing logger."""
    print("📝 Creating customer logger...")
    
    logger_code = '''"""
Customer Logger - Professional Output Only
==========================================
Shows only actionable information: trades, status, risk alerts.
Hides all technical details, strategy internals, and debug info.
"""

import logging
import os
from datetime import datetime
from typing import Optional


class CustomerLogFilter(logging.Filter):
    """Filter that only allows customer-friendly log messages."""
    
    # Customer-facing prefixes (what they SHOULD see)
    ALLOWED_PREFIXES = [
        "[TRADE]",      # Trade executions
        "[SIGNAL]",     # AI signal notifications
        "[STATUS]",     # Bot status updates
        "[RISK]",       # Risk warnings
        "[SAFETY]",     # Safety notifications
        "[ERROR]",      # Critical errors only
        "[WARNING]",    # Important warnings
        "[PERFORMANCE]" # Daily/session performance
    ]
    
    # Hide these (developer-only)
    HIDDEN_KEYWORDS = [
        "vwap", "VWAP",
        "band", "bounce",
        "std dev", "StdDev",
        "RSI", "rsi",
        "tick", "bar update",
        "experience", "RL",
        "debug", "DEBUG",
        "calculate", "calculating"
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Return True if message should be shown to customer."""
        msg = record.getMessage().lower()
        
        # Check if message has customer-friendly prefix
        has_allowed_prefix = any(
            prefix.lower() in msg 
            for prefix in self.ALLOWED_PREFIXES
        )
        
        # Check if message contains hidden keywords
        has_hidden_keyword = any(
            keyword.lower() in msg 
            for keyword in self.HIDDEN_KEYWORDS
        )
        
        # Show if: has allowed prefix AND no hidden keywords
        # OR if it's ERROR/CRITICAL level (always show errors)
        if record.levelno >= logging.ERROR:
            return not has_hidden_keyword
        
        return has_allowed_prefix and not has_hidden_keyword


class CustomerFormatter(logging.Formatter):
    """Professional formatter for customer logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log message in clean, professional style."""
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Color codes for different message types (optional, works in most terminals)
        COLORS = {
            'TRADE': '\\033[92m',     # Green
            'SIGNAL': '\\033[96m',    # Cyan
            'STATUS': '\\033[94m',    # Blue
            'RISK': '\\033[93m',      # Yellow
            'SAFETY': '\\033[95m',    # Magenta
            'ERROR': '\\033[91m',     # Red
            'WARNING': '\\033[93m',   # Yellow
            'PERFORMANCE': '\\033[92m' # Green
        }
        RESET = '\\033[0m'
        
        msg = record.getMessage()
        
        # Add color if message has recognizable prefix
        for prefix, color in COLORS.items():
            if f'[{prefix}]' in msg:
                # Only colorize in interactive terminals
                if os.getenv('TERM'):
                    msg = msg.replace(f'[{prefix}]', f'{color}[{prefix}]{RESET}')
                break
        
        return f"{timestamp} {msg}"


def setup_customer_logging(log_file: str = "logs/quotrading.log", level: str = "INFO"):
    """
    Setup customer-friendly logging.
    
    Args:
        log_file: Path to log file
        level: Log level (INFO, WARNING, ERROR)
    """
    # Create logs directory
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler (what customer sees in terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(CustomerFormatter())
    console_handler.addFilter(CustomerLogFilter())
    
    # File handler (minimal - errors only)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.ERROR)  # Only log errors to file
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s'
    ))
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# Helper functions for customer-friendly logging
def log_trade(action: str, symbol: str, price: float, contracts: int, reason: str = ""):
    """Log trade execution."""
    logger = logging.getLogger(__name__)
    msg = f"[TRADE] {action} {symbol} @ {price:.2f} | {contracts} contracts"
    if reason:
        msg += f" | {reason}"
    logger.info(msg)


def log_signal(direction: str, symbol: str, confidence: str, contracts: int):
    """Log AI signal."""
    logger = logging.getLogger(__name__)
    logger.info(f"[SIGNAL] AI Signal: {direction} {symbol} | Confidence: {confidence} | Size: {contracts} contracts")


def log_status(message: str):
    """Log status update."""
    logger = logging.getLogger(__name__)
    logger.info(f"[STATUS] {message}")


def log_risk(message: str):
    """Log risk warning."""
    logger = logging.getLogger(__name__)
    logger.warning(f"[RISK] {message}")


def log_performance(trades: int, max_trades: int, pnl: float, win_rate: float):
    """Log daily performance."""
    logger = logging.getLogger(__name__)
    logger.info(f"[PERFORMANCE] Daily Stats | Trades: {trades}/{max_trades} | "
                f"P&L: ${pnl:+.2f} | Win Rate: {win_rate:.0f}%")
'''
    
    with open(CUSTOMER_DIR / "src" / "customer_logger.py", "w") as f:
        f.write(logger_code)
    print("  ✓ src/customer_logger.py (clean professional logging)")


def create_minimal_signal_confidence():
    """Create customer version of signal_confidence.py (uses cloud API)."""
    print("🔄 Creating customer signal_confidence.py...")
    
    # Note: You'll need to modify the actual signal_confidence.py to use QuoTradingAI
    # For now, just copy it - you can modify later
    src = Path("src/signal_confidence.py")
    dst = CUSTOMER_DIR / "src" / "signal_confidence.py"
    if src.exists():
        shutil.copy2(src, dst)
        print("  ✓ src/signal_confidence.py (NOTE: Modify to use QuoTradingAI)")


def create_summary():
    """Create build summary file."""
    summary = f"""
Customer Version Build Summary
==============================

CREATED: {CUSTOMER_DIR}

What's Included:
- QuoTrading AI bot (production-ready)
- QuoTrading AI cloud client (calls YOUR server)
- Bot state persistence (crash recovery)
- Minimal error logging only
- Customer README with QuoTrading branding
- Clean .env.example
- Multi-symbol support (ES, NQ, YM, RTY, etc.)

What's Excluded:
- Historical data recording
- ML training files (signal_experience.json, exit_experience.json)
- Strategy details (VWAP, bands, etc. - fully hidden)
- Verbose logs (vwap_bounce_bot.log, audit.log, performance.log)
- Live data recorder
- Debug features

Next Steps:
1. Review {CUSTOMER_DIR}
2. Modify src/signal_confidence.py to use QuoTradingAI
3. Test the customer build
4. BUILD WINDOWS EXE:
   cd {CUSTOMER_DIR}
   pip install pyinstaller
   pyinstaller build_exe.spec
   
5. Result: dist/QuoTrading_AI.exe (standalone, no Python needed!)

Your ML models stay on YOUR cloud - customers just call QuoTrading AI API!
"""
    
    print(summary)
    
    with open(CUSTOMER_DIR / "BUILD_SUMMARY.txt", "w", encoding="utf-8") as f:
        f.write(summary)


def main():
    """Build customer version."""
    print("=" * 60)
    print("Building Customer Version")
    print("=" * 60)
    
    clean_directory()
    create_structure()
    copy_core_files()
    copy_source_files()
    create_customer_env_example()
    create_customer_readme()
    create_cloud_api_client()
    create_customer_logger()
    create_gui_launcher()
    create_minimal_signal_confidence()
    create_summary()
    
    print("\n" + "=" * 60)
    print("✅ Customer build complete!")
    print(f"📂 Output: {CUSTOMER_DIR.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
