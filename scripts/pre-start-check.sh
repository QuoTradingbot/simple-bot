#!/bin/bash
#
# Pre-start checks for VWAP Bounce Bot
# Validates environment before starting the bot
#

set -e

BOT_DIR="/opt/vwap-bot/simple-bot"
LOG_DIR="$BOT_DIR/logs"
DATA_DIR="$BOT_DIR/data"

echo "==================================="
echo "VWAP Bot Pre-Start Checks"
echo "==================================="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check 1: Verify we're in the correct directory
echo "[1/8] Checking working directory..."
if [ ! -f "$BOT_DIR/main.py" ]; then
    echo "ERROR: Bot files not found in $BOT_DIR"
    exit 1
fi
echo "  ✓ Working directory OK"

# Check 2: Verify Python virtual environment
echo "[2/8] Checking Python environment..."
if [ ! -f "$BOT_DIR/venv/bin/python" ]; then
    echo "ERROR: Python virtual environment not found"
    exit 1
fi

PYTHON_VERSION=$($BOT_DIR/venv/bin/python --version 2>&1 | cut -d' ' -f2)
echo "  ✓ Python $PYTHON_VERSION found"

# Check 3: Verify configuration file exists
echo "[3/8] Checking configuration..."
if [ ! -f "$BOT_DIR/.env.production" ]; then
    echo "ERROR: Production configuration file not found"
    echo "Create .env.production with required settings"
    exit 1
fi
echo "  ✓ Configuration file exists"

# Check 4: Validate configuration
echo "[4/8] Validating configuration..."
$BOT_DIR/venv/bin/python -c "
from config import load_config
try:
    config = load_config('production')
    config.validate()
    print('  ✓ Configuration valid')
except Exception as e:
    print(f'  ✗ Configuration error: {e}')
    exit(1)
"

# Check 5: Check disk space
echo "[5/8] Checking disk space..."
AVAILABLE_GB=$(df -BG "$BOT_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -lt 5 ]; then
    echo "WARNING: Low disk space (${AVAILABLE_GB}GB available, 5GB minimum recommended)"
else
    echo "  ✓ Disk space OK (${AVAILABLE_GB}GB available)"
fi

# Check 6: Create necessary directories
echo "[6/8] Ensuring required directories exist..."
mkdir -p "$LOG_DIR" "$DATA_DIR"
chmod 755 "$LOG_DIR" "$DATA_DIR"
echo "  ✓ Directories OK"

# Check 7: Check network connectivity
echo "[7/8] Checking network connectivity..."
if ping -c 1 -W 2 8.8.8.8 > /dev/null 2>&1; then
    echo "  ✓ Network connectivity OK"
else
    echo "WARNING: Network connectivity issue detected"
    echo "  Bot may not be able to connect to broker"
fi

# Check 8: Verify no stale PID files
echo "[8/8] Checking for stale processes..."
if [ -f "$BOT_DIR/bot.pid" ]; then
    OLD_PID=$(cat "$BOT_DIR/bot.pid")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "WARNING: Bot appears to be already running (PID: $OLD_PID)"
        echo "  If this is incorrect, remove $BOT_DIR/bot.pid"
    else
        echo "  Removing stale PID file"
        rm -f "$BOT_DIR/bot.pid"
    fi
fi
echo "  ✓ No stale processes"

echo ""
echo "==================================="
echo "All pre-start checks passed!"
echo "Starting bot..."
echo "==================================="
echo ""

exit 0
