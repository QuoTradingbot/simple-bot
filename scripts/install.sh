#!/bin/bash
#
# Installation script for VWAP Bounce Bot
# Sets up virtual environment and installs dependencies
#

set -e

echo "==================================="
echo "VWAP Bounce Bot - Installation"
echo "==================================="
echo ""

# Detect the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$BOT_DIR"

# Check Python version
echo "[1/6] Checking Python version..."
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.12 or newer."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.12"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    echo "ERROR: Python $REQUIRED_VERSION or newer required (found $PYTHON_VERSION)"
    exit 1
fi

echo "  ✓ Python $PYTHON_VERSION found"

# Create virtual environment
echo "[2/6] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "  Virtual environment already exists. Skipping creation."
else
    $PYTHON_CMD -m venv venv
    echo "  ✓ Virtual environment created"
fi

# Activate virtual environment
echo "[3/6] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "[4/6] Upgrading pip..."
pip install --upgrade pip --quiet
echo "  ✓ Pip upgraded to $(pip --version | cut -d' ' -f2)"

# Install dependencies
echo "[5/6] Installing dependencies..."
if [ -f "requirements-pinned.txt" ]; then
    echo "  Using pinned requirements (production mode)..."
    pip install -r requirements-pinned.txt
else
    echo "  Using standard requirements..."
    pip install -r requirements.txt
fi
echo "  ✓ Dependencies installed"

# Verify installation
echo "[6/6] Verifying installation..."
python -c "import pytz, dotenv; print('  ✓ Core dependencies OK')" || {
    echo "ERROR: Failed to import core dependencies"
    exit 1
}

# Check if tests can run
if [ -f "test_backtest_monitoring.py" ]; then
    echo "  Running quick test..."
    python -c "
import sys
sys.path.insert(0, '.')
from backtesting import BacktestConfig
from monitoring import HealthChecker
print('  ✓ Modules load successfully')
" || {
        echo "WARNING: Some modules failed to load. Check dependencies."
    }
fi

echo ""
echo "==================================="
echo "Installation completed successfully!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env.production and configure"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Run tests: python -m pytest test_backtest_monitoring.py"
echo "4. Start bot: python main.py --mode live --dry-run"
echo ""
echo "For production deployment, see DEPLOYMENT.md"
echo ""

exit 0
