#!/bin/bash
#
# Configuration validation script for VWAP Bounce Bot
# Validates environment configuration before deployment
#

set -e

# Parse arguments
ENVIRONMENT="${1:-development}"

if [ "$ENVIRONMENT" != "development" ] && [ "$ENVIRONMENT" != "staging" ] && [ "$ENVIRONMENT" != "production" ]; then
    echo "Usage: $0 [development|staging|production]"
    echo "Example: $0 production"
    exit 1
fi

# Detect the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$BOT_DIR"

echo "==================================="
echo "Configuration Validation"
echo "==================================="
echo "Environment: $ENVIRONMENT"
echo "Working Directory: $BOT_DIR"
echo ""

# Check 1: Verify environment file exists
echo "[1/7] Checking environment file..."
ENV_FILE=".env.$ENVIRONMENT"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: Environment file not found: $ENV_FILE"
    echo "Create $ENV_FILE with required configuration."
    exit 1
fi
echo "  ✓ Environment file exists: $ENV_FILE"

# Check 2: Verify Python environment
echo "[2/7] Checking Python environment..."
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found. Run scripts/install.sh first."
    exit 1
fi
source venv/bin/activate
echo "  ✓ Virtual environment activated"

# Check 3: Validate configuration with Python
echo "[3/7] Validating configuration structure..."
python -c "
import os
import sys
from pathlib import Path

# Load environment file
env_file = '.env.$ENVIRONMENT'
env_vars = {}

with open(env_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            env_vars[key] = value

# Set in environment
for key, value in env_vars.items():
    os.environ[key] = value

# Validate with config module
try:
    from config import load_config
    config = load_config('$ENVIRONMENT')
    config.validate()
    print('  ✓ Configuration structure valid')
except Exception as e:
    print(f'  ✗ Configuration error: {e}')
    sys.exit(1)
" || exit 1

# Check 4: Verify API token is set (for non-backtest)
echo "[4/7] Checking credentials..."
source "$ENV_FILE"

if [ -z "$TOPSTEP_API_TOKEN" ] || [ "$TOPSTEP_API_TOKEN" = "your_token_here" ]; then
    echo "WARNING: TOPSTEP_API_TOKEN not set or using placeholder value"
    if [ "$ENVIRONMENT" = "production" ]; then
        echo "ERROR: Production environment requires valid API token"
        exit 1
    fi
else
    echo "  ✓ API token configured"
fi

# Check 5: Verify environment-specific settings
echo "[5/7] Checking environment-specific settings..."

case "$ENVIRONMENT" in
    "production")
        if [ "$BOT_DRY_RUN" = "true" ]; then
            echo "WARNING: Production environment has dry-run enabled"
            echo "  Set BOT_DRY_RUN=false for live trading"
        fi
        if [ -z "$CONFIRM_LIVE_TRADING" ] || [ "$CONFIRM_LIVE_TRADING" != "1" ]; then
            echo "WARNING: CONFIRM_LIVE_TRADING not set to 1"
            echo "  This is required for live trading"
        fi
        ;;
    "staging")
        if [ "$BOT_DRY_RUN" != "true" ]; then
            echo "WARNING: Staging should use dry-run mode"
        fi
        ;;
    "development")
        echo "  ✓ Development environment detected"
        ;;
esac

echo "  ✓ Environment settings checked"

# Check 6: Verify log directory permissions
echo "[6/7] Checking file permissions..."
if [ -d "logs" ]; then
    if [ ! -w "logs" ]; then
        echo "ERROR: logs directory not writable"
        exit 1
    fi
fi
echo "  ✓ Directory permissions OK"

# Check 7: Security checks
echo "[7/7] Running security checks..."

# Check file permissions on environment file
ENV_PERMS=$(stat -c %a "$ENV_FILE" 2>/dev/null || stat -f %A "$ENV_FILE" 2>/dev/null || echo "unknown")
if [ "$ENV_PERMS" != "600" ] && [ "$ENV_PERMS" != "unknown" ]; then
    echo "WARNING: $ENV_FILE has permissions $ENV_PERMS (should be 600)"
    echo "  Run: chmod 600 $ENV_FILE"
fi

# Check that .env files are in .gitignore
if [ -f ".gitignore" ]; then
    if ! grep -q "\.env\.\*" .gitignore && ! grep -q "\.env$" .gitignore; then
        echo "WARNING: .env files may not be in .gitignore"
        echo "  Ensure credentials are not committed to git"
    fi
fi

echo "  ✓ Security checks completed"

echo ""
echo "==================================="
echo "Validation completed successfully!"
echo "==================================="
echo ""
echo "Configuration summary for $ENVIRONMENT:"
echo "  - Environment file: $ENV_FILE"
echo "  - API token: $([ -n "$TOPSTEP_API_TOKEN" ] && echo "Configured" || echo "Not set")"
echo "  - Dry run: ${BOT_DRY_RUN:-not set}"
echo "  - Max trades/day: ${BOT_MAX_TRADES_PER_DAY:-not set}"
echo ""

if [ "$ENVIRONMENT" = "production" ]; then
    echo "IMPORTANT: Production environment detected"
    echo "  - Ensure CONFIRM_LIVE_TRADING=1 is set for live trading"
    echo "  - Verify API token is for production broker account"
    echo "  - Review all settings in $ENV_FILE before deploying"
    echo ""
fi

exit 0
