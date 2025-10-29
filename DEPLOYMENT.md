# VWAP Bounce Bot - Deployment Guide

## Table of Contents
1. [Server Requirements](#server-requirements)
2. [Pre-Deployment](#pre-deployment)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Process Management](#process-management)
6. [Deployment Checklist](#deployment-checklist)
7. [Monitoring](#monitoring)
8. [Rollback Procedure](#rollback-procedure)

---

## Server Requirements

### Hardware Requirements
- **CPU**: 2+ cores (4 recommended for production)
- **RAM**: 2GB minimum (4GB recommended)
- **Disk**: 10GB minimum (50GB recommended for logs and data)
- **Network**: Stable internet connection with low latency to broker API

### Software Requirements
- **OS**: Linux (Ubuntu 20.04 LTS or newer recommended)
  - Also supports: CentOS 7+, Debian 10+
  - Windows Server 2019+ supported but not recommended
- **Python**: 3.12+ (3.12.3 tested and recommended)
- **Systemd**: For service management (Linux)

### Network Requirements
- **Outbound**: HTTPS (443) to broker API endpoints
- **Inbound** (optional): Port 8080 for health checks (configurable)
- **Firewall**: Allow outbound connections to broker
- **Latency**: <100ms to broker API (lower is better for live trading)

---

## Pre-Deployment

### 1. Verify Prerequisites

```bash
# Check Python version (must be 3.12+)
python3 --version

# Check available disk space (need 10GB+)
df -h

# Check available RAM (need 2GB+)
free -h

# Verify internet connectivity
ping -c 3 8.8.8.8
```

### 2. Create Deployment User (Linux)

```bash
# Create dedicated user for running the bot
sudo useradd -r -s /bin/bash -d /opt/vwap-bot -m vwap-bot

# Create necessary directories
sudo mkdir -p /opt/vwap-bot/{logs,data,backups}
sudo chown -R vwap-bot:vwap-bot /opt/vwap-bot
```

### 3. Clone Repository

```bash
# Switch to bot user
sudo su - vwap-bot

# Clone the repository
cd /opt/vwap-bot
git clone https://github.com/Quotraders/simple-bot.git
cd simple-bot
```

---

## Installation

### 1. Create Virtual Environment

```bash
# Run the installation script
./scripts/install.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate

# Install pinned dependencies
pip install --upgrade pip
pip install -r requirements-pinned.txt
```

### 2. Verify Installation

```bash
# Activate virtual environment
source venv/bin/activate

# Run tests
python -m pytest test_backtest_monitoring.py -v

# Verify all modules load
python -c "import vwap_bounce_bot; import monitoring; import backtesting; print('OK')"
```

---

## Configuration

### 1. Environment-Specific Configuration

Create separate configuration for each environment:

**Development** (`.env.development`):
```bash
BOT_ENVIRONMENT=development
BOT_DRY_RUN=true
BOT_MAX_TRADES_PER_DAY=5
BOT_DAILY_LOSS_LIMIT=50
TOPSTEP_API_TOKEN=dev_token_here
```

**Staging** (`.env.staging`):
```bash
BOT_ENVIRONMENT=staging
BOT_DRY_RUN=true
BOT_MAX_TRADES_PER_DAY=3
BOT_DAILY_LOSS_LIMIT=100
TOPSTEP_API_TOKEN=staging_token_here
```

**Production** (`.env.production`):
```bash
BOT_ENVIRONMENT=production
BOT_DRY_RUN=false
BOT_MAX_TRADES_PER_DAY=3
BOT_DAILY_LOSS_LIMIT=200
TOPSTEP_API_TOKEN=prod_token_here
CONFIRM_LIVE_TRADING=1
```

### 2. Secure Credentials

```bash
# Set proper permissions on environment files
chmod 600 .env.production
chmod 600 .env.staging

# Never commit credentials to git
echo ".env.*" >> .gitignore
```

### 3. Validate Configuration

```bash
# Run configuration validation
./scripts/validate-config.sh production

# Or manually:
python -c "from config import load_config; config = load_config('production'); config.validate(); print('Config OK')"
```

---

## Process Management

### Linux (Systemd)

#### 1. Install Service File

```bash
# Copy service file
sudo cp deployment/vwap-bot.service /etc/systemd/system/

# Edit service file to match your setup
sudo nano /etc/systemd/system/vwap-bot.service

# Reload systemd
sudo systemctl daemon-reload
```

#### 2. Enable and Start Service

```bash
# Enable auto-start on boot
sudo systemctl enable vwap-bot

# Start the service
sudo systemctl start vwap-bot

# Check status
sudo systemctl status vwap-bot

# View logs
sudo journalctl -u vwap-bot -f
```

#### 3. Service Management Commands

```bash
# Stop service
sudo systemctl stop vwap-bot

# Restart service
sudo systemctl restart vwap-bot

# Disable auto-start
sudo systemctl disable vwap-bot
```

### Windows (Optional)

See `deployment/windows-service.md` for Windows deployment instructions.

---

## Deployment Checklist

### Pre-Deployment Validation

- [ ] Server meets hardware requirements (CPU, RAM, disk)
- [ ] Python 3.12+ installed and verified
- [ ] Network connectivity to broker verified
- [ ] Firewall rules configured
- [ ] Deployment user created (Linux)
- [ ] Repository cloned to correct location

### Installation Validation

- [ ] Virtual environment created successfully
- [ ] All dependencies installed (pinned versions)
- [ ] Unit tests pass (28/28 tests)
- [ ] Module imports successful
- [ ] Configuration files created for all environments

### Configuration Validation

- [ ] Environment-specific config files created
- [ ] API credentials verified and working
- [ ] File permissions set correctly (600 for .env files)
- [ ] Configuration validation passes
- [ ] Dry-run mode tested successfully

### Deployment Execution

- [ ] Systemd service file installed (Linux)
- [ ] Service enabled for auto-start
- [ ] Service starts without errors
- [ ] Health check endpoint responding (if enabled)
- [ ] Logs being written correctly
- [ ] No errors in initial 5 minutes of operation

### Post-Deployment Verification

- [ ] Bot connects to broker API successfully
- [ ] Market data being received
- [ ] Health check returns 200 OK
- [ ] Alerts configured and tested
- [ ] Monitoring dashboard accessible (if configured)
- [ ] Backup procedure tested
- [ ] Rollback procedure documented and tested

### Success Criteria

- [ ] Bot runs for 24 hours without crashes
- [ ] All configured alerts working
- [ ] Logs rotating correctly
- [ ] Service restarts automatically after reboot
- [ ] Performance metrics within acceptable ranges
- [ ] Team can access logs and metrics

---

## Monitoring

### Health Checks

```bash
# Check service status
sudo systemctl status vwap-bot

# Check health endpoint
curl http://localhost:8080/health

# View recent logs
sudo journalctl -u vwap-bot -n 100

# Monitor live logs
sudo journalctl -u vwap-bot -f
```

### Log Locations

- **Service Logs**: `/var/log/vwap-bot/` or via `journalctl`
- **Application Logs**: `/opt/vwap-bot/simple-bot/logs/`
- **Audit Logs**: `/opt/vwap-bot/simple-bot/logs/audit.log`
- **Performance Logs**: `/opt/vwap-bot/simple-bot/logs/performance.log`

### Key Metrics to Monitor

- **Service uptime**: Should be continuous
- **API latency**: Should be <100ms average
- **Daily P&L**: Track against limits
- **Error rate**: Should be <1%
- **Memory usage**: Should be stable
- **Disk usage**: Logs should rotate properly

---

## Rollback Procedure

### Quick Rollback

```bash
# Stop the service
sudo systemctl stop vwap-bot

# Switch to previous version
cd /opt/vwap-bot/simple-bot
git checkout <previous-commit-hash>

# Restore previous environment file (if needed)
cp /opt/vwap-bot/backups/.env.production.backup .env.production

# Restart service
sudo systemctl start vwap-bot

# Verify status
sudo systemctl status vwap-bot
curl http://localhost:8080/health
```

### Full Rollback with Backup

```bash
# Stop service
sudo systemctl stop vwap-bot

# Restore from backup
cd /opt/vwap-bot
rm -rf simple-bot
tar -xzf backups/simple-bot-backup-YYYYMMDD.tar.gz

# Restore environment
cp backups/.env.production simple-bot/

# Restart service
cd simple-bot
sudo systemctl start vwap-bot
```

### Rollback Validation

```bash
# Verify service is running
sudo systemctl status vwap-bot

# Check logs for errors
sudo journalctl -u vwap-bot -n 50

# Verify health endpoint
curl http://localhost:8080/health

# Monitor for 10 minutes to ensure stability
watch -n 10 'curl -s http://localhost:8080/health | jq .'
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check service logs
sudo journalctl -u vwap-bot -n 100 --no-pager

# Check permissions
ls -la /opt/vwap-bot/simple-bot

# Verify Python environment
sudo -u vwap-bot /opt/vwap-bot/simple-bot/venv/bin/python --version

# Test manual start
sudo -u vwap-bot bash
cd /opt/vwap-bot/simple-bot
source venv/bin/activate
python main.py --mode live --dry-run
```

### High Memory Usage

```bash
# Check current memory usage
ps aux | grep python

# Review log file sizes
du -sh /opt/vwap-bot/simple-bot/logs/*

# Rotate logs manually if needed
sudo systemctl restart vwap-bot
```

### Connection Issues

```bash
# Test broker connectivity
ping broker.api.endpoint

# Check firewall rules
sudo iptables -L -n

# Verify API token
python -c "import os; from config import load_config; c = load_config('production'); print('Token configured' if c.api_token else 'No token')"
```

---

## Support

For issues or questions:
- Review logs: `sudo journalctl -u vwap-bot -f`
- Check health: `curl http://localhost:8080/health`
- Consult README.md and BACKTESTING_MONITORING.md

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-29  
**Maintained By**: Development Team
