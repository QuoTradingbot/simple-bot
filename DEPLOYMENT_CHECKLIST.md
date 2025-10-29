# VWAP Bounce Bot - Deployment Checklist

Use this checklist to ensure a successful deployment. Check off each item as you complete it.

## Pre-Deployment Phase

### Server Preparation
- [ ] Server meets minimum hardware requirements (2+ CPU cores, 2GB+ RAM, 10GB+ disk)
- [ ] Operating system is Linux (Ubuntu 20.04 LTS or newer recommended)
- [ ] Python 3.12+ is installed (`python3 --version`)
- [ ] Git is installed (`git --version`)
- [ ] Network connectivity verified (`ping 8.8.8.8`)
- [ ] Firewall configured to allow outbound HTTPS (port 443)
- [ ] Server timezone set correctly (`timedatectl`)

### User & Directory Setup
- [ ] Deployment user created (`sudo useradd -r -s /bin/bash -d /opt/vwap-bot -m vwap-bot`)
- [ ] Required directories created (`/opt/vwap-bot/{logs,data,backups}`)
- [ ] Proper ownership set (`sudo chown -R vwap-bot:vwap-bot /opt/vwap-bot`)
- [ ] Repository cloned to `/opt/vwap-bot/simple-bot`

### Credentials & Configuration
- [ ] API token obtained from broker
- [ ] API token tested and verified
- [ ] Environment file created (`.env.production`)
- [ ] Configuration customized for your trading parameters
- [ ] Environment file permissions set (`chmod 600 .env.production`)
- [ ] Environment file NOT committed to git

---

## Installation Phase

### Virtual Environment
- [ ] Installation script executed (`./scripts/install.sh`)
- [ ] Virtual environment created (`venv/` directory exists)
- [ ] Dependencies installed successfully
- [ ] Pinned requirements used (`requirements-pinned.txt`)

### Validation
- [ ] Configuration validated (`./scripts/validate-config.sh production`)
- [ ] All unit tests pass (`python -m pytest test_backtest_monitoring.py`)
- [ ] Module imports successful
- [ ] No errors in validation output

---

## Configuration Phase

### Environment Configuration
- [ ] Correct environment selected (development/staging/production)
- [ ] Trading mode set correctly (`BOT_DRY_RUN=false` for live, `true` for paper)
- [ ] Live trading confirmation set (`CONFIRM_LIVE_TRADING=1` for production)
- [ ] Risk parameters reviewed and set appropriately:
  - [ ] Risk per trade (default: 1%)
  - [ ] Max contracts (default: 2)
  - [ ] Max trades per day (default: 3)
  - [ ] Daily loss limit (default: $200)
  - [ ] Max drawdown (default: 5%)
- [ ] Trading hours verified for your timezone
- [ ] Instrument symbol correct (default: MES)

### Security Review
- [ ] API token is for correct broker account (dev/staging/prod)
- [ ] `.env.*` files listed in `.gitignore`
- [ ] No credentials in source code
- [ ] Environment file has restricted permissions (600)
- [ ] Service user has minimal required permissions

---

## Service Setup Phase

### Systemd Configuration (Linux)
- [ ] Service file copied (`sudo cp deployment/vwap-bot.service /etc/systemd/system/`)
- [ ] Service file edited for your paths and environment
- [ ] Systemd daemon reloaded (`sudo systemctl daemon-reload`)
- [ ] Service enabled for auto-start (`sudo systemctl enable vwap-bot`)

### Startup Scripts
- [ ] Pre-start check script is executable (`chmod +x scripts/pre-start-check.sh`)
- [ ] Pre-start checks pass when run manually
- [ ] Graceful shutdown script is executable (`chmod +x scripts/graceful-shutdown.sh`)
- [ ] Shutdown script tested

---

## Deployment Execution Phase

### Initial Start
- [ ] Service started (`sudo systemctl start vwap-bot`)
- [ ] Service status is active (`sudo systemctl status vwap-bot`)
- [ ] No errors in service logs (`sudo journalctl -u vwap-bot -n 50`)
- [ ] Bot process is running (`ps aux | grep python`)

### First 5 Minutes
- [ ] Bot connects to broker API successfully
- [ ] Market data being received (check logs)
- [ ] No critical errors in logs
- [ ] Health check endpoint responds (`curl http://localhost:8080/health`)
- [ ] Memory usage is normal (check with `top` or `htop`)

### First Hour
- [ ] Service remains running without crashes
- [ ] Logs are being written correctly
- [ ] Log rotation is working (if applicable)
- [ ] No unexpected warnings or errors
- [ ] Performance metrics within normal ranges

---

## Post-Deployment Phase

### Monitoring Setup
- [ ] Health check endpoint accessible
- [ ] Alert handlers configured (if applicable)
- [ ] Dashboard configured (if applicable)
- [ ] Team has access to logs and metrics
- [ ] Monitoring alerts tested

### 24-Hour Verification
- [ ] Service has run for 24 hours without crashes
- [ ] Auto-restart works (test with `sudo systemctl restart vwap-bot`)
- [ ] Logs rotate properly
- [ ] No memory leaks observed
- [ ] Disk usage is stable
- [ ] All configured alerts are working

### Documentation
- [ ] Deployment documented (who, when, what version)
- [ ] Access credentials documented and secured
- [ ] Monitoring access documented
- [ ] Escalation procedures documented
- [ ] Rollback procedure documented and tested

---

## Backup & Rollback Preparation

### Backup
- [ ] Current configuration backed up
- [ ] Backup location documented
- [ ] Backup tested (can restore successfully)
- [ ] Backup retention policy defined

### Rollback Procedure
- [ ] Previous version identified (git commit hash)
- [ ] Rollback steps documented
- [ ] Rollback tested in staging
- [ ] Team knows how to execute rollback

---

## Success Criteria

Mark these items to confirm successful deployment:

### Functional
- [ ] Bot starts automatically on system reboot
- [ ] Bot connects to broker API successfully
- [ ] Market data is received in real-time
- [ ] Trading logic executes as expected (in dry-run mode first)
- [ ] Orders are placed correctly (if live trading)
- [ ] Stop losses and targets function properly
- [ ] Daily limits are enforced

### Operational
- [ ] Service runs continuously for 24+ hours without crashes
- [ ] Service restarts automatically after crashes (if any)
- [ ] Logs are captured properly
- [ ] Log rotation works correctly
- [ ] Disk space usage is under control
- [ ] Memory usage is stable
- [ ] CPU usage is acceptable

### Monitoring
- [ ] Health check endpoint returns 200 OK
- [ ] All metrics are being collected
- [ ] Alerts fire when expected
- [ ] Team can access logs and metrics
- [ ] Monitoring dashboard shows correct data (if applicable)

### Security
- [ ] Service runs as non-root user
- [ ] Environment files have proper permissions (600)
- [ ] No credentials exposed in logs
- [ ] API token is for correct environment
- [ ] Security best practices followed

---

## Sign-Off

- **Deployed By**: ___________________________
- **Deployment Date**: ___________________________
- **Environment**: [ ] Development [ ] Staging [ ] Production
- **Version/Commit**: ___________________________
- **Approved By**: ___________________________

---

## Notes

Use this section to document any issues, deviations from standard procedure, or important observations:

```
[Add your notes here]
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-29  
**Review Frequency**: Before each deployment
