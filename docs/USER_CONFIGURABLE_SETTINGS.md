# User Configurable Settings

## Overview
This bot is designed for subscription-based deployment. Users can customize their **risk management** and **broker settings**, while the core trading logic remains optimized for best performance.

---

## ✅ USER CONFIGURABLE Parameters

### **1. Risk Management** (Primary Settings)
These are the MAIN settings your customers will adjust:

| Parameter | Description | Default | Range | Example |
|-----------|-------------|---------|-------|---------|
| `max_contracts` | Maximum contracts per trade | 3 | 1-25 | 5 |
| `max_trades_per_day` | Daily trade limit | 9999 (unlimited) | 1-9999 | 10 |
| `risk_per_trade` | Risk per trade as % of account | 0.012 (1.2%) | 0.005-0.02 | 0.01 (1%) |
| `daily_loss_limit` | Max $ loss per day | Auto-calculated | 100-5000 | 1000 |
| `daily_loss_percent` | Max daily loss as % | 2.0% | 1.0-5.0% | 2.5% |
| `max_drawdown_percent` | Max account drawdown % | 4.0% | 2.0-10.0% | 5.0% |
| `auto_calculate_limits` | Auto-calc limits from balance | true | true/false | false |

### **2. Broker Configuration**
Customers can use different brokers:

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `broker` | Broker name | "TopStep" | TopStep, Tradovate, Rithmic, NinjaTrader, Interactive Brokers, etc. |
| `api_token` | Broker API token | "" | (broker-specific) |
| `username` | Broker username/email | "" | user@example.com |

### **3. RL/AI Customization**
Advanced users can tune the AI layer:

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `rl_confidence_threshold` | Minimum signal confidence | 0.65 (65%) | 0.50-0.80 |
| `rl_exploration_rate` | Learning exploration % | 0.05 (5%) | 0.01-0.20 |
| `rl_min_exploration_rate` | Minimum exploration | 0.05 (5%) | 0.01-0.10 |
| `rl_exploration_decay` | Exploration decay rate | 0.995 | 0.99-1.0 |

---

## ❌ NOT CONFIGURABLE (Optimized for Performance)

These parameters are **locked** because they're optimized for best results:

### **Trading Logic** (DO NOT CHANGE)
- **Confidence level thresholds** (30%, 50%, 70%, 85%) - optimized position sizing tiers
- **Early loss cut** (75% of stop distance) - proven exit strategy
- **Volatility regime detection** (1.2x high, 0.8x low) - adaptive to market conditions
- **VWAP standard deviations** (2.5, 2.1, 3.7) - proven entry/exit zones
- **RSI levels** (35 oversold, 65 overbought) - optimal mean reversion signals
- **Technical filters** (trend, RSI, VWAP direction) - proven combination

### **Time-Based Rules** (Platform-Specific)
- **Flatten time** (4:45 PM ET) - futures session management
- **Market hours** (6 PM - 5 PM ET) - ES futures schedule
- **Early exit time** (3:30 PM) - pre-close risk management

---

## 📝 How to Configure

### **Option 1: config.json** (Recommended for Admin)
```json
{
  "broker": "TopStep",
  "max_contracts": 3,
  "max_trades_per_day": 9999,
  "risk_per_trade": 0.012,
  "daily_loss_percent": 2.0,
  "max_drawdown_percent": 4.0,
  "rl_confidence_threshold": 0.65
}
```

### **Option 2: .env file** (Recommended for Customers)
```bash
# Broker Settings
BOT_BROKER=Tradovate
BOT_API_TOKEN=your_api_token_here
BOT_USERNAME=trader@email.com

# Risk Management
BOT_MAX_CONTRACTS=5
BOT_MAX_TRADES_PER_DAY=10
BOT_RISK_PER_TRADE=0.01
BOT_DAILY_LOSS_PERCENT=2.5
BOT_MAX_DRAWDOWN_PERCENT=5.0

# AI Settings (Advanced)
BOT_RL_CONFIDENCE_THRESHOLD=0.70
BOT_RL_EXPLORATION_RATE=0.05
```

### **Option 3: GUI Launcher** (Customer-Facing)
Your GUI launcher should update the `.env` file with customer selections.

**Priority**: `.env` > `config.json` > defaults

---

## 🎯 Subscription Model Best Practices

### **For Conservative Customers:**
```
max_contracts: 2
max_trades_per_day: 5
risk_per_trade: 0.01 (1%)
daily_loss_percent: 2.0%
```

### **For Aggressive Customers:**
```
max_contracts: 5
max_trades_per_day: 20
risk_per_trade: 0.015 (1.5%)
daily_loss_percent: 3.0%
```

### **For Evaluation Accounts (TopStep, etc.):**
```
broker: TopStep
daily_loss_percent: 2.0% (enforced)
max_drawdown_percent: 4.0% (enforced)
auto_calculate_limits: true
```

---

## ⚠️ Important Notes

1. **Trading Logic is Locked**: The confidence thresholds, exit logic, and technical indicators are optimized through extensive backtesting. Changing them will degrade performance.

2. **Broker-Specific Rules**: When `broker` is set to "TopStep" or "TopStepX", the bot enforces their 2%/4% rules automatically.

3. **Auto-Calculate Limits**: When `auto_calculate_limits=true`, the bot calculates `daily_loss_limit` from `daily_loss_percent` × account balance.

4. **Environment Variable Priority**: `.env` settings override `config.json`, allowing per-customer customization while keeping defaults intact.

5. **Risk Validation**: The bot validates all risk settings on startup and will reject unsafe configurations.

---

## 🚀 Customer Onboarding Checklist

- [ ] Customer creates account with their broker
- [ ] Customer receives API token from broker
- [ ] Customer sets broker name, API token, username in GUI
- [ ] Customer configures risk limits (contracts, daily trades, risk %)
- [ ] Bot validates configuration
- [ ] Bot starts trading with customer's settings
- [ ] Customer monitors performance via dashboard

---

## 📊 Example Customer Profiles

### **Profile 1: "Safe Sam" - Funded Account**
```json
{
  "broker": "TopStep",
  "max_contracts": 2,
  "max_trades_per_day": 5,
  "risk_per_trade": 0.01,
  "daily_loss_percent": 2.0,
  "max_drawdown_percent": 4.0
}
```
**Goal**: Protect funded account, conservative approach

---

### **Profile 2: "Aggressive Andy" - Evaluation Account**
```json
{
  "broker": "TopStep",
  "max_contracts": 5,
  "max_trades_per_day": 15,
  "risk_per_trade": 0.015,
  "daily_loss_percent": 2.0,
  "max_drawdown_percent": 4.0
}
```
**Goal**: Pass evaluation quickly, higher risk acceptable

---

### **Profile 3: "Pro Trader Pat" - Personal Account**
```json
{
  "broker": "Tradovate",
  "max_contracts": 10,
  "max_trades_per_day": 9999,
  "risk_per_trade": 0.02,
  "daily_loss_percent": 3.0,
  "max_drawdown_percent": 6.0,
  "rl_confidence_threshold": 0.60
}
```
**Goal**: Maximize profits, full control, experienced trader

---

## 🔒 Security Best Practices

1. **Never hardcode API tokens** - always use environment variables
2. **Store customer configs separately** - one `.env` per customer
3. **Validate all inputs** - bot has built-in validation
4. **Log all config changes** - audit trail for troubleshooting
5. **Encrypt sensitive data** - API tokens, usernames

---

**Version**: 1.0  
**Last Updated**: November 6, 2025  
**Contact**: support@quotrading.com
