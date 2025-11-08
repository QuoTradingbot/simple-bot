# QuoTrading VWAP Bounce Bot

Professional AI-powered mean reversion trading bot for futures markets.

## 🎯 Overview

An event-driven algorithmic trading system that executes high-probability mean reversion trades based on VWAP (Volume Weighted Average Price) standard deviation bands with reinforcement learning optimization.

### Key Features

- ✅ Real-time tick data processing
- ✅ VWAP deviation-based entry signals  
- ✅ Reinforcement learning for signal confidence
- ✅ Adaptive exit strategies
- ✅ TopStep/Tradovate integration
- ✅ Risk management with position sizing
- ✅ Professional GUI launcher for customers

## 📊 Performance

- **60-day backtest**: +$19,015 (+38% return)
- **Win rate**: 76%
- **Sharpe Ratio**: 11.53
- **3,480+ signal experiences**
- **216+ exit experiences**

## 📁 Repository Structure

```
simple-bot-1/
├── src/                          # Core trading bot source code
│   ├── main.py                  # Main entry point
│   ├── vwap_bounce_bot.py       # Core trading logic
│   ├── signal_confidence.py     # RL signal optimization
│   ├── adaptive_exits.py        # RL exit optimization
│   ├── broker_interface.py      # Broker API integration
│   ├── event_loop.py            # Event-driven architecture
│   ├── monitoring.py            # Performance monitoring
│   └── ...
│
├── customer/                     # Customer-facing distribution
│   ├── QuoTrading_Launcher.py   # Professional GUI launcher
│   ├── build_exe.spec           # PyInstaller build config
│   └── .gitkeep
│
├── templates/                    # Code generation templates
│   └── customer_launcher_template.py
│
├── scripts/                      # Build and utility scripts
│   ├── build_customer_version.py
│   └── prepare_customer_bot.py
│
├── docs/                         # Documentation
│   ├── BUILD_EXE_INSTRUCTIONS.md
│   ├── ENV_CONFIGURATION_GUIDE.md
│   └── POSITION_SIZING_GUIDE.md
│
├── data/                         # Runtime data (gitignored)
│   ├── historical_data/         # Backtesting data
│   ├── bot_state.json           # Bot state persistence
│   ├── exit_experience.json     # Exit RL training data
│   ├── signal_experience.json   # Signal RL training data
│   └── .gitkeep
│
├── logs/                         # Log files (gitignored)
│
├── .env.example                  # Environment template
├── requirements.txt              # Python dependencies
├── requirements-pinned.txt       # Locked dependency versions
└── run.py                        # Development entry point
```

## 🚀 Quick Start

### For Developers

1. **Clone the repository**
   ```bash
   git clone https://github.com/Quotraders/simple-bot.git
   cd simple-bot-1
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements-pinned.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your broker credentials
   ```

4. **Run the bot**
   ```bash
   python run.py
   ```

### For Customers

1. **Launch the GUI**
   ```bash
   cd customer
   python QuoTrading_Launcher.py
   ```

2. **Enter your license key** (provided via email)

3. **Configure broker credentials**
   - TopStep API Token
   - TopStep Username
   - Select your broker

4. **Configure trading settings**
   - Symbol (MES, MNQ, etc.)
   - Max contracts
   - Max trades per day
   - Risk per trade

5. **Start trading** - Bot runs in background

## 🔧 Configuration

### Environment Variables

See `.env.example` for all configuration options:

- `BROKER` - Trading broker (TopStep, Tradovate, etc.)
- `TOPSTEP_API_TOKEN` - Your TopStep API token
- `TOPSTEP_USERNAME` - Your TopStep username/email
- `SYMBOL` - Trading instrument (MES, MNQ, etc.)
- `MAX_CONTRACTS` - Maximum position size
- `MAX_TRADES_PER_DAY` - Daily trade limit
- `RISK_PER_TRADE` - Risk amount per trade ($)

### Documentation

- **[Build EXE Instructions](docs/BUILD_EXE_INSTRUCTIONS.md)** - Create customer executables
- **[Environment Guide](docs/ENV_CONFIGURATION_GUIDE.md)** - Detailed config reference
- **[Position Sizing Guide](docs/POSITION_SIZING_GUIDE.md)** - Risk management settings

## 🏗️ Building Customer Version

```bash
cd scripts
python build_customer_version.py
```

This creates a standalone executable in `customer/dist/QuoTrading_Launcher.exe`

## ☁️ Cloud Deployment

The QuoTrading Cloud API can be deployed to multiple cloud platforms:

### Deploy to Render
See [cloud-api/DEPLOYMENT.md](cloud-api/DEPLOYMENT.md) for Render deployment guide.

### Deploy to Azure
See [cloud-api/AZURE_DEPLOYMENT.md](cloud-api/AZURE_DEPLOYMENT.md) for Azure CLI deployment guide.

**Quick Azure Deployment:**
```bash
cd cloud-api
chmod +x deploy-azure.sh
./deploy-azure.sh
```

After deployment, set environment variable:
```bash
export QUOTRADING_API_URL="https://your-app.azurewebsites.net"
```

## 📝 License

Proprietary - QuoTrading LLC

## 🔑 Admin Access

For development/testing, use the admin master key:
```
QUOTRADING_ADMIN_MASTER_2025
```

This bypasses all validation and grants immediate access.

## 📧 Support

- **Email**: support@quotrading.com
- **Website**: https://quotrading.com

---

**Built with ❤️ by QuoTrading**
