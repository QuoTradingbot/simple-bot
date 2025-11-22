# QuoTrading Bot

A professional trading bot for futures markets with cloud-based signal generation and local execution capabilities.

## Features

- 🤖 **Reinforcement Learning**: AI-powered trading signals with confidence scoring
- 📊 **VWAP Strategy**: Mean reversion trading based on VWAP bands
- ☁️ **Cloud API**: Centralized signal generation and license management
- 🔄 **Adaptive Exits**: Dynamic stop-loss and take-profit management
- 📈 **Multiple Markets**: Support for ES, NQ, YM, and other futures
- 🛡️ **Risk Management**: TopStep-compliant risk controls and position sizing
- 📱 **Desktop Launcher**: Easy-to-use GUI for bot management

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Configure your settings
nano .env

# Run the bot
python -m src.main
```

### Using the Desktop Launcher

1. Download the latest release
2. Run `QuoTrading_Launcher.exe` (Windows) or launcher script
3. Configure your broker credentials
4. Start trading

## Cloud Deployment

The QuoTrading Bot includes comprehensive Azure container deployment support:

### 🚀 Deploy to Azure

Choose your deployment method:

1. **Docker Compose** (Local testing)
   ```bash
   docker-compose up -d
   ```

2. **Azure Container Apps** (Recommended for production)
   ```bash
   ./scripts/deploy-azure.sh
   ```

3. **Azure Kubernetes Service** (Enterprise)
   ```bash
   kubectl apply -f kubernetes/
   ```

### 📚 Documentation

Complete deployment guides available:

- **[Quick Start - Azure Deployment](docs/QUICK_START_AZURE.md)** - Get running on Azure in minutes
- **[Azure Container Apps Setup](docs/AZURE_CONTAINER_SETUP.md)** - Full deployment guide
- **[Kubernetes (AKS) Deployment](docs/azure/aks-deployment.md)** - Enterprise Kubernetes setup
- **[Troubleshooting Guide](docs/azure/troubleshooting.md)** - Common issues and solutions

See [docs/README.md](docs/README.md) for complete documentation index.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    QuoTrading Ecosystem                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Desktop Client          Cloud Infrastructure               │
│  ┌──────────────┐       ┌──────────────────┐               │
│  │ Trading Bot  │──────▶│  Signal Engine   │               │
│  │  (Local)     │       │   (Azure Cloud)  │               │
│  └──────┬───────┘       └────────┬─────────┘               │
│         │                         │                          │
│         ▼                         ▼                          │
│  ┌──────────────┐       ┌──────────────────┐               │
│  │   Broker     │       │   PostgreSQL     │               │
│  │   (TopStep)  │       │   + Redis Cache  │               │
│  └──────────────┘       └──────────────────┘               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Components

- **Trading Bot** (`src/`): Local bot that executes trades
- **Cloud API** (`cloud-api/`): Signal generation and license management
- **Desktop Launcher** (`launcher/`): User-friendly GUI
- **Deployment** (`kubernetes/`, `docker-compose.yml`): Cloud infrastructure

## Configuration

All configuration is managed through environment variables. See `.env.example` for available options:

### Essential Settings

```bash
# Broker Credentials
TOPSTEP_API_TOKEN=your_token
TOPSTEP_USERNAME=your_email

# Cloud API
QUOTRADING_API_URL=https://your-api.azurecontainerapps.io

# Trading Configuration
BOT_INSTRUMENT=ES
BOT_MAX_CONTRACTS=3
BOT_DRY_RUN=false

# Risk Management
BOT_USE_TOPSTEP_RULES=true
BOT_DAILY_LOSS_PERCENT=2.0
BOT_MAX_DRAWDOWN_PERCENT=4.0
```

See `.env.example` for complete configuration options.

## Risk Management

The bot includes comprehensive risk controls:

- ✅ Daily loss limits
- ✅ Maximum drawdown protection
- ✅ Position size limits
- ✅ Trading hour restrictions
- ✅ Maximum trades per day
- ✅ TopStep rule compliance

**Always test in paper trading mode first!**

## Development

### Project Structure

```
simple-bot/
├── src/                    # Trading bot source code
│   ├── main.py            # Main entry point
│   ├── broker_interface.py
│   ├── quotrading_engine.py
│   └── ...
├── cloud-api/             # Cloud signal engine
│   ├── Dockerfile
│   └── database.py
├── launcher/              # Desktop GUI
├── kubernetes/            # Kubernetes manifests
├── docs/                  # Documentation
├── scripts/               # Deployment scripts
├── docker-compose.yml     # Local development
└── .github/workflows/     # CI/CD pipelines
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python test_market_conditions.py
```

### Building for Production

```bash
# Build Docker images
docker build -t quotrading/signal-engine -f cloud-api/Dockerfile .
docker build -t quotrading/trading-bot -f Dockerfile.bot .

# Build Windows launcher
cd launcher
python build_exe.py
```

## Deployment Options

### Cloud Services Comparison

| Feature | Container Apps | ACI | AKS |
|---------|---------------|-----|-----|
| **Setup Time** | 15 min | 10 min | 1-2 hours |
| **Cost (monthly)** | $45-300 | $30-50 | $300-1000+ |
| **Auto-scaling** | ✅ Yes | ❌ No | ✅ Yes |
| **Best For** | Production | Simple deployments | Enterprise |
| **Complexity** | Low | Very Low | High |

See [docs/README.md](docs/README.md) for detailed comparisons.

### Infrastructure as Code

All deployment configurations are included:

- ✅ Docker Compose for local dev
- ✅ Kubernetes manifests for AKS
- ✅ GitHub Actions for CI/CD
- ✅ Automated deployment scripts

## Monitoring

### Health Checks

```bash
# Check signal engine health
curl https://your-api.azurecontainerapps.io/health

# View container logs
az containerapp logs show --name signal-engine --resource-group quotrading-rg

# Kubernetes monitoring
kubectl get pods -n quotrading
kubectl logs -f -n quotrading -l app=signal-engine
```

### Metrics

The bot tracks:
- Trade performance (win rate, profit factor, etc.)
- Risk metrics (drawdown, exposure)
- System health (uptime, API latency)
- RL confidence scores

## Security

### Best Practices Implemented

- 🔒 TLS/HTTPS encryption
- 🔐 Secrets stored in Azure Key Vault / Kubernetes Secrets
- 🛡️ Network isolation options
- 🔑 Azure RBAC for access control
- 📝 Comprehensive audit logging
- 🔄 Automated security updates

See deployment guides for security configuration details.

## Support

### Documentation

- [Azure Deployment Quick Start](docs/QUICK_START_AZURE.md)
- [Complete Azure Setup Guide](docs/AZURE_CONTAINER_SETUP.md)
- [Troubleshooting Guide](docs/azure/troubleshooting.md)
- [Kubernetes Deployment](docs/azure/aks-deployment.md)

### Getting Help

1. Check the [Troubleshooting Guide](docs/azure/troubleshooting.md)
2. Review logs and error messages
3. Open an issue with diagnostic information
4. Contact support with deployment details

## License

Proprietary - QuoTrading Inc.

This software requires a valid license key for operation. Contact sales@quotrading.com for licensing information.

## Disclaimer

**IMPORTANT**: Trading futures involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. This software is provided "as is" without warranty. Always test thoroughly in a paper trading environment before risking real capital.

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

See development guides in `docs/` for more details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

---

**Ready to deploy?** Start with the [Azure Quick Start Guide](docs/QUICK_START_AZURE.md)!
