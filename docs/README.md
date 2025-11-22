# QuoTrading Bot - Azure Deployment Documentation

This directory contains comprehensive documentation for deploying the QuoTrading Bot to Microsoft Azure using various container services.

## 📚 Documentation Index

### Quick Start
- **[Quick Start Guide](QUICK_START_AZURE.md)** - Get running on Azure in minutes

### Deployment Guides
- **[Azure Container Setup](AZURE_CONTAINER_SETUP.md)** - Complete guide for Azure Container Apps deployment (recommended)
- **[AKS Deployment](azure/aks-deployment.md)** - Azure Kubernetes Service deployment for enterprise use
- **[Troubleshooting](azure/troubleshooting.md)** - Common issues and solutions

### Configuration Files
- **[Kubernetes Manifests](../kubernetes/README.md)** - Kubernetes YAML files for AKS deployment
- **[Docker Compose](../docker-compose.yml)** - Local development setup
- **[GitHub Actions Workflow](../.github/workflows/azure-deploy.yml)** - CI/CD pipeline

## 🚀 Choose Your Deployment Method

### 1. Local Development (Start Here)
**Best for:** Testing and development
**Time:** 5 minutes
**Cost:** Free

```bash
docker-compose up -d
```

See: [Quick Start Guide](QUICK_START_AZURE.md#option-1-local-development-fastest)

### 2. Azure Container Apps (Recommended)
**Best for:** Most production deployments
**Time:** 15-30 minutes
**Cost:** ~$45-300/month
**Features:**
- Serverless scaling (0-N instances)
- Built-in HTTPS and DNS
- Simplified management
- Pay only for what you use

See: [Azure Container Setup Guide](AZURE_CONTAINER_SETUP.md)

### 3. Azure Container Instances
**Best for:** Simple, single-instance deployments
**Time:** 10 minutes
**Cost:** ~$30-50/month
**Features:**
- Simple VM-based containers
- Fixed resources
- No orchestration overhead

See: [Azure Container Setup Guide - Option 2](AZURE_CONTAINER_SETUP.md#option-2-azure-container-instances-aci)

### 4. Azure Kubernetes Service (AKS)
**Best for:** Enterprise deployments with complex requirements
**Time:** 1-2 hours
**Cost:** ~$300-1000+/month
**Features:**
- Full Kubernetes orchestration
- Advanced networking
- Multi-region support
- Custom configurations

See: [AKS Deployment Guide](azure/aks-deployment.md)

## 📖 Documentation Structure

```
docs/
├── QUICK_START_AZURE.md          # Start here!
├── AZURE_CONTAINER_SETUP.md      # Main deployment guide
└── azure/
    ├── aks-deployment.md         # Kubernetes deployment
    └── troubleshooting.md        # Problem solving

Root directory:
├── docker-compose.yml            # Local development
├── Dockerfile.bot                # Trading bot container
├── .env.docker                   # Docker environment template
├── kubernetes/                   # Kubernetes manifests
│   ├── README.md
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secrets.yaml.template
│   ├── signal-engine-deployment.yaml
│   ├── trading-bot-statefulset.yaml
│   ├── ingress.yaml
│   └── hpa.yaml
├── scripts/
│   └── deploy-azure.sh          # Automated deployment script
└── .github/workflows/
    └── azure-deploy.yml         # CI/CD pipeline
```

## 🎯 Getting Started Checklist

- [ ] Read the [Quick Start Guide](QUICK_START_AZURE.md)
- [ ] Choose your deployment method
- [ ] Set up Azure subscription and CLI
- [ ] Test locally with Docker Compose
- [ ] Deploy to Azure
- [ ] Configure monitoring
- [ ] Set up backups
- [ ] Implement CI/CD

## 🔧 Prerequisites

Before deploying to Azure, ensure you have:

1. **Azure Account**
   - Active Azure subscription
   - Sufficient quota for your deployment type

2. **Tools Installed**
   ```bash
   # Azure CLI
   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
   
   # Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   
   # kubectl (for AKS)
   az aks install-cli
   ```

3. **Credentials Ready**
   - Database password
   - Stripe API keys (for payments)
   - Broker API credentials (for trading)

## 💰 Cost Estimates

### Development/Testing
- Local Docker: **Free**
- Container Apps (Basic): **~$45/month**

### Production
- Container Apps: **~$100-300/month**
- AKS: **~$300-1000+/month**

Costs include:
- Container hosting
- PostgreSQL database
- Container registry
- Data transfer
- Optional: Redis cache, monitoring

Use the [Azure Pricing Calculator](https://azure.microsoft.com/en-us/pricing/calculator/) for detailed estimates.

## 🔒 Security Considerations

All deployment guides include:
- ✅ HTTPS/TLS encryption
- ✅ Secrets management
- ✅ Network isolation options
- ✅ Database SSL connections
- ✅ Container security scanning
- ✅ Azure RBAC integration

See individual guides for detailed security configurations.

## 📊 Monitoring & Logging

All deployments include:
- Health check endpoints
- Container logs
- Azure Monitor integration
- Application Insights (optional)
- Custom metrics

See [Troubleshooting Guide](azure/troubleshooting.md#monitoring-and-debugging) for details.

## 🆘 Getting Help

1. **Check Documentation**
   - [Quick Start](QUICK_START_AZURE.md)
   - [Troubleshooting](azure/troubleshooting.md)

2. **Review Logs**
   ```bash
   # Container Apps
   az containerapp logs show --name signal-engine --resource-group quotrading-rg
   
   # Kubernetes
   kubectl logs -n quotrading -l app=signal-engine
   ```

3. **Check Azure Status**
   - [Azure Status](https://status.azure.com/)
   - Service health in Azure Portal

4. **Community Support**
   - GitHub Issues
   - Azure Support

## 🔄 Updates & Maintenance

### Updating Containers

**Container Apps:**
```bash
az containerapp update \
  --name signal-engine \
  --resource-group quotrading-rg \
  --image quotradingacr.azurecr.io/signal-engine:v2.0
```

**Kubernetes:**
```bash
kubectl set image deployment/signal-engine \
  signal-engine=quotradingacr.azurecr.io/signal-engine:v2.0 \
  -n quotrading
```

### Database Maintenance
- Automated backups (configurable retention)
- Point-in-time restore
- Automated patching

See deployment guides for configuration details.

## 📝 Contributing

When updating documentation:
1. Keep guides practical and example-driven
2. Test all commands before documenting
3. Include troubleshooting tips
4. Update this index when adding new guides

## 🔗 External Resources

### Microsoft Azure
- [Azure Container Apps Documentation](https://learn.microsoft.com/en-us/azure/container-apps/)
- [Azure Kubernetes Service Documentation](https://learn.microsoft.com/en-us/azure/aks/)
- [PostgreSQL Flexible Server Documentation](https://learn.microsoft.com/en-us/azure/postgresql/flexible-server/)
- [Azure Container Registry Documentation](https://learn.microsoft.com/en-us/azure/container-registry/)

### Docker & Kubernetes
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

### Tools
- [Azure CLI Reference](https://learn.microsoft.com/en-us/cli/azure/)
- [kubectl Reference](https://kubernetes.io/docs/reference/kubectl/)

---

**Ready to deploy?** Start with the [Quick Start Guide](QUICK_START_AZURE.md)!
