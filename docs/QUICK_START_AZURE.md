# Azure Container Deployment - Quick Start Guide

Get your QuoTrading Bot running on Azure in minutes!

## Overview

This repository includes everything you need to deploy the QuoTrading Bot to Azure using containers:

- **Docker Compose** for local development
- **Azure Container Apps** for serverless container deployment (recommended)
- **Azure Container Instances** for simple VM-based containers
- **Azure Kubernetes Service** for enterprise-grade orchestration

## Prerequisites

- Azure subscription
- Azure CLI installed: `curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash`
- Docker installed: `curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh`

## Option 1: Local Development (Fastest)

Perfect for testing before deploying to Azure.

```bash
# 1. Copy environment file
cp .env.docker .env

# 2. Edit with your credentials
nano .env

# 3. Start all services
docker-compose up -d

# 4. Check status
docker-compose ps
docker-compose logs -f signal-engine

# 5. Test the API
curl http://localhost:8000/health
```

## Option 2: Azure Container Apps (Recommended for Production)

Automated deployment script handles everything.

```bash
# 1. Login to Azure
az login

# 2. Run deployment script
./scripts/deploy-azure.sh

# 3. Follow the prompts
# The script will:
# - Create all Azure resources
# - Build and push Docker images
# - Deploy containers
# - Configure networking
```

**Manual deployment:**

See the complete guide at [docs/AZURE_CONTAINER_SETUP.md](docs/AZURE_CONTAINER_SETUP.md)

## Option 3: Azure Kubernetes Service (For Advanced Users)

Full Kubernetes deployment for enterprise needs.

```bash
# 1. Create AKS cluster
./scripts/setup-aks.sh  # (see docs/azure/aks-deployment.md)

# 2. Update configuration
nano kubernetes/secrets.yaml
nano kubernetes/configmap.yaml

# 3. Deploy to AKS
kubectl apply -f kubernetes/

# 4. Check status
kubectl get pods -n quotrading
```

See the full AKS guide at [docs/azure/aks-deployment.md](docs/azure/aks-deployment.md)

## What Gets Deployed

### Infrastructure
- **Azure Container Registry**: Stores your Docker images
- **PostgreSQL Flexible Server**: Database for RL experiences and user data
- **Redis Cache** (optional): For caching and session management
- **Container Apps Environment**: Managed container hosting

### Applications
- **Signal Engine**: Cloud API for trading signals (auto-scales 1-10 replicas)
- **Trading Bot**: Executes trades based on signals (1+ instances)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Your Client                           │
│                     (Trading Bot Instance)                   │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTPS
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Azure Container Apps / AKS                      │
│                                                               │
│  ┌──────────────┐       ┌──────────────┐                   │
│  │Signal Engine │◄─────►│ Trading Bots │                   │
│  │   (API)      │       │  (Multiple)  │                   │
│  └──────┬───────┘       └──────────────┘                   │
│         │                                                     │
│         ▼                                                     │
│  ┌──────────────┐       ┌──────────────┐                   │
│  │  PostgreSQL  │       │    Redis     │                   │
│  │   Database   │       │    Cache     │                   │
│  └──────────────┘       └──────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Required Environment Variables

```bash
# Database
DB_HOST=quotrading-db.postgres.database.azure.com
DB_NAME=quotrading
DB_USER=quotradingadmin
DB_PASSWORD=your_secure_password

# Stripe (for payments)
STRIPE_API_KEY=sk_live_xxxxx
STRIPE_WEBHOOK_SECRET=whsec_xxxxx

# Trading (optional, for bot instances)
TOPSTEP_API_TOKEN=your_token
TOPSTEP_USERNAME=your_email@example.com
```

See `.env.example` for all available options.

## Deployment Costs (Estimated)

### Basic Setup (Dev/Testing)
- Container Apps: ~$15/month
- PostgreSQL Basic: ~$25/month
- Container Registry: ~$5/month
- **Total: ~$45/month**

### Production Setup
- Container Apps (with scaling): ~$100-300/month
- PostgreSQL General Purpose: ~$150/month
- Container Registry Premium: ~$40/month
- Redis Cache: ~$15/month
- **Total: ~$305-505/month**

Use the [Azure Pricing Calculator](https://azure.microsoft.com/en-us/pricing/calculator/) for accurate estimates.

## Monitoring

### Check Deployment Status

```bash
# Container Apps
az containerapp show \
  --name signal-engine \
  --resource-group quotrading-rg

# Get logs
az containerapp logs show \
  --name signal-engine \
  --resource-group quotrading-rg \
  --tail 100

# For Kubernetes
kubectl get pods -n quotrading
kubectl logs -n quotrading -l app=signal-engine --tail=100 -f
```

### Health Endpoints

- Signal Engine: `https://your-app.azurecontainerapps.io/health`
- Database check: `https://your-app.azurecontainerapps.io/health/db`

## Troubleshooting

Common issues and solutions are documented at [docs/azure/troubleshooting.md](docs/azure/troubleshooting.md)

### Quick Checks

```bash
# Test database connection
az postgres flexible-server connect \
  --name quotrading-db \
  --admin-user quotradingadmin

# View container logs
az containerapp logs show \
  --name signal-engine \
  --resource-group quotrading-rg \
  --tail 50

# Check image in registry
az acr repository list --name quotradingacr
```

## CI/CD Setup

GitHub Actions workflow is included for automated deployments.

### Setup Steps

1. Create Azure service principal:
   ```bash
   az ad sp create-for-rbac \
     --name quotrading-github \
     --role contributor \
     --scopes /subscriptions/{subscription-id}/resourceGroups/quotrading-rg \
     --sdk-auth
   ```

2. Add to GitHub secrets as `AZURE_CREDENTIALS`

3. Push to `main` branch to trigger deployment

See `.github/workflows/azure-deploy.yml` for workflow details.

## Scaling

### Auto-scaling (Container Apps)

Automatically scales based on HTTP requests and CPU:

```bash
# Set scaling rules
az containerapp update \
  --name signal-engine \
  --resource-group quotrading-rg \
  --min-replicas 1 \
  --max-replicas 10
```

### Manual Scaling (Kubernetes)

```bash
# Scale signal engine
kubectl scale deployment/signal-engine --replicas=5 -n quotrading

# Enable HPA (Horizontal Pod Autoscaler)
kubectl apply -f kubernetes/hpa.yaml
```

## Security Best Practices

1. **Use Managed Identities** instead of connection strings
2. **Enable HTTPS only** with custom domains
3. **Implement VNET integration** for network isolation
4. **Use Azure Key Vault** for secrets management
5. **Enable Azure Defender** for container security scanning
6. **Regular updates** of base images and dependencies

## Backup and Recovery

### Database Backups

```bash
# Enable automated backups (retention: 30 days)
az postgres flexible-server update \
  --resource-group quotrading-rg \
  --name quotrading-db \
  --backup-retention 30

# Manual backup
az postgres flexible-server backup create \
  --resource-group quotrading-rg \
  --name quotrading-db \
  --backup-name manual-backup-$(date +%Y%m%d)
```

### Container Image Backups

All images in Azure Container Registry are automatically replicated and backed up.

## Additional Resources

### Documentation
- [Complete Azure Container Apps Guide](docs/AZURE_CONTAINER_SETUP.md)
- [AKS Deployment Guide](docs/azure/aks-deployment.md)
- [Troubleshooting Guide](docs/azure/troubleshooting.md)
- [Kubernetes Manifests README](kubernetes/README.md)

### Azure Documentation
- [Azure Container Apps](https://learn.microsoft.com/en-us/azure/container-apps/)
- [PostgreSQL Flexible Server](https://learn.microsoft.com/en-us/azure/postgresql/flexible-server/)
- [Azure Container Registry](https://learn.microsoft.com/en-us/azure/container-registry/)
- [Azure Kubernetes Service](https://learn.microsoft.com/en-us/azure/aks/)

### Scripts
- `scripts/deploy-azure.sh` - Automated deployment to Container Apps
- Docker Compose files for local development

## Support

For issues or questions:

1. Check the [Troubleshooting Guide](docs/azure/troubleshooting.md)
2. Review Azure service health
3. Check application logs
4. Open an issue with diagnostic information

## Next Steps

1. ✅ Choose your deployment option (local/Container Apps/AKS)
2. ✅ Configure environment variables
3. ✅ Deploy infrastructure
4. ✅ Test the deployment
5. ✅ Set up monitoring and alerts
6. ✅ Configure CI/CD pipeline
7. ✅ Implement backup strategy
8. ✅ Review security settings

---

**Ready to deploy?** Start with local development using Docker Compose, then move to Azure Container Apps when ready for production!
