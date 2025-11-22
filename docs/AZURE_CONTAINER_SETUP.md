# Azure Container Setup Guide

This guide provides comprehensive instructions for deploying the QuoTrading Bot to Azure using various container services.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Architecture Overview](#architecture-overview)
- [Deployment Options](#deployment-options)
  - [Option 1: Azure Container Apps (Recommended)](#option-1-azure-container-apps-recommended)
  - [Option 2: Azure Container Instances (ACI)](#option-2-azure-container-instances-aci)
  - [Option 3: Azure Kubernetes Service (AKS)](#option-3-azure-kubernetes-service-aks)
- [Azure Services Setup](#azure-services-setup)
- [Local Development with Docker](#local-development-with-docker)
- [CI/CD Pipeline](#cicd-pipeline)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before deploying to Azure, ensure you have:

1. **Azure Account**: Active Azure subscription
2. **Azure CLI**: Installed and authenticated
   ```bash
   # Install Azure CLI
   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
   
   # Login to Azure
   az login
   ```

3. **Docker**: Installed for local testing
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   ```

4. **Required Environment Variables**: See `.env.example` for all required variables

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Azure Cloud Infrastructure                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │  Container Apps  │      │  Container Apps  │            │
│  │  Signal Engine   │◄────►│  Trading Bot(s)  │            │
│  │  (Cloud API)     │      │  (Multiple)      │            │
│  └────────┬─────────┘      └─────────┬────────┘            │
│           │                          │                      │
│           ▼                          ▼                      │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │   PostgreSQL     │      │   Redis Cache    │            │
│  │   Flexible Srv   │      │   (Optional)     │            │
│  └──────────────────┘      └──────────────────┘            │
│                                                               │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │ Azure Container  │      │  Azure Monitor   │            │
│  │    Registry      │      │   & App Insights │            │
│  └──────────────────┘      └──────────────────┘            │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Deployment Options

### Option 1: Azure Container Apps (Recommended)

Azure Container Apps is the recommended deployment option for its simplicity, auto-scaling, and cost-effectiveness.

#### Step 1: Set up Azure Resources

```bash
# Set variables
RESOURCE_GROUP="quotrading-rg"
LOCATION="eastus"
ACR_NAME="quotradingacr"
POSTGRES_SERVER="quotrading-db"
POSTGRES_DB="quotrading"
POSTGRES_USER="quotradingadmin"

# Create resource group
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION

# Create Azure Container Registry
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Basic \
  --admin-enabled true

# Create PostgreSQL Flexible Server
az postgres flexible-server create \
  --resource-group $RESOURCE_GROUP \
  --name $POSTGRES_SERVER \
  --location $LOCATION \
  --admin-user $POSTGRES_USER \
  --admin-password "YourSecurePassword123!" \
  --sku-name Standard_B1ms \
  --tier Burstable \
  --version 15 \
  --storage-size 32 \
  --public-access 0.0.0.0

# Create database
az postgres flexible-server db create \
  --resource-group $RESOURCE_GROUP \
  --server-name $POSTGRES_SERVER \
  --database-name $POSTGRES_DB

# Allow Azure services to access PostgreSQL
az postgres flexible-server firewall-rule create \
  --resource-group $RESOURCE_GROUP \
  --name $POSTGRES_SERVER \
  --rule-name AllowAzureServices \
  --start-ip-address 0.0.0.0 \
  --end-ip-address 0.0.0.0
```

#### Step 2: Build and Push Container Images

```bash
# Login to ACR
az acr login --name $ACR_NAME

# Build and push signal engine image
docker build -t $ACR_NAME.azurecr.io/signal-engine:latest -f cloud-api/Dockerfile .
docker push $ACR_NAME.azurecr.io/signal-engine:latest

# Build and push trading bot image (optional)
docker build -t $ACR_NAME.azurecr.io/trading-bot:latest -f Dockerfile.bot .
docker push $ACR_NAME.azurecr.io/trading-bot:latest
```

#### Step 3: Create Container Apps Environment

```bash
# Install Container Apps extension
az extension add --name containerapp --upgrade

# Create Container Apps environment
az containerapp env create \
  --name quotrading-env \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)
```

#### Step 4: Deploy Signal Engine

```bash
# Create signal engine container app
az containerapp create \
  --name signal-engine \
  --resource-group $RESOURCE_GROUP \
  --environment quotrading-env \
  --image $ACR_NAME.azurecr.io/signal-engine:latest \
  --registry-server $ACR_NAME.azurecr.io \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 3 \
  --cpu 0.5 \
  --memory 1.0Gi \
  --env-vars \
    DB_HOST=quotrading-db.postgres.database.azure.com \
    DB_NAME=quotrading \
    DB_USER=quotradingadmin \
    DB_PASSWORD=secretref:db-password \
    ENVIRONMENT=production \
  --secrets \
    db-password="YourSecurePassword123!"

# Get the signal engine URL
SIGNAL_ENGINE_URL=$(az containerapp show \
  --name signal-engine \
  --resource-group $RESOURCE_GROUP \
  --query properties.configuration.ingress.fqdn -o tsv)

echo "Signal Engine URL: https://$SIGNAL_ENGINE_URL"
```

#### Step 5: Deploy Trading Bot(s)

```bash
# Create trading bot container app
az containerapp create \
  --name trading-bot \
  --resource-group $RESOURCE_GROUP \
  --environment quotrading-env \
  --image $ACR_NAME.azurecr.io/trading-bot:latest \
  --registry-server $ACR_NAME.azurecr.io \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --min-replicas 1 \
  --max-replicas 1 \
  --cpu 0.25 \
  --memory 0.5Gi \
  --env-vars \
    QUOTRADING_API_URL=https://$SIGNAL_ENGINE_URL \
    BOT_ENVIRONMENT=production \
    BOT_DRY_RUN=false \
    BOT_INSTRUMENT=ES \
    TOPSTEP_API_TOKEN=secretref:topstep-token \
    TOPSTEP_USERNAME=secretref:topstep-username \
  --secrets \
    topstep-token="your-api-token" \
    topstep-username="your-email@example.com"
```

### Option 2: Azure Container Instances (ACI)

For simpler deployments without auto-scaling:

```bash
# Create signal engine container instance
az container create \
  --resource-group $RESOURCE_GROUP \
  --name signal-engine-aci \
  --image $ACR_NAME.azurecr.io/signal-engine:latest \
  --registry-login-server $ACR_NAME.azurecr.io \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --dns-name-label quotrading-signals \
  --ports 8000 \
  --cpu 1 \
  --memory 2 \
  --environment-variables \
    DB_HOST=quotrading-db.postgres.database.azure.com \
    DB_NAME=quotrading \
    DB_USER=quotradingadmin \
    ENVIRONMENT=production \
  --secure-environment-variables \
    DB_PASSWORD="YourSecurePassword123!"

# Get the FQDN
az container show \
  --resource-group $RESOURCE_GROUP \
  --name signal-engine-aci \
  --query ipAddress.fqdn \
  --output tsv
```

### Option 3: Azure Kubernetes Service (AKS)

For production-grade deployments with advanced orchestration:

See `docs/azure/aks-deployment.md` for detailed AKS deployment instructions.

## Azure Services Setup

### Azure Container Registry (ACR)

```bash
# Create ACR with geo-replication (for production)
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Premium \
  --admin-enabled true

# Add replication (optional, for high availability)
az acr replication create \
  --registry $ACR_NAME \
  --location westus
```

### PostgreSQL Flexible Server

```bash
# Create with high availability (for production)
az postgres flexible-server create \
  --resource-group $RESOURCE_GROUP \
  --name $POSTGRES_SERVER \
  --location $LOCATION \
  --admin-user $POSTGRES_USER \
  --admin-password "YourSecurePassword123!" \
  --sku-name Standard_D2s_v3 \
  --tier GeneralPurpose \
  --version 15 \
  --storage-size 128 \
  --high-availability Enabled \
  --zone 1 \
  --standby-zone 2

# Configure backup retention
az postgres flexible-server parameter set \
  --resource-group $RESOURCE_GROUP \
  --server-name $POSTGRES_SERVER \
  --name backup_retention_days \
  --value 30
```

### Redis Cache (Optional)

```bash
# Create Redis cache
az redis create \
  --resource-group $RESOURCE_GROUP \
  --name quotrading-redis \
  --location $LOCATION \
  --sku Basic \
  --vm-size c0

# Get connection string
az redis show \
  --resource-group $RESOURCE_GROUP \
  --name quotrading-redis \
  --query hostName -o tsv
```

### Azure Monitor & Application Insights

```bash
# Create Application Insights
az monitor app-insights component create \
  --app quotrading-insights \
  --location $LOCATION \
  --resource-group $RESOURCE_GROUP \
  --application-type web

# Get instrumentation key
INSTRUMENTATION_KEY=$(az monitor app-insights component show \
  --app quotrading-insights \
  --resource-group $RESOURCE_GROUP \
  --query instrumentationKey -o tsv)

echo "Application Insights Key: $INSTRUMENTATION_KEY"
```

## Local Development with Docker

### Using Docker Compose

```bash
# Copy environment file
cp .env.docker .env

# Edit .env with your values
nano .env

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f signal-engine

# Stop all services
docker-compose down

# Start with bot
docker-compose --profile bot up -d
```

### Building Individual Containers

```bash
# Build signal engine
docker build -t quotrading/signal-engine:dev -f cloud-api/Dockerfile .

# Build trading bot
docker build -t quotrading/trading-bot:dev -f Dockerfile.bot .

# Run signal engine locally
docker run -d \
  --name signal-engine \
  -p 8000:8000 \
  -e DB_HOST=host.docker.internal \
  -e DB_NAME=quotrading \
  -e DB_USER=quotradingadmin \
  -e DB_PASSWORD=yourpassword \
  quotrading/signal-engine:dev
```

## CI/CD Pipeline

### GitHub Actions

Create `.github/workflows/azure-deploy.yml`:

```yaml
name: Deploy to Azure Container Apps

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Login to Azure
      uses: azure/login@v1
      with:
        creds: ${{ secrets.AZURE_CREDENTIALS }}
    
    - name: Login to ACR
      run: az acr login --name quotradingacr
    
    - name: Build and push signal engine
      run: |
        docker build -t quotradingacr.azurecr.io/signal-engine:${{ github.sha }} -f cloud-api/Dockerfile .
        docker push quotradingacr.azurecr.io/signal-engine:${{ github.sha }}
    
    - name: Deploy to Container Apps
      run: |
        az containerapp update \
          --name signal-engine \
          --resource-group quotrading-rg \
          --image quotradingacr.azurecr.io/signal-engine:${{ github.sha }}
```

### Azure DevOps

See `docs/azure/azure-devops-pipeline.md` for Azure DevOps configuration.

## Troubleshooting

### Common Issues

#### 1. Container fails to start

```bash
# Check container logs
az containerapp logs show \
  --name signal-engine \
  --resource-group quotrading-rg \
  --tail 50

# Or for ACI
az container logs \
  --resource-group quotrading-rg \
  --name signal-engine-aci
```

#### 2. Database connection issues

```bash
# Test PostgreSQL connectivity
az postgres flexible-server connect \
  --name quotrading-db \
  --admin-user quotradingadmin

# Check firewall rules
az postgres flexible-server firewall-rule list \
  --resource-group quotrading-rg \
  --name quotrading-db
```

#### 3. Image pull errors

```bash
# Verify ACR credentials
az acr credential show --name quotradingacr

# Test image pull
docker pull quotradingacr.azurecr.io/signal-engine:latest
```

#### 4. Performance issues

```bash
# Check metrics
az containerapp show \
  --name signal-engine \
  --resource-group quotrading-rg \
  --query properties.template.containers[0].resources

# Scale up
az containerapp update \
  --name signal-engine \
  --resource-group quotrading-rg \
  --cpu 1.0 \
  --memory 2.0Gi
```

### Health Check Endpoints

- Signal Engine: `https://your-signal-engine.azurecontainerapps.io/health`
- Database check: `https://your-signal-engine.azurecontainerapps.io/health/db`

### Monitoring

```bash
# View Application Insights metrics
az monitor app-insights metrics show \
  --app quotrading-insights \
  --resource-group quotrading-rg \
  --metric requests/count

# Set up alerts
az monitor metrics alert create \
  --name high-cpu-alert \
  --resource-group quotrading-rg \
  --scopes /subscriptions/{subscription-id}/resourceGroups/quotrading-rg/providers/Microsoft.App/containerApps/signal-engine \
  --condition "avg Percentage CPU > 80" \
  --window-size 5m \
  --evaluation-frequency 1m
```

## Security Best Practices

1. **Use Managed Identities**: Avoid storing credentials in environment variables
2. **Enable HTTPS only**: Configure custom domains with SSL certificates
3. **Network Isolation**: Use VNet integration for production
4. **Secrets Management**: Use Azure Key Vault for sensitive data
5. **Regular Updates**: Keep base images and dependencies updated

## Cost Optimization

1. **Right-size resources**: Monitor and adjust CPU/memory allocations
2. **Use auto-scaling**: Set appropriate min/max replicas
3. **Reserved instances**: For predictable workloads
4. **Development environments**: Use Basic SKUs and stop when not in use

## Next Steps

1. Review and customize environment variables in `.env.docker`
2. Set up monitoring and alerting
3. Configure backup and disaster recovery
4. Implement CI/CD pipeline
5. Set up staging environment

For more information:
- [Azure Container Apps Documentation](https://learn.microsoft.com/en-us/azure/container-apps/)
- [PostgreSQL Flexible Server Documentation](https://learn.microsoft.com/en-us/azure/postgresql/flexible-server/)
- [Azure Container Registry Documentation](https://learn.microsoft.com/en-us/azure/container-registry/)
