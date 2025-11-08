# QuoTrading Cloud API - Azure Deployment Guide

## 🚀 Deploy to Azure using Azure CLI

This guide shows you how to deploy the QuoTrading Cloud API to Azure App Service using the Azure CLI.

### Prerequisites

1. **Azure Account**
   - Sign up at [portal.azure.com](https://portal.azure.com)
   - Get 12 months free tier + $200 credit for new accounts

2. **Azure CLI Installed**
   ```bash
   # Install Azure CLI
   # Windows (PowerShell):
   winget install Microsoft.AzureCLI
   
   # macOS:
   brew install azure-cli
   
   # Linux:
   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
   ```

3. **Login to Azure**
   ```bash
   az login
   ```

---

## 📋 Quick Deployment Steps

### Step 1: Set Configuration Variables

```bash
# Set your configuration
RESOURCE_GROUP="quotrading-rg"
LOCATION="eastus"
APP_NAME="quotrading-api"
DB_SERVER_NAME="quotrading-db-server"
DB_NAME="quotrading-db"
DB_ADMIN_USER="quotradingadmin"
DB_ADMIN_PASSWORD="YourSecurePassword123!"  # Change this!
PLAN_NAME="quotrading-plan"
```

### Step 2: Create Resource Group

```bash
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION
```

### Step 3: Create PostgreSQL Database

```bash
# Create PostgreSQL server
az postgres flexible-server create \
  --resource-group $RESOURCE_GROUP \
  --name $DB_SERVER_NAME \
  --location $LOCATION \
  --admin-user $DB_ADMIN_USER \
  --admin-password $DB_ADMIN_PASSWORD \
  --sku-name Standard_B1ms \
  --tier Burstable \
  --version 14 \
  --storage-size 32

# Create database
az postgres flexible-server db create \
  --resource-group $RESOURCE_GROUP \
  --server-name $DB_SERVER_NAME \
  --database-name $DB_NAME

# Configure firewall to allow Azure services
az postgres flexible-server firewall-rule create \
  --resource-group $RESOURCE_GROUP \
  --name $DB_SERVER_NAME \
  --rule-name AllowAzureServices \
  --start-ip-address 0.0.0.0 \
  --end-ip-address 0.0.0.0

# Get connection string
DB_CONNECTION_STRING="postgresql://$DB_ADMIN_USER:$DB_ADMIN_PASSWORD@$DB_SERVER_NAME.postgres.database.azure.com:5432/$DB_NAME?sslmode=require"
echo "Database URL: $DB_CONNECTION_STRING"
```

### Step 4: Create App Service Plan

```bash
# Create App Service Plan (B1 tier - suitable for production)
az appservice plan create \
  --resource-group $RESOURCE_GROUP \
  --name $PLAN_NAME \
  --location $LOCATION \
  --sku B1 \
  --is-linux
```

### Step 5: Create Web App

```bash
# Create Web App with Python runtime
az webapp create \
  --resource-group $RESOURCE_GROUP \
  --plan $PLAN_NAME \
  --name $APP_NAME \
  --runtime "PYTHON:3.11" \
  --deployment-local-git
```

### Step 6: Configure Application Settings

```bash
# Generate API secret
API_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

# Set environment variables
az webapp config appsettings set \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --settings \
    DATABASE_URL="$DB_CONNECTION_STRING" \
    STRIPE_SECRET_KEY="sk_test_YOUR_KEY_HERE" \
    STRIPE_WEBHOOK_SECRET="whsec_YOUR_SECRET_HERE" \
    API_SECRET_KEY="$API_SECRET" \
    SCM_DO_BUILD_DURING_DEPLOYMENT=true \
    WEBSITES_PORT=8000
```

### Step 7: Configure Startup Command

```bash
az webapp config set \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --startup-file "gunicorn main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000"
```

### Step 8: Deploy from GitHub

**Option A: Deploy from Local Git**
```bash
# Get deployment credentials
az webapp deployment list-publishing-credentials \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME

# Add Azure remote and push
cd cloud-api
git init
git add .
git commit -m "Initial Azure deployment"

# Get Git URL
GIT_URL=$(az webapp deployment source config-local-git \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --query url \
  --output tsv)

git remote add azure $GIT_URL
git push azure main
```

**Option B: Deploy from GitHub (Recommended)**
```bash
# Configure GitHub deployment
az webapp deployment source config \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --repo-url https://github.com/Quotraders/simple-bot \
  --branch main \
  --manual-integration
```

### Step 9: Enable HTTPS Only

```bash
az webapp update \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --https-only true
```

### Step 10: Test Your Deployment

```bash
# Get app URL
APP_URL=$(az webapp show \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --query defaultHostName \
  --output tsv)

echo "Your API is deployed at: https://$APP_URL"

# Test health endpoint
curl https://$APP_URL/

# Test registration
curl -X POST https://$APP_URL/api/v1/users/register \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com"}'
```

---

## 🔧 Configuration Management

### View Current Settings
```bash
az webapp config appsettings list \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --output table
```

### Update a Setting
```bash
az webapp config appsettings set \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --settings STRIPE_SECRET_KEY="sk_live_YOUR_NEW_KEY"
```

### View Logs
```bash
# Enable logging
az webapp log config \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --docker-container-logging filesystem \
  --level information

# Stream logs
az webapp log tail \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME
```

---

## 💰 Pricing Estimates

### Free Tier (Development)
- **App Service**: F1 Free tier (1 GB RAM, 60 min/day CPU)
- **PostgreSQL**: Burstable B1ms (~$12/month with free credit)
- **Total**: ~$0-12/month (with free credits)

### Production Tier
- **App Service**: B1 Basic (~$13/month)
- **PostgreSQL**: Burstable B1ms (~$12/month)
- **Total**: ~$25/month

### Scale Up Options
```bash
# Upgrade to S1 Standard (better performance)
az appservice plan update \
  --resource-group $RESOURCE_GROUP \
  --name $PLAN_NAME \
  --sku S1

# Upgrade database to General Purpose
az postgres flexible-server update \
  --resource-group $RESOURCE_GROUP \
  --name $DB_SERVER_NAME \
  --sku-name Standard_D2s_v3 \
  --tier GeneralPurpose
```

---

## 🔒 Security Best Practices

### 1. Enable Managed Identity
```bash
az webapp identity assign \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME
```

### 2. Restrict Database Access
```bash
# Remove public access
az postgres flexible-server firewall-rule delete \
  --resource-group $RESOURCE_GROUP \
  --name $DB_SERVER_NAME \
  --rule-name AllowAzureServices

# Add VNet integration (requires Premium tier)
az webapp vnet-integration add \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --vnet MyVNet \
  --subnet MySubnet
```

### 3. Use Azure Key Vault for Secrets
```bash
# Create Key Vault
az keyvault create \
  --resource-group $RESOURCE_GROUP \
  --name quotrading-vault \
  --location $LOCATION

# Store secrets
az keyvault secret set \
  --vault-name quotrading-vault \
  --name stripe-secret-key \
  --value "sk_live_YOUR_KEY"

# Reference in app settings
az webapp config appsettings set \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --settings STRIPE_SECRET_KEY="@Microsoft.KeyVault(SecretUri=https://quotrading-vault.vault.azure.net/secrets/stripe-secret-key/)"
```

---

## 📊 Monitoring & Diagnostics

### Enable Application Insights
```bash
# Create Application Insights
az monitor app-insights component create \
  --resource-group $RESOURCE_GROUP \
  --app quotrading-insights \
  --location $LOCATION \
  --application-type web

# Get instrumentation key
INSIGHTS_KEY=$(az monitor app-insights component show \
  --resource-group $RESOURCE_GROUP \
  --app quotrading-insights \
  --query instrumentationKey \
  --output tsv)

# Configure app to use it
az webapp config appsettings set \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --settings APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=$INSIGHTS_KEY"
```

### View Metrics
```bash
# View app metrics
az monitor metrics list \
  --resource-group $RESOURCE_GROUP \
  --resource $APP_NAME \
  --resource-type "Microsoft.Web/sites" \
  --metric "Requests" \
  --start-time 2025-01-01T00:00:00Z
```

---

## 🔄 CI/CD with GitHub Actions

Create `.github/workflows/azure-deploy.yml`:

```yaml
name: Deploy to Azure

on:
  push:
    branches: [ main ]
    paths:
      - 'cloud-api/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Azure Login
      uses: azure/login@v1
      with:
        creds: ${{ secrets.AZURE_CREDENTIALS }}
    
    - name: Deploy to Azure Web App
      uses: azure/webapps-deploy@v2
      with:
        app-name: 'quotrading-api'
        package: './cloud-api'
```

---

## 🐛 Troubleshooting

### App Won't Start
```bash
# Check logs
az webapp log tail --resource-group $RESOURCE_GROUP --name $APP_NAME

# Common issues:
# 1. Check DATABASE_URL format includes ?sslmode=require
# 2. Verify startup command is correct
# 3. Check requirements-azure.txt is present
```

### Database Connection Failed
```bash
# Test connection
az postgres flexible-server connect \
  --name $DB_SERVER_NAME \
  --admin-user $DB_ADMIN_USER \
  --admin-password $DB_ADMIN_PASSWORD

# Check firewall rules
az postgres flexible-server firewall-rule list \
  --resource-group $RESOURCE_GROUP \
  --name $DB_SERVER_NAME
```

### SSL/TLS Issues
```bash
# Ensure connection string includes sslmode
DATABASE_URL="postgresql://user:pass@server.postgres.database.azure.com:5432/db?sslmode=require"
```

---

## 🧹 Cleanup Resources

```bash
# Delete everything (be careful!)
az group delete \
  --name $RESOURCE_GROUP \
  --yes \
  --no-wait
```

---

## 📚 Additional Resources

- [Azure CLI Documentation](https://docs.microsoft.com/en-us/cli/azure/)
- [Azure App Service Python](https://docs.microsoft.com/en-us/azure/app-service/quickstart-python)
- [Azure PostgreSQL Flexible Server](https://docs.microsoft.com/en-us/azure/postgresql/flexible-server/)
- [Azure Pricing Calculator](https://azure.microsoft.com/en-us/pricing/calculator/)

---

## ✅ Next Steps

1. **Test API Endpoints** - Verify all endpoints work
2. **Configure Stripe** - Set up production webhook
3. **Update Bot** - Change `CLOUD_API_BASE_URL` in launcher
4. **Monitor Performance** - Use Application Insights
5. **Set Up Alerts** - Configure Azure Monitor alerts
6. **Backup Database** - Enable automated backups

---

**Built with ❤️ by QuoTrading**
