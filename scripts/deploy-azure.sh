#!/bin/bash
# Quick setup script for Azure Container deployment
# This script automates the deployment of QuoTrading Bot to Azure Container Apps

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}QuoTrading Bot - Azure Container Setup${NC}"
echo -e "${GREEN}==================================================${NC}"
echo ""

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo -e "${RED}Error: Azure CLI is not installed${NC}"
    echo "Please install Azure CLI: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Login to Azure
echo -e "${YELLOW}Checking Azure login status...${NC}"
if ! az account show &> /dev/null; then
    echo -e "${YELLOW}Please login to Azure${NC}"
    az login
fi

echo ""
echo -e "${GREEN}Step 1: Configuration${NC}"
echo "Please provide the following information:"
echo ""

# Get configuration from user
read -p "Resource Group name [quotrading-rg]: " RESOURCE_GROUP
RESOURCE_GROUP=${RESOURCE_GROUP:-quotrading-rg}

read -p "Azure Location [eastus]: " LOCATION
LOCATION=${LOCATION:-eastus}

read -p "Container Registry name [quotradingacr]: " ACR_NAME
ACR_NAME=${ACR_NAME:-quotradingacr}

read -p "PostgreSQL Server name [quotrading-db]: " POSTGRES_SERVER
POSTGRES_SERVER=${POSTGRES_SERVER:-quotrading-db}

read -p "PostgreSQL Admin username [quotradingadmin]: " POSTGRES_USER
POSTGRES_USER=${POSTGRES_USER:-quotradingadmin}

read -sp "PostgreSQL Admin password: " POSTGRES_PASSWORD
echo ""

if [ -z "$POSTGRES_PASSWORD" ]; then
    echo -e "${RED}Error: PostgreSQL password cannot be empty${NC}"
    exit 1
fi

# Confirm settings
echo ""
echo -e "${YELLOW}Deployment Configuration:${NC}"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Location: $LOCATION"
echo "  Container Registry: $ACR_NAME"
echo "  PostgreSQL Server: $POSTGRES_SERVER"
echo "  PostgreSQL User: $POSTGRES_USER"
echo ""
read -p "Continue with these settings? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Deployment cancelled"
    exit 0
fi

# Create resource group
echo ""
echo -e "${GREEN}Step 2: Creating Resource Group${NC}"
if az group exists --name $RESOURCE_GROUP | grep -q true; then
    echo "Resource group $RESOURCE_GROUP already exists"
else
    az group create --name $RESOURCE_GROUP --location $LOCATION
    echo -e "${GREEN}✓ Resource group created${NC}"
fi

# Create Azure Container Registry
echo ""
echo -e "${GREEN}Step 3: Creating Azure Container Registry${NC}"
if az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo "Container Registry $ACR_NAME already exists"
else
    az acr create \
      --resource-group $RESOURCE_GROUP \
      --name $ACR_NAME \
      --sku Basic \
      --admin-enabled true
    echo -e "${GREEN}✓ Container Registry created${NC}"
fi

# Create PostgreSQL Flexible Server
echo ""
echo -e "${GREEN}Step 4: Creating PostgreSQL Flexible Server${NC}"
echo "This may take several minutes..."
if az postgres flexible-server show --name $POSTGRES_SERVER --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo "PostgreSQL server $POSTGRES_SERVER already exists"
else
    az postgres flexible-server create \
      --resource-group $RESOURCE_GROUP \
      --name $POSTGRES_SERVER \
      --location $LOCATION \
      --admin-user $POSTGRES_USER \
      --admin-password "$POSTGRES_PASSWORD" \
      --sku-name Standard_B1ms \
      --tier Burstable \
      --version 15 \
      --storage-size 32 \
      --public-access 0.0.0.0
    
    # Create database
    az postgres flexible-server db create \
      --resource-group $RESOURCE_GROUP \
      --server-name $POSTGRES_SERVER \
      --database-name quotrading
    
    # Allow Azure services
    az postgres flexible-server firewall-rule create \
      --resource-group $RESOURCE_GROUP \
      --name $POSTGRES_SERVER \
      --rule-name AllowAzureServices \
      --start-ip-address 0.0.0.0 \
      --end-ip-address 0.0.0.0
    
    echo -e "${GREEN}✓ PostgreSQL server created${NC}"
fi

# Build and push Docker images
echo ""
echo -e "${GREEN}Step 5: Building and Pushing Docker Images${NC}"

# Login to ACR
az acr login --name $ACR_NAME

# Build signal engine
echo "Building signal engine image..."
docker build -t $ACR_NAME.azurecr.io/signal-engine:latest -f cloud-api/Dockerfile .
docker push $ACR_NAME.azurecr.io/signal-engine:latest
echo -e "${GREEN}✓ Signal engine image pushed${NC}"

# Build trading bot (optional)
read -p "Build and push trading bot image? (yes/no): " BUILD_BOT
if [ "$BUILD_BOT" = "yes" ]; then
    echo "Building trading bot image..."
    docker build -t $ACR_NAME.azurecr.io/trading-bot:latest -f Dockerfile.bot .
    docker push $ACR_NAME.azurecr.io/trading-bot:latest
    echo -e "${GREEN}✓ Trading bot image pushed${NC}"
fi

# Create Container Apps Environment
echo ""
echo -e "${GREEN}Step 6: Creating Container Apps Environment${NC}"

# Install/upgrade containerapp extension
az extension add --name containerapp --upgrade --yes

if az containerapp env show --name quotrading-env --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo "Container Apps environment already exists"
else
    az containerapp env create \
      --name quotrading-env \
      --resource-group $RESOURCE_GROUP \
      --location $LOCATION
    echo -e "${GREEN}✓ Container Apps environment created${NC}"
fi

# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)

# Deploy Signal Engine
echo ""
echo -e "${GREEN}Step 7: Deploying Signal Engine${NC}"

if az containerapp show --name signal-engine --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo "Updating existing signal-engine container app..."
    az containerapp update \
      --name signal-engine \
      --resource-group $RESOURCE_GROUP \
      --image $ACR_NAME.azurecr.io/signal-engine:latest
else
    echo "Creating signal-engine container app..."
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
        DB_HOST=$POSTGRES_SERVER.postgres.database.azure.com \
        DB_NAME=quotrading \
        DB_USER=$POSTGRES_USER \
        ENVIRONMENT=production \
      --secrets \
        db-password="$POSTGRES_PASSWORD"
fi

# Get Signal Engine URL
SIGNAL_ENGINE_URL=$(az containerapp show \
  --name signal-engine \
  --resource-group $RESOURCE_GROUP \
  --query properties.configuration.ingress.fqdn -o tsv)

echo -e "${GREEN}✓ Signal engine deployed${NC}"

# Deploy Trading Bot (optional)
if [ "$BUILD_BOT" = "yes" ]; then
    echo ""
    echo -e "${GREEN}Step 8: Deploying Trading Bot${NC}"
    
    read -p "Enter TOPSTEP_API_TOKEN: " TOPSTEP_TOKEN
    read -p "Enter TOPSTEP_USERNAME: " TOPSTEP_USERNAME
    
    if az containerapp show --name trading-bot --resource-group $RESOURCE_GROUP &> /dev/null; then
        echo "Updating existing trading-bot container app..."
        az containerapp update \
          --name trading-bot \
          --resource-group $RESOURCE_GROUP \
          --image $ACR_NAME.azurecr.io/trading-bot:latest
    else
        echo "Creating trading-bot container app..."
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
          --secrets \
            topstep-token="$TOPSTEP_TOKEN" \
            topstep-username="$TOPSTEP_USERNAME"
    fi
    
    echo -e "${GREEN}✓ Trading bot deployed${NC}"
fi

# Summary
echo ""
echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}==================================================${NC}"
echo ""
echo "Resources created:"
echo "  - Resource Group: $RESOURCE_GROUP"
echo "  - Container Registry: $ACR_NAME.azurecr.io"
echo "  - PostgreSQL Server: $POSTGRES_SERVER.postgres.database.azure.com"
echo "  - Signal Engine URL: https://$SIGNAL_ENGINE_URL"
echo ""
echo "Next steps:"
echo "  1. Test the signal engine: curl https://$SIGNAL_ENGINE_URL/health"
echo "  2. Initialize the database: ./scripts/init-azure-db.sh"
echo "  3. Configure monitoring and alerts"
echo "  4. Set up custom domain (optional)"
echo ""
echo -e "${YELLOW}Important: Save your credentials securely!${NC}"
echo ""
