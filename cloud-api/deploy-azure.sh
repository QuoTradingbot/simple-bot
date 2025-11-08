#!/bin/bash

# QuoTrading Cloud API - Azure Deployment Script
# This script automates the deployment of the QuoTrading API to Azure

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    print_error "Azure CLI is not installed. Please install it first:"
    echo "  Windows: winget install Microsoft.AzureCLI"
    echo "  macOS: brew install azure-cli"
    echo "  Linux: curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash"
    exit 1
fi

print_info "Azure CLI found: $(az --version | head -n 1)"

# Check if logged in to Azure
if ! az account show &> /dev/null; then
    print_error "Not logged in to Azure. Please run: az login"
    exit 1
fi

SUBSCRIPTION=$(az account show --query name -o tsv)
print_info "Using Azure subscription: $SUBSCRIPTION"

# Configuration - Can be overridden with environment variables
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-quotrading-rg}"
LOCATION="${AZURE_LOCATION:-eastus}"
APP_NAME="${AZURE_APP_NAME:-quotrading-api-$(date +%s)}"
DB_SERVER_NAME="${AZURE_DB_SERVER:-quotrading-db-$(date +%s)}"
DB_NAME="${AZURE_DB_NAME:-quotrading_db}"
DB_ADMIN_USER="${AZURE_DB_USER:-quotradingadmin}"
PLAN_NAME="${AZURE_PLAN_NAME:-quotrading-plan}"
SKU="${AZURE_SKU:-B1}"

echo ""
print_info "Deployment Configuration:"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Location: $LOCATION"
echo "  App Name: $APP_NAME"
echo "  Database Server: $DB_SERVER_NAME"
echo "  Database Name: $DB_NAME"
echo "  App Service Plan: $PLAN_NAME"
echo "  SKU: $SKU"
echo ""

# Prompt for database password if not set
if [ -z "$AZURE_DB_PASSWORD" ]; then
    read -sp "Enter database admin password (min 8 chars, must include uppercase, lowercase, number): " DB_ADMIN_PASSWORD
    echo ""
    
    # Validate password
    if [ ${#DB_ADMIN_PASSWORD} -lt 8 ]; then
        print_error "Password must be at least 8 characters"
        exit 1
    fi
else
    DB_ADMIN_PASSWORD="$AZURE_DB_PASSWORD"
fi

# Prompt for Stripe keys
read -p "Enter Stripe Secret Key (or press Enter to set later): " STRIPE_SECRET_KEY
STRIPE_SECRET_KEY="${STRIPE_SECRET_KEY:-sk_test_CHANGEME}"

read -p "Enter Stripe Webhook Secret (or press Enter to set later): " STRIPE_WEBHOOK_SECRET
STRIPE_WEBHOOK_SECRET="${STRIPE_WEBHOOK_SECRET:-whsec_CHANGEME}"

# Generate API secret
API_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))" 2>/dev/null || openssl rand -base64 32)

echo ""
print_warning "Review configuration above. Continue? (y/n)"
read -p "> " CONFIRM
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    print_info "Deployment cancelled"
    exit 0
fi

# Step 1: Create Resource Group
print_info "Step 1/8: Creating resource group..."
if az group show --name $RESOURCE_GROUP &> /dev/null; then
    print_warning "Resource group already exists, skipping"
else
    az group create \
      --name $RESOURCE_GROUP \
      --location $LOCATION \
      --output none
    print_info "✓ Resource group created"
fi

# Step 2: Create PostgreSQL Server
print_info "Step 2/8: Creating PostgreSQL server (this may take 5-10 minutes)..."
if az postgres flexible-server show --resource-group $RESOURCE_GROUP --name $DB_SERVER_NAME &> /dev/null; then
    print_warning "PostgreSQL server already exists, skipping"
else
    az postgres flexible-server create \
      --resource-group $RESOURCE_GROUP \
      --name $DB_SERVER_NAME \
      --location $LOCATION \
      --admin-user $DB_ADMIN_USER \
      --admin-password "$DB_ADMIN_PASSWORD" \
      --sku-name Standard_B1ms \
      --tier Burstable \
      --version 14 \
      --storage-size 32 \
      --public-access 0.0.0.0 \
      --yes \
      --output none
    print_info "✓ PostgreSQL server created"
fi

# Step 3: Create Database
print_info "Step 3/8: Creating database..."
if az postgres flexible-server db show --resource-group $RESOURCE_GROUP --server-name $DB_SERVER_NAME --database-name $DB_NAME &> /dev/null; then
    print_warning "Database already exists, skipping"
else
    az postgres flexible-server db create \
      --resource-group $RESOURCE_GROUP \
      --server-name $DB_SERVER_NAME \
      --database-name $DB_NAME \
      --output none
    print_info "✓ Database created"
fi

# Step 4: Configure Firewall
print_info "Step 4/8: Configuring database firewall..."
az postgres flexible-server firewall-rule create \
  --resource-group $RESOURCE_GROUP \
  --name $DB_SERVER_NAME \
  --rule-name AllowAzureServices \
  --start-ip-address 0.0.0.0 \
  --end-ip-address 0.0.0.0 \
  --output none 2>/dev/null || print_warning "Firewall rule may already exist"
print_info "✓ Firewall configured"

# Build connection string
DB_CONNECTION_STRING="postgresql://$DB_ADMIN_USER:$DB_ADMIN_PASSWORD@$DB_SERVER_NAME.postgres.database.azure.com:5432/$DB_NAME?sslmode=require"

# Step 5: Create App Service Plan
print_info "Step 5/8: Creating App Service plan..."
if az appservice plan show --resource-group $RESOURCE_GROUP --name $PLAN_NAME &> /dev/null; then
    print_warning "App Service plan already exists, skipping"
else
    az appservice plan create \
      --resource-group $RESOURCE_GROUP \
      --name $PLAN_NAME \
      --location $LOCATION \
      --sku $SKU \
      --is-linux \
      --output none
    print_info "✓ App Service plan created"
fi

# Step 6: Create Web App
print_info "Step 6/8: Creating web app..."
if az webapp show --resource-group $RESOURCE_GROUP --name $APP_NAME &> /dev/null; then
    print_warning "Web app already exists, skipping"
else
    az webapp create \
      --resource-group $RESOURCE_GROUP \
      --plan $PLAN_NAME \
      --name $APP_NAME \
      --runtime "PYTHON:3.11" \
      --output none
    print_info "✓ Web app created"
fi

# Step 7: Configure App Settings
print_info "Step 7/8: Configuring application settings..."
az webapp config appsettings set \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --settings \
    DATABASE_URL="$DB_CONNECTION_STRING" \
    STRIPE_SECRET_KEY="$STRIPE_SECRET_KEY" \
    STRIPE_WEBHOOK_SECRET="$STRIPE_WEBHOOK_SECRET" \
    API_SECRET_KEY="$API_SECRET" \
    SCM_DO_BUILD_DURING_DEPLOYMENT=true \
    WEBSITES_PORT=8000 \
  --output none
print_info "✓ Application settings configured"

# Step 8: Configure Startup Command
print_info "Step 8/8: Configuring startup command..."
az webapp config set \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --startup-file "gunicorn main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000" \
  --output none
print_info "✓ Startup command configured"

# Enable HTTPS only
az webapp update \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --https-only true \
  --output none

# Get app URL
APP_URL=$(az webapp show \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --query defaultHostName \
  --output tsv)

echo ""
print_info "=========================================="
print_info "Deployment Complete! 🎉"
print_info "=========================================="
echo ""
echo "Your API will be available at: https://$APP_URL"
echo ""
print_info "Next steps:"
echo "  1. Deploy your code:"
echo "     cd cloud-api"
echo "     az webapp up --resource-group $RESOURCE_GROUP --name $APP_NAME --runtime PYTHON:3.11"
echo ""
echo "  2. Or configure GitHub deployment:"
echo "     az webapp deployment source config --resource-group $RESOURCE_GROUP --name $APP_NAME \\"
echo "       --repo-url https://github.com/Quotraders/simple-bot --branch main --manual-integration"
echo ""
echo "  3. Configure your bot to use this Azure deployment:"
echo "     export QUOTRADING_API_URL=\"https://$APP_URL\""
echo "     # Or add to .env file: QUOTRADING_API_URL=https://$APP_URL"
echo ""
echo "  4. Test your API:"
echo "     curl https://$APP_URL/"
echo ""
echo "  5. View logs:"
echo "     az webapp log tail --resource-group $RESOURCE_GROUP --name $APP_NAME"
echo ""
print_warning "Important: Update Stripe webhook URL to: https://$APP_URL/api/v1/webhooks/stripe"
echo ""
print_info "Configuration saved to: deployment-info.txt"

# Save deployment info
cat > deployment-info.txt <<EOF
QuoTrading Azure Deployment Information
Generated: $(date)

Resource Group: $RESOURCE_GROUP
Location: $LOCATION
App Name: $APP_NAME
App URL: https://$APP_URL
Database Server: $DB_SERVER_NAME.postgres.database.azure.com
Database Name: $DB_NAME
Database User: $DB_ADMIN_USER
App Service Plan: $PLAN_NAME

Stripe Webhook URL: https://$APP_URL/api/v1/webhooks/stripe

Environment Variables Set:
- DATABASE_URL
- STRIPE_SECRET_KEY
- STRIPE_WEBHOOK_SECRET
- API_SECRET_KEY
- SCM_DO_BUILD_DURING_DEPLOYMENT
- WEBSITES_PORT

To use this Azure deployment, set environment variable:
export QUOTRADING_API_URL="https://$APP_URL"

Or add to your .env file:
QUOTRADING_API_URL=https://$APP_URL
EOF

print_info "Deployment information saved!"
echo ""
