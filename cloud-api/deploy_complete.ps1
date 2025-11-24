# QuoTrading - Complete Azure Deployment Script
# This deploys Signal Engine with WebSocket + Auto-Scaling + Redis

Write-Host "🚀 QuoTrading Azure Deployment - WebSocket + Auto-Scaling" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Configuration
$RESOURCE_GROUP = "quotrading-rg"
$LOCATION = "eastus"
$SIGNAL_APP_NAME = "quotrading-signal-engine"
$REDIS_NAME = "quotrading-cache"
$APP_SERVICE_PLAN = "quotrading-signal-plan"

# Step 1: Check if logged in
Write-Host "🔐 Step 1: Checking Azure login..." -ForegroundColor Yellow
try {
    $account = az account show 2>$null | ConvertFrom-Json
    Write-Host "✅ Logged in as: $($account.user.name)" -ForegroundColor Green
} catch {
    Write-Host "❌ Not logged in to Azure" -ForegroundColor Red
    Write-Host "Run: az login" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# Step 2: Check/Create Resource Group
Write-Host "📦 Step 2: Checking resource group..." -ForegroundColor Yellow
$rgExists = az group exists --name $RESOURCE_GROUP
if ($rgExists -eq "false") {
    Write-Host "Creating resource group: $RESOURCE_GROUP" -ForegroundColor Cyan
    az group create --name $RESOURCE_GROUP --location $LOCATION
} else {
    Write-Host "✅ Resource group exists: $RESOURCE_GROUP" -ForegroundColor Green
}
Write-Host ""

# Step 3: Check/Create App Service Plan
Write-Host "🖥️  Step 3: Checking App Service Plan..." -ForegroundColor Yellow
$planExists = az appservice plan show --name $APP_SERVICE_PLAN --resource-group $RESOURCE_GROUP 2>$null
if (-not $planExists) {
    Write-Host "Creating App Service Plan (S1 Standard)..." -ForegroundColor Cyan
    az appservice plan create `
        --name $APP_SERVICE_PLAN `
        --resource-group $RESOURCE_GROUP `
        --location $LOCATION `
        --sku S1 `
        --is-linux
    Write-Host "✅ App Service Plan created" -ForegroundColor Green
} else {
    Write-Host "✅ App Service Plan exists" -ForegroundColor Green
}
Write-Host ""

# Step 4: Deploy Signal Engine
Write-Host "🚀 Step 4: Deploying Signal Engine with WebSocket..." -ForegroundColor Yellow
Write-Host "   This may take 3-5 minutes..." -ForegroundColor Gray

Push-Location cloud-api
try {
    az webapp up `
        --name $SIGNAL_APP_NAME `
        --resource-group $RESOURCE_GROUP `
        --plan $APP_SERVICE_PLAN `
        --runtime "PYTHON:3.11" `
        --sku S1
    
    Write-Host "✅ Signal Engine deployed!" -ForegroundColor Green
} catch {
    Write-Host "❌ Deployment failed: $_" -ForegroundColor Red
} finally {
    Pop-Location
}
Write-Host ""

# Step 5: Create Redis Cache (Optional - costs $16/month)
Write-Host "🔴 Step 5: Setting up Redis Cache..." -ForegroundColor Yellow
$createRedis = Read-Host "Create Azure Redis Cache? ($16/month for Basic tier) [y/N]"

if ($createRedis -eq 'y' -or $createRedis -eq 'Y') {
    Write-Host "Creating Redis Cache (this takes 10-15 minutes)..." -ForegroundColor Cyan
    
    az redis create `
        --resource-group $RESOURCE_GROUP `
        --name $REDIS_NAME `
        --location $LOCATION `
        --sku Basic `
        --vm-size c0 `
        --enable-non-ssl-port false
    
    Write-Host "✅ Redis Cache created!" -ForegroundColor Green
    
    # Get Redis key
    Write-Host "Getting Redis access key..." -ForegroundColor Cyan
    $redisKeys = az redis list-keys --resource-group $RESOURCE_GROUP --name $REDIS_NAME | ConvertFrom-Json
    $primaryKey = $redisKeys.primaryKey
    
    # Set Redis URL in App Service
    $redisUrl = "redis://$REDIS_NAME.redis.cache.windows.net:6380?ssl=True&password=$primaryKey"
    
    Write-Host "Setting REDIS_URL in App Service..." -ForegroundColor Cyan
    az webapp config appsettings set `
        --resource-group $RESOURCE_GROUP `
        --name $SIGNAL_APP_NAME `
        --settings REDIS_URL="$redisUrl"
    
    Write-Host "✅ Redis configured!" -ForegroundColor Green
} else {
    Write-Host "⏭️  Skipping Redis (will use in-memory fallback)" -ForegroundColor Yellow
}
Write-Host ""

# Step 6: Enable Auto-Scaling
Write-Host "📊 Step 6: Configuring Auto-Scaling (1-10 instances)..." -ForegroundColor Yellow

# Check if autoscale already exists
$autoscaleExists = az monitor autoscale show `
    --resource-group $RESOURCE_GROUP `
    --name "$APP_SERVICE_PLAN-autoscale" 2>$null

if (-not $autoscaleExists) {
    Write-Host "Creating autoscale rules..." -ForegroundColor Cyan
    
    # Create autoscale setting
    az monitor autoscale create `
        --resource-group $RESOURCE_GROUP `
        --resource "/subscriptions/$($account.id)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Web/serverfarms/$APP_SERVICE_PLAN" `
        --resource-type "Microsoft.Web/serverfarms" `
        --name "$APP_SERVICE_PLAN-autoscale" `
        --min-count 1 `
        --max-count 10 `
        --count 2
    
    # Scale out rule: CPU > 70%
    az monitor autoscale rule create `
        --resource-group $RESOURCE_GROUP `
        --autoscale-name "$APP_SERVICE_PLAN-autoscale" `
        --condition "Percentage CPU > 70 avg 5m" `
        --scale out 2 `
        --cooldown 5
    
    # Scale in rule: CPU < 30%
    az monitor autoscale rule create `
        --resource-group $RESOURCE_GROUP `
        --autoscale-name "$APP_SERVICE_PLAN-autoscale" `
        --condition "Percentage CPU < 30 avg 10m" `
        --scale in 1 `
        --cooldown 10
    
    Write-Host "✅ Auto-scaling enabled!" -ForegroundColor Green
} else {
    Write-Host "✅ Auto-scaling already configured" -ForegroundColor Green
}
Write-Host ""

# Step 7: Restart App Service
Write-Host "🔄 Step 7: Restarting App Service..." -ForegroundColor Yellow
az webapp restart --name $SIGNAL_APP_NAME --resource-group $RESOURCE_GROUP
Write-Host "✅ App Service restarted" -ForegroundColor Green
Write-Host ""

# Step 8: Display Summary
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "🎉 DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Deployment Summary:" -ForegroundColor Yellow
Write-Host "   Resource Group:    $RESOURCE_GROUP" -ForegroundColor White
Write-Host "   Signal Engine:     https://$SIGNAL_APP_NAME.azurewebsites.net" -ForegroundColor White
Write-Host "   WebSocket:         wss://$SIGNAL_APP_NAME.azurewebsites.net/ws/signals/{license_key}" -ForegroundColor White
Write-Host "   Admin Dashboard:   https://$SIGNAL_APP_NAME.azurewebsites.net/admin-dashboard/" -ForegroundColor White
Write-Host "   Auto-Scaling:      1-10 instances (CPU-based)" -ForegroundColor White
if ($createRedis -eq 'y' -or $createRedis -eq 'Y') {
    Write-Host "   Redis Cache:       $REDIS_NAME.redis.cache.windows.net" -ForegroundColor White
} else {
    Write-Host "   Redis Cache:       In-memory fallback (no external Redis)" -ForegroundColor White
}
Write-Host ""

# Step 9: Test Deployment
Write-Host "🧪 Step 9: Testing deployment..." -ForegroundColor Yellow
try {
    $healthUrl = "https://$SIGNAL_APP_NAME.azurewebsites.net/health"
    Write-Host "   Testing: $healthUrl" -ForegroundColor Gray
    
    Start-Sleep -Seconds 5  # Wait for service to be ready
    
    $response = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 10
    Write-Host "✅ Health check passed!" -ForegroundColor Green
    Write-Host "   Status: $($response.status)" -ForegroundColor White
} catch {
    Write-Host "⚠️  Health check failed (service may still be starting up)" -ForegroundColor Yellow
    Write-Host "   Try again in 1-2 minutes: $healthUrl" -ForegroundColor Gray
}
Write-Host ""

# Display Next Steps
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "📋 Next Steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Test WebSocket connection:" -ForegroundColor White
Write-Host "   Open browser console and run:" -ForegroundColor Gray
Write-Host "   const ws = new WebSocket('wss://$SIGNAL_APP_NAME.azurewebsites.net/ws/signals/YOUR_LICENSE_KEY');" -ForegroundColor Cyan
Write-Host "   ws.onmessage = (e) => console.log(JSON.parse(e.data));" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. Check deployment logs:" -ForegroundColor White
Write-Host "   az webapp log tail --name $SIGNAL_APP_NAME --resource-group $RESOURCE_GROUP" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Monitor auto-scaling:" -ForegroundColor White
Write-Host "   Azure Portal → $APP_SERVICE_PLAN → Metrics" -ForegroundColor Cyan
Write-Host ""
Write-Host "4. View WebSocket stats:" -ForegroundColor White
Write-Host "   https://$SIGNAL_APP_NAME.azurewebsites.net/api/ws/stats?license_key=ADMIN_KEY" -ForegroundColor Cyan
Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "🚀 Your Signal Engine is ready for 1,000+ users!" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Cyan
