# QuoTrading Azure Deployment Script
# This script deploys the Flask API and runs database migrations

param(
    [switch]$SkipMigration,
    [switch]$SkipDeploy
)

Write-Host "🚀 QuoTrading Azure Deployment" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$DB_HOST = "quotrading-db.postgres.database.azure.com"
$DB_NAME = "quotrading"
$DB_USER = "quotradingadmin"
$WEBAPP_NAME = "quotrading-flask-api"
$RESOURCE_GROUP = "quotrading-rg"  # UPDATE THIS if different

# Step 1: Database Migration
if (-not $SkipMigration) {
    Write-Host "📊 Step 1: Running Database Migration (Stripe -> Whop)..." -ForegroundColor Yellow
    Write-Host ""
    
    $migrationScript = "flask-api/migrate_stripe_to_whop.py"
    
    if (Test-Path $migrationScript) {
        Write-Host "Migration script found: $migrationScript" -ForegroundColor Green
        
        # Check if python is available
        $pythonExists = Get-Command python -ErrorAction SilentlyContinue
        
        if ($pythonExists) {
            Write-Host "Running migration script..." -ForegroundColor Cyan
            # Set DB password for migration (ensure this matches your actual DB password)
            $env:DB_PASSWORD = "QuoTrade2024!SecureDB" 
            
            python $migrationScript
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ Database migration completed successfully!" -ForegroundColor Green
            } else {
                Write-Host "❌ Database migration failed!" -ForegroundColor Red
                exit 1
            }
        } else {
            Write-Host "⚠️  Python not found! Skipping migration." -ForegroundColor Yellow
        }
    } else {
        Write-Host "⚠️  Migration script not found: $migrationScript" -ForegroundColor Yellow
    }
    
    Write-Host ""
} else {
    Write-Host "⏭️  Skipping database migration" -ForegroundColor Yellow
    Write-Host ""
}

# Step 2: Deploy Flask API
if (-not $SkipDeploy) {
    Write-Host "🌐 Step 2: Deploying Flask API to Azure..." -ForegroundColor Yellow
    Write-Host ""
    
    # Check if Azure CLI is installed
    $azExists = Get-Command az -ErrorAction SilentlyContinue
    
    if (-not $azExists) {
        Write-Host "❌ Azure CLI not found!" -ForegroundColor Red
        Write-Host "Install from: https://aka.ms/installazurecliwindows" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Or use VS Code Azure extension:" -ForegroundColor Yellow
        Write-Host "1. Install 'Azure App Service' extension" -ForegroundColor Cyan
        Write-Host "2. Right-click 'cloud-api/flask-api' folder" -ForegroundColor Cyan
        Write-Host "3. Select 'Deploy to Web App'" -ForegroundColor Cyan
        Write-Host "4. Choose '$WEBAPP_NAME'" -ForegroundColor Cyan
        exit 1
    }
    
    # Navigate to Flask API directory
    Push-Location flask-api
    
    try {
        Write-Host "Logging into Azure..." -ForegroundColor Cyan
        az login --only-show-errors
        
        Write-Host "Deploying to $WEBAPP_NAME..." -ForegroundColor Cyan
        az webapp up --name $WEBAPP_NAME --resource-group $RESOURCE_GROUP --runtime "PYTHON:3.11" --sku B1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "✅ Deployment completed successfully!" -ForegroundColor Green
            Write-Host ""
            Write-Host "🌐 Your API is live at:" -ForegroundColor Green
            Write-Host "   https://$WEBAPP_NAME.azurewebsites.net" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "📊 Admin Dashboard:" -ForegroundColor Green
            Write-Host "   https://$WEBAPP_NAME.azurewebsites.net/admin-dashboard/" -ForegroundColor Cyan
            Write-Host ""
        } else {
            Write-Host "❌ Deployment failed!" -ForegroundColor Red
            exit 1
        }
    } finally {
        Pop-Location
    }
} else {
    Write-Host "⏭️  Skipping deployment" -ForegroundColor Yellow
    Write-Host ""
}

# Step 3: Verify Deployment
Write-Host "🔍 Step 3: Verifying Deployment..." -ForegroundColor Yellow
Write-Host ""

$healthUrl = "https://$WEBAPP_NAME.azurewebsites.net/health"
Write-Host "Testing health endpoint: $healthUrl" -ForegroundColor Cyan

try {
    $response = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 10
    Write-Host "✅ Health check passed!" -ForegroundColor Green
    Write-Host "Response: $($response | ConvertTo-Json -Compress)" -ForegroundColor Gray
} catch {
    Write-Host "⚠️  Health check failed (may take a few minutes for deployment to complete)" -ForegroundColor Yellow
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "🎉 Deployment Complete!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Test license validation: https://$WEBAPP_NAME.azurewebsites.net/api/validate-license?license_key=YOUR_KEY" -ForegroundColor Cyan
Write-Host "2. Check admin dashboard: https://$WEBAPP_NAME.azurewebsites.net/admin-dashboard/" -ForegroundColor Cyan
Write-Host "3. Monitor logs: az webapp log tail --name $WEBAPP_NAME --resource-group $RESOURCE_GROUP" -ForegroundColor Cyan
Write-Host ""
