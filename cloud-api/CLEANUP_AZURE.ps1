# Azure Resource Cleanup Script
# This removes UNUSED resources and saves $140-280/month
# 
# ⚠️ WARNING: This will DELETE resources permanently!
# Review AZURE_VERIFICATION_REPORT.md before running

param(
    [switch]$DryRun,  # Show what would be deleted without deleting
    [switch]$Force    # Skip confirmation prompts
)

$RESOURCE_GROUP = "quotrading-rg"

Write-Host "🗑️  Azure Resource Cleanup Script" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

if ($DryRun) {
    Write-Host "🔍 DRY RUN MODE - No resources will be deleted" -ForegroundColor Yellow
    Write-Host ""
}

# Resources to DELETE (unused/duplicate)
$resourcesToDelete = @{
    "Web Apps (Duplicate)" = @(
        "quotrading-api",
        "quotrading-api-v2"
    )
    "Storage Accounts (Unused)" = @(
        "quotradingrlstorage",
        "quotradingstore9283",
        "quotradingfuncstore"
    )
    "Container Registry (Unused)" = @(
        "cada7631c732acr"
    )
    "Relay (Unused)" = @(
        "quotrading-relay"
    )
    "Managed Environment (Stopped)" = @(
        "quotrading-env"
    )
    "App Service Plans (Extra)" = @(
        "EastUSLinuxDynamicPlan"
        # Keep WestUS2LinuxDynamicPlan if used by quotrading-asp
    )
}

# Resources to KEEP (essential)
$resourcesToKeep = @(
    "quotrading-flask-api",
    "quotrading-db",
    "quotrading-asp",
    "quotrading-logs",
    "quotrading-api-v3",  # App Insights component
    "Application Insights Smart Detection",
    "WestUS2LinuxDynamicPlan"  # May be same as quotrading-asp
)

Write-Host "📊 Cleanup Plan:" -ForegroundColor Yellow
Write-Host ""

$totalCost = 0
foreach ($category in $resourcesToDelete.Keys) {
    Write-Host "  $category" -ForegroundColor Red
    foreach ($resource in $resourcesToDelete[$category]) {
        Write-Host "    ❌ $resource" -ForegroundColor DarkRed
        
        # Estimate cost
        if ($category -match "Web Apps") { $totalCost += 70 }
        elseif ($category -match "Storage") { $totalCost += 5 }
        elseif ($category -match "Registry") { $totalCost += 5 }
        elseif ($category -match "Relay") { $totalCost += 10 }
        elseif ($category -match "Environment") { $totalCost += 25 }
        elseif ($category -match "App Service") { $totalCost += 35 }
    }
    Write-Host ""
}

Write-Host "💰 Estimated Monthly Savings: ~`$$totalCost" -ForegroundColor Green
Write-Host ""

Write-Host "✅ Resources to KEEP:" -ForegroundColor Green
foreach ($resource in $resourcesToKeep) {
    Write-Host "    ✓ $resource" -ForegroundColor DarkGreen
}
Write-Host ""

# Confirmation
if (-not $Force -and -not $DryRun) {
    Write-Host "⚠️  WARNING: This will PERMANENTLY DELETE the resources listed above!" -ForegroundColor Yellow
    Write-Host ""
    $confirm = Read-Host "Type 'DELETE' to continue, or anything else to cancel"
    
    if ($confirm -ne "DELETE") {
        Write-Host "❌ Cleanup cancelled" -ForegroundColor Red
        exit 0
    }
    Write-Host ""
}

# Check Azure login
Write-Host "🔐 Checking Azure login..." -ForegroundColor Cyan
try {
    $account = az account show 2>$null | ConvertFrom-Json
    Write-Host "✅ Logged in as: $($account.user.name)" -ForegroundColor Green
} catch {
    Write-Host "❌ Not logged in to Azure" -ForegroundColor Red
    Write-Host "Run: az login" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# Function to delete resource
function Remove-AzureResource {
    param(
        [string]$ResourceName,
        [string]$ResourceType,
        [bool]$IsDryRun
    )
    
    if ($IsDryRun) {
        Write-Host "  [DRY RUN] Would delete: $ResourceName ($ResourceType)" -ForegroundColor DarkYellow
        return
    }
    
    Write-Host "  Deleting: $ResourceName..." -ForegroundColor Yellow
    
    try {
        switch ($ResourceType) {
            "webapp" {
                az webapp delete --name $ResourceName --resource-group $RESOURCE_GROUP --yes 2>$null
            }
            "storage" {
                az storage account delete --name $ResourceName --resource-group $RESOURCE_GROUP --yes 2>$null
            }
            "acr" {
                az acr delete --name $ResourceName --resource-group $RESOURCE_GROUP --yes 2>$null
            }
            "relay" {
                az relay namespace delete --name $ResourceName --resource-group $RESOURCE_GROUP 2>$null
            }
            "containerapp-env" {
                az containerapp env delete --name $ResourceName --resource-group $RESOURCE_GROUP --yes 2>$null
            }
            "appserviceplan" {
                az appservice plan delete --name $ResourceName --resource-group $RESOURCE_GROUP --yes 2>$null
            }
        }
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "    ✅ Deleted successfully" -ForegroundColor Green
        } else {
            Write-Host "    ⚠️  May have failed or already deleted" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "    ❌ Error: $_" -ForegroundColor Red
    }
}

# Delete Web Apps
Write-Host "🌐 Deleting duplicate web apps..." -ForegroundColor Cyan
foreach ($webapp in $resourcesToDelete["Web Apps (Duplicate)"]) {
    Remove-AzureResource -ResourceName $webapp -ResourceType "webapp" -IsDryRun $DryRun
}
Write-Host ""

# Delete Storage Accounts
Write-Host "💾 Deleting unused storage accounts..." -ForegroundColor Cyan
foreach ($storage in $resourcesToDelete["Storage Accounts (Unused)"]) {
    Remove-AzureResource -ResourceName $storage -ResourceType "storage" -IsDryRun $DryRun
}
Write-Host ""

# Delete Container Registry
Write-Host "📦 Deleting container registry..." -ForegroundColor Cyan
foreach ($acr in $resourcesToDelete["Container Registry (Unused)"]) {
    Remove-AzureResource -ResourceName $acr -ResourceType "acr" -IsDryRun $DryRun
}
Write-Host ""

# Delete Relay
Write-Host "🔗 Deleting Azure Relay..." -ForegroundColor Cyan
foreach ($relay in $resourcesToDelete["Relay (Unused)"]) {
    Remove-AzureResource -ResourceName $relay -ResourceType "relay" -IsDryRun $DryRun
}
Write-Host ""

# Delete Managed Environment
Write-Host "🏗️  Deleting managed environment..." -ForegroundColor Cyan
foreach ($env in $resourcesToDelete["Managed Environment (Stopped)"]) {
    Remove-AzureResource -ResourceName $env -ResourceType "containerapp-env" -IsDryRun $DryRun
}
Write-Host ""

# Delete Extra App Service Plans (CAREFUL - check dependencies first)
Write-Host "📋 Checking App Service Plans..." -ForegroundColor Cyan
foreach ($plan in $resourcesToDelete["App Service Plans (Extra)"]) {
    # Check if plan has any apps
    $apps = az webapp list --query "[?appServicePlanId contains(@, '$plan')].name" -o tsv 2>$null
    
    if ($apps) {
        Write-Host "  ⚠️  Skipping $plan - has apps: $apps" -ForegroundColor Yellow
    } else {
        Remove-AzureResource -ResourceName $plan -ResourceType "appserviceplan" -IsDryRun $DryRun
    }
}
Write-Host ""

# Summary
Write-Host "=" * 70 -ForegroundColor Cyan
if ($DryRun) {
    Write-Host "✅ Dry run complete - no changes made" -ForegroundColor Green
    Write-Host ""
    Write-Host "To actually delete resources, run:" -ForegroundColor Yellow
    Write-Host "  .\CLEANUP_AZURE.ps1 -Force" -ForegroundColor Cyan
} else {
    Write-Host "✅ Cleanup complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "💰 Estimated monthly savings: ~`$$totalCost" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Verify remaining resources: az resource list --resource-group $RESOURCE_GROUP --output table" -ForegroundColor Cyan
    Write-Host "2. Update bot code to use quotrading-flask-api" -ForegroundColor Cyan
    Write-Host "3. Test bot connection to RL engine" -ForegroundColor Cyan
}
Write-Host ""
