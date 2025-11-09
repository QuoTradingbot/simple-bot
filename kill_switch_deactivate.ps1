# QuoTrading Kill Switch - DEACTIVATE
# This allows ALL customer bots to reconnect and resume trading

$body = @{
    admin_key = "QUOTRADING_ADMIN_2025"
    active = $false
} | ConvertTo-Json

Write-Host "🟢 DEACTIVATING KILL SWITCH..." -ForegroundColor Green
Write-Host "All customer bots will reconnect and resume trading" -ForegroundColor Yellow

try {
    $response = Invoke-WebRequest -Uri "https://quotrading-signals.icymeadow-86b2969e.eastus.azurecontainerapps.io/api/admin/kill_switch" -Method POST -Body $body -ContentType "application/json"
    Write-Host "✅ Kill switch DEACTIVATED successfully" -ForegroundColor Green
    Write-Host "Status: Bots are reconnecting to brokers" -ForegroundColor Yellow
} catch {
    Write-Host "❌ Error deactivating kill switch: $_" -ForegroundColor Red
}

Write-Host "`nPress any key to close..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
