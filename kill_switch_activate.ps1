# QuoTrading Kill Switch - ACTIVATE
# This stops ALL customer bots immediately

$body = @{
    admin_key = "QUOTRADING_ADMIN_2025"
    active = $true
    reason = "Emergency stop - manual activation"
} | ConvertTo-Json

Write-Host "🔴 ACTIVATING KILL SWITCH..." -ForegroundColor Red
Write-Host "All customer bots will disconnect within 30 seconds" -ForegroundColor Yellow

try {
    $response = Invoke-WebRequest -Uri "https://quotrading-signals.icymeadow-86b2969e.eastus.azurecontainerapps.io/api/admin/kill_switch" -Method POST -Body $body -ContentType "application/json"
    Write-Host "✅ Kill switch ACTIVATED successfully" -ForegroundColor Green
    Write-Host "Status: All bots are disconnecting from brokers" -ForegroundColor Yellow
} catch {
    Write-Host "❌ Error activating kill switch: $_" -ForegroundColor Red
}

Write-Host "`nPress any key to close..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
