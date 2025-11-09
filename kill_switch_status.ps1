# QuoTrading Kill Switch - CHECK STATUS
# View current kill switch state

Write-Host "Checking kill switch status..." -ForegroundColor Cyan

try {
    $response = Invoke-WebRequest -Uri "https://quotrading-signals.icymeadow-86b2969e.eastus.azurecontainerapps.io/api/kill_switch/status" -Method GET
    $status = $response.Content | ConvertFrom-Json
    
    Write-Host "`n========================================" -ForegroundColor Cyan
    if ($status.kill_switch_active) {
        Write-Host "🔴 KILL SWITCH: ACTIVE" -ForegroundColor Red
        Write-Host "Reason: $($status.reason)" -ForegroundColor Yellow
        Write-Host "Activated: $($status.activated_at)" -ForegroundColor Yellow
        Write-Host "`nAll customer bots are DISCONNECTED" -ForegroundColor Red
    } else {
        Write-Host "🟢 KILL SWITCH: INACTIVE" -ForegroundColor Green
        Write-Host "`nAll customer bots are RUNNING normally" -ForegroundColor Green
    }
    Write-Host "========================================`n" -ForegroundColor Cyan
} catch {
    Write-Host "❌ Error checking status: $_" -ForegroundColor Red
}

Write-Host "Press any key to close..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
