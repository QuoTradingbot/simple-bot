# Fix Azure PostgreSQL Firewall - Allow App Service to Connect
# Run this to enable database access from your Flask API

# 1. Get your App Service outbound IPs
Write-Host "Getting App Service outbound IP addresses..." -ForegroundColor Yellow
az webapp show --resource-group quotrading-rg --name quotrading-flask-api --query outboundIpAddresses --output tsv

# 2. Add firewall rule to allow App Service IPs
Write-Host "`nAdding firewall rules to PostgreSQL..." -ForegroundColor Yellow

# Get the IPs as an array
$outboundIPs = (az webapp show --resource-group quotrading-rg --name quotrading-flask-api --query outboundIpAddresses --output tsv) -split ','

$ruleNum = 1
foreach ($ip in $outboundIPs) {
    $ip = $ip.Trim()
    Write-Host "Adding rule for IP: $ip" -ForegroundColor Cyan
    
    az postgres flexible-server firewall-rule create `
        --resource-group quotrading-rg `
        --name quotrading-db `
        --rule-name "AppService-IP-$ruleNum" `
        --start-ip-address $ip `
        --end-ip-address $ip
    
    $ruleNum++
}

# 3. Also allow all Azure services (recommended for App Service)
Write-Host "`nAllowing Azure services..." -ForegroundColor Yellow
az postgres flexible-server firewall-rule create `
    --resource-group quotrading-rg `
    --name quotrading-db `
    --rule-name "AllowAzureServices" `
    --start-ip-address 0.0.0.0 `
    --end-ip-address 0.0.0.0

Write-Host "`n✅ Firewall rules configured!" -ForegroundColor Green
Write-Host "Your Flask API can now connect to PostgreSQL" -ForegroundColor Green
Write-Host "`nTest the cloud RL again in 30 seconds..." -ForegroundColor Cyan
