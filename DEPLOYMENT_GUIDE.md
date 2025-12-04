# Azure Deployment Guide - Profile Endpoint Update

## Overview
This guide explains how to deploy the new `/api/profile` endpoint to Azure App Service.

The changes are **backend-only** (Python Flask API) - no changes to the launcher or bot are required.

---

## What Changed

### Files Modified
- `cloud-api/flask-api/app.py` - Added new `/api/profile` endpoint (~220 lines)

### Files Added (Documentation Only - Not Deployed)
- `USER_PROFILE_AUDIT.md` - Audit report
- `API_PROFILE_DOCUMENTATION.md` - API documentation
- `PROFILE_IMPLEMENTATION_SUMMARY.md` - Implementation summary
- `cloud-api/flask-api/test_profile_endpoint.py` - Test script

**Important:** Only `app.py` needs to be deployed to Azure. The documentation files stay in the repository.

---

## Deployment Methods

### Method 1: Azure CLI Deployment (Recommended)

**Prerequisites:**
- Azure CLI installed
- Logged in to Azure (`az login`)
- Access to quotrading-rg resource group

**Steps:**

```bash
# 1. Navigate to Flask API directory
cd cloud-api/flask-api

# 2. Create deployment ZIP (only necessary files)
zip deploy_profile_update.zip app.py requirements.txt

# 3. Deploy to Azure App Service
az webapp deployment source config-zip \
    --resource-group quotrading-rg \
    --name quotrading-flask-api \
    --src deploy_profile_update.zip

# 4. Restart the app to apply changes
az webapp restart \
    --resource-group quotrading-rg \
    --name quotrading-flask-api

# 5. Verify deployment
curl "https://quotrading-flask-api.azurewebsites.net/api/hello"
```

**Expected output:**
```json
{
  "status": "success",
  "message": "✅ QuoTrading Cloud API - Data Collection Only",
  "endpoints": [
    "POST /api/rl/submit-outcome - Submit trade outcome",
    "GET /api/profile - Get user profile and trading statistics",
    "GET /api/hello - Health check"
  ]
}
```

### Method 2: Local Deployment from Cloud Directory

If you prefer using the existing deployment script structure:

```powershell
# Navigate to Flask API directory
cd C:\path\to\simple-bot\cloud-api\flask-api

# Create deployment package
Compress-Archive -Path app.py,requirements.txt -DestinationPath deploy_profile_update.zip -Force

# Deploy using Azure CLI
az webapp deployment source config-zip `
    --resource-group quotrading-rg `
    --name quotrading-flask-api `
    --src deploy_profile_update.zip

# Restart
az webapp restart `
    --resource-group quotrading-rg `
    --name quotrading-flask-api
```

### Method 3: Azure Portal Deployment

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to **quotrading-flask-api** App Service
3. Go to **Deployment Center**
4. Choose **Local Git** or **ZIP Deploy**
5. Upload `app.py` (or the entire `cloud-api/flask-api` folder)
6. Restart the app

---

## Testing After Deployment

### 1. Test Health Endpoint
```bash
curl "https://quotrading-flask-api.azurewebsites.net/api/hello"
```

Should show the new profile endpoint in the list.

### 2. Test Profile Endpoint (with valid license key)
```bash
# Replace YOUR-LICENSE-KEY with a real key
curl "https://quotrading-flask-api.azurewebsites.net/api/profile?license_key=YOUR-LICENSE-KEY"
```

Expected response:
```json
{
  "status": "success",
  "profile": { ... },
  "trading_stats": { ... },
  "recent_activity": { ... }
}
```

### 3. Test Invalid License Key (Should Fail)
```bash
curl "https://quotrading-flask-api.azurewebsites.net/api/profile?license_key=INVALID"
```

Expected: `401 Unauthorized` with `{"error": "Invalid license key"}`

### 4. Run Automated Tests
```bash
# Set environment variable
export TEST_LICENSE_KEY="your-real-license-key"

# Run test script
python cloud-api/flask-api/test_profile_endpoint.py
```

---

## Deployment Checklist

- [ ] Back up current `app.py` (optional, but recommended)
- [ ] Verify Azure CLI is installed and authenticated
- [ ] Navigate to `cloud-api/flask-api` directory
- [ ] Create deployment ZIP with `app.py` and `requirements.txt`
- [ ] Deploy to Azure using `az webapp deployment source config-zip`
- [ ] Restart the Azure App Service
- [ ] Test `/api/hello` endpoint (verify profile endpoint listed)
- [ ] Test `/api/profile` with valid license key
- [ ] Test `/api/profile` with invalid license key (should return 401)
- [ ] Monitor Azure logs for any errors
- [ ] Update launcher/bot (optional - not required for this change)

---

## Rollback Procedure (If Needed)

If something goes wrong, you can rollback to the previous version:

```bash
# Get the previous deployment ID
az webapp deployment list \
    --resource-group quotrading-rg \
    --name quotrading-flask-api \
    --query "[0].id" -o tsv

# Rollback to previous deployment
az webapp deployment source revert \
    --resource-group quotrading-rg \
    --name quotrading-flask-api \
    --slot production

# Or restore from backup
# If you saved the old app.py, just re-deploy it
```

---

## Post-Deployment Monitoring

### Check Application Logs
```bash
# Stream logs
az webapp log tail \
    --resource-group quotrading-rg \
    --name quotrading-flask-api

# Download logs
az webapp log download \
    --resource-group quotrading-rg \
    --name quotrading-flask-api \
    --log-file azure-logs.zip
```

### Monitor for Errors
Look for these log messages after deployment:
- ✅ `✅ Profile accessed: us***@example.com, 150 trades, $5420.50 PnL` - Success
- ⚠️ `⚠️ Invalid license key in /api/profile:` - Expected for invalid keys
- ⚠️ `⚠️ Rate limit exceeded for /api/profile:` - Expected if rate limit hit
- ❌ `Profile query error:` - Database connection issue (investigate)

---

## Environment Variables (Already Configured)

The following environment variables are already set in Azure and don't need changes:

- `DB_HOST` - PostgreSQL host
- `DB_NAME` - Database name
- `DB_USER` - Database user
- `DB_PASSWORD` - Database password
- `DB_PORT` - Database port
- `ADMIN_API_KEY` - Admin API key
- `CORS_ORIGINS` - CORS allowed origins

**No new environment variables are needed for the profile endpoint.**

---

## Database Changes

**No database migrations required.** The profile endpoint uses existing tables:
- `users` - Already exists
- `rl_experiences` - Already exists
- `api_logs` - Already exists

---

## Performance Impact

**Expected Impact:**
- Additional 5 database queries per profile request
- Response time: ~200ms (well within acceptable range)
- No impact on existing endpoints
- Rate limit: 100 requests/minute per user (shared with other endpoints)

**Scaling:**
The current S2 tier can handle:
- ~1000 users checking their profile once per day
- ~100 users checking profile every hour
- Concurrent requests: Limited by existing rate limiting

---

## Troubleshooting

### Issue: 500 Internal Server Error
**Cause:** Database connection failure  
**Solution:** Check database connection string and credentials

### Issue: 401 Unauthorized
**Cause:** Invalid license key  
**Solution:** This is expected behavior - verify the license key is correct

### Issue: 403 Forbidden
**Cause:** Account is suspended  
**Solution:** Check user's license_status in database

### Issue: 429 Too Many Requests
**Cause:** Rate limit exceeded (100 req/min)  
**Solution:** This is expected behavior - user should wait

### Issue: Profile endpoint not showing in /api/hello
**Cause:** Deployment didn't complete or app didn't restart  
**Solution:** Restart the app service

---

## Quick Deployment Script

Save this as `deploy_profile_endpoint.sh`:

```bash
#!/bin/bash
set -e

echo "========================================="
echo "Deploying Profile Endpoint to Azure"
echo "========================================="

# Navigate to directory
cd "$(dirname "$0")"

# Create deployment package
echo "[1/4] Creating deployment package..."
zip -q deploy_profile_update.zip app.py requirements.txt
echo "✅ Package created"

# Deploy to Azure
echo "[2/4] Deploying to Azure..."
az webapp deployment source config-zip \
    --resource-group quotrading-rg \
    --name quotrading-flask-api \
    --src deploy_profile_update.zip
echo "✅ Deployed"

# Restart app
echo "[3/4] Restarting app..."
az webapp restart \
    --resource-group quotrading-rg \
    --name quotrading-flask-api
echo "✅ Restarted"

# Verify
echo "[4/4] Verifying deployment..."
sleep 5
curl -s "https://quotrading-flask-api.azurewebsites.net/api/hello" | grep -q "profile" && \
    echo "✅ Profile endpoint is live!" || \
    echo "❌ Profile endpoint not found - check logs"

echo "========================================="
echo "Deployment Complete!"
echo "========================================="
echo "Test the endpoint with:"
echo "curl 'https://quotrading-flask-api.azurewebsites.net/api/profile?license_key=YOUR-KEY'"
```

Make it executable: `chmod +x deploy_profile_endpoint.sh`

---

## Summary

**What to deploy:** Only `app.py` from `cloud-api/flask-api/`  
**How to deploy:** Use Azure CLI with ZIP deployment  
**Database changes:** None required  
**Environment variables:** None required  
**Downtime:** ~30 seconds during restart  
**Rollback:** Available via Azure deployment history  

**After deployment, the profile endpoint will be available at:**
```
https://quotrading-flask-api.azurewebsites.net/api/profile
```

---

*Deployment guide for profile endpoint update - December 4, 2025*
