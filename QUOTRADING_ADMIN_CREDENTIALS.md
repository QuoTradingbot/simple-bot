# QuoTrading Admin Credentials & API Access

## 🔐 Admin Access

### Admin API Key
```
ADMIN-DEV-KEY-2024
```

**Usage**: Include in requests as header:
```
X-Admin-Key: ADMIN-DEV-KEY-2024
```

---

## 🌐 Production API Server

### Base URL
```
https://quotrading-flask-api.azurewebsites.net
```

### Health Check
```bash
GET https://quotrading-flask-api.azurewebsites.net/api/hello
```

---

## 👤 Your License

### Account Details
- **Email**: kevin@quotrading.com
- **Account ID**: ACC-59B9D665C1F1541F
- **License Key**: `KTAO-XBBX-NWA1-L14N`
- **License Type**: Lifetime
- **Expires**: November 5, 2125 (100 years)

---

## 📡 Admin API Endpoints

### 1. List All Licenses
```bash
GET /api/admin/list-licenses
Header: X-Admin-Key: ADMIN-DEV-KEY-2024
```

**PowerShell Example**:
```powershell
$headers = @{'X-Admin-Key' = 'ADMIN-DEV-KEY-2024'}
Invoke-WebRequest -Uri "https://quotrading-flask-api.azurewebsites.net/api/admin/list-licenses" -Headers $headers
```

### 2. Create New License
```bash
POST /api/admin/create-license
Header: X-Admin-Key: ADMIN-DEV-KEY-2024
Body: {
  "email": "customer@example.com",
  "license_type": "Monthly",
  "duration_days": 30
}
```

**PowerShell Example**:
```powershell
$headers = @{'X-Admin-Key' = 'ADMIN-DEV-KEY-2024'; 'Content-Type' = 'application/json'}
$body = @{
    email = 'customer@example.com'
    license_type = 'Monthly'
    duration_days = 30
} | ConvertTo-Json

Invoke-WebRequest -Uri "https://quotrading-flask-api.azurewebsites.net/api/admin/create-license" -Headers $headers -Method POST -Body $body
```

### 3. Dashboard Stats
```bash
GET /api/admin/dashboard-stats
Header: X-Admin-Key: ADMIN-DEV-KEY-2024
```

### 4. Recent Activity
```bash
GET /api/admin/recent-activity
Header: X-Admin-Key: ADMIN-DEV-KEY-2024
```

### 5. Online Users
```bash
GET /api/admin/online-users
Header: X-Admin-Key: ADMIN-DEV-KEY-2024
```

### 6. Update License Status
```bash
POST /api/admin/update-license-status
Header: X-Admin-Key: ADMIN-DEV-KEY-2024
Body: {
  "license_key": "XXXX-XXXX-XXXX-XXXX",
  "new_status": "active"  // or "suspended", "expired"
}
```

---

## 🗄️ Database Access

### PostgreSQL Connection
- **Host**: `quotrading-db.postgres.database.azure.com`
- **Database**: `quotrading`
- **Username**: `quotradingadmin`
- **Password**: `QuoTrade2024!SecureDB`
- **Port**: 5432
- **SSL**: Required

**Connection String**:
```
postgresql://quotradingadmin:QuoTrade2024!SecureDB@quotrading-db.postgres.database.azure.com:5432/quotrading?sslmode=require
```

### Tables
- `users` - License management
- `rl_experiences` - Trading experiences
- `api_logs` - API request logs
- `heartbeats` - Bot heartbeats

---

## 🔄 Testing Bot License Validation

### Test License Validation
```powershell
$headers = @{'Content-Type' = 'application/json'}
$body = @{
    license_key = 'KTAO-XBBX-NWA1-L14N'
    signal_type = 'LONG'
} | ConvertTo-Json

Invoke-WebRequest -Uri "https://quotrading-flask-api.azurewebsites.net/api/main" -Headers $headers -Method POST -Body $body
```

### Test Admin Key as License
```powershell
$headers = @{'Content-Type' = 'application/json'}
$body = @{
    license_key = 'ADMIN-DEV-KEY-2024'
    signal_type = 'LONG'
} | ConvertTo-Json

Invoke-WebRequest -Uri "https://quotrading-flask-api.azurewebsites.net/api/main" -Headers $headers -Method POST -Body $body
```

---

## ☁️ Azure Resources

### Resource Group
```
quotrading-rg (East US)
```

### Resources
- **quotrading-flask-api**: Web App Service (S2 tier)
- **quotrading-db**: PostgreSQL Flexible Server
- **quotrading-asp**: App Service Plan (S2)
- **quotrading-logs**: Application Insights

### Subscription
```
Azure subscription 1
ID: c2a4b238-2c5e-46b5-9d4d-b0d8f9714bd2
Email: kevinsuero072897@gmail.com
```

---

## 🔧 Deployment

### Redeploy API
```powershell
cd C:\Users\kevin\Downloads\simple-bot\cloud-api\flask-api
Compress-Archive -Path app.py,requirements.txt -DestinationPath deploy.zip -Force
az webapp deployment source config-zip --name quotrading-flask-api --resource-group quotrading-rg --src deploy.zip
```

### Restart API
```bash
az webapp restart --name quotrading-flask-api --resource-group quotrading-rg
```

### View Logs
```bash
az webapp log tail --name quotrading-flask-api --resource-group quotrading-rg
```

---

## ✅ Verified Working

- ✅ Admin key authentication (`ADMIN-DEV-KEY-2024`)
- ✅ License validation endpoint (`/api/main`)
- ✅ License creation (`/api/admin/create-license`)
- ✅ License listing (`/api/admin/list-licenses`)
- ✅ Database tables created (users, rl_experiences, api_logs, heartbeats)
- ✅ Your lifetime license (`KTAO-XBBX-NWA1-L14N`)
- ✅ Admin key works as license key (no expiration)

---

## 📝 Notes

- Admin key has unlimited access and never expires
- Your personal license expires in 100 years (Nov 2125)
- API server is running on S2 tier (300-500ms response time)
- All endpoints require proper authentication
- Database is PostgreSQL Flexible Server with SSL required

---

**Last Updated**: November 29, 2025
**Status**: ✅ All systems operational
