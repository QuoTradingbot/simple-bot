# QuoTrading Cloud API

Subscription management and license validation API for QuoTrading bot.

## 🎯 What This Does

- **User Registration**: Create accounts and generate API keys
- **License Validation**: Bot validates subscription on startup
- **Stripe Integration**: Handle payments and subscriptions
- **Usage Tracking**: Monitor user logins and activity
- **Admin Access**: Master key for testing and support

## 🏗️ Architecture

```
Bot (Customer) → QuoTrading API (Render) → PostgreSQL (Render)
                        ↓
                 Stripe Webhooks
```

## 📁 Files

- `main.py` - FastAPI application (core API)
- `requirements.txt` - Python dependencies
- `DEPLOYMENT.md` - Complete deployment guide
- `admin_tool.py` - CLI tool for testing and user management
- `.env.example` - Environment variables template

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
cd cloud-api
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your values

# Run locally
python main.py
# API runs at http://localhost:8000
```

### Deploy to Cloud

**Option 1: Deploy to Render**

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete deployment instructions.

**TL;DR:**
1. Create PostgreSQL database on Render
2. Create Web Service pointing to this folder
3. Set environment variables
4. Deploy!

**Option 2: Deploy to Azure**

See [AZURE_DEPLOYMENT.md](AZURE_DEPLOYMENT.md) for complete Azure CLI deployment instructions.

**TL;DR:**
```bash
cd cloud-api
chmod +x deploy-azure.sh
./deploy-azure.sh
```

Or deploy manually with Azure CLI commands in the guide.

## 🔑 API Endpoints

### Health Check
```http
GET /
```

### Register User
```http
POST /api/v1/users/register
{
  "email": "user@example.com"
}
```

### Validate License (Bot uses this)
```http
POST /api/v1/license/validate
{
  "email": "user@example.com",
  "api_key": "quo_xxxxxxxxxxxxxxxx"
}
```

### Create Subscription
```http
POST /api/v1/subscriptions/create
{
  "email": "user@example.com",
  "tier": "pro",
  "payment_method_id": "pm_xxxxx"
}
```

### Get User Info
```http
GET /api/v1/users/{email}
```

### Stripe Webhook
```http
POST /api/v1/webhooks/stripe
```

## 💰 Subscription Tiers

| Tier | Price | Max Contracts | Max Accounts |
|------|-------|---------------|--------------|
| Basic | $99/mo | 3 | 1 |
| Pro | $199/mo | 10 | 3 |
| Enterprise | $499/mo | 25 | 10 |

## 🔐 Admin Access

**Master Key:** `QUOTRADING_ADMIN_MASTER_2025`

- Bypasses all license checks
- No subscription required
- Use for testing and support
- Gives enterprise-level limits

## 🧪 Testing

```bash
# Run admin tool
python admin_tool.py

# Quick health check
python admin_tool.py health

# Test admin key
python admin_tool.py admin

# Full test suite
python admin_tool.py test
```

## 🔧 Environment Variables

```env
DATABASE_URL=postgresql://user:pass@host:5432/db
STRIPE_SECRET_KEY=sk_test_xxxxx
STRIPE_WEBHOOK_SECRET=whsec_xxxxx
API_SECRET_KEY=random-secret
```

## 📊 Database Schema

### users table
- `id` - Primary key
- `email` - Unique email address
- `api_key` - Generated API key (quo_xxxxx)
- `subscription_status` - active/inactive/past_due/canceled
- `subscription_tier` - basic/pro/enterprise
- `subscription_start` - Start date
- `subscription_end` - Expiration date
- `stripe_customer_id` - Stripe customer ID
- `stripe_subscription_id` - Stripe subscription ID
- `max_contract_size` - Contract limit for tier
- `max_accounts` - Account limit for tier
- `last_login` - Last login timestamp
- `total_logins` - Login counter
- `is_active` - Account active flag

## 🔄 Subscription Flow

1. **User Signs Up**
   - POST /api/v1/users/register
   - Receives API key via email
   - Status: inactive

2. **User Subscribes**
   - POST /api/v1/subscriptions/create
   - Stripe processes payment
   - Status: active
   - Limits applied based on tier

3. **Bot Validates**
   - POST /api/v1/license/validate
   - Returns subscription details
   - Bot enforces contract limits

4. **Payment Events**
   - Stripe webhooks update status
   - Past due → Bot shows warning
   - Canceled → Bot blocks access

## 🛠️ Bot Integration

The launcher automatically validates on startup:

```python
# In QuoTrading_Launcher.py
response = requests.post(
    "https://quotrading-api.onrender.com/api/v1/license/validate",
    json={"email": email, "api_key": api_key}
)

if response.status_code == 200:
    data = response.json()
    max_contracts = data["max_contract_size"]
    max_accounts = data["max_accounts"]
    # Apply limits to bot
```

## 📈 Monitoring

**Render Dashboard:**
- View logs in real-time
- Monitor API requests
- Check error rates
- Database connections

**Stripe Dashboard:**
- Track subscriptions
- Monitor revenue
- Handle failed payments
- View customer details

## 🐛 Troubleshooting

### API won't start
- Check logs in Render dashboard
- Verify DATABASE_URL is set
- Ensure all environment variables present

### License validation fails
- Check user exists in database
- Verify API key matches exactly
- Ensure subscription is active
- Check subscription_end date

### Stripe webhooks not working
- Verify webhook secret is correct
- Check webhook URL matches Render URL
- Ensure events are configured

## 📝 Next Steps

1. **Deploy to Render** (see DEPLOYMENT.md)
2. **Set up Stripe account** (get API keys)
3. **Test with admin key** (python admin_tool.py)
4. **Create test subscription** (use Stripe test mode)
5. **Update bot** (set QUOTRADING_API_URL env var)
6. **Go live!** (switch to Stripe live mode)

## 💡 Tips

- Start with Stripe test mode (keys start with `sk_test_`)
- Use admin key for testing without subscriptions
- Monitor Render logs for errors
- Test webhooks with Stripe CLI
- Keep DATABASE_URL secret!

## 📞 Support

Questions? Contact: support@quotrading.com
