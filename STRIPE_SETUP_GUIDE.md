# 🚀 Stripe Integration Setup Guide

## ✅ What's Been Done

1. **Added to cloud API (signal_engine_v2.py):**
   - License validation endpoints
   - Stripe webhook handler (auto-creates licenses)
   - Checkout session creation
   - In-memory license storage

2. **Created payment page (payment.html):**
   - Beautiful subscription page
   - Test mode ready (fake card: 4242 4242 4242 4242)
   - Stripe Checkout integration

3. **Added dependencies:**
   - stripe==7.4.0 in requirements-signal.txt

---

## 🔧 Next Steps (5 minutes)

### Step 1: Get Your Stripe Price ID

1. Go to: https://dashboard.stripe.com/test/products
2. Click on "QuoTrading Bot" product
3. Copy the **Price ID** (starts with `price_...`)

### Step 2: Update Cloud API

Open `cloud-api/signal_engine_v2.py` and find line ~843:

```python
PRICE_ID = "price_PLACEHOLDER"  # TODO: Update with actual price ID
```

Replace with your actual price ID:
```python
PRICE_ID = "price_1abc123..."  # Your price ID from Stripe
```

### Step 3: Set Up Stripe Webhook (Optional for beta)

For test mode, you can skip this - webhook will work without signature verification.

For production later:
1. Go to: https://dashboard.stripe.com/test/webhooks
2. Click "Add endpoint"
3. URL: `https://quotrading-signals.icymeadow-86b2969e.eastus.azurecontainerapps.io/api/stripe/webhook`
4. Events to listen: 
   - `checkout.session.completed`
   - `customer.subscription.deleted`
   - `invoice.payment_failed`
5. Copy the **Signing secret** (starts with `whsec_...`)
6. Update `signal_engine_v2.py` line ~31:
   ```python
   STRIPE_WEBHOOK_SECRET = "whsec_your_secret_here"
   ```

---

## 🚀 Deployment

### Option 1: Deploy Now (Test Price ID Needed)

```powershell
cd c:\Users\kevin\Downloads\simple-bot-1\cloud-api

# Build and deploy
az acr build --registry quotradingsignals --image quotrading-signals:v6-stripe --file Dockerfile .

# Update container app
az containerapp update --name quotrading-signals --resource-group quotrading-rg --image quotradingsignals.azurecr.io/quotrading-signals:v6-stripe
```

### Option 2: Test Locally First

```powershell
# Install dependencies
pip install stripe==7.4.0

# Run locally
cd cloud-api
python signal_engine_v2.py

# Test in browser: http://localhost:8000/api/ml/stats
```

---

## 🧪 Testing the Payment Flow

### Manual License Creation (For Beta Testing)

```powershell
# Create a license for a beta tester
Invoke-WebRequest -Uri "https://quotrading-signals.icymeadow-86b2969e.eastus.azurecontainerapps.io/api/license/activate" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"email": "beta@test.com", "days": 30}'

# Response will include license key:
# {"license_key": "ABC123XYZ...", "email": "beta@test.com", "expires_at": "2025-12-08T..."}
```

### Test Stripe Payment (After Price ID Updated)

1. Open `cloud-api/payment.html` in browser
2. Click "Subscribe Now"
3. Use test card: `4242 4242 4242 4242`
4. Any future expiry date, any CVC
5. "Payment" succeeds (no real money)
6. Check license was created:
   ```powershell
   Invoke-WebRequest -Uri "https://quotrading-signals.icymeadow-86b2969e.eastus.azurecontainerapps.io/api/license/list"
   ```

### Validate a License

```powershell
Invoke-WebRequest -Uri "https://quotrading-signals.icymeadow-86b2969e.eastus.azurecontainerapps.io/api/license/validate" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"license_key": "YOUR_LICENSE_KEY_HERE"}'
```

---

## 📋 API Endpoints

### License Management

- **POST** `/api/license/validate` - Check if license is valid
  - Body: `{"license_key": "ABC123..."}`
  - Response: `{"valid": true, "email": "...", "expires_at": "..."}`

- **POST** `/api/license/activate` - Manually create license (admin)
  - Body: `{"email": "user@example.com", "days": 30}`
  - Response: `{"license_key": "ABC123...", "email": "...", "expires_at": "..."}`

- **GET** `/api/license/list` - List all active licenses (admin)
  - Response: `{"total": 5, "licenses": [...]}`

### Stripe

- **POST** `/api/stripe/create-checkout` - Create Stripe Checkout session
  - Response: `{"session_id": "cs_test_..."}`

- **POST** `/api/stripe/webhook` - Stripe webhook handler (auto-called by Stripe)

---

## 🔐 Security Notes

### Test Mode (Current)
- Stripe keys are TEST keys (safe to commit)
- No real money involved
- Perfect for beta testing

### Production Mode (Later)
- Replace with LIVE keys (`sk_live_...`, `pk_live_...`)
- Store keys as Azure environment variables (not in code)
- Enable webhook signature verification
- Add rate limiting to prevent abuse

---

## 🎯 Beta Launch Flow

### For Beta Testers (Free):

1. You manually create licenses:
   ```powershell
   curl -X POST https://quotrading-signals.../api/license/activate \
     -H "Content-Type: application/json" \
     -d '{"email": "beta@test.com", "days": 30}'
   ```

2. Send license key via email/DM

3. They enter it in bot → start trading

### For Paid Users (After Beta):

1. They visit payment page
2. Pay with real card
3. Stripe webhook auto-creates license
4. Email sent with license key (we'll add this)
5. They enter key in bot → start trading

---

## ✅ What's Next

- [ ] Get Price ID from Stripe
- [ ] Update `PRICE_ID` in `signal_engine_v2.py`
- [ ] Deploy v6 to Azure
- [ ] Test manual license creation
- [ ] Test Stripe payment flow (fake card)
- [ ] Add bot license check (coming next)
- [ ] Add email automation (later)

---

## 💡 Pro Tips

**For Beta Launch Tomorrow:**
- Skip Stripe checkout for now
- Manually create 5-10 licenses for beta testers
- Focus on: Does the bot work? Is it profitable?

**After Beta Success:**
- Enable Stripe checkout
- Switch to live mode
- Start accepting real $200/month payments

**The beauty:** Everything's already built! Just flip the switch when ready. 🚀
