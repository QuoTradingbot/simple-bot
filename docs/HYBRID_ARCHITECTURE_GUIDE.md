# QuoTrading Hybrid Architecture - Testing Guide

## Overview

Your bot now supports **TWO MODES**:

1. **LOCAL MODE** (Testing) - Bot runs VWAP logic locally
2. **CLOUD MODE** (Production) - Bot polls Azure for ML/RL signals

This lets you test everything works BEFORE deploying to cloud and compiling to EXE.

---

## How It Works

### Configuration Flag

Located in `customer/QuoTrading_Launcher.py` (line ~37):

```python
USE_CLOUD_SIGNALS = False  # Local mode
# USE_CLOUD_SIGNALS = True  # Cloud mode (production)
```

### Local Mode (Testing)
- **What**: Bot uses local VWAP logic from `src/vwap_bounce_bot.py`
- **When**: Development, testing, debugging
- **Pros**: Fast, no API calls, works offline
- **Cons**: Your ML/RL IP is in the EXE (can be decompiled)

### Cloud Mode (Production)
- **What**: Bot sends market data to Azure, gets signals back
- **When**: Production deployment, customer distribution
- **Pros**: Protects your ML/RL IP, instant updates, per-user signals
- **Cons**: Requires internet, 50-300ms latency (acceptable for VWAP)

---

## Testing Workflow

### Step 1: Test Local Mode
```bash
# customer/QuoTrading_Launcher.py
USE_CLOUD_SIGNALS = False

# Run bot
python customer/QuoTrading_Launcher.py
```

**Verify:**
- ✅ Broker login works
- ✅ Account selection works
- ✅ Signals generate (watch console logs)
- ✅ Orders execute via TopStep
- ✅ Position management works
- ✅ License validation works

### Step 2: Deploy Signal API to Azure

```bash
# Deploy cloud-api to Azure
# (Instructions in separate Azure deployment guide)
```

### Step 3: Test Cloud Mode

```python
# customer/QuoTrading_Launcher.py
USE_CLOUD_SIGNALS = True
CLOUD_API_BASE_URL = "https://your-azure-app.azurewebsites.net"
```

**Verify:**
- ✅ Bot polls Azure every 5 seconds
- ✅ Signals generate from cloud
- ✅ Latency < 500ms
- ✅ Everything else still works

### Step 4: Compile to EXE

```bash
# Only after BOTH modes tested
pip install pyinstaller
pyinstaller --onefile --windowed customer/QuoTrading_Launcher.py
```

---

## Cloud Signal Flow

```
Customer PC (Every 5 seconds):
  ↓
  1. Collect market data (bars, position, settings)
  ↓
  2. HTTP POST to Azure: /api/v1/signals/generate
     Headers: X-API-Key: user_api_key_here
     Body: {
       user_id: "user@email.com",
       symbol: "ES",
       bars: [...],  # Last 30 1-min bars
       current_position: {...},
       settings: {...}  # Account size, risk, filters
     }
  ↓
  3. Azure runs VWAP/ML/RL
  ↓
  4. Returns signal: {
       action: "LONG",
       contracts: 5,
       entry: 5800.25,
       stop: 5798.00,
       target: 5804.50,
       confidence: 0.85,
       reason: "VWAP bounce + RSI oversold"
     }
  ↓
  5. Client executes trade via TopStep (locally)
```

---

## What Changes Between Modes?

### LOCAL MODE
- Bot calculates signals internally
- Uses `src/vwap_bounce_bot.py` logic
- No internet required
- Faster (no API calls)

### CLOUD MODE  
- Bot polls `CLOUD_SIGNAL_ENDPOINT`
- Sends market data to Azure
- Receives pre-calculated signal
- Executes signal locally via TopStep
- Still connects to TopStep directly (credentials never sent to cloud)

---

## Code Changes Made

### New Files
- `cloud-api/signal_engine.py` - Core VWAP/ML/RL logic
- `cloud-api/main.py` - Updated with `/api/v1/signals/generate` endpoint
- `test_signal_engine.py` - Local testing script

### Modified Files
- `customer/QuoTrading_Launcher.py`:
  - Added `USE_CLOUD_SIGNALS` flag
  - Added `CLOUD_API_BASE_URL` config
  - Added `CLOUD_SIGNAL_ENDPOINT` config
  - TODO: Add cloud polling function

---

## Next Steps

1. **Test local mode thoroughly** ✅
   - Verify all functions work
   - Document any bugs

2. **Deploy to Azure** ⏳
   - Create Azure App Service
   - Deploy `cloud-api/`
   - Test `/api/v1/signals/generate` endpoint

3. **Test cloud mode** ⏳
   - Switch `USE_CLOUD_SIGNALS = True`
   - Verify signals match local mode
   - Measure latency

4. **Compile to EXE** ⏳
   - PyInstaller build
   - Test on clean Windows machine
   - Distribute to customers

---

## Important Notes

### Latency
- Local mode: 0ms (instant)
- Cloud mode: 50-300ms (acceptable for VWAP strategy)
- NOT for HFT/scalping (use local mode for that)

### IP Protection
- **Local mode**: VWAP logic visible in EXE (can be decompiled)
- **Cloud mode**: VWAP/ML/RL stays in Azure (protected)

### Costs
- **Local mode**: FREE (no cloud costs)
- **Cloud mode**: 
  - Render (license API): $7/mo
  - Azure (signal API): FREE for 6 months (credits), then ~$30/mo
  - Total: $7/mo now, $37/mo after credits

### Updates
- **Local mode**: Customers must download new EXE for updates
- **Cloud mode**: Push to GitHub → auto-deploy → instant updates

---

## Troubleshooting

### Signals not generating in cloud mode?
- Check `CLOUD_API_BASE_URL` is correct
- Verify API key is valid (`X-API-Key` header)
- Check internet connection
- Look at console logs for errors

### High latency (>500ms)?
- Azure region might be far from customer
- Use Azure region closest to customers
- Check Azure pricing tier (higher = faster)

### Signals different between local/cloud?
- Cloud has bugs (fix in `cloud-api/signal_engine.py`)
- Settings mismatch (verify user settings sent correctly)
- Bar data mismatch (verify bars sent correctly)

---

## Questions?

This is a **LOT** of changes, but the flow is:
1. Test local → Works? → Deploy cloud → Test cloud → Works? → Compile EXE

You're currently on step 1 (test local). Once everything works, we deploy to Azure!
