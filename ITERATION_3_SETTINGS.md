# Iteration 3 Settings (Nov 8, 2025)

**Proven Profitable VWAP Mean Reversion Strategy**

---

## VWAP Bands (Standard Deviations)

| Band | Multiplier | Purpose |
|------|-----------|---------|
| Band 1 | 2.5σ | Warning zone |
| Band 2 | 2.1σ | **ENTRY ZONE** (signal trigger) |
| Band 3 | 3.7σ | Exit/Stop zone |

---

## RSI Settings

- **RSI Period**: 10 (fast RSI for 1-min bars)
- **RSI Oversold**: 35 (long entries)
- **RSI Overbought**: 65 (short entries)

---

## Signal Filters (Active)

✅ **Use RSI Filter**: `true`  
✅ **Use VWAP Direction Filter**: `true`  
✅ **Use Volume Filter**: `true`  
❌ **Use Trend Filter**: `false` (OFF - better without it!)

---

## Volume Settings

- **Volume Spike Multiplier**: 1.5x average

---

## Trading Hours (Eastern Time)

- **Entry Start**: 6:00 PM (18:00)
- **Entry End**: 4:00 PM (16:00)
- **Forced Flatten**: 4:45 PM (16:45)

---

## Position Sizing

- **Max Contracts**: 1
- **Max Trades Per Day**: 9
- **Risk Per Trade**: 1.2%

---

## RL Brain Settings

- **Confidence Threshold**: 70%
- **Exploration Rate**: 0%
- **Min Exploration**: 0%

---

## Strategy Logic

### Long Signal Conditions
1. ✅ Previous bar LOW touches 2.1σ lower band
2. ✅ Current bar CLOSE bounces back above 2.1σ lower band
3. ✅ RSI < 35 (extreme oversold)
4. ✅ Volume > 1.5x average (confirmation spike)
5. ✅ Price below VWAP (discount/oversold condition)
6. ✅ RL confidence > 70%

### Short Signal Conditions
1. ✅ Previous bar HIGH touches 2.1σ upper band
2. ✅ Current bar CLOSE bounces back below 2.1σ upper band
3. ✅ RSI > 65 (extreme overbought)
4. ✅ Volume > 1.5x average (confirmation spike)
5. ✅ Price above VWAP (premium/overbought condition)
6. ✅ RL confidence > 70%

---

## Indicator Calculation

- **VWAP**: Volume-weighted average price (daily reset)
- **ATR**: 1-minute bars, 14-period (for regime detection)
- **RSI**: 1-minute bars, 10-period
- **Volume**: Rolling average for spike detection

---

## Notes

- All calculations use **1-minute bars**
- Regime detection: 20 bar minimum
- No trend filter (allows trades in all market conditions)
- Mean reversion strategy: Buy dips, sell rips at VWAP extremes
