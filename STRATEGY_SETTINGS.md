# Daily Reversal Strategy Settings

These are the optimized settings for the daily reversal trading strategy. The strategy catches big moves and plays the reversal, targeting ~1 trade per day.

## 88-Day Backtest Results
```
Period:           2025-08-31 to 2025-11-27
Total Trades:     71 (Wins: 50, Losses: 21)
Win Rate:         70.4%
Profit Factor:    4.32
Net P&L:          +$17,430.00 (+34.86%)
Max Drawdown:     $1,042.50 (1.54%)
Avg Trade:        31.7 minutes
```

---

## Signal Detection Thresholds

### Flush/Move Detection
| Parameter | Value | Description |
|-----------|-------|-------------|
| `MIN_FLUSH_TICKS` | 12 | Minimum move size (12 ticks = $3 on ES) |
| `MIN_VELOCITY_TICKS_PER_BAR` | 2 | Speed of move (2 ticks per 1-min bar) |
| `FLUSH_LOOKBACK_BARS` | 7 | Look at last 7 one-minute bars |
| `NEAR_EXTREME_TICKS` | 8 | Entry must be within 8 ticks of extreme |

### RSI Thresholds
| Parameter | Value | Description |
|-----------|-------|-------------|
| `RSI_OVERSOLD_EXTREME` | 35 | RSI < 35 for LONG entry |
| `RSI_OVERBOUGHT_EXTREME` | 65 | RSI > 65 for SHORT entry |

### Volume Confirmation
| Parameter | Value | Description |
|-----------|-------|-------------|
| `VOLUME_SPIKE_THRESHOLD` | 1.5 | Current volume >= 1.5x 20-bar average |

### Allowed Market Regimes
- `HIGH_VOL_TRENDING`
- `HIGH_VOL_CHOPPY`
- `NORMAL_TRENDING`
- `NORMAL_CHOPPY`
- `NORMAL`

---

## Trade Management

### Stop Loss
| Parameter | Value | Description |
|-----------|-------|-------------|
| `STOP_BUFFER_TICKS` | 2 | Stop placed 2 ticks beyond flush extreme |

### Breakeven
| Parameter | Value | Description |
|-----------|-------|-------------|
| `BREAKEVEN_TRIGGER_TICKS` | 12 | Move stop to entry after 12 ticks profit |
| `BREAKEVEN_OFFSET_TICKS` | 1 | Entry + 1 tick buffer |

### Trailing Stop
| Parameter | Value | Description |
|-----------|-------|-------------|
| `TRAILING_TRIGGER_TICKS` | 15 | Start trailing after 15 ticks profit |
| `TRAILING_DISTANCE_TICKS` | 8 | Trail 8 ticks behind peak |

### Time Management
| Parameter | Value | Description |
|-----------|-------|-------------|
| `MAX_HOLD_BARS` | 20 | Optional time stop after 20 bars |

---

## Entry Conditions (All 9 Must Be TRUE)

### For LONG Entry (After Flush Down):
1. ✅ Flush Happened - Range of last 7 bars >= 12 ticks
2. ✅ Flush Was Fast - Velocity >= 2 ticks per bar
3. ✅ Near Bottom - Within 8 ticks of flush low
4. ✅ RSI Oversold - RSI < 35
5. ✅ Volume Spike - Current volume >= 1.5x average
6. ✅ Stopped Making New Lows - Current bar low >= previous bar low
7. ✅ Reversal Candle - Current bar closes green (close > open)
8. ✅ Below VWAP - Buying at discount
9. ✅ Regime Allows - HIGH_VOL or NORMAL regime

### For SHORT Entry (After Flush Up):
1. ✅ Pump Happened - Range of last 7 bars >= 12 ticks
2. ✅ Pump Was Fast - Velocity >= 2 ticks per bar
3. ✅ Near Top - Within 8 ticks of flush high
4. ✅ RSI Overbought - RSI > 65
5. ✅ Volume Spike - Current volume >= 1.5x average
6. ✅ Stopped Making New Highs - Current bar high <= previous bar high
7. ✅ Reversal Candle - Current bar closes red (close < open)
8. ✅ Above VWAP - Selling at premium
9. ✅ Regime Allows - HIGH_VOL or NORMAL regime

---

## ES Futures Specifications
| Parameter | Value |
|-----------|-------|
| Tick Size | $0.25 |
| Tick Value | $12.50 |
| Symbol | ES |

---

## File Locations
- Strategy Logic: `src/capitulation_detector.py`
- Main Engine: `src/quotrading_engine.py`
- Backtest Runner: `dev/run_backtest.py`
- Experiences: `experiences/ES/signal_experience.json`
