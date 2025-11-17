# Exit Strategy Fix: Enable Partials and Let Runners Run

## Problem Identified (Fifth Backtest - 20% Threshold)

**Issues Found:**
1. **NO partial exits**: 0% of trades triggered partials
2. **Not letting runners run**: 0.34R average (target 2-3R)
3. **Early exits**: Winners held only 5 minutes average

## Root Cause Analysis

### Current Exit Parameter Defaults (Too Conservative)

**Partial Exit Thresholds:**
```python
partial_1_r: 1.2R (default) - TOO HIGH for scalping
partial_1_min_profit_ticks: 6 ticks - Reasonable but R-multiple blocking
partial_2_r: 2.0R (default) - TOO HIGH, rarely reached
partial_3_r: 3.5R (default) - RUNNER target, rarely reached
```

**Why Partials Don't Trigger:**
- Average winner exits at 0.34R (via profit_drawdown)
- 1.2R partial threshold never reached before main exit fires
- profit_drawdown exits entire position before partials can trigger
- Result: 0% partial exits, bot can't learn partial strategy

**Why Runners Don't Run:**
- Trailing stops too tight (10 ticks default)
- Profit_drawdown exits too early
- Time decay starts tightening after 40 bars
- Average trade duration: 5-6 minutes (too short for 2-3R)

## Recommended Fixes

### Fix 1: Lower Partial Exit Thresholds

**File:** `src/exit_params_config.py`

```python
# BEFORE (Current Defaults)
'partial_1_r': {
    'min': 0.8, 'max': 2.0, 'default': 1.2,  # ← TOO HIGH
    'description': 'First partial exit target (R-multiple)',
    'category': 'partials'
},
'partial_1_min_profit_ticks': {
    'min': 4, 'max': 15, 'default': 6,
    'description': 'Minimum profit in ticks for first partial',
    'category': 'partials'
},

# AFTER (Recommended)
'partial_1_r': {
    'min': 0.3, 'max': 1.0, 'default': 0.5,  # ← LOWERED to trigger at 0.5R
    'description': 'First partial exit target (R-multiple) - SCALPING OPTIMIZED',
    'category': 'partials'
},
'partial_1_min_profit_ticks': {
    'min': 2, 'max': 8, 'default': 3,  # ← LOWERED to 3 ticks minimum
    'description': 'Minimum profit in ticks for first partial - LOWERED',
    'category': 'partials'
},
```

**Second and Third Partials:**
```python
'partial_2_r': {
    'min': 0.8, 'max': 2.0, 'default': 1.0,  # ← Was 2.0R, now 1.0R
    'description': 'Second partial exit target (R-multiple) - LOWERED',
    'category': 'partials'
},
'partial_3_r': {
    'min': 1.5, 'max': 4.0, 'default': 2.5,  # ← Was 3.5R, now 2.5R (runner)
    'description': 'Third partial exit target (R-multiple) - RUNNER',
    'category': 'partials'
},
```

### Fix 2: Let Winners Run - Widen Trailing Stops

```python
# BEFORE
'trailing_distance_ticks': {
    'min': 6, 'max': 24, 'default': 10,  # ← TOO TIGHT
    'description': 'How far to trail behind price',
    'category': 'trailing'
},
'trailing_min_profit_ticks': {
    'min': 8, 'max': 20, 'default': 12,  # ← Starts too late
    'description': 'Minimum profit before trailing activates',
    'category': 'trailing'
},

# AFTER
'trailing_distance_ticks': {
    'min': 10, 'max': 30, 'default': 16,  # ← WIDENED to give room
    'description': 'How far to trail behind price - WIDENED',
    'category': 'trailing'
},
'trailing_min_profit_ticks': {
    'min': 6, 'max': 15, 'default': 8,  # ← LOWERED to start earlier
    'description': 'Minimum profit before trailing activates - LOWERED',
    'category': 'trailing'
},
'trailing_activation_r': {
    'min': 1.0, 'max': 3.0, 'default': 2.0,  # ← Was 1.5R, now 2.0R
    'description': 'R-multiple required to activate trailing stop - RAISED for runners',
    'category': 'trailing'
},
```

### Fix 3: Increase Max Hold Time for Runners

```python
# BEFORE
'max_hold_duration_minutes': {
    'min': 45, 'max': 90, 'default': 60,  # ← TOO SHORT for 2-3R targets
    'description': 'Absolute maximum trade duration',
    'category': 'time_based'
},

# AFTER
'max_hold_duration_minutes': {
    'min': 60, 'max': 120, 'default': 90,  # ← EXTENDED to allow runners
    'description': 'Absolute maximum trade duration - EXTENDED',
    'category': 'time_based'
},
'time_decay_start_bar': {
    'min': 30, 'max': 100, 'default': 60,  # ← Was 40, now 60 bars
    'description': 'Bar number to start tightening stops - DELAYED',
    'category': 'time_based'
},
```

### Fix 4: Adjust Profit Lock for Big Winners

```python
# BEFORE
'profit_lock_threshold_r': {
    'min': 3.0, 'max': 6.0, 'default': 4.0,  # ← When to protect profits
    'description': 'R-multiple that triggers profit protection mode',
    'category': 'adverse'
},

# AFTER
'profit_lock_threshold_r': {
    'min': 4.0, 'max': 8.0, 'default': 5.0,  # ← RAISED to let bigger wins develop
    'description': 'R-multiple that triggers profit protection mode - RAISED',
    'category': 'adverse'
},
'profit_lock_min_acceptable_r': {
    'min': 2.0, 'max': 5.0, 'default': 3.0,  # ← Was 2.5R, now 3.0R
    'description': 'Minimum R to accept after locking profit - RAISED',
    'category': 'adverse'
},
```

## Expected Results After Fixes

### Partial Exits
- **First partial** at 0.5R: Should trigger in 60-70% of winners
- **Second partial** at 1.0R: Should trigger in 30-40% of winners
- **Third partial (runner)** at 2.5R: Should trigger in 10-20% of winners

### Average R-Multiple Improvement
- **Current**: 0.34R average winner
- **After Fix**: 0.8-1.2R average winner (+135-250% improvement)
- **With Runners**: Occasional 2-5R wins (vs current max ~1.5R)

### Win Rate Impact
- **Current**: 70-75% WR (constant across runs)
- **After Fix**: May drop to 65-70% WR (holding longer = more stop-outs)
- **Net Profit**: Should increase despite lower WR (bigger wins compensate)

## Learning Benefit

**Why This Fixes the Learning Problem:**

1. **Partials Now Trigger**:
   - Current: 0% partial exits → no partial data collected
   - After fix: 60-70% partial exits → rich data for learning

2. **Exit Model Can Learn**:
   - Collects experiences with partial_1, partial_2, partial_3
   - Learns optimal partial sizes and timing
   - Adapts to different market conditions

3. **Gradual Optimization**:
   - Start with aggressive partials (0.5R, 1.0R, 2.5R)
   - Collect 500+ partial experiences
   - Retrain exit model
   - Model finds optimal balance
   - Gradually raise thresholds as model learns

## Implementation Steps

1. **Update `src/exit_params_config.py`** with new defaults
2. **Retrain exit model** (optional - can use current model with new params)
3. **Run backtest with 10% threshold** to validate
4. **Collect partial exit experiences** (100+ trades)
5. **Retrain exit model again** with partial data
6. **Fine-tune thresholds** based on learned behavior

## Risk Considerations

**Potential Downsides:**
- Holding longer = more exposure to reversals
- May see temporary drop in win rate (65-70% vs 70-75%)
- Larger drawdowns from peak (wider stops)

**Mitigations:**
- Daily loss limit still enforced ($1,000)
- Breakeven moves protect capital
- Adverse condition exits still active
- Can revert to tighter stops if needed

## Summary

**Current State:**
- 0% partial exits (thresholds too high)
- 0.34R average winners (exits too early)
- Bot can't learn partial strategy (no data)

**After Fixes:**
- 60-70% partial exits (realistic thresholds)
- 0.8-1.2R average winners (+135-250%)
- 10-20% runners hit 2-5R targets
- Rich learning data for exit model optimization

**Bottom Line:** Lower thresholds → partials trigger → bot learns → better exits → higher profits
