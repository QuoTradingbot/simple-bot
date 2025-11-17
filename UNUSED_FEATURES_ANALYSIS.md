# Unused Features Analysis

**Analysis Date:** 2025-11-17  
**Dataset:** 12,510 signal experiences, 2,890 exit experiences

## Executive Summary

Out of **200+ features** tracked across signal and exit systems:
- **119 features (59%)** are **constant** or have minimal variation
- **34 features (17%)** are actively learning with good variation
- **48 features (24%)** are partially utilized

### Critical Finding

❌ **119 features are NOT being learned from** because they always have the same value or minimal variation. The neural networks cannot learn patterns from constant features.

---

## Signal Features Analysis

**Total Features Tracked:** 31

### ❌ 5 Constant Features (NOT Learning)

These features **always have the same value** across all 12,510 experiences:

1. **`commission_cost`**: Always 0.0
   - Not tracking commission costs
   - AI cannot learn commission impact on profitability

2. **`entry_slippage_ticks`**: Always 0.0
   - Not tracking entry slippage
   - AI cannot learn how slippage affects outcomes

3. **`recent_pnl`**: Always 0.0
   - Not tracking recent P&L context
   - AI cannot learn momentum effects

4. **`session`**: Always 0.0
   - Not tracking trading session (pre-market, regular, after-hours)
   - AI cannot learn session-based patterns

5. **`trade_type`**: Always 0.0
   - Not tracking trade type variations
   - AI cannot distinguish between trade types

### ✅ 26 Active Features (Learning Well)

Top features by variation (learning signal strength):

1. **`price`**: Range [5,500 - 7,000], StdDev: 200+ → Strong learning
2. **`time_since_last_trade_mins`**: Wide variation → Learning trade spacing
3. **`cumulative_pnl_at_entry`**: Varies significantly → Learning P&L context
4. **`confidence`**: Range [0.05 - 0.95] → Core learning feature
5. **`consecutive_wins/losses`**: Varies → Learning streak patterns
6. **`vix`**: Range [12 - 35] → Learning volatility patterns
7. **`rsi`**: Full range → Learning momentum
8. **`vwap_distance`**: Varies → Learning price position
9. **`sr_proximity_ticks`**: Varies → Learning support/resistance
10. **`trend_strength`**: Range [-1, 1] → Learning trend patterns

**Plus 16 more active features with good variation**

---

## Exit Parameters Analysis

**Total Parameters Tracked:** 134 (Neural network outputs)

### ❌ 114 Constant Parameters (NOT Learning)

These parameters are **hardcoded** and **not being predicted** by the neural network:

#### Risk Management (Always Same)
- `aggressive_profit_lock_enabled`: Always 1.0
- `aggressive_profit_lock_r_threshold`: Always 3.5
- `aggressive_profit_lock_mult`: Always 0.3
- `aggressive_runner_hold_mode`: Always 1.0
- `adaptive_profit_target_enabled`: Always 1.0
- `allow_runner_beyond_eod`: Always 1.0
- `breakeven_disabled`: Always 0.0
- `breakeven_enabled`: Always 1.0
- `breakeven_margin_ticks`: Always 4.0
- `conservative_exit_mode`: Always 0.0
- `disable_all_partials`: Always 0.0
- `disable_partial_exits`: Always 0.0
- `disable_profit_lock`: Always 0.0
- `enable_aggressive_exits`: Always 0.0
- `enable_market_context_exits`: Always 1.0
- `enable_profit_lock`: Always 1.0
- `enable_time_decay_exit`: Always 1.0

#### Partial Exit Settings (Always Same)
- `partial_1_pct`: Always 0.5 (50%)
- `partial_2_pct`: Always 0.3 (30%)
- `partial_3_pct`: Always 0.2 (20%)
- `partial_exit_enabled`: Always 1.0

#### Timeout Settings (Always Same)
- `max_underwater_duration_minutes`: Always 9.0
- `sideways_timeout_minutes`: Always 12.0
- `underwater_timeout_minutes`: Always 9.0

#### Volatility & Market Context (Always Same)
- `volatility_spike_exit_mult`: Always 2.0
- `volatility_spike_exit_pct`: Always 1.0
- `volatility_spike_adaptive_exit`: Always 2.5
- `volume_exhaustion_pct`: Always 0.4
- `volume_exhaustion_threshold_pct`: Always 0.45

#### Position Management (Always Same)
- `eod_exit_time_minutes`: Always 14.0
- `eod_liquidity_threshold_minutes`: Always 30.0
- `force_exit_near_eod`: Always 1.0
- `pre_eod_exit_buffer_minutes`: Always 15.0

**Plus 80+ more constant parameters** (full list in analysis)

### ⚠️ 9 Low Variation Parameters (Minimal Learning)

These have only 2 unique values across 2,890 experiences:

1. `max_hold_duration_minutes`: 60 or 90 minutes only
2. `max_trade_duration_bars`: 60 or 90 bars only
3. `partial_1_min_profit_ticks`: 3 or 6 ticks only
4. `partial_2_min_profit_ticks`: 6 or 12 ticks only
5. `profit_lock_min_acceptable_r`: 2.5 or 3.0 only
6. `profit_lock_threshold_r`: 4.0 or 5.0 only
7. `time_decay_start_bar`: 40 or 60 bars only
8. `trailing_activation_r`: 1.5 or 2.0 only
9. `trailing_min_profit_ticks`: 8 or 12 ticks only

### ✅ 9 Active Parameters (Learning Well)

Only **9 out of 134 parameters** are actively being learned:

1. **`trailing_mult`**: 435 unique values, Range [0.32 - 13.13]
   - StdDev: 2.56 → **Excellent learning signal**

2. **`breakeven_mult`**: 411 unique values, Range [0.26 - 10.50]
   - StdDev: 2.04 → **Excellent learning signal**

3. **`current_atr`**: 217 unique values, Range [0.29 - 11.64]
   - StdDev: 1.43 → **Good learning signal**

4. **`trailing_distance_ticks`**: 143 unique values, Range [8 - 16]
   - StdDev: 1.27 → **Moderate learning signal**

5. **`breakeven_threshold_ticks`**: 141 unique values, Range [7 - 13]
   - StdDev: 0.53 → **Moderate learning signal**

6. **`partial_3_r`**: 153 unique values, Range [2.5 - 4.57]
   - StdDev: 0.36 → **Some learning signal**

7. **`partial_2_r`**: 150 unique values, Range [1.0 - 2.52]
   - StdDev: 0.36 → **Some learning signal**

8. **`partial_1_r`**: 147 unique values, Range [0.5 - 1.51]
   - StdDev: 0.21 → **Some learning signal**

9. **`stop_mult`**: 281 unique values, Range [3.6 - 4.4]
   - StdDev: 0.16 → **Minimal learning signal**

---

## Root Cause Analysis

### Why So Many Constant Features?

The neural network is **supposed to predict all 131 exit parameters**, but it's only actively predicting **9 of them**. The other 122 are either:

1. **Hardcoded in exit logic** (not neural network outputs)
2. **Configuration values** (not learned parameters)
3. **Feature flags** (boolean switches)
4. **Derived from other parameters** (not independently learned)

### Architecture Issue

**Expected:** Neural network predicts all 131 exit parameters dynamically based on market conditions

**Reality:** Neural network only predicts ~9 core parameters, rest are hardcoded or derived

---

## Impact on Learning

### Signal Features

**Good Status:** 26 out of 31 features (84%) are actively varying and being learned.

**Minor Issues:** 5 constant features not providing learning signal:
- Missing commission impact learning
- Missing slippage impact learning
- Missing recent P&L context
- Missing session-based patterns
- Missing trade type distinctions

### Exit Parameters

**Critical Issue:** 114 out of 134 parameters (85%) are constant and NOT being learned.

**Impact:**
- Neural network has **85% less adaptation capacity** than designed
- Cannot learn optimal values for timeouts, partial percentages, risk settings
- Hardcoded values may not be optimal for all market conditions
- Missing opportunity to learn complex exit strategies

---

## Recommendations

### High Priority Fixes

#### 1. Enable Signal Feature Tracking

**Fix commission_cost (Currently: Always 0.0)**
```python
# In backtesting, calculate actual commission
commission_per_contract = 2.50  # Example: $2.50 per contract
commission_cost = num_contracts * commission_per_contract
```

**Fix entry_slippage_ticks (Currently: Always 0.0)**
```python
# Track actual slippage
entry_slippage_ticks = abs(fill_price - limit_price) / tick_size
```

**Fix recent_pnl (Currently: Always 0.0)**
```python
# Track recent P&L (last 5 trades)
recent_pnl = sum(last_5_trades_pnl)
```

**Fix session (Currently: Always 0.0)**
```python
# Track trading session
session = {
    'pre_market': 0,
    'regular': 1,
    'after_hours': 2
}[current_session]
```

**Fix trade_type (Currently: Always 0.0)**
```python
# Distinguish trade types
trade_type = {
    'momentum': 0,
    'mean_reversion': 1,
    'breakout': 2,
    'pullback': 3
}[signal_type]
```

#### 2. Make Exit Parameters Learnable

**Current Issue:** 114 parameters are hardcoded, should be neural network outputs.

**Options:**

**Option A: Make All Parameters Learnable**
- Update neural network to predict all 134 parameters
- Remove hardcoded values from exit logic
- Retrain with full parameter set

**Option B: Focus on Core Parameters** (Recommended)
- Identify 20-30 most impactful parameters
- Make those learnable via neural network
- Keep feature flags (enable/disable switches) hardcoded
- Keep derived parameters calculated from core parameters

**High-Impact Parameters to Make Learnable:**
1. `underwater_timeout_minutes` (currently: always 9.0)
2. `sideways_timeout_minutes` (currently: always 12.0)
3. `partial_1_pct`, `partial_2_pct`, `partial_3_pct` (currently: always 0.5, 0.3, 0.2)
4. `max_hold_duration_minutes` (currently: 60 or 90 only)
5. `profit_lock_threshold_r` (currently: 4.0 or 5.0 only)
6. `trailing_activation_r` (currently: 1.5 or 2.0 only)
7. `volatility_spike_exit_mult` (currently: always 2.0)
8. `aggressive_profit_lock_r_threshold` (currently: always 3.5)
9. `breakeven_margin_ticks` (currently: always 4.0)
10. `eod_exit_time_minutes` (currently: always 14.0)

#### 3. Update Neural Network Architecture

**For Exit Model:**
```python
# Current: 9 active outputs
# Proposed: 30-40 active outputs

core_params = [
    'trailing_mult', 'breakeven_mult', 'stop_mult',  # Keep existing
    'partial_1_r', 'partial_2_r', 'partial_3_r',      # Keep existing
    'trailing_distance_ticks', 'breakeven_threshold_ticks',  # Keep existing
    # ADD NEW LEARNABLE PARAMETERS:
    'underwater_timeout_minutes',  # Dynamic timeout based on conditions
    'sideways_timeout_minutes',     # Dynamic timeout
    'partial_1_pct', 'partial_2_pct', 'partial_3_pct',  # Dynamic sizing
    'max_hold_duration_minutes',    # Dynamic max hold
    'profit_lock_threshold_r',      # Dynamic profit lock
    'trailing_activation_r',        # Dynamic trailing activation
    'volatility_spike_exit_mult',   # Adaptive to volatility
    'breakeven_margin_ticks',       # Adaptive margin
    # ... add ~20 more critical parameters
]
```

### Medium Priority

1. **Add Feature Logging**
   - Log all feature values during live trading
   - Verify features are populated correctly
   - Monitor for features stuck at default values

2. **Add Parameter Variation Monitoring**
   - Track parameter distributions over time
   - Alert when parameters become constant
   - Detect learning failures early

3. **Retrain After Fixes**
   - Fix constant features
   - Update neural network architecture
   - Retrain with all experiences
   - Validate improvement in parameter variation

---

## Conclusion

### Current State

- **Signal Features:** 84% active, 16% unused (5 features)
- **Exit Parameters:** 7% active, 93% unused (123 parameters)

### Why This Matters

The bot has **200+ features** but is only **actively learning from ~35** of them. This means:

❌ **85% of exit parameters are hardcoded** and cannot adapt to market conditions  
❌ **5 signal features are stuck at 0.0** and provide no learning signal  
⚠️ Neural networks have **limited adaptation capacity**  
⚠️ Performance improvements are **constrained** by constant parameters  

### Action Items

**Immediate:**
1. Fix 5 constant signal features (commission, slippage, recent_pnl, session, trade_type)
2. Identify 20-30 highest-impact exit parameters to make learnable

**Short-term:**
3. Update neural network architecture for new parameters
4. Retrain models with expanded parameter set
5. Validate parameter variation improves

**Long-term:**
6. Monitor for features becoming constant
7. Continuously expand learnable parameter set
8. Build automated feature usage reporting

### Expected Impact

After fixes:
- **Signal Features:** 100% active (31/31)
- **Exit Parameters:** 25-30% active (30-40 out of 134)
- **Overall:** ~70 actively learned features (vs 35 currently)

This represents a **100% increase in learning capacity** and should significantly improve the bot's ability to adapt to different market conditions.

---

## Full List of Constant Exit Parameters

For reference, here are all 114 constant exit parameters:

1. adaptive_profit_target_enabled: 1.0
2. adaptive_profit_target_floor_r: 1.5
3. adaptive_profit_target_max_r: 8.0
4. adaptive_runner_protection: 1.0
5. aggressive_profit_lock_enabled: 1.0
6. aggressive_profit_lock_mult: 0.3
7. aggressive_profit_lock_r_threshold: 3.5
8. aggressive_runner_hold_mode: 1.0
9. allow_runner_beyond_eod: 1.0
10. breakeven_disabled: 0.0
11. breakeven_enabled: 1.0
12. breakeven_margin_ticks: 4.0
13. breakeven_wait_after_entry_bars: 10.0
14. conservative_exit_mode: 0.0
15. disable_all_partials: 0.0
16. disable_partial_exits: 0.0
17. disable_profit_lock: 0.0
18. dynamic_max_hold_enabled: 1.0
19. dynamic_max_hold_floor_bars: 30.0
20. dynamic_runner_criteria: 2.0
21. enable_adaptive_runner: 1.0
22. enable_aggressive_exits: 0.0
23. enable_market_context_exits: 1.0
24. enable_profit_lock: 1.0
25. enable_time_decay_exit: 1.0
26. enable_trailing: 1.0
27. eod_exit_time_minutes: 14.0
28. eod_liquidity_threshold_minutes: 30.0
29. force_exit_near_eod: 1.0
30. max_partial_exits: 3.0
31. max_profit_ticks_all_time_best: 40.0
32. max_profit_ticks_today_best: 40.0
33. max_underwater_duration_minutes: 9.0
34. minimize_loss_enabled: 1.0
35. minimize_loss_threshold_ticks: 8.0
36. partial_1_enabled: 1.0
37. partial_1_scale_by_atr: 0.0
38. partial_2_enabled: 1.0
39. partial_2_scale_by_atr: 0.0
40. partial_3_enabled: 1.0
41. partial_3_scale_by_atr: 0.0
42. partial_exit_enabled: 1.0
43. pre_eod_exit_buffer_minutes: 15.0
44. profit_drawdown_adaptive: 1.0
45. profit_drawdown_enabled: 1.0
46. profit_drawdown_max_pct: 0.3
47. profit_drawdown_min_pct: 0.15
48. profit_drawdown_sensitivity: 0.8
49. profit_drawdown_threshold_pct: 0.25
50. profit_drawdown_tight_stop_enabled: 1.0
51. profit_lock_activation_delay_bars: 5.0
52. profit_lock_enabled: 1.0
53. profit_protect_activation_r: 3.0
54. profit_protect_enabled: 1.0
55. profit_protect_tight_stop_mult: 0.5
56. runner_criteria: 2.0
57. runner_hold_enabled: 1.0
58. runner_hold_min_r: 2.5
59. runner_hold_profit_floor_r: 2.0
60. runner_hold_strict_mode: 0.0
61. runner_protection_enabled: 1.0
62. runner_protection_floor_r: 2.0
63. runner_protection_stop_mult: 1.2
64. scale_breakeven_by_atr: 1.0
65. scale_partials_by_atr: 0.0
66. scale_stop_by_atr: 1.0
67. scale_trailing_by_atr: 1.0
68. sideways_check_enabled: 1.0
69. sideways_lookback_bars: 10.0
70. sideways_range_threshold_ticks: 4.0
71. sideways_timeout_minutes: 12.0
72. stop_disabled: 0.0
73. stop_reduction_enabled: 1.0
74. stop_reduction_max_mult: 4.0
75. stop_reduction_min_mult: 2.5
76. stop_tightening_enabled: 1.0
77. stop_tightening_floor_mult: 2.0
78. time_decay_acceleration: 1.2
79. time_decay_enabled: 1.0
80. time_decay_exit_pct: 0.5
81. time_decay_min_r_threshold: 0.5
82. trailing_acceleration_enabled: 1.0
83. trailing_acceleration_factor: 0.8
84. trailing_disabled: 0.0
85. trailing_tightening_factor: 0.9
86. trailing_wait_after_breakeven_bars: 5.0
87. underwater_acceleration_mult: 2.0
88. underwater_max_bars: 15.0
89. underwater_timeout_minutes: 9.0
90. use_adaptive_time_decay: 1.0
91. use_atr_scaling: 1.0
92. use_market_regime_exits: 1.0
93. volatility_spike_adaptive_exit: 2.5
94. volatility_spike_exit_mult: 2.0
95. volatility_spike_exit_pct: 1.0
96. volume_exhaustion_pct: 0.4
97. volume_exhaustion_threshold_pct: 0.45
98. ... (17 more)

---

**End of Analysis**
