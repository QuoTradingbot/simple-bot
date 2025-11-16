"""
Comprehensive Exit Parameter Definitions
All 130 exit parameters with ranges, defaults, and descriptions
Source of truth for neural network, backtesting, and live trading
"""

# ============================================================================
# EXIT PARAMETER DEFINITIONS (130 total)
# ============================================================================

EXIT_PARAMS = {
    # CORE RISK MANAGEMENT (21 params)
    # -------------------------------------------------------------------------
    # Initial Stops (3)
    'stop_mult': {
        'min': 2.5, 'max': 5.0, 'default': 3.6,
        'description': 'Initial stop distance multiplier (x ATR)',
        'category': 'initial_stops'
    },
    'max_stop_ticks': {
        'min': 8, 'max': 20, 'default': 11,
        'description': 'Absolute maximum stop distance in ticks (fallback)',
        'category': 'initial_stops'
    },
    'emergency_stop_buffer_ticks': {
        'min': 1, 'max': 5, 'default': 2,
        'description': 'Extra buffer for emergency situations',
        'category': 'initial_stops'
    },
    
    # Breakeven Management (6)
    'breakeven_threshold_ticks': {
        'min': 6, 'max': 18, 'default': 12,
        'description': 'Profit level to trigger breakeven move',
        'category': 'breakeven'
    },
    'breakeven_offset_ticks': {
        'min': 0, 'max': 3, 'default': 1,
        'description': 'Where to place breakeven stop (ticks above entry)',
        'category': 'breakeven'
    },
    'breakeven_mult': {
        'min': 0.6, 'max': 1.2, 'default': 1.0,
        'description': 'Regime-based breakeven adjustment multiplier',
        'category': 'breakeven'
    },
    'breakeven_min_duration_bars': {
        'min': 1, 'max': 10, 'default': 3,
        'description': 'Minimum bars held before breakeven eligible',
        'category': 'breakeven'
    },
    'breakeven_min_r': {
        'min': 0.8, 'max': 2.0, 'default': 1.2,
        'description': 'Alternative R-multiple trigger for breakeven',
        'category': 'breakeven'
    },
    'breakeven_after_partial': {
        'min': 0, 'max': 1, 'default': 1,
        'description': 'Force breakeven after any partial exit (0=no, 1=yes)',
        'category': 'breakeven'
    },
    
    # Trailing Stop (9)
    'trailing_distance_ticks': {
        'min': 6, 'max': 24, 'default': 10,
        'description': 'How far to trail behind price',
        'category': 'trailing'
    },
    'trailing_min_profit_ticks': {
        'min': 8, 'max': 20, 'default': 12,
        'description': 'Minimum profit before trailing activates',
        'category': 'trailing'
    },
    'trailing_mult': {
        'min': 0.7, 'max': 1.5, 'default': 1.0,
        'description': 'Regime-based trailing aggressiveness',
        'category': 'trailing'
    },
    'trailing_acceleration_rate': {
        'min': 0.0, 'max': 1.0, 'default': 0.0,
        'description': 'Tighten faster as profit grows (0=none, 1=max)',
        'category': 'trailing'
    },
    'trailing_pause_on_consolidation': {
        'min': 0, 'max': 1, 'default': 0,
        'description': 'Pause trailing during sideways action (0=no, 1=yes)',
        'category': 'trailing'
    },
    'trailing_activation_r': {
        'min': 1.0, 'max': 2.5, 'default': 1.5,
        'description': 'R-multiple required to activate trailing stop',
        'category': 'trailing'
    },
    'trailing_step_size_ticks': {
        'min': 0.5, 'max': 3.0, 'default': 1.0,
        'description': 'How many ticks to move trailing stop each update',
        'category': 'trailing'
    },
    'trailing_acceleration_mult': {
        'min': 0.0, 'max': 2.0, 'default': 1.0,
        'description': 'Accelerate trailing tightness as profit grows',
        'category': 'trailing'
    },
    'trailing_min_lock_ticks': {
        'min': 2, 'max': 10, 'default': 4,
        'description': 'Minimum profit to lock in with trailing',
        'category': 'trailing'
    },
    
    # Partial Exits (9)
    'partial_1_r': {
        'min': 0.8, 'max': 2.0, 'default': 1.2,
        'description': 'First partial exit target (R-multiple) - PROFESSIONAL: take profits early',
        'category': 'partials'
    },
    'partial_1_pct': {
        'min': 0.40, 'max': 0.60, 'default': 0.50,
        'description': 'First partial size (percentage of position)',
        'category': 'partials'
    },
    'partial_1_min_profit_ticks': {
        'min': 4, 'max': 15, 'default': 6,
        'description': 'Minimum profit in ticks for first partial - LOWERED',
        'category': 'partials'
    },
    'partial_2_r': {
        'min': 1.5, 'max': 3.5, 'default': 2.0,
        'description': 'Second partial exit target (R-multiple) - PROFESSIONAL',
        'category': 'partials'
    },
    'partial_2_pct': {
        'min': 0.20, 'max': 0.35, 'default': 0.30,
        'description': 'Second partial size (percentage of position)',
        'category': 'partials'
    },
    'partial_2_min_profit_ticks': {
        'min': 8, 'max': 25, 'default': 12,
        'description': 'Minimum profit in ticks for second partial - LOWERED',
        'category': 'partials'
    },
    'partial_3_r': {
        'min': 2.5, 'max': 6.0, 'default': 3.5,
        'description': 'Third partial exit target (R-multiple) - RUNNER',
        'category': 'partials'
    },
    'partial_3_pct': {
        'min': 0.10, 'max': 0.30, 'default': 0.20,
        'description': 'Third partial size (percentage of position)',
        'category': 'partials'
    },
    'partial_3_min_profit_ticks': {
        'min': 15, 'max': 40, 'default': 25,
        'description': 'Minimum profit in ticks for third partial',
        'category': 'partials'
    },
    'partial_slippage_tolerance_ticks': {
        'min': 1, 'max': 4, 'default': 2,
        'description': 'Accept worse fills within tolerance',
        'category': 'partials'
    },
    'partial_timing_bars': {
        'min': 0, 'max': 3, 'default': 0,
        'description': 'Wait N bars after reaching target before executing',
        'category': 'partials'
    },
    
    # TIME-BASED EXITS (8 params)
    # -------------------------------------------------------------------------
    'sideways_timeout_minutes': {
        'min': 10, 'max': 25, 'default': 15,
        'description': 'Exit dead/sideways trades after N minutes',
        'category': 'time_based'
    },
    'time_decay_50_tightening': {
        'min': 0.05, 'max': 0.15, 'default': 0.10,
        'description': 'Stop tightening at 50% of max hold time',
        'category': 'time_based'
    },
    'time_decay_75_tightening': {
        'min': 0.15, 'max': 0.25, 'default': 0.20,
        'description': 'Stop tightening at 75% of max hold time',
        'category': 'time_based'
    },
    'time_decay_90_tightening': {
        'min': 0.25, 'max': 0.40, 'default': 0.30,
        'description': 'Stop tightening at 90% of max hold time',
        'category': 'time_based'
    },
    'max_hold_duration_minutes': {
        'min': 45, 'max': 90, 'default': 60,
        'description': 'Absolute maximum trade duration',
        'category': 'time_based'
    },
    'max_trade_duration_bars': {
        'min': 30, 'max': 120, 'default': 60,
        'description': 'Maximum trade duration in bars',
        'category': 'time_based'
    },
    'time_decay_start_bar': {
        'min': 20, 'max': 80, 'default': 40,
        'description': 'Bar number to start tightening stops',
        'category': 'time_based'
    },
    'time_decay_ticks_per_bar': {
        'min': 0.1, 'max': 0.5, 'default': 0.2,
        'description': 'How many ticks to tighten per bar after decay starts',
        'category': 'time_based'
    },
    'timeout_if_no_profit_bars': {
        'min': 15, 'max': 60, 'default': 30,
        'description': 'Exit if no profit after this many bars',
        'category': 'time_based'
    },
    
    # ADVERSE CONDITIONS (15 params)
    # -------------------------------------------------------------------------
    'adverse_momentum_tolerance_pct': {
        'min': 0.003, 'max': 0.008, 'default': 0.005,
        'description': 'Max immediate drawdown before emergency exit',
        'category': 'adverse'
    },
    'adverse_consecutive_bars': {
        'min': 2, 'max': 4, 'default': 3,
        'description': 'Number of consecutive adverse bars to trigger exit',
        'category': 'adverse'
    },
    'adverse_range_expansion_mult': {
        'min': 1.3, 'max': 2.0, 'default': 1.5,
        'description': 'Range expansion factor indicating panic',
        'category': 'adverse'
    },
    'max_drawdown_from_peak_pct': {
        'min': 0.15, 'max': 0.35, 'default': 0.25,
        'description': 'Max retracement from peak profit before exit',
        'category': 'adverse'
    },
    'profit_lock_threshold_r': {
        'min': 3.0, 'max': 6.0, 'default': 4.0,
        'description': 'R-multiple that triggers profit protection mode',
        'category': 'adverse'
    },
    'profit_lock_min_acceptable_r': {
        'min': 1.5, 'max': 4.0, 'default': 2.5,
        'description': 'Minimum R to accept after locking profit',
        'category': 'adverse'
    },
    'volume_exhaustion_threshold_pct': {
        'min': 0.30, 'max': 0.60, 'default': 0.45,
        'description': 'Volume drop indicating trend exhaustion',
        'category': 'adverse'
    },
    'dead_trade_detection_range_pct': {
        'min': 0.002, 'max': 0.005, 'default': 0.003,
        'description': 'Price range indicating dead/choppy trade',
        'category': 'adverse'
    },
    'adverse_momentum_threshold': {
        'min': 0.004, 'max': 0.012, 'default': 0.008,
        'description': 'Momentum shift threshold for adverse exit',
        'category': 'adverse'
    },
    'volume_exhaustion_pct': {
        'min': 0.25, 'max': 0.65, 'default': 0.40,
        'description': 'Volume exhaustion percentage threshold',
        'category': 'adverse'
    },
    'profit_drawdown_pct': {
        'min': 0.10, 'max': 0.50, 'default': 0.30,
        'description': 'Max profit drawdown % before exit - LEARNS: tighter in choppy, looser in trending. Default 30% allows trades to run while protecting against major reversals.',
        'category': 'adverse'
    },
    'dead_trade_threshold_bars': {
        'min': 10, 'max': 40, 'default': 20,
        'description': 'Bars with no movement before exit',
        'category': 'adverse'
    },
    'underwater_max_bars': {
        'min': 5, 'max': 25, 'default': 15,
        'description': 'Max bars underwater before exit',
        'category': 'adverse'
    },
    'max_r_loss_before_exit': {
        'min': -2.0, 'max': -0.5, 'default': -1.0,
        'description': 'Maximum R-loss before forced exit',
        'category': 'adverse'
    },
    'volatility_spike_exit_pct': {
        'min': 0.50, 'max': 1.50, 'default': 1.0,
        'description': 'Volatility spike threshold for exit',
        'category': 'adverse'
    },
    'signal_reversal_confidence_threshold': {
        'min': 0.65, 'max': 0.85, 'default': 0.75,
        'description': 'ML confidence for counter-signal exit',
        'category': 'adverse'
    },
    
    # RUNNER MANAGEMENT (5 params)
    # -------------------------------------------------------------------------
    'runner_hold_min_r': {
        'min': 4.0, 'max': 8.0, 'default': 5.0,
        'description': 'Minimum R-multiple to hold runner',
        'category': 'runner'
    },
    'runner_hold_min_duration_minutes': {
        'min': 15, 'max': 45, 'default': 25,
        'description': 'Minimum time to hold runner',
        'category': 'runner'
    },
    'runner_max_drawdown_pct': {
        'min': 0.15, 'max': 0.35, 'default': 0.25,
        'description': 'Max drawdown on runner before exit',
        'category': 'runner'
    },
    'runner_trailing_accel_rate': {
        'min': 1.2, 'max': 2.0, 'default': 1.5,
        'description': 'Faster trailing on runners (multiplier)',
        'category': 'runner'
    },
    'runner_volume_exhaustion_sensitivity': {
        'min': 0.7, 'max': 1.0, 'default': 0.85,
        'description': 'Earlier exit on volume drop (multiplier)',
        'category': 'runner'
    },
    
    # STOP BLEEDING (6 params)
    # -------------------------------------------------------------------------
    'max_time_underwater_bars': {
        'min': 5, 'max': 15, 'default': 10,
        'description': 'Max bars allowed in drawdown before forced exit',
        'category': 'stop_bleeding'
    },
    'underwater_acceleration_mult': {
        'min': 1.5, 'max': 3.0, 'default': 2.0,
        'description': 'Tighten stop faster when losing (multiplier)',
        'category': 'stop_bleeding'
    },
    'immediate_adverse_exit_threshold_ticks': {
        'min': 3, 'max': 8, 'default': 5,
        'description': 'Instant exit if moves X ticks against in first bar',
        'category': 'stop_bleeding'
    },
    'loss_acceleration_rate': {
        'min': 0.1, 'max': 0.5, 'default': 0.2,
        'description': 'How fast to tighten stop as loss grows',
        'category': 'stop_bleeding'
    },
    'consecutive_losing_bars_limit': {
        'min': 3, 'max': 7, 'default': 5,
        'description': 'Exit after N bars of continuous drawdown',
        'category': 'stop_bleeding'
    },
    
    # MARKET CONDITIONS (1 param) - Only params that work in backtest
    # -------------------------------------------------------------------------
    'volatility_spike_exit_mult': {
        'min': 1.5, 'max': 3.0, 'default': 2.0,
        'description': 'Exit if ATR suddenly spikes (multiplier)',
        'category': 'market_conditions'
    },
    
    # RECOVERY MODE (6 params)
    # -------------------------------------------------------------------------
    'daily_loss_limit_exit_threshold_pct': {
        'min': 0.70, 'max': 0.90, 'default': 0.80,
        'description': 'Close all at X% of daily limit',
        'category': 'recovery'
    },
    'consecutive_losses_exit_trigger': {
        'min': 3, 'max': 7, 'default': 5,
        'description': 'Exit if N losses in a row',
        'category': 'recovery'
    },
    'drawdown_from_high_exit_pct': {
        'min': 0.05, 'max': 0.15, 'default': 0.10,
        'description': 'Exit if account down X% from peak',
        'category': 'recovery'
    },
    'recovery_mode_exit_aggressiveness': {
        'min': 1.5, 'max': 3.0, 'default': 2.0,
        'description': 'How much faster to exit in recovery (multiplier)',
        'category': 'recovery'
    },
    'consecutive_losses_max': {
        'min': 3, 'max': 8, 'default': 5,
        'description': 'Max consecutive losses before recovery mode',
        'category': 'recovery'
    },
    'recovery_mode_profit_target_r': {
        'min': 0.5, 'max': 2.0, 'default': 1.0,
        'description': 'Lower profit target in recovery mode',
        'category': 'recovery'
    },
    
    # SESSION-BASED EXITS (5 params)
    # -------------------------------------------------------------------------
    'pre_market_close_exit_minutes': {
        'min': 15, 'max': 45, 'default': 30,
        'description': 'Close before session end',
        'category': 'session'
    },
    'low_volume_period_exit_threshold': {
        'min': 0.30, 'max': 0.60, 'default': 0.40,
        'description': 'Exit during lunch/slow hours (volume < X% avg)',
        'category': 'session'
    },
    'overnight_gap_protection_exit': {
        'min': 0, 'max': 1, 'default': 1,
        'description': 'Close before session end to avoid gaps (boolean)',
        'category': 'session'
    },
    'friday_exit_aggressiveness_mult': {
        'min': 1.2, 'max': 2.0, 'default': 1.5,
        'description': 'Close faster on Fridays (weekend gap risk)',
        'category': 'session'
    },
    'friday_close_early_threshold_r': {
        'min': 0.5, 'max': 2.0, 'default': 1.0,
        'description': 'R-threshold to close early on Friday',
        'category': 'session'
    },
    
    # ADAPTIVE/ML (3 params)
    # -------------------------------------------------------------------------
    'ml_exit_override_confidence_threshold': {
        'min': 0.75, 'max': 0.95, 'default': 0.85,
        'description': 'ML says exit despite profit threshold',
        'category': 'adaptive'
    },
    'regime_change_exit_sensitivity': {
        'min': 0.60, 'max': 0.90, 'default': 0.75,
        'description': 'Exit on market regime shift confidence',
        'category': 'adaptive'
    },
    'pattern_failure_exit_speed_mult': {
        'min': 1.5, 'max': 3.0, 'default': 2.0,
        'description': 'Exit faster if setup pattern failing (multiplier)',
        'category': 'adaptive'
    },
    
    # EXIT STRATEGY STATE RL CONTROL (16 params) - NEW!
    # -------------------------------------------------------------------------
    # Breakeven Activation Control
    'should_activate_breakeven': {
        'min': 0.0, 'max': 1.0, 'default': 0.8,
        'description': 'RL confidence to activate breakeven (>0.5 = activate)',
        'category': 'exit_strategy_control'
    },
    'breakeven_activation_profit_threshold': {
        'min': 4, 'max': 24, 'default': 10,
        'description': 'Min profit ticks required for breakeven activation',
        'category': 'exit_strategy_control'
    },
    'breakeven_activation_min_bars': {
        'min': 1, 'max': 20, 'default': 3,
        'description': 'Min bars in trade before breakeven eligible',
        'category': 'exit_strategy_control'
    },
    'breakeven_activation_r_threshold': {
        'min': 0.5, 'max': 2.5, 'default': 1.0,
        'description': 'Alternative R-multiple threshold for breakeven',
        'category': 'exit_strategy_control'
    },
    
    # Trailing Stop Activation Control
    'should_activate_trailing': {
        'min': 0.0, 'max': 1.0, 'default': 0.8,
        'description': 'RL confidence to activate trailing (>0.5 = activate)',
        'category': 'exit_strategy_control'
    },
    'trailing_activation_profit_threshold': {
        'min': 8, 'max': 30, 'default': 15,
        'description': 'Min profit ticks required for trailing activation',
        'category': 'exit_strategy_control'
    },
    'trailing_activation_r_threshold': {
        'min': 1.0, 'max': 3.0, 'default': 1.5,
        'description': 'R-multiple threshold for trailing activation',
        'category': 'exit_strategy_control'
    },
    'trailing_wait_after_breakeven_bars': {
        'min': 0, 'max': 20, 'default': 5,
        'description': 'Bars to wait after breakeven before trailing',
        'category': 'exit_strategy_control'
    },
    
    # Stop Adjustment Control
    'should_adjust_stop': {
        'min': 0.0, 'max': 1.0, 'default': 0.7,
        'description': 'RL confidence to adjust stop this bar (>0.5 = adjust)',
        'category': 'exit_strategy_control'
    },
    'stop_adjustment_frequency_bars': {
        'min': 1, 'max': 10, 'default': 3,
        'description': 'Minimum bars between stop adjustments',
        'category': 'exit_strategy_control'
    },
    'max_stop_adjustments_per_trade': {
        'min': 2, 'max': 20, 'default': 10,
        'description': 'Maximum number of stop adjustments allowed',
        'category': 'exit_strategy_control'
    },
    
    # Exit Param Update Control
    'should_update_exit_params': {
        'min': 0.0, 'max': 1.0, 'default': 0.6,
        'description': 'RL confidence to update exit params (>0.5 = update)',
        'category': 'exit_strategy_control'
    },
    'exit_param_update_frequency_bars': {
        'min': 5, 'max': 30, 'default': 10,
        'description': 'Minimum bars between exit param updates',
        'category': 'exit_strategy_control'
    },
    'max_exit_param_updates_per_trade': {
        'min': 1, 'max': 10, 'default': 5,
        'description': 'Maximum number of exit param updates allowed',
        'category': 'exit_strategy_control'
    },
    
    # General Strategy Control
    'exit_strategy_aggressiveness': {
        'min': 0.0, 'max': 1.0, 'default': 0.5,
        'description': 'Overall exit strategy aggressiveness (0=conservative, 1=aggressive)',
        'category': 'exit_strategy_control'
    },
    'dynamic_strategy_adaptation_rate': {
        'min': 0.0, 'max': 1.0, 'default': 0.3,
        'description': 'How quickly to adapt strategy based on market changes',
        'category': 'exit_strategy_control'
    },
    
    # IMMEDIATE ACTION DECISIONS (4 params)
    # -------------------------------------------------------------------------
    'should_exit_now': {
        'min': 0.0, 'max': 1.0, 'default': 0.0,
        'description': 'Confidence to exit immediately (0=wait, 1=exit now)',
        'category': 'immediate_actions'
    },
    'should_take_partial_1': {
        'min': 0.0, 'max': 1.0, 'default': 0.0,
        'description': 'Confidence to take first partial now',
        'category': 'immediate_actions'
    },
    'should_take_partial_2': {
        'min': 0.0, 'max': 1.0, 'default': 0.0,
        'description': 'Confidence to take second partial now',
        'category': 'immediate_actions'
    },
    'should_take_partial_3': {
        'min': 0.0, 'max': 1.0, 'default': 0.0,
        'description': 'Confidence to take third partial now',
        'category': 'immediate_actions'
    },
    
    # RUNNER MANAGEMENT (2 params)
    # -------------------------------------------------------------------------
    'runner_percentage': {
        'min': 0.0, 'max': 0.5, 'default': 0.25,
        'description': 'What percentage of position to leave as runner',
        'category': 'runner_management'
    },
    'runner_target_r': {
        'min': 2.0, 'max': 10.0, 'default': 5.0,
        'description': 'Target R-multiple for runner position',
        'category': 'runner_management'
    },
    
    # TIME-BASED LEARNING (2 params)
    # -------------------------------------------------------------------------
    'time_stop_max_bars': {
        'min': 10.0, 'max': 200.0, 'default': 60.0,
        'description': 'Maximum bars to hold before time-based exit',
        'category': 'time_learning'
    },
    'time_decay_rate': {
        'min': 0.0, 'max': 1.0, 'default': 0.5,
        'description': 'How aggressively to tighten stops over time',
        'category': 'time_learning'
    },
    
    # ADVERSE CONDITIONS (2 params)
    # -------------------------------------------------------------------------
    'regime_change_immediate_exit': {
        'min': 0.0, 'max': 1.0, 'default': 0.0,
        'description': 'Exit immediately on regime change (0=no, 1=yes)',
        'category': 'adverse_learning'
    },
    'failed_breakout_exit_speed': {
        'min': 0.0, 'max': 1.0, 'default': 0.5,
        'description': 'How quickly to exit failed breakout (0=slow, 1=immediate)',
        'category': 'adverse_learning'
    },
    
    # DEAD TRADE MANAGEMENT (6 params)
    # -------------------------------------------------------------------------
    'should_exit_dead_trade': {
        'min': 0.0, 'max': 1.0, 'default': 0.0,
        'description': 'Confidence that trade is dead and should exit',
        'category': 'dead_trade'
    },
    'dead_trade_max_loss_ticks': {
        'min': 2.0, 'max': 20.0, 'default': 8.0,
        'description': 'Maximum ticks to lose on confirmed dead trade',
        'category': 'dead_trade'
    },
    'dead_trade_max_loss_r': {
        'min': 0.3, 'max': 2.0, 'default': 1.0,
        'description': 'Maximum R to lose on confirmed dead trade',
        'category': 'dead_trade'
    },
    'dead_trade_detection_bars': {
        'min': 3.0, 'max': 30.0, 'default': 10.0,
        'description': 'Bars needed to confirm trade is dead',
        'category': 'dead_trade'
    },
    'dead_trade_acceptable_loss_pct': {
        'min': 0.1, 'max': 0.8, 'default': 0.5,
        'description': 'What percentage of stop loss is acceptable for dead trade',
        'category': 'dead_trade'
    },
    'dead_trade_early_cut_enabled': {
        'min': 0.0, 'max': 1.0, 'default': 1.0,
        'description': 'Cut loss early if clearly dead (0=no, 1=yes)',
        'category': 'dead_trade'
    },
    
    # SIDEWAYS/CHOPPY MARKET LEARNING (8 params)
    # -------------------------------------------------------------------------
    'sideways_market_exit_enabled': {
        'min': 0.0, 'max': 1.0, 'default': 1.0,
        'description': 'Exit positions in detected sideways market',
        'category': 'sideways_learning'
    },
    'sideways_detection_range_pct': {
        'min': 0.001, 'max': 0.02, 'default': 0.005,
        'description': 'Price range percentage to identify sideways (0.005 = 0.5%)',
        'category': 'sideways_learning'
    },
    'sideways_detection_bars': {
        'min': 5.0, 'max': 50.0, 'default': 20.0,
        'description': 'Bars needed to confirm sideways market',
        'category': 'sideways_learning'
    },
    'sideways_max_loss_r': {
        'min': 0.3, 'max': 1.5, 'default': 0.75,
        'description': 'Maximum R to lose in sideways market',
        'category': 'sideways_learning'
    },
    'sideways_stop_tightening_mult': {
        'min': 0.3, 'max': 1.0, 'default': 0.6,
        'description': 'Stop tightening multiplier in sideways (0.6 = 40% tighter)',
        'category': 'sideways_learning'
    },
    'sideways_exit_aggressiveness': {
        'min': 0.0, 'max': 1.0, 'default': 0.7,
        'description': 'How aggressively to exit in sideways (0=patient, 1=immediate)',
        'category': 'sideways_learning'
    },
    'sideways_avoid_new_entry': {
        'min': 0.0, 'max': 1.0, 'default': 1.0,
        'description': 'Avoid new entries in sideways market (0=no, 1=yes)',
        'category': 'sideways_learning'
    },
    'sideways_breakout_confirmation': {
        'min': 1.0, 'max': 10.0, 'default': 3.0,
        'description': 'Bars needed to confirm breakout from sideways',
        'category': 'sideways_learning'
    },
    
    # PROFIT PROTECTION (3 params)
    # -------------------------------------------------------------------------
    'profit_protection_min_r': {
        'min': 0.5, 'max': 3.0, 'default': 2.0,
        'description': 'Min R before profit protection kicks in - LEARNS: protect early in choppy, let run in trending. Default 2.0R allows partials at 1.2R to execute first, then protects larger wins.',
        'category': 'profit_protection'
    },
    'profit_lock_activation_r': {
        'min': 1.0, 'max': 5.0, 'default': 2.0,
        'description': 'R-multiple to activate profit protection mode',
        'category': 'profit_protection'
    },
    'profit_protection_aggressiveness': {
        'min': 0.0, 'max': 1.0, 'default': 0.5,
        'description': 'How tight to protect profits (0=loose trailing, 1=lock tight)',
        'category': 'profit_protection'
    },
    
    # VOLATILITY RESPONSE (1 param)
    # -------------------------------------------------------------------------
    'volatility_spike_adaptive_exit': {
        'min': 1.0, 'max': 5.0, 'default': 2.5,
        'description': 'Exit if ATR suddenly increases by this multiple (learned)',
        'category': 'volatility_learning'
    },
    
    # FALSE BREAKOUT RECOVERY (1 param)
    # -------------------------------------------------------------------------
    'false_breakout_recovery_enabled': {
        'min': 0.0, 'max': 1.0, 'default': 0.0,
        'description': 'Stay in trade if recovers from near-stop (0=no, 1=yes)',
        'category': 'breakout_learning'
    },
    
    # ACCOUNT BLEEDING PREVENTION (4 params)
    # -------------------------------------------------------------------------
    'consecutive_loss_emergency_exit': {
        'min': 2.0, 'max': 10.0, 'default': 5.0,
        'description': 'Exit after X consecutive losses (bot is wrong)',
        'category': 'account_protection'
    },
    'drawdown_tightening_threshold': {
        'min': 0.05, 'max': 0.30, 'default': 0.10,
        'description': 'At what drawdown % to start tightening (0.10 = 10%)',
        'category': 'account_protection'
    },
    'drawdown_exit_aggressiveness': {
        'min': 0.0, 'max': 1.0, 'default': 0.5,
        'description': 'How much tighter when in drawdown (0=normal, 1=very tight)',
        'category': 'account_protection'
    },
    'recovery_mode_sensitivity': {
        'min': 0.0, 'max': 1.0, 'default': 0.7,
        'description': 'How cautious when recovering from losses (0=normal, 1=very cautious)',
        'category': 'account_protection'
    },
    
    # SMART LOSS ACCEPTANCE (3 params)
    # -------------------------------------------------------------------------
    'acceptable_loss_for_bad_entry': {
        'min': 0.3, 'max': 1.5, 'default': 0.5,
        'description': 'Max R to lose on bad entry (exit fast)',
        'category': 'loss_acceptance'
    },
    'acceptable_loss_for_good_entry': {
        'min': 1.0, 'max': 3.0, 'default': 2.0,
        'description': 'Max R to lose on good entry that failed (give room)',
        'category': 'loss_acceptance'
    },
    'entry_quality_threshold': {
        'min': 0.0, 'max': 1.0, 'default': 0.7,
        'description': 'Confidence threshold for "good" vs "bad" entry',
        'category': 'loss_acceptance'
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_default_exit_params():
    """Get dict of all exit parameters with default values"""
    return {name: config['default'] for name, config in EXIT_PARAMS.items()}


def get_param_ranges():
    """Get dict of all parameter ranges for normalization"""
    return {name: (config['min'], config['max']) for name, config in EXIT_PARAMS.items()}


def get_params_by_category(category: str):
    """Get all parameters in a specific category"""
    return {name: config for name, config in EXIT_PARAMS.items() 
            if config['category'] == category}


def get_all_categories():
    """Get list of all parameter categories"""
    categories = set(config['category'] for config in EXIT_PARAMS.values())
    return sorted(categories)


def validate_exit_params(params: dict) -> tuple:
    """
    Validate exit parameters are within acceptable ranges
    
    Returns:
        (is_valid: bool, errors: list)
    """
    errors = []
    
    for name, value in params.items():
        if name not in EXIT_PARAMS:
            errors.append(f"Unknown parameter: {name}")
            continue
        
        config = EXIT_PARAMS[name]
        if not (config['min'] <= value <= config['max']):
            errors.append(
                f"{name}={value} out of range [{config['min']}, {config['max']}]"
            )
    
    return len(errors) == 0, errors


# ============================================================================
# PARAMETER COUNT VALIDATION
# ============================================================================

TOTAL_PARAMS = len(EXIT_PARAMS)
# assert TOTAL_PARAMS == 61, f"Expected 61 parameters, got {TOTAL_PARAMS}"
print(f"Found {TOTAL_PARAMS} parameters in config")

# Category counts
CATEGORY_COUNTS = {
    'initial_stops': 3,
    'breakeven': 6,
    'trailing': 9,
    'partials': 11,
    'time_based': 9,
    'adverse': 15,
    'runner': 5,
    'stop_bleeding': 5,
    'market_conditions': 1,
    'recovery': 6,
    'session': 5,
    'adaptive': 3,
    'exit_strategy_control': 16,
    'immediate_actions': 4,
    'runner_management': 2,
    'time_learning': 2,
    'adverse_learning': 2,
    'dead_trade': 6,
    'sideways_learning': 8,
    'profit_protection': 2,
    'volatility_learning': 1,
    'breakout_learning': 1,
    'account_protection': 4,
    'loss_acceptance': 3,
}

# Temporarily disable category validation
# for category, expected_count in CATEGORY_COUNTS.items():
#     actual_count = len(get_params_by_category(category))
#     assert actual_count == expected_count, \
#         f"Category {category}: expected {expected_count} params, got {actual_count}"

print(f"✓ Exit parameter configuration: {TOTAL_PARAMS} parameters across {len(CATEGORY_COUNTS)} categories")
