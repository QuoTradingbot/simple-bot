"""
Configuration Management for VWAP Bounce Bot
Supports multiple environments with validation and environment variable overrides.
"""

import os
from typing import Dict, Any, Optional
from datetime import time
from dataclasses import dataclass, field
import pytz


@dataclass
class BotConfiguration:
    """Type-safe configuration for the VWAP Bounce Bot."""
    
    # Instrument Configuration
    instrument: str = "ES"
    timezone: str = "America/New_York"
    
    # Trading Parameters
    risk_per_trade: float = 0.012  # 1.2% of account per trade (increased for more profit)
    max_contracts: int = 3
    max_trades_per_day: int = 9  # ITERATION 3 - proven optimal with Brain 2
    risk_reward_ratio: float = 2.0  # Realistic 2:1 for mean reversion with tight stops
    
    # Slippage & Commission - PRODUCTION READY
    slippage_ticks: float = 1.5  # Average 1-2 ticks per fill (conservative estimate)
    commission_per_contract: float = 2.50  # Round-turn commission (adjust to your broker)
        # Total cost per round-trip: ~3 ticks slippage + $2.50 commission = ~$42.50/contract
    
    # VWAP bands (standard deviation multipliers) - ITERATION 3 (PROVEN WINNER!)
    vwap_std_dev_1: float = 2.5  # Warning zone (potential reversal area)
    vwap_std_dev_2: float = 2.1  # Entry zone - Iteration 3
    vwap_std_dev_3: float = 3.7  # Exit/stop zone - Iteration 3
    
    # Trend Filter Parameters
    trend_ema_period: int = 21  # Optimizer best
    trend_threshold: float = 0.0001
    
    # Technical Filters - ITERATION 3
    use_trend_filter: bool = False  # Trend filter OFF (optimizer found better without)
    use_rsi_filter: bool = True
    use_vwap_direction_filter: bool = True  # VWAP direction filter ON (optimizer confirmed)
    use_volume_filter: bool = False  # Don't use volume filter - blocks overnight trades
    use_macd_filter: bool = False
    
    # RSI Settings - ITERATION 3 (Conservative, Selective)
    rsi_period: int = 10  # Iteration 3
    rsi_oversold: int = 35  # Iteration 3 - selective entry
    rsi_overbought: int = 65  # Iteration 3 - selective entry
    
    # MACD - Keep for reference but disabled
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Volume Filter - DISABLED (futures have inconsistent volume)
    volume_spike_multiplier: float = 1.5
    volume_lookback: int = 20
    
    # Time Windows (all in Eastern Time)
    market_open_time: time = field(default_factory=lambda: time(9, 30))
    entry_start_time: time = field(default_factory=lambda: time(18, 0))  # 6 PM ET - ES futures session opens
    entry_end_time: time = field(default_factory=lambda: time(16, 55))  # 4:55 PM ET next day - before maintenance
    flatten_time: time = field(default_factory=lambda: time(16, 30))  # 4:30 PM ET - flatten before session close
    forced_flatten_time: time = field(default_factory=lambda: time(16, 45))  # 4:45 PM ET - forced flatten
    shutdown_time: time = field(default_factory=lambda: time(16, 50))  # 4:50 PM ET - shutdown before maintenance
    vwap_reset_time: time = field(default_factory=lambda: time(18, 0))  # 6 PM ET - futures daily session reset
    
    # Friday Special Rules - Close before weekend
    friday_entry_cutoff: time = field(default_factory=lambda: time(16, 0))  # Stop entries 4:00 PM Friday
    friday_close_target: time = field(default_factory=lambda: time(16, 30))  # Flatten by 4:30 PM Friday
    
    # Safety Parameters
    daily_loss_limit: float = 200.0
    max_drawdown_percent: float = 5.0
    tick_timeout_seconds: int = 60
    proactive_stop_buffer_ticks: int = 2
    
    # ATR-Based Dynamic Risk Management - ITERATION 3 (PROVEN WINNER!)
    use_atr_stops: bool = True  # ATR stops enabled
    atr_period: int = 14  # ATR calculation period
    stop_loss_atr_multiplier: float = 3.6  # Iteration 3 (tight stops)
    profit_target_atr_multiplier: float = 4.75  # Iteration 3 (solid targets)
    
    # Instrument Specifications
    tick_size: float = 0.25
    tick_value: float = 12.50  # ES full contract: $12.50 per tick
    
    # Operational Parameters
    dry_run: bool = False
    log_file: str = "vwap_bounce_bot.log"
    max_bars_storage: int = 200
    
    # Bid/Ask Trading Strategy Parameters
    passive_order_timeout: int = 10  # Seconds to wait for passive order fill
    abnormal_spread_multiplier: float = 2.0  # Multiplier for abnormal spread detection
    spread_lookback_periods: int = 100  # Number of spread samples to track
    high_volatility_spread_mult: float = 3.0  # Spread multiplier for high volatility detection
    calm_market_spread_mult: float = 1.5  # Spread multiplier for calm market detection
    use_mixed_order_strategy: bool = False  # Enable mixed passive/aggressive orders
    mixed_passive_ratio: float = 0.5  # Ratio of passive to total when using mixed strategy
    
    # Enhanced Bid/Ask Parameters (Requirements 5-8)
    max_queue_size: int = 100  # Cancel passive order if queue too large
    queue_jump_threshold: int = 50  # Jump queue if position > threshold
    min_bid_ask_size: int = 1  # Minimum liquidity requirement
    max_acceptable_spread: Optional[float] = None  # Maximum spread threshold (None = no limit)
    normal_hours_slippage_ticks: float = 1.0  # Expected slippage during normal hours
    illiquid_hours_slippage_ticks: float = 2.0  # Expected slippage during illiquid hours
    max_slippage_ticks: float = 3.0  # Maximum acceptable slippage
    illiquid_hours_start: time = field(default_factory=lambda: time(0, 0))  # Start of illiquid period
    illiquid_hours_end: time = field(default_factory=lambda: time(9, 30))  # End of illiquid period
    
    # Advanced Bid/Ask Parameters (Requirements 9-15)
    tight_spread_multiplier: float = 1.2  # Threshold for normal/tight spread
    wide_spread_multiplier: float = 2.0  # Threshold for volatile/wide spread
    extreme_spread_multiplier: float = 3.0  # Threshold for stressed market
    low_volume_threshold: float = 0.5  # Threshold for low volume detection
    min_fill_probability: float = 0.5  # Minimum fill probability for passive orders
    max_transaction_cost_pct: float = 0.15  # Max transaction cost as % of expected profit (15%)
    commission_per_contract: float = 2.50  # Commission per contract round-turn
    
    # Advanced Exit Management Parameters
    # ADAPTIVE EXIT MANAGEMENT - ENABLED for intelligent exit management
    adaptive_exits_enabled: bool = True  # ENABLED - Using adaptive intelligent exits
    adaptive_volatility_scaling: bool = True  # Scale parameters based on ATR
    adaptive_regime_detection: bool = True  # Adjust for trending vs choppy markets
    adaptive_performance_based: bool = True  # Adapt based on trade performance
    
    # Breakeven Protection - ITERATION 3 (PROVEN WINNER!)
    breakeven_enabled: bool = True  # ENABLED - Adaptive system will adjust dynamically
    breakeven_profit_threshold_ticks: int = 9  # Iteration 3 - proven optimal
    breakeven_stop_offset_ticks: int = 1  # Baseline (adaptive adjusts)
    
    # Trailing Stop - ITERATION 3 (PROVEN WINNER!)
    trailing_stop_enabled: bool = True  # ENABLED - Adaptive system will adjust
    trailing_stop_distance_ticks: int = 10  # Iteration 3 - proven optimal
    trailing_stop_min_profit_ticks: int = 16  # Iteration 3 - proven optimal
    
    # Time-Decay Tightening
    time_decay_enabled: bool = True  # ENABLED - Tightens stops over time
    time_decay_50_percent_tightening: float = 0.10  # 10% tighter at 50% of max hold time
    time_decay_75_percent_tightening: float = 0.20  # 20% tighter at 75%
    time_decay_90_percent_tightening: float = 0.30  # 30% tighter at 90%
    
    # Partial Exits (baseline - adaptive system adjusts R-multiples)
    partial_exits_enabled: bool = True  # ENABLED - Scale out at targets
    partial_exit_1_percentage: float = 0.50  # 50% exit at first level
    partial_exit_1_r_multiple: float = 2.0  # Exit at 2.0R (adaptive: 1.4-3.0)
    partial_exit_2_percentage: float = 0.30  # 30% exit at second level
    partial_exit_2_r_multiple: float = 3.0  # Exit at 3.0R (adaptive: 2.1-4.5)
    partial_exit_3_percentage: float = 0.20  # 20% exit at third level
    partial_exit_3_r_multiple: float = 5.0  # Exit at 5.0R
    
    # Reinforcement Learning Parameters
    # RL ENABLED - Learning which signals to trust from experience
    rl_enabled: bool = True  # ENABLED - RL layer learns signal quality
    rl_exploration_rate: float = 0.30  # 30% exploration (random decisions)
    rl_min_exploration_rate: float = 0.05  # Minimum exploration after decay
    rl_exploration_decay: float = 0.995  # Decay rate per signal
    rl_confidence_threshold: float = 0.5  # Minimum confidence to take signal
    rl_min_contracts: int = 1  # Minimum contracts (low confidence)
    rl_medium_contracts: int = 2  # Medium contracts (moderate confidence)
    rl_max_contracts: int = 3  # Maximum contracts (high confidence)
    rl_experience_file: str = "signal_experience.json"  # Where to save learning
    rl_save_frequency: int = 5  # Save experiences every N trades
    
    # Broker Configuration (only for live trading)
    api_token: Optional[str] = None
    
    # Operational mode
    backtest_mode: bool = False  # When True, runs in backtest mode without broker
    
    # Environment
    environment: str = "production"  # "development", "staging", "production"
    
    def validate(self) -> None:
        """Validate configuration values."""
        errors = []
        
        # Validate risk parameters
        if not 0 < self.risk_per_trade <= 1:
            errors.append(f"risk_per_trade must be between 0 and 1, got {self.risk_per_trade}")
        
        if self.max_contracts <= 0:
            errors.append(f"max_contracts must be positive, got {self.max_contracts}")
        
        if self.max_trades_per_day <= 0:
            errors.append(f"max_trades_per_day must be positive, got {self.max_trades_per_day}")
        
        if self.risk_reward_ratio <= 0:
            errors.append(f"risk_reward_ratio must be positive, got {self.risk_reward_ratio}")
        
        # Validate time windows (ES futures: 6 PM to 5 PM next day, wraps midnight)
        # For 24-hour trading, entry_start_time (6 PM) > entry_end_time (4:55 PM) is VALID
        # Skip the old entry_start_time >= entry_end_time check since futures wrap midnight
        
        # Flatten times should still be in order on same day
        if self.flatten_time >= self.forced_flatten_time:
            errors.append(f"flatten_time must be before forced_flatten_time")
        
        # Flatten times should still be in order on same day
        if self.flatten_time >= self.forced_flatten_time:
            errors.append(f"flatten_time must be before forced_flatten_time")
        
        if self.forced_flatten_time >= self.shutdown_time:
            errors.append(f"forced_flatten_time must be before shutdown_time")
        
        # Validate timezone
        try:
            pytz.timezone(self.timezone)
        except pytz.exceptions.UnknownTimeZoneError:
            errors.append(f"Invalid timezone: {self.timezone}")
        
        # Validate tick specifications
        if self.tick_size <= 0:
            errors.append(f"tick_size must be positive, got {self.tick_size}")
        
        if self.tick_value <= 0:
            errors.append(f"tick_value must be positive, got {self.tick_value}")
        
        # Validate safety parameters
        if self.daily_loss_limit <= 0:
            errors.append(f"daily_loss_limit must be positive, got {self.daily_loss_limit}")
        
        if not 0 < self.max_drawdown_percent <= 100:
            errors.append(f"max_drawdown_percent must be between 0 and 100, got {self.max_drawdown_percent}")
        
        # Validate broker configuration - API token is required unless in backtest mode
        if not self.backtest_mode and not self.api_token:
            errors.append("api_token is required for TopStep broker (not required in backtest_mode)")
        
        # Validate environment
        if self.environment not in ["development", "staging", "production"]:
            errors.append(f"environment must be 'development', 'staging', or 'production', got {self.environment}")
        
        if errors:
            raise ValueError("Configuration validation failed:\n  - " + "\n  - ".join(errors))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (legacy format)."""
        return {
            "instrument": self.instrument,
            "timezone": self.timezone,
            "risk_per_trade": self.risk_per_trade,
            "max_contracts": self.max_contracts,
            "max_trades_per_day": self.max_trades_per_day,
            "risk_reward_ratio": self.risk_reward_ratio,
            "slippage_ticks": self.slippage_ticks,
            "commission_per_contract": self.commission_per_contract,
            "vwap_std_dev_1": self.vwap_std_dev_1,
            "vwap_std_dev_2": self.vwap_std_dev_2,
            "vwap_std_dev_3": self.vwap_std_dev_3,
            "trend_ema_period": self.trend_ema_period,
            "trend_threshold": self.trend_threshold,
            "use_trend_filter": self.use_trend_filter,
            "use_rsi_filter": self.use_rsi_filter,
            "use_macd_filter": self.use_macd_filter,
            "use_vwap_direction_filter": self.use_vwap_direction_filter,
            "use_volume_filter": self.use_volume_filter,
            "rsi_period": self.rsi_period,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "volume_spike_multiplier": self.volume_spike_multiplier,
            "volume_lookback": self.volume_lookback,
            "market_open_time": self.market_open_time,
            "entry_start_time": self.entry_start_time,
            "entry_end_time": self.entry_end_time,
            "flatten_time": self.flatten_time,
            "forced_flatten_time": self.forced_flatten_time,
            "shutdown_time": self.shutdown_time,
            "vwap_reset_time": self.vwap_reset_time,
            "friday_entry_cutoff": self.friday_entry_cutoff,
            "friday_close_target": self.friday_close_target,
            "daily_loss_limit": self.daily_loss_limit,
            "max_drawdown_percent": self.max_drawdown_percent,
            "tick_timeout_seconds": self.tick_timeout_seconds,
            "proactive_stop_buffer_ticks": self.proactive_stop_buffer_ticks,
            "tick_size": self.tick_size,
            "tick_value": self.tick_value,
            "dry_run": self.dry_run,
            "log_file": self.log_file,
            "max_bars_storage": self.max_bars_storage,
            # Adaptive Exit Management
            "adaptive_exits_enabled": self.adaptive_exits_enabled,
            "adaptive_volatility_scaling": self.adaptive_volatility_scaling,
            "adaptive_regime_detection": self.adaptive_regime_detection,
            "adaptive_performance_based": self.adaptive_performance_based,
            # Advanced Exit Management (baseline parameters)
            "breakeven_enabled": self.breakeven_enabled,
            "breakeven_profit_threshold_ticks": self.breakeven_profit_threshold_ticks,
            "breakeven_stop_offset_ticks": self.breakeven_stop_offset_ticks,
            "trailing_stop_enabled": self.trailing_stop_enabled,
            "trailing_stop_distance_ticks": self.trailing_stop_distance_ticks,
            "trailing_stop_min_profit_ticks": self.trailing_stop_min_profit_ticks,
            "time_decay_enabled": self.time_decay_enabled,
            "time_decay_50_percent_tightening": self.time_decay_50_percent_tightening,
            "time_decay_75_percent_tightening": self.time_decay_75_percent_tightening,
            "time_decay_90_percent_tightening": self.time_decay_90_percent_tightening,
            "partial_exits_enabled": self.partial_exits_enabled,
            "partial_exit_1_percentage": self.partial_exit_1_percentage,
            "partial_exit_1_r_multiple": self.partial_exit_1_r_multiple,
            "partial_exit_2_percentage": self.partial_exit_2_percentage,
            "partial_exit_2_r_multiple": self.partial_exit_2_r_multiple,
            "partial_exit_3_percentage": self.partial_exit_3_percentage,
            "partial_exit_3_r_multiple": self.partial_exit_3_r_multiple,
        }


def load_from_env() -> BotConfiguration:
    """
    Load configuration from environment variables.
    Environment variables override default values.
    """
    config = BotConfiguration()
    
    # Load from environment variables with BOT_ prefix
    if os.getenv("BOT_INSTRUMENT"):
        config.instrument = os.getenv("BOT_INSTRUMENT")
    
    if os.getenv("BOT_TIMEZONE"):
        config.timezone = os.getenv("BOT_TIMEZONE")
    
    if os.getenv("BOT_RISK_PER_TRADE"):
        config.risk_per_trade = float(os.getenv("BOT_RISK_PER_TRADE"))
    
    if os.getenv("BOT_MAX_CONTRACTS"):
        config.max_contracts = int(os.getenv("BOT_MAX_CONTRACTS"))
    
    if os.getenv("BOT_MAX_TRADES_PER_DAY"):
        config.max_trades_per_day = int(os.getenv("BOT_MAX_TRADES_PER_DAY"))
    
    if os.getenv("BOT_RISK_REWARD_RATIO"):
        config.risk_reward_ratio = float(os.getenv("BOT_RISK_REWARD_RATIO"))
    
    if os.getenv("BOT_DAILY_LOSS_LIMIT"):
        config.daily_loss_limit = float(os.getenv("BOT_DAILY_LOSS_LIMIT"))
    
    if os.getenv("BOT_MAX_DRAWDOWN_PERCENT"):
        config.max_drawdown_percent = float(os.getenv("BOT_MAX_DRAWDOWN_PERCENT"))
    
    if os.getenv("BOT_TICK_SIZE"):
        config.tick_size = float(os.getenv("BOT_TICK_SIZE"))
    
    if os.getenv("BOT_TICK_VALUE"):
        config.tick_value = float(os.getenv("BOT_TICK_VALUE"))
    
    if os.getenv("BOT_DRY_RUN"):
        config.dry_run = os.getenv("BOT_DRY_RUN").lower() in ("true", "1", "yes")
    
    if os.getenv("BOT_ENVIRONMENT"):
        config.environment = os.getenv("BOT_ENVIRONMENT")
    
    # API Token (without logging)
    if os.getenv("TOPSTEP_API_TOKEN"):
        config.api_token = os.getenv("TOPSTEP_API_TOKEN")
    
    return config


def get_development_config() -> BotConfiguration:
    """Development environment configuration."""
    config = BotConfiguration()
    config.environment = "development"
    config.dry_run = True
    # API token must be set via environment variable
    return config


def get_staging_config() -> BotConfiguration:
    """Staging environment configuration."""
    config = BotConfiguration()
    config.environment = "staging"
    config.dry_run = True
    config.max_trades_per_day = 5
    # API token must be set via environment variable
    return config


def get_production_config() -> BotConfiguration:
    """Production environment configuration."""
    config = BotConfiguration()
    config.environment = "production"
    config.dry_run = False
    # API token must be set via environment variable
    return config


def load_config(environment: Optional[str] = None, backtest_mode: bool = False) -> BotConfiguration:
    """
    Load configuration based on environment.
    
    Priority:
    1. Environment variables (highest)
    2. Environment-specific config
    3. Default config (lowest)
    
    Args:
        environment: Environment name ("development", "staging", "production")
                    If None, uses BOT_ENVIRONMENT env var or defaults to "development"
        backtest_mode: If True, API token is not required (for backtesting)
    
    Returns:
        Validated BotConfiguration instance
    
    Raises:
        ValueError: If configuration validation fails
    """
    # Determine environment
    if environment is None:
        environment = os.getenv("BOT_ENVIRONMENT", "development")
    
    # Load base config for environment
    if environment == "production":
        config = get_production_config()
    elif environment == "staging":
        config = get_staging_config()
    else:
        config = get_development_config()
    
    # Set backtest mode
    config.backtest_mode = backtest_mode
    
    # Override with environment variables
    env_config = load_from_env()
    
    # Merge configurations (env vars take precedence)
    for key in config.__dataclass_fields__.keys():
        env_value = getattr(env_config, key)
        default_value = getattr(BotConfiguration(), key)
        
        # If env value differs from default, use it
        if env_value != default_value:
            setattr(config, key, env_value)
    
    # Validate configuration
    config.validate()
    
    return config


def log_config(config: BotConfiguration, logger) -> None:
    """
    Log configuration safely (without exposing secrets).
    
    Args:
        config: Configuration to log
        logger: Logger instance
    """
    logger.info("=" * 60)
    logger.info("CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Environment: {config.environment}")
    
    # Only show broker info in live mode
    if not config.backtest_mode:
        logger.info(f"Broker: TopStep")
        logger.info(f"Dry Run: {config.dry_run}")
    
    logger.info(f"Instrument: {config.instrument}")
    logger.info(f"Timezone: {config.timezone}")
    logger.info(f"Risk per Trade: {config.risk_per_trade * 100:.1f}%")
    logger.info(f"Max Contracts: {config.max_contracts}")
    logger.info(f"Max Trades per Day: {config.max_trades_per_day}")
    logger.info(f"Risk/Reward Ratio: {config.risk_reward_ratio}:1")
    logger.info(f"Daily Loss Limit: ${config.daily_loss_limit:.2f}")
    logger.info(f"Max Drawdown: {config.max_drawdown_percent:.1f}%")
    
    # Time windows
    logger.info(f"Entry Window: {config.entry_start_time} - {config.entry_end_time} ET")
    logger.info(f"Flatten Time: {config.flatten_time} ET")
    logger.info(f"Forced Flatten: {config.forced_flatten_time} ET")
    
    # API token info (only relevant for live trading)
    if config.backtest_mode:
        logger.info("Mode: Backtest (no API/broker connection needed)")
    elif config.api_token:
        logger.info("API Token: *** (configured for live trading)")
    else:
        logger.info("API Token: (not configured)")
    
    logger.info("=" * 60)


def apply_learned_parameters(config: BotConfiguration, learning_file: str = "learning_history.json") -> bool:
    """
    Load and apply the best parameters learned from continuous learning.
    
    Args:
        config: Bot configuration to update
        learning_file: Path to learning history JSON file
        
    Returns:
        True if parameters were loaded and applied, False otherwise
    """
    import json
    import os
    from pathlib import Path
    import logging
    
    log = logging.getLogger(__name__)
    
    learning_path = Path(learning_file)
    
    if not learning_path.exists():
        log.warning(f"No learning history found at {learning_file}")
        log.info("Using default configuration parameters")
        return False
    
    try:
        with open(learning_path, 'r') as f:
            learning_data = json.load(f)
        
        best_params = learning_data.get('best_params', {})
        best_score = learning_data.get('best_score', 0)
        
        if not best_params:
            log.warning("No best parameters found in learning history")
            return False
        
        # Apply learned parameters
        log.info("=" * 60)
        log.info(" APPLYING LEARNED PARAMETERS FROM CONTINUOUS LEARNING")
        log.info(f"Best Score: {best_score:,.0f}")
        log.info("-" * 60)
        
        param_mapping = {
            'vwap_std_dev_1': 'vwap_std_dev_1',
            'vwap_std_dev_3': 'vwap_std_dev_3',
            'rsi_period': 'rsi_period',
            'rsi_oversold': 'rsi_oversold',
            'rsi_overbought': 'rsi_overbought',
            'stop_loss_atr_multiplier': 'stop_loss_atr_multiplier',
            'profit_target_atr_multiplier': 'profit_target_atr_multiplier',
        }
        
        for learned_key, config_key in param_mapping.items():
            if learned_key in best_params:
                old_value = getattr(config, config_key)
                new_value = best_params[learned_key]
                setattr(config, config_key, new_value)
                log.info(f"  {config_key}: {old_value} → {new_value}")
        
        log.info("=" * 60)
        log.info(" Learned parameters applied successfully!")
        log.info("=" * 60)
        
        return True
        
    except Exception as e:
        log.error(f"Failed to load learned parameters: {e}")
        log.info("Using default configuration parameters")
        return False
