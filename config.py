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
    risk_per_trade: float = 0.01  # 1% of account per trade
    max_contracts: int = 2
    max_trades_per_day: int = 3  # Limit to best quality setups
    risk_reward_ratio: float = 2.0  # Realistic 2:1 for mean reversion with tight stops
    
    # Slippage & Commission - PRODUCTION READY
    slippage_ticks: float = 1.5  # Average 1-2 ticks per fill (conservative estimate)
    commission_per_contract: float = 2.50  # Round-turn commission (adjust to your broker)
    # Total cost per round-trip: ~3 ticks slippage + $2.50 commission = ~$42.50/contract
    
    # VWAP Parameters - BALANCED FOR REAL TRADING
    vwap_std_dev_1: float = 1.5  # Warning zone
    vwap_std_dev_2: float = 2.0  # Entry zone (BALANCED - not too extreme)
    vwap_std_dev_3: float = 3.0  # Stop zone
    
    # Trend Filter Parameters
    trend_ema_period: int = 20
    trend_threshold: float = 0.0001
    
    # Technical Indicator Parameters - BACK TO PROVEN SETUP
    use_trend_filter: bool = False  # ❌ OFF - conflicts with mean reversion
    use_rsi_filter: bool = True  # ✅ RSI extremes (28/72)
    use_macd_filter: bool = False  # ❌ OFF - lags reversals
    use_vwap_direction_filter: bool = True  # ✅ Price moving toward VWAP
    use_volume_filter: bool = False  # ❌ OFF - blocks overnight trades
    
        # RSI Settings - Balanced for quality
    rsi_period: int = 14
    rsi_oversold: int = 28  # Slightly tighter than 30 for better extremes
    rsi_overbought: int = 72  # Slightly tighter than 70
    
    # MACD - Keep for reference but disabled
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Volume Filter - DISABLED (futures have inconsistent volume)
    volume_spike_multiplier: float = 1.5
    volume_lookback: int = 20
    
    # Time Windows (all in Eastern Time)
    market_open_time: time = field(default_factory=lambda: time(9, 30))
    entry_start_time: time = field(default_factory=lambda: time(0, 0))  # 24/5 - trades any time
    entry_end_time: time = field(default_factory=lambda: time(23, 50))  # Accept entries nearly all day
    flatten_time: time = field(default_factory=lambda: time(23, 55))  # Only used for Friday close
    forced_flatten_time: time = field(default_factory=lambda: time(23, 57))  # Only used for Friday close
    shutdown_time: time = field(default_factory=lambda: time(23, 59))  # Daily reset at midnight
    vwap_reset_time: time = field(default_factory=lambda: time(18, 0))  # 6 PM ET - futures daily session reset
    
    # Friday Special Rules - Close before weekend
    friday_entry_cutoff: time = field(default_factory=lambda: time(16, 45))  # Stop entries 4:45 PM Friday
    friday_close_target: time = field(default_factory=lambda: time(16, 30))  # Flatten by 4:30 PM Friday
    
    # Safety Parameters
    daily_loss_limit: float = 200.0
    max_drawdown_percent: float = 5.0
    tick_timeout_seconds: int = 60
    proactive_stop_buffer_ticks: int = 2
    
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
        
        # Validate time windows
        if self.entry_start_time >= self.entry_end_time:
            errors.append(f"entry_start_time must be before entry_end_time")
        
        if self.entry_end_time >= self.flatten_time:
            errors.append(f"entry_end_time must be before flatten_time")
        
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
