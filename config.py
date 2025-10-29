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
    instrument: str = "MES"
    timezone: str = "America/New_York"
    
    # Trading Parameters
    risk_per_trade: float = 0.01  # 1% of account per trade
    max_contracts: int = 2
    max_trades_per_day: int = 3
    risk_reward_ratio: float = 1.5
    
    # VWAP Parameters
    vwap_std_dev_1: float = 1.0
    vwap_std_dev_2: float = 2.0
    
    # Trend Filter Parameters
    trend_ema_period: int = 20
    trend_threshold: float = 0.0001
    
    # Time Windows (all in Eastern Time)
    market_open_time: time = field(default_factory=lambda: time(9, 30))
    entry_start_time: time = field(default_factory=lambda: time(9, 0))
    entry_end_time: time = field(default_factory=lambda: time(14, 30))
    flatten_time: time = field(default_factory=lambda: time(16, 30))
    forced_flatten_time: time = field(default_factory=lambda: time(16, 45))
    shutdown_time: time = field(default_factory=lambda: time(17, 0))
    vwap_reset_time: time = field(default_factory=lambda: time(9, 30))
    
    # Friday Special Rules
    friday_entry_cutoff: time = field(default_factory=lambda: time(13, 0))
    friday_close_target: time = field(default_factory=lambda: time(15, 0))
    
    # Safety Parameters
    daily_loss_limit: float = 200.0
    max_drawdown_percent: float = 5.0
    tick_timeout_seconds: int = 60
    proactive_stop_buffer_ticks: int = 2
    
    # Instrument Specifications
    tick_size: float = 0.25
    tick_value: float = 1.25
    
    # Operational Parameters
    dry_run: bool = False
    log_file: str = "vwap_bounce_bot.log"
    max_bars_storage: int = 200
    
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
            "vwap_std_dev_1": self.vwap_std_dev_1,
            "vwap_std_dev_2": self.vwap_std_dev_2,
            "trend_ema_period": self.trend_ema_period,
            "trend_threshold": self.trend_threshold,
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
