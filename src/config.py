"""
Configuration Management for VWAP Bounce Bot
Supports multiple environments with validation and environment variable overrides.
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import time
from dataclasses import dataclass, field
import pytz


@dataclass
class BotConfiguration:
    """Type-safe configuration for the VWAP Bounce Bot."""
    
    # Instrument Configuration
    instrument: str = "ES"  # Single instrument (legacy support)
    instruments: list = field(default_factory=lambda: ["ES"])  # Multi-symbol support
    timezone: str = "America/New_York"
    
    # Broker Configuration
    api_token: str = ""
    username: str = ""  # TopStep username/email
    
    # Trading Parameters
    risk_per_trade: float = 0.012  # 1.2% of account per trade (increased for more profit)
    max_contracts: int = 3  # USER CONFIGURABLE - maximum contracts allowed (user sets their own limit)
    max_trades_per_day: int = 9  # USER CONFIGURABLE - customers can adjust
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
    
    # Testing/Debug Parameters
    force_test_trade: bool = True  # Set to True to force a test trade on next bar (ignores all filters)
    
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
    shutdown_time: time = field(default_factory=lambda: time(16, 50))  # 4:50 PM ET - shutdown for 5-6 PM maintenance
    vwap_reset_time: time = field(default_factory=lambda: time(18, 0))  # 6 PM ET - futures daily session reset
    
    # Friday Special Rules - Close before weekend
    friday_entry_cutoff: time = field(default_factory=lambda: time(16, 0))  # Stop entries 4:00 PM Friday
    friday_close_target: time = field(default_factory=lambda: time(16, 30))  # Flatten by 4:30 PM Friday
    
    # Safety Parameters
    daily_loss_limit: float = 1000.0  # Default - will be calculated from account size
    max_drawdown_percent: float = 4.0  # Default: 4% (standard safe limit)
    daily_loss_percent: float = 2.0  # Default: 2% (standard safe limit)
    auto_calculate_limits: bool = True  # If True, bot auto-calculates limits from account balance
    tick_timeout_seconds: int = 999999  # Disabled for testing
    proactive_stop_buffer_ticks: int = 2
    flatten_buffer_ticks: int = 2  # Buffer for flatten price calculation
    
    def get_daily_loss_limit(self, account_balance: float) -> float:
        """
        Calculate dynamic daily loss limit based on account rules.
        Works for TopStep accounts OR personal accounts with custom risk %.
        
        TopStep Rules (as of 2025):
        - All accounts: 2% of starting balance
        
        Personal Account:
        - Uses daily_loss_percent setting (default: 2%, but customizable)
        
        Args:
            account_balance: Current account balance
            
        Returns:
            Daily loss limit in dollars
        """
        # Use custom percentage or TopStep standard (2%)
        loss_percent = self.daily_loss_percent if hasattr(self, 'daily_loss_percent') else 2.0
        return account_balance * (loss_percent / 100.0)
    
    def get_max_drawdown_dollars(self, account_balance: float) -> float:
        """
        Calculate max trailing drawdown in dollars based on TopStep rules.
        Works for ALL TopStep account sizes.
        
        TopStep Trailing Drawdown Rules:
        - Calculated from HIGHEST equity reached (not starting balance)
        - Express ($25K): 4% = $1,000 max drawdown from peak
        - Step 1 ($50K): 4% = $2,000 max drawdown from peak
        - Step 2 ($100K): 4% = $4,000 max drawdown from peak
        - Step 2 ($150K): 4% = $6,000 max drawdown from peak
        - Funded (All sizes): 4% max drawdown from peak
        
        Example: Start at $50K, grow to $55K → Max drawdown = $55K - ($55K × 0.04) = $52,800
        
        Args:
            account_balance: Current/highest account balance reached
            
        Returns:
            Max trailing drawdown in dollars (4% of highest balance)
        """
        return account_balance * (self.max_drawdown_percent / 100.0)
    
    def get_profit_target(self, account_balance: float) -> float:
        """
        Calculate profit target based on TopStep rules.
        Users must hit profit target to advance to next step or get funded.
        
        TopStep Profit Targets:
        - Express ($25K): $1,500 (6%)
        - Step 1 ($50K): $3,000 (6%)
        - Step 2 ($100K): $6,000 (6%)
        - Step 2 ($150K): $9,000 (6%)
        - Funded: No target, keep profits!
        
        Args:
            account_balance: Starting account balance
            
        Returns:
            Profit target in dollars (6% for evaluation accounts)
        """
        # 6% profit target for evaluation accounts
        return account_balance * 0.06
    
    def get_account_type(self, account_balance: float) -> str:
        """
        Determine TopStep account type based on balance.
        
        Args:
            account_balance: Current account balance
            
        Returns:
            Account type string
        """
        if account_balance <= 25000:
            return "Express ($25K)"
        elif account_balance <= 50000:
            return "Step 1 ($50K) or Funded ($50K)"
        elif account_balance <= 100000:
            return "Step 2 ($100K) or Funded ($100K)"
        elif account_balance <= 150000:
            return "Step 2 ($150K) or Funded ($150K)"
        elif account_balance <= 250000:
            return "Funded ($250K)"
        else:
            return f"Funded (${account_balance/1000:.0f}K)"
    
    def auto_configure_for_account(self, account_balance: float, logger=None, force: bool = False) -> bool:
        """
        Automatically configure all risk limits based on account balance.
        Called when bot connects or when balance changes significantly.
        
        Supports BOTH TopStep accounts and personal trading accounts:
        - TopStep: Enforces 2% daily loss, 4% max DD (mandatory rules)
        - Personal: Uses custom daily_loss_percent and max_drawdown_percent
        
        NOTE: max_contracts and max_trades_per_day are USER CONFIGURABLE and NOT changed here.
        
        Args:
            account_balance: Current account balance from broker
            logger: Optional logger for info messages
            force: Force reconfiguration even if balance seems invalid
            
        Returns:
            True if configuration was successful, False if balance invalid
        """
        # Safety check: Validate balance is reasonable
        if account_balance <= 0 and not force:
            if logger:
                logger.error(f"[WARNING] Invalid account balance: ${account_balance:,.2f}")
                logger.error("Skipping auto-configuration to prevent misconfiguration")
            return False
        
        # Safety check: Prevent extreme values (likely API error)
        if account_balance > 10_000_000 and not force:  # $10M+ seems wrong
            if logger:
                logger.warning(f"[WARNING] Unusually high balance: ${account_balance:,.2f}")
                logger.warning("This may be an API error - skipping reconfiguration")
            return False
        
        # Calculate dynamic limits based on account type
        self.daily_loss_limit = self.get_daily_loss_limit(account_balance)
        max_dd_dollars = self.get_max_drawdown_dollars(account_balance)
        profit_target = self.get_profit_target(account_balance)
        account_type = self.get_account_type(account_balance)
        
        # Auto-calculate limits based on account balance (standard 2%/4% rules)
        if self.auto_calculate_limits:
            self.max_drawdown_percent = 4.0  # Standard safe limit
            account_label = "AUTO-CALCULATED LIMITS"
        else:
            # Keep user's custom setting (already set in config)
            account_label = "MANUAL LIMITS"
        
        # NOTE: max_contracts and max_trades_per_day are NOT changed here
        # Those are user preferences that customers can configure themselves
        
        if logger:
            logger.info("=" * 80)
            logger.info(f"[AUTO-CONFIG] Configured for {account_label}")
            logger.info("=" * 80)
            logger.info(f"Account Type: {account_type}")
            logger.info(f"Account Balance: ${account_balance:,.2f}")
            logger.info("")
            if self.auto_calculate_limits:
                logger.info("Auto-Calculated Limits (Based on Account Balance):")
            else:
                logger.info(f"Manual Limits (Custom {self.daily_loss_percent}%/{self.max_drawdown_percent}%):")
            logger.info(f"  Daily Loss Limit: ${self.daily_loss_limit:,.2f} ({self.daily_loss_percent}% of balance)")
            logger.info(f"  Max Trailing Drawdown: ${max_dd_dollars:,.2f} ({self.max_drawdown_percent}% from peak)")
            if self.auto_calculate_limits:
                logger.info(f"  Profit Target: ${profit_target:,.2f} (6% target)")
            logger.info("")
            logger.info("User Configurable Settings (Not Changed):")
            logger.info(f"  Max Contracts: {self.max_contracts}")
            logger.info(f"  Max Trades/Day: {self.max_trades_per_day}")
            logger.info("[SUCCESS] Risk limits applied successfully")
        
        return True
    
    def get_current_risk_limits(self) -> Dict[str, Any]:
        """
        Get current risk limits configured for the account.
        Useful for monitoring and validation.
        
        Returns:
            Dictionary with all current risk parameters
        """
        return {
            "daily_loss_limit": self.daily_loss_limit,
            "max_drawdown_percent": self.max_drawdown_percent,
            "max_contracts": self.max_contracts,
            "max_trades_per_day": self.max_trades_per_day,
            "risk_per_trade": self.risk_per_trade,
            "risk_reward_ratio": self.risk_reward_ratio,
        }
    
    def validate_topstep_compliance(self, account_balance: float, logger=None) -> bool:
        """
        Validate that current risk limits comply with TopStep rules.
        
        Args:
            account_balance: Current account balance
            logger: Optional logger for validation messages
            
        Returns:
            True if compliant, False if violations detected
        """
        violations = []
        
        # Check daily loss limit (must be 2% of balance)
        expected_daily_loss = self.get_daily_loss_limit(account_balance)
        if abs(self.daily_loss_limit - expected_daily_loss) > 1.0:  # $1 tolerance
            violations.append(
                f"Daily loss limit mismatch: Expected ${expected_daily_loss:,.2f}, "
                f"Got ${self.daily_loss_limit:,.2f}"
            )
        
        # Check max drawdown (must be 4%)
        if self.max_drawdown_percent != 4.0:
            violations.append(
                f"Max drawdown must be 4% (TopStep rule), got {self.max_drawdown_percent}%"
            )
        
        if violations:
            if logger:
                logger.error("[WARNING] TOPSTEP COMPLIANCE VIOLATIONS DETECTED:")
                for violation in violations:
                    logger.error(f"  - {violation}")
                logger.error("Auto-reconfiguring to fix violations...")
            return False
        
        if logger:
            logger.info("[SUCCESS] All TopStep compliance rules validated successfully")
        return True
    
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
    shadow_mode: bool = False  # Shadow mode - simulates full trading with live data (no account login, tracks positions/P&L locally)
    log_file: str = "logs/vwap_bounce_bot.log"
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
    rl_experience_file: str = "data/signal_experience.json"  # Where to save learning
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
        warnings = []
        
        # Validate risk parameters
        if not 0 < self.risk_per_trade <= 1:
            errors.append(f"risk_per_trade must be between 0 and 1, got {self.risk_per_trade}")
        
        if self.max_contracts <= 0:
            errors.append(f"max_contracts must be positive, got {self.max_contracts}")
        
        # Validate max_contracts with warnings and hard caps
        if self.max_contracts > 25:
            errors.append(f"max_contracts exceeds safety limit: {self.max_contracts} (maximum allowed: 25)")
        elif self.max_contracts > 15:
            warnings.append(f"max_contracts is high: {self.max_contracts} (recommended max: 15 for most traders)")
        
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
        # Shadow mode needs API token for live data streaming (but no account login)
        if not self.backtest_mode and not self.api_token:
            errors.append("api_token is required for TopStep broker (not required only in backtest_mode)")
        
        # Validate environment
        if self.environment not in ["development", "staging", "production"]:
            errors.append(f"environment must be 'development', 'staging', or 'production', got {self.environment}")
        
        # Log warnings (non-fatal)
        if warnings:
            logger = logging.getLogger(__name__)
            logger.warning("Configuration warnings:")
            for warning in warnings:
                logger.warning(f"  [WARN] {warning}")
        
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
            "flatten_buffer_ticks": self.flatten_buffer_ticks,
            "tick_size": self.tick_size,
            "tick_value": self.tick_value,
            "dry_run": self.dry_run,
            "shadow_mode": self.shadow_mode,
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
    # Multi-symbol support: BOT_INSTRUMENTS takes precedence over BOT_INSTRUMENT
    if os.getenv("BOT_INSTRUMENTS"):
        # Parse comma-separated list of instruments
        instruments_str = os.getenv("BOT_INSTRUMENTS")
        config.instruments = [s.strip() for s in instruments_str.split(",")]
        config.instrument = config.instruments[0]  # First symbol is primary
    elif os.getenv("BOT_INSTRUMENT"):
        # Legacy single instrument support
        config.instrument = os.getenv("BOT_INSTRUMENT")
        config.instruments = [config.instrument]
    
    if os.getenv("BOT_TIMEZONE"):
        config.timezone = os.getenv("BOT_TIMEZONE")
    
    if os.getenv("BOT_RISK_PER_TRADE"):
        config.risk_per_trade = float(os.getenv("BOT_RISK_PER_TRADE"))
    
    if os.getenv("BOT_MAX_CONTRACTS"):
        config.max_contracts = int(os.getenv("BOT_MAX_CONTRACTS"))
    
    if os.getenv("BOT_MAX_TRADES_PER_DAY"):
        config.max_trades_per_day = int(os.getenv("BOT_MAX_TRADES_PER_DAY"))
    
    # BOT_MIN_RISK_REWARD is alias for BOT_RISK_REWARD_RATIO (GUI uses MIN_RISK_REWARD)
    if os.getenv("BOT_MIN_RISK_REWARD"):
        config.risk_reward_ratio = float(os.getenv("BOT_MIN_RISK_REWARD"))
    elif os.getenv("BOT_RISK_REWARD_RATIO"):
        config.risk_reward_ratio = float(os.getenv("BOT_RISK_REWARD_RATIO"))
    
    if os.getenv("BOT_DAILY_LOSS_LIMIT"):
        config.daily_loss_limit = float(os.getenv("BOT_DAILY_LOSS_LIMIT"))
    
    if os.getenv("BOT_MAX_DRAWDOWN_PERCENT"):
        config.max_drawdown_percent = float(os.getenv("BOT_MAX_DRAWDOWN_PERCENT"))
    
    if os.getenv("BOT_DAILY_LOSS_PERCENT"):
        config.daily_loss_percent = float(os.getenv("BOT_DAILY_LOSS_PERCENT"))
    
    # Auto-calculate limits (supports both old and new env var names)
    if os.getenv("BOT_AUTO_CALCULATE_LIMITS"):
        config.auto_calculate_limits = os.getenv("BOT_AUTO_CALCULATE_LIMITS").lower() in ("true", "1", "yes")
    elif os.getenv("BOT_USE_TOPSTEP_RULES"):  # Legacy support
        config.auto_calculate_limits = os.getenv("BOT_USE_TOPSTEP_RULES").lower() in ("true", "1", "yes")
    
    if os.getenv("BOT_TICK_SIZE"):
        config.tick_size = float(os.getenv("BOT_TICK_SIZE"))
    
    if os.getenv("BOT_TICK_VALUE"):
        config.tick_value = float(os.getenv("BOT_TICK_VALUE"))
    
    if os.getenv("BOT_DRY_RUN"):
        config.dry_run = os.getenv("BOT_DRY_RUN").lower() in ("true", "1", "yes")
    
    if os.getenv("BOT_SHADOW_MODE"):
        config.shadow_mode = os.getenv("BOT_SHADOW_MODE").lower() in ("true", "1", "yes")
    
    if os.getenv("BOT_ENVIRONMENT"):
        config.environment = os.getenv("BOT_ENVIRONMENT")
    
    # API Token (support both old TOPSTEP and new TOPSTEPX variable names)
    if os.getenv("TOPSTEPX_API_TOKEN"):
        config.api_token = os.getenv("TOPSTEPX_API_TOKEN")
    elif os.getenv("TOPSTEP_API_TOKEN"):
        config.api_token = os.getenv("TOPSTEP_API_TOKEN")
    
    # Username (support both old TOPSTEP and new TOPSTEPX variable names)
    if os.getenv("TOPSTEPX_USERNAME"):
        config.username = os.getenv("TOPSTEPX_USERNAME")
    elif os.getenv("TOPSTEP_USERNAME"):
        config.username = os.getenv("TOPSTEP_USERNAME")
    
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


# ========================================
# EXECUTION RISK CONTROLS (Production-Ready)
# ========================================
# These parameters protect against real-world execution failures:
# - Price deterioration (signal price vs current bid/ask)
# - Partial fills (1 of 3 contracts filled)
# - Order rejections (broker/exchange failures)
# - Fast markets (spread widening, volatility spikes)

# Fix #1: Price Deterioration Protection (ADAPTIVE)
MAX_ENTRY_PRICE_DETERIORATION_TICKS = 3
"""Maximum ticks price can move away from signal before aborting entry.
Example: Long signal at $6,934, current ask at $6,937 (3 ticks) = OK
         Long signal at $6,934, current ask at $6,945 (11 ticks) = ABORT
Prevents chasing bad entries in fast markets or during news events."""

ENTRY_PRICE_WAIT_ENABLED = True
"""Enable adaptive waiting for price to come back before aborting.
If True: Will wait up to ENTRY_PRICE_WAIT_MAX_SECONDS for price to improve
If False: Abort immediately if price > max ticks away (old rigid behavior)
Example: Price jumps 8 ticks but comes back to 2 ticks in 1 second → ENTER ✅"""

ENTRY_PRICE_WAIT_MAX_SECONDS = 5
"""Maximum seconds to wait for price to come back into acceptable range.
During wait, checks price every ENTRY_PRICE_CHECK_INTERVAL (200ms default).
Prevents giving up on good setups due to temporary price oscillations."""

ENTRY_PRICE_CHECK_INTERVAL = 0.2
"""Seconds between price re-checks during adaptive wait.
0.2s = checks 5x per second (fast enough to catch quick reversions)
Lower = more CPU/API calls but catches faster movements"""

ENTRY_ABORT_IF_WORSE_THAN_TICKS = 10
"""HARD ABORT if price deteriorates beyond this (prevents chasing disasters).
Example: Signal at $6,934, price jumps to $6,936.50 (10+ ticks) = IMMEDIATE ABORT
This is a safety net - bot won't wait for price this far gone to come back.
Should be 2-3x MAX_ENTRY_PRICE_DETERIORATION_TICKS for safety."""

# Fix #2: Partial Fill Handling
MIN_ACCEPTABLE_FILL_RATIO = 0.5
"""Minimum fill ratio to accept partial fills (0.5 = 50%).
Example: Want 3 contracts, get 2 filled (66.7%) = ACCEPT PARTIAL
         Want 3 contracts, get 1 filled (33.3%) = CLOSE POSITION, SKIP TRADE
Prevents over-positioning from re-ordering after partial fills."""

PASSIVE_ORDER_TIMEOUT = 10
"""Seconds to wait for passive order fill before checking for partial fills.
Longer timeout = more chances for passive fill (save spread)
Shorter timeout = faster fallback to aggressive (guaranteed fill)"""

# Fix #3: Order Rejection Recovery (Retry Logic)
MAX_ENTRY_RETRIES = 3
"""Maximum retry attempts for failed order placements.
Each retry uses better price (jumps queue by 1 tick).
Exponential backoff between attempts: 0.5s, 1.0s, 1.5s"""

ENTRY_RETRY_BACKOFF = 0.5
"""Base seconds for exponential backoff between retries.
Retry 1: 0.5s wait, Retry 2: 1.0s wait, Retry 3: 1.5s wait"""

# Fix #4: Fast Market Detection
FAST_MARKET_SKIP_ENABLED = True
"""Skip entries when market moving too fast (dangerous conditions).
Checks: 1) Spread widening detection
        2) Current bar volatility > 2x average
Prevents terrible fills during flash crashes, NFP releases, etc."""

FAST_MARKET_VOLATILITY_MULTIPLIER = 2.0
"""Current bar range must be > this multiplier of average range to trigger fast market skip.
Example: Average 5-bar range = 10 points, current bar = 25 points (2.5x) = SKIP ENTRY"""

# ============================================================================
# EXIT EXECUTION PROTECTION (Critical for Protecting Winning Trades)
# ============================================================================

# Fix #5: Target Order Validation
TARGET_FILL_VALIDATION_TICKS = 2
"""Maximum ticks between target price and current price to trust limit fill.
When price hits target, we check if it's still near target:
- If current price within 2 ticks of target → Likely filled at target
- If current price >2 ticks away → Price reversed, use current price instead

Example (LONG):
  Target: $6,940, High: $6,940.25 (hit!), Close: $6,940.50 (1 tick away) → Use $6,940 ✅
  Target: $6,940, High: $6,940.50 (hit!), Close: $6,936.00 (16 ticks away) → Use $6,936 ⚠️

⚠️ WARNING: Don't set this too high or you'll assume fills that didn't happen!
Recommended: 2 ticks for ES, adjust for other instruments based on tick size."""

# Fix #6: Exit Slippage Alert Threshold  
EXIT_SLIPPAGE_ALERT_TICKS = 2
"""Alert threshold for high exit slippage (especially on stop losses).
Normal slippage: 0-1 tick. High slippage: 2+ ticks.

When slippage exceeds this on a stop loss, bot will log critical warning:
- Expected: ${stop_price}
- Actual: ${fill_price}  
- Slippage: X ticks ($Y cost)
- Recommendation: Consider tighter stops or avoid fast markets

⚠️ This is for ALERTING only - doesn't prevent exits.
Use alerts to optimize entry timing and avoid high-slippage conditions."""

# Fix #7a: Entry Slippage Alert Threshold (Live Trading)
ENTRY_SLIPPAGE_ALERT_TICKS = 2
"""Alert threshold for high entry slippage (live trading only).
Normal entry slippage: 0-1 tick. High slippage: 2+ ticks.

In backtesting: Entry slippage is simulated via slippage_ticks parameter.
In live trading: Bot validates actual fill price vs expected ask/bid.

When entry slippage exceeds this threshold:
- Expected: ${ask/bid_price}
- Actual: ${fill_price}
- Slippage: X ticks ($Y cost)
- Recommendation: Tighten price validation or avoid volatile entry conditions

⚠️ This is for ALERTING only - doesn't prevent entries.
Use to identify when market conditions cause excessive entry costs."""

# Gap #2: Queue Monitoring Configuration
QUEUE_MONITORING_ENABLED = True
"""Enable passive limit order queue monitoring (Gap #2 fix).

When enabled, bot will:
- Monitor passive limit orders for up to passive_order_timeout seconds
- Check fill status every 500ms
- Cancel if price moves 2+ ticks away
- Cancel on timeout and switch to aggressive (market) order

This prevents orders sitting in queue when price moves away.
Improves fill rates by adapting to market conditions.

⚠️ Only works in live trading with broker queue position support."""

PASSIVE_ORDER_TIMEOUT = 10
"""Maximum seconds to wait for passive limit order fill.
Default: 10 seconds

After timeout, order is cancelled and bot switches to aggressive (market) execution.

Tuning:
- Lower (5s): More aggressive, fewer passive fills, higher spread costs
- Higher (15s): More patient, more passive fills, but risk missing entries
- Default (10s): Balanced approach for most market conditions

Related to MAX_ENTRY_PRICE_WAIT_SECONDS but applies to queue monitoring."""

QUEUE_PRICE_MOVE_CANCEL_TICKS = 2
"""Cancel passive order if price moves this many ticks away.
Default: 2 ticks

Example (ES futures, tick=$12.50):
- Enter long limit at bid $5000.00
- If ask moves to $5000.75+ (2 ticks), cancel and reassess
- Prevents chasing a moving market with stale limit orders

Tuning:
- Lower (1 tick): Very sensitive, cancels quickly
- Higher (3+ ticks): More patient, but risk not filling at all"""

# Gap #3: Bid/Ask Imbalance Detection
IMBALANCE_DETECTION_ENABLED = True
"""Enable bid/ask size imbalance detection (Gap #3 fix).

When enabled, bot will:
- Calculate bid_size / ask_size ratio
- Detect strong buying pressure (>3:1 ratio)
- Detect strong selling pressure (<1:3 ratio)
- Adjust entry urgency based on imbalance

Example:
- Bid: 1000 contracts, Ask: 100 contracts = 10:1 ratio (strong buying)
- Long entry: Use more aggressive routing (strong demand)
- Short entry: Use passive routing (weak demand)

This optimizes fill quality by reading market pressure."""

IMBALANCE_THRESHOLD_RATIO = 3.0
"""Bid/ask size imbalance threshold for urgency adjustment.
Default: 3.0 (3:1 ratio)

Imbalance signals:
- Ratio > 3.0: "strong_bid" (heavy buying, aggressive on longs)
- Ratio < 0.33: "strong_ask" (heavy selling, aggressive on shorts)
- 0.33 ≤ ratio ≤ 3.0: "balanced" (normal routing)

Examples (ES futures):
- Bid: 500, Ask: 100 → Ratio 5.0 → Strong bid (go aggressive on longs)
- Bid: 100, Ask: 500 → Ratio 0.2 → Strong ask (go aggressive on shorts)
- Bid: 300, Ask: 250 → Ratio 1.2 → Balanced (normal routing)

Tuning:
- Lower (2.0): More sensitive, triggers more aggressive routing
- Higher (5.0): Less sensitive, requires extreme imbalance"""

# Fix #7: Forced Flatten Retry Configuration
# NOTE: These are SAFETY-CRITICAL - don't change unless you understand the risk!

FORCED_FLATTEN_MAX_RETRIES = 5
"""Maximum retry attempts for forced flatten at market close.
⚠️ SAFETY CRITICAL - DO NOT REDUCE BELOW 5!

Retry sequence with 5 attempts:
- Attempt 1: Immediate (0s)
- Attempt 2: After 1s  
- Attempt 3: After 2s (total 3s elapsed)
- Attempt 4: After 3s (total 6s elapsed)
- Attempt 5: After 4s (total 10s elapsed)

Total time: 10 seconds before giving up and alerting for manual intervention.

Why 5 attempts?
- Market closes at 4:45 PM ET (forced flatten time)
- Need aggressive retries but can't wait forever
- 5 attempts with 10s total is maximum safe retry window
- More attempts = risk missing market close entirely

⚠️ CRITICAL: If all 5 attempts fail, position will be held overnight (CATASTROPHIC RISK!)"""

FORCED_FLATTEN_RETRY_BACKOFF_BASE = 1
"""Base seconds for retry delays (1s, 2s, 3s, 4s).
⚠️ SAFETY CRITICAL - Don't increase unless you understand the risk!

With base = 1:
- Retry 1→2: 1 second wait
- Retry 2→3: 2 seconds wait  
- Retry 3→4: 3 seconds wait
- Retry 4→5: 4 seconds wait

Increasing this means slower retries = more risk of missing market close.
Decreasing means faster retries but might hit rate limits.

Keep at 1 second unless you have specific broker requirements."""


