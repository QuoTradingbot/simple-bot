"""
Configuration Management for Capitulation Reversal Bot
Supports multiple environments with validation and environment variable overrides.

STRATEGY: Capitulation Reversal
- Wait for panic selling/buying (flush)
- Enter on exhaustion confirmation
- Target VWAP for mean reversion

STOP LOSS: User configurable via GUI
- Primary stop: 2 ticks below flush low (long) or above flush high (short)
- Emergency max: max_stop_loss_dollars setting (user configurable via GUI)
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import time
from dataclasses import dataclass, field
import pytz

# Default configuration constants
# This is the EMERGENCY MAX stop loss - user can configure this via GUI
# The actual stop is placed 2 ticks beyond the flush extreme
# This setting acts as a safety net in case of gaps or broken logic
DEFAULT_MAX_STOP_LOSS_DOLLARS = 400.0  # Default emergency max loss per trade (user configurable via GUI)


@dataclass
class BotConfiguration:
    """Type-safe configuration for the Capitulation Reversal Bot."""
    
    # Default account size constant
    DEFAULT_ACCOUNT_SIZE: float = 50000.0
    
    # Instrument Configuration (USER CONFIGURABLE - no hardcoded defaults)
    instrument: str = ""  # Single instrument (set by user, no default)
    instruments: list = field(default_factory=list)  # Multi-symbol support (empty by default)
    timezone: str = "US/Eastern"  # CME futures use US Eastern wall-clock time (handles DST automatically)
    
    # Broker Configuration
    broker: str = ""  # USER CONFIGURABLE - TopStep, Tradovate, Rithmic, NinjaTrader, etc.
    api_token: str = ""
    username: str = ""  # Broker username/email
    
    # QuoTrading License
    quotrading_license: str = ""  # QuoTrading license key for cloud RL access
    
    # Trading Parameters
    risk_per_trade: float = 0.012  # 1.2% of account per trade (increased for more profit)
    max_contracts: int = 3  # USER CONFIGURABLE - maximum contracts allowed (user sets their own limit)
    max_trades_per_day: int = 9999  # USER CONFIGURABLE - customers can adjust (9999 = unlimited)
    risk_reward_ratio: float = 2.0  # Realistic 2:1 for mean reversion with tight stops
    
    # Slippage & Commission - PRODUCTION READY
    slippage_ticks: float = 1.5  # Average 1-2 ticks per fill (conservative estimate)
    commission_per_contract: float = 2.50  # Round-turn commission (adjust to your broker)
        # Total cost per round-trip: ~3 ticks slippage + $2.50 commission = ~$42.50/contract
    
    # ==========================================================================
    # CAPITULATION REVERSAL STRATEGY - HARDCODED PARAMETERS
    # ==========================================================================
    # These parameters are HARDCODED in capitulation_detector.py
    # They are NOT configurable - the strategy uses fixed rules for all trades.
    # 
    # HARDCODED VALUES (see capitulation_detector.py for implementation):
    # - flush_min_ticks: 20 (minimum 5 dollars on ES)
    # - flush_lookback_bars: 7 (last 7 one-minute bars)
    # - flush_min_velocity: 3.0 (at least 3 ticks per bar)
    # - flush_near_extreme_ticks: 5 (within 5 ticks of flush extreme)
    # - rsi_extreme_long: 25 (RSI < 25 for long entry)
    # - rsi_extreme_short: 75 (RSI > 75 for short entry)
    # - volume_climax_mult: 2.0 (2x 20-bar average volume)
    # - stop_buffer_ticks: 2 (2 ticks below/above flush extreme)
    # - breakeven_trigger_ticks: 12 (move stop to entry after 12 ticks profit)
    # - breakeven_offset_ticks: 1 (entry + 1 tick)
    # - trailing_trigger_ticks: 15 (start trailing after 15 ticks profit)
    # - trailing_distance_ticks: 8 (trail 8 ticks behind peak)
    # - max_hold_bars: 20 (time stop after 20 bars)
    #
    # USER CONFIGURABLE via GUI:
    # - max_stop_loss_dollars: Emergency max stop (caps stop loss in dollars)
    # - max_contracts: Position size limit
    # - max_trades_per_day: Trade count limit
    # - daily_loss_limit: Daily loss cap
    # - confidence_threshold: AI signal confidence filter
    # - time_exit_enabled: Time-based exit after 20 bars (optional)
    # ==========================================================================
    
    # RSI calculation period (standard)
    rsi_period: int = 14  # Standard RSI period
    volume_lookback: int = 20  # 20-bar volume average
    
    # ==========================================================================
    # LEGACY PARAMETERS (kept for backwards compatibility, not used in new strategy)
    # ==========================================================================
    # VWAP bands - NOT USED in capitulation strategy (VWAP is target, not entry zone)
    vwap_std_dev_1: float = 2.5  # Legacy - not used
    vwap_std_dev_2: float = 2.1  # Legacy - not used
    vwap_std_dev_3: float = 3.7  # Legacy - not used
    
    # Technical Filters - All handled by CapitulationDetector now
    use_trend_filter: bool = False  # NOT USED - replaced by regime go/no-go filter
    use_rsi_filter: bool = True  # RSI still used but with new thresholds (25/75)
    use_vwap_direction_filter: bool = False  # NOT USED - VWAP is target, not filter
    use_volume_filter: bool = True  # Volume still used but with 2x threshold
    use_macd_filter: bool = False  # NOT USED - removed from strategy
    
    # Legacy RSI settings (kept for backwards compatibility)
    rsi_oversold: int = 25  # Now using rsi_extreme_long
    rsi_overbought: int = 75  # Now using rsi_extreme_short
    
    # Legacy trend settings - NOT USED
    trend_ema_period: int = 21
    trend_threshold: float = 0.0001
    
    # Legacy MACD - NOT USED
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Legacy volume settings (kept for reference)
    volume_spike_multiplier: float = 2.0  # Now using volume_climax_mult
    
    # Time Windows (US Eastern - CME Futures Wall-Clock Schedule)
    # Note: Bot can trade anytime when market is open. These are for maintenance only.
    market_open_time: time = field(default_factory=lambda: time(9, 30))  # Legacy stock market alignment - not used for VWAP reset
    entry_start_time: time = field(default_factory=lambda: time(18, 0))  # 6:00 PM Eastern - CME futures session opens
    entry_end_time: time = field(default_factory=lambda: time(16, 0))  # 4:00 PM Eastern - no new entries after this (can hold positions until 4:45 PM)
    forced_flatten_time: time = field(default_factory=lambda: time(16, 45))  # 4:45 PM Eastern - force close all positions before maintenance
    shutdown_time: time = field(default_factory=lambda: time(18, 0))  # 6:00 PM Eastern - market reopens after maintenance
    vwap_reset_time: time = field(default_factory=lambda: time(18, 0))  # 6:00 PM Eastern - daily session reset at market open
    
    # Safety Parameters - USER CONFIGURABLE
    daily_loss_limit: float = 1000.0  # USER CONFIGURABLE - max $ loss per day (or auto-calculated)
    daily_loss_percent: float = 2.0  # USER CONFIGURABLE - max daily loss as % of account
    account_size: float = 50000.0  # USER CONFIGURABLE - account size for risk calculations (needed for recovery mode to track initial balance)
    auto_calculate_limits: bool = True  # USER CONFIGURABLE - auto-calculate limits from account balance
    tick_timeout_seconds: int = 999999  # Disabled for testing
    proactive_stop_buffer_ticks: int = 2
    flatten_buffer_ticks: int = 2  # Buffer for flatten price calculation
    
    def get_daily_loss_limit(self, account_balance: float) -> float:
        """
        Calculate dynamic daily loss limit based on account rules.
        Works for any broker - uses configured risk percentage.
        
        If auto_calculate_limits=True:
        - Uses daily_loss_percent setting (default: 2%)
        
        If auto_calculate_limits=False:
        - Uses fixed daily_loss_limit value
        
        Args:
            account_balance: Current account balance
            
        Returns:
            Daily loss limit in dollars
        """
        # Use configured percentage (default 2%)
        loss_percent = self.daily_loss_percent if hasattr(self, 'daily_loss_percent') else 2.0
        return account_balance * (loss_percent / 100.0)
    
    def get_profit_target(self, account_balance: float) -> float:
        """
        Calculate profit target (optional - for evaluation/funded accounts).
        
        Default: 6% profit target
        Users can customize this based on their broker's requirements.
        
        Args:
            account_balance: Starting account balance
            
        Returns:
            Profit target in dollars
        """
        # Default 6% profit target (common for evaluation accounts)
        return account_balance * 0.06
    
    def get_account_type(self, account_balance: float) -> str:
        """
        Determine account type based on balance (informational only).
        
        Args:
            account_balance: Current account balance
            
        Returns:
            Account type string
        """
        if account_balance <= 25000:
            return "Small ($25K)"
        elif account_balance <= 50000:
            return "Medium ($50K)"
        elif account_balance <= 100000:
            return "Large ($100K)"
        elif account_balance <= 150000:
            return "XLarge ($150K)"
        elif account_balance <= 250000:
            return "Pro ($250K)"
        else:
            return f"Funded (${account_balance/1000:.0f}K)"
    
    def auto_configure_for_account(self, account_balance: float, logger=None, force: bool = False) -> bool:
        """
        Automatically configure all risk limits based on account balance.
        Called when bot connects or when balance changes significantly.
        
        Uses daily_loss_percent to calculate daily loss limit.
        
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
        profit_target = self.get_profit_target(account_balance)
        account_type = self.get_account_type(account_balance)
        
        # Auto-calculate limits based on account balance
        if self.auto_calculate_limits:
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
                logger.info(f"Manual Limits (Custom {self.daily_loss_percent}%):")
            logger.info(f"  Daily Loss Limit: ${self.daily_loss_limit:,.2f} ({self.daily_loss_percent}% of balance)")
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
            "max_contracts": self.max_contracts,
            "max_trades_per_day": self.max_trades_per_day,
            "max_stop_loss_dollars": self.max_stop_loss_dollars,
            "risk_per_trade": self.risk_per_trade,
            "risk_reward_ratio": self.risk_reward_ratio,
        }
    
    # Removed: validate_topstep_compliance() - broker-specific logic not needed
    
    # ATR-Based Dynamic Risk Management - ITERATION 3 (PROVEN WINNER!)
    use_atr_stops: bool = False  # ATR stops disabled - using proven 11-tick fixed stops
    atr_period: int = 14  # ATR calculation period
    stop_loss_atr_multiplier: float = 3.6  # Iteration 3 (tight stops)
    
    # Stop Loss Configuration - USER CONFIGURABLE
    max_stop_loss_dollars: float = DEFAULT_MAX_STOP_LOSS_DOLLARS  # USER CONFIGURABLE - max loss per trade in dollars
    # This is the "Max Loss Per Trade" setting from the GUI
    # Position closes automatically if trade loses this amount
    
    # Instrument Specifications
    tick_size: float = 0.25
    tick_value: float = 12.50  # ES full contract: $12.50 per tick
    
    # Operational Parameters
    shadow_mode: bool = False  # Signal-only mode - shows trading signals without executing trades (manual trading)
    ai_mode: bool = False  # AI position management mode - user trades manually, AI manages stops/exits
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
    
    # ==========================================================================
    # EXIT MANAGEMENT - HARDCODED IN CAPITULATION_DETECTOR.PY
    # ==========================================================================
    # All exit management rules are HARDCODED in the bot code.
    # These values are NOT configurable - the strategy uses fixed rules.
    #
    # HARDCODED EXIT RULES (see capitulation_detector.py):
    # - Breakeven: Move stop to entry + 1 tick after 12 ticks profit
    # - Trailing: Trail 8 ticks behind peak after 15 ticks profit
    # - Time Stop: Exit after 20 bars if no target/stop hit
    # - Target: VWAP (mean reversion destination)
    #
    # The following flags remain for compatibility but use hardcoded values:
    breakeven_enabled: bool = True  # ENABLED - uses hardcoded 12-tick trigger
    breakeven_profit_threshold_ticks: int = 12  # HARDCODED - Move stop to entry after 12 ticks profit
    breakeven_stop_offset_ticks: int = 1  # HARDCODED - Entry + 1 tick buffer
    trailing_stop_enabled: bool = True  # ENABLED - uses hardcoded 15-tick trigger, 8-tick trail
    trailing_stop_trigger_ticks: int = 15  # HARDCODED - Start trailing after 15 ticks profit
    trailing_stop_distance_ticks: int = 8  # HARDCODED - Trail 8 ticks behind peak
    
    # Time-Based Exit (USER CONFIGURABLE via GUI checkbox)
    time_stop_enabled: bool = False  # USER CONFIGURABLE - Exit after max_hold_bars if no resolution
    max_hold_bars: int = 20  # HARDCODED - Time stop after 20 bars (20 min on 1-min chart)
    
    # Partial Exits (static R-multiples) - Capitulation Strategy exits
    # For capitulation trades, VWAP is the primary target. Partial exits work as follows:
    # - 50% exit at 1.5R (approximately halfway to VWAP from entry)
    # - 30% exit at 2.0R (near VWAP or slightly beyond)
    # - 20% runner for extended moves
    # Note: The R-multiples are based on initial risk (stop distance), not VWAP distance
    partial_exits_enabled: bool = True  # ENABLED - Take 50% at halfway to VWAP
    partial_exit_1_percentage: float = 0.50  # 50% exit at first level
    partial_exit_1_r_multiple: float = 1.5  # Exit at 1.5R (halfway to VWAP)
    partial_exit_2_percentage: float = 0.30  # 30% exit at second level
    partial_exit_2_r_multiple: float = 2.0  # Exit at 2.0R (near VWAP)
    partial_exit_3_percentage: float = 0.20  # 20% exit at third level
    partial_exit_3_r_multiple: float = 3.0  # Exit at 3.0R (runner)
    
    # Reinforcement Learning Parameters
    # RL confidence filtering - uses RL experience to filter out low-confidence signals
    # USER CONFIGURABLE - threshold determines which signals to take
    rl_enabled: bool = True  # ENABLED - RL layer filters signals based on confidence
    rl_exploration_rate: float = 0.30  # 30% exploration (for learning)
    rl_min_exploration_rate: float = 0.05  # Minimum exploration after decay
    rl_exploration_decay: float = 0.995  # Decay rate per signal
    rl_confidence_threshold: float = 0.5  # USER CONFIGURABLE via GUI - minimum confidence to take signal
    # NOTE: Contracts are FIXED at user's max_contracts setting (no dynamic scaling)
    # NOTE: For production, RL is cloud-based. Local files only for backtesting/development.
    rl_experience_file: str = None  # Path to local RL experience file (None = cloud-based RL)
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
        
        # Time Window Validation
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
            "broker": self.broker,
            "quotrading_license": self.quotrading_license,
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
            "forced_flatten_time": self.forced_flatten_time,
            "shutdown_time": self.shutdown_time,
            "vwap_reset_time": self.vwap_reset_time,
            "daily_loss_limit": self.daily_loss_limit,
            "tick_timeout_seconds": self.tick_timeout_seconds,
            "proactive_stop_buffer_ticks": self.proactive_stop_buffer_ticks,
            "flatten_buffer_ticks": self.flatten_buffer_ticks,
            "tick_size": self.tick_size,
            "tick_value": self.tick_value,
            "shadow_mode": self.shadow_mode,
            "ai_mode": self.ai_mode,
            "max_bars_storage": self.max_bars_storage,
            # Advanced Exit Management (baseline parameters)
            "breakeven_enabled": self.breakeven_enabled,
            "breakeven_profit_threshold_ticks": self.breakeven_profit_threshold_ticks,
            "breakeven_trigger_ticks": self.breakeven_profit_threshold_ticks,  # Alias for engine compatibility
            "breakeven_stop_offset_ticks": self.breakeven_stop_offset_ticks,
            "breakeven_offset_ticks": self.breakeven_stop_offset_ticks,  # Alias for engine compatibility
            # Trailing Stop Settings (Capitulation Strategy)
            "trailing_stop_enabled": self.trailing_stop_enabled,
            "trailing_stop_trigger_ticks": self.trailing_stop_trigger_ticks,
            "trailing_trigger_ticks": self.trailing_stop_trigger_ticks,  # Alias for engine compatibility
            "trailing_stop_distance_ticks": self.trailing_stop_distance_ticks,
            "trailing_distance_ticks": self.trailing_stop_distance_ticks,  # Alias for engine compatibility
            # Time Stop Settings (Capitulation Strategy - USER CONFIGURABLE)
            "time_stop_enabled": self.time_stop_enabled,
            "max_hold_bars": self.max_hold_bars,
            "time_stop_bars": self.max_hold_bars,  # Alias for engine compatibility
            # Partial Exits
            "partial_exits_enabled": self.partial_exits_enabled,
            "partial_exit_1_percentage": self.partial_exit_1_percentage,
            "partial_exit_1_r_multiple": self.partial_exit_1_r_multiple,
            "partial_exit_2_percentage": self.partial_exit_2_percentage,
            "partial_exit_2_r_multiple": self.partial_exit_2_r_multiple,
            "partial_exit_3_percentage": self.partial_exit_3_percentage,
            "partial_exit_3_r_multiple": self.partial_exit_3_r_multiple,
            # RL Configuration
            "rl_confidence_threshold": self.rl_confidence_threshold,
            "rl_exploration_rate": self.rl_exploration_rate,
            "rl_min_exploration_rate": self.rl_min_exploration_rate,
            "rl_exploration_decay": self.rl_exploration_decay,
            # Account Settings
            "account_size": self.account_size,
            # Stop Loss Configuration
            "max_stop_loss_dollars": self.max_stop_loss_dollars,
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
    
    if os.getenv("BOT_MAX_LOSS_PER_TRADE"):
        config.max_stop_loss_dollars = float(os.getenv("BOT_MAX_LOSS_PER_TRADE"))
    
    if os.getenv("BOT_DAILY_LOSS_PERCENT"):
        config.daily_loss_percent = float(os.getenv("BOT_DAILY_LOSS_PERCENT"))
    
    # Auto-calculate limits (supports both old and new env var names)
    if os.getenv("BOT_AUTO_CALCULATE_LIMITS"):
        config.auto_calculate_limits = os.getenv("BOT_AUTO_CALCULATE_LIMITS").lower() in ("true", "1", "yes")
    elif os.getenv("BOT_USE_TOPSTEP_RULES"):  # Legacy support
        config.auto_calculate_limits = os.getenv("BOT_USE_TOPSTEP_RULES").lower() in ("true", "1", "yes")
    
    # Account Size (for risk calculations)
    if os.getenv("ACCOUNT_SIZE"):
        # Handle both numeric and string formats (e.g., "50000", "50k", "50K")
        account_size_str = os.getenv("ACCOUNT_SIZE")
        try:
            # Try parsing as float first
            config.account_size = float(account_size_str)
        except ValueError:
            # Handle "50k", "50K", "100k", "100K" format
            try:
                account_size_str_lower = account_size_str.lower().replace("k", "000")
                config.account_size = float(account_size_str_lower)
            except ValueError:
                # If still fails, log error and use default
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Invalid ACCOUNT_SIZE format: {account_size_str}. Using default: {config.account_size}")
    
    # RL/AI Configuration from GUI
    if os.getenv("BOT_CONFIDENCE_THRESHOLD"):
        # GUI provides confidence as percentage (0-100), config expects decimal (0-1)
        # Values > 1.0 are treated as percentages and converted
        # Values <= 1.0 are treated as already in decimal form
        threshold = float(os.getenv("BOT_CONFIDENCE_THRESHOLD"))
        # Note: 1.0 is treated as 100% confidence (decimal), not 1% confidence
        # If you want 1% confidence, use 0.01 or set to 1 (which will be converted to 0.01)
        if threshold > 1.0:
            threshold = threshold / 100.0
        config.rl_confidence_threshold = threshold
    
    if os.getenv("BOT_TICK_SIZE"):
        config.tick_size = float(os.getenv("BOT_TICK_SIZE"))
    
    if os.getenv("BOT_TICK_VALUE"):
        config.tick_value = float(os.getenv("BOT_TICK_VALUE"))
    
    if os.getenv("BOT_SHADOW_MODE"):
        config.shadow_mode = os.getenv("BOT_SHADOW_MODE").lower() in ("true", "1", "yes")
    
    if os.getenv("BOT_AI_MODE"):
        config.ai_mode = os.getenv("BOT_AI_MODE").lower() in ("true", "1", "yes")
    
    # Time-Based Exit (USER CONFIGURABLE via GUI checkbox)
    if os.getenv("BOT_TIME_EXIT_ENABLED"):
        config.time_stop_enabled = os.getenv("BOT_TIME_EXIT_ENABLED").lower() in ("true", "1", "yes")
    
    if os.getenv("BOT_ENVIRONMENT"):
        config.environment = os.getenv("BOT_ENVIRONMENT")
    
    # Broker name (USER CONFIGURABLE)
    if os.getenv("BOT_BROKER"):
        config.broker = os.getenv("BOT_BROKER")
    elif os.getenv("BROKER"):
        config.broker = os.getenv("BROKER")
    
    # API Token (support both old TOPSTEP and new TOPSTEPX variable names, plus generic)
    if os.getenv("BOT_API_TOKEN"):
        config.api_token = os.getenv("BOT_API_TOKEN")
    elif os.getenv("BROKER_API_TOKEN"):
        config.api_token = os.getenv("BROKER_API_TOKEN")
    elif os.getenv("TOPSTEPX_API_TOKEN"):
        config.api_token = os.getenv("TOPSTEPX_API_TOKEN")
    elif os.getenv("TOPSTEP_API_TOKEN"):
        config.api_token = os.getenv("TOPSTEP_API_TOKEN")
    
    # Username (support both old TOPSTEP and new TOPSTEPX variable names, plus generic)
    if os.getenv("BOT_USERNAME"):
        config.username = os.getenv("BOT_USERNAME")
    elif os.getenv("BROKER_USERNAME"):
        config.username = os.getenv("BROKER_USERNAME")
    elif os.getenv("TOPSTEPX_USERNAME"):
        config.username = os.getenv("TOPSTEPX_USERNAME")
    elif os.getenv("TOPSTEP_USERNAME"):
        config.username = os.getenv("TOPSTEP_USERNAME")
    
    return config


def _load_config_from_json(config: BotConfiguration) -> BotConfiguration:
    """
    Helper function to load and apply JSON config values.
    
    Args:
        config: BotConfiguration instance to update
        
    Returns:
        Updated BotConfiguration instance
    """
    from pathlib import Path
    config_file = Path(__file__).parent.parent / "data" / "config.json"
    if os.path.exists(config_file):
        import json
        with open(config_file, 'r') as f:
            json_config = json.load(f)
            # Update config with JSON values
            for key, value in json_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            # Handle legacy 'symbols' field (GUI format) - map to 'instrument' and 'instruments'
            if 'symbols' in json_config and json_config['symbols']:
                config.instrument = json_config['symbols'][0]
                config.instruments = json_config['symbols']
    
    return config


def get_development_config() -> BotConfiguration:
    """Development environment configuration."""
    config = BotConfiguration()
    config.environment = "development"
    return _load_config_from_json(config)


def get_staging_config() -> BotConfiguration:
    """Staging environment configuration."""
    config = BotConfiguration()
    config.environment = "staging"
    return _load_config_from_json(config)


def get_production_config() -> BotConfiguration:
    """Production environment configuration."""
    config = BotConfiguration()
    config.environment = "production"
    return _load_config_from_json(config)


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
    # We need to check which env vars were actually set, not just compare to defaults
    # Load environment variables again to check which ones are set
    env_vars_set = set()
    
    # Check which environment variables are actually set
    if os.getenv("BOT_INSTRUMENTS") or os.getenv("BOT_INSTRUMENT"):
        env_vars_set.add("instruments")
        env_vars_set.add("instrument")
    if os.getenv("BOT_TIMEZONE"):
        env_vars_set.add("timezone")
    if os.getenv("BOT_RISK_PER_TRADE"):
        env_vars_set.add("risk_per_trade")
    if os.getenv("BOT_MAX_CONTRACTS"):
        env_vars_set.add("max_contracts")
    if os.getenv("BOT_MAX_TRADES_PER_DAY"):
        env_vars_set.add("max_trades_per_day")
    if os.getenv("BOT_MIN_RISK_REWARD") or os.getenv("BOT_RISK_REWARD_RATIO"):
        env_vars_set.add("risk_reward_ratio")
    if os.getenv("BOT_DAILY_LOSS_LIMIT"):
        env_vars_set.add("daily_loss_limit")
    if os.getenv("BOT_DAILY_LOSS_PERCENT"):
        env_vars_set.add("daily_loss_percent")
    if os.getenv("BOT_MAX_LOSS_PER_TRADE"):
        env_vars_set.add("max_stop_loss_dollars")
    if os.getenv("BOT_AUTO_CALCULATE_LIMITS") or os.getenv("BOT_USE_TOPSTEP_RULES"):
        env_vars_set.add("auto_calculate_limits")
    if os.getenv("ACCOUNT_SIZE"):
        env_vars_set.add("account_size")
    if os.getenv("BOT_CONFIDENCE_THRESHOLD"):
        env_vars_set.add("rl_confidence_threshold")
    if os.getenv("BOT_TICK_SIZE"):
        env_vars_set.add("tick_size")
    if os.getenv("BOT_TICK_VALUE"):
        env_vars_set.add("tick_value")
    if os.getenv("BOT_DRY_RUN"):
        env_vars_set.add("dry_run")
    if os.getenv("BOT_SHADOW_MODE"):
        env_vars_set.add("shadow_mode")
    if os.getenv("BOT_AI_MODE"):
        env_vars_set.add("ai_mode")
    if os.getenv("BOT_ENVIRONMENT"):
        env_vars_set.add("environment")
    if os.getenv("BOT_BROKER") or os.getenv("BROKER"):
        env_vars_set.add("broker")
    if os.getenv("BOT_API_TOKEN") or os.getenv("TOPSTEPX_API_TOKEN") or os.getenv("TOPSTEP_API_TOKEN") or os.getenv("BROKER_API_TOKEN"):
        env_vars_set.add("api_token")
    if os.getenv("BOT_USERNAME") or os.getenv("TOPSTEPX_USERNAME") or os.getenv("TOPSTEP_USERNAME") or os.getenv("BROKER_USERNAME"):
        env_vars_set.add("username")
    
    # Apply env vars that were actually set
    for key in config.__dataclass_fields__.keys():
        if key in env_vars_set:
            env_value = getattr(env_config, key)
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
        logger.info(f"Broker: {config.broker}")
    
    logger.info(f"Instrument: {config.instrument}")
    logger.info(f"Timezone: {config.timezone}")
    logger.info(f"Risk per Trade: {config.risk_per_trade * 100:.1f}%")
    logger.info(f"Max Contracts: {config.max_contracts}")
    logger.info(f"Max Trades per Day: {config.max_trades_per_day}")
    logger.info(f"Risk/Reward Ratio: {config.risk_reward_ratio}:1")
    logger.info(f"Daily Loss Limit: ${config.daily_loss_limit:.2f}")
    
    # Time windows
    logger.info(f"Entry Window: {config.entry_start_time} - {config.entry_end_time} ET")
    logger.info(f"Forced Flatten: {config.forced_flatten_time} ET")
    
    # API token info (only relevant for live trading)
    if config.backtest_mode:
        logger.info("Mode: Backtest (no API/broker connection needed)")
    elif config.api_token:
        logger.info("API Token: *** (configured for live trading)")
    else:
        logger.info("API Token: (not configured)")
    
    logger.info("=" * 60)


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


