"""
COMPLETE TRADING SYSTEM BACKTEST
=================================
Full simulation of QuoTrading bot with ALL features:
- Cloud ML confidence filtering (70% threshold)
- Adaptive exits with RL learning
- Partial exits (runners at 2R, 3R, 5R)
- Breakeven protection
- Trailing stops
- Time-based exits (flatten mode)
- Position sizing based on confidence
- ATR-based stops
- VWAP bounce strategy
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
import json
import os
import random
from typing import Dict, List, Tuple, Optional
import statistics
import aiohttp
import asyncio
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from adaptive_exits_backtest import get_adaptive_exit_params
from adaptive_exits import AdaptiveExitManager  # EXIT RL LEARNING

# ========================================
# CONFIGURATION (matches bot)
# ========================================

CONFIG = {
    "tick_size": 0.25,
    "tick_value": 12.50,  # ES full contract = $12.50 per tick (MATCHES LIVE BOT)
    "max_contracts": 3,
    "rl_confidence_threshold": 0.50,  # 50% threshold (moderate selectivity)
    "exploration_rate": 0.30,  # 30% chance to take rejected trades (build experience)
    
    # OPTIMIZED - Adaptive exit parameters (MATCHES LIVE BOT ITERATION 3)
    "breakeven_threshold_ticks": 9,  # Move to BE at +9 ticks (Iteration 3 - matches breakeven_profit_threshold_ticks)
    "trailing_distance_ticks": 12,   # Trail 12 ticks (simplified - live bot uses adaptive exits)
    
    # Partial exits (runners) - Iteration 3
    "partial_exit_1_r_multiple": 2.0,
    "partial_exit_1_percentage": 0.50,  # 50% at 2R
    "partial_exit_2_r_multiple": 3.0,
    "partial_exit_2_percentage": 0.30,  # 30% at 3R
    "partial_exit_3_r_multiple": 5.0,
    "partial_exit_3_percentage": 0.20,  # 20% at 5R,  
    
    # Time-based exits (MATCHES LIVE BOT - 4:45-5:00 PM flatten window)
    "daily_entry_cutoff": time(16, 0),    # 4:00 PM - no new entries (matches live bot)
    "flatten_start_time": time(16, 45),  # 4:45 PM - flatten mode (matches live bot)
    "forced_flatten_time": time(17, 0),   # 5:00 PM - force close (matches live bot)
    
    # ITERATION 3 - ATR settings (MATCHES LIVE BOT)
    "atr_period": 14,
    "atr_stop_multiplier": 3.6,  # Iteration 3 - Stop at 3.6x ATR (matches stop_loss_atr_multiplier)
    # NO HARDCODED TARGET - Learned partials control ALL profit exits (2R, 3R, 5R adaptive)
    
    # ITERATION 3 - RSI settings
    "rsi_period": 10,  # Fast RSI
    "rsi_oversold": 35.0,  # LONG entry threshold
    "rsi_overbought": 65.0,  # SHORT entry threshold
}


# ========================================
# CLOUD RL CONFIDENCE API (Reinforcement Learning from 6,880+ trade experiences)
# ========================================

# Cloud API endpoint (from production bot)
CLOUD_RL_API_URL = "https://quotrading-signals.icymeadow-86b2969e.eastus.azurecontainerapps.io"

# Generate same user ID as production bot
import hashlib
import json as json_module
with open('config.json') as f:
    config_data = json_module.load(f)
    broker_username = config_data.get("broker_username", "default_user")
    BACKTEST_USER_ID = hashlib.md5(broker_username.encode()).hexdigest()[:12]
    print(f"Using User ID: {BACKTEST_USER_ID} (from {broker_username})")

async def get_rl_confidence_async(rl_state: Dict, side: str, user_id: str = None, symbol: str = "MES") -> Tuple[bool, float, str]:
    """
    Get RL confidence from REAL cloud API (pattern matching across all past trades).
    Returns: (take_signal, confidence, reason)
    """
    if user_id is None:
        user_id = BACKTEST_USER_ID
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Prepare payload for cloud API (correct format for /api/ml/should_take_signal)
            payload = {
                "user_id": user_id,
                "symbol": symbol,
                "signal": side.upper(),  # 'LONG' or 'SHORT'
                "entry_price": rl_state.get("entry_price", 0),
                "vwap": rl_state.get("vwap", 0),
                "rsi": rl_state.get("rsi", 50),
                "vix": rl_state.get("vix", 15.0),
                "volume_ratio": rl_state.get("volume_ratio", 1.0),
                "hour": rl_state.get("hour", 12),
                "day_of_week": rl_state.get("day_of_week", 0),
                "recent_pnl": rl_state.get("recent_pnl", 0.0),
                "streak": rl_state.get("streak", 0)
            }
            
            # Increase timeout on retries
            timeout_seconds = 5.0 + (attempt * 2.0)  # 5s, 7s, 9s
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{CLOUD_RL_API_URL}/api/ml/should_take_signal",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        confidence = data.get("confidence", 0.5)
                        take_signal = data.get("take_trade", False)
                        reason = data.get("reason", "Cloud RL evaluation")
                        
                        # Debug: Show what cloud RL returned
                        print(f"    Cloud RL: {side.upper()} conf={confidence:.1%} take={take_signal} reason={reason[:50]}")
                        
                        if attempt > 0:
                            print(f"[CLOUD RL] [OK] Success on retry {attempt + 1}")
                        
                        return take_signal, confidence, reason
                    else:
                        print(f"⚠️ Cloud RL API returned status {response.status} (attempt {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(0.5 * (attempt + 1))
                            continue
                        print("[X] Cloud RL API failed after all retries - REJECTING TRADE for safety")
                        return False, 0.0, "Cloud RL API error - rejecting trade for safety"
                        
        except asyncio.TimeoutError:
            print(f"⚠️ Cloud RL API timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            print("[X] Cloud RL API timeout after all retries - REJECTING TRADE for safety")
            return False, 0.0, "RL API timeout - rejecting trade for safety"
        except Exception as e:
            print(f"⚠️ Cloud RL API error: {e} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            print(f"[X] Cloud RL API error after all retries - REJECTING TRADE for safety: {e}")
            return False, 0.0, f"RL API error - rejecting trade for safety"
    
    # Should never reach here
    print("[X] Unknown error in RL API call - REJECTING TRADE for safety")
    return False, 0.0, "Unknown error - rejecting trade for safety"


def get_rl_confidence(rl_state: Dict, side: str) -> Tuple[bool, float, str]:
    """
    Synchronous wrapper for cloud RL API (runs async call).
    """
    try:
        # Small delay to avoid rate limiting (429 errors)
        import time
        time.sleep(0.1)  # 100ms between calls
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(get_rl_confidence_async(rl_state, side))
        loop.close()
        return result
    except Exception as e:
        print(f"[X] Error in RL confidence sync wrapper: {e}")
        return False, 0.0, f"Wrapper error: {e}"


# ========================================
# POSITION SIZING (confidence-based)
# ========================================

def calculate_position_size(confidence: float, account_size: float = 50000.0) -> int:
    """
    Scale position size based on ML confidence.
    Higher confidence = more contracts (up to max).
    
    DYNAMIC SCALING based on your confidence threshold:
    - Divides range from threshold to 100% into equal tiers
    - Each tier gets 1 more contract
    
    Example with threshold=50%, max=3:
      50-66% → 1 contract
      67-83% → 2 contracts
      84-100% → 3 contracts
    
    Example with threshold=30%, max=3:
      30-53% → 1 contract
      54-76% → 2 contracts
      77-100% → 3 contracts
    """
    max_contracts = CONFIG['max_contracts']
    threshold = CONFIG['rl_confidence_threshold']
    
    # Calculate tier size: range from threshold to 100%, divided by max_contracts
    range_size = 1.0 - threshold  # e.g., if threshold=0.50, range is 0.50 (50% to 100%)
    tier_size = range_size / max_contracts  # e.g., 0.50 / 3 = 0.167 (16.7% per tier)
    
    # Calculate which tier this confidence falls into
    if confidence < threshold:
        contracts = 1  # Below threshold still gets 1 contract
    else:
        # How far above threshold? (0.0 to 1.0)
        above_threshold = confidence - threshold
        
        # Which tier? (1 to max_contracts)
        tier = int(above_threshold / tier_size) + 1
        contracts = min(tier, max_contracts)
    
    return contracts


# ========================================
# INDICATOR CALCULATIONS
# ========================================

def calculate_rsi(prices: pd.Series, period: int = 10) -> float:
    """Calculate RSI indicator (Iteration 3 uses period=10 for fast RSI)."""
    deltas = prices.diff()
    gains = deltas.where(deltas > 0, 0.0)
    losses = -deltas.where(deltas < 0, 0.0)
    
    avg_gain = gains.rolling(window=period).mean().iloc[-1]
    avg_loss = losses.rolling(window=period).mean().iloc[-1]
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range."""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean().iloc[-1]
    
    return atr


def calculate_vwap_bands(df_day: pd.DataFrame) -> Dict:
    """Calculate daily VWAP with standard deviation bands."""
    df_day['typical_price'] = (df_day['high'] + df_day['low'] + df_day['close']) / 3
    df_day['tpv'] = df_day['typical_price'] * df_day['volume']
    
    cumulative_tpv = df_day['tpv'].cumsum()
    cumulative_volume = df_day['volume'].cumsum()
    
    vwap = cumulative_tpv / cumulative_volume
    
    # Standard deviation
    df_day['vwap'] = vwap
    df_day['price_diff_sq'] = (df_day['typical_price'] - df_day['vwap']) ** 2
    df_day['weighted_diff_sq'] = df_day['price_diff_sq'] * df_day['volume']
    
    cumulative_weighted_diff_sq = df_day['weighted_diff_sq'].cumsum()
    variance = cumulative_weighted_diff_sq / cumulative_volume
    std_dev = np.sqrt(variance)
    
    latest_vwap = vwap.iloc[-1]
    latest_std = std_dev.iloc[-1]
    
    # ITERATION 3 band multipliers (matches production bot)
    return {
        'vwap': latest_vwap,
        'upper_1': latest_vwap + (2.5 * latest_std),
        'upper_2': latest_vwap + (2.1 * latest_std),  # Entry zone
        'upper_3': latest_vwap + (3.7 * latest_std),  # Exit/stop
        'lower_1': latest_vwap - (2.5 * latest_std),
        'lower_2': latest_vwap - (2.1 * latest_std),  # Entry zone
        'lower_3': latest_vwap - (3.7 * latest_std),  # Exit/stop
        'std_dev': latest_std
    }


# ========================================
# TRADE SIMULATION CLASS
# ========================================

class Trade:
    """Represents an active trade with ADAPTIVE exit logic (matches live bot)."""
    
    def __init__(self, entry_bar: pd.Series, side: str, contracts: int, 
                 confidence: float, atr: float, vwap_bands: Dict, all_bars: List,
                 adaptive_manager=None, entry_market_state: Dict = None):
        self.entry_time = entry_bar['timestamp']
        self.entry_price = entry_bar['close']
        self.side = side
        self.contracts = contracts
        self.original_contracts = contracts
        self.confidence = confidence
        self.all_bars = all_bars  # Need full bar history for adaptive exits
        self.adaptive_manager = adaptive_manager  # For learned exit parameters
        self.entry_market_state = entry_market_state or {}  # Market state at entry for scaling
        
        # Calculate stop based on ATR (keep stop hardcoded for risk management)
        tick_size = CONFIG['tick_size']
        atr_ticks = atr / tick_size
        
        stop_distance_ticks = int(atr_ticks * CONFIG['atr_stop_multiplier'])
        # NO HARDCODED TARGET - Let learned partials control ALL profit exits
        
        if side == 'long':
            self.stop_price = entry_bar['close'] - (stop_distance_ticks * tick_size)
        else:  # short
            self.stop_price = entry_bar['close'] + (stop_distance_ticks * tick_size)
        
        self.initial_risk_ticks = stop_distance_ticks
        
        # Exit tracking
        self.breakeven_active = False
        self.trailing_active = False
        self.partial_exits = []
        self.highest_price = entry_bar['close']
        self.lowest_price = entry_bar['close']
        
        # Exit flags
        self.partial_1_done = False
        self.partial_2_done = False
        self.partial_3_done = False
        
        # ADAPTIVE: Will be recalculated each bar
        self.current_exit_params = None
    
    def update(self, bar: pd.Series, bar_index: int) -> Optional[Tuple[str, float, int]]:
        """
        Update trade with new bar, check exits.
        Uses ADAPTIVE exits that adjust to market conditions (like live bot).
        Returns: (exit_reason, exit_price, contracts_closed) or None
        """
        current_price = bar['close']
        current_time = bar['timestamp'].time()
        
        # Update price extremes
        self.highest_price = max(self.highest_price, bar['high'])
        self.lowest_price = min(self.lowest_price, bar['low'])
        
        # Calculate current P&L
        tick_size = CONFIG['tick_size']
        tick_value = CONFIG['tick_value']
        
        if self.side == 'long':
            profit_ticks = (current_price - self.entry_price) / tick_size
        else:
            profit_ticks = (self.entry_price - current_price) / tick_size
        
        r_multiple = profit_ticks / self.initial_risk_ticks if self.initial_risk_ticks > 0 else 0
        
        # ========================================
        # GET ADAPTIVE EXIT PARAMETERS (RECALCULATE EACH BAR)
        # ========================================
        # Convert recent bars to list of dicts for adaptive exit function
        recent_bars = []
        start_idx = max(0, bar_index - 50)  # Last 50 bars for context
        for idx in range(start_idx, bar_index + 1):
            b = self.all_bars.iloc[idx]
            recent_bars.append({
                'timestamp': b['timestamp'],
                'open': b['open'],
                'high': b['high'],
                'low': b['low'],
                'close': b['close'],
                'volume': b['volume'],
                'atr': b.get('atr', 2.0)
            })
        
        # Get adaptive exit params based on current conditions
        self.current_exit_params = get_adaptive_exit_params(
            bars=recent_bars,
            position={'entry_time': self.entry_time},
            current_price=current_price,
            config=CONFIG,
            entry_time=self.entry_time,
            adaptive_manager=self.adaptive_manager  # Pass manager for learned params
        )
        
        # Extract adaptive parameters
        breakeven_threshold = self.current_exit_params['breakeven_threshold_ticks']
        trailing_distance_ticks = self.current_exit_params['trailing_distance_ticks']
        
        # ========================================
        # EXIT PRIORITY ORDER (matches bot)
        # ========================================
        
        # 1. TIME-BASED EXITS (highest priority)
        if current_time >= CONFIG['forced_flatten_time']:
            return ('forced_flatten', current_price, self.contracts)
        
        if current_time >= CONFIG['flatten_start_time'] and profit_ticks > 0:
            # Take profit during flatten window
            return ('flatten_profit', current_price, self.contracts)
        
        # 2. PARTIAL EXITS (runners) - CHECK FIRST before stop/target
        partial_exit = self.check_partial_exits(r_multiple, current_price)
        if partial_exit:
            return partial_exit
        
        # 3. STOP LOSS
        if self.side == 'long' and bar['low'] <= self.stop_price:
            return ('stop_loss', self.stop_price, self.contracts)
        if self.side == 'short' and bar['high'] >= self.stop_price:
            return ('stop_loss', self.stop_price, self.contracts)
        
        # 4. ADAPTIVE BREAKEVEN PROTECTION (uses learned threshold)
        if not self.breakeven_active and profit_ticks >= breakeven_threshold:
            self.breakeven_active = True
            self.stop_price = self.entry_price
        
        # 5. ADAPTIVE TRAILING STOP (uses learned trail distance)
        if self.breakeven_active:
            self.update_trailing_stop_adaptive(current_price, trailing_distance_ticks)
        
        return None
    
    def check_partial_exits(self, r_multiple: float, current_price: float) -> Optional[Tuple[str, float, int]]:
        """Check and execute partial exits using LEARNED scaling strategy."""
        
        # Get learned scaling strategy from adaptive exit manager
        from adaptive_exits_backtest import get_recommended_scaling_strategy, detect_market_regime
        
        # Build market state for scaling decision
        market_state = {
            'rsi': self.entry_market_state.get('rsi', 50),
            'volume_ratio': self.entry_market_state.get('volume_ratio', 1.0),
            'hour': self.entry_market_state.get('hour', 12),
            'streak': self.entry_market_state.get('streak', 0)
        }
        
        # Detect regime (simplified for backtest)
        regime = self.entry_market_state.get('regime', 'NORMAL')
        
        # Get adaptive scaling strategy (works for ALL contract sizes)
        scaling = get_recommended_scaling_strategy(market_state, regime, self.adaptive_manager)
        
        # SINGLE CONTRACT: Take full position at learned target
        if self.original_contracts <= 1:
            learned_target = scaling.get('single_contract_target', scaling['partial_3_r'])
            if r_multiple >= learned_target:
                return ('learned_target', current_price, self.contracts)
            return None
        
        # MULTI-CONTRACT: Execute learned partial scaling
        
        # First partial (LEARNED r-multiple and percentage)
        if r_multiple >= scaling['partial_1_r'] and not self.partial_1_done:
            contracts_to_close = int(self.original_contracts * scaling['partial_1_pct'])
            if contracts_to_close >= 1:
                self.partial_1_done = True
                self.contracts -= contracts_to_close
                self.partial_exits.append({
                    'r_multiple': r_multiple,
                    'price': current_price,
                    'contracts': contracts_to_close,
                    'level': 1,
                    'strategy': scaling['strategy']
                })
                return ('partial_1', current_price, contracts_to_close)
        
        # Second partial (LEARNED)
        if r_multiple >= scaling['partial_2_r'] and not self.partial_2_done:
            contracts_to_close = int(self.original_contracts * scaling['partial_2_pct'])
            if contracts_to_close >= 1 and self.contracts >= contracts_to_close:
                self.partial_2_done = True
                self.contracts -= contracts_to_close
                self.partial_exits.append({
                    'r_multiple': r_multiple,
                    'price': current_price,
                    'contracts': contracts_to_close,
                    'level': 2,
                    'strategy': scaling['strategy']
                })
                return ('partial_2', current_price, contracts_to_close)
        
        # Third partial (LEARNED - final runner)
        if r_multiple >= scaling['partial_3_r'] and not self.partial_3_done:
            if self.contracts >= 1:
                contracts_to_close = self.contracts
                self.partial_3_done = True
                self.contracts = 0
                self.partial_exits.append({
                    'r_multiple': r_multiple,
                    'price': current_price,
                    'contracts': contracts_to_close,
                    'level': 3,
                    'strategy': scaling['strategy']
                })
                return ('partial_3', current_price, contracts_to_close)
        
        return None
    
    def update_trailing_stop_adaptive(self, current_price: float, trailing_distance_ticks: int):
        """Update trailing stop using ADAPTIVE distance (not fixed)."""
        tick_size = CONFIG['tick_size']
        trail_distance = trailing_distance_ticks * tick_size
        
        if self.side == 'long':
            # Trail below highest high
            new_stop = self.highest_price - trail_distance
            if new_stop > self.stop_price:
                self.stop_price = new_stop
                self.trailing_active = True
        else:  # short
            # Trail above lowest low
            new_stop = self.lowest_price + trail_distance
            if new_stop < self.stop_price:
                self.stop_price = new_stop
                self.trailing_active = True


# ========================================
# EXPERIENCE RECORDING FOR RL LEARNING
# ========================================

def save_signal_experience(rl_state: Dict, took_trade: bool, outcome: Dict, backtest_mode: bool = False):
    """
    Save signal experience to JSON for RL learning.
    Appends to cloud-api/signal_experience.json (MAIN file used by live bot).
    
    Args:
        rl_state: State at signal time (with all features)
        took_trade: Whether cloud RL approved the trade
        outcome: Trade result (pnl, duration, exit_reason, etc.)
        backtest_mode: Ignored - always saves to main file
    """
    from datetime import datetime
    import json
    import os
    
    experience = {
        'timestamp': datetime.now().isoformat(),
        'state': rl_state,
        'action': {
            'took_trade': took_trade,
            'exploration_rate': 0.05  # Backtest uses cloud RL decisions (no exploration)
        },
        'reward': outcome.get('pnl', 0),
        'duration': outcome.get('duration_min', 0) * 60,  # Convert to seconds
        'execution': {
            'exit_reason': outcome.get('exit_reason', 'unknown'),
            'partial_fill': outcome.get('partial_exits', 0) > 0,
            'fill_ratio': 1.0,  # Backtest assumes full fills
            'entry_slippage_ticks': 1.0,  # Estimate 1 tick slippage
            'order_type_used': 'market',
            'held_full_duration': outcome.get('duration_min', 0) > 30
        }
    }
    
    # ALWAYS use MAIN signal experience file (live bot reads this)
    experience_file = 'cloud-api/signal_experience.json'
    
    # Load existing experiences
    if os.path.exists(experience_file):
        with open(experience_file, 'r') as f:
            try:
                data = json.load(f)
                # Handle both dict and list formats
                if isinstance(data, dict):
                    experiences = data
                else:
                    experiences = {str(i): exp for i, exp in enumerate(data)}
            except:
                experiences = {}
    else:
        experiences = {}
    
    # Add new experience with unique key
    next_key = str(len(experiences))
    experiences[next_key] = experience
    
    # Save back to file
    with open(experience_file, 'w') as f:
        json.dump(experiences, f, indent=2)
    
    return len(experiences)


# ========================================
# BACKTEST ENGINE
# ========================================

def run_full_backtest(csv_file: str, days: int = 15):
    """
    Run complete backtest with ALL bot features.
    """
    print("=" * 80)
    print("QUOTRADING BOT - FULL SYSTEM BACKTEST")
    print("=" * 80)
    print(f"Data: {csv_file}")
    print(f"Period: Last {days} days")
    print(f"ML Confidence Threshold: {CONFIG['rl_confidence_threshold']:.0%}")
    print(f"Exploration Rate: {CONFIG['exploration_rate']:.0%} (randomly take some rejected trades)")
    print(f"Max Contracts: {CONFIG['max_contracts']}")
    print(f"Partial Exits: ADAPTIVE (learned from market context)")
    print(f"  - Aggressive: 70% @ 2R, 25% @ 3R (choppy/overbought)")
    print(f"  - Hold Full: 0% @ 3R, 30% @ 4R, 70% @ 6R (trending)")
    print(f"  - Balanced: 50% @ 2R, 30% @ 3R, 20% @ 5R (normal)")
    print(f"  - NO HARDCODED PROFIT TARGET - Partials control ALL exits")
    print("=" * 80)
    print()
    
    # Load data
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # CRITICAL: Convert timestamps to Eastern Time to match live bot
    # Historical data is assumed to be in UTC (from broker feed)
    # Live bot operates in America/New_York timezone
    import pytz
    utc_tz = pytz.UTC
    et_tz = pytz.timezone('America/New_York')
    
    # If timestamps are naive (no timezone), assume UTC
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(utc_tz)
    
    # Convert to Eastern Time to match live bot
    df['timestamp'] = df['timestamp'].dt.tz_convert(et_tz)
    
    # Filter to last N days
    last_date = df['timestamp'].max()
    start_date = last_date - pd.Timedelta(days=days)
    df = df[df['timestamp'] >= start_date].reset_index(drop=True)
    
    print(f"Loaded {len(df):,} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Timezone: Eastern Time (America/New_York) - matches live bot")
    print()
    
    # Initialize EXIT RL MANAGER for learning exit patterns
    print("Initializing Exit RL Manager with CLOUD API for pattern learning...")
    adaptive_exit_manager = AdaptiveExitManager(
        config=CONFIG,
        experience_file='cloud-api/exit_experience.json',  # Local fallback only
        cloud_api_url=CLOUD_RL_API_URL  # Fetch/save to cloud for 10k+ shared experiences
    )
    print(f"  ✅ Loaded {len(adaptive_exit_manager.exit_experiences):,} past exit experiences from {'CLOUD' if adaptive_exit_manager.use_cloud else 'LOCAL'}")
    print()
    
    # Trading state
    active_trade = None
    signals_detected = 0
    signals_ml_approved = 0
    signals_ml_rejected = 0
    completed_trades = []
    
    # RL Learning - Track performance for state
    current_streak = 0  # Positive = wins, negative = losses
    recent_pnl_sum = 0.0
    
    # Process each bar
    for idx in range(100, len(df)):  # Start at 100 for indicators
        bar = df.iloc[idx]
        current_date = bar['timestamp'].date()
        
        # Get daily data for VWAP
        df_day = df[df['timestamp'].dt.date == current_date].iloc[:idx - df[df['timestamp'].dt.date == current_date].index[0] + 1]
        
        if len(df_day) < 30:
            continue  # Need enough bars for VWAP
        
        vwap_bands = calculate_vwap_bands(df_day.copy())
        
        # Calculate indicators (ITERATION 3 parameters)
        recent_closes = df.iloc[idx-50:idx]['close']
        rsi = calculate_rsi(recent_closes, period=CONFIG['rsi_period'])
        atr = calculate_atr(df.iloc[idx-20:idx], period=CONFIG['atr_period'])
        
        # Calculate volume ratio for RL state
        if idx >= 20:
            avg_volume = df.iloc[idx-20:idx]['volume'].mean()
            volume_ratio = bar['volume'] / avg_volume if avg_volume > 0 else 1.0
        else:
            volume_ratio = 1.0
        
        # Calculate VWAP distance for RL state
        vwap_distance = (bar['close'] - vwap_bands['vwap']) / vwap_bands['vwap'] if vwap_bands['vwap'] > 0 else 0.0
        
        # Update active trade (pass idx for adaptive exits)
        if active_trade:
            exit_result = active_trade.update(bar, idx)
            
            if exit_result:
                exit_reason, exit_price, contracts_closed = exit_result
                
                # Calculate P&L
                tick_size = CONFIG['tick_size']
                tick_value = CONFIG['tick_value']
                
                if active_trade.side == 'long':
                    profit_ticks = (exit_price - active_trade.entry_price) / tick_size
                else:
                    profit_ticks = (active_trade.entry_price - exit_price) / tick_size
                
                pnl = profit_ticks * tick_value * contracts_closed
                
                # For partial exits, record partial trade
                if 'partial' in exit_reason:
                    print(f"  └─ {exit_reason.upper()}: Closed {contracts_closed} @ ${exit_price:.2f} | P&L: ${pnl:+.2f}")
                    
                    # RECORD PARTIAL EXIT FOR EXIT RL LEARNING
                    if hasattr(active_trade, 'current_exit_params') and active_trade.current_exit_params:
                        try:
                            # Capture current market state for exit learning
                            market_state = {
                                'rsi': rsi,
                                'volume_ratio': volume_ratio,
                                'hour': bar['timestamp'].hour,
                                'day_of_week': bar['timestamp'].weekday(),
                                'streak': current_streak,
                                'recent_pnl': recent_pnl_sum,
                                'vix': 15.0,
                                'vwap_distance': vwap_distance,
                                'atr': atr
                            }
                            
                            # Record partial exit outcome
                            adaptive_exit_manager.record_exit_outcome(
                                regime=active_trade.current_exit_params.get('market_regime', 'UNKNOWN'),
                                exit_params=active_trade.current_exit_params,
                                trade_outcome={
                                    'pnl': pnl,
                                    'duration': 0,  # Not tracking for partials
                                    'exit_reason': exit_reason,
                                    'side': active_trade.side,
                                    'contracts': contracts_closed,
                                    'win': pnl > 0,
                                    'partial': True,
                                    'r_multiple': (profit_ticks / active_trade.initial_risk_ticks) if active_trade.initial_risk_ticks > 0 else 0
                                },
                                market_state=market_state
                            )
                        except Exception as e:
                            print(f"    [WARN] Partial exit learning failed: {e}")
                    
                    # Don't end trade, just reduce position
                    if active_trade.contracts > 0:
                        continue  # Keep trade open
                
                # Full exit - record completed trade
                duration = (bar['timestamp'] - active_trade.entry_time).total_seconds() / 60
                
                # Calculate total P&L including partials
                total_pnl = pnl
                for partial in active_trade.partial_exits:
                    if active_trade.side == 'long':
                        partial_profit_ticks = (partial['price'] - active_trade.entry_price) / tick_size
                    else:
                        partial_profit_ticks = (active_trade.entry_price - partial['price']) / tick_size
                    total_pnl += partial_profit_ticks * tick_value * partial['contracts']
                
                # Update streak tracking for RL
                if total_pnl > 0:
                    current_streak = current_streak + 1 if current_streak > 0 else 1
                else:
                    current_streak = current_streak - 1 if current_streak < 0 else -1
                
                # Update recent P&L (last 5 trades rolling sum)
                recent_pnl_sum += total_pnl
                if len(completed_trades) >= 5:
                    recent_pnl_sum -= completed_trades[-5]['pnl']
                
                completed_trades.append({
                    'entry_time': active_trade.entry_time,
                    'exit_time': bar['timestamp'],
                    'side': active_trade.side,
                    'contracts': active_trade.original_contracts,
                    'entry_price': active_trade.entry_price,
                    'exit_price': exit_price,
                    'pnl': total_pnl,
                    'exit_reason': exit_reason,
                    'duration_min': duration,
                    'confidence': active_trade.confidence,
                    'partial_exits': len(active_trade.partial_exits),
                    'r_multiple': (profit_ticks / active_trade.initial_risk_ticks) if active_trade.initial_risk_ticks > 0 else 0
                })
                
                # SAVE SIGNAL EXPERIENCE FOR RL LEARNING
                # Store the entry state and trade outcome for pattern learning
                if hasattr(active_trade, 'entry_state'):
                    outcome = {
                        'pnl': total_pnl,
                        'duration_min': duration,
                        'exit_reason': exit_reason,
                        'partial_exits': len(active_trade.partial_exits)
                    }
                    total_experiences = save_signal_experience(
                        active_trade.entry_state, 
                        True,  # Trade was taken (cloud RL approved)
                        outcome,
                        backtest_mode=False  # Save to MAIN file for live bot learning
                    )
                    if len(completed_trades) % 5 == 0:  # Log every 5 trades
                        print(f"  [RL LEARNING] Saved to signal_experience.json #{total_experiences} | Recent P&L: ${recent_pnl_sum:+.2f} | Streak: {current_streak:+d}")
                
                # RECORD FULL EXIT FOR EXIT RL LEARNING
                if hasattr(active_trade, 'current_exit_params') and active_trade.current_exit_params:
                    try:
                        # Capture current market state for exit learning
                        market_state = {
                            'rsi': rsi,
                            'volume_ratio': volume_ratio,
                            'hour': bar['timestamp'].hour,
                            'day_of_week': bar['timestamp'].weekday(),
                            'streak': current_streak,
                            'recent_pnl': recent_pnl_sum,
                            'vix': 15.0,
                            'vwap_distance': vwap_distance,
                            'atr': atr
                        }
                        
                        # Record final exit outcome with full market context
                        adaptive_exit_manager.record_exit_outcome(
                            regime=active_trade.current_exit_params.get('market_regime', 'UNKNOWN'),
                            exit_params=active_trade.current_exit_params,
                            trade_outcome={
                                'pnl': total_pnl,
                                'duration': duration,
                                'exit_reason': exit_reason,
                                'side': active_trade.side,
                                'contracts': contracts_closed,
                                'win': total_pnl > 0
                            },
                            market_state=market_state
                        )
                        
                        if len(completed_trades) % 5 == 0:  # Log every 5 trades
                            print(f"  [EXIT RL] Saved exit experience | Regime: {active_trade.current_exit_params.get('market_regime', 'UNKNOWN')} | RSI: {rsi:.1f} | Vol: {volume_ratio:.2f}x")
                    except Exception as e:
                        print(f"  [WARN] Exit learning failed: {e}")
                
                # Print trade result
                win_or_loss = "WIN" if total_pnl > 0 else "LOSS"
                print(f"{'[OK]' if total_pnl > 0 else '[X]'} {win_or_loss}: {active_trade.side.upper()} {active_trade.original_contracts}x | "
                      f"Entry: ${active_trade.entry_price:.2f} | Exit: ${exit_price:.2f} | "
                      f"P&L: ${total_pnl:+.2f} | {exit_reason} | {duration:.0f}min | "
                      f"Partials: {len(active_trade.partial_exits)}")
                
                active_trade = None
                continue
        
        # Check for new signals (only if no active trade)
        if active_trade is None:
            # TRADING HOURS CHECK (matches live bot)
            # Live bot trading window: Sunday 6 PM - Friday 5 PM ET
            # Daily entry cutoff: 4:00 PM (16:00) - no new entries after this
            # Daily flatten: 4:45-5:00 PM (16:45-17:00)
            # Daily maintenance: 5:00-6:00 PM (17:00-18:00)
            bar_time = bar['timestamp'].time()
            
            # Skip signals after 4:00 PM daily entry cutoff
            if bar_time >= CONFIG['daily_entry_cutoff']:
                continue  # After 4 PM - no new entries (can hold existing)
            
            # Skip signals during flatten mode (4:45-5:00 PM)
            if bar_time >= CONFIG['flatten_start_time'] and bar_time < CONFIG['forced_flatten_time']:
                continue  # Flatten mode - no new entries
            
            # Skip signals during maintenance window (5:00-6:00 PM / 17:00-18:00)
            if bar_time >= time(17, 0) and bar_time < time(18, 0):
                continue  # Maintenance - market closed
            
            prev_bar = df.iloc[idx - 1]
            
            # LONG signal check: VWAP bounce + RSI oversold
            if (prev_bar['low'] <= vwap_bands['lower_2'] and 
                bar['close'] > prev_bar['close'] and
                rsi < CONFIG['rsi_oversold']):  # RSI must be < 35 (Iteration 3)
                
                signals_detected += 1
                
                # Build COMPLETE RL state (matches live bot format)
                rl_state = {
                    'entry_price': bar['close'],
                    'vwap': vwap_bands['vwap'],
                    'rsi': rsi,
                    'atr': atr,
                    'vix': 15.0,  # Default VIX (TODO: fetch real historical VIX)
                    'price': bar['close'],
                    'hour': bar['timestamp'].hour,
                    'day_of_week': bar['timestamp'].weekday(),  # 0=Monday, 6=Sunday
                    'volume_ratio': volume_ratio,
                    'vwap_distance': vwap_distance,
                    'recent_pnl': recent_pnl_sum,
                    'streak': current_streak,
                    'side': 'long'
                }
                
                # Get RL confidence from cloud (pattern matching 6,880+ experiences)
                take_signal, confidence, reason = get_rl_confidence(rl_state, 'long')
                
                # EXPLORATION: Sometimes take rejected trades to build experience
                explored = False
                if not take_signal and random.random() < CONFIG['exploration_rate']:
                    take_signal = True
                    explored = True
                    reason = f"EXPLORATION (was {confidence:.1%})"
                
                if take_signal:
                    signals_ml_approved += 1
                    contracts = calculate_position_size(confidence)
                    
                    print(f"\nLONG SIGNAL @ {bar['timestamp']} | ${bar['close']:.2f}")
                    print(f"  RL Confidence: {confidence:.1%} APPROVED")
                    print(f"  Position Size: {contracts} contracts ({reason})")
                    
                    active_trade = Trade(bar, 'long', contracts, confidence, atr, vwap_bands, df, adaptive_exit_manager, rl_state)
                    active_trade.entry_state = rl_state  # Store for RL learning
                else:
                    signals_ml_rejected += 1
                    print(f"  X LONG @ {bar['timestamp']:.19s} REJECTED: {confidence:.1%} ({reason})")
            
            # SHORT signal check: VWAP bounce + RSI overbought
            if (prev_bar['high'] >= vwap_bands['upper_2'] and 
                bar['close'] < prev_bar['close'] and
                rsi > CONFIG['rsi_overbought']):  # RSI must be > 65 (Iteration 3)
                
                signals_detected += 1
                
                # Build COMPLETE RL state (matches live bot format)
                rl_state = {
                    'entry_price': bar['close'],
                    'vwap': vwap_bands['vwap'],
                    'rsi': rsi,
                    'atr': atr,
                    'vix': 15.0,  # Default VIX (TODO: fetch real historical VIX)
                    'price': bar['close'],
                    'hour': bar['timestamp'].hour,
                    'day_of_week': bar['timestamp'].weekday(),  # 0=Monday, 6=Sunday
                    'volume_ratio': volume_ratio,
                    'vwap_distance': vwap_distance,
                    'recent_pnl': recent_pnl_sum,
                    'streak': current_streak,
                    'side': 'short'
                }
                
                # Get RL confidence from cloud (pattern matching 6,880+ experiences)
                take_signal, confidence, reason = get_rl_confidence(rl_state, 'short')
                
                # EXPLORATION: Sometimes take rejected trades to build experience
                explored = False
                if not take_signal and random.random() < CONFIG['exploration_rate']:
                    take_signal = True
                    explored = True
                    reason = f"EXPLORATION (was {confidence:.1%})"
                
                if take_signal:
                    signals_ml_approved += 1
                    contracts = calculate_position_size(confidence)
                    
                    print(f"\nSHORT SIGNAL @ {bar['timestamp']} | ${bar['close']:.2f}")
                    print(f"  RL Confidence: {confidence:.1%} APPROVED")
                    print(f"  Position Size: {contracts} contracts ({reason})")
                    
                    active_trade = Trade(bar, 'short', contracts, confidence, atr, vwap_bands, df, adaptive_exit_manager, rl_state)
                    active_trade.entry_state = rl_state  # Store RL state for learning
                else:
                    signals_ml_rejected += 1
                    print(f"  X SHORT @ {bar['timestamp']:.19s} REJECTED: {confidence:.1%} ({reason})")
    
    # ========================================
    # PERFORMANCE METRICS
    # ========================================
    
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    
    if not completed_trades:
        print("No completed trades!")
        return
    
    df_trades = pd.DataFrame(completed_trades)
    
    # Basic stats
    total_trades = len(df_trades)
    winning_trades = len(df_trades[df_trades['pnl'] > 0])
    losing_trades = len(df_trades[df_trades['pnl'] <= 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # P&L stats
    total_pnl = df_trades['pnl'].sum()
    avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = df_trades[df_trades['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
    
    gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df_trades[df_trades['pnl'] <= 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Risk metrics
    largest_win = df_trades['pnl'].max()
    largest_loss = df_trades['pnl'].min()
    
    # Calculate max drawdown
    cumulative_pnl = df_trades['pnl'].cumsum()
    running_max = cumulative_pnl.expanding().max()
    drawdown = cumulative_pnl - running_max
    max_drawdown = drawdown.min()
    
    # R-multiple stats
    avg_r_multiple = df_trades['r_multiple'].mean()
    
    # Duration stats
    avg_duration = df_trades['duration_min'].mean()
    
    # Partial exit stats
    trades_with_partials = len(df_trades[df_trades['partial_exits'] > 0])
    
    # Print results
    print(f"\nSIGNAL DETECTION:")
    print(f"  Total Signals Detected: {signals_detected}")
    print(f"  ML Approved: {signals_ml_approved} ({signals_ml_approved/signals_detected*100:.1f}%)")
    print(f"  ML Rejected: {signals_ml_rejected} ({signals_ml_rejected/signals_detected*100:.1f}%)")
    
    print(f"\nTRADING PERFORMANCE:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Winning Trades: {winning_trades}")
    print(f"  Losing Trades: {losing_trades}")
    print(f"  Win Rate: {win_rate:.1f}%")
    
    print(f"\nPROFIT & LOSS:")
    print(f"  Total P&L: ${total_pnl:+,.2f}")
    print(f"  Gross Profit: ${gross_profit:,.2f}")
    print(f"  Gross Loss: ${gross_loss:,.2f}")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Average Win: ${avg_win:+,.2f}")
    print(f"  Average Loss: ${avg_loss:+,.2f}")
    print(f"  Largest Win: ${largest_win:+,.2f}")
    print(f"  Largest Loss: ${largest_loss:+,.2f}")
    
    print(f"\nRISK METRICS:")
    print(f"  Max Drawdown: ${max_drawdown:+,.2f}")
    print(f"  Average R-Multiple: {avg_r_multiple:+.2f}R")
    
    print(f"\nTRADE CHARACTERISTICS:")
    print(f"  Average Duration: {avg_duration:.0f} minutes")
    print(f"  Trades with Partials: {trades_with_partials} ({trades_with_partials/total_trades*100:.1f}%)")
    
    # Exit reason breakdown
    print(f"\n🚪 EXIT REASONS:")
    exit_counts = df_trades['exit_reason'].value_counts()
    for reason, count in exit_counts.items():
        print(f"  {reason}: {count} ({count/total_trades*100:.1f}%)")
    
    # Show first 5 and last 5 trades
    print(f"\n📋 SAMPLE TRADES:")
    print(f"\nFirst 5 trades:")
    for idx, trade in df_trades.head(5).iterrows():
        print(f"  {trade['entry_time'].strftime('%m/%d %H:%M')} {trade['side'].upper():<5} "
              f"${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} | "
              f"P&L: ${trade['pnl']:+7.2f} | {trade['exit_reason']:<20} | "
              f"Conf: {trade['confidence']:.0%} | Partials: {trade['partial_exits']}")
    
    print(f"\nLast 5 trades:")
    for idx, trade in df_trades.tail(5).iterrows():
        print(f"  {trade['entry_time'].strftime('%m/%d %H:%M')} {trade['side'].upper():<5} "
              f"${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} | "
              f"P&L: ${trade['pnl']:+7.2f} | {trade['exit_reason']:<20} | "
              f"Conf: {trade['confidence']:.0%} | Partials: {trade['partial_exits']}")
    
    print("\n" + "=" * 80)
    
    # Save trades to CSV for analysis
    df_trades.to_csv('data/backtest_trades.csv', index=False)
    print(f"\n[OK] Full trade log saved to: data/backtest_trades.csv")
    
    return df_trades


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    csv_file = "data/historical_data/ES_1min.csv"
    
    if not os.path.exists(csv_file):
        print(f"ERROR: Data file not found: {csv_file}")
        print("Please ensure historical data is available.")
    else:
        df_trades = run_full_backtest(csv_file, days=15)
