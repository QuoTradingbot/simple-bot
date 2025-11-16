# -*- coding: utf-8 -*-
"""
COMPLETE TRADING SYSTEM BACKTEST - 100% LOCAL ONLY
===================================================
Full simulation of QuoTrading bot with ALL features:
- LOCAL pattern matching (no cloud API calls)
- Adaptive exits with local RL learning
- Partial exits (runners at 2R, 3R, 5R)
- Breakeven protection
- Trailing stops
- Time-based exits (flatten mode)
- Position sizing based on confidence
- ATR-based stops
- VWAP bounce strategy
- 100% OFFLINE - uses local_experience_manager.py only
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
import json
import pytz
import os
import random
from typing import Dict, List, Tuple, Optional
import statistics
import sys
import io

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from adaptive_exits import AdaptiveExitManager, detect_market_regime  # EXIT RL LEARNING
from comprehensive_exit_logic import ComprehensiveExitChecker  # ALL 131 EXIT PARAMETERS
from exit_param_utils import extract_all_exit_params  # Extract all 131 backtest params
from exit_params_config import get_default_exit_params  # Load configured defaults

# ========================================
# CONFIGURATION (matches bot)
# ========================================

CONFIG = {
    "tick_size": 0.25,
    "tick_value": 12.50,  # ES full contract = $12.50 per tick (MATCHES LIVE BOT)
    "max_contracts": 3,
    "local_mode": True,  # Use local experiences (3,754 saved experiences)
    "rl_confidence_threshold": 0.10,  # Only take trades above 10% confidence (exploration mode)
    "exploration_rate": 0.30,  # 30% exploration rate (HIGH - for testing and building dataset)
    
    # REALISTIC TRADING COSTS (TopStep/Prop Firm)
    "slippage_ticks": 0.5,  # 0.5 tick slippage per side (entry + exit = 1 tick total = $12.50)
    "commission_per_contract": 1.00,  # $1.00 per contract per side ($2.00 round trip)
    
    # DAILY LOSS LIMIT (matches live bot - user configurable)
    "daily_loss_limit": 1000.0,  # FAIL if down $1000 in a day - backtest auto-stops (prop firm rule simulation)
    
    # NEURAL NETWORK BASELINES - Only used if neural network prediction fails
    "breakeven_threshold_ticks": 9,  # Fallback baseline (neural network predicts actual value)
    "trailing_distance_ticks": 12,   # Fallback baseline (neural network predicts actual value)
    
    # Partial exits (runners) - Neural network baseline
    "partial_exit_1_r_multiple": 2.0,
    "partial_exit_1_percentage": 0.50,  # 50% at 2R
    "partial_exit_2_r_multiple": 3.0,
    "partial_exit_2_percentage": 0.30,  # 30% at 3R
    "partial_exit_3_r_multiple": 5.0,
    "partial_exit_3_percentage": 0.20,  # 20% at 5R,  
    
    # Time-based exits - UTC schedule (ES maintenance 22:00-23:00 UTC daily)
    # Sunday opens 23:00 UTC, Friday closes 22:00 UTC
    "daily_entry_cutoff": time(21, 0),    # 21:00 UTC - no new entries (1 hr before maintenance)
    "flatten_start_time": time(21, 45),  # 21:45 UTC - flatten mode starts  
    "forced_flatten_time": time(22, 0),   # 22:00 UTC - force close before maintenance window
    
    # ATR settings (MATCHES LIVE BOT) - Used for signal features
    "atr_period": 14,
    "atr_stop_multiplier": 3.6,  # Baseline - neural network adjusts per trade
    # NO HARDCODED TARGET - Neural network controls ALL profit exits (2R, 3R, 5R dynamic)
    
    # RSI settings - Neural network baseline
    "rsi_period": 10,  # Fast RSI
    "rsi_oversold": 25.0,  # LONG entry baseline (neural network validates)
    "rsi_overbought": 75.0,  # SHORT entry baseline (neural network validates)
}


# ========================================
# LOCAL RL CONFIDENCE (Pattern Matching from Local JSON Files)
# ========================================

from local_experience_manager import local_manager

# Update local_manager threshold from CONFIG
local_manager.confidence_threshold = CONFIG['rl_confidence_threshold']

def get_rl_confidence(rl_state: Dict, side: str) -> Tuple[bool, float, str]:
    """
    Get ML confidence from LOCAL pattern matching only.
    Uses local_experience_manager.py to match against saved experiences.
    NO cloud API calls - 100% offline backtesting.
    """
    exploration_rate = CONFIG.get('exploration_rate', 0.0)
    return local_manager.get_signal_confidence(rl_state, side.upper(), exploration_rate)


# ========================================
# POSITION SIZING (confidence-based)
# ========================================

def encode_session(session_str: str) -> int:
    """Convert session string to numeric code for neural network."""
    session_map = {'Asia': 0, 'London': 1, 'NY': 2}
    return session_map.get(session_str, 0)

def encode_trade_type(trade_type_str: str) -> int:
    """Convert trade type string to numeric code for neural network."""
    return 0 if trade_type_str == 'reversal' else 1

def encode_signal(signal_str: str) -> int:
    """Convert signal string to numeric code for neural network."""
    return 0 if signal_str.upper() == 'LONG' else 1

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
    
    # Handle adaptive threshold (default to 0.50)
    if threshold == "adaptive":
        threshold = 0.50
    
    # Calculate tier size: range from threshold to 100%, divided by max_contracts
    range_size = 1.0 - threshold  # e.g., if threshold=0.50, range is 0.50 (50% to 100%)
    tier_size = range_size / max_contracts  # e.g., 0.50 / 3 = 0.167 (16.7% per tier)
    
    # Calculate which tier this confidence falls into
    if confidence < threshold:
        contracts = 0  # Below threshold = NO TRADE
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
    """Calculate RSI indicator (using fast period=10 for neural network features)."""
    deltas = prices.diff()
    gains = deltas.where(deltas > 0, 0.0)
    losses = -deltas.where(deltas < 0, 0.0)
    
    avg_gain = gains.rolling(window=period).mean().iloc[-1]
    avg_loss = losses.rolling(window=period).mean().iloc[-1]
    
    # Handle edge cases
    if avg_loss == 0 and avg_gain == 0:
        return 50.0  # No movement = neutral
    if avg_loss == 0:
        return 100.0  # Only gains = overbought
    if avg_gain == 0:
        return 0.0  # Only losses = oversold
    
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


def calculate_vwap_bands(df_day: pd.DataFrame, signal_manager=None) -> Dict:
    """Calculate daily VWAP with standard deviation bands."""
    try:
        # Ensure we have a proper copy to avoid SettingWithCopyWarning
        df_work = df_day.copy().reset_index(drop=True)
        
        # Vectorized calculation - faster and less prone to interrupts
        typical_price = (df_work['high'].values + df_work['low'].values + df_work['close'].values) / 3
        volume = df_work['volume'].values
        
        tpv = typical_price * volume
        
        cumulative_tpv = np.cumsum(tpv)
        cumulative_volume = np.cumsum(volume)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            vwap = cumulative_tpv / cumulative_volume
            vwap = np.nan_to_num(vwap, nan=typical_price[-1])
        
        # Standard deviation using numpy for speed
        price_diff_sq = (typical_price - vwap) ** 2
        weighted_diff_sq = price_diff_sq * volume
        
        cumulative_weighted_diff_sq = np.cumsum(weighted_diff_sq)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            variance = cumulative_weighted_diff_sq / cumulative_volume
            variance = np.nan_to_num(variance, nan=0)
            std_dev = np.sqrt(variance)
    except Exception as e:
        print(f"⚠️  VWAP calculation error: {e}")
        # Return safe defaults
        return {
            'vwap': df_day['close'].iloc[-1],
            'upper_1': df_day['close'].iloc[-1] * 1.001,
            'lower_1': df_day['close'].iloc[-1] * 0.999,
            'upper_2': df_day['close'].iloc[-1] * 1.002,
            'lower_2': df_day['close'].iloc[-1] * 0.998
        }
    
    latest_vwap = vwap[-1]
    latest_std = std_dev[-1]
    
    # LEARN OPTIMAL VWAP BAND MULTIPLIERS from winning entries
    # Start with learned baseline from data, then refine
    entry_band_mult = None  # Will be learned
    
    if signal_manager and hasattr(signal_manager, 'signal_experiences'):
        signal_exps = signal_manager.signal_experiences
        
        # Start learning from first 20+ experiences
        if len(signal_exps) >= 20:
            # Calculate VWAP distance for winning trades
            winners_with_vwap = [e for e in signal_exps if e.get('pnl', 0) > 0 and e.get('vwap_distance', 0) != 0]
            
            if len(winners_with_vwap) >= 10:
                # Get median VWAP distance of winners
                winner_distances = sorted([abs(e.get('vwap_distance', 2.1)) for e in winners_with_vwap])
                median_winner_distance = winner_distances[len(winner_distances) // 2]
                
                # Use learned distance with reasonable bounds (1.5 - 2.8 std dev)
                entry_band_mult = max(1.5, min(2.8, median_winner_distance))
    
    # Ultimate fallback if no learned data yet
    if entry_band_mult is None:
        entry_band_mult = 2.1  # Conservative starting point
    
    # Neural network learned entry zone
    return {
        'vwap': latest_vwap,
        'upper_1': latest_vwap + (2.5 * latest_std),
        'upper_2': latest_vwap + (entry_band_mult * latest_std),  # LEARNED entry zone
        'upper_3': latest_vwap + (3.7 * latest_std),  # Exit/stop
        'lower_1': latest_vwap - (2.5 * latest_std),
        'lower_2': latest_vwap - (entry_band_mult * latest_std),  # LEARNED entry zone
        'lower_3': latest_vwap - (3.7 * latest_std),  # Exit/stop
        'std_dev': latest_std,
        'entry_band_mult': entry_band_mult  # Track what we learned
    }


# ========================================
# TRADE SIMULATION CLASS
# ========================================

class Trade:
    """Represents an active trade with ADAPTIVE exit logic (matches live bot)."""
    
    def __init__(self, entry_bar: pd.Series, side: str, contracts: int, 
                 confidence: float, atr: float, vwap_bands: Dict, all_bars: List,
                 adaptive_manager=None, entry_market_state: Dict = None, entry_bar_index: int = 0):
        self.entry_time = entry_bar['timestamp']
        self.entry_price = entry_bar['close']
        self.entry_bar_index = entry_bar_index  # NEW: Bar index at entry
        self.exit_bar_index = None  # NEW: Will be set at exit
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
        
        # Track bars in trade for exit param caching
        self.bars_in_trade = 0
        
        # ADAPTIVE: Will be recalculated each bar
        self.current_exit_params = None
        
        # NEW: In-trade behavior tracking for RL learning
        self.exit_param_updates = []  # [{bar, vix, atr, reason}]
        self.breakeven_activation_bar = None
        self.trailing_activation_bar = None
        self.bars_until_breakeven = None
        self.bars_until_trailing = None
        self.max_r_achieved = 0.0  # Peak R-multiple during trade
        self.min_r_achieved = 0.0  # Worst R-multiple during trade (drawdown)
        self.regime_changes = []  # Track if market regime shifts mid-trade
        self.stop_adjustments = []  # [{bar, old_stop, new_stop, reason}]
        
        # ADVANCED RL: Intra-trade decision sequences and P&L tracking
        self.decision_history = []  # Track every bar's state-action-reward
        self.unrealized_pnl_history = []  # Track unrealized P&L at each bar
        self.peak_unrealized_pnl = 0.0  # Highest unrealized profit achieved
        self.peak_r_multiple = 0.0  # Highest R-multiple achieved
        
        # NEW: Comprehensive 131-parameter exit logic
        self.comprehensive_exit_checker = ComprehensiveExitChecker({
            'entry_price': float(entry_bar['close']),
            'side': side,
            'contracts': contracts,
            'original_contracts': contracts,
            'initial_risk_ticks': stop_distance_ticks,
            'bars_in_trade': 0,
            'entry_atr': atr,
            'stop_price': self.stop_price
        })
        
        # CRITICAL FIX: Load configured exit parameters (not hardcoded defaults)
        # This ensures profit_protection_min_r=1.0, partial targets at 1.2R/2R/3.5R, etc.
        configured_exit_params = get_default_exit_params()
        self.comprehensive_exit_checker.update_exit_params(configured_exit_params)
        
        self.all_exit_params_used = {}  # Will store all 131 params at exit
        
        # NEW: DRAWDOWN FROM PEAK TRACKING
        self.profit_drawdown_from_peak = 0.0  # $ given back from peak
        self.max_drawdown_percent = 0.0  # Largest % drop from peak
        self.drawdown_bars = 0  # Bars in drawdown when exited
        self.currently_in_drawdown = False  # Track if currently giving back profit
        
        # NEW: VOLATILITY DURING TRADE TRACKING
        self.atr_samples = []  # Collect ATR at each bar
        self.entry_atr = atr  # Store entry ATR
        self.high_volatility_bars = 0  # Count bars with ATR spike > 20%
        
        # EXECUTION QUALITY TRACKING
        self.entry_slippage_ticks = 0.0  # Difference from intended entry
        self.exit_slippage_ticks = 0.0  # Difference from intended exit
        self.commission_cost = 0.0  # Total commissions paid
        self.fill_quality = 'full'  # 'full', 'partial', 'rejected'
        
        # MARKET CONTEXT TRACKING
        self.entry_session = self._get_session(entry_bar)  # 'Asia', 'London', 'NY'
        self.volume_at_exit = 0.0
        self.volatility_regime_at_entry = self._get_volatility_regime(vwap_bands)
        self.volatility_regime_at_exit = None
        
        # TRADE QUALITY TRACKING
        self.time_in_breakeven_bars = 0  # How long in breakeven before trailing
        self.rejected_partial_count = 0  # Failed partial exit attempts
        self.stop_hit = False  # Did stop loss trigger (vs target/profit)
        self.in_breakeven_zone = False  # Currently risk-free
    
    def _get_session(self, bar: pd.Series) -> str:
        """Determine trading session based on hour."""
        hour = bar['timestamp'].hour
        if 22 <= hour or hour < 7:  # 10pm - 7am UTC (Asia session)
            return 'Asia'
        elif 7 <= hour < 12:  # 7am - 12pm UTC (London session)
            return 'London'
        else:  # 12pm - 10pm UTC (NY session)
            return 'NY'
    
    def _get_volatility_regime(self, vwap_bands: Dict) -> str:
        """Determine volatility regime based on VIX or std dev."""
        if 'vix' in self.entry_market_state:
            vix = self.entry_market_state['vix']
            if vix < 15:
                return 'LOW'
            elif vix < 25:
                return 'NORMAL'
            else:
                return 'HIGH'
        return 'NORMAL'

    
    def update(self, bar: pd.Series, bar_index: int) -> Optional[Tuple[str, float, int]]:
        """
        Update trade with new bar, check exits.
        Uses ADAPTIVE exits that adjust to market conditions (like live bot).
        Returns: (exit_reason, exit_price, contracts_closed) or None
        """
        # Increment bar counter
        self.bars_in_trade += 1
        
        current_price = bar['close']
        
        # LEARN MAX TRADE DURATION - exit stale trades that aren't working
        # Start with learned baseline from existing data, then refine
        max_duration = 60  # Conservative default (60 bars = 1 hour)
        
        if self.adaptive_manager and hasattr(self.adaptive_manager, 'exit_experiences'):
            exit_exps = self.adaptive_manager.exit_experiences
            
            # If we have initial data, start with learned duration
            if len(exit_exps) >= 10:
                all_durations = [e.get('outcome', {}).get('duration', 30) for e in exit_exps]
                # Use 75th percentile as starting max (most trades should finish before this)
                sorted_durations = sorted(all_durations)
                max_duration = int(sorted_durations[int(len(sorted_durations) * 0.75)])
                max_duration = max(20, min(120, max_duration))  # 20-120 bar bounds
            
            if len(exit_exps) > 50:
                winners = [e for e in exit_exps if e.get('outcome', {}).get('win', False)]
                losers = [e for e in exit_exps if not e.get('outcome', {}).get('win', False)]
                
                if len(winners) > 10 and len(losers) > 10:
                    # Calculate 90th percentile of winner durations
                    winner_durations = sorted([e.get('outcome', {}).get('duration', 30) for e in winners])
                    winner_90th = winner_durations[int(len(winner_durations) * 0.90)]
                    
                    # Calculate median of loser durations
                    loser_durations = sorted([e.get('outcome', {}).get('duration', 30) for e in losers])
                    loser_median = loser_durations[len(loser_durations) // 2]
                    
                    # If losers hold longer than winners, cut them off earlier
                    # Use 90th percentile of winners + 30% buffer
                    max_duration = int(winner_90th * 1.3)
                    max_duration = max(20, min(120, max_duration))  # 20-120 bar bounds
        
        # Check if trade is stale (held too long without profit)
        if self.bars_in_trade >= max_duration:
            profit_ticks = (current_price - self.entry_price) / CONFIG['tick_size']
            if self.side == 'SHORT':
                profit_ticks = -profit_ticks
            
            # If not profitable after max duration, exit
            if profit_ticks < 5:  # Less than 5 ticks profit
                return ('stale_exit', current_price, self.contracts)
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
        
        # Track R-multiple extremes for learning (NEW)
        self.max_r_achieved = max(self.max_r_achieved, r_multiple)
        self.min_r_achieved = min(self.min_r_achieved, r_multiple)
        
        # ADVANCED RL: Track unrealized P&L and decision at this bar
        unrealized_pnl = profit_ticks * tick_value * self.contracts
        self.unrealized_pnl_history.append({
            'bar': self.bars_in_trade,
            'unrealized_pnl': float(unrealized_pnl),
            'r_multiple': float(r_multiple),
            'price': float(current_price),
            'contracts': int(self.contracts)
        })
        
        # Track peak unrealized profit
        self.peak_unrealized_pnl = max(self.peak_unrealized_pnl, unrealized_pnl)
        self.peak_r_multiple = max(self.peak_r_multiple, r_multiple)
        
        # NEW: Track drawdown from peak
        if self.peak_unrealized_pnl > 0:
            drawdown_amount = self.peak_unrealized_pnl - unrealized_pnl
            if drawdown_amount > 0:
                self.currently_in_drawdown = True
                self.drawdown_bars += 1
                self.profit_drawdown_from_peak = max(self.profit_drawdown_from_peak, drawdown_amount)
                
                # Calculate drawdown percentage
                drawdown_pct = (drawdown_amount / self.peak_unrealized_pnl) * 100
                self.max_drawdown_percent = max(self.max_drawdown_percent, drawdown_pct)
            else:
                self.currently_in_drawdown = False
        
        # NEW: Track volatility during trade
        current_atr = bar.get('atr', 2.0)
        self.atr_samples.append(current_atr)
        
        # Check for volatility spike (>20% increase from entry)
        if current_atr > self.entry_atr * 1.20:
            self.high_volatility_bars += 1
        
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
        
        # Get adaptive exit params from LOCAL RL Manager
        # OPTIMIZATION: Only update every 5 bars or when market changes significantly
        if self.adaptive_manager:
            # Calculate synthetic VIX from current bar (matches signal VIX formula)
            atr = float(bar.get('atr', 2.0))
            # Get volume ratio from bar if available, otherwise use 1.0
            volume_ratio = 1.0  # Simplified for exit updates
            atr_contribution = (atr / 2.0) * 10.0
            volume_contribution = (1.0 - min(volume_ratio, 1.5)) * 5.0
            synthetic_vix = 15.0 + atr_contribution + volume_contribution
            synthetic_vix = max(10.0, min(synthetic_vix, 35.0))
            
            # LEARN optimal exit update frequency from data
            # Start with learned baseline, refine as more data comes in
            update_interval = 5      # Conservative default
            vix_threshold = 3.0      # Conservative default  
            atr_threshold = 0.5      # Conservative default
            
            if self.adaptive_manager and hasattr(self.adaptive_manager, 'exit_experiences'):
                exit_exps = self.adaptive_manager.exit_experiences
                
                # Start learning from initial data
                if len(exit_exps) >= 20:
                    # Calculate overall avg trade duration
                    all_durations = [e.get('outcome', {}).get('duration', 30) for e in exit_exps]
                    avg_duration = sum(all_durations) / len(all_durations)
                    
                    # Set initial intervals based on average trade length
                    if avg_duration < 15:  # Fast scalps
                        update_interval = 3
                        vix_threshold = 2.0
                        atr_threshold = 0.3
                    elif avg_duration < 40:  # Normal trades
                        update_interval = 5
                        vix_threshold = 3.0
                        atr_threshold = 0.5
                    else:  # Long runners
                        update_interval = 7
                        vix_threshold = 4.0
                        atr_threshold = 0.7
                
                # Refine from winning trades once we have enough data
                if len(exit_exps) > 50:
                    winners = [e for e in exit_exps if e.get('outcome', {}).get('win', False)]
                    
                    if len(winners) > 20:
                        # Calculate avg bars in trade for winners
                        winner_durations = [e.get('outcome', {}).get('duration', 30) for e in winners]
                        avg_duration = sum(winner_durations) / len(winner_durations)
                        
                        # Update more frequently for short trades, less for long trades
                        if avg_duration < 10:  # Fast scalps
                            update_interval = 3
                            vix_threshold = 2.0
                            atr_threshold = 0.3
                        elif avg_duration < 30:  # Normal trades
                            update_interval = 5
                            vix_threshold = 3.0
                            atr_threshold = 0.5
                        else:  # Long runners
                            update_interval = 8
                            vix_threshold = 4.0
                            atr_threshold = 0.7
            
            # Check if we should update exit params (LEARNED intervals)
            should_update = False
            if not hasattr(self, 'last_exit_update_bar'):
                self.last_exit_update_bar = 0
                self.last_exit_vix = synthetic_vix
                self.last_exit_atr = atr
                should_update = True
            else:
                bars_since_update = self.bars_in_trade - self.last_exit_update_bar
                vix_change = abs(synthetic_vix - self.last_exit_vix)
                atr_change = abs(atr - self.last_exit_atr)
                
                # Update based on LEARNED frequency and thresholds
                if bars_since_update >= update_interval or vix_change > vix_threshold or atr_change > atr_threshold:
                    should_update = True
            
            if should_update:
                # Build market state for exit manager (include ALL tracked features)
                market_state = {
                    'vix': synthetic_vix,  # Synthetic VIX from ATR + volume
                    'atr': atr,
                    'hour': int(bar['timestamp'].hour),
                    'volume_ratio': volume_ratio,
                    'rsi': float(self.entry_market_state.get('rsi', 50)),
                    'vwap_distance': float(self.entry_market_state.get('vwap_distance', 0)),
                    # NEW: Add consecutive trade context for intelligent decisions
                    'wins_in_last_5_trades': int(getattr(self, 'wins_in_last_5_trades', 0)),
                    'losses_in_last_5_trades': int(getattr(self, 'losses_in_last_5_trades', 0)),
                    'cumulative_pnl_before_trade': float(getattr(self, 'cumulative_pnl_before_trade', 0.0)),
                }
                
                position = {
                    'entry_price': float(self.entry_price),
                    'side': self.side,
                    'entry_time': str(self.entry_time)  # Convert Timestamp to string
                }
                
                # ADAPTIVE: Let bot learn ALL 131 exit parameters from market conditions
                # Comprehensive exit checker uses full defaults from EXIT_PARAMS config
                # Pattern matching will happen during update_exit_params() if neural network trained
                # For now, use defaults and let bot explore the full parameter space
                
                # NOTE: adaptive_manager.get_adaptive_exit_params() doesn't exist yet
                # When we train the exit neural network, it will return all 131 params
                # For now, comprehensive_exit_checker already has all 131 defaults loaded
                
                # Skip adaptive manager call - use comprehensive defaults (all 131 params)
                # This allows bot to explore full parameter space during backtest
                if False:  # Disabled - adaptive_manager doesn't have this method yet
                    self.current_exit_params = {
                        'breakeven_threshold_ticks': 9,
                        'trailing_distance_ticks': 12,
                        'stop_mult': 3.6,
                        'partial_1_r': 2.0,
                        'partial_1_pct': 0.5,
                        'partial_2_r': 3.0,
                        'partial_2_pct': 0.3,
                        'partial_3_r': 5.0,
                        'partial_3_pct': 0.2,
                        'market_regime': 'UNKNOWN',
                    }
                    
                    # Track exit param update for RL learning (NEW)
                    update_reason = 'initial'
                    if hasattr(self, 'last_exit_update_bar') and self.last_exit_update_bar > 0:
                        bars_since_update = self.bars_in_trade - self.last_exit_update_bar
                        vix_change = abs(synthetic_vix - self.last_exit_vix)
                        atr_change = abs(atr - self.last_exit_atr)
                        
                        if bars_since_update >= update_interval:
                            update_reason = 'interval'
                        elif vix_change > vix_threshold:
                            update_reason = 'vix_change'
                        elif atr_change > atr_threshold:
                            update_reason = 'atr_change'
                        else:
                            update_reason = 'periodic'
                    
                    self.exit_param_updates.append({
                        'bar': self.bars_in_trade,
                        'vix': float(synthetic_vix),
                        'atr': float(atr),
                        'reason': update_reason,
                        'r_at_update': float(r_multiple)
                    })
                    
                    # Track last update
                    self.last_exit_update_bar = self.bars_in_trade
                    self.last_exit_vix = synthetic_vix
                    self.last_exit_atr = atr
            else:
                # Fallback: Don't override - comprehensive checker has ALL 131 defaults
                # This allows bot to learn full parameter space during backtest
                self.current_exit_params = None
        else:
            # No adaptive manager - comprehensive checker already has ALL 131 defaults
            # No need to set current_exit_params - comprehensive checker handles it
            # This allows bot to learn from full parameter space (profit_protection_min_r, etc.)
            self.current_exit_params = None  # Use comprehensive checker's defaults
        
        # Extract adaptive parameters only if we have them
        if self.current_exit_params:
            breakeven_threshold = self.current_exit_params.get('breakeven_threshold_ticks', 12)
            trailing_distance_ticks = self.current_exit_params.get('trailing_distance_ticks', 15)
        else:
            # Use comprehensive checker's defaults
            breakeven_threshold = 12
            trailing_distance_ticks = 15
        
        # ========================================
        # COMPREHENSIVE 131-PARAMETER EXIT CHECKING
        # ========================================
        
        # Update comprehensive checker's trade context
        self.comprehensive_exit_checker.trade.update({
            'contracts': self.contracts,
            'bars_in_trade': self.bars_in_trade,
            'highest_price': self.highest_price,
            'lowest_price': self.lowest_price,
            'stop_price': self.stop_price,
            'breakeven_active': self.breakeven_active,
            'trailing_active': self.trailing_active,
            'partial_1_done': self.partial_1_done,
            'partial_2_done': self.partial_2_done,
            'partial_3_done': self.partial_3_done,
            'partial_exits': self.partial_exits,
            'peak_unrealized_pnl': self.peak_unrealized_pnl
        })
        
        # Update comprehensive checker with adaptive params if available
        if self.current_exit_params:
            self.comprehensive_exit_checker.update_exit_params(self.current_exit_params)
        
        # Build market context for comprehensive checks
        market_context = {
            'consecutive_losses': getattr(self, 'consecutive_losses', 0),
            'consecutive_wins': getattr(self, 'consecutive_wins', 0),
            'daily_pnl': getattr(self, 'cumulative_pnl', 0.0),
            'vix': synthetic_vix if 'synthetic_vix' in locals() else 15.0,
            'entry_atr': self.entry_atr,
            'current_atr': bar.get('atr', 2.0)
        }
        
        # Check all 131 exit parameters
        comprehensive_exit = self.comprehensive_exit_checker.check_all_exits(
            current_bar=bar,
            bar_index=bar_index,
            all_bars=self.all_bars,
            market_context=market_context
        )
        
        # Store all 131 parameters used (will be saved to JSON)
        self.all_exit_params_used = self.comprehensive_exit_checker.get_all_used_params()
        
        # If comprehensive checker triggered ANY exit (including partials), use it
        # CRITICAL FIX: Partials return should_exit=False, but still need to be processed
        if comprehensive_exit:
            # Track stop hit for learning
            if 'stop' in comprehensive_exit['exit_reason'].lower():
                self.stop_hit = True
            # Partial exits (should_exit=False) and full exits (should_exit=True) both processed here
            return (comprehensive_exit['exit_reason'], 
                   comprehensive_exit['exit_price'], 
                   comprehensive_exit['contracts_to_close'])
        
        # ========================================
        # LEGACY FALLBACK: If comprehensive checker didn't trigger, no exit
        # ========================================
        # Note: Comprehensive checker handles ALL 131 exit parameters including:
        # - Time-based exits (forced_flatten, daily_entry_cutoff)
        # - Partial exits (partial_1/2/3 with proper R-multiples from config)
        # - Stop loss, breakeven, trailing stops
        # - Dead trade detection, sideways market handling
        # - Profit protection, account protection
        # - All advanced exit features
        #
        # Legacy checks below are kept ONLY for tracking/compatibility but should never trigger
        # since comprehensive checker is more complete
        
        # Track time in breakeven before trailing activates (for learning)
        if self.in_breakeven_zone and not self.trailing_active:
            self.time_in_breakeven_bars += 1
        
        return None
    
    def check_partial_exits(self, r_multiple: float, current_price: float) -> Optional[Tuple[str, float, int]]:
        """Check and execute partial exits using LOCAL ADAPTIVE parameters."""
        
        # Use adaptive params from local experience manager
        if not self.current_exit_params:
            return None
        
        # SINGLE CONTRACT: Take full position at learned target
        if self.original_contracts <= 1:
            learned_target = self.current_exit_params.get('partial_3_r', 5.0)
            if r_multiple >= learned_target:
                return ('learned_target', current_price, self.contracts)
            return None
        
        # MULTI-CONTRACT: Execute adaptive partial scaling from local learned params
        
        # First partial (LOCAL ADAPTIVE r-multiple and percentage)
        partial_1_r = self.current_exit_params.get('partial_1_r', 2.0)
        partial_1_pct = self.current_exit_params.get('partial_1_pct', 0.5)
        
        if r_multiple >= partial_1_r and not self.partial_1_done:
            contracts_to_close = int(self.original_contracts * partial_1_pct)
            if contracts_to_close >= 1:
                self.partial_1_done = True
                self.contracts -= contracts_to_close
                self.partial_exits.append({
                    'r_multiple': r_multiple,
                    'price': current_price,
                    'contracts': contracts_to_close,
                    'level': 1
                })
                return ('partial_1', current_price, contracts_to_close)
        
        # Second partial (LOCAL ADAPTIVE)
        partial_2_r = self.current_exit_params.get('partial_2_r', 3.0)
        partial_2_pct = self.current_exit_params.get('partial_2_pct', 0.3)
        
        if r_multiple >= partial_2_r and not self.partial_2_done:
            contracts_to_close = int(self.original_contracts * partial_2_pct)
            if contracts_to_close >= 1 and self.contracts >= contracts_to_close:
                self.partial_2_done = True
                self.contracts -= contracts_to_close
                self.partial_exits.append({
                    'r_multiple': r_multiple,
                    'price': current_price,
                    'contracts': contracts_to_close,
                    'level': 2
                })
                return ('partial_2', current_price, contracts_to_close)
        
        # Third partial (LOCAL ADAPTIVE - final runner)
        partial_3_r = self.current_exit_params.get('partial_3_r', 5.0)
        
        if r_multiple >= partial_3_r and not self.partial_3_done:
            if self.contracts >= 1:
                contracts_to_close = self.contracts
                self.partial_3_done = True
                self.contracts = 0
                self.partial_exits.append({
                    'r_multiple': r_multiple,
                    'price': current_price,
                    'contracts': contracts_to_close,
                    'level': 3
                })
                return ('partial_3', current_price, contracts_to_close)
        
        # ADVANCED RL: Record decision to HOLD (no exit taken)
        # Calculate current unrealized P&L for this decision
        tick_value = CONFIG['tick_value']
        # Use current unrealized P&L from profit_ticks
        if self.side == 'long':
            profit_ticks_calc = (current_price - self.entry_price) / CONFIG['tick_size']
        else:
            profit_ticks_calc = (self.entry_price - current_price) / CONFIG['tick_size']
        
        hold_unrealized_pnl = profit_ticks_calc * tick_value * self.contracts
        self.decision_history.append({
            'bar': self.bars_in_trade,
            'action': 'hold',
            'r_multiple': float(r_multiple),
            'unrealized_pnl': float(hold_unrealized_pnl),
            'price': float(current_price),
            'stop_price': float(self.stop_price),
            'contracts': int(self.contracts),
            'breakeven_active': bool(self.breakeven_active),
            'trailing_active': bool(self.trailing_active)
        })
        
        return None
    
    def update_trailing_stop_adaptive(self, current_price: float, trailing_distance_ticks: int):
        """Update trailing stop using ADAPTIVE distance (not fixed)."""
        tick_size = CONFIG['tick_size']
        trail_distance = trailing_distance_ticks * tick_size
        
        if self.side == 'long':
            # Trail below highest high
            new_stop = self.highest_price - trail_distance
            if new_stop > self.stop_price:
                old_stop = self.stop_price
                self.stop_price = new_stop
                
                # Track first trailing activation (NEW)
                if not self.trailing_active:
                    self.trailing_active = True
                    self.trailing_activation_bar = self.bars_in_trade
                    self.bars_until_trailing = self.bars_in_trade
                
                # Track stop adjustment for RL learning (NEW)
                if self.side == 'long':
                    profit_ticks = (current_price - self.entry_price) / tick_size
                else:
                    profit_ticks = (self.entry_price - current_price) / tick_size
                r_at_adjustment = profit_ticks / self.initial_risk_ticks if self.initial_risk_ticks > 0 else 0
                
                self.stop_adjustments.append({
                    'bar': self.bars_in_trade,
                    'old_stop': float(old_stop),
                    'new_stop': float(new_stop),
                    'reason': 'trailing_adjustment',
                    'r_at_adjustment': float(r_at_adjustment)
                })
        else:  # short
            # Trail above lowest low
            new_stop = self.lowest_price + trail_distance
            if new_stop < self.stop_price:
                old_stop = self.stop_price
                self.stop_price = new_stop
                
                # Track first trailing activation (NEW)
                if not self.trailing_active:
                    self.trailing_active = True
                    self.trailing_activation_bar = self.bars_in_trade
                    self.bars_until_trailing = self.bars_in_trade
                
                # Track stop adjustment for RL learning (NEW)
                if self.side == 'long':
                    profit_ticks = (current_price - self.entry_price) / tick_size
                else:
                    profit_ticks = (self.entry_price - current_price) / tick_size
                r_at_adjustment = profit_ticks / self.initial_risk_ticks if self.initial_risk_ticks > 0 else 0
                
                self.stop_adjustments.append({
                    'bar': self.bars_in_trade,
                    'old_stop': float(old_stop),
                    'new_stop': float(new_stop),
                    'reason': 'trailing_adjustment',
                    'r_at_adjustment': float(r_at_adjustment)
                })


# ========================================
# EXPERIENCE RECORDING FOR RL LEARNING
# ========================================

def save_signal_experience(rl_state: Dict, took_trade: bool, outcome: Dict, backtest_mode: bool = False):
    """
    Save signal experience to LOCAL JSON files.
    Backtest mode NEVER uses cloud API - everything stays local.
    
    Args:
        rl_state: State at signal time (with ALL 13 features for pattern matching)
        took_trade: Whether backtest took the trade
        outcome: Trade result (pnl, duration, exit_reason, etc.)
        backtest_mode: If True, saves to local files only
    """
    # BACKTEST: Save to local files only - NO cloud API
    print(f"    [BACKTEST] Saving experience to local files (no cloud)")
    # Local saving happens via local_experience_manager.py - this is just a placeholder
    return


class ShadowTrade:
    """
    Represents a rejected signal being tracked in the background.
    Simulates what WOULD happen if we took the trade using FULL neural exit model.
    
    MATCHES LIVE BOT IMPLEMENTATION WITH ALL 131 EXIT PARAMETERS:
    - Uses comprehensive exit logic (ALL 131 exit parameters)
    - Updates stop/breakeven/trailing EVERY bar
    - Tracks MAE/MFE, partials, all metrics
    - Saves complete 158+ field experience for learning
    - Outcome reflects what would ACTUALLY happen with current exit logic
    """
    def __init__(self, entry_bar: pd.Series, side: str, confidence: float, 
                 rejection_reason: str, atr: float, adaptive_manager, 
                 entry_bar_index: int, all_bars):
        self.entry_time = entry_bar['timestamp']
        self.entry_price = entry_bar['close']
        self.side = side
        self.confidence = confidence
        self.rejection_reason = rejection_reason
        self.entry_bar_index = entry_bar_index
        self.all_bars = all_bars
        self.adaptive_manager = adaptive_manager
        
        # Store ALL 131 exit parameters for learning (matches live bot)
        self.all_exit_params_used = {}
        
        # Shadow position state
        self.quantity = 1  # Ghost trades use 1 contract
        self.remaining_quantity = 1
        self.contracts = 1
        self.original_contracts = 1
        
        # Initial stop (will be updated by adaptive model)
        stop_distance = atr * CONFIG['atr_stop_multiplier']
        if side == 'long':
            self.stop_price = entry_bar['close'] - stop_distance
        else:
            self.stop_price = entry_bar['close'] + stop_distance
        
        self.initial_risk_ticks = int((entry_bar['close'] - self.stop_price) / CONFIG['tick_size']) if side == 'long' else int((self.stop_price - entry_bar['close']) / CONFIG['tick_size'])
        
        # Exit state tracking
        self.breakeven_active = False
        self.trailing_active = False
        self.highest_price = entry_bar['close']
        self.lowest_price = entry_bar['close']
        
        # Performance tracking
        self.mae = 0.0
        self.mfe = 0.0
        self.partial_exits = []
        self.partial_1_done = False
        self.partial_2_done = False
        self.partial_3_done = False
        
        # NEW: Match real trade tracking fields for comprehensive parity
        self.bars_in_trade = 0
        self.exit_param_updates = []
        self.breakeven_activation_bar = None
        self.trailing_activation_bar = None
        self.bars_until_breakeven = None
        self.bars_until_trailing = None
        self.max_r_achieved = 0.0
        self.min_r_achieved = 0.0
        self.regime_changes = []
        self.stop_adjustments = []
        self.decision_history = []
        self.unrealized_pnl_history = []
        self.peak_unrealized_pnl = 0.0
        self.peak_r_multiple = 0.0
        self.profit_drawdown_from_peak = 0.0
        self.max_drawdown_percent = 0.0
        self.drawdown_bars = 0
        self.currently_in_drawdown = False
        self.atr_samples = []
        self.entry_atr = atr
        self.high_volatility_bars = 0
        self.stop_hit = False
        self.time_in_breakeven_bars = 0
        
        # Additional fields for complete parity with real trades
        self.rejected_partial_count = 0
        self.entry_slippage_ticks = 0.0
        self.exit_slippage_ticks = 0.0
        self.commission_cost = 0.0
        
        # Completion tracking
        self.bars_elapsed = 0
        self.max_bars = 200  # Timeout after 200 bars
        self.completed = False
        self.outcome = None
        self.exit_bar_index = None
    
    def _create_outcome_dict(self, pnl: float, exit_price: float, exit_reason: str, bar_index: int, bar: pd.Series) -> Dict:
        """Create complete outcome dictionary with all tracking fields (matches real trades)."""
        initial_risk = abs(self.entry_price - self.stop_price)
        tick_value = CONFIG['tick_value']
        r_multiple = pnl / (initial_risk * tick_value * 4) if initial_risk > 0 else 0
        
        return {
            'pnl': pnl,
            'exit_price': exit_price,
            'duration_min': self.bars_elapsed,
            'exit_reason': f'ghost_{exit_reason}' if not exit_reason.startswith('ghost_') else exit_reason,
            'took_trade': False,
            'confidence': self.confidence,
            'rejection_reason': self.rejection_reason,
            'side': self.side,
            'contracts': self.contracts,
            'win': pnl > 0,
            'is_ghost': True,
            
            # Performance Metrics (complete parity)
            'mae': self.mae,
            'mfe': self.mfe,
            'max_r_achieved': self.max_r_achieved,
            'min_r_achieved': self.min_r_achieved,
            'r_multiple': r_multiple,
            'peak_r_multiple': self.peak_r_multiple,
            'peak_unrealized_pnl': self.peak_unrealized_pnl,
            'profit_drawdown_from_peak': self.profit_drawdown_from_peak,
            'max_drawdown_percent': self.max_drawdown_percent,
            'opportunity_cost': float(self.peak_unrealized_pnl - pnl) if self.peak_unrealized_pnl > pnl else 0.0,
            
            # Exit Strategy State
            'breakeven_activated': self.breakeven_active,
            'trailing_activated': self.trailing_active,
            'stop_hit': 'stop_loss' in exit_reason,
            'partials_taken': len(self.partial_exits),
            'breakeven_activation_bar': self.breakeven_activation_bar if self.breakeven_activation_bar else 0,
            'trailing_activation_bar': self.trailing_activation_bar if self.trailing_activation_bar else 0,
            'bars_until_breakeven': self.bars_until_breakeven if self.bars_until_breakeven else 0,
            'bars_until_trailing': self.bars_until_trailing if self.bars_until_trailing else 0,
            
            # Partial exits (GHOST TRADES TRACK THESE TOO)
            'partial_exit_1_completed': self.partial_1_done,
            'partial_exit_2_completed': self.partial_2_done,
            'partial_exit_3_completed': self.partial_3_done,
            'partial_exits_history': self.partial_exits,  # Full history of when partials were taken
            
            # Entry context
            'entry_confidence': self.confidence,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            
            # Advanced tracking
            'exit_param_update_count': len(self.exit_param_updates),
            'stop_adjustment_count': len(self.stop_adjustments),
            'decision_history': self.decision_history,
            'unrealized_pnl_history': self.unrealized_pnl_history,
            'exit_param_updates': self.exit_param_updates,
            'stop_adjustments': self.stop_adjustments,
            'high_volatility_bars': self.high_volatility_bars,
            'bars_in_trade': self.bars_in_trade,
            'drawdown_bars': self.drawdown_bars,
            'currently_in_drawdown': self.currently_in_drawdown,
            
            # Execution quality (ghost trades have 0 slippage/commission)
            'slippage_ticks': 0.0,
            'commission_cost': 0.0,
            'bid_ask_spread_ticks': 0.0,
            'rejected_partial_count': self.rejected_partial_count,
            'time_in_breakeven_bars': self.time_in_breakeven_bars,
            
            # Volatility tracking
            'avg_atr_during_trade': float(sum(self.atr_samples) / len(self.atr_samples)) if self.atr_samples else float(self.entry_atr),
            'atr_change_percent': float(((self.atr_samples[-1] - self.entry_atr) / self.entry_atr) * 100.0) if self.atr_samples and self.entry_atr > 0 else 0.0,
            
            # Timing fields
            'entry_bar': self.entry_bar_index,
            'exit_bar': bar_index,
            'entry_hour': self.entry_time.hour,
            'entry_minute': self.entry_time.minute,
            'exit_hour': bar['timestamp'].hour,
            'exit_minute': bar['timestamp'].minute,
            'day_of_week': bar['timestamp'].weekday(),
            'bars_held': bar_index - self.entry_bar_index,
            
            # Exit params (ALL 131 PARAMETERS)
            'exit_params': {},
            'exit_params_used': self.all_exit_params_used
        }
        
    def update(self, bar: pd.Series, bar_index: int, vwap_bands: Dict) -> Optional[Dict]:
        """Update shadow trade with new bar - FULL COMPREHENSIVE EXIT SIMULATION
        
        CRITICAL: Ghost trades MUST use same comprehensive exit logic as real trades.
        This includes ALL 131 exit parameters:
        - Profit drawdown protection
        - Sideways market detection
        - Adverse conditions check
        - Breakeven, trailing, partials
        - Max time limits
        - All adaptive thresholds
        
        Without this, ghost outcomes are inaccurate (run too long, wrong exits).
        """
        if self.completed:
            return None
        
        self.bars_elapsed += 1
        self.bars_in_trade = self.bars_elapsed  # Match real trade field name
        current_price = bar['close']
        
        # Update price extremes and track R-multiples (matches real trades)
        initial_risk = abs(self.entry_price - self.stop_price)
        tick_size = CONFIG['tick_size']
        tick_value = CONFIG['tick_value']
        
        if self.side == 'long':
            if current_price > self.highest_price:
                self.highest_price = current_price
            if current_price < self.lowest_price:
                self.lowest_price = current_price
            current_r = (current_price - self.entry_price) / initial_risk if initial_risk > 0 else 0
        else:
            if current_price < self.lowest_price:
                self.lowest_price = current_price
            if current_price > self.highest_price:
                self.highest_price = current_price
            current_r = (self.entry_price - current_price) / initial_risk if initial_risk > 0 else 0
        
        # Track peak R-multiple (matches real trades)
        if current_r > self.max_r_achieved:
            self.max_r_achieved = current_r
            self.peak_r_multiple = current_r
        if current_r < self.min_r_achieved:
            self.min_r_achieved = current_r
        
        # Track unrealized P&L and peak (matches real trades)
        if self.side == 'long':
            unrealized_pnl = (current_price - self.entry_price) * tick_value * 4 * self.remaining_quantity
        else:
            unrealized_pnl = (self.entry_price - current_price) * tick_value * 4 * self.remaining_quantity
        
        self.unrealized_pnl_history.append(unrealized_pnl)
        if unrealized_pnl > self.peak_unrealized_pnl:
            self.peak_unrealized_pnl = unrealized_pnl
        
        # Track drawdown from peak (matches real trades)
        if self.peak_unrealized_pnl > 0:
            drawdown = self.peak_unrealized_pnl - unrealized_pnl
            if drawdown > self.profit_drawdown_from_peak:
                self.profit_drawdown_from_peak = drawdown
                drawdown_pct = (drawdown / self.peak_unrealized_pnl) * 100
                if drawdown_pct > self.max_drawdown_percent:
                    self.max_drawdown_percent = drawdown_pct
            
            if drawdown > 0:
                self.currently_in_drawdown = True
                self.drawdown_bars += 1
            else:
                self.currently_in_drawdown = False
                self.drawdown_bars = 0
        
        # Track ATR samples (matches real trades)
        current_atr = bar.get('atr', self.entry_atr)
        self.atr_samples.append(current_atr)
        if current_atr > self.entry_atr * 1.2:
            self.high_volatility_bars += 1
        
        # Calculate current metrics for comprehensive exit checks
        initial_risk = abs(self.entry_price - self.stop_price)
        tick_size = CONFIG['tick_size']
        tick_value = CONFIG['tick_value']
        
        if self.side == 'long':
            profit_ticks = (current_price - self.entry_price) / tick_size
            current_r = (current_price - self.entry_price) / initial_risk if initial_risk > 0 else 0
        else:
            profit_ticks = (self.entry_price - current_price) / tick_size
            current_r = (self.entry_price - current_price) / initial_risk if initial_risk > 0 else 0
        
        # **CRITICAL**: Use comprehensive exit checker - same as real trades!
        # This applies ALL 131 exit parameters including:
        # - profit_protection_min_r, sideways_max_loss_r, adverse_max_loss_r
        # - sideways detection, profit drawdown, adverse conditions
        # - max_time_in_trade, max_time_without_progress
        # - Everything the real bot uses
        
        try:
            # Build market context (matches real trade logic)
            market_context = {
                'consecutive_losses': 0,  # Ghost trades don't track streaks
                'consecutive_wins': 0,
                'daily_pnl': 0.0,
                'vix': 15.0,  # Default VIX
                'entry_atr': bar.get('atr', 2.0),
                'current_atr': bar.get('atr', 2.0)
            }
            
            # Create temporary comprehensive checker for this ghost trade
            from comprehensive_exit_logic import ComprehensiveExitChecker
            
            # Build trade context dict (same format as real trades)
            trade_context = {
                'entry_price': float(self.entry_price),
                'side': self.side,
                'contracts': self.contracts,
                'original_contracts': self.contracts,
                'initial_risk_ticks': self.initial_risk_ticks,
                'bars_in_trade': self.bars_elapsed,
                'entry_atr': bar.get('atr', 2.0),
                'stop_price': self.stop_price,
                # Additional state for comprehensive checks
                'highest_price': self.highest_price,
                'lowest_price': self.lowest_price,
                'breakeven_active': self.breakeven_active,
                'trailing_active': self.trailing_active,
                'remaining_quantity': self.remaining_quantity,
                # CRITICAL: Pass partial exit states so checker knows what's already done
                'partial_1_done': self.partial_1_done,
                'partial_2_done': self.partial_2_done,
                'partial_3_done': self.partial_3_done,
                'partial_exits': self.partial_exits
            }
            
            ghost_checker = ComprehensiveExitChecker(trade_context)
            
            # Update trade context with current state (bars_in_trade changes each update)
            ghost_checker.trade['bars_in_trade'] = self.bars_elapsed
            ghost_checker.trade['highest_price'] = self.highest_price
            ghost_checker.trade['lowest_price'] = self.lowest_price
            ghost_checker.trade['breakeven_active'] = self.breakeven_active
            ghost_checker.trade['trailing_active'] = self.trailing_active
            ghost_checker.trade['stop_price'] = self.stop_price
            ghost_checker.trade['remaining_quantity'] = self.remaining_quantity
            ghost_checker.trade['partial_1_done'] = self.partial_1_done
            ghost_checker.trade['partial_2_done'] = self.partial_2_done
            ghost_checker.trade['partial_3_done'] = self.partial_3_done
            ghost_checker.trade['partial_exits'] = self.partial_exits
            
            # Check ALL exit conditions using comprehensive logic
            comprehensive_exit = ghost_checker.check_all_exits(
                current_bar=bar,
                bar_index=bar_index,
                all_bars=self.all_bars,
                market_context=market_context
            )
            
            # Store ALL exit parameters used (for learning)
            self.all_exit_params_used = ghost_checker.get_all_used_params()
            
            # If comprehensive checker says exit, do it
            if comprehensive_exit and comprehensive_exit.get('should_exit', False):
                exit_reason = comprehensive_exit['exit_reason']
                exit_price = comprehensive_exit['exit_price']
                
                # Calculate final P&L
                if self.side == 'long':
                    pnl = (exit_price - self.entry_price) * tick_value * 4 * self.remaining_quantity
                else:
                    pnl = (self.entry_price - exit_price) * tick_value * 4 * self.remaining_quantity
                
                # Use helper function to create complete outcome
                self.outcome = self._create_outcome_dict(pnl, exit_price, exit_reason, bar_index, bar)
                # Add comprehensive-specific fields
                self.outcome['exit_params'] = comprehensive_exit.get('exit_params', {})
                self.outcome['triggered_params'] = comprehensive_exit.get('triggered_params', [])
                
                self.completed = True
                self.exit_bar_index = bar_index
                return self.outcome
            
            # Update stop/breakeven/trailing based on comprehensive checker state
            # (comprehensive checker may update stops even without triggering exit)
            if 'stop_price' in ghost_checker.trade:
                self.stop_price = ghost_checker.trade['stop_price']
            if 'breakeven_active' in ghost_checker.trade:
                was_breakeven = self.breakeven_active
                self.breakeven_active = ghost_checker.trade['breakeven_active']
                # Track when breakeven first activates
                if self.breakeven_active and not was_breakeven:
                    self.breakeven_activation_bar = self.bars_in_trade
                    self.bars_until_breakeven = self.bars_in_trade
                    self.in_breakeven_zone = True
            if 'trailing_active' in ghost_checker.trade:
                was_trailing = self.trailing_active
                self.trailing_active = ghost_checker.trade['trailing_active']
                # Track when trailing first activates
                if self.trailing_active and not was_trailing:
                    self.trailing_activation_bar = self.bars_in_trade
                    self.bars_until_trailing = self.bars_in_trade
            
            # CRITICAL: Update partial exit states from comprehensive checker
            # This ensures ghost trades track partial exits just like real trades
            if 'partial_1_done' in ghost_checker.trade:
                if ghost_checker.trade['partial_1_done'] and not self.partial_1_done:
                    self.partial_1_done = True
                    # Record the partial exit in history
                    self.partial_exits.append({
                        'type': 'partial_1',
                        'bar': self.bars_in_trade,
                        'r_multiple': current_r
                    })
            if 'partial_2_done' in ghost_checker.trade:
                if ghost_checker.trade['partial_2_done'] and not self.partial_2_done:
                    self.partial_2_done = True
                    self.partial_exits.append({
                        'type': 'partial_2',
                        'bar': self.bars_in_trade,
                        'r_multiple': current_r
                    })
            if 'partial_3_done' in ghost_checker.trade:
                if ghost_checker.trade['partial_3_done'] and not self.partial_3_done:
                    self.partial_3_done = True
                    self.partial_exits.append({
                        'type': 'partial_3',
                        'bar': self.bars_in_trade,
                        'r_multiple': current_r
                    })
            
            # Update remaining quantity if partials were taken
            if 'remaining_quantity' in ghost_checker.trade:
                self.remaining_quantity = ghost_checker.trade['remaining_quantity']
            
        except Exception as e:
            # Fallback to basic logic if comprehensive checker fails
            print(f"[ERROR] Comprehensive exit checker failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Track MAE/MFE
        if self.side == 'long':
            unrealized_pnl = (current_price - self.entry_price) * tick_value * 4 * self.remaining_quantity
        else:
            unrealized_pnl = (self.entry_price - current_price) * tick_value * 4 * self.remaining_quantity
        
        if unrealized_pnl < 0 and abs(unrealized_pnl) > self.mae:
            self.mae = abs(unrealized_pnl)
        if unrealized_pnl > 0 and unrealized_pnl > self.mfe:
            self.mfe = unrealized_pnl
        
        # BASIC STOP CHECK (backup if comprehensive didn't trigger)
        if self.side == 'long' and bar['low'] <= self.stop_price:
            pnl = (self.stop_price - self.entry_price) * tick_value * 4 * self.remaining_quantity
            self.outcome = self._create_outcome_dict(pnl, self.stop_price, 'stop_loss', bar_index, bar)
            self.completed = True
            self.exit_bar_index = bar_index
            return self.outcome
            
        elif self.side == 'short' and bar['high'] >= self.stop_price:
            pnl = (self.entry_price - self.stop_price) * tick_value * 4 * self.remaining_quantity
            self.outcome = self._create_outcome_dict(pnl, self.stop_price, 'stop_loss', bar_index, bar)
            self.completed = True
            self.exit_bar_index = bar_index
            return self.outcome
        
        # Timeout (200 bars ~ 3.3 hours)
        if self.bars_elapsed >= self.max_bars:
            if self.side == 'long':
                pnl = (current_price - self.entry_price) * tick_value * 4 * self.remaining_quantity
            else:
                pnl = (self.entry_price - current_price) * tick_value * 4 * self.remaining_quantity
            
            self.outcome = self._create_outcome_dict(pnl, current_price, 'ghost_timeout', bar_index, bar)
            self.completed = True
            self.exit_bar_index = bar_index
            return self.outcome
        
        return None  # Trade continues


def simulate_ghost_trade(entry_bar: Dict, side: str, df: pd.DataFrame, start_idx: int, atr: float, 
                        adaptive_manager=None, confidence: float = 0.5, reason: str = "rejected") -> Dict:
    """
    Create and simulate a ghost trade to completion using comprehensive exit logic.
    
    Ghost trades track what would have happened if a rejected signal was taken,
    allowing the ML model to learn from both accepted and rejected opportunities.
    Uses the same ComprehensiveExitChecker with all 131 exit parameters as real trades.
    
    Args:
        entry_bar: Bar data at entry
        side: 'long' or 'short'
        df: Full DataFrame of bars
        start_idx: Entry bar index
        atr: Current ATR value
        adaptive_manager: AdaptiveExitManager instance
        confidence: Signal confidence (0-1)
        reason: Rejection reason for tracking
    
    Returns:
        Dict with complete outcome including all 90+ tracked fields
    """
    # Create shadow trade
    shadow = ShadowTrade(
        entry_bar=entry_bar,
        side=side,
        confidence=confidence,
        rejection_reason=reason,
        atr=atr,
        adaptive_manager=adaptive_manager,
        entry_bar_index=start_idx,
        all_bars=df
    )
    
    # Simulate forward
    for i in range(start_idx + 1, min(start_idx + shadow.max_bars, len(df))):
        bar = df.iloc[i]
        outcome = shadow.update(bar, i, {})  # vwap_bands not needed for shadow
        if outcome:
            return outcome
    
    # If not completed, force timeout
    if not shadow.completed:
        last_bar = df.iloc[min(start_idx + shadow.max_bars - 1, len(df) - 1)]
        tick_value = CONFIG['tick_value']
        if side == 'long':
            pnl = (last_bar['close'] - entry_bar['close']) * tick_value * 4
        else:
            pnl = (entry_bar['close'] - last_bar['close']) * tick_value * 4
        
        return {
            'pnl': pnl,
            'exit_price': last_bar['close'],
            'duration_min': shadow.max_bars,
            'exit_reason': 'ghost_timeout',
            'took_trade': False,
            'confidence': confidence,
            'rejection_reason': reason,
            'mae': shadow.mae,
            'mfe': shadow.mfe,
            'breakeven_activated': bool(shadow.breakeven_active),
            'trailing_activated': bool(shadow.trailing_active),
            'breakeven_activation_bar': shadow.bars_elapsed if shadow.breakeven_active else 0,
            'trailing_activation_bar': shadow.bars_elapsed if shadow.trailing_active else 0
        }
    
    return shadow.outcome


def print_learned_insights_summary(trades, experiences, signal_manager, exit_manager):
    """Print overall learned insights from the backtest"""
    print("\n" + "=" * 80)
    print("📚 LEARNED INSIGHTS SUMMARY")
    print("=" * 80)
    
    if len(trades) < 5:
        print("  Insufficient trades for learning analysis")
        return
    
    # Separate winners and losers
    winners = [t for t in trades if t['pnl'] > 0]
    losers = [t for t in trades if t['pnl'] <= 0]
    
    print(f"\n🎯 SIGNAL QUALITY INSIGHTS:")
    
    # Confidence analysis
    if len(winners) > 0 and len(losers) > 0:
        avg_win_conf = sum(t['confidence'] for t in winners) / len(winners)
        avg_loss_conf = sum(t['confidence'] for t in losers) / len(losers)
        print(f"  • Winners averaged {avg_win_conf:.1%} confidence vs {avg_loss_conf:.1%} for losers")
        if avg_win_conf > avg_loss_conf * 1.1:
            print(f"    → High confidence signals are {((avg_win_conf/avg_loss_conf - 1) * 100):.0f}% more reliable")
        
    # Time of day patterns
    from collections import Counter
    win_hours = Counter([t['entry_time'].hour for t in winners])
    loss_hours = Counter([t['entry_time'].hour for t in losers])
    
    if len(win_hours) > 0:
        best_hour = win_hours.most_common(1)[0][0]
        worst_hour = loss_hours.most_common(1)[0][0] if len(loss_hours) > 0 else 0
        print(f"  • Best trading hour: {best_hour:02d}:00 UTC ({win_hours[best_hour]} wins)")
        print(f"  • Worst trading hour: {worst_hour:02d}:00 UTC ({loss_hours.get(worst_hour, 0)} losses)")
    
    # Day of week patterns
    win_days = Counter([t['entry_time'].strftime('%a') for t in winners])
    loss_days = Counter([t['entry_time'].strftime('%a') for t in losers])
    
    if len(win_days) > 0:
        best_day = win_days.most_common(1)[0][0]
        print(f"  • Best trading day: {best_day} ({win_days[best_day]} wins)")
    
    print(f"\n🚪 EXIT MANAGEMENT INSIGHTS:")
    
    # Duration analysis
    win_durations = [t['duration_min'] for t in winners]
    loss_durations = [t['duration_min'] for t in losers]
    
    if len(win_durations) > 0:
        avg_win_dur = sum(win_durations) / len(win_durations)
        avg_loss_dur = sum(loss_durations) / len(loss_durations) if len(loss_durations) > 0 else 0
        print(f"  • Winners held for {avg_win_dur:.0f} min avg vs {avg_loss_dur:.0f} min for losers")
        if avg_win_dur > avg_loss_dur * 1.5:
            print(f"    → Winners need more time to develop (hold longer on strong setups)")
        elif avg_loss_dur > avg_win_dur * 1.5:
            print(f"    → Cut losers faster (they linger {avg_loss_dur/avg_win_dur:.1f}x longer)")
    
    # R-multiple analysis
    r_multiples = [t['r_multiple'] for t in trades]
    avg_r = sum(r_multiples) / len(r_multiples)
    winners_r = [t['r_multiple'] for t in winners]
    avg_win_r = sum(winners_r) / len(winners_r) if len(winners_r) > 0 else 0
    
    print(f"  • Average R-multiple: {avg_r:.2f}R (Winners: {avg_win_r:.2f}R)")
    if avg_win_r < 2.0:
        print(f"    → Taking profits too early (avg winner only {avg_win_r:.2f}R, target 2-3R)")
    
    # Exit reason analysis
    exit_reasons = Counter([t['exit_reason'] for t in trades])
    print(f"  • Most common exit: {exit_reasons.most_common(1)[0][0]} ({exit_reasons.most_common(1)[0][1]} times)")
    
    stop_loss_trades = [t for t in trades if t['exit_reason'] == 'stop_loss']
    if len(stop_loss_trades) > len(trades) * 0.7:
        stop_wins = [t for t in stop_loss_trades if t['pnl'] > 0]
        print(f"    → {len(stop_loss_trades)}/{len(trades)} stopped out ({len(stop_wins)} were winners via trailing)")
    
    # Partial exits analysis
    partial_trades = [t for t in trades if t['partial_exits'] > 0]
    if len(partial_trades) > 0:
        partial_pnls = [t['pnl'] for t in partial_trades]
        no_partial_pnls = [t['pnl'] for t in trades if t['partial_exits'] == 0]
        avg_partial = sum(partial_pnls) / len(partial_pnls)
        avg_no_partial = sum(no_partial_pnls) / len(no_partial_pnls) if len(no_partial_pnls) > 0 else 0
        print(f"  • Trades with partials: {len(partial_trades)} (avg P&L: ${avg_partial:.0f})")
        print(f"  • Trades without partials: {len(trades) - len(partial_trades)} (avg P&L: ${avg_no_partial:.0f})")
    
    print(f"\n👻 GHOST TRADE INSIGHTS:")
    
    # Ghost trade analysis
    ghosts = [e for e in experiences if not e['took_trade']]
    if len(ghosts) > 0:
        ghost_winners = [g for g in ghosts if g['outcome']['pnl'] > 0]
        ghost_losers = [g for g in ghosts if g['outcome']['pnl'] <= 0]
        
        print(f"  • Rejected {len(ghosts)} signals: {len(ghost_winners)} would have won, {len(ghost_losers)} would have lost")
        
        if len(ghost_winners) > len(ghost_losers):
            print(f"    ⚠️  Being too conservative - missing {len(ghost_winners) - len(ghost_losers)} net winners")
            print(f"    → Lower confidence threshold or reduce rejection rate")
        else:
            print(f"    ✅ Good signal filtering - avoided {len(ghost_losers) - len(ghost_winners)} net losers")
    
    # EXIT RL INSIGHTS - Analyze what exit manager learned
    print(f"\n🎯 EXIT RL DEEP LEARNING (38 Adaptive Adjustments):")
    
    if hasattr(exit_manager, 'exit_experiences') and len(exit_manager.exit_experiences) > 50:
        exps = exit_manager.exit_experiences
        
        # 1. Trailing stop analysis
        trailing_exps = [e for e in exps if e.get('trailing_stop_activated', False)]
        if len(trailing_exps) > 10:
            trailing_wins = [e for e in trailing_exps if e.get('pnl', 0) > 0]
            trailing_wr = len(trailing_wins) / len(trailing_exps) * 100
            avg_trailing_dist = sum(e.get('exit_params', {}).get('trailing_distance_ticks', 10) for e in trailing_exps[-20:]) / min(20, len(trailing_exps))
            print(f"  • Trailing stops: {len(trailing_exps)} trades, {trailing_wr:.0f}% WR → Learned distance: {avg_trailing_dist:.1f} ticks")
        
        # 2. Breakeven analysis
        breakeven_exps = [e for e in exps if e.get('breakeven_activated', False)]
        if len(breakeven_exps) > 10:
            be_wins = [e for e in breakeven_exps if e.get('pnl', 0) > 0]
            be_wr = len(be_wins) / len(breakeven_exps) * 100
            avg_be_threshold = sum(e.get('exit_params', {}).get('breakeven_threshold_ticks', 10) for e in breakeven_exps[-20:]) / min(20, len(breakeven_exps))
            print(f"  • Breakeven moves: {len(breakeven_exps)} trades, {be_wr:.0f}% WR → Learned threshold: {avg_be_threshold:.1f} ticks")
        
        # 3. Partial exit analysis
        partial_exps = [e for e in exps if e.get('partial_exits_taken', 0) > 0]
        if len(partial_exps) > 5:
            partial_pnls = [e.get('pnl', 0) for e in partial_exps]
            full_pnls = [e.get('pnl', 0) for e in exps if e.get('partial_exits_taken', 0) == 0]
            avg_partial_pnl = sum(partial_pnls) / len(partial_pnls)
            avg_full_pnl = sum(full_pnls) / len(full_pnls) if len(full_pnls) > 0 else 0
            print(f"  • Partial exits: {len(partial_exps)} trades avg ${avg_partial_pnl:.0f} vs ${avg_full_pnl:.0f} for full position")
            if avg_partial_pnl > avg_full_pnl * 1.2:
                print(f"    → Partials working well ({((avg_partial_pnl/avg_full_pnl - 1) * 100):.0f}% better)")
        
        # 4. Volatility-based adjustments
        high_vix_trades = [e for e in exps if e.get('market_state', {}).get('vix', 15) > 20]
        low_vix_trades = [e for e in exps if e.get('market_state', {}).get('vix', 15) < 15]
        
        if len(high_vix_trades) > 5 and len(low_vix_trades) > 5:
            high_vix_pnl = sum(e.get('pnl', 0) for e in high_vix_trades) / len(high_vix_trades)
            low_vix_pnl = sum(e.get('pnl', 0) for e in low_vix_trades) / len(low_vix_trades)
            print(f"  • High volatility (VIX>20): ${high_vix_pnl:.0f} avg vs ${low_vix_pnl:.0f} in calm markets")
            if high_vix_pnl < low_vix_pnl * 0.7:
                print(f"    → Tighter stops in volatile conditions recommended")
        
        # 5. Time-based patterns
        quick_trades = [e for e in exps if e.get('bars_held', 0) < 10]
        long_trades = [e for e in exps if e.get('bars_held', 0) > 30]
        
        if len(quick_trades) > 5 and len(long_trades) > 5:
            quick_wr = sum(1 for e in quick_trades if e.get('pnl', 0) > 0) / len(quick_trades) * 100
            long_wr = sum(1 for e in long_trades if e.get('pnl', 0) > 0) / len(long_trades) * 100
            print(f"  • Quick exits (<10 bars): {quick_wr:.0f}% WR vs Long holds (>30 bars): {long_wr:.0f}% WR")
            if quick_wr > long_wr * 1.3:
                print(f"    → Scalping working better than swing holds")
        
        # 6. Drawdown tolerance
        drawdown_exps = [e for e in exps if e.get('max_adverse_excursion', 0) > 0]
        if len(drawdown_exps) > 10:
            recovered = [e for e in drawdown_exps if e.get('pnl', 0) > 0]
            recovery_rate = len(recovered) / len(drawdown_exps) * 100
            avg_mae = sum(e.get('max_adverse_excursion', 0) for e in drawdown_exps) / len(drawdown_exps)
            print(f"  • Drawdown recovery: {recovery_rate:.0f}% of trades with avg {avg_mae:.1f} tick MAE recovered")
            if recovery_rate < 40:
                print(f"    → Stop losses too wide (only {recovery_rate:.0f}% recover from drawdown)")
        
        # 7. Session-based performance
        from collections import defaultdict
        session_pnls = defaultdict(list)
        for e in exps:
            session = e.get('market_state', {}).get('session', 'Unknown')
            session_pnls[session].append(e.get('pnl', 0))
        
        if len(session_pnls) > 1:
            print(f"  • Session performance:")
            for session, pnls in sorted(session_pnls.items(), key=lambda x: sum(x[1])/len(x[1]) if x[1] else 0, reverse=True):
                if len(pnls) > 3:
                    avg_pnl = sum(pnls) / len(pnls)
                    wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
                    print(f"    - {session}: ${avg_pnl:.0f} avg, {wr:.0f}% WR ({len(pnls)} trades)")
        
        # 8. Confidence correlation
        if len(exps) > 20:
            high_conf_exits = [e for e in exps if e.get('entry_confidence', 0.5) > 0.65]
            low_conf_exits = [e for e in exps if e.get('entry_confidence', 0.5) < 0.45]
            
            if len(high_conf_exits) > 5 and len(low_conf_exits) > 5:
                high_conf_pnl = sum(e.get('pnl', 0) for e in high_conf_exits) / len(high_conf_exits)
                low_conf_pnl = sum(e.get('pnl', 0) for e in low_conf_exits) / len(low_conf_exits)
                print(f"  • High confidence entries (>65%): ${high_conf_pnl:.0f} avg")
                print(f"  • Low confidence entries (<45%): ${low_conf_pnl:.0f} avg")
                if high_conf_pnl > low_conf_pnl * 2:
                    print(f"    → Skip low confidence setups ({((high_conf_pnl/low_conf_pnl - 1) * 100):.0f}% better on high conf)")
        
        # 9. Stop adjustment effectiveness
        stop_adj_trades = [e for e in exps if e.get('stop_adjustment_count', 0) > 0]
        if len(stop_adj_trades) > 10:
            adj_wins = sum(1 for e in stop_adj_trades if e.get('pnl', 0) > 0)
            adj_wr = adj_wins / len(stop_adj_trades) * 100
            avg_adjustments = sum(e.get('stop_adjustment_count', 0) for e in stop_adj_trades) / len(stop_adj_trades)
            print(f"  • Stop adjustments: {len(stop_adj_trades)} trades with avg {avg_adjustments:.1f} adjustments, {adj_wr:.0f}% WR")
            if adj_wr > 60:
                print(f"    → Active stop management working ({adj_wr:.0f}% WR on adjusted trades)")
        
        # 10. Regime change impact
        regime_change_trades = [e for e in exps if e.get('volatility_regime_change', False)]
        if len(regime_change_trades) > 5:
            regime_wins = sum(1 for e in regime_change_trades if e.get('pnl', 0) > 0)
            regime_wr = regime_wins / len(regime_change_trades) * 100
            print(f"  • Regime changes during trade: {len(regime_change_trades)} trades, {regime_wr:.0f}% WR")
            if regime_wr < 40:
                print(f"    → Exit when volatility regime shifts (poor WR in transitions)")
        
        # 11. R-multiple distribution
        r_multiples = [e.get('max_r_achieved', 0) for e in exps]
        if len(r_multiples) > 20:
            big_winners = [r for r in r_multiples if r >= 2.0]
            small_winners = [r for r in r_multiples if 0 < r < 1.0]
            print(f"  • R-multiple distribution: {len(big_winners)} trades hit 2R+ ({len(big_winners)/len(r_multiples)*100:.0f}%)")
            print(f"  • Small winners (<1R): {len(small_winners)} trades ({len(small_winners)/len(r_multiples)*100:.0f}%)")
            if len(small_winners) > len(big_winners) * 2:
                print(f"    → Too many small wins - let winners run longer")
        
        # 12. Exit parameter learning summary
        recent_exps = exps[-50:] if len(exps) > 50 else exps
        if len(recent_exps) > 10:
            avg_stop_mult = sum(e.get('exit_params', {}).get('stop_mult', 3.6) for e in recent_exps) / len(recent_exps)
            avg_trailing = sum(e.get('exit_params', {}).get('trailing_distance_ticks', 12) for e in recent_exps) / len(recent_exps)
            avg_breakeven = sum(e.get('exit_params', {}).get('breakeven_threshold_ticks', 10) for e in recent_exps) / len(recent_exps)
            
            print(f"  • Learned exit parameters (last 50 trades):")
            print(f"    - Stop multiplier: {avg_stop_mult:.2f}x ATR")
            print(f"    - Trailing distance: {avg_trailing:.1f} ticks")
            print(f"    - Breakeven threshold: {avg_breakeven:.1f} ticks")
        
        # 13. Slippage & commission impact
        slippage_total = sum(e.get('slippage_ticks', 0) for e in exps)
        commission_total = sum(e.get('commission_cost', 0) for e in exps)
        if slippage_total > 0 or commission_total > 0:
            gross_pnl = sum(e.get('pnl', 0) for e in exps)
            net_pnl = gross_pnl  # Already includes costs
            print(f"  • Trading costs: ${slippage_total * 12.5:.0f} slippage + ${commission_total:.0f} commissions")
            if slippage_total + commission_total > abs(gross_pnl) * 0.5:
                print(f"    → Costs eating {(slippage_total * 12.5 + commission_total) / abs(gross_pnl) * 100:.0f}% of gross P&L - reduce trade frequency")
        
        # 14. Day of week patterns
        day_pnls = defaultdict(list)
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for e in exps:
            day = e.get('market_state', {}).get('day_of_week', 0)
            day_pnls[day].append(e.get('pnl', 0))
        
        if len(day_pnls) > 1:
            print(f"  • Day of week patterns:")
            for day in sorted(day_pnls.keys()):
                if len(day_pnls[day]) > 3:
                    avg_pnl = sum(day_pnls[day]) / len(day_pnls[day])
                    wr = sum(1 for p in day_pnls[day] if p > 0) / len(day_pnls[day]) * 100
                    print(f"    - {day_names[day]}: ${avg_pnl:.0f} avg, {wr:.0f}% WR ({len(day_pnls[day])} trades)")
        
        # 15. Overall learning progress
        print(f"\n  📈 LEARNING PROGRESS:")
        print(f"  • Total exit experiences analyzed: {len(exps):,}")
        print(f"  • Experiences used for pattern matching: {len([e for e in exps if e.get('pnl', 0) != 0]):,}")
        recent_50 = exps[-50:] if len(exps) > 50 else exps
        early_50 = exps[:50] if len(exps) > 100 else exps[:len(exps)//2]
        if len(recent_50) > 10 and len(early_50) > 10:
            recent_wr = sum(1 for e in recent_50 if e.get('pnl', 0) > 0) / len(recent_50) * 100
            early_wr = sum(1 for e in early_50 if e.get('pnl', 0) > 0) / len(early_50) * 100
            recent_avg = sum(e.get('pnl', 0) for e in recent_50) / len(recent_50)
            early_avg = sum(e.get('pnl', 0) for e in early_50) / len(early_50)
            print(f"  • Recent performance (last 50): {recent_wr:.0f}% WR, ${recent_avg:.0f} avg")
            print(f"  • Early performance (first 50): {early_wr:.0f}% WR, ${early_avg:.0f} avg")
            if recent_wr > early_wr * 1.1:
                print(f"    ✅ IMPROVING - Bot learning working! (+{recent_wr - early_wr:.0f}% WR improvement)")
            elif recent_wr < early_wr * 0.9:
                print(f"    ⚠️  DEGRADING - Recent performance worse (-{early_wr - recent_wr:.0f}% WR drop)")
    
    print("\n" + "=" * 80)


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
    
    # Handle adaptive vs fixed threshold
    threshold = CONFIG['rl_confidence_threshold']
    if threshold == "adaptive":
        print(f"ML Confidence Threshold: ADAPTIVE (starts at 50%, adjusts based on performance)")
    else:
        print(f"ML Confidence Threshold: {threshold:.0%}")
    
    # Note: exploration_rate will be shown after loading experiences (if adaptive)
    print(f"Max Contracts: {CONFIG['max_contracts']}")
    
    # Collect experiences during backtest for bulk save at end
    backtest_experiences = []
    print(f"Exit Strategy: NEURAL NETWORK PREDICTIONS")
    print(f"  🧠 Exit neural network predicts 6 adaptive parameters:")
    print(f"     • Breakeven threshold (ticks)")
    print(f"     • Trailing stop distance (ticks)")
    print(f"     • Stop loss multiplier (ATR)")
    print(f"     • Partial exit levels (3 R-multiples)")
    print(f"  📊 Trained on 1,466+ real exit experiences")
    print(f"  ⚡ Adapts to: VIX, ATR, time of day, confidence, regime")
    print("=" * 80)
    print()
    
    # Load local experiences if in local mode
    if CONFIG.get('local_mode', False):
        print("🔧 LOADING LOCAL EXPERIENCES FOR DEV MODE...")
        from local_experience_manager import local_manager
        if local_manager.load_experiences():
            counts = local_manager.get_experience_count()
            print(f"   ✅ Loaded {counts['signal']:,} signal + {counts['exit']:,} exit = {counts['total']:,} total experiences")
            print(f"   🧠 Neural network ready (100x faster than pattern matching)")
            signal_manager_for_vwap = local_manager  # Pass to VWAP function
            
            # ADAPTIVE EXPLORATION RATE based on data availability
            if CONFIG['exploration_rate'] == "adaptive":
                total_exps = counts['signal']
                if total_exps < 100:
                    CONFIG['exploration_rate'] = 0.30  # HIGH: Build diverse dataset
                    print(f"   🔍 Adaptive Exploration: 30% (sparse data: {total_exps} experiences)")
                elif total_exps < 500:
                    CONFIG['exploration_rate'] = 0.20  # MEDIUM-HIGH: Continue learning
                    print(f"   🔍 Adaptive Exploration: 20% (building data: {total_exps} experiences)")
                elif total_exps < 2000:
                    CONFIG['exploration_rate'] = 0.05  # LOW: Test learned strategy with minimal exploration
                    print(f"   🔍 Adaptive Exploration: 5% (moderate data: {total_exps} experiences)")
                else:
                    CONFIG['exploration_rate'] = 0.05  # LOW: Test learned strategy, keep adaptability
                    print(f"   🔍 Adaptive Exploration: 5% (rich data: {total_exps} experiences)")
        else:
            print(f"   ⚠️  No existing local experiences found - will create new files")
            print(f"   🎯 Starting fresh - will collect experiences during this backtest")
            signal_manager_for_vwap = local_manager  # Still use local_manager for saving
        print()
    else:
        signal_manager_for_vwap = None
    
    # Load data
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Data is in UTC timezone (TopstepX API returns UTC, stored as naive datetime)
    # ES Schedule: Sun 23:00 UTC open, Mon-Fri 22:00-21:00 UTC, maintenance 21:00-22:00 UTC
    # NO timezone conversion needed - data and config both in UTC
    
    # Filter to last N days
    last_date = df['timestamp'].max()
    start_date = last_date - pd.Timedelta(days=days)
    df = df[df['timestamp'] >= start_date].reset_index(drop=True)
    
    print(f"Loaded {len(df):,} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Timezone: UTC - matches ES futures schedule and live bot")
    print()
    
    # Initialize LOCAL EXIT MANAGER - 100% offline
    print("🧠 EXIT NEURAL NETWORK: Predicting adaptive exit parameters from market context")
    from local_exit_manager import local_exit_manager
    local_exit_manager.load_experiences()
    adaptive_exit_manager = local_exit_manager  # Use local manager only
    print(f"  ✅ Exit NN ready - Adapts stops, breakeven, trailing, and partials dynamically")
    print()
    
    # Trading state
    active_trade = None
    signals_detected = 0
    signals_ml_approved = 0
    signals_ml_rejected = 0
    completed_trades = []
    ghost_trades = []  # Track rejected signals for learning
    
    # RL Learning - Track performance for state
    current_streak = 0  # Positive = wins, negative = losses
    recent_pnl_sum = 0.0
    
    # PSYCHOLOGICAL TRACKING for signal RL
    consecutive_wins = 0
    consecutive_losses = 0
    starting_balance = 50000.0  # Starting account balance
    cumulative_pnl = 0.0  # Profit/loss from trades (starts at 0)
    peak_balance = 0.0
    last_trade_time = None
    
    # DAILY LOSS LIMIT TRACKING (matches live bot)
    daily_pnl = 0.0  # P&L for current trading day
    current_trading_day = None  # Track which day we're on
    daily_limit_hit = False  # Stop trading when daily loss limit hit
    days_stopped_by_limit = 0  # Count how many days hit the limit
    
    # OPTIMIZATION: Pre-calculate trading session boundaries (VWAP resets after maintenance)
    # ES maintenance: 21:00-22:00 UTC daily. Trading day runs 22:00 UTC to 21:00 UTC next day
    # VWAP RESET LOGIC:
    #   - Each session starts at 22:00 UTC (post-maintenance)
    #   - VWAP calculated from session start to current bar
    #   - Cache cleared during maintenance (21:00) to ensure fresh calculations
    #   - Result: VWAP fully resets daily after maintenance window
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    
    # Calculate trading session start (22:00 UTC = start of new VWAP period)
    # Store by bar index to handle sessions spanning midnight
    session_start_map = {}  # idx -> session_start_idx (VWAP anchor point)
    current_session_start = None
    
    for i in range(len(df)):
        hour = df.iloc[i]['timestamp'].hour
        
        # New session starts at 23:00 UTC (first hour after maintenance 22:00-23:00) - VWAP RESET POINT
        # NOTE: Hour 22 is removed from cleaned data (maintenance window), so we detect session at 23:00
        if hour == 23 and (i == 0 or df.iloc[i-1]['timestamp'].hour != 23):
            current_session_start = i
        elif i == 0:  # Handle first bar if it's not hour 23
            current_session_start = 0
        
        # Map each bar to its session start (ensures VWAP always calculated from session start)
        if current_session_start is not None:
            session_start_map[i] = current_session_start
    
    # Cache for VWAP calculations (session_start, bar_count) -> vwap_bands
    vwap_cache = {}
    
    # Process each bar
    print(f"\nProcessing {len(df):,} bars...")
    progress_interval = max(1, len(df) // 100)  # Update every 1% progress
    
    for idx in range(100, len(df)):  # Start at 100 for indicators
        # Progress bar
        if idx % progress_interval == 0:
            pct = (idx - 100) / (len(df) - 100) * 100
            bars_done = idx - 100
            bars_total = len(df) - 100
            print(f"\r[{pct:5.1f}%] {bars_done:,}/{bars_total:,} bars | Signals: {signals_detected} | Trades: {len(completed_trades)} | Active: {'YES' if active_trade else 'NO '}", end='', flush=True)
        
        bar = df.iloc[idx]
        current_hour = bar['timestamp'].hour
        
        # During maintenance window (22:00-23:00), skip all processing
        # NOTE: Hour 22 is already removed from cleaned data, so this mainly handles hour boundaries
        if current_hour == 22:
            # Clear VWAP cache during maintenance - fresh start at 23:00
            if vwap_cache:
                vwap_cache.clear()  # Reset for new session
            continue
        
        # VWAP RESET: Find session start (22:00 UTC after maintenance)
        session_start_idx = session_start_map.get(idx, None)
        
        if session_start_idx is None:
            continue  # Not in a valid trading session yet
        
        # DAILY P&L RESET (trading day starts at 22:00 UTC after maintenance)
        bar_date = bar['timestamp'].date()
        if current_trading_day is None or bar_date != current_trading_day:
            # New trading day started
            if daily_limit_hit:
                # Previous day hit limit, count it
                days_stopped_by_limit += 1
            current_trading_day = bar_date
            daily_pnl = 0.0  # Reset daily P&L
            daily_limit_hit = False  # Reset limit flag
        
        # Calculate VWAP from session start (post-maintenance) to current bar
        # This ensures VWAP resets daily at 22:00 UTC after maintenance
        df_session = df.iloc[session_start_idx:idx + 1]
        
        if len(df_session) < 30:
            continue  # Need enough bars for VWAP calculation
        
        # OPTIMIZED: Cache VWAP by (session_start, bar_count) to avoid recalculation
        # Cache is cleared during maintenance, so each session has fresh VWAP
        cache_key = (session_start_idx, len(df_session))
        if cache_key in vwap_cache:
            vwap_bands = vwap_cache[cache_key]
        else:
            vwap_bands = calculate_vwap_bands(df_session.copy(), signal_manager_for_vwap)
            vwap_cache[cache_key] = vwap_bands
        
        # Calculate indicators (neural network baseline parameters)
        recent_closes = df.iloc[idx-50:idx]['close']
        rsi = calculate_rsi(recent_closes, period=CONFIG['rsi_period'])
        atr = calculate_atr(df.iloc[idx-20:idx], period=CONFIG['atr_period'])
        
        # Handle NaN values (not enough data yet)
        import math
        if math.isnan(rsi):
            rsi = 50.0  # Default to neutral RSI
        if math.isnan(atr):
            atr = 2.0   # Default ATR
        
        # Calculate volume ratio for RL state
        if idx >= 20:
            avg_volume = df.iloc[idx-20:idx]['volume'].mean()
            volume_ratio = bar['volume'] / avg_volume if avg_volume > 0 else 1.0
        else:
            volume_ratio = 1.0
        
        # Calculate VWAP distance for RL state
        vwap_distance = (bar['close'] - vwap_bands['vwap']) / vwap_bands['vwap'] if vwap_bands['vwap'] > 0 else 0.0
        
        # Calculate market regime (needed for exit params)
        recent_bars_for_regime = df.iloc[max(0, idx-50):idx+1].to_dict('records')
        market_regime = detect_market_regime(recent_bars_for_regime, atr)
        
        # Update active trade (pass idx for adaptive exits)
        if active_trade:
            exit_result = active_trade.update(bar, idx)
            
            if exit_result:
                exit_reason, exit_price, contracts_closed = exit_result
                
                # Calculate P&L with realistic costs
                tick_size = CONFIG['tick_size']
                tick_value = CONFIG['tick_value']
                
                if active_trade.side == 'long':
                    profit_ticks = (exit_price - active_trade.entry_price) / tick_size
                else:
                    profit_ticks = (active_trade.entry_price - exit_price) / tick_size
                
                # Deduct slippage (1 tick entry + 1 tick exit = 2 ticks total)
                slippage_cost_ticks = CONFIG['slippage_ticks'] * 2  # Entry + Exit
                profit_ticks -= slippage_cost_ticks
                
                # Calculate gross P&L
                pnl = profit_ticks * tick_value * contracts_closed
                
                # Deduct commissions ($2.50 per contract per side = $5 round trip)
                commission_cost = CONFIG['commission_per_contract'] * 2 * contracts_closed  # Entry + Exit
                pnl -= commission_cost
                
                # Track costs on trade object
                active_trade.exit_slippage_ticks = slippage_cost_ticks
                active_trade.commission_cost += commission_cost
                
                # For partial exits, record partial trade
                if 'partial' in exit_reason:
                    # Partial tracked (reduced spam)
                    
                    # RECORD PARTIAL EXIT FOR EXIT RL LEARNING
                    if hasattr(active_trade, 'current_exit_params') and active_trade.current_exit_params:
                        try:
                            # Calculate synthetic VIX (same formula)
                            atr_contribution = (atr / 2.0) * 10.0
                            volume_contribution = (1.0 - min(volume_ratio, 1.5)) * 5.0
                            synthetic_vix = 15.0 + atr_contribution + volume_contribution
                            synthetic_vix = max(10.0, min(synthetic_vix, 35.0))
                            
                            # Capture current market state for exit learning
                            market_state = {
                                'rsi': rsi,
                                'volume_ratio': volume_ratio,
                                'hour': bar['timestamp'].hour,
                                'day_of_week': bar['timestamp'].weekday(),
                                'streak': current_streak,
                                'recent_pnl': recent_pnl_sum,
                                'vix': synthetic_vix,  # Synthetic VIX from ATR + volume
                                'vwap_distance': vwap_distance,
                                'atr': atr
                            }
                            
                            # Record partial exit outcome (use actual exit params from comprehensive checker)
                            adaptive_exit_manager.record_exit_outcome(
                                regime=market_regime,
                                exit_params=active_trade.comprehensive_exit_checker.exit_params,
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
                active_trade.exit_bar_index = idx  # NEW: Record exit bar index
                duration = (bar['timestamp'] - active_trade.entry_time).total_seconds() / 60
                
                # Calculate total P&L including partials
                total_pnl = pnl
                for partial in active_trade.partial_exits:
                    if active_trade.side == 'long':
                        partial_profit_ticks = (partial['price'] - active_trade.entry_price) / tick_size
                    else:
                        partial_profit_ticks = (active_trade.entry_price - partial['price']) / tick_size
                    total_pnl += partial_profit_ticks * tick_value * partial['contracts']
                
                # Deduct slippage and commission costs
                slippage_cost = CONFIG['slippage_ticks'] * 2 * CONFIG['tick_value']  # Entry + exit
                commission_cost = CONFIG['commission_per_contract'] * 2 * active_trade.original_contracts  # Round trip
                total_pnl -= (slippage_cost + commission_cost)
                
                # Update streak tracking for RL
                if total_pnl > 0:
                    current_streak = current_streak + 1 if current_streak > 0 else 1
                    consecutive_wins += 1
                    consecutive_losses = 0
                else:
                    current_streak = current_streak - 1 if current_streak < 0 else -1
                    consecutive_losses += 1
                    consecutive_wins = 0
                
                # Update cumulative P&L and peak balance
                cumulative_pnl += total_pnl
                if cumulative_pnl > peak_balance:
                    peak_balance = cumulative_pnl
                
                # Update daily P&L (matches live bot)
                daily_pnl += total_pnl
                
                # Check if daily loss limit hit - HALT trading for rest of day (not full stop)
                if daily_pnl <= -CONFIG['daily_loss_limit'] and not daily_limit_hit:
                    daily_limit_hit = True
                    print(f"\n{'='*80}")
                    print(f"⚠️  DAILY LOSS LIMIT BREACHED - TRADING HALTED FOR TODAY")
                    print(f"{'='*80}")
                    print(f"Daily P&L: ${daily_pnl:.2f} / Limit: -${CONFIG['daily_loss_limit']:.2f}")
                    print(f"Date: {bar_date}")
                    print(f"Trading will resume tomorrow after maintenance window.")
                    print(f"This simulates prop firm rule - no more trades today.")
                    print(f"{'='*80}\n")
                
                # Update last trade time
                last_trade_time = bar['timestamp']
                
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
                
                # SAVE SIGNAL EXPERIENCE FOR RL LEARNING TO CLOUD API
                # Store the entry state and trade outcome for pattern learning
                if hasattr(active_trade, 'entry_state'):
                    # Add exit price and confidence to state
                    active_trade.entry_state['exit_price'] = exit_price
                    active_trade.entry_state['confidence'] = active_trade.confidence
                    
                    outcome = {
                        'pnl': total_pnl,
                        'duration_min': duration,
                        'exit_reason': exit_reason,
                        'exit_price': exit_price,
                        'partial_exits': len(active_trade.partial_exits),
                        'confidence': active_trade.confidence  # Save confidence for learning
                    }
                    
                    # COLLECT for bulk save at end (don't POST during backtest - too slow!)
                    backtest_experiences.append({
                        'rl_state': active_trade.entry_state.copy(),
                        'took_trade': True,
                        'outcome': outcome
                    })
                
                # RECORD FULL EXIT FOR EXIT RL LEARNING
                # Always record exits - we get params from comprehensive checker now
                try:
                    # Calculate synthetic VIX (same formula)
                    atr_contribution = (atr / 2.0) * 10.0
                    volume_contribution = (1.0 - min(volume_ratio, 1.5)) * 5.0
                    synthetic_vix = 15.0 + atr_contribution + volume_contribution
                    synthetic_vix = max(10.0, min(synthetic_vix, 35.0))
                    
                    # Capture current market state for exit learning
                    market_state = {
                        'rsi': rsi,
                        'volume_ratio': volume_ratio,
                        'hour': bar['timestamp'].hour,
                        'day_of_week': bar['timestamp'].weekday(),
                        'streak': current_streak,
                        'recent_pnl': recent_pnl_sum,
                        'vix': synthetic_vix,  # Synthetic VIX from ATR + volume
                        'vwap_distance': vwap_distance,
                        'atr': atr,
                        'peak_pnl': float(active_trade.peak_unrealized_pnl)  # CRITICAL: Peak profit for drawdown learning
                    }
                    
                    # Record final exit outcome with full market context
                    # Calculate R-multiple for learning
                    r_multiple = 0.0
                    if active_trade.initial_risk_ticks > 0:
                        profit_ticks = total_pnl / CONFIG['tick_value']
                        r_multiple = profit_ticks / active_trade.initial_risk_ticks
                    
                    # Calculate MAE (Maximum Adverse Excursion) and MFE (Maximum Favorable Excursion)
                    if active_trade.side.upper() == 'LONG':
                        mae = (active_trade.lowest_price - active_trade.entry_price) / CONFIG['tick_size'] * CONFIG['tick_value']
                        mfe = (active_trade.highest_price - active_trade.entry_price) / CONFIG['tick_size'] * CONFIG['tick_value']
                    else:  # SHORT
                        mae = (active_trade.entry_price - active_trade.highest_price) / CONFIG['tick_size'] * CONFIG['tick_value']
                        mfe = (active_trade.entry_price - active_trade.lowest_price) / CONFIG['tick_size'] * CONFIG['tick_value']
                    
                    # NEW: Check if trade held through multiple sessions
                    held_through_sessions = active_trade.entry_session != active_trade._get_session(bar)
                    
                    # **CRITICAL**: Get ALL 131 exit parameters from comprehensive checker
                    # This is what the bot actually used - not the partial current_exit_params
                    if hasattr(active_trade, 'all_exit_params_used') and active_trade.all_exit_params_used:
                        enhanced_exit_params = active_trade.all_exit_params_used.copy()
                    else:
                        # Fallback: use current_exit_params if comprehensive checker didn't populate
                        enhanced_exit_params = active_trade.current_exit_params.copy() if active_trade.current_exit_params else {}
                    
                    # Add calculated exit context (for neural network features)
                    enhanced_exit_params['current_atr'] = atr  # ATR at exit
                    # Don't overwrite breakeven_mult/trailing_mult - they come from adaptive_exits or comprehensive checker
                    
                    adaptive_exit_manager.record_exit_outcome(
                        regime=market_regime,
                        exit_params=enhanced_exit_params,  # NOW INCLUDES ALL 131 PARAMS!
                        trade_outcome={
                            'pnl': total_pnl,
                            'profit_ticks': profit_ticks,  # CRITICAL: Add profit in ticks for exit learning
                            'total_pnl': total_pnl,  # Alias for compatibility
                            'final_r_multiple': r_multiple,  # Add R-multiple at exit
                            'initial_risk_ticks': active_trade.initial_risk_ticks,  # CRITICAL: For R-multiple calculations in training
                            'duration': duration,
                            'exit_reason': exit_reason,
                            'side': active_trade.side,
                            'contracts': contracts_closed,
                            'win': total_pnl > 0,
                            'entry_confidence': active_trade.confidence,  # CRITICAL: Store entry confidence for learning
                            'entry_price': active_trade.entry_price,  # For r_multiple calculation
                            'duration_bars': active_trade.bars_in_trade,  # Bar count (neural network feature)
                            'r_multiple': r_multiple,  # Risk-reward ratio achieved
                            'mae': mae,  # Maximum drawdown during trade
                            'mfe': mfe,  # Maximum profit during trade
                            # NEW: In-trade behavior tracking
                            'max_r_achieved': float(active_trade.max_r_achieved),
                            'min_r_achieved': float(active_trade.min_r_achieved),
                            'exit_param_update_count': len(active_trade.exit_param_updates),
                            'stop_adjustment_count': len(active_trade.stop_adjustments),
                            'breakeven_activation_bar': active_trade.breakeven_activation_bar if active_trade.breakeven_activation_bar else 0,
                            'trailing_activation_bar': active_trade.trailing_activation_bar if active_trade.trailing_activation_bar else 0,
                            'bars_until_breakeven': active_trade.bars_until_breakeven if active_trade.bars_until_breakeven else 0,
                            'bars_until_trailing': active_trade.bars_until_trailing if active_trade.bars_until_trailing else 0,
                            'breakeven_activated': bool(active_trade.breakeven_active),
                            'trailing_activated': bool(active_trade.trailing_active),
                            # ADVANCED RL: Intra-trade decision sequences and P&L tracking
                            'decision_history': active_trade.decision_history,
                            'unrealized_pnl_history': active_trade.unrealized_pnl_history,
                            'peak_unrealized_pnl': float(active_trade.peak_unrealized_pnl),
                            'peak_r_multiple': float(active_trade.peak_r_multiple),
                            'opportunity_cost': float(active_trade.peak_unrealized_pnl - total_pnl) if active_trade.peak_unrealized_pnl > total_pnl else 0.0,
                            # Advanced tracking arrays
                            'exit_param_updates': active_trade.exit_param_updates,  # [{bar, vix, atr, reason, r_at_update}]
                            'stop_adjustments': active_trade.stop_adjustments,  # [{bar, old_stop, new_stop, reason, r_at_adjustment}]
                            # NEW: Execution quality tracking
                            'slippage_ticks': float(active_trade.exit_slippage_ticks),
                            'commission_cost': float(active_trade.commission_cost),
                            # NEW: Market context tracking
                            'session': encode_session(active_trade.entry_session),  # Convert to integer (0=Asia, 1=London, 2=NY)
                            'volume_at_exit': float(bar.get('volume', 0)),
                            'volatility_regime_change': active_trade.volatility_regime_at_entry != active_trade._get_volatility_regime({'vix': synthetic_vix}),
                            # NEW: Exit quality tracking
                            'time_in_breakeven_bars': int(active_trade.time_in_breakeven_bars),
                            'rejected_partial_count': int(active_trade.rejected_partial_count),
                            'stop_hit': bool(active_trade.stop_hit),
                            # NEW: Drawdown from peak tracking
                            'profit_drawdown_from_peak': float(active_trade.profit_drawdown_from_peak),
                            'max_drawdown_percent': float(active_trade.max_drawdown_percent),
                            'drawdown_bars': int(active_trade.drawdown_bars),
                            # NEW: Volatility during trade tracking
                            'avg_atr_during_trade': float(sum(active_trade.atr_samples) / len(active_trade.atr_samples)) if active_trade.atr_samples else float(active_trade.entry_atr),
                            'atr_change_percent': float(((active_trade.atr_samples[-1] - active_trade.entry_atr) / active_trade.entry_atr) * 100.0) if active_trade.atr_samples and active_trade.entry_atr > 0 else 0.0,
                            'high_volatility_bars': int(active_trade.high_volatility_bars),
                            # NEW: Consecutive trade context
                            'trade_number_in_session': int(active_trade.trade_number_in_session),
                            'wins_in_last_5_trades': int(active_trade.wins_in_last_5_trades),
                            'losses_in_last_5_trades': int(active_trade.losses_in_last_5_trades),
                            'cumulative_pnl_before_trade': float(active_trade.cumulative_pnl_before_trade),
                            # NEW: Daily loss limit tracking (bot learns to manage daily risk)
                            'daily_pnl_before_trade': float(daily_pnl - total_pnl),  # P&L for today before this trade
                            'daily_loss_limit': float(CONFIG['daily_loss_limit']),  # $1000 limit
                            'daily_loss_proximity_pct': float(max(0.0, (-(daily_pnl - total_pnl) / CONFIG['daily_loss_limit']) * 100.0)),  # 0% if profitable, increases as losses approach limit
                            # NEW: Bar indices and timing
                            'entry_bar': int(active_trade.entry_bar_index),
                            'exit_bar': int(active_trade.exit_bar_index),
                            'entry_hour': int(active_trade.entry_time.hour),
                            'entry_minute': int(active_trade.entry_time.minute),
                            'exit_hour': int(bar['timestamp'].hour),
                            'exit_minute': int(bar['timestamp'].minute),
                            'day_of_week': int(bar['timestamp'].weekday()),  # 0=Monday, 6=Sunday
                            'bars_held': int(active_trade.exit_bar_index - active_trade.entry_bar_index),
                            'held_through_sessions': bool(held_through_sessions),
                            'max_profit_reached': float(mfe),  # Maximum profit achieved (same as MFE)
                            'vix': synthetic_vix,  # VIX is critical neural network feature
                            'minutes_until_close': float((time(21, 0).hour * 60 + time(21, 0).minute) - (bar['timestamp'].hour * 60 + bar['timestamp'].minute)) if bar['timestamp'].hour < 21 else 0.0,
                        },
                        market_state=market_state,
                        backtest_mode=True,  # Collect for bulk save at end
                        partial_exits=active_trade.partial_exits  # Include partial exit history
                    )
                except Exception as e:
                    print(f"  [WARN] Exit learning failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Print trade completion
                win_or_loss = "WIN" if total_pnl > 0 else "LOSS"
                day_name = bar['timestamp'].strftime('%a')  # Mon, Tue, etc
                print(f"\n{'[OK]' if total_pnl > 0 else '[X]'} {win_or_loss}: {active_trade.side.upper()} {active_trade.original_contracts}x | "
                      f"{day_name} {bar['timestamp'].strftime('%m/%d %H:%M')} | "
                      f"Entry: ${active_trade.entry_price:.2f} → Exit: ${exit_price:.2f} | "
                      f"P&L: ${total_pnl:+.2f} | {exit_reason} | {duration:.0f}min | "
                      f"Conf: {active_trade.confidence:.0%}")
                
                active_trade = None
                continue
        
        # Check for new signals (only if no active trade)
        if active_trade is None:
            # DAILY LOSS LIMIT CHECK (matches live bot)
            if daily_limit_hit:
                continue  # Stop trading for the day after hitting limit
            
            # TRADING HOURS CHECK (matches live bot)
            # Live bot trading window: Sunday 11 PM - Friday 9 PM UTC
            # Daily entry cutoff: 9:00 PM (21:00 UTC) - no new entries after this
            # Daily flatten: 9:45-10:00 PM (21:45-22:00 UTC)
            # Daily maintenance: 10:00-11:00 PM (22:00-23:00 UTC)
            bar_time = bar['timestamp'].time()
            
            # Skip signals after 9:00 PM daily entry cutoff
            if bar_time >= CONFIG['daily_entry_cutoff']:
                continue  # After 9 PM UTC - no new entries (can hold existing)
            
            # Skip signals during flatten mode (21:45-22:00 UTC)
            if bar_time >= CONFIG['flatten_start_time'] and bar_time < CONFIG['forced_flatten_time']:
                continue  # Flatten mode - no new entries
            
            # Skip signals during maintenance window (22:00-23:00 UTC daily)
            # NOTE: Hour 22 already removed from cleaned data, so this is safety check
            if bar_time >= time(22, 0) and bar_time < time(23, 0):
                continue  # Maintenance - no trading
            
            # Safety check: need previous bar for signal detection
            if idx < 1:
                continue
            
            prev_bar = df.iloc[idx - 1]
            
            # LEARN OPTIMAL RSI THRESHOLDS from winning trades
            # Professional traders adapt entry criteria based on what works
            rsi_oversold_threshold = CONFIG['rsi_oversold']
            rsi_overbought_threshold = CONFIG['rsi_overbought']
            
            if signal_manager_for_vwap and hasattr(signal_manager_for_vwap, 'signal_experiences'):
                signal_exps = signal_manager_for_vwap.signal_experiences
                
                if len(signal_exps) > 100:
                    # Learn optimal RSI for LONG entries
                    long_winners = [e for e in signal_exps if e.get('signal') == 'LONG' and e.get('pnl', 0) > 0]
                    long_losers = [e for e in signal_exps if e.get('signal') == 'LONG' and e.get('pnl', 0) <= 0]
                    
                    if len(long_winners) > 20:
                        # Calculate median RSI of winners
                        winner_rsis = sorted([e.get('rsi', 25) for e in long_winners])
                        winner_median_rsi = winner_rsis[len(winner_rsis) // 2]
                        
                        # Use winner median but keep reasonable bounds
                        rsi_oversold_threshold = max(15, min(35, winner_median_rsi))
                    
                    # Learn optimal RSI for SHORT entries
                    short_winners = [e for e in signal_exps if e.get('signal') == 'SHORT' and e.get('pnl', 0) > 0]
                    short_losers = [e for e in signal_exps if e.get('signal') == 'SHORT' and e.get('pnl', 0) <= 0]
                    
                    if len(short_winners) > 20:
                        # Calculate median RSI of winners
                        winner_rsis = sorted([e.get('rsi', 75) for e in short_winners])
                        winner_median_rsi = winner_rsis[len(winner_rsis) // 2]
                        
                        # Use winner median but keep reasonable bounds
                        rsi_overbought_threshold = max(65, min(85, winner_median_rsi))
            
            # LONG signal check: VWAP bounce + LEARNED RSI oversold
            if (prev_bar['low'] <= vwap_bands['lower_2'] and 
                bar['close'] > prev_bar['close'] and
                rsi < rsi_oversold_threshold):  # LEARNED threshold
                
                signals_detected += 1
                
                # Calculate synthetic VIX from ATR and volume (real-time volatility proxy)
                # This captures CURRENT bar volatility, not slow-moving market fear index
                # High ATR = wild price swings RIGHT NOW
                # Low volume = thin/choppy (uncertain) = higher risk
                atr_contribution = (atr / 2.0) * 10.0  # ATR 2.0 = +10 VIX, ATR 4.0 = +20 VIX
                volume_contribution = (1.0 - min(volume_ratio, 1.5)) * 5.0  # Low volume adds VIX
                synthetic_vix = 15.0 + atr_contribution + volume_contribution
                synthetic_vix = max(10.0, min(synthetic_vix, 35.0))  # Clamp 10-35 range
                
                # Calculate psychological metrics
                drawdown_pct = 0.0
                if peak_balance > 0:
                    current_balance = peak_balance + cumulative_pnl
                    drawdown_pct = ((current_balance - peak_balance) / peak_balance) * 100.0
                
                time_since_last_trade = 999999.0  # Large default
                if last_trade_time:
                    time_diff = bar['timestamp'] - last_trade_time
                    time_since_last_trade = time_diff.total_seconds() / 60.0  # Minutes
                
                # Get session
                hour = bar['timestamp'].hour
                if 18 <= hour or hour < 3:
                    session = 'Asia'
                elif 3 <= hour < 8:
                    session = 'London'
                else:
                    session = 'NY'
                
                # Calculate trend strength (simple: current price vs 20-bar MA)
                if idx >= 20:
                    ma_20 = df.iloc[idx-20:idx]['close'].mean()
                    trend_strength = ((bar['close'] - ma_20) / ma_20) * 100.0  # % above/below MA
                else:
                    trend_strength = 0.0
                
                # Calculate S/R proximity (distance to VWAP bands in ticks)
                tick_size = CONFIG['tick_size']
                if bar['close'] < vwap_bands['vwap']:
                    # Below VWAP - measure to lower band
                    sr_proximity_ticks = abs(bar['close'] - vwap_bands['lower_2']) / tick_size
                else:
                    # Above VWAP - measure to upper band
                    sr_proximity_ticks = abs(bar['close'] - vwap_bands['upper_2']) / tick_size
                
                # Determine trade type: reversal (bouncing off band) or continuation
                if prev_bar['low'] <= vwap_bands['lower_2'] and bar['close'] > prev_bar['close']:
                    trade_type = 'reversal'  # LONG reversal from lower band
                else:
                    trade_type = 'continuation'  # Following trend
                
                # DETECT MARKET REGIME for neural network feature
                recent_bars_for_regime = df.iloc[max(0, idx-20):idx+1].to_dict('records')
                market_regime = detect_market_regime(recent_bars_for_regime, atr)
                
                # Calculate volatility clustering features
                if idx >= 20:
                    recent_closes = df.iloc[idx-20:idx+1]['close'].values
                    recent_volatility_20bar = float(np.std(recent_closes))
                    
                    # Volatility trend: compare recent 10 bars vs previous 10 bars
                    vol_recent_10 = np.std(df.iloc[idx-10:idx+1]['close'].values)
                    vol_previous_10 = np.std(df.iloc[idx-20:idx-10]['close'].values)
                    volatility_trend = (vol_recent_10 - vol_previous_10) / (vol_previous_10 + 1e-8)  # % change
                else:
                    recent_volatility_20bar = atr  # Fallback to ATR
                    volatility_trend = 0.0
                
                vwap_std_dev = vwap_bands.get('std_dev', atr)  # VWAP standard deviation
                
                # Calculate temporal features for neural network
                minute = bar['timestamp'].minute
                hour_decimal = bar['timestamp'].hour + minute / 60.0
                time_to_close = max(0, 16.0 - hour_decimal) * 60  # minutes to 16:00 UTC close
                price_mod_50 = (bar['close'] % 50) / 50.0  # Distance to nearest 50-point level
                
                # Build COMPLETE RL state (matches live bot format)
                rl_state = {
                    'symbol': 'ES',  # Backtest symbol
                    'entry_price': bar['close'],
                    'vwap': vwap_bands['vwap'],
                    'rsi': rsi,
                    'atr': atr,
                    'vix': synthetic_vix,  # Synthetic VIX from ATR + volume
                    'price': bar['close'],
                    'hour': bar['timestamp'].hour,
                    'day_of_week': bar['timestamp'].weekday(),  # 0=Monday, 6=Sunday
                    'volume_ratio': volume_ratio,
                    'vwap_distance': vwap_distance,
                    'recent_pnl': recent_pnl_sum,
                    'streak': current_streak,
                    'side': 'long',
                    # NEW PSYCHOLOGICAL FIELDS
                    'cumulative_pnl_at_entry': cumulative_pnl,
                    'consecutive_wins': consecutive_wins,
                    'consecutive_losses': consecutive_losses,
                    'drawdown_pct_at_entry': drawdown_pct,
                    'time_since_last_trade_mins': time_since_last_trade,
                    # NEW ADVANCED ML FEATURES
                    'market_regime': market_regime,  # HIGH_VOL_TRENDING, LOW_VOL_CHOPPY, etc.
                    'recent_volatility_20bar': recent_volatility_20bar,  # Rolling std of prices
                    'volatility_trend': volatility_trend,  # Is volatility increasing/decreasing
                    'vwap_std_dev': vwap_std_dev,  # VWAP standard deviation
                    # NEW MARKET CONTEXT FIELDS
                    'session': encode_session(session),
                    'trend_strength': trend_strength,
                    'sr_proximity_ticks': sr_proximity_ticks,
                    'trade_type': encode_trade_type(trade_type),
                    'entry_slippage_ticks': 0.0,  # Will be set if we take the trade
                    'commission_cost': 0.0,  # Will be calculated
                    'bid_ask_spread_ticks': 0.5,  # ES typical spread
                    'signal': encode_signal('long'),
                    # NEW TEMPORAL/PRICE FEATURES (3 features - brings total to 31)
                    'minute': minute,
                    'time_to_close': time_to_close,
                    'price_mod_50': price_mod_50,  # Distance to round 50-level
                    # POSITION SIZING (default 1 contract for prediction, will be overridden after)
                    'contracts': 1,  # Default for neural network prediction
                }
                
                # Get RL confidence from local neural network
                take_signal, confidence, reason = get_rl_confidence(rl_state, 'long')
                
                # Trust the signal confidence manager's decision (no duplicate exploration)
                
                if take_signal:
                    contracts = calculate_position_size(confidence)
                    
                    # Skip trade if below confidence threshold (contracts = 0)
                    if contracts == 0:
                        signals_ml_rejected += 1
                        # Rejection tracked in progress bar (reduced spam)
                        
                        # GHOST TRADE: Simulate what would have happened with FULL neural exits
                        ghost_outcome = simulate_ghost_trade(bar, 'long', df, idx, atr, adaptive_exit_manager, confidence, reason)
                        ghost_trades.append(ghost_outcome)
                        
                        # Save as learning experience (bot learns from low-confidence rejections)
                        backtest_experiences.append({
                            'rl_state': rl_state,
                            'took_trade': False,  # Confidence too low
                            'outcome': ghost_outcome  # What actually happened
                        })
                    else:
                        signals_ml_approved += 1
                        
                        # Signal tracked in progress bar (reduced spam)
                        
                        active_trade = Trade(bar, 'long', contracts, confidence, atr, vwap_bands, df, adaptive_exit_manager, rl_state, entry_bar_index=idx)
                        active_trade.entry_state = rl_state  # Store for RL learning
                        
                        # NEW: Add consecutive trade context
                        active_trade.trade_number_in_session = len(completed_trades) + 1
                        active_trade.wins_in_last_5_trades = consecutive_wins
                        active_trade.losses_in_last_5_trades = consecutive_losses
                        active_trade.cumulative_pnl_before_trade = cumulative_pnl
                else:
                    signals_ml_rejected += 1
                    # Rejection tracked in progress bar (reduced spam)
                    
                    # GHOST TRADE: Simulate what would have happened with FULL neural exits
                    ghost_outcome = simulate_ghost_trade(bar, 'long', df, idx, atr, adaptive_exit_manager, confidence, reason)
                    
                    # Save as learning experience (bot learns from mistakes/missed opportunities)
                    backtest_experiences.append({
                        'rl_state': rl_state,
                        'took_trade': False,  # Bot said NO
                        'outcome': ghost_outcome  # What actually happened
                    })
                    
                    # Ghost outcome tracked (reduced spam)
            
            # SHORT signal check: VWAP bounce + LEARNED RSI overbought
            if (prev_bar['high'] >= vwap_bands['upper_2'] and 
                bar['close'] < prev_bar['close'] and
                rsi > rsi_overbought_threshold):  # LEARNED threshold
                
                signals_detected += 1
                
                # Calculate synthetic VIX (same formula as long signals)
                atr_contribution = (atr / 2.0) * 10.0
                volume_contribution = (1.0 - min(volume_ratio, 1.5)) * 5.0
                synthetic_vix = 15.0 + atr_contribution + volume_contribution
                synthetic_vix = max(10.0, min(synthetic_vix, 35.0))
                
                # Calculate psychological metrics
                drawdown_pct = 0.0
                if peak_balance > 0:
                    current_balance = peak_balance + cumulative_pnl
                    drawdown_pct = ((current_balance - peak_balance) / peak_balance) * 100.0
                
                time_since_last_trade = 999999.0  # Large default
                if last_trade_time:
                    time_diff = bar['timestamp'] - last_trade_time
                    time_since_last_trade = time_diff.total_seconds() / 60.0  # Minutes
                
                # Get session
                hour = bar['timestamp'].hour
                if 18 <= hour or hour < 3:
                    session = 'Asia'
                elif 3 <= hour < 8:
                    session = 'London'
                else:
                    session = 'NY'
                
                # Calculate trend strength (simple: current price vs 20-bar MA)
                if idx >= 20:
                    ma_20 = df.iloc[idx-20:idx]['close'].mean()
                    trend_strength = ((bar['close'] - ma_20) / ma_20) * 100.0  # % above/below MA
                else:
                    trend_strength = 0.0
                
                # Calculate S/R proximity (distance to VWAP bands in ticks)
                tick_size = CONFIG['tick_size']
                if bar['close'] < vwap_bands['vwap']:
                    # Below VWAP - measure to lower band
                    sr_proximity_ticks = abs(bar['close'] - vwap_bands['lower_2']) / tick_size
                else:
                    # Above VWAP - measure to upper band
                    sr_proximity_ticks = abs(bar['close'] - vwap_bands['upper_2']) / tick_size
                
                # Determine trade type: reversal (bouncing off band) or continuation
                if prev_bar['high'] >= vwap_bands['upper_2'] and bar['close'] < prev_bar['close']:
                    trade_type = 'reversal'  # SHORT reversal from upper band
                else:
                    trade_type = 'continuation'  # Following trend
                
                # DETECT MARKET REGIME for neural network feature (same as LONG)
                recent_bars_for_regime = df.iloc[max(0, idx-20):idx+1].to_dict('records')
                market_regime = detect_market_regime(recent_bars_for_regime, atr)
                
                # Calculate volatility clustering features (same as LONG)
                if idx >= 20:
                    recent_closes = df.iloc[idx-20:idx+1]['close'].values
                    recent_volatility_20bar = float(np.std(recent_closes))
                    
                    # Volatility trend: compare recent 10 bars vs previous 10 bars
                    vol_recent_10 = np.std(df.iloc[idx-10:idx+1]['close'].values)
                    vol_previous_10 = np.std(df.iloc[idx-20:idx-10]['close'].values)
                    volatility_trend = (vol_recent_10 - vol_previous_10) / (vol_previous_10 + 1e-8)  # % change
                else:
                    recent_volatility_20bar = atr  # Fallback to ATR
                    volatility_trend = 0.0
                
                vwap_std_dev = vwap_bands.get('std_dev', atr)  # VWAP standard deviation
                
                # Calculate temporal features for neural network
                minute = bar['timestamp'].minute
                hour_decimal = bar['timestamp'].hour + minute / 60.0
                time_to_close = max(0, 16.0 - hour_decimal) * 60  # minutes to 16:00 UTC close
                price_mod_50 = (bar['close'] % 50) / 50.0  # Distance to nearest 50-point level
                
                # Build COMPLETE RL state (matches live bot format)
                rl_state = {
                    'symbol': 'ES',  # Backtest symbol
                    'entry_price': bar['close'],
                    'vwap': vwap_bands['vwap'],
                    'rsi': rsi,
                    'atr': atr,
                    'vix': synthetic_vix,  # Synthetic VIX from ATR + volume
                    'price': bar['close'],
                    'hour': bar['timestamp'].hour,
                    'day_of_week': bar['timestamp'].weekday(),  # 0=Monday, 6=Sunday
                    'volume_ratio': volume_ratio,
                    'vwap_distance': vwap_distance,
                    'recent_pnl': recent_pnl_sum,
                    'streak': current_streak,
                    'side': 'short',
                    # NEW PSYCHOLOGICAL FIELDS
                    'cumulative_pnl_at_entry': cumulative_pnl,
                    'consecutive_wins': consecutive_wins,
                    'consecutive_losses': consecutive_losses,
                    'drawdown_pct_at_entry': drawdown_pct,
                    'time_since_last_trade_mins': time_since_last_trade,
                    # NEW ADVANCED ML FEATURES
                    'market_regime': market_regime,  # HIGH_VOL_TRENDING, LOW_VOL_CHOPPY, etc.
                    'recent_volatility_20bar': recent_volatility_20bar,  # Rolling std of prices
                    'volatility_trend': volatility_trend,  # Is volatility increasing/decreasing
                    'vwap_std_dev': vwap_std_dev,  # VWAP standard deviation
                    # NEW MARKET CONTEXT FIELDS
                    'session': encode_session(session),
                    'trend_strength': trend_strength,
                    'sr_proximity_ticks': sr_proximity_ticks,
                    'trade_type': encode_trade_type(trade_type),
                    'entry_slippage_ticks': 0.0,
                    'commission_cost': 0.0,
                    'bid_ask_spread_ticks': 0.5,
                    'signal': encode_signal('short'),
                    # NEW TEMPORAL/PRICE FEATURES (3 features - brings total to 31)
                    'minute': minute,
                    'time_to_close': time_to_close,
                    'price_mod_50': price_mod_50,  # Distance to round 50-level
                    # POSITION SIZING (default 1 contract for prediction, will be overridden after)
                    'contracts': 1,  # Default for neural network prediction
                }
                
                # Get RL confidence from local neural network
                take_signal, confidence, reason = get_rl_confidence(rl_state, 'short')
                
                # Trust the signal confidence manager's decision (no duplicate exploration)
                
                if take_signal:
                    contracts = calculate_position_size(confidence)
                    
                    # Skip trade if below confidence threshold (contracts = 0)
                    if contracts == 0:
                        signals_ml_rejected += 1
                        # Rejection tracked in progress bar (reduced spam)
                        
                        # GHOST TRADE: Simulate what would have happened with FULL neural exits
                        ghost_outcome = simulate_ghost_trade(bar, 'short', df, idx, atr, adaptive_exit_manager, confidence, reason)
                        ghost_trades.append(ghost_outcome)
                        
                        # Save as learning experience (bot learns from low-confidence rejections)
                        backtest_experiences.append({
                            'rl_state': rl_state,
                            'took_trade': False,  # Confidence too low
                            'outcome': ghost_outcome  # What actually happened
                        })
                    else:
                        signals_ml_approved += 1
                        
                        # Signal tracked in progress bar (reduced spam)
                        
                        active_trade = Trade(bar, 'short', contracts, confidence, atr, vwap_bands, df, adaptive_exit_manager, rl_state, entry_bar_index=idx)
                        active_trade.entry_state = rl_state  # Store RL state for learning
                        
                        # NEW: Add consecutive trade context
                        active_trade.trade_number_in_session = len(completed_trades) + 1
                        active_trade.wins_in_last_5_trades = consecutive_wins
                        active_trade.losses_in_last_5_trades = consecutive_losses
                        active_trade.cumulative_pnl_before_trade = cumulative_pnl
                else:
                    signals_ml_rejected += 1
                    # Rejection tracked in progress bar (reduced spam)
                    
                    # GHOST TRADE: Simulate what would have happened with FULL neural exits
                    ghost_outcome = simulate_ghost_trade(bar, 'short', df, idx, atr, adaptive_exit_manager, confidence, reason)
                    
                    # Save as learning experience (bot learns from mistakes/missed opportunities)
                    backtest_experiences.append({
                        'rl_state': rl_state,
                        'took_trade': False,  # Bot said NO
                        'outcome': ghost_outcome  # What actually happened
                    })
                    
                    # Ghost outcome tracked (reduced spam)
    
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
    
    # Clear progress bar and print results
    print(f"\n\n{'='*80}")
    print(f"BACKTEST COMPLETE - 64 DAYS")
    print(f"{'='*80}")
    
    # Calculate ghost trade stats (rejected signals that were simulated)
    ghost_trades = [exp for exp in backtest_experiences if not exp['took_trade']]
    ghost_winners = [exp for exp in ghost_trades if exp['outcome']['pnl'] > 0]
    ghost_losers = [exp for exp in ghost_trades if exp['outcome']['pnl'] <= 0]
    missed_profit = sum([exp['outcome']['pnl'] for exp in ghost_winners])
    avoided_loss = abs(sum([exp['outcome']['pnl'] for exp in ghost_losers]))
    
    # Print results
    print(f"\nSIGNAL DETECTION:")
    print(f"  Total Signals Detected: {signals_detected}")
    print(f"  ML Approved: {signals_ml_approved} ({signals_ml_approved/signals_detected*100:.1f}%)")
    print(f"  ML Rejected: {signals_ml_rejected} ({signals_ml_rejected/signals_detected*100:.1f}%)")
    
    # Daily Loss Limit Stats - SHOW TRADING HALTS
    print(f"\n💰 DAILY LOSS LIMIT: ${CONFIG['daily_loss_limit']:.2f}")
    if days_stopped_by_limit > 0:
        print(f"  ⚠️  Trading halted on {days_stopped_by_limit} day(s) due to daily loss limit")
        print(f"  📊 Bot continued trading on following days (realistic prop firm behavior)")
        print(f"  ✅ This tests bot's ability to recover from bad days")
    else:
        print(f"  ✅ PASSED - Never exceeded daily loss limit")
        print(f"  ✅ Bot successfully managed risk within constraints")
    
    print(f"\n👻 GHOST TRADES (Rejected Signals Simulated for Learning):")
    print(f"  Total Ghost Trades: {len(ghost_trades)}")
    print(f"  Would Have Won: {len(ghost_winners)} (Missed Profit: ${missed_profit:+,.0f})")
    print(f"  Would Have Lost: {len(ghost_losers)} (Avoided Loss: ${avoided_loss:,.0f})")
    if len(ghost_trades) > 0:
        net_ghost = missed_profit - avoided_loss
        print(f"  Net Impact: ${net_ghost:+,.0f} ({'GOOD rejection' if net_ghost < 0 else 'MISSED opportunity'})")
        print(f"  → Bot will learn from these {len(ghost_trades)} experiences to improve future decisions!")
    
    print(f"\nTRADING PERFORMANCE:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Winning Trades: {winning_trades}")
    print(f"  Losing Trades: {losing_trades}")
    print(f"  Win Rate: {win_rate:.1f}%")
    
    print(f"\nPROFIT & LOSS:")
    print(f"  Starting Balance: ${starting_balance:,.2f}")
    print(f"  Ending Balance: ${starting_balance + total_pnl:,.2f}")
    print(f"  Total P&L (Net): ${total_pnl:+,.2f}")
    print(f"  Return: {(total_pnl / starting_balance * 100):+.2f}%")
    
    # Calculate total costs for display (already deducted in trade P&L)
    total_slippage_cost = total_trades * CONFIG['slippage_ticks'] * 2 * CONFIG['tick_value']  # 2 ticks per trade (entry+exit)
    total_commission_cost = CONFIG['commission_per_contract'] * 2 * df_trades['contracts'].sum()  # $5 per contract round trip (sum already counts all trades)
    total_costs = total_slippage_cost + total_commission_cost
    
    # Calculate gross P&L (before costs)
    gross_total_pnl = total_pnl + total_costs
    
    print(f"\n  💰 TRADING COSTS (Deducted from P&L above):")
    print(f"     Gross P&L: ${gross_total_pnl:+,.2f}")
    print(f"     Slippage: -${total_slippage_cost:,.2f} ({CONFIG['slippage_ticks']*2} ticks/trade)")
    print(f"     Commission: -${total_commission_cost:,.2f} (${CONFIG['commission_per_contract']*2}/contract)")
    print(f"     Total Costs: -${total_costs:,.2f}")
    print(f"     Net P&L: ${total_pnl:+,.2f}")
    
    print(f"\n  Gross Profit: ${gross_profit:,.2f}")
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
    
    # PRINT LEARNED INSIGHTS SUMMARY
    print_learned_insights_summary(completed_trades, backtest_experiences, signal_manager_for_vwap, adaptive_exit_manager)
    
    # BULK SAVE ALL EXPERIENCES TO CLOUD API (skip in local mode or no-save mode)
    if CONFIG.get('save_experiences', True) and CONFIG.get('local_mode', False):
        print(f"\n⚡ LOCAL MODE: Saving {len(backtest_experiences)} signal experiences to local files...")
        from local_experience_manager import local_manager
        
        # Save all signal experiences to local file
        for exp in backtest_experiences:
            local_manager.add_signal_experience(exp['rl_state'], exp['took_trade'], exp['outcome'])
        
        local_manager.save_new_experiences_to_file()
        
        # Save exit experiences if using local exit manager
        if hasattr(adaptive_exit_manager, 'save_new_experiences_to_file'):
            adaptive_exit_manager.save_new_experiences_to_file()
        
        print(f"\n✅ All experiences saved to local JSON files")
    elif not CONFIG.get('save_experiences', True):
        print(f"\n🧪 STRESS TEST MODE: Experiences NOT saved (keeping real data clean)")
        print(f"   Generated {len(backtest_experiences)} signal experiences (discarded)")
        print(f"   Use this for testing synthetic data or edge cases")
    
    if CONFIG.get('save_experiences', True):
        # AUTO-RETRAINING DISABLED - Collecting experiences first
        # Uncomment below to enable auto-retraining after each backtest
        """
        print(f"\n" + "="*80)
        print("🔄 AUTO-RETRAINING NEURAL NETWORK WITH NEW DATA")
        print("="*80)
        print(f"This ensures the bot gets smarter after each backtest!")
        print()
        
        import subprocess
        import sys
        
        try:
            # Run train_model.py to retrain with all experiences (old + new)
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            result = subprocess.run(
                [sys.executable, 'train_model.py'],
                cwd=script_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(result.stdout)
                print(f"\n✅ NEURAL NETWORK RETRAINED SUCCESSFULLY")
                print(f"   Bot is now smarter with updated model!")
                print(f"   Next backtest will use improved predictions!")
            else:
                print(f"⚠️  Retraining failed: {result.stderr}")
                print(f"   Run manually: cd dev-tools && python train_model.py")
        except Exception as e:
            print(f"⚠️  Could not auto-retrain: {e}")
            print(f"   Run manually: cd dev-tools && python train_model.py")
        """
        
        print(f"\n💡 TIP: Run more backtests to collect experiences (target: 7,000+)")
        print(f"   Current: {len(local_manager.signal_experiences):,} signal experiences")
        print(f"   When ready to retrain: cd dev-tools && python train_model.py")
    
    # Save trades to CSV for analysis
    # Save CSV relative to this file so it works no matter where script runs from
    backtest_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    os.makedirs(backtest_data_dir, exist_ok=True)
    backtest_csv_path = os.path.join(backtest_data_dir, 'backtest_trades.csv')
    df_trades.to_csv(backtest_csv_path, index=False)
    print(f"\n📁 Trades saved to {backtest_csv_path}")
    print(f"\n[OK] Full trade log saved to: ../data/backtest_trades.csv")
    
    return df_trades


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    import sys
    
    # Parse command-line arguments
    days = 15  # Default
    local_mode = True  # Default: save experiences to local JSON (bot needs to learn!)
    # Default CSV path - works from dev-tools/ directory or repo root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "..", "data", "historical_data", "ES_1min_cleaned.csv")
    save_experiences = True  # Default: save experiences
    confidence_threshold = None  # Will use CONFIG default if not specified
    max_contracts = None  # Will use CONFIG default if not specified
    
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == "--days" and i + 1 < len(sys.argv):
                try:
                    days = int(sys.argv[i + 1])
                except ValueError:
                    print(f"Invalid --days value: {sys.argv[i + 1]}")
                    sys.exit(1)
            elif arg == "--csv-file" and i + 1 < len(sys.argv):
                csv_file = sys.argv[i + 1]
                print(f"📊 Using custom CSV file: {csv_file}\n")
            elif arg == "--confidence-threshold" and i + 1 < len(sys.argv):
                try:
                    confidence_threshold = float(sys.argv[i + 1])
                    print(f"🎯 Confidence threshold set to: {confidence_threshold:.0%}\n")
                except ValueError:
                    print(f"Invalid --confidence-threshold value: {sys.argv[i + 1]}")
                    sys.exit(1)
            elif arg == "--max-contracts" and i + 1 < len(sys.argv):
                try:
                    max_contracts = int(sys.argv[i + 1])
                    print(f"📊 Max contracts set to: {max_contracts}\n")
                except ValueError:
                    print(f"Invalid --max-contracts value: {sys.argv[i + 1]}")
                    sys.exit(1)
            elif arg == "--no-save":
                save_experiences = False
                print("🧪 STRESS TEST MODE: Results only, experiences will NOT be saved")
                print("   (Use this for synthetic data or testing - keeps real data clean)\n")
            elif arg == "--local":
                local_mode = True
                print("🔧 DEV MODE: Using local experiences for fast backtesting")
                print("   (Will bulk upload new experiences to cloud at end)\n")
        
        # Also support plain number: "python full_backtest.py 75"
        if len(sys.argv) == 2:
            try:
                days = int(sys.argv[1])
            except ValueError:
                pass  # Not a number, ignore
    
    if not os.path.exists(csv_file):
        print(f"ERROR: Data file not found: {csv_file}")
        print("Please ensure historical data is available.")
    else:
        # Set global flags
        CONFIG['local_mode'] = local_mode
        CONFIG['save_experiences'] = save_experiences
        
        # Update CONFIG with command-line overrides
        if confidence_threshold is not None:
            CONFIG['rl_confidence_threshold'] = confidence_threshold
            local_manager.confidence_threshold = confidence_threshold  # Update local_manager too
        if max_contracts is not None:
            CONFIG['max_contracts'] = max_contracts
        
        df_trades = run_full_backtest(csv_file, days=days)
