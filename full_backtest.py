# -*- coding: utf-8 -*-
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
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
    
    # Time-based exits - UTC schedule (ES maintenance 21:00-22:00 UTC daily)
    # Sunday opens 23:00 UTC, Friday closes 21:00 UTC
    "daily_entry_cutoff": time(20, 0),    # 20:00 UTC - no new entries (1 hr before maintenance)
    "flatten_start_time": time(20, 45),  # 20:45 UTC - flatten mode starts  
    "forced_flatten_time": time(21, 0),   # 21:00 UTC - force close before maintenance window
    
    # ITERATION 3 - ATR settings (MATCHES LIVE BOT)
    "atr_period": 14,
    "atr_stop_multiplier": 3.6,  # Iteration 3 - Stop at 3.6x ATR (matches stop_loss_atr_multiplier)
    # NO HARDCODED TARGET - Learned partials control ALL profit exits (2R, 3R, 5R adaptive)
    
    # ITERATION 3 - RSI settings
    "rsi_period": 10,  # Fast RSI
    "rsi_oversold": 25.0,  # LONG entry threshold (more extreme oversold)
    "rsi_overbought": 75.0,  # SHORT entry threshold (more extreme overbought)
}


# ========================================
# LOCAL RL CONFIDENCE (Pattern Matching from Local JSON Files)
# ========================================

from local_experience_manager import local_manager

def get_rl_confidence(rl_state: Dict, side: str) -> Tuple[bool, float, str]:
    """
    Get ML confidence from LOCAL pattern matching only.
    Uses local_experience_manager.py to match against saved experiences.
    NO cloud API calls - 100% offline backtesting.
    """
    return local_manager.get_signal_confidence(rl_state, side.upper())


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


def calculate_vwap_bands(df_day: pd.DataFrame, signal_manager=None) -> Dict:
    """Calculate daily VWAP with standard deviation bands."""
    # Ensure we have a proper copy to avoid SettingWithCopyWarning
    df_work = df_day.copy().reset_index(drop=True)
    
    df_work['typical_price'] = (df_work['high'] + df_work['low'] + df_work['close']) / 3
    df_work['tpv'] = df_work['typical_price'] * df_work['volume']
    
    cumulative_tpv = df_work['tpv'].cumsum()
    cumulative_volume = df_work['volume'].cumsum()
    
    vwap = cumulative_tpv / cumulative_volume
    
    # Standard deviation
    df_work['vwap'] = vwap
    df_work['price_diff_sq'] = (df_work['typical_price'] - df_work['vwap']) ** 2
    df_work['weighted_diff_sq'] = df_work['price_diff_sq'] * df_work['volume']
    
    cumulative_weighted_diff_sq = df_work['weighted_diff_sq'].cumsum()
    variance = cumulative_weighted_diff_sq / cumulative_volume
    std_dev = np.sqrt(variance)
    
    latest_vwap = vwap.iloc[-1]
    latest_std = std_dev.iloc[-1]
    
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
    
    # ITERATION 3 band multipliers (LEARNED entry zone)
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
        
        # EXECUTION QUALITY TRACKING
        self.entry_slippage_ticks = 0.0  # Difference from intended entry
        self.exit_slippage_ticks = 0.0  # Difference from intended exit
        self.commission_cost = 0.0  # Total commissions paid
        self.entry_bid_ask_spread_ticks = 0.0  # Spread at entry
        self.exit_bid_ask_spread_ticks = 0.0  # Spread at exit
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
                # Build market state for exit manager
                market_state = {
                    'vix': synthetic_vix,  # Synthetic VIX from ATR + volume
                    'atr': atr,
                    'hour': int(bar['timestamp'].hour),
                    'volume_ratio': volume_ratio,
                    'rsi': float(self.entry_market_state.get('rsi', 50)),
                    'vwap_distance': float(self.entry_market_state.get('vwap_distance', 0))
                }
                
                position = {
                    'entry_price': float(self.entry_price),
                    'side': self.side,
                    'entry_time': str(self.entry_time)  # Convert Timestamp to string
                }
                
                # Use LOCAL exit manager only
                exit_params = self.adaptive_manager.get_adaptive_exit_params(
                    market_state=market_state,
                    position=position,
                    entry_confidence=float(self.confidence)
                )
                
                # Convert response to expected format (handle None)
                if exit_params:
                    self.current_exit_params = {
                        'breakeven_threshold_ticks': exit_params.get('breakeven_threshold_ticks', 9),
                        'trailing_distance_ticks': exit_params.get('trailing_distance_ticks', 12),
                        'stop_mult': exit_params.get('stop_mult', 3.6),
                        'partial_1_r': exit_params.get('partial_1_r', 2.0),
                        'partial_1_pct': exit_params.get('partial_1_pct', 0.5),
                        'partial_2_r': exit_params.get('partial_2_r', 3.0),
                        'partial_2_pct': exit_params.get('partial_2_pct', 0.3),
                        'partial_3_r': exit_params.get('partial_3_r', 5.0),
                        'partial_3_pct': exit_params.get('partial_3_pct', 0.2),
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
                # Fallback: Use adaptive manager defaults (learned from experiences)
                if hasattr(self.adaptive_manager, '_get_default_params'):
                    self.current_exit_params = self.adaptive_manager._get_default_params()
                else:
                    # Ultimate fallback if manager has no default method
                    self.current_exit_params = {
                        'breakeven_threshold_ticks': 12,
                        'trailing_distance_ticks': 15,
                        'stop_mult': 4.0,
                        'partial_1_r': 1.5,
                        'partial_1_pct': 0.5,
                        'partial_2_r': 2.5,
                        'partial_2_pct': 0.3,
                        'partial_3_r': 4.0,
                        'partial_3_pct': 0.2,
                    }
        else:
            # No adaptive manager - use conservative defaults
            self.current_exit_params = {
                'breakeven_threshold_ticks': 12,
                'trailing_distance_ticks': 15,
                'stop_mult': 4.0,
                'partial_1_r': 1.5,
                'partial_1_pct': 0.5,
                'partial_2_r': 2.5,
                'partial_2_pct': 0.3,
                'partial_3_r': 4.0,
                'partial_3_pct': 0.2,
            }
        
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
            self.stop_hit = True
            return ('stop_loss', self.stop_price, self.contracts)
        if self.side == 'short' and bar['high'] >= self.stop_price:
            self.stop_hit = True
            return ('stop_loss', self.stop_price, self.contracts)
        
        # 4. ADAPTIVE BREAKEVEN PROTECTION (uses learned threshold)
        if not self.breakeven_active and profit_ticks >= breakeven_threshold:
            self.breakeven_active = True
            self.in_breakeven_zone = True  # Mark as risk-free
            old_stop = self.stop_price
            self.stop_price = self.entry_price
            
            # Track breakeven activation for RL learning (NEW)
            self.breakeven_activation_bar = self.bars_in_trade
            self.bars_until_breakeven = self.bars_in_trade
            self.stop_adjustments.append({
                'bar': self.bars_in_trade,
                'old_stop': float(old_stop),
                'new_stop': float(self.stop_price),
                'reason': 'breakeven_activation',
                'r_at_adjustment': float(r_multiple)
            })
        
        # Track time in breakeven before trailing activates
        if self.in_breakeven_zone and not self.trailing_active:
            self.time_in_breakeven_bars += 1
        
        # 5. ADAPTIVE TRAILING STOP (uses learned trail distance)
        if self.breakeven_active:
            self.update_trailing_stop_adaptive(current_price, trailing_distance_ticks)
        
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


def simulate_ghost_trade(entry_bar: Dict, side: str, df: pd.DataFrame, start_idx: int, atr: float) -> Dict:
    """
    Simulate what WOULD have happened if we took a rejected signal.
    This allows the bot to learn from missed opportunities.
    
    Args:
        entry_bar: Bar where signal occurred
        side: 'long' or 'short'
        df: Full dataframe for forward simulation
        start_idx: Index where signal occurred
        atr: ATR at signal time
        
    Returns:
        Dict with simulated outcome (pnl, duration, exit_reason)
    """
    entry_price = entry_bar['close']
    stop_distance = atr * 3.6  # Same as real trades
    
    # Define stop loss levels
    if side == 'long':
        stop_loss = entry_price - stop_distance
    else:  # short
        stop_loss = entry_price + stop_distance
    
    # Simulate forward from entry
    max_bars_to_check = 200  # Don't simulate forever (max ~3 hours)
    
    for i in range(start_idx + 1, min(start_idx + max_bars_to_check, len(df))):
        bar = df.iloc[i]
        duration_min = (bar['timestamp'] - entry_bar['timestamp']).total_seconds() / 60
        
        # Check if stop loss hit
        if side == 'long':
            if bar['low'] <= stop_loss:
                # Stopped out
                pnl = (stop_loss - entry_price) * 50 * 1  # 1 contract, $50 per point
                return {
                    'pnl': pnl,
                    'exit_price': stop_loss,
                    'duration_min': duration_min,
                    'exit_reason': 'ghost_stop_loss',
                    'took_trade': False  # Ghost trade marker
                }
            # Simple target: 2R winner
            target_price = entry_price + (stop_distance * 2)
            if bar['high'] >= target_price:
                pnl = (target_price - entry_price) * 50 * 1
                return {
                    'pnl': pnl,
                    'exit_price': target_price,
                    'duration_min': duration_min,
                    'exit_reason': 'ghost_target',
                    'took_trade': False
                }
        else:  # short
            if bar['high'] >= stop_loss:
                pnl = (entry_price - stop_loss) * 50 * 1
                return {
                    'pnl': pnl,
                    'exit_price': stop_loss,
                    'duration_min': duration_min,
                    'exit_reason': 'ghost_stop_loss',
                    'took_trade': False
                }
            target_price = entry_price - (stop_distance * 2)
            if bar['low'] <= target_price:
                pnl = (entry_price - target_price) * 50 * 1
                return {
                    'pnl': pnl,
                    'exit_price': target_price,
                    'duration_min': duration_min,
                    'exit_reason': 'ghost_target',
                    'took_trade': False
                }
    
    # Timed out without hitting stop or target
    last_bar = df.iloc[min(start_idx + max_bars_to_check - 1, len(df) - 1)]
    if side == 'long':
        pnl = (last_bar['close'] - entry_price) * 50 * 1
    else:
        pnl = (entry_price - last_bar['close']) * 50 * 1
    
    return {
        'pnl': pnl,
        'exit_price': last_bar['close'],
        'duration_min': max_bars_to_check,
        'exit_reason': 'ghost_timeout',
        'took_trade': False
    }


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
    
    # Collect experiences during backtest for bulk save at end
    backtest_experiences = []
    print(f"Partial Exits: ADAPTIVE (learned from market context)")
    print(f"  - Aggressive: 70% @ 2R, 25% @ 3R (choppy/overbought)")
    print(f"  - Hold Full: 0% @ 3R, 30% @ 4R, 70% @ 6R (trending)")
    print(f"  - Balanced: 50% @ 2R, 30% @ 3R, 20% @ 5R (normal)")
    print(f"  - NO HARDCODED PROFIT TARGET - Partials control ALL exits")
    print("=" * 80)
    print()
    
    # Load local experiences if in local mode
    if CONFIG.get('local_mode', False):
        print("🔧 LOADING LOCAL EXPERIENCES FOR DEV MODE...")
        from local_experience_manager import local_manager
        if local_manager.load_experiences():
            counts = local_manager.get_experience_count()
            print(f"   ✅ Loaded {counts['signal']:,} signal + {counts['exit']:,} exit = {counts['total']:,} total experiences")
            print(f"   📊 Local pattern matching ready (10x faster than cloud API)")
            signal_manager_for_vwap = local_manager  # Pass to VWAP function
        else:
            print(f"   ❌ Failed to load local experiences - switching to cloud mode")
            CONFIG['local_mode'] = False
            signal_manager_for_vwap = None
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
    print("⚡ LOCAL MODE: Using local exit manager (pattern matching from local data)")
    from local_exit_manager import local_exit_manager
    local_exit_manager.load_experiences()
    adaptive_exit_manager = local_exit_manager  # Use local manager only
    print(f"  ✅ Local exit experiences loaded - NO cloud API calls")
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
    
    # PSYCHOLOGICAL TRACKING for signal RL
    consecutive_wins = 0
    consecutive_losses = 0
    cumulative_pnl = 0.0
    peak_balance = 0.0
    last_trade_time = None
    
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
        current_date = bar['timestamp'].date()
        
        # Get daily data for VWAP (optimize date comparison)
        day_mask = df['timestamp'].dt.date == current_date
        day_indices = df.index[day_mask]
        if len(day_indices) == 0:
            continue
        
        day_start_idx = day_indices[0]
        df_day = df.iloc[day_start_idx:idx + 1]
        
        if len(df_day) < 30:
            continue  # Need enough bars for VWAP
        
        vwap_bands = calculate_vwap_bands(df_day.copy(), signal_manager_for_vwap)
        
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
                        
                        # Enhance exit_params with current market context
                        enhanced_exit_params = active_trade.current_exit_params.copy()
                        enhanced_exit_params['current_atr'] = atr  # ATR at exit
                        enhanced_exit_params['breakeven_mult'] = enhanced_exit_params.get('breakeven_threshold_ticks', 12) / (atr / CONFIG['tick_size']) if atr > 0 else 1.0
                        enhanced_exit_params['trailing_mult'] = enhanced_exit_params.get('trailing_distance_ticks', 15) / (atr / CONFIG['tick_size']) if atr > 0 else 1.0
                        
                        adaptive_exit_manager.record_exit_outcome(
                            regime=active_trade.current_exit_params.get('market_regime', 'UNKNOWN'),
                            exit_params=enhanced_exit_params,  # Include ATR and multipliers
                            trade_outcome={
                                'pnl': total_pnl,
                                'duration': duration,
                                'exit_reason': exit_reason,
                                'side': active_trade.side,
                                'contracts': contracts_closed,
                                'win': total_pnl > 0,
                                'entry_confidence': active_trade.confidence,  # CRITICAL: Store entry confidence for learning
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
                                # Advanced tracking arrays
                                'exit_param_updates': active_trade.exit_param_updates,  # [{bar, vix, atr, reason, r_at_update}]
                                'stop_adjustments': active_trade.stop_adjustments,  # [{bar, old_stop, new_stop, reason, r_at_adjustment}]
                                # NEW: Execution quality tracking
                                'slippage_ticks': float(active_trade.exit_slippage_ticks),
                                'commission_cost': float(active_trade.commission_cost),
                                'bid_ask_spread_ticks': float(active_trade.exit_bid_ask_spread_ticks),
                                # NEW: Market context tracking
                                'session': str(active_trade.entry_session),
                                'volume_at_exit': float(bar.get('volume', 0)),
                                'volatility_regime_change': active_trade.volatility_regime_at_entry != active_trade._get_volatility_regime({'vix': synthetic_vix}),
                                # NEW: Exit quality tracking
                                'time_in_breakeven_bars': int(active_trade.time_in_breakeven_bars),
                                'rejected_partial_count': int(active_trade.rejected_partial_count),
                                'stop_hit': bool(active_trade.stop_hit),
                            },
                            market_state=market_state,
                            backtest_mode=True,  # Collect for bulk save at end
                            partial_exits=active_trade.partial_exits  # Include partial exit history
                        )
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
            # Live bot trading window: Sunday 11 PM - Friday 9 PM UTC
            # Daily entry cutoff: 8:00 PM (20:00 UTC) - no new entries after this
            # Daily flatten: 8:45-9:00 PM (20:45-21:00 UTC)
            # Daily maintenance: 9:00-10:00 PM (21:00-22:00 UTC)
            bar_time = bar['timestamp'].time()
            
            # Skip signals after 8:00 PM daily entry cutoff
            if bar_time >= CONFIG['daily_entry_cutoff']:
                continue  # After 8 PM UTC - no new entries (can hold existing)
            
            # Skip signals during flatten mode (20:45-21:00 UTC)
            if bar_time >= CONFIG['flatten_start_time'] and bar_time < CONFIG['forced_flatten_time']:
                continue  # Flatten mode - no new entries
            
            # Skip signals during maintenance window (21:00-22:00 UTC daily)
            if bar_time >= time(21, 0) and bar_time < time(22, 0):
                continue  # Maintenance - no trading
            
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
                    # NEW MARKET CONTEXT FIELDS
                    'session': session,
                    'trend_strength': trend_strength,
                    'sr_proximity_ticks': sr_proximity_ticks,
                    'trade_type': trade_type,
                    'entry_slippage_ticks': 0.0,  # Will be set if we take the trade
                    'commission_cost': 0.0,  # Will be calculated
                    'bid_ask_spread_ticks': 0.5,  # ES typical spread
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
                    
                    # GHOST TRADE: Simulate what would have happened
                    ghost_outcome = simulate_ghost_trade(bar, 'long', df, idx, atr)
                    ghost_outcome['confidence'] = confidence  # Store rejection confidence
                    
                    # Save as learning experience (bot learns from mistakes/missed opportunities)
                    backtest_experiences.append({
                        'rl_state': rl_state,
                        'took_trade': False,  # Bot said NO
                        'outcome': ghost_outcome  # What actually happened
                    })
                    
                    # Show if we missed a winner
                    if ghost_outcome['pnl'] > 0:
                        print(f"    👻 GHOST: Would have WON ${ghost_outcome['pnl']:.0f} ({ghost_outcome['exit_reason']}) - BOT LEARNS FROM THIS")
            
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
                    # NEW MARKET CONTEXT FIELDS
                    'session': session,
                    'trend_strength': trend_strength,
                    'sr_proximity_ticks': sr_proximity_ticks,
                    'trade_type': trade_type,
                    'entry_slippage_ticks': 0.0,
                    'commission_cost': 0.0,
                    'bid_ask_spread_ticks': 0.5,
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
                    
                    # GHOST TRADE: Simulate what would have happened
                    ghost_outcome = simulate_ghost_trade(bar, 'short', df, idx, atr)
                    ghost_outcome['confidence'] = confidence  # Store rejection confidence
                    
                    # Save as learning experience (bot learns from mistakes/missed opportunities)
                    backtest_experiences.append({
                        'rl_state': rl_state,
                        'took_trade': False,  # Bot said NO
                        'outcome': ghost_outcome  # What actually happened
                    })
                    
                    # Show if we missed a winner
                    if ghost_outcome['pnl'] > 0:
                        print(f"    👻 GHOST: Would have WON ${ghost_outcome['pnl']:.0f} ({ghost_outcome['exit_reason']}) - BOT LEARNS FROM THIS")
    
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
    
    # BULK SAVE ALL EXPERIENCES TO CLOUD API (skip in local mode)
    if CONFIG.get('local_mode', False):
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
    
    # Save trades to CSV for analysis
    df_trades.to_csv('data/backtest_trades.csv', index=False)
    print(f"\n[OK] Full trade log saved to: data/backtest_trades.csv")
    
    return df_trades


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    import sys
    
    # Parse command-line arguments
    days = 15  # Default
    local_mode = False  # Default: use cloud API
    
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == "--days" and i + 1 < len(sys.argv):
                try:
                    days = int(sys.argv[i + 1])
                except ValueError:
                    print(f"Invalid --days value: {sys.argv[i + 1]}")
                    sys.exit(1)
            elif arg == "--local":
                local_mode = True
                print("🔧 DEV MODE: Using local experiences for fast backtesting")
                print("   (Will bulk upload new experiences to cloud at end)\n")
    
    csv_file = "data/historical_data/ES_1min.csv"
    
    if not os.path.exists(csv_file):
        print(f"ERROR: Data file not found: {csv_file}")
        print("Please ensure historical data is available.")
    else:
        # Set global flag for local mode
        CONFIG['local_mode'] = local_mode
        df_trades = run_full_backtest(csv_file, days=days)
