"""
Signal Generation Engine - VWAP + ML/RL Brain
==============================================
This is the HEART of the trading system - your intellectual property.

Runs in Azure cloud, generates personalized trading signals for each user.
Client sends market data + settings, receives back trading decisions.

Input: {user_id, symbol, price, volume, user_settings, positions}
Output: {action, contracts, entry, stop, target, confidence}

Architecture:
- Client polls this API every 1-5 seconds
- Generates signals based on VWAP/RSI/ML/RL
- Per-user personalization (each user gets unique signals)
- Instant updates (push to GitHub → auto-deploy)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from collections import deque
import statistics

logger = logging.getLogger(__name__)


class VWAPSignalEngine:
    """
    Core signal generation logic extracted from local bot.
    Stateless per request - client sends all needed context.
    """
    
    def __init__(self):
        """Initialize signal engine."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_vwap_bands(self, bars: List[Dict[str, Any]], std_dev_multiplier: float = 2.0) -> Dict[str, float]:
        """
        Calculate VWAP and standard deviation bands.
        
        Args:
            bars: List of 1-minute bars with {timestamp, open, high, low, close, volume}
            std_dev_multiplier: Standard deviation multiplier for bands (default 2.0)
        
        Returns:
            {vwap, upper_1, upper_2, lower_1, lower_2, std_dev}
        """
        if not bars or len(bars) < 2:
            return {"vwap": None, "upper_1": None, "upper_2": None, "lower_1": None, "lower_2": None, "std_dev": None}
        
        # Calculate VWAP: sum(price * volume) / sum(volume)
        total_pv = sum(bar["close"] * bar["volume"] for bar in bars)
        total_volume = sum(bar["volume"] for bar in bars)
        
        if total_volume == 0:
            return {"vwap": None, "upper_1": None, "upper_2": None, "lower_1": None, "lower_2": None, "std_dev": None}
        
        vwap = total_pv / total_volume
        
        # Calculate standard deviation
        squared_diffs = [(bar["close"] - vwap) ** 2 * bar["volume"] for bar in bars]
        variance = sum(squared_diffs) / total_volume
        std_dev = variance ** 0.5
        
        # Calculate bands
        bands = {
            "vwap": vwap,
            "upper_1": vwap + std_dev,
            "upper_2": vwap + (std_dev * std_dev_multiplier),
            "lower_1": vwap - std_dev,
            "lower_2": vwap - (std_dev * std_dev_multiplier),
            "std_dev": std_dev
        }
        
        return bands
    
    def calculate_rsi(self, bars: List[Dict[str, Any]], period: int = 14) -> Optional[float]:
        """
        Calculate RSI (Relative Strength Index).
        
        Args:
            bars: List of bars with close prices
            period: RSI period (default 14)
        
        Returns:
            RSI value (0-100) or None if insufficient data
        """
        if len(bars) < period + 1:
            return None
        
        # Calculate price changes
        closes = [bar["close"] for bar in bars[-(period + 1):]]
        changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        
        # Separate gains and losses
        gains = [max(change, 0) for change in changes]
        losses = [abs(min(change, 0)) for change in changes]
        
        # Calculate average gain and loss
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def check_long_signal(self, current_bar: Dict[str, Any], prev_bar: Dict[str, Any], 
                         vwap_bands: Dict[str, float], rsi: Optional[float],
                         settings: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if LONG signal conditions are met.
        
        Strategy: Buy dips in uptrends (mean reversion to VWAP from below)
        
        Args:
            current_bar: Current 1-min bar
            prev_bar: Previous 1-min bar
            vwap_bands: VWAP bands dictionary
            rsi: Current RSI value
            settings: User configuration settings
        
        Returns:
            (signal_triggered, reason)
        """
        # PRIMARY: VWAP bounce condition (touched lower band 2, bounced back)
        touched_lower = prev_bar["low"] <= vwap_bands["lower_2"]
        bounced_back = current_bar["close"] > vwap_bands["lower_2"]
        
        if not (touched_lower and bounced_back):
            return False, "No VWAP lower band bounce detected"
        
        # FILTER 1: Price below VWAP (discount/oversold confirmation)
        use_vwap_direction = settings.get("use_vwap_direction_filter", False)
        if use_vwap_direction and vwap_bands["vwap"] is not None:
            if current_bar["close"] >= vwap_bands["vwap"]:
                return False, f"Price above VWAP: {current_bar['close']:.2f} >= {vwap_bands['vwap']:.2f}"
        
        # FILTER 2: RSI extreme oversold
        use_rsi = settings.get("use_rsi_filter", True)
        rsi_oversold = settings.get("rsi_oversold", 25.0)
        if use_rsi and rsi is not None:
            if rsi >= rsi_oversold:
                return False, f"RSI not extreme: {rsi:.2f} >= {rsi_oversold}"
        
        # FILTER 3: Volume spike (if enabled in settings)
        use_volume = settings.get("use_volume_filter", True)
        if use_volume and "avg_volume" in settings:
            volume_mult = settings.get("volume_spike_multiplier", 1.5)
            avg_volume = settings["avg_volume"]
            if avg_volume > 0 and current_bar["volume"] < avg_volume * volume_mult:
                return False, f"No volume spike: {current_bar['volume']} < {avg_volume * volume_mult:.0f}"
        
        return True, f"LONG signal: VWAP bounce at {current_bar['close']:.2f} (RSI: {rsi:.1f if rsi else 'N/A'})"
    
    def check_short_signal(self, current_bar: Dict[str, Any], prev_bar: Dict[str, Any],
                          vwap_bands: Dict[str, float], rsi: Optional[float],
                          settings: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if SHORT signal conditions are met.
        
        Strategy: Sell rallies in downtrends (mean reversion to VWAP from above)
        
        Args:
            current_bar: Current 1-min bar
            prev_bar: Previous 1-min bar
            vwap_bands: VWAP bands dictionary
            rsi: Current RSI value
            settings: User configuration settings
        
        Returns:
            (signal_triggered, reason)
        """
        # PRIMARY: VWAP bounce condition (touched upper band 2, bounced back)
        touched_upper = prev_bar["high"] >= vwap_bands["upper_2"]
        bounced_back = current_bar["close"] < vwap_bands["upper_2"]
        
        if not (touched_upper and bounced_back):
            return False, "No VWAP upper band bounce detected"
        
        # FILTER 1: Price above VWAP (premium/overbought confirmation)
        use_vwap_direction = settings.get("use_vwap_direction_filter", False)
        if use_vwap_direction and vwap_bands["vwap"] is not None:
            if current_bar["close"] <= vwap_bands["vwap"]:
                return False, f"Price below VWAP: {current_bar['close']:.2f} <= {vwap_bands['vwap']:.2f}"
        
        # FILTER 2: RSI extreme overbought
        use_rsi = settings.get("use_rsi_filter", True)
        rsi_overbought = settings.get("rsi_overbought", 75.0)
        if use_rsi and rsi is not None:
            if rsi <= rsi_overbought:
                return False, f"RSI not extreme: {rsi:.2f} <= {rsi_overbought}"
        
        # FILTER 3: Volume spike (if enabled in settings)
        use_volume = settings.get("use_volume_filter", True)
        if use_volume and "avg_volume" in settings:
            volume_mult = settings.get("volume_spike_multiplier", 1.5)
            avg_volume = settings["avg_volume"]
            if avg_volume > 0 and current_bar["volume"] < avg_volume * volume_mult:
                return False, f"No volume spike: {current_bar['volume']} < {avg_volume * volume_mult:.0f}"
        
        return True, f"SHORT signal: VWAP bounce at {current_bar['close']:.2f} (RSI: {rsi:.1f if rsi else 'N/A'})"
    
    def calculate_position_size(self, account_size: float, risk_per_trade: float,
                               entry_price: float, stop_price: float,
                               tick_size: float, tick_value: float,
                               max_contracts: int = 25) -> Tuple[int, float, float]:
        """
        Calculate position size based on risk management.
        
        Args:
            account_size: Account balance
            risk_per_trade: Risk percentage per trade (e.g., 0.01 = 1%)
            entry_price: Entry price
            stop_price: Stop loss price
            tick_size: Minimum price increment
            tick_value: Dollar value per tick
            max_contracts: Maximum contracts allowed
        
        Returns:
            (contracts, stop_price, target_price)
        """
        # Calculate risk amount in dollars
        risk_dollars = account_size * risk_per_trade
        
        # Calculate stop distance in ticks
        stop_distance = abs(entry_price - stop_price)
        stop_distance_ticks = stop_distance / tick_size
        
        # Calculate position size
        # risk_dollars = contracts * stop_distance_ticks * tick_value
        # contracts = risk_dollars / (stop_distance_ticks * tick_value)
        if stop_distance_ticks == 0:
            return 0, stop_price, entry_price
        
        contracts = int(risk_dollars / (stop_distance_ticks * tick_value))
        contracts = max(1, min(contracts, max_contracts))
        
        # Calculate target (2:1 reward-risk ratio)
        target_distance = stop_distance * 2
        if entry_price > stop_price:  # Long
            target_price = entry_price + target_distance
        else:  # Short
            target_price = entry_price - target_distance
        
        return contracts, stop_price, target_price
    
    def generate_signal(self, user_id: str, symbol: str, bars: List[Dict[str, Any]],
                       current_position: Optional[Dict[str, Any]], settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point: Generate trading signal for user.
        
        Args:
            user_id: Unique user identifier
            symbol: Trading symbol (e.g., "ES")
            bars: List of recent 1-min bars (at least 20+)
            current_position: Current position {side, quantity, entry_price} or None
            settings: User settings {account_size, risk_per_trade, rsi_oversold, etc.}
        
        Returns:
            {
                action: "LONG" | "SHORT" | "HOLD" | "CLOSE",
                contracts: int,
                entry: float,
                stop: float,
                target: float,
                confidence: float,
                reason: str,
                timestamp: str
            }
        """
        try:
            # Validate input
            if not bars or len(bars) < 2:
                return {
                    "action": "HOLD",
                    "contracts": 0,
                    "entry": 0.0,
                    "stop": 0.0,
                    "target": 0.0,
                    "confidence": 0.0,
                    "reason": "Insufficient bar data",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Get current and previous bars
            current_bar = bars[-1]
            prev_bar = bars[-2]
            
            # Calculate VWAP bands
            vwap_bands = self.calculate_vwap_bands(bars, settings.get("vwap_std_dev", 2.0))
            if vwap_bands["vwap"] is None:
                return {
                    "action": "HOLD",
                    "contracts": 0,
                    "entry": 0.0,
                    "stop": 0.0,
                    "target": 0.0,
                    "confidence": 0.0,
                    "reason": "VWAP not ready",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Calculate RSI
            rsi = self.calculate_rsi(bars, settings.get("rsi_period", 14))
            
            # If already in position, check exit conditions
            if current_position and current_position.get("active"):
                # For now, let client handle exits (can add exit logic later)
                return {
                    "action": "HOLD",
                    "contracts": 0,
                    "entry": 0.0,
                    "stop": 0.0,
                    "target": 0.0,
                    "confidence": 0.0,
                    "reason": "Position active - monitoring",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Check for LONG signal
            long_signal, long_reason = self.check_long_signal(current_bar, prev_bar, vwap_bands, rsi, settings)
            if long_signal:
                # Calculate position size
                stop_distance_ticks = settings.get("stop_loss_ticks", 8)
                tick_size = settings.get("tick_size", 0.25)
                tick_value = settings.get("tick_value", 12.50)
                
                entry_price = current_bar["close"]
                stop_price = entry_price - (stop_distance_ticks * tick_size)
                
                contracts, stop_price, target_price = self.calculate_position_size(
                    account_size=settings.get("account_size", 50000),
                    risk_per_trade=settings.get("risk_per_trade", 0.01),
                    entry_price=entry_price,
                    stop_price=stop_price,
                    tick_size=tick_size,
                    tick_value=tick_value,
                    max_contracts=settings.get("max_contracts", 25)
                )
                
                return {
                    "action": "LONG",
                    "contracts": contracts,
                    "entry": entry_price,
                    "stop": stop_price,
                    "target": target_price,
                    "confidence": 0.75,  # TODO: Add ML/RL confidence scoring
                    "reason": long_reason,
                    "vwap": vwap_bands["vwap"],
                    "rsi": rsi,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Check for SHORT signal
            short_signal, short_reason = self.check_short_signal(current_bar, prev_bar, vwap_bands, rsi, settings)
            if short_signal:
                # Calculate position size
                stop_distance_ticks = settings.get("stop_loss_ticks", 8)
                tick_size = settings.get("tick_size", 0.25)
                tick_value = settings.get("tick_value", 12.50)
                
                entry_price = current_bar["close"]
                stop_price = entry_price + (stop_distance_ticks * tick_size)
                
                contracts, stop_price, target_price = self.calculate_position_size(
                    account_size=settings.get("account_size", 50000),
                    risk_per_trade=settings.get("risk_per_trade", 0.01),
                    entry_price=entry_price,
                    stop_price=stop_price,
                    tick_size=tick_size,
                    tick_value=tick_value,
                    max_contracts=settings.get("max_contracts", 25)
                )
                
                return {
                    "action": "SHORT",
                    "contracts": contracts,
                    "entry": entry_price,
                    "stop": stop_price,
                    "target": target_price,
                    "confidence": 0.75,  # TODO: Add ML/RL confidence scoring
                    "reason": short_reason,
                    "vwap": vwap_bands["vwap"],
                    "rsi": rsi,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # No signal
            return {
                "action": "HOLD",
                "contracts": 0,
                "entry": 0.0,
                "stop": 0.0,
                "target": 0.0,
                "confidence": 0.0,
                "reason": "No entry conditions met",
                "vwap": vwap_bands["vwap"],
                "rsi": rsi,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Signal generation error for user {user_id}: {e}", exc_info=True)
            return {
                "action": "HOLD",
                "contracts": 0,
                "entry": 0.0,
                "stop": 0.0,
                "target": 0.0,
                "confidence": 0.0,
                "reason": f"Error: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }


# Global engine instance
signal_engine = VWAPSignalEngine()
