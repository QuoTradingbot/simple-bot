"""
QuoTrading AI - Professional Trading Bot
=========================================
Hybrid architecture with local execution and cloud ML intelligence.

Architecture:
- VWAP/RSI Calculation: Local (proven Iteration 3 settings)
- ML/RL Confidence: Cloud (Azure - shared learning across all users)
- Signal Generation: Local (fast response with cloud validation)
- Trade Execution: Local (TopStep WebSocket)
- License Validation: Cloud (Render API)

Author: QuoTraders Team
Version: 2.0 (Production)
"""

import asyncio
import json
import logging
import os
import sys
from collections import deque
from datetime import datetime, time as datetime_time
from typing import Optional, Dict, Any, List
import aiohttp
import pytz

# Import local execution modules
from src.broker_interface import create_broker, BrokerInterface
from src.config import load_config
from src.notifications import NotificationManager
from src.monitoring import PerformanceMonitor
from src.session_state import SessionState

# ============================================================================
# CONFIGURATION
# ============================================================================

# Cloud API Configuration
ML_API_URL = "https://quotrading-signals.icymeadow-86b2969e.eastus.azurecontainerapps.io"
LICENSE_API_URL = "https://quotrading-license.onrender.com"

# Trading Configuration - ITERATION 3 SETTINGS (PROVEN PROFITABLE!)
VWAP_STD_DEV_1 = 2.5  # Warning zone
VWAP_STD_DEV_2 = 2.1  # ENTRY ZONE ✅
VWAP_STD_DEV_3 = 3.7  # EXIT/STOP ZONE ✅

RSI_PERIOD = 10
RSI_OVERSOLD = 35   # LONG entries ✅
RSI_OVERBOUGHT = 65  # SHORT entries ✅

# Trading hours (Eastern Time)
ENTRY_START_TIME = datetime_time(18, 0)   # 6 PM
ENTRY_END_TIME = datetime_time(16, 55)    # 4:55 PM

TIMEZONE = pytz.timezone("America/New_York")

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/customer_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOMER BOT CLASS
# ============================================================================

class QuoTradingBot:
    """QuoTrading AI - Professional automated trading system"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = load_config(config_path)
        self.running = False
        
        # Cloud API session
        self.http_session: Optional[aiohttp.ClientSession] = None
        
        # Trading components
        self.broker: Optional[BrokerInterface] = None
        self.notifications = NotificationManager(self.config)
        self.monitor = PerformanceMonitor()
        self.session_state = SessionState()
        
        # Market data tracking (for local VWAP/RSI calculation)
        self.bars_1min: deque = deque(maxlen=390)  # 1 trading day of 1-min bars
        self.current_1min_bar: Optional[Dict[str, Any]] = None
        
        # Calculated indicators (local)
        self.vwap: float = 0.0
        self.vwap_std_dev: float = 0.0
        self.vwap_bands: Dict[str, float] = {
            "upper_1": 0.0, "upper_2": 0.0, "upper_3": 0.0,
            "lower_1": 0.0, "lower_2": 0.0, "lower_3": 0.0
        }
        self.rsi: float = 50.0
        
        # Position tracking
        self.in_position = False
        self.position_side: Optional[str] = None  # 'LONG' or 'SHORT'
        self.entry_price: Optional[float] = None
        self.entry_time: Optional[datetime] = None
        self.entry_vwap: Optional[float] = None
        self.entry_rsi: Optional[float] = None
        self.ml_confidence_used: Optional[float] = None
        
        # User settings (from config)
        self.symbol = self.config.instrument_symbol  # e.g., "ES", "NQ", "CL"
        self.min_confidence_threshold = getattr(self.config, 'ml_confidence_threshold', 0.70)
        self.position_size = self.config.position_size_contracts
        
        self.license_valid = False
        
        logger.info(f"QuoTrading Bot initialized for {self.symbol}")
        logger.info(f"ML confidence threshold: {self.min_confidence_threshold:.0%}")
        logger.info(f"Position size: {self.position_size} contracts")
    
    async def validate_license(self) -> bool:
        """Validate license with Render API"""
        try:
            license_key = getattr(self.config, 'license_key', None)
            if not license_key:
                logger.error("No license key found in config")
                return False
            
            async with self.http_session.post(
                f"{LICENSE_API_URL}/validate",
                json={"license_key": license_key}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.license_valid = data.get('valid', False)
                    logger.info(f"License validation: {'VALID' if self.license_valid else 'INVALID'}")
                    return self.license_valid
                else:
                    logger.error(f"License validation failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"License validation error: {e}")
            return False
    
    # ========================================================================
    # LOCAL VWAP/RSI CALCULATION (Extracted from vwap_bounce_bot.py)
    # ========================================================================
    
    def calculate_vwap(self) -> None:
        """Calculate VWAP and standard deviation bands from 1-minute bars"""
        if len(self.bars_1min) == 0:
            return
        
        # Calculate cumulative VWAP
        total_pv = 0.0  # price * volume
        total_volume = 0.0
        
        for bar in self.bars_1min:
            typical_price = (bar["high"] + bar["low"] + bar["close"]) / 3.0
            pv = typical_price * bar["volume"]
            total_pv += pv
            total_volume += bar["volume"]
        
        if total_volume == 0:
            return
        
        # VWAP = sum(price * volume) / sum(volume)
        vwap = total_pv / total_volume
        self.vwap = vwap
        
        # Calculate standard deviation (volume-weighted)
        variance_sum = 0.0
        for bar in self.bars_1min:
            typical_price = (bar["high"] + bar["low"] + bar["close"]) / 3.0
            squared_diff = (typical_price - vwap) ** 2
            variance_sum += squared_diff * bar["volume"]
        
        variance = variance_sum / total_volume
        std_dev = variance ** 0.5
        self.vwap_std_dev = std_dev
        
        # Calculate bands using Iteration 3 settings
        self.vwap_bands["upper_1"] = vwap + (std_dev * VWAP_STD_DEV_1)
        self.vwap_bands["upper_2"] = vwap + (std_dev * VWAP_STD_DEV_2)
        self.vwap_bands["upper_3"] = vwap + (std_dev * VWAP_STD_DEV_3)
        self.vwap_bands["lower_1"] = vwap - (std_dev * VWAP_STD_DEV_1)
        self.vwap_bands["lower_2"] = vwap - (std_dev * VWAP_STD_DEV_2)
        self.vwap_bands["lower_3"] = vwap - (std_dev * VWAP_STD_DEV_3)
    
    def calculate_rsi(self, prices: List[float], period: int = RSI_PERIOD) -> Optional[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return None
        
        # Calculate price changes
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Separate gains and losses
        gains = [change if change > 0 else 0 for change in changes]
        losses = [-change if change < 0 else 0 for change in changes]
        
        # Calculate initial average gain and loss (SMA)
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        # Calculate smoothed averages (EMA style)
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        # Calculate RSI
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    def update_1min_bar(self, price: float, volume: int, timestamp: datetime) -> None:
        """Update or create 1-minute bars for VWAP calculation"""
        # Get current minute boundary
        minute_boundary = timestamp.replace(second=0, microsecond=0)
        
        if self.current_1min_bar is None or self.current_1min_bar["timestamp"] != minute_boundary:
            # Finalize previous bar if exists
            if self.current_1min_bar is not None:
                self.bars_1min.append(self.current_1min_bar)
                logger.debug(f"1min bar closed: ${self.current_1min_bar['close']:.2f}, Vol: {self.current_1min_bar['volume']}")
                
                # Calculate VWAP after new bar added
                self.calculate_vwap()
                
                # Update RSI
                if len(self.bars_1min) >= RSI_PERIOD + 1:
                    closes = [bar["close"] for bar in self.bars_1min]
                    self.rsi = self.calculate_rsi(closes, RSI_PERIOD) or 50.0
            
            # Create new bar
            self.current_1min_bar = {
                "timestamp": minute_boundary,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume
            }
        else:
            # Update existing bar
            self.current_1min_bar["high"] = max(self.current_1min_bar["high"], price)
            self.current_1min_bar["low"] = min(self.current_1min_bar["low"], price)
            self.current_1min_bar["close"] = price
            self.current_1min_bar["volume"] += volume
    
    def check_trading_hours(self) -> bool:
        """Check if we're in trading hours (24/5 except weekends)"""
        now = datetime.now(TIMEZONE)
        current_time = now.time()
        weekday = now.weekday()
        
        # Weekend check
        if weekday == 5:  # Saturday
            return False
        if weekday == 6 and current_time < ENTRY_START_TIME:  # Sunday before 6 PM
            return False
        
        # Friday close check
        if weekday == 4 and current_time >= ENTRY_END_TIME:  # Friday after 4:55 PM
            return False
        
        return True
    
    # ========================================================================
    # LOCAL SIGNAL GENERATION + CLOUD ML CONFIDENCE
    # ========================================================================
    
    async def generate_local_signal(self, current_price: float) -> Optional[Dict[str, Any]]:
        """
        Generate signal locally using VWAP/RSI + get ML confidence from cloud
        
        Returns signal dict with ML confidence or None
        """
        # Need at least 2 bars for signal generation
        if len(self.bars_1min) < 2:
            return None
        
        # Check trading hours
        if not self.check_trading_hours():
            return None
        
        # Get previous and current bars
        prev_bar = list(self.bars_1min)[-2]
        current_bar = list(self.bars_1min)[-1]
        
        # Check LONG signal conditions (Iteration 3)
        # 1. Price bounces off lower band 2 (2.1 std dev)
        # 2. RSI < 35 (oversold)
        # 3. Price closing back above lower band 2
        touched_lower = prev_bar["low"] <= self.vwap_bands["lower_2"]
        bounced_back = current_bar["close"] > self.vwap_bands["lower_2"]
        rsi_oversold = self.rsi < RSI_OVERSOLD
        price_below_vwap = current_price < self.vwap  # VWAP direction filter
        
        if touched_lower and bounced_back and rsi_oversold and price_below_vwap:
            # Get ML confidence from cloud
            ml_confidence = await self.get_ml_confidence(
                signal="LONG",
                price=current_price,
                vwap=self.vwap,
                rsi=self.rsi
            )
            
            if ml_confidence and ml_confidence >= self.min_confidence_threshold:
                logger.info(f"🟢 LONG SIGNAL | Price: ${current_price:.2f} | VWAP: ${self.vwap:.2f} | RSI: {self.rsi:.1f} | ML: {ml_confidence:.0%}")
                return {
                    "action": "LONG",
                    "price": current_price,
                    "vwap": self.vwap,
                    "rsi": self.rsi,
                    "ml_confidence": ml_confidence,
                    "stop_loss": self.vwap_bands["lower_3"],
                    "take_profit": self.vwap_bands["upper_3"]
                }
        
        # Check SHORT signal conditions (Iteration 3)
        # 1. Price bounces off upper band 2 (2.1 std dev)
        # 2. RSI > 65 (overbought)
        # 3. Price closing back below upper band 2
        touched_upper = prev_bar["high"] >= self.vwap_bands["upper_2"]
        bounced_back_short = current_bar["close"] < self.vwap_bands["upper_2"]
        rsi_overbought = self.rsi > RSI_OVERBOUGHT
        price_above_vwap = current_price > self.vwap  # VWAP direction filter
        
        if touched_upper and bounced_back_short and rsi_overbought and price_above_vwap:
            # Get ML confidence from cloud
            ml_confidence = await self.get_ml_confidence(
                signal="SHORT",
                price=current_price,
                vwap=self.vwap,
                rsi=self.rsi
            )
            
            if ml_confidence and ml_confidence >= self.min_confidence_threshold:
                logger.info(f"🔴 SHORT SIGNAL | Price: ${current_price:.2f} | VWAP: ${self.vwap:.2f} | RSI: {self.rsi:.1f} | ML: {ml_confidence:.0%}")
                return {
                    "action": "SHORT",
                    "price": current_price,
                    "vwap": self.vwap,
                    "rsi": self.rsi,
                    "ml_confidence": ml_confidence,
                    "stop_loss": self.vwap_bands["upper_3"],
                    "take_profit": self.vwap_bands["lower_3"]
                }
        
        return None
    
    async def get_ml_confidence(self, signal: str, price: float, vwap: float, rsi: float) -> Optional[float]:
        """Get ML confidence score from cloud API"""
        try:
            payload = {
                "symbol": self.symbol,
                "vwap": vwap,
                "rsi": rsi,
                "price": price,
                "volume": self.current_1min_bar["volume"] if self.current_1min_bar else 0,
                "signal": signal
            }
            
            async with self.http_session.post(
                f"{ML_API_URL}/api/ml/get_confidence",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('ml_confidence', 0.0)
                else:
                    logger.warning(f"ML API error: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting ML confidence: {e}")
            return None
    
    
    async def save_trade_experience(self, side: str, entry_price: float, exit_price: float, pnl: float):
        """Save trade experience to cloud for RL learning"""
        try:
            duration = (datetime.now(TIMEZONE) - self.entry_time).total_seconds()
            
            payload = {
                "symbol": self.symbol,
                "side": side.lower(),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "entry_vwap": self.entry_vwap,
                "entry_rsi": self.entry_rsi,
                "exit_vwap": self.vwap,
                "exit_rsi": self.rsi,
                "ml_confidence": self.ml_confidence_used,
                "duration": duration
            }
            
            async with self.http_session.post(
                f"{ML_API_URL}/api/ml/save_trade",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Trade experience saved: ID {data.get('experience_id')}")
                else:
                    logger.warning(f"Failed to save trade experience: {response.status}")
        except Exception as e:
            logger.error(f"Error saving trade experience: {e}")
    
    # ========================================================================
    # TRADE EXECUTION
    # ========================================================================
    
    async def enter_position(self, signal_data: Dict[str, Any]):
        """Enter a new position"""
        try:
            side = signal_data['action']
            current_price = signal_data['price']
            
            # Place market order
            order_side = "BUY" if side == "LONG" else "SELL"
            logger.info(f"Entering {side} position at ${current_price:.2f}, size: {self.position_size} contracts")
            
            # Execute order via broker
            order_result = self.broker.place_market_order(
                symbol=self.symbol,
                side=order_side,
                quantity=self.position_size
            )
            
            if not order_result:
                logger.error("Failed to place entry order")
                await self.notifications.send_alert("❌ ENTRY FAILED", "Order rejected by broker")
                return
            
            # Update state - track entry details for RL
            self.in_position = True
            self.position_side = side
            self.entry_price = current_price
            self.entry_time = datetime.now(TIMEZONE)
            self.entry_vwap = signal_data['vwap']
            self.entry_rsi = signal_data['rsi']
            self.ml_confidence_used = signal_data['ml_confidence']
            
            # Send notification
            await self.notifications.send_alert(
                f"🟢 ENTERED {side} @ ${current_price:.2f}",
                f"VWAP: ${signal_data['vwap']:.2f}\n"
                f"RSI: {signal_data['rsi']:.1f}\n"
                f"ML Confidence: {signal_data['ml_confidence']:.0%}\n"
                f"Stop: ${signal_data['stop_loss']:.2f}\n"
                f"Target: ${signal_data['take_profit']:.2f}"
            )
            
            logger.info(f"Position entered: {side} @ ${current_price:.2f} | ML: {signal_data['ml_confidence']:.0%}")
            
        except Exception as e:
            logger.error(f"Error entering position: {e}")
            await self.notifications.send_alert("❌ ENTRY ERROR", str(e))
    
    async def exit_position(self, reason: str = "Signal"):
        """Exit current position and save experience to cloud"""
        try:
            if not self.in_position:
                logger.warning("No position to exit")
                return
            
            # Get current market price
            current_price = self.broker.get_last_price(self.symbol)
            if not current_price:
                logger.warning("No market data available, cannot exit position")
                return
            
            # Calculate P&L (ES tick value = $50)
            tick_value = 50.0  # TODO: Make this configurable per symbol
            if self.position_side == 'LONG':
                pnl = (current_price - self.entry_price) * tick_value * self.position_size
            else:  # SHORT
                pnl = (self.entry_price - current_price) * tick_value * self.position_size
            
            logger.info(f"Exiting {self.position_side} @ ${current_price:.2f}, P&L: ${pnl:.2f}")
            
            # Close position via broker
            exit_side = "SELL" if self.position_side == 'LONG' else "BUY"
            
            order_result = self.broker.place_market_order(
                symbol=self.symbol,
                side=exit_side,
                quantity=self.position_size
            )
            
            if not order_result:
                logger.error("Failed to place exit order")
                await self.notifications.send_alert("❌ EXIT FAILED", "Order rejected by broker")
                return
            
            # Save trade experience to cloud for RL learning
            await self.save_trade_experience(
                side=self.position_side,
                entry_price=self.entry_price,
                exit_price=current_price,
                pnl=pnl
            )
            
            # Send notification
            pnl_emoji = "💰" if pnl >= 0 else "📉"
            await self.notifications.send_alert(
                f"{pnl_emoji} EXITED {self.position_side}",
                f"Entry: ${self.entry_price:.2f}\n"
                f"Exit: ${current_price:.2f}\n"
                f"P&L: ${pnl:.2f}\n"
                f"Reason: {reason}\n"
                f"ML Conf: {self.ml_confidence_used:.0%}"
            )
            
            # Reset state
            side = self.position_side
            self.in_position = False
            self.position_side = None
            self.entry_price = None
            self.entry_time = None
            self.entry_vwap = None
            self.entry_rsi = None
            self.ml_confidence_used = None
            
            logger.info(f"Position closed: {side} | P&L: ${pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error exiting position: {e}")
            await self.notifications.send_alert("❌ EXIT ERROR", str(e))
    
    
    async def on_tick(self, symbol: str, price: float, volume: int, timestamp: int):
        """
        Handle incoming tick data from broker
        
        Flow:
        1. Update 1-min bars → triggers VWAP/RSI calculation
        2. Generate signal locally (VWAP/RSI bounce conditions)
        3. Get ML confidence from cloud
        4. Execute if confidence >= threshold
        """
        try:
            # Convert timestamp to datetime
            dt = datetime.fromtimestamp(timestamp / 1000, tz=TIMEZONE)
            
            # Update 1-minute bars (triggers VWAP/RSI calc)
            self.update_1min_bar(price=price, volume=volume, timestamp=dt)
            
            # Skip if already in position
            if self.in_position:
                return
            
            # Generate signal locally with ML confidence check
            signal_data = await self.generate_local_signal(current_price=price)
            
            if signal_data:
                await self.enter_position(signal_data)
            
        except Exception as e:
            logger.error(f"Error handling tick: {e}")
    
    async def start(self):
        """Start the QuoTrading bot"""
        logger.info("=" * 70)
        logger.info("QuoTrading AI v2.0 - Professional Trading System")
        logger.info("=" * 70)
        logger.info(f"ML API: {ML_API_URL}")
        logger.info(f"License API: {LICENSE_API_URL}")
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"ML Confidence Threshold: {self.min_confidence_threshold:.0%}")
        logger.info(f"Position Size: {self.position_size} contracts")
        logger.info("=" * 70)
        logger.info("Architecture: Cloud ML/RL + Local VWAP/RSI")
        logger.info(f"VWAP Bands: {VWAP_STD_DEV_2:.1f} std (entry), {VWAP_STD_DEV_3:.1f} std (stop)")
        logger.info(f"RSI Levels: {RSI_OVERSOLD}/{RSI_OVERBOUGHT} (period {RSI_PERIOD})")
        logger.info("=" * 70)
        
        try:
            # Create HTTP session
            self.http_session = aiohttp.ClientSession()
            
            # Validate license
            if not await self.validate_license():
                logger.error("❌ Invalid license! Bot will not start.")
                await self.http_session.close()
                return
            
            logger.info("✅ License validated")
            
            # Initialize broker connection
            api_token = self.config.topstep_api_token
            username = self.config.topstep_username
            
            self.broker = create_broker(
                api_token=api_token,
                username=username,
                instrument=self.symbol
            )
            
            # Connect to broker
            connected = await self.broker.connect_async()
            if not connected:
                logger.error("❌ Failed to connect to broker!")
                await self.http_session.close()
                return
            
            logger.info(f"✅ Connected to TopStep broker ({self.symbol})")
            
            # Subscribe to market data (triggers on_tick for each price update)
            self.broker.subscribe_market_data(self.symbol, self.on_tick)
            
            # Bot is now running - on_tick handles everything
            self.running = True
            logger.info("🚀 Bot is LIVE - Calculating signals locally, using cloud ML confidence")
            logger.info("Waiting for market data...")
            
            # Keep alive loop
            while self.running:
                await asyncio.sleep(1)
            
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Gracefully shutdown the bot"""
        logger.info("Shutting down...")
        self.running = False
        
        # Close any open position
        if self.in_position:
            logger.info("Closing open position before shutdown")
            await self.exit_position(reason="Bot shutdown")
        
        # Disconnect broker
        if self.broker:
            self.broker.disconnect()
        
        # Close HTTP session
        if self.http_session:
            await self.http_session.close()
        
        logger.info("Shutdown complete")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Main entry point for QuoTrading Bot"""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Create and start bot
    bot = QuoTradingBot('config.json')
    await bot.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
