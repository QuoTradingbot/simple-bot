"""
Cloud API Client for User Bots
================================
Simple client that reports trade outcomes to cloud for data collection.
Bots make decisions locally using their own RL brain.
"""

import logging
import requests
import aiohttp
import asyncio
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class CloudAPIClient:
    """
    Simple API client for user bots to report trade outcomes to cloud.
    
    User bots use this to:
    1. Report "here's what happened" after trade closes
    
    Decision-making happens locally in each bot's RL brain.
    """
    
    def __init__(self, api_url: str, license_key: str, timeout: int = 10, max_retries: int = 2):
        """
        Initialize cloud API client.
        
        Args:
            api_url: Cloud API URL (e.g., "https://quotrading-flask-api.azurewebsites.net")
            license_key: User's license key for authentication
            timeout: Request timeout in seconds (default 10s)
            max_retries: Number of retry attempts on connection failure (default 2)
        """
        self.api_url = api_url.rstrip('/')
        self.license_key = license_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.license_valid = True  # Set to False only on 401 license errors
        self.session: Optional[aiohttp.ClientSession] = None
        
        pass  # Silent - cloud API is transparent to customer

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the shared ClientSession"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close(self):
        """Close the session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def report_trade_outcome(self, state: Dict, took_trade: bool, pnl: float, duration: float, execution_data: Optional[Dict] = None) -> bool:
        """
        Report trade outcome to cloud for data collection.
        
        Args:
            state: Market state when trade was taken (17 fields: timestamp, symbol, price, etc.)
            took_trade: Whether trade was actually taken
            pnl: Profit/loss in dollars
            duration: Trade duration in seconds
            execution_data: Optional execution quality metrics (mfe, mae, exit_reason, order_type_used, entry_slippage_ticks)
        
        Returns:
            True if successfully reported, False otherwise
            
        Example:
            client.report_trade_outcome(
                state=original_state,
                took_trade=True,
                pnl=125.50,
                duration=1800,
                execution_data={"mfe": 200.0, "mae": 50.0, "exit_reason": "target_hit"}
            )
        """
        # Skip reporting if license is invalid
        if not self.license_valid:
            pass  # Silent - license check internal
            return False
        
        try:
            # FLAT FORMAT: All 24 fields at root level (no nesting)
            payload = {
                "license_key": self.license_key,
                # Market state (17 fields)
                **state,  # timestamp, symbol, price, returns, vwap_distance, vwap_slope, atr, atr_slope, rsi, macd_hist, stoch_k, volume_ratio, volume_slope, hour, session, regime, volatility_regime
                # Trade outcomes (7 fields)
                "took_trade": took_trade,
                "pnl": pnl,
                "duration": duration,
                "exploration_rate": state.get("exploration_rate", 0.0),
                "mfe": execution_data.get("mfe", 0.0) if execution_data else 0.0,
                "mae": execution_data.get("mae", 0.0) if execution_data else 0.0,
                "order_type_used": execution_data.get("order_type_used", "market") if execution_data else "market",
                "entry_slippage_ticks": execution_data.get("entry_slippage_ticks", 0.0) if execution_data else 0.0,
                "exit_reason": execution_data.get("exit_reason", "unknown") if execution_data else "unknown"
            }
            
            response = requests.post(
                f"{self.api_url}/api/rl/submit-outcome",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                total_exp = data.get('total_experiences', '?')
                win_rate = data.get('win_rate', 0) * 100
                pass  # Silent - cloud sync is transparent
                return True
            else:
                logger.warning(f"⚠️ Failed to report outcome: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            pass  # Silent - cloud sync failure is non-critical
            return False

    async def report_trade_outcome_async(self, state: Dict, took_trade: bool, pnl: float, duration: float, execution_data: Optional[Dict] = None) -> bool:
        """
        Async version of report_trade_outcome using aiohttp.
        """
        # Skip reporting if license is invalid
        if not self.license_valid:
            pass  # Silent - license check internal
            return False
        
        try:
            # FLAT FORMAT: All 24 fields at root level (no nesting)
            payload = {
                "license_key": self.license_key,
                # Market state (17 fields)
                **state,  # timestamp, symbol, price, returns, vwap_distance, vwap_slope, atr, atr_slope, rsi, macd_hist, stoch_k, volume_ratio, volume_slope, hour, session, regime, volatility_regime
                # Trade outcomes (7 fields)
                "took_trade": took_trade,
                "pnl": pnl,
                "duration": duration,
                "exploration_rate": state.get("exploration_rate", 0.0),
                "mfe": execution_data.get("mfe", 0.0) if execution_data else 0.0,
                "mae": execution_data.get("mae", 0.0) if execution_data else 0.0,
                "order_type_used": execution_data.get("order_type_used", "market") if execution_data else "market",
                "entry_slippage_ticks": execution_data.get("entry_slippage_ticks", 0.0) if execution_data else 0.0,
                "exit_reason": execution_data.get("exit_reason", "unknown") if execution_data else "unknown"
            }
            
            session = await self._get_session()
            async with session.post(
                f"{self.api_url}/api/rl/submit-outcome",
                json=payload,
                timeout=self.timeout
            ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        total_exp = data.get('total_experiences', '?')
                        win_rate = data.get('win_rate', 0) * 100
                        pass  # Silent - cloud sync is transparent
                        return True
                    else:
                        logger.warning(f"⚠️ Failed to report outcome: HTTP {response.status}")
                        return False
                
        except Exception as e:
            pass  # Silent - cloud sync failure is non-critical (async)
            return False
    
    def set_license_valid(self, valid: bool):
        """
        Set license validity status.
        Only call this if you need to re-enable after fixing license issues.
        """
        self.license_valid = valid
        status = "valid" if valid else "invalid"
        pass  # Silent - license status is internal
