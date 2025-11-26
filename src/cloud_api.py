"""
Cloud API Client for User Bots
================================
Simple client that reports trade outcomes to cloud for data collection.
Bots make decisions locally using their own RL brain.
"""

import logging
import requests
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
        
        logger.info(f"🌐 Cloud API client initialized: {self.api_url} (data collection only)")
    
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
            logger.debug("License invalid - skipping outcome report")
            return False
        
        try:
            payload = {
                "license_key": self.license_key,
                "state": state,
                "took_trade": took_trade,
                "pnl": pnl,
                "duration": duration
            }
            
            # Add execution data if provided
            if execution_data:
                payload["execution_data"] = execution_data
            
            response = requests.post(
                f"{self.api_url}/api/rl/submit-outcome",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                total_exp = data.get('total_experiences', '?')
                win_rate = data.get('win_rate', 0) * 100
                logger.info(f"✅ Outcome reported to cloud ({total_exp} experiences, {win_rate:.0f}% WR)")
                return True
            else:
                logger.warning(f"⚠️ Failed to report outcome: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.debug(f"Non-critical: Could not report outcome to cloud: {e}")
            return False
    
    def set_license_valid(self, valid: bool):
        """
        Set license validity status.
        Only call this if you need to re-enable after fixing license issues.
        """
        self.license_valid = valid
        status = "valid" if valid else "invalid"
        logger.info(f"License marked as {status}")

