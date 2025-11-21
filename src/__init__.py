"""
VWAP Bounce Bot - Core Trading Module

A sophisticated VWAP-based mean reversion trading bot for production use.
For backtesting and development, see the dev/ folder.
"""

__version__ = "1.0.0"
__author__ = "Kevin Suero"

# Core production modules
from . import config
from . import quotrading_engine as vwap_bounce_bot  # Alias for backward compatibility
from . import broker_interface

__all__ = [
    "config",
    "vwap_bounce_bot", 
    "broker_interface",
]
