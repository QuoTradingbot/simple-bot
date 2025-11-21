"""
VWAP Bounce Bot - Core Trading Module

A sophisticated VWAP-based mean reversion trading bot for production use.
For backtesting and development, see the dev/ folder.

Note: The main trading engine is in quotrading_engine.py but is aliased as 
'vwap_bounce_bot' for backward compatibility with existing code that imports
'from src import vwap_bounce_bot'.
"""

__version__ = "1.0.0"
__author__ = "Kevin Suero"

# Core production modules
from . import config
from . import quotrading_engine as vwap_bounce_bot  # Main engine aliased for compatibility
from . import broker_interface

__all__ = [
    "config",
    "vwap_bounce_bot",  # Actually quotrading_engine.py
    "broker_interface",
]
