"""
QuoTrading Bot - Core Trading Module

A sophisticated algorithmic trading bot for futures markets.
For backtesting and development, see the dev/ folder.
"""

__version__ = "1.0.0"
__author__ = "Kevin Suero"

# Core production modules
from . import config
from . import quotrading_engine as quotrading_bot  # Main trading engine
from . import broker_interface

__all__ = [
    "config",
    "quotrading_bot",  # Main engine (quotrading_engine.py)
    "broker_interface",
]
