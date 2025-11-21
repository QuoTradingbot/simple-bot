"""
Development Environment for VWAP Bounce Bot

This folder contains the backtesting framework and development tools.
Separated from production code in src/ for clean architecture.
"""

__version__ = "1.0.0"

# Export backtesting components
from .backtesting import (
    BacktestConfig,
    BacktestEngine,
    HistoricalDataLoader,
    PerformanceMetrics,
    ReportGenerator,
    Trade
)

__all__ = [
    'BacktestConfig',
    'BacktestEngine',
    'HistoricalDataLoader',
    'PerformanceMetrics',
    'ReportGenerator',
    'Trade'
]
