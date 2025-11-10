"""
Dashboard Display Module for QuoTrading AI Bot
===============================================
Fixed dashboard view that updates in place without scrolling.
Shows bot status, settings, and real-time market data for selected symbols.

Compatible with Windows PowerShell, Linux terminals, and macOS.
"""

import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import platform


class Dashboard:
    """Fixed dashboard display that updates in place."""
    
    def __init__(self, symbols: List[str], config: Dict[str, Any]):
        """
        Initialize the dashboard.
        
        Args:
            symbols: List of trading symbols to display
            config: Bot configuration dictionary
        """
        self.symbols = symbols
        self.config = config
        self.running = False
        self.symbol_data = {symbol: self._get_default_symbol_data() for symbol in symbols}
        
        # Bot-level data
        self.bot_data = {
            "version": "v2.0",
            "server_status": "✓",
            "server_latency": "45ms",
            "server_latency_ms": 45,
            "account_balance": config.get("account_size", 50000),
            "max_contracts": config.get("max_contracts", 3),
            "daily_loss_limit": config.get("daily_loss_limit", 1000),
            "confidence_threshold": config.get("confidence_threshold", 65),
            "recovery_mode": config.get("recovery_mode", False),
            "confidence_trading": config.get("confidence_trading", False),
            "critical_errors": [],  # Track critical errors for display
            "contract_adjustment": None,  # Track contract adjustments (confidence/recovery mode)
            "fomc_active": False,  # FOMC/economic event block active
            "fomc_message": "",  # FOMC block message
        }
        
        # Platform detection
        self.is_windows = platform.system() == "Windows"
        
        # Initialize display
        self._init_display()
    
    def _init_display(self):
        """Initialize the display (clear screen and set up)."""
        self._clear_screen()
        # Hide cursor for cleaner display
        if self.is_windows:
            os.system("")  # Enable ANSI escape codes on Windows 10+
        else:
            sys.stdout.write("\033[?25l")  # Hide cursor
            sys.stdout.flush()
    
    def _clear_screen(self):
        """Clear the terminal screen."""
        if self.is_windows:
            os.system('cls')
        else:
            os.system('clear')
    
    def _move_cursor_home(self):
        """Move cursor to top-left corner."""
        # ANSI escape code to move cursor to home position
        sys.stdout.write("\033[H")
        sys.stdout.flush()
    
    def _get_default_symbol_data(self) -> Dict[str, Any]:
        """Get default data structure for a symbol."""
        return {
            "market_status": "LOADING",
            "maintenance_in": "-- h --m",
            "bid": 0.0,
            "bid_size": 0,
            "ask": 0.0,
            "ask_size": 0,
            "spread": 0.0,
            "condition": "Initializing...",
            "position": "FLAT",
            "position_qty": 0,
            "position_side": None,
            "pnl_today": 0.0,
            "pnl_percent": 0.0,
            "last_signal": "--",
            "last_signal_time": "",
            "last_signal_confidence": "",
            "last_signal_approved": None,  # True=approved, False=rejected, None=no signal
            "last_rejected_signal": "",  # Last rejected signal info
            "skip_reason": "",  # Why bot is not trading (e.g., "Market closed", "Waiting for pullback", etc.)
            "status": "Starting..."
        }
    
    def update_symbol_data(self, symbol: str, data: Dict[str, Any]):
        """
        Update data for a specific symbol.
        
        Args:
            symbol: Symbol to update
            data: Dictionary of data to update (partial updates supported)
        """
        if symbol in self.symbol_data:
            self.symbol_data[symbol].update(data)
    
    def update_bot_data(self, data: Dict[str, Any]):
        """
        Update bot-level data.
        
        Args:
            data: Dictionary of data to update (partial updates supported)
        """
        self.bot_data.update(data)
    
    def add_critical_error(self, error_message: str):
        """
        Add a critical error to be displayed.
        
        Args:
            error_message: The error message to display
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        error_entry = f"{timestamp} - {error_message}"
        
        # Keep only last 3 critical errors
        if len(self.bot_data["critical_errors"]) >= 3:
            self.bot_data["critical_errors"].pop(0)
        
        self.bot_data["critical_errors"].append(error_entry)
    
    def clear_critical_errors(self):
        """Clear all critical errors from display."""
        self.bot_data["critical_errors"] = []
    
    def _format_symbol_name(self, symbol: str) -> str:
        """
        Get formatted name for a symbol.
        
        Args:
            symbol: Symbol code (e.g., 'MES', 'NQ')
        
        Returns:
            Formatted name
        """
        symbol_names = {
            "ES": "E-mini S&P 500",
            "MES": "Micro E-mini S&P 500",
            "NQ": "E-mini Nasdaq",
            "MNQ": "Micro E-mini Nasdaq",
            "YM": "E-mini Dow",
            "MYM": "Micro E-mini Dow",
            "RTY": "E-mini Russell 2000",
            "M2K": "Micro E-mini Russell 2000",
            "CL": "Crude Oil",
            "GC": "Gold",
            "NG": "Natural Gas",
            "6E": "Euro FX",
            "ZN": "10-Year Treasury Note",
            "MBTX": "Micro Bitcoin"
        }
        return symbol_names.get(symbol, symbol)
    
    def _render_header(self) -> List[str]:
        """Render the header section."""
        lines = []
        lines.append("=" * 60)
        
        # Determine server status icon based on latency
        latency_ms = self.bot_data.get('server_latency_ms', 999)
        if latency_ms < 100:
            server_icon = "✓"
        elif latency_ms < 200:
            server_icon = "⚠"
        else:
            server_icon = "✗"
        
        lines.append(f"QuoTrading AI Bot {self.bot_data['version']} | "
                    f"Server: {server_icon} {self.bot_data['server_latency']} | "
                    f"Account: ${self.bot_data['account_balance']:,.0f}")
        lines.append("=" * 60)
        return lines
    
    def _render_settings(self) -> List[str]:
        """Render the active settings section."""
        lines = []
        lines.append("")
        lines.append("Active Settings:")
        lines.append(f"  Max Contracts: {self.bot_data['max_contracts']} per symbol")
        lines.append(f"  Daily Loss Limit: ${self.bot_data['daily_loss_limit']:,.0f}")
        
        # Confidence mode display
        confidence_mode = "Standard"
        if self.bot_data.get("confidence_trading"):
            confidence_mode = f"Confidence Trading ({self.bot_data['confidence_threshold']}%)"
        else:
            confidence_mode = f"Standard ({self.bot_data['confidence_threshold']}%)"
        lines.append(f"  Confidence Mode: {confidence_mode}")
        
        # Recovery mode display
        recovery_status = "Enabled" if self.bot_data.get("recovery_mode") else "Disabled"
        lines.append(f"  Recovery Mode: {recovery_status}")
        
        return lines
    
    def _render_fomc_notification(self) -> List[str]:
        """Render FOMC/economic event notification if active."""
        lines = []
        
        if self.bot_data.get("fomc_active"):
            msg = self.bot_data.get("fomc_message", "Economic event - halting new trades")
            lines.append("")
            lines.append("⚠️  ECONOMIC EVENT BLOCK ACTIVE:")
            lines.append(f"  {msg}")
            lines.append("  No new trades during this period")
            lines.append("")
        
        return lines
    
    def _render_contract_adjustment(self) -> List[str]:
        """Render contract adjustment notification if active."""
        lines = []
        
        if self.bot_data.get("contract_adjustment"):
            adj = self.bot_data["contract_adjustment"]
            lines.append("")
            lines.append("⚙️  CONTRACT ADJUSTMENT ACTIVE:")
            lines.append(f"  {adj}")
            lines.append("")
        
        return lines
    
    def _render_symbol(self, symbol: str) -> List[str]:
        """
        Render the display section for a single symbol.
        
        Args:
            symbol: Symbol to render
        
        Returns:
            List of lines to display
        """
        data = self.symbol_data[symbol]
        lines = []
        
        # Symbol header
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"{symbol} | {self._format_symbol_name(symbol)}")
        lines.append("=" * 60)
        
        # Market status line
        lines.append(f"Market: {data['market_status']} | Maintenance in: {data['maintenance_in']}")
        
        # Bid/Ask line
        bid_str = f"${data['bid']:.2f} x {data['bid_size']}"
        ask_str = f"${data['ask']:.2f} x {data['ask_size']}"
        spread_str = f"${data['spread']:.2f}"
        lines.append(f"Bid: {bid_str} | Ask: {ask_str} | Spread: {spread_str}")
        
        # Market condition
        lines.append(f"Condition: {data['condition']}")
        
        # Position and P&L - show percentage if we have it
        pnl_sign = "+" if data['pnl_today'] > 0 else ""
        pnl_str = f"${pnl_sign}{data['pnl_today']:.2f}"
        
        # Add percentage if available
        if data.get('pnl_percent', 0) != 0:
            pnl_percent_sign = "+" if data['pnl_percent'] > 0 else ""
            pnl_str += f" ({pnl_percent_sign}{data['pnl_percent']:.2f}%)"
        
        lines.append(f"Position: {data['position']} | P&L Today: {pnl_str}")
        
        # Last signal - show with time, confidence, and approval status
        signal_line = f"Last Signal: {data['last_signal']}"
        if data.get('last_signal_time'):
            signal_line = f"Last Signal: {data['last_signal_time']} {data['last_signal']}"
        if data.get('last_signal_confidence'):
            signal_line += f" (Confidence: {data['last_signal_confidence']})"
        
        # Add approval status indicator
        if data.get('last_signal_approved') is True:
            signal_line += " ✅ APPROVED"
        elif data.get('last_signal_approved') is False:
            signal_line += " ❌ REJECTED"
        
        lines.append(signal_line)
        
        # Show last rejected signal if there is one
        if data.get('last_rejected_signal'):
            lines.append(f"Last Rejected: {data['last_rejected_signal']}")
        
        # Show skip/wait reason if bot is not actively trading
        if data.get('skip_reason') and data.get('position') == "FLAT":
            lines.append(f"Reason: {data['skip_reason']}")
        
        # Status
        lines.append(f"Status: {data['status']}")
        
        lines.append("=" * 60)
        
        return lines
    
    def render(self) -> str:
        """
        Render the complete dashboard.
        
        Returns:
            Complete dashboard as a string
        """
        lines = []
        
        # Header
        lines.extend(self._render_header())
        
        # Settings
        lines.extend(self._render_settings())
        
        # FOMC/Economic event notification (if active) - shows before contract adjustments
        lines.extend(self._render_fomc_notification())
        
        # Contract adjustment notification (if active)
        lines.extend(self._render_contract_adjustment())
        
        # Symbols (only show selected symbols)
        for symbol in self.symbols:
            lines.extend(self._render_symbol(symbol))
        
        # Critical errors (if any)
        if self.bot_data.get("critical_errors"):
            lines.append("")
            lines.append("🚨 CRITICAL ERRORS:")
            for error in self.bot_data["critical_errors"]:
                lines.append(f"  {error}")
            lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def display(self):
        """Display the dashboard (move cursor to home and print)."""
        self._move_cursor_home()
        content = self.render()
        sys.stdout.write(content)
        sys.stdout.flush()
    
    def start(self):
        """Start the dashboard display."""
        self.running = True
        self._init_display()
        self.display()
    
    def stop(self):
        """Stop the dashboard and cleanup."""
        self.running = False
        # Show cursor again
        if not self.is_windows:
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()
    
    def calculate_maintenance_time(self) -> str:
        """
        Calculate time until next maintenance window.
        
        Returns:
            Formatted string like "4h 23m"
        """
        now = datetime.now()
        current_time = now.time()
        
        # Determine next maintenance based on current time
        # Maintenance is 5:00 PM - 6:00 PM ET daily
        maintenance_hour = 17  # 5 PM
        
        # If before 5 PM today, next maintenance is today at 5 PM
        if current_time.hour < maintenance_hour:
            next_maintenance = now.replace(hour=maintenance_hour, minute=0, second=0, microsecond=0)
        # If between 5 PM and 6 PM, we're in maintenance (show as "NOW")
        elif current_time.hour == maintenance_hour:
            return "NOW"
        # If after 6 PM, next maintenance is tomorrow at 5 PM
        else:
            tomorrow = now + timedelta(days=1)
            next_maintenance = tomorrow.replace(hour=maintenance_hour, minute=0, second=0, microsecond=0)
        
        # Calculate time difference
        time_diff = next_maintenance - now
        hours = int(time_diff.total_seconds() // 3600)
        minutes = int((time_diff.total_seconds() % 3600) // 60)
        
        return f"{hours}h {minutes}m"


# Singleton instance
_dashboard_instance: Optional[Dashboard] = None


def get_dashboard(symbols: List[str] = None, config: Dict[str, Any] = None) -> Dashboard:
    """
    Get or create the global dashboard instance.
    
    Args:
        symbols: List of symbols (only used on first call)
        config: Configuration dict (only used on first call)
    
    Returns:
        Dashboard instance
    """
    global _dashboard_instance
    
    if _dashboard_instance is None:
        if symbols is None or config is None:
            raise ValueError("symbols and config required for first dashboard initialization")
        _dashboard_instance = Dashboard(symbols, config)
    
    return _dashboard_instance
