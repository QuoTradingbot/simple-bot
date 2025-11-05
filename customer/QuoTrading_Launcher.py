"""
QuoTrading AI - Customer Launcher
==================================
Professional GUI application for easy setup and launch.
Customers enter credentials, configure settings, and start trading.

Flow:
1. EXE opens → Clean GUI for credentials & settings
2. Click START → Saves config, closes GUI
3. PowerShell terminal opens → Shows live bot logs
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
import json
from pathlib import Path
from datetime import datetime
import sys
import subprocess


class QuoTradingLauncher:
    """Professional GUI launcher for QuoTrading AI bot."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("QuoTrading AI - Professional Trading Platform")
        self.root.geometry("650x580")  # Rectangle: wider than tall, full content visible
        self.root.resizable(False, False)
        
        # Dark blue color scheme
        self.colors = {
            'primary': '#1E3A8A',      # Deep blue
            'secondary': '#3B82F6',    # Bright blue
            'success': '#10B981',      # Green
            'warning': '#F59E0B',      # Orange/yellow warning
            'background': '#1E293B',   # Dark blue-gray background
            'card': '#334155',         # Dark blue card
            'text': '#F1F5F9',         # Light text
            'text_light': '#94A3B8',   # Medium light gray
            'text_secondary': '#64748B', # Secondary text color
            'border': '#475569'        # Dark border
        }
        
        self.root.configure(bg=self.colors['background'])
        
        # Load saved config
        self.config_file = Path("config.json")
        self.config = self.load_config()
        
        # Start with credentials screen
        self.setup_credentials_screen()
    
    def setup_credentials_screen(self):
        """Setup the credentials entry screen (Screen 1)."""
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        self.root.title("QuoTrading AI - Broker Setup")
        
        # Modern gradient-style header - compact
        header = tk.Frame(self.root, bg=self.colors['primary'], height=90)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        # Logo/Title section - centered
        title = tk.Label(
            header,
            text="QuoTrading AI",
            font=("Segoe UI", 18, "bold"),
            bg=self.colors['primary'],
            fg="white"
        )
        title.pack(pady=(12, 0))
        
        subtitle = tk.Label(
            header,
            text="Professional AI Trading",
            font=("Segoe UI", 10),
            bg=self.colors['primary'],
            fg="#93C5FD"
        )
        subtitle.pack(pady=(4, 16))
        
        # Main content area with minimal padding
        main_container = tk.Frame(self.root, bg=self.colors['background'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        # Main card - minimal padding
        main = tk.Frame(main_container, bg=self.colors['card'], relief=tk.FLAT, bd=0)
        main.pack(fill=tk.BOTH, expand=True)
        main.configure(highlightbackground=self.colors['border'], highlightthickness=1)
        
        # Card content with minimal padding
        card_content = tk.Frame(main, bg=self.colors['card'])
        card_content.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Section: QuoTrading License - very compact
        tk.Label(
            card_content,
            text="QuoTrading License",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 2))
        
        tk.Label(
            card_content,
            text="Enter your license key (received via email)",
            font=("Segoe UI", 7),
            bg=self.colors['card'],
            fg=self.colors['text_light']
        ).pack(anchor=tk.W, pady=(0, 4))
        
        # Modern entry field - very compact
        license_frame = tk.Frame(card_content, bg=self.colors['card'])
        license_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.license_entry = tk.Entry(
            license_frame,
            font=("Segoe UI", 8),
            relief=tk.SOLID,
            bd=1,
            highlightthickness=2,
            highlightbackground=self.colors['border'],
            highlightcolor=self.colors['secondary']
        )
        self.license_entry.pack(fill=tk.X, ipady=3)
        self.license_entry.insert(0, self.config.get("quotrading_license", ""))
        
        # Divider - very thin
        tk.Frame(card_content, height=1, bg=self.colors['border']).pack(fill=tk.X, pady=8)
        
        # Section: Market & Broker Selection - category-based
        tk.Label(
            card_content,
            text="Select Market Type",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 6))
        
        # Market type selection (Futures, Forex, Crypto) - centered
        self.market_var = tk.StringVar(value=self.config.get("market_type", "Futures"))
        market_container = tk.Frame(card_content, bg=self.colors['card'])
        market_container.pack(pady=(0, 8))
        
        # Create centered frame for market options
        market_options_frame = tk.Frame(market_container, bg=self.colors['card'])
        market_options_frame.pack()
        
        # Market type radio buttons - centered with Options added
        markets = ["Futures", "Forex", "Crypto", "Options"]
        for market in markets:
            market_radio = tk.Radiobutton(
                market_options_frame,
                text=market,
                variable=self.market_var,
                value=market,
                bg=self.colors['card'],
                fg=self.colors['text'],
                activebackground=self.colors['card'],
                activeforeground=self.colors['text'],
                selectcolor=self.colors['secondary'],
                font=("Segoe UI", 8, "bold"),
                command=self.update_broker_dropdown
            )
            market_radio.pack(side=tk.LEFT, padx=10)
        
        # Broker dropdown (changes based on market type)
        tk.Label(
            card_content,
            text="Select Broker:",
            font=("Segoe UI", 8, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(8, 3))
        
        self.broker_var = tk.StringVar(value=self.config.get("broker", "TopStep"))
        self.broker_dropdown = ttk.Combobox(
            card_content,
            textvariable=self.broker_var,
            state="readonly",
            font=("Segoe UI", 8),
            width=35
        )
        self.broker_dropdown.pack(fill=tk.X, pady=(0, 10))
        self.broker_dropdown.bind('<<ComboboxSelected>>', lambda e: self.update_broker_fields())
        
        # Divider - very thin
        tk.Frame(card_content, height=1, bg=self.colors['border']).pack(fill=tk.X, pady=8)
        
        # Section: Broker Credentials - very compact
        tk.Label(
            card_content,
            text="Broker Credentials",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 6))
        
        # API Token field - very compact
        self.token_label = tk.Label(
            card_content,
            text="TopStep API Token:",
            font=("Segoe UI", 8, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        self.token_label.pack(anchor=tk.W, pady=(0, 2))
        
        token_frame = tk.Frame(card_content, bg=self.colors['card'])
        token_frame.pack(fill=tk.X, pady=(0, 6))
        
        self.token_entry = tk.Entry(
            token_frame,
            font=("Segoe UI", 8),
            show="●",
            relief=tk.SOLID,
            bd=1,
            highlightthickness=2,
            highlightbackground=self.colors['border'],
            highlightcolor=self.colors['secondary']
        )
        self.token_entry.pack(fill=tk.X, ipady=3)
        self.token_entry.insert(0, self.config.get("broker_token", ""))
        
        # Username field - very compact
        self.username_label = tk.Label(
            card_content,
            text="TopStep Username/Email:",
            font=("Segoe UI", 8, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        self.username_label.pack(anchor=tk.W, pady=(0, 2))
        
        username_frame = tk.Frame(card_content, bg=self.colors['card'])
        username_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.username_entry = tk.Entry(
            username_frame,
            font=("Segoe UI", 8),
            relief=tk.SOLID,
            bd=1,
            highlightthickness=2,
            highlightbackground=self.colors['border'],
            highlightcolor=self.colors['secondary']
        )
        self.username_entry.pack(fill=tk.X, ipady=3)
        self.username_entry.insert(0, self.config.get("broker_username", ""))
        
        # Initialize broker dropdown options and update credential fields
        self.update_broker_dropdown()
        
        # Update labels based on saved broker
        self.update_broker_fields()
        
        # Bottom button area - very compact
        button_container = tk.Frame(main_container, bg=self.colors['background'])
        button_container.pack(fill=tk.X, pady=(10, 0))
        
        # Modern confirm button - compact
        confirm_btn = tk.Button(
            button_container,
            text="CONFIRM",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['secondary'],
            fg="white",
            activebackground=self.colors['primary'],
            activeforeground="white",
            relief=tk.FLAT,
            command=self.validate_and_continue,
            cursor="hand2",
            bd=0
        )
        confirm_btn.pack(fill=tk.X, ipady=8)
    
    def update_broker_fields(self):
        """Update field labels based on selected broker."""
        broker_full = self.broker_var.get()
        # Extract broker name (before the dash)
        broker = broker_full.split(" - ")[0] if " - " in broker_full else broker_full
        
        # Define credential requirements for each broker
        broker_credentials = {
            # Futures brokers
            "TopStep": {
                "field1_label": "TopStep API Token:",
                "field2_label": "TopStep Username/Email:",
                "field1_show": "●",
                "field2_show": ""
            },
            "Tradovate": {
                "field1_label": "Tradovate API Key:",
                "field2_label": "Tradovate Username:",
                "field1_show": "●",
                "field2_show": ""
            },
            "Rithmic": {
                "field1_label": "Rithmic Username:",
                "field2_label": "Rithmic Password:",
                "field1_show": "",
                "field2_show": "●"
            },
            "Interactive Brokers": {
                "field1_label": "IBKR Account ID:",
                "field2_label": "IBKR Username:",
                "field1_show": "",
                "field2_show": ""
            },
            "NinjaTrader": {
                "field1_label": "NinjaTrader License Key:",
                "field2_label": "NinjaTrader Username:",
                "field1_show": "●",
                "field2_show": ""
            },
            # Forex brokers
            "OANDA": {
                "field1_label": "OANDA API Token:",
                "field2_label": "OANDA Account ID:",
                "field1_show": "●",
                "field2_show": ""
            },
            "FXCM": {
                "field1_label": "FXCM API Token:",
                "field2_label": "FXCM Account ID:",
                "field1_show": "●",
                "field2_show": ""
            },
            "TD Ameritrade": {
                "field1_label": "TD Ameritrade API Key:",
                "field2_label": "TD Ameritrade Username:",
                "field1_show": "●",
                "field2_show": ""
            },
            "IG Markets": {
                "field1_label": "IG API Key:",
                "field2_label": "IG Username:",
                "field1_show": "●",
                "field2_show": ""
            },
            # Crypto brokers
            "Binance": {
                "field1_label": "Binance API Key:",
                "field2_label": "Binance API Secret:",
                "field1_show": "●",
                "field2_show": "●"
            },
            "Coinbase Pro": {
                "field1_label": "Coinbase API Key:",
                "field2_label": "Coinbase API Secret:",
                "field1_show": "●",
                "field2_show": "●"
            },
            "Kraken": {
                "field1_label": "Kraken API Key:",
                "field2_label": "Kraken Private Key:",
                "field1_show": "●",
                "field2_show": "●"
            },
            "Bybit": {
                "field1_label": "Bybit API Key:",
                "field2_label": "Bybit API Secret:",
                "field1_show": "●",
                "field2_show": "●"
            },
            "Bitget": {
                "field1_label": "Bitget API Key:",
                "field2_label": "Bitget Secret Key:",
                "field1_show": "●",
                "field2_show": "●"
            },
            # Options brokers
            "Tastytrade": {
                "field1_label": "Tastytrade API Token:",
                "field2_label": "Tastytrade Account Number:",
                "field1_show": "●",
                "field2_show": ""
            },
            "E*TRADE": {
                "field1_label": "E*TRADE Consumer Key:",
                "field2_label": "E*TRADE Consumer Secret:",
                "field1_show": "●",
                "field2_show": "●"
            },
            "Charles Schwab": {
                "field1_label": "Schwab API Key:",
                "field2_label": "Schwab Account Number:",
                "field1_show": "●",
                "field2_show": ""
            }
        }
        
        # Get credentials for selected broker
        creds = broker_credentials.get(broker, broker_credentials["TopStep"])
        
        # Update labels and show/hide settings
        self.token_label.config(text=creds["field1_label"])
        self.username_label.config(text=creds["field2_label"])
        self.token_entry.config(show=creds["field1_show"])
        self.username_entry.config(show=creds["field2_show"])
    
    def update_broker_dropdown(self):
        """Update broker dropdown options based on selected market type."""
        market = self.market_var.get()
        
        # Define brokers for each market type
        broker_options = {
            "Futures": [
                "TopStep - Funded trader program",
                "Tradovate - Cloud-based platform",
                "Rithmic - Professional API access",
                "Interactive Brokers - Global markets",
                "NinjaTrader - Platform + brokerage"
            ],
            "Forex": [
                "OANDA - Forex specialist",
                "Interactive Brokers - Multi-asset",
                "FXCM - Forex trading",
                "TD Ameritrade - Forex & more",
                "IG Markets - Global forex"
            ],
            "Crypto": [
                "Binance - Largest crypto exchange",
                "Coinbase Pro - US-based exchange",
                "Kraken - Secure crypto trading",
                "Bybit - Derivatives exchange",
                "Bitget - Copy trading platform"
            ],
            "Options": [
                "Tastytrade - Options specialist",
                "TD Ameritrade - thinkorswim platform",
                "Interactive Brokers - Advanced options",
                "E*TRADE - Options trading tools",
                "Charles Schwab - Full-service options"
            ]
        }
        
        # Update dropdown values
        self.broker_dropdown['values'] = broker_options.get(market, broker_options["Futures"])
        
        # Set default selection if current broker not in new list
        current_broker = self.broker_var.get()
        if not any(current_broker in option for option in self.broker_dropdown['values']):
            self.broker_dropdown.current(0)
        
        # Update credential fields
        self.update_broker_fields()
    
    def validate_and_continue(self):
        """Validate license key and move to settings screen. Broker credentials validated later."""
        broker_full = self.broker_var.get()
        broker = broker_full.split(" - ")[0] if " - " in broker_full else broker_full
        market_type = self.market_var.get()
        
        # Step 1: Validate license key (REQUIRED)
        license_key = self.license_entry.get()
        
        if not license_key:
            messagebox.showerror("Missing License Key", 
                "QuoTrading License Key is required!\n\n"
                "You should have received this via email after purchase.\n"
                "Contact support@quotrading.com if you need help.")
            return
        
        # Admin master key - instant access (bypass all checks)
        if license_key == "QUOTRADING_ADMIN_MASTER_2025":
            self.config["quotrading_license"] = license_key
            self.config["market_type"] = market_type
            self.config["broker"] = broker
            self.config["broker_token"] = self.token_entry.get() or "admin_token"
            self.config["broker_username"] = self.username_entry.get() or "admin@quotrading.com"
            self.save_config()
            self.setup_settings_screen()
            return
        
        # Validate license key format (basic check for now)
        if len(license_key) < 20:
            messagebox.showerror("Invalid License Key", 
                "License key appears to be invalid.\n\n"
                "Please check your email for the correct license key.\n"
                "Format: QUOTRADING-XXXX-XXXX-XXXX-XXXX")
            return
        
        # Step 2: Check broker credentials are entered (NOT validated yet - that happens at START TRADING)
        if not self.token_entry.get():
            messagebox.showwarning(f"Missing {broker} Credentials", 
                f"Please enter your {broker} API credentials.\n\n"
                f"You can get these from your {broker} account dashboard.\n"
                f"These will be validated when you start trading.")
            return
        
        if not self.username_entry.get():
            messagebox.showwarning(f"Missing {broker} Username", 
                f"Please enter your {broker} username/email.")
            return
        
        # Step 3: Save everything and proceed to settings screen
        self.config["quotrading_license"] = license_key
        self.config["market_type"] = market_type
        self.config["broker"] = broker
        self.config["broker_token"] = self.token_entry.get()
        self.config["broker_username"] = self.username_entry.get()
        self.save_config()
        
        # Move to settings screen (broker credentials will be validated when clicking START TRADING)
        self.setup_settings_screen()
    
    def setup_settings_screen(self):
        """Setup the trading settings screen (Screen 2)."""
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        self.root.title("QuoTrading AI - Configure Trading")
        
        # Compact header
        header = tk.Frame(self.root, bg=self.colors['primary'], height=50)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        title = tk.Label(
            header, 
            text="QuoTrading AI - Trading Settings", 
            font=("Segoe UI", 14, "bold"),
            bg=self.colors['primary'],
            fg="white"
        )
        title.pack(pady=12)
        
        # Main container with dark background - minimal padding
        main_container = tk.Frame(self.root, bg=self.colors['background'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Main card with rounded appearance
        main = tk.Frame(main_container, bg=self.colors['card'], relief=tk.FLAT, bd=0)
        main.pack(fill=tk.BOTH, expand=True)
        main.configure(highlightbackground=self.colors['border'], highlightthickness=1)
        
        # Card content - minimal padding
        card_content = tk.Frame(main, bg=self.colors['card'])
        card_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=3)
        
        # Trading Settings Frame (don't expand, just fill)
        settings = tk.Frame(card_content, bg=self.colors['card'])
        settings.pack(fill=tk.X, expand=False, pady=(0, 5))
        
        # Section 1: Trading Symbols - Checkbox Grid
        tk.Label(
            settings, 
            text="Trading Symbols", 
            font=("Segoe UI", 11, "bold"), 
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 5))
        
        # Checkbox grid for symbols
        symbol_grid = tk.Frame(settings, bg=self.colors['card'])
        symbol_grid.pack(fill=tk.X, pady=(0, 3))
        
        # Available symbols - All major futures contracts
        self.all_symbols = [
            ("ES", "E-mini S&P 500"),
            ("NQ", "E-mini Nasdaq 100"),
            ("YM", "E-mini Dow"),
            ("RTY", "E-mini Russell 2000"),
            ("CL", "Crude Oil"),
            ("GC", "Gold"),
            ("NG", "Natural Gas"),
            ("6E", "Euro FX"),
            ("ZN", "10-Year Treasury Note"),
            ("MES", "Micro E-mini S&P 500"),
            ("MNQ", "Micro E-mini Nasdaq 100")
        ]
        
        # Store checkbox variables
        self.symbol_vars = {}
        saved_symbols = self.config.get("symbols", ["ES"])
        if isinstance(saved_symbols, str):
            saved_symbols = [saved_symbols]
        
        # Create 2-column grid of checkboxes
        for i, (code, name) in enumerate(self.all_symbols):
            row = i // 2
            col = i % 2
            
            var = tk.BooleanVar(value=(code in saved_symbols))
            self.symbol_vars[code] = var
            
            cb_frame = tk.Frame(symbol_grid, bg=self.colors['card'])
            cb_frame.grid(row=row, column=col, sticky=tk.W, padx=(0, 20), pady=1)
            
            cb = tk.Checkbutton(
                cb_frame,
                text=f"{code} - {name}",
                variable=var,
                font=("Segoe UI", 9),
                bg=self.colors['card'],
                fg=self.colors['text'],
                selectcolor=self.colors['background'],
                activebackground=self.colors['card'],
                activeforeground=self.colors['secondary'],
                cursor="hand2"
            )
            cb.pack(anchor=tk.W)
        
        # Section 2: Account Size & Risk - Side by side
        account_risk_frame = tk.Frame(settings, bg=self.colors['card'])
        account_risk_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Account Size (left side)
        account_col = tk.Frame(account_risk_frame, bg=self.colors['card'])
        account_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        tk.Label(
            account_col, 
            text="Account Size ($)", 
            font=("Segoe UI", 9),
            bg=self.colors['card'],
            fg=self.colors['text_light']
        ).pack(anchor=tk.W)
        
        account_entry_frame = tk.Frame(account_col, bg=self.colors['card'])
        account_entry_frame.pack(fill=tk.X, pady=(3, 0))
        
        self.account_entry = tk.Entry(
            account_entry_frame,
            font=("Segoe UI", 9),
            relief=tk.SOLID,
            bd=1,
            highlightthickness=2,
            highlightbackground=self.colors['border'],
            highlightcolor=self.colors['secondary']
        )
        self.account_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=2)
        self.account_entry.insert(0, self.config.get("account_size", "10000"))
        
        fetch_btn = tk.Button(
            account_entry_frame,
            text="Fetch from Broker",
            font=("Segoe UI", 7),
            bg=self.colors['secondary'],
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            padx=8,
            pady=4,
            command=self.fetch_account_size
        )
        fetch_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        # Risk per Trade (right side)
        risk_col = tk.Frame(account_risk_frame, bg=self.colors['card'])
        risk_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(
            risk_col, 
            text="Risk per Trade (%)", 
            font=("Segoe UI", 9),
            bg=self.colors['card'],
            fg=self.colors['text_light']
        ).pack(anchor=tk.W)
        
        self.risk_var = tk.DoubleVar(value=self.config.get("risk_per_trade", 1.2))
        risk_spin = ttk.Spinbox(
            risk_col,
            from_=0.5, 
            to=5.0, 
            increment=0.1, 
            textvariable=self.risk_var,
            width=15,
            format="%.1f"
        )
        risk_spin.pack(fill=tk.X, pady=(3, 0), ipady=0)
        
        # Daily Loss Limit
        daily_loss_col = tk.Frame(account_risk_frame, bg=self.colors['card'])
        daily_loss_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        
        tk.Label(
            daily_loss_col, 
            text="Daily Loss Limit ($)", 
            font=("Segoe UI", 9),
            bg=self.colors['card'],
            fg=self.colors['text_light']
        ).pack(anchor=tk.W)
        
        self.daily_loss_var = tk.IntVar(value=self.config.get("daily_loss_limit", 2000))
        loss_spin = ttk.Spinbox(
            daily_loss_col,
            from_=500, 
            to=10000, 
            increment=100, 
            textvariable=self.daily_loss_var,
            width=15
        )
        loss_spin.pack(fill=tk.X, pady=(3, 0), ipady=0)
        
        # Section 3: Strategy Settings - 3 columns - reduce padding
        strategy_frame = tk.Frame(settings, bg=self.colors['card'])
        strategy_frame.pack(fill=tk.X, pady=(0, 3))
        
        # Max Contracts
        contracts_col = tk.Frame(strategy_frame, bg=self.colors['card'])
        contracts_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        tk.Label(
            contracts_col, 
            text="Max Contracts", 
            font=("Segoe UI", 9),
            bg=self.colors['card'],
            fg=self.colors['text_light']
        ).pack(anchor=tk.W)
        
        self.contracts_var = tk.IntVar(value=self.config.get("max_contracts", 3))
        contracts_spin = ttk.Spinbox(
            contracts_col,
            from_=1, 
            to=25, 
            textvariable=self.contracts_var,
            width=12
        )
        contracts_spin.pack(fill=tk.X, pady=(3, 0), ipady=0)
        
        # Min Risk/Reward
        rr_col = tk.Frame(strategy_frame, bg=self.colors['card'])
        rr_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        tk.Label(
            rr_col, 
            text="Min Risk/Reward", 
            font=("Segoe UI", 9),
            bg=self.colors['card'],
            fg=self.colors['text_light']
        ).pack(anchor=tk.W)
        
        self.risk_reward_var = tk.DoubleVar(value=self.config.get("min_risk_reward", 2.0))
        rr_spin = ttk.Spinbox(
            rr_col,
            from_=1.0, 
            to=5.0, 
            increment=0.1, 
            textvariable=self.risk_reward_var,
            width=12,
            format="%.1f"
        )
        rr_spin.pack(fill=tk.X, pady=(3, 0), ipady=0)
        
        # Max Trades/Day
        trades_col = tk.Frame(strategy_frame, bg=self.colors['card'])
        trades_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(
            trades_col, 
            text="Max Trades/Day", 
            font=("Segoe UI", 9),
            bg=self.colors['card'],
            fg=self.colors['text_light']
        ).pack(anchor=tk.W)
        
        self.trades_var = tk.IntVar(value=self.config.get("max_trades", 10))
        trades_spin = ttk.Spinbox(
            trades_col,
            from_=1, 
            to=50, 
            textvariable=self.trades_var,
            width=12
        )
        trades_spin.pack(fill=tk.X, pady=(3, 0), ipady=0)
        
        # Section 5: Auto-Calculate Toggle & Shadow Mode (compact row)
        toggles_frame = tk.Frame(settings, bg=self.colors['card'])
        toggles_frame.pack(fill=tk.X, pady=(3, 0))
        
        # Left column: Auto-calculate
        self.auto_calculate_var = tk.BooleanVar(value=self.config.get("auto_calculate_limits", True))
        auto_check = tk.Checkbutton(
            toggles_frame,
            text="Auto-calculate Limits",
            variable=self.auto_calculate_var,
            font=("Segoe UI", 9),
            bg=self.colors['card'],
            fg=self.colors['text'],
            selectcolor=self.colors['background'],
            activebackground=self.colors['card'],
            activeforeground=self.colors['secondary'],
            cursor="hand2"
        )
        auto_check.pack(side=tk.LEFT, anchor=tk.W)
        
        # Right column: Shadow mode (compact)
        self.shadow_mode_var = tk.BooleanVar(value=self.config.get("shadow_mode", False))
        shadow_check = tk.Checkbutton(
            toggles_frame,
            text="🌙 Shadow Mode",
            variable=self.shadow_mode_var,
            font=("Segoe UI", 9),
            bg=self.colors['card'],
            fg=self.colors['warning'],
            selectcolor=self.colors['background'],
            activebackground=self.colors['card'],
            activeforeground=self.colors['warning'],
            cursor="hand2"
        )
        shadow_check.pack(side=tk.LEFT, anchor=tk.W, padx=(20, 0))
        
        # Shadow mode explanation (small text below)
        shadow_info = tk.Label(
            settings,
            text="Shadow mode = Simulates full trading with live data (no account login, tracks positions/P&L locally)",
            font=("Segoe UI", 7, "italic"),
            bg=self.colors['card'],
            fg=self.colors['text_secondary'],
            justify=tk.LEFT
        )
        shadow_info.pack(anchor=tk.W, pady=(2, 0))
        
        # Separator line - reduce padding
        separator = tk.Frame(card_content, bg=self.colors['border'], height=1)
        separator.pack(fill=tk.X, pady=(3, 5))
        
        # Button Container - Side by side
        button_container = tk.Frame(card_content, bg=self.colors['card'])
        button_container.pack(fill=tk.X, pady=(0, 5))
        
        # Start Bot Button (left)
        self.start_btn = tk.Button(
            button_container,
            text="▶ Start Bot",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['success'],
            fg="white",
            activebackground="#059669",
            activeforeground="white",
            command=self.start_bot,
            cursor="hand2",
            relief=tk.FLAT,
            bd=0
        )
        self.start_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(0, 5))
        
        # Stop Bot Button (right)
        self.stop_btn = tk.Button(
            button_container,
            text="■ Stop Bot",
            font=("Segoe UI", 11, "bold"),
            bg="#EF4444",
            fg="white",
            activebackground="#DC2626",
            activeforeground="white",
            command=self.stop_bot,
            cursor="hand2",
            relief=tk.FLAT,
            bd=0,
            state=tk.DISABLED  # Disabled until bot starts
        )
        self.stop_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(5, 0))
        
        # Store bot process reference
        self.bot_process = None
        
        # Console/Status Frame (compact)
        console_separator = tk.Frame(card_content, bg=self.colors['border'], height=1)
        console_separator.pack(fill=tk.X, pady=(5, 3))
        
        console_label = tk.Label(
            card_content,
            text="📊 Status",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        console_label.pack(anchor=tk.W, pady=(0, 3))
        
        # Compact console text area (3 lines)
        console_frame = tk.Frame(card_content, bg=self.colors['background'])
        console_frame.pack(fill=tk.X)
        
        self.console_text = tk.Text(
            console_frame,
            height=3,
            font=("Consolas", 7),
            bg="#1a1a1a",
            fg="#00ff00",
            insertbackground="#00ff00",
            relief=tk.FLAT,
            padx=5,
            pady=3,
            wrap=tk.WORD
        )
        self.console_text.pack(fill=tk.X)
        
        # Initial console message
        self.console_log("Ready - Configure settings and click 'Start Bot'")
        self.console_text.config(state=tk.DISABLED)
    
    def fetch_account_size(self):
        """Fetch account size from broker credentials."""
        # Get broker credentials
        broker = self.config.get("broker", "TopStep")
        broker_token = self.config.get("broker_token", "")
        broker_username = self.config.get("broker_username", "")
        
        if not broker_token or not broker_username:
            messagebox.showwarning(
                "Credentials Required",
                f"Please enter your {broker} credentials first!\n\n"
                f"API Token and Username are required to fetch account balance."
            )
            return
        
        # Show working dialog
        result = messagebox.showinfo(
            "Fetching Balance...",
            f"Connecting to {broker} to fetch account balance...\n\n"
            f"This feature requires the bot to be running.\n"
            f"For now, please enter your account size manually.\n\n"
            f"Future update: Real-time balance fetching!"
        )
        
        # TODO: Implement actual broker API call when bot is running
        # For now, suggest common account sizes
        suggested = messagebox.askyesno(
            "Suggest Account Size?",
            "Would you like to use a common account size?\n\n"
            "TopStep Account Sizes:\n"
            "• Express: $25,000\n"
            "• Step 1: $50,000\n"
            "• Step 2: $100,000 or $150,000\n\n"
            "Click YES to select, NO to enter manually."
        )
        
        if suggested:
            # Show selection dialog
            from tkinter import simpledialog
            choice = simpledialog.askstring(
                "Select Account Size",
                "Enter account size:\n"
                "25000 (Express)\n"
                "50000 (Step 1)\n"
                "100000 (Step 2)\n"
                "150000 (Step 2 Large)\n"
            )
            if choice and choice.isdigit():
                self.account_entry.delete(0, tk.END)
                self.account_entry.insert(0, choice)
    
    def load_config(self):
        """Load saved configuration."""
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def save_config(self):
        """Save current configuration (only saves what exists)."""
        config = self.config.copy()  # Start with existing config
        
        # Try to save credentials if they still exist (Screen 1)
        try:
            if hasattr(self, 'license_entry'):
                config["quotrading_license"] = self.license_entry.get()
        except:
            pass
        
        try:
            if hasattr(self, 'token_entry'):
                config["broker_token"] = self.token_entry.get()
        except:
            pass
        
        try:
            if hasattr(self, 'username_entry'):
                config["broker_username"] = self.username_entry.get()
        except:
            pass
        
        # Save trading settings if they exist (Screen 2)
        try:
            if hasattr(self, 'symbol_vars'):
                # Get selected symbols from checkboxes
                selected_symbols = [code for code, var in self.symbol_vars.items() if var.get()]
                config["symbols"] = selected_symbols if selected_symbols else ["ES"]
        except:
            pass
        
        try:
            if hasattr(self, 'account_entry'):
                config["account_size"] = self.account_entry.get()
        except:
            pass
        
        try:
            if hasattr(self, 'contracts_var'):
                config["max_contracts"] = self.contracts_var.get()
        except:
            pass
        
        try:
            if hasattr(self, 'trades_var'):
                config["max_trades"] = self.trades_var.get()
        except:
            pass
        
        try:
            if hasattr(self, 'risk_var'):
                config["risk_per_trade"] = self.risk_var.get()
        except:
            pass
        
        try:
            if hasattr(self, 'risk_reward_var'):
                config["min_risk_reward"] = self.risk_reward_var.get()
        except:
            pass
        
        try:
            if hasattr(self, 'daily_loss_var'):
                config["daily_loss_limit"] = self.daily_loss_var.get()
        except:
            pass
        
        try:
            if hasattr(self, 'auto_calculate_var'):
                config["auto_calculate_limits"] = self.auto_calculate_var.get()
        except:
            pass
        
        try:
            if hasattr(self, 'shadow_mode_var'):
                config["shadow_mode"] = self.shadow_mode_var.get()
        except:
            pass
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def console_log(self, message):
        """Add message to console with timestamp."""
        if not hasattr(self, 'console_text'):
            return  # Console not initialized yet
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console_text.config(state=tk.NORMAL)
        self.console_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.console_text.see(tk.END)  # Auto-scroll to bottom
        self.console_text.config(state=tk.DISABLED)
    
    def stop_bot(self):
        """Stop the running bot process."""
        if self.bot_process is None or self.bot_process.poll() is not None:
            messagebox.showinfo(
                "No Bot Running",
                "No bot process is currently running."
            )
            return
        
        result = messagebox.askyesno(
            "Stop Bot?",
            "⚠ Stop Trading Bot ⚠\n\n"
            "This will stop the bot process.\n"
            "Any open positions will remain open!\n\n"
            "Are you sure?"
        )
        
        if result:
            try:
                self.console_log("Stopping bot process...")
                self.bot_process.terminate()
                self.bot_process.wait(timeout=5)
                self.bot_process = None
                
                # Update UI
                self.start_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)
                
                self.console_log("✓ Bot stopped successfully")
                messagebox.showinfo(
                    "Bot Stopped",
                    "✓ Bot process terminated.\n\n"
                    "IMPORTANT: Check broker for any open positions!"
                )
            except Exception as e:
                self.console_log(f"✗ Failed to stop bot: {e}")
                messagebox.showerror(
                    "Stop Failed",
                    f"Failed to stop bot:\n{str(e)}"
                )
    
    def kill_bot(self):
        """Legacy method - redirects to stop_bot."""
        self.stop_bot()
    
    def start_bot(self):
        """Validate broker credentials and start the trading bot."""
        # Step 0: Validate at least one symbol is selected
        selected_symbols = [code for code, var in self.symbol_vars.items() if var.get()]
        
        if not selected_symbols:
            self.console_log("✗ Error: No symbols selected")
            messagebox.showerror(
                "No Symbols Selected",
                "⚠ Please select at least one symbol to trade!\n\n"
                "Use the checkboxes to choose ES, NQ, GC, or other instruments."
            )
            return
        
        # Step 1: Validate broker credentials (required for both live and shadow mode)
        broker = self.config.get("broker", "TopStep")
        broker_token = self.config.get("broker_token", "")
        broker_username = self.config.get("broker_username", "")
        license_key = self.config.get("quotrading_license", "")
        shadow_mode = self.shadow_mode_var.get()
        
        # Validate broker credentials (required even in shadow mode for live data streaming)
        # Admin key bypasses broker validation
        if license_key != "QUOTRADING_ADMIN_MASTER_2025":
            # Validate broker credentials are present
            if not broker_token or not broker_username:
                self.console_log("✗ Error: Missing broker credentials")
                mode_desc = "simulated trading with live data (no account)" if shadow_mode else "live trading"
                messagebox.showerror(
                    "Missing Broker Credentials",
                    f"Your {broker} credentials are required for {mode_desc}!\n\n"
                    f"Shadow mode simulates full trading using live market data\n"
                    f"without logging into your trading account.\n\n"
                    f"Please go back and enter your API credentials."
                )
                return
        
        if shadow_mode:
            self.console_log("🌙 Shadow mode enabled - simulating trades with live data (no account)")
        
        # Step 2: Save final config
        self.console_log("Saving configuration...")
        self.save_config()
        
        # Step 3: Create .env file
        self.console_log("Creating .env file...")
        self.create_env_file()
        
        # Step 4: Show confirmation
        symbols_str = ", ".join(selected_symbols)
        mode_str = "🌙 Shadow Mode (Simulated Trading)" if shadow_mode else f"{broker} Live Trading"
        
        result = messagebox.askyesno(
            "Launch Trading Bot?",
            f"Ready to start bot with these settings:\n\n"
            f"Mode: {mode_str}\n"
            f"Broker: {broker}\n"
            f"Symbols: {symbols_str}\n"
            f"Max Contracts: {self.contracts_var.get()}\n"
            f"Max Trades/Day: {self.trades_var.get()}\n"
            f"Risk/Trade: {self.risk_var.get()}%\n"
            f"Min R:R Ratio: {self.risk_reward_var.get()}:1\n"
            f"Daily Loss Limit: ${self.daily_loss_var.get()}\n\n"
            f"{'⚠️ Shadow Mode: Simulates full trading with live data\n(tracks positions/P&L locally without account login).\n\n' if shadow_mode else ''}"
            f"This will open a PowerShell terminal with live logs.\n"
            f"Use the STOP BOT button to stop trading.\n\n"
            f"Continue?"
        )
        
        if not result:
            self.console_log("Launch cancelled by user")
            return
        
        # Launch bot in PowerShell terminal
        try:
            self.console_log(f"Launching bot for {symbols_str}...")
            
            # Get the parent directory (where run.py is located)
            bot_dir = Path(__file__).parent.parent.absolute()
            
            # PowerShell command to run the bot
            ps_command = [
                "powershell.exe",
                "-NoExit",  # Keep window open
                "-Command",
                f"cd '{bot_dir}'; python run.py"
            ]
            
            # Start PowerShell process in a NEW SYSTEM WINDOW
            self.bot_process = subprocess.Popen(
                ps_command, 
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                cwd=str(bot_dir)
            )
            
            # Update UI
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
            # Log success
            self.console_log(f"✓ Bot launched successfully (PID: {self.bot_process.pid})")
            self.console_log(f"Mode: {mode_str}")
            self.console_log(f"Symbols: {symbols_str}")
            self.console_log("PowerShell terminal opened - check for live logs")
            self.console_log("Use 'Stop Bot' button to stop trading")
            
            # Success notification (don't close GUI)
            messagebox.showinfo(
                "Bot Launched!",
                f"✓ Bot launched successfully!\n\n"
                f"Mode: {mode_str}\n"
                f"Symbols: {symbols_str}\n\n"
                f"PowerShell terminal opened with live logs.\n"
                f"This window will stay open for monitoring.\n"
                f"Use the STOP BOT button to stop trading."
            )
            
        except Exception as e:
            self.console_log(f"✗ Launch failed: {e}")
            messagebox.showerror(
                "Launch Error",
                f"Failed to launch bot:\n{str(e)}\n\n"
                f"Make sure Python is installed and run.py exists."
            )
    
    def create_env_file(self):
        """Create .env file from GUI settings."""
        # Get selected symbols from checkboxes (NOT from config)
        selected_symbols = [code for code, var in self.symbol_vars.items() if var.get()]
        if not selected_symbols:
            selected_symbols = ["ES"]  # Fallback
        
        symbols_str = ",".join(selected_symbols)
        
        broker = self.config.get("broker", "TopStep")
        
        # Get the bot directory (parent of customer folder)
        bot_dir = Path(__file__).parent.parent.absolute()
        env_path = bot_dir / '.env'
        
        env_content = f"""# QuoTrading AI - Auto-generated Configuration
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# DO NOT EDIT MANUALLY - Use the launcher to change settings

# License & API Keys
QUOTRADING_LICENSE_KEY={self.config.get("quotrading_license", "")}
BROKER={broker}
BROKER_API_TOKEN={self.config.get("broker_token", "")}
BROKER_USERNAME={self.config.get("broker_username", "")}

# Legacy TopStep/Tradovate variables (for compatibility)
TOPSTEP_API_TOKEN={self.config.get("broker_token", "")}
TOPSTEP_USERNAME={self.config.get("broker_username", "")}
TRADOVATE_API_KEY={self.config.get("broker_token", "")}
TRADOVATE_USERNAME={self.config.get("broker_username", "")}

# Trading Configuration - Multi-Symbol Support
# SELECTED SYMBOLS: {symbols_str}
BOT_INSTRUMENTS={symbols_str}
BOT_MAX_CONTRACTS={self.contracts_var.get()}
BOT_MAX_TRADES_PER_DAY={self.trades_var.get()}
BOT_RISK_PER_TRADE={self.risk_var.get() / 100}
BOT_MIN_RISK_REWARD={self.risk_reward_var.get()}
BOT_DAILY_LOSS_LIMIT={self.daily_loss_var.get()}
BOT_AUTO_CALCULATE_LIMITS={str(self.auto_calculate_var.get()).lower()}

# Trading Mode
BOT_SHADOW_MODE={str(self.shadow_mode_var.get()).lower()}
BOT_DRY_RUN=false

# Environment
BOT_ENVIRONMENT=production
CONFIRM_LIVE_TRADING=1
BOT_LOG_LEVEL=INFO
QUOTRADING_API_URL=https://api.quotrading.com/v1/signals
"""
        
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        # Log which symbols were saved
        print(f"✓ .env file created with {len(selected_symbols)} symbols: {symbols_str}")
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


if __name__ == "__main__":
    app = QuoTradingLauncher()
    app.run()
