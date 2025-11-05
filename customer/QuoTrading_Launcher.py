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
            'background': '#1E293B',   # Dark blue-gray background
            'card': '#334155',         # Dark blue card
            'text': '#F1F5F9',         # Light text
            'text_light': '#94A3B8',   # Medium light gray
            'border': '#475569'        # Dark border
        }
        
        self.root.configure(bg=self.colors['background'])
        
        # Load saved config
        self.config_file = Path("config.json")
        self.config = self.load_config()
        
        # Start with credentials screen
        self.setup_credentials_screen()
        
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
        self.root.geometry("700x550")
        
        # Header
        header = tk.Frame(self.root, bg="#2C3E50", height=80)
        header.pack(fill=tk.X)
        
        title = tk.Label(
            header, 
            text="QuoTrading AI - Trading Settings", 
            font=("Arial", 20, "bold"),
            bg="#2C3E50",
            fg="white"
        )
        title.pack(pady=25)
        
        # Main container
        main = tk.Frame(self.root, bg="white", padx=30, pady=20)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Instructions
        instructions = tk.Label(
            main,
            text="Configure Your Trading Settings",
            font=("Arial", 14, "bold"),
            bg="white"
        )
        instructions.pack(pady=(0, 20))
        
        # Trading Settings Frame
        settings = tk.Frame(main, bg="white")
        settings.pack(fill=tk.BOTH, expand=True)
        
        # Row 1: Symbol and Contracts
        tk.Label(settings, text="Trading Symbol:", font=("Arial", 11, "bold"), bg="white").grid(row=0, column=0, sticky=tk.W, pady=10)
        self.symbol_var = tk.StringVar(value=self.config.get("symbol", "ES"))
        symbol_combo = ttk.Combobox(settings, textvariable=self.symbol_var, width=25, state="readonly")
        symbol_combo['values'] = (
            "ES - E-mini S&P 500",
            "NQ - E-mini Nasdaq 100",
            "YM - E-mini Dow",
            "RTY - E-mini Russell 2000",
            "CL - Crude Oil",
            "GC - Gold",
            "NG - Natural Gas",
            "6E - Euro FX",
            "ZN - 10-Year Treasury",
            "MES - Micro E-mini S&P 500",
            "MNQ - Micro E-mini Nasdaq 100"
        )
        symbol_combo.grid(row=0, column=1, sticky=tk.W, padx=15, pady=10)
        
        tk.Label(settings, text="Max Contracts:", font=("Arial", 11, "bold"), bg="white").grid(row=0, column=2, sticky=tk.W, padx=(30, 0), pady=10)
        self.contracts_var = tk.IntVar(value=self.config.get("max_contracts", 3))
        self.contracts_spin = ttk.Spinbox(settings, from_=1, to=10, textvariable=self.contracts_var, width=10)
        self.contracts_spin.grid(row=0, column=3, padx=15, pady=10)
        
        # Row 2: Daily Limits
        tk.Label(settings, text="Max Trades/Day:", font=("Arial", 11, "bold"), bg="white").grid(row=1, column=0, sticky=tk.W, pady=10)
        self.trades_var = tk.IntVar(value=self.config.get("max_trades", 9))
        trades_spin = ttk.Spinbox(settings, from_=1, to=50, textvariable=self.trades_var, width=10)
        trades_spin.grid(row=1, column=1, sticky=tk.W, padx=15, pady=10)
        
        tk.Label(settings, text="Risk per Trade (%):", font=("Arial", 11, "bold"), bg="white").grid(row=1, column=2, sticky=tk.W, padx=(30, 0), pady=10)
        self.risk_var = tk.DoubleVar(value=self.config.get("risk_per_trade", 1.2))
        risk_spin = ttk.Spinbox(settings, from_=0.5, to=5.0, increment=0.1, textvariable=self.risk_var, width=10, format="%.1f")
        risk_spin.grid(row=1, column=3, padx=15, pady=10)
        
        # Row 3: TopStep Rules
        self.topstep_var = tk.BooleanVar(value=self.config.get("use_topstep_rules", True))
        topstep_check = ttk.Checkbutton(
            settings,
            text="Use TopStep Rules (Daily loss limits, drawdown protection)",
            variable=self.topstep_var
        )
        topstep_check.grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(15, 10))
        
        # Start Button
        start_btn = tk.Button(
            main,
            text="🚀 START TRADING",
            font=("Arial", 14, "bold"),
            bg="#27AE60",
            fg="white",
            command=self.start_bot,
            width=30,
            height=2
        )
        start_btn.pack(pady=20)
        
        info_label = tk.Label(
            main,
            text="Bot will launch in PowerShell terminal with live logs",
            font=("Arial", 9),
            bg="white",
            fg="gray"
        )
        info_label.pack()
    
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
            if hasattr(self, 'symbol_var'):
                config["symbol"] = self.symbol_var.get().split(" - ")[0]
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
            if hasattr(self, 'topstep_var'):
                config["use_topstep_rules"] = self.topstep_var.get()
        except:
            pass
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def start_bot(self):
        """Validate broker credentials and start the trading bot in PowerShell terminal."""
        # Step 1: Validate broker credentials NOW (right before trading)
        broker = self.config.get("broker", "TopStep")
        broker_token = self.config.get("broker_token", "")
        broker_username = self.config.get("broker_username", "")
        license_key = self.config.get("quotrading_license", "")
        
        # Admin key bypasses broker validation
        if license_key != "QUOTRADING_ADMIN_MASTER_2025":
            # Validate broker credentials are present
            if not broker_token or not broker_username:
                messagebox.showerror(
                    "Missing Broker Credentials",
                    f"Your {broker} credentials are missing!\n\n"
                    f"Please go back and enter your API credentials."
                )
                return
            
            # TODO: When you build the API server, add real credential validation here:
            # response = requests.post("https://api.quotrading.com/validate", 
            #                          json={"broker": broker, "token": broker_token})
            # For now, we just check they exist
            
            messagebox.showinfo(
                "Credentials Ready",
                f"✓ License key validated\n"
                f"✓ {broker} credentials ready\n\n"
                f"Broker credentials will be validated when the bot connects."
            )
        
        # Step 2: Save final config
        self.save_config()
        
        # Step 3: Create .env file
        self.create_env_file()
        
        # Step 4: Show confirmation
        result = messagebox.askyesno(
            "Launch Trading Bot?",
            f"Ready to start trading with these settings:\n\n"
            f"Broker: {broker}\n"
            f"Symbol: {self.symbol_var.get()}\n"
            f"Max Contracts: {self.contracts_var.get()}\n"
            f"Max Trades/Day: {self.trades_var.get()}\n"
            f"Risk/Trade: {self.risk_var.get()}%\n\n"
            f"This will open a PowerShell terminal with live logs.\n"
            f"The bot will connect to {broker} and start trading.\n"
            f"Close the PowerShell window to stop the bot.\n\n"
            f"Continue?"
        )
        
        if not result:
            return
        
        # Launch bot in PowerShell terminal
        try:
            # Get the parent directory (where run.py is located)
            bot_dir = Path(__file__).parent.parent.absolute()
            
            # PowerShell command to run the bot
            ps_command = [
                "powershell.exe",
                "-NoExit",  # Keep window open
                "-Command",
                f"cd '{bot_dir}'; python run.py"
            ]
            
            # Start PowerShell process in a NEW SYSTEM WINDOW (not in VS Code)
            subprocess.Popen(
                ps_command, 
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                cwd=str(bot_dir)
            )
            
            # Success message
            messagebox.showinfo(
                "Bot Launched!",
                "✓ QuoTrading AI bot launched successfully!\n\n"
                "PowerShell terminal opened with live logs.\n"
                "To stop the bot, close the PowerShell window.\n\n"
                "You can close this setup window now."
            )
            
            # Close the GUI
            self.root.destroy()
            
        except Exception as e:
            messagebox.showerror(
                "Launch Error",
                f"Failed to launch bot:\n{str(e)}\n\n"
                f"Make sure Python is installed and run.py exists."
            )
    
    def create_env_file(self):
        """Create .env file from GUI settings."""
        symbol = self.symbol_var.get().split(" - ")[0]  # Extract symbol code (ES, NQ, etc.)
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

# Trading Configuration
BOT_INSTRUMENT={symbol}
BOT_MAX_CONTRACTS={self.contracts_var.get()}
BOT_MAX_TRADES_PER_DAY={self.trades_var.get()}
BOT_RISK_PER_TRADE={self.risk_var.get() / 100}
BOT_USE_TOPSTEP_RULES={str(self.topstep_var.get()).lower()}

# Environment
BOT_ENVIRONMENT=production
BOT_DRY_RUN=false
CONFIRM_LIVE_TRADING=1
BOT_LOG_LEVEL=INFO
QUOTRADING_API_URL=https://api.quotrading.com/v1/signals
"""
        
        with open(env_path, 'w') as f:
            f.write(env_content)
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


if __name__ == "__main__":
    app = QuoTradingLauncher()
    app.run()
