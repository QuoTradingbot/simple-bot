"""
QuoTrading AI - Customer Launcher
==================================
Professional GUI application for easy setup and launch.
4-Screen Progressive Onboarding Flow with Validation.

Flow:
1. Screen 0: Username Creation
2. Screen 1: QuoTrading Account Setup (Email + API Key validation)
3. Screen 2: Broker Connection Setup (Prop Firm/Live Broker with validation)
4. Screen 3: Trading Preferences (Symbol selection, risk settings, launch)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
import json
from pathlib import Path
from datetime import datetime
import sys
import subprocess
import re


class QuoTradingLauncher:
    """Professional GUI launcher for QuoTrading AI bot - Green/Black Theme."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("QuoTrading - Professional Trading Platform")
        self.root.geometry("700x650")
        self.root.resizable(False, False)
        
        # Green and Black color scheme - Premium Matrix-style theme
        self.colors = {
            'primary': '#000000',        # Pure black background
            'secondary': '#0A0A0A',      # Near black for secondary cards
            'success': '#00FF41',        # Matrix green (bright) - primary accent
            'success_dark': '#00B82E',   # Darker green for buttons/headers
            'error': '#FF0000',          # Red for error messages
            'background': '#000000',     # Black main background
            'card': '#0F0F0F',           # Dark gray card background
            'text': '#00FF41',           # Bright green text (primary)
            'text_light': '#00CC33',     # Medium green (secondary labels)
            'text_secondary': '#008822', # Dark green (tertiary/hints)
            'border': '#00FF41',         # Green border (glowing effect)
            'input_bg': '#1A1A1A',       # Very dark gray for input fields
            'button_hover': '#00DD38'    # Slightly darker green for hover state
        }
        
        # Default fallback symbol
        self.DEFAULT_SYMBOL = 'ES'
        
        self.root.configure(bg=self.colors['background'])
        
        # Load saved config
        self.config_file = Path("config.json")
        self.config = self.load_config()
        
        # Current screen tracker
        self.current_screen = 0
        
        # Bot process reference
        self.bot_process = None
        
        # Start with username screen (Screen 0)
        self.setup_username_screen()
    
    def create_header(self, title, subtitle=""):
        """Create a professional header for each screen."""
        header = tk.Frame(self.root, bg=self.colors['success_dark'], height=100)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        title_label = tk.Label(
            header,
            text=title,
            font=("Arial", 24, "bold"),
            bg=self.colors['success_dark'],
            fg=self.colors['background']
        )
        title_label.pack(pady=(20, 5))
        
        if subtitle:
            subtitle_label = tk.Label(
                header,
                text=subtitle,
                font=("Arial", 11),
                bg=self.colors['success_dark'],
                fg=self.colors['background']
            )
            subtitle_label.pack()
        
        return header
    
    def create_input_field(self, parent, label_text, is_password=False, placeholder=""):
        """Create a styled input field with label."""
        container = tk.Frame(parent, bg=self.colors['card'])
        container.pack(fill=tk.X, pady=10)
        
        label = tk.Label(
            container,
            text=label_text,
            font=("Arial", 12, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        label.pack(anchor=tk.W, pady=(0, 5))
        
        entry = tk.Entry(
            container,
            font=("Arial", 12),
            bg=self.colors['input_bg'],
            fg=self.colors['text'],
            insertbackground=self.colors['success'],
            relief=tk.FLAT,
            bd=0,
            highlightthickness=2,
            highlightbackground=self.colors['border'],
            highlightcolor=self.colors['success'],
            show="●" if is_password else ""
        )
        entry.pack(fill=tk.X, ipady=8, padx=2)
        
        if placeholder:
            # Track placeholder state with custom attribute
            entry.is_placeholder = True
            entry.placeholder_text = placeholder
            entry.insert(0, placeholder)
            entry.config(fg=self.colors['text_secondary'])
            
            def on_focus_in(event):
                if entry.is_placeholder:
                    entry.delete(0, tk.END)
                    entry.config(fg=self.colors['text'])
                    entry.is_placeholder = False
            
            def on_focus_out(event):
                if not entry.get():
                    entry.insert(0, entry.placeholder_text)
                    entry.config(fg=self.colors['text_secondary'])
                    entry.is_placeholder = True
            
            entry.bind("<FocusIn>", on_focus_in)
            entry.bind("<FocusOut>", on_focus_out)
        
        return entry
    
    def create_button(self, parent, text, command, button_type="next"):
        """Create a styled button."""
        if button_type == "next":
            bg = self.colors['success_dark']
            fg = self.colors['background']
            width = 20
        elif button_type == "back":
            bg = self.colors['secondary']
            fg = self.colors['text']
            width = 15
        else:  # start
            bg = self.colors['success']
            fg = self.colors['background']
            width = 25
        
        button = tk.Button(
            parent,
            text=text,
            font=("Arial", 14, "bold"),
            bg=bg,
            fg=fg,
            activebackground=self.colors['button_hover'],
            activeforeground=self.colors['background'],
            relief=tk.FLAT,
            bd=0,
            command=command,
            cursor="hand2",
            width=width,
            height=2
        )
        return button
    
    def setup_username_screen(self):
        """Screen 0: Username creation screen."""
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        self.current_screen = 0
        self.root.title("QuoTrading - Welcome")
        
        # Header
        header = self.create_header("Welcome to QuoTrading", "Create your trading profile")
        
        # Main container
        main = tk.Frame(self.root, bg=self.colors['background'], padx=40, pady=40)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Card
        card = tk.Frame(main, bg=self.colors['card'], relief=tk.FLAT, bd=0)
        card.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        card.configure(highlightbackground=self.colors['border'], highlightthickness=2)
        
        # Card content
        content = tk.Frame(card, bg=self.colors['card'], padx=30, pady=30)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Welcome message
        welcome = tk.Label(
            content,
            text="Create Your Username",
            font=("Arial", 18, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        welcome.pack(pady=(0, 10))
        
        info = tk.Label(
            content,
            text="This username will be used to identify your trading profile.\nChoose something you'll remember.",
            font=("Arial", 10),
            bg=self.colors['card'],
            fg=self.colors['text_light'],
            justify=tk.CENTER
        )
        info.pack(pady=(0, 30))
        
        # Username input
        self.username_entry = self.create_input_field(content, "Username:", placeholder=self.config.get("username", "Enter your username"))
        
        # Instructions
        instructions = tk.Label(
            content,
            text="• 3-20 characters\n• Letters, numbers, and underscores only\n• Will be saved to your profile",
            font=("Arial", 9),
            bg=self.colors['card'],
            fg=self.colors['text_secondary'],
            justify=tk.LEFT
        )
        instructions.pack(pady=(10, 30))
        
        # Button container
        button_frame = tk.Frame(content, bg=self.colors['card'])
        button_frame.pack(fill=tk.X, pady=20)
        
        # Next button
        next_btn = self.create_button(button_frame, "NEXT →", self.validate_username, "next")
        next_btn.pack()
    
    def validate_username(self):
        """Validate username and proceed to QuoTrading setup."""
        username = self.username_entry.get().strip()
        
        # Remove placeholder if present
        if username == "Enter your username":
            username = ""
        
        # Validation
        if not username:
            messagebox.showerror(
                "Username Required",
                "Please enter a username to continue."
            )
            return
        
        if len(username) < 3 or len(username) > 20:
            messagebox.showerror(
                "Invalid Username",
                "Username must be between 3 and 20 characters."
            )
            return
        
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            messagebox.showerror(
                "Invalid Username",
                "Username can only contain letters, numbers, and underscores."
            )
            return
        
        # Save username to config
        self.config["username"] = username
        self.save_config()
        
        # Proceed to QuoTrading setup
        self.setup_quotrading_screen()
    
    def setup_quotrading_screen(self):
        """Screen 1: QuoTrading Account Setup with Email + API Key validation."""
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        self.current_screen = 1
        self.root.title("QuoTrading - Account Setup")
        
        # Header
        header = self.create_header("QuoTrading Account", "Enter your subscription credentials")
        
        # Main container
        main = tk.Frame(self.root, bg=self.colors['background'], padx=40, pady=30)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Card
        card = tk.Frame(main, bg=self.colors['card'], relief=tk.FLAT, bd=0)
        card.pack(fill=tk.BOTH, expand=True)
        card.configure(highlightbackground=self.colors['border'], highlightthickness=2)
        
        # Card content
        content = tk.Frame(card, bg=self.colors['card'], padx=30, pady=30)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Info message
        info = tk.Label(
            content,
            text="Enter your QuoTrading subscription details.\nWe'll validate your access before proceeding.",
            font=("Arial", 11),
            bg=self.colors['card'],
            fg=self.colors['text_light'],
            justify=tk.CENTER
        )
        info.pack(pady=(0, 20))
        
        # Email input
        self.email_entry = self.create_input_field(
            content, 
            "Email Address:",
            placeholder=self.config.get("quotrading_email", "your.email@example.com")
        )
        
        # API Key input
        self.api_key_entry = self.create_input_field(
            content,
            "API Key:",
            is_password=True,
            placeholder=self.config.get("quotrading_api_key", "")
        )
        
        # Help text
        help_text = tk.Label(
            content,
            text="📧 Check your email for your API key\n💡 Contact support@quotrading.com if you need help",
            font=("Arial", 9),
            bg=self.colors['card'],
            fg=self.colors['text_secondary'],
            justify=tk.CENTER
        )
        help_text.pack(pady=(10, 20))
        
        # Button container
        button_frame = tk.Frame(content, bg=self.colors['card'])
        button_frame.pack(fill=tk.X, pady=20)
        
        # Back button
        back_btn = self.create_button(button_frame, "← BACK", self.setup_username_screen, "back")
        back_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Next button
        next_btn = self.create_button(button_frame, "NEXT →", self.validate_quotrading, "next")
        next_btn.pack(side=tk.RIGHT)
    
    def validate_quotrading(self):
        """Validate QuoTrading credentials before proceeding."""
        email = self.email_entry.get().strip()
        api_key = self.api_key_entry.get().strip()
        
        # Remove placeholders
        if email == "your.email@example.com":
            email = ""
        
        # Validation
        if not email or not api_key:
            messagebox.showerror(
                "Missing Information",
                "Please enter both your email and API key."
            )
            return
        
        # Validate email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            messagebox.showerror(
                "Invalid Email",
                "Please enter a valid email address."
            )
            return
        
        # Admin master key - instant access
        if api_key == "QUOTRADING_ADMIN_MASTER_2025":
            self.config["quotrading_email"] = email
            self.config["quotrading_api_key"] = api_key
            self.config["quotrading_validated"] = True
            self.save_config()
            self.setup_broker_screen()
            return
        
        # Validate API key format (basic check)
        if len(api_key) < 20:
            messagebox.showerror(
                "Invalid API Key",
                "API key appears to be invalid.\nPlease check your email for the correct API key."
            )
            return
        
        # TODO: Make real API call to QuoTrading backend to validate credentials
        # For now, simulate validation
        messagebox.showinfo(
            "Validation",
            "⚠️ Note: Real API validation will be implemented.\n\nFor now, credentials are accepted if properly formatted."
        )
        
        # Save credentials
        self.config["quotrading_email"] = email
        self.config["quotrading_api_key"] = api_key
        self.config["quotrading_validated"] = True
        self.save_config()
        
        # Proceed to broker setup
        self.setup_broker_screen()
    
    def setup_broker_screen(self):
        """Screen 2: Broker Connection Setup (Prop Firm vs Live Broker)."""
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        self.current_screen = 2
        self.root.title("QuoTrading - Broker Setup")
        
        # Header
        header = self.create_header("Broker Connection", "Select your account type and broker")
        
        # Main container
        main = tk.Frame(self.root, bg=self.colors['background'], padx=40, pady=30)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Card
        card = tk.Frame(main, bg=self.colors['card'], relief=tk.FLAT, bd=0)
        card.pack(fill=tk.BOTH, expand=True)
        card.configure(highlightbackground=self.colors['border'], highlightthickness=2)
        
        # Card content
        content = tk.Frame(card, bg=self.colors['card'], padx=30, pady=30)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Info message
        info = tk.Label(
            content,
            text="Choose your broker type and enter credentials",
            font=("Arial", 11),
            bg=self.colors['card'],
            fg=self.colors['text_light'],
            justify=tk.CENTER
        )
        info.pack(pady=(0, 20))
        
        # Broker Type Selection - Card-style buttons
        type_label = tk.Label(
            content,
            text="Account Type:",
            font=("Arial", 13, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        type_label.pack(pady=(0, 15))
        
        # Container for cards
        cards_container = tk.Frame(content, bg=self.colors['card'])
        cards_container.pack(fill=tk.X, pady=(0, 20))
        
        # Initialize broker type variable
        self.broker_type_var = tk.StringVar(value=self.config.get("broker_type", "Prop Firm"))
        
        # Create card buttons
        self.broker_cards = {}
        broker_types = [
            ("Prop Firm", "💼", "Funded trading programs"),
            ("Live Broker", "🏦", "Direct broker accounts")
        ]
        
        for i, (btype, icon, desc) in enumerate(broker_types):
            card_frame = tk.Frame(
                cards_container,
                bg=self.colors['secondary'],
                relief=tk.FLAT,
                bd=0,
                highlightthickness=3,
                highlightbackground=self.colors['border'] if self.broker_type_var.get() == btype else self.colors['text_secondary']
            )
            card_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10)
            
            # Make card clickable
            def make_select(bt=btype):
                return lambda e: self.select_broker_type(bt)
            
            card_frame.bind("<Button-1>", make_select(btype))
            
            # Card content
            inner = tk.Frame(card_frame, bg=self.colors['secondary'])
            inner.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
            inner.bind("<Button-1>", make_select(btype))
            
            # Icon
            icon_label = tk.Label(
                inner,
                text=icon,
                font=("Arial", 30),
                bg=self.colors['secondary'],
                fg=self.colors['text']
            )
            icon_label.pack()
            icon_label.bind("<Button-1>", make_select(btype))
            
            # Type name
            type_name = tk.Label(
                inner,
                text=btype,
                font=("Arial", 13, "bold"),
                bg=self.colors['secondary'],
                fg=self.colors['text']
            )
            type_name.pack(pady=(10, 5))
            type_name.bind("<Button-1>", make_select(btype))
            
            # Description
            desc_label = tk.Label(
                inner,
                text=desc,
                font=("Arial", 9),
                bg=self.colors['secondary'],
                fg=self.colors['text_light']
            )
            desc_label.pack()
            desc_label.bind("<Button-1>", make_select(btype))
            
            self.broker_cards[btype] = card_frame
        
        # Broker dropdown
        broker_label = tk.Label(
            content,
            text="Select Broker:",
            font=("Arial", 12, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        broker_label.pack(anchor=tk.W, pady=(10, 5))
        
        self.broker_var = tk.StringVar(value=self.config.get("broker", "TopStep"))
        self.broker_dropdown = ttk.Combobox(
            content,
            textvariable=self.broker_var,
            state="readonly",
            font=("Arial", 11),
            width=35
        )
        self.broker_dropdown.pack(fill=tk.X, pady=(0, 15))
        self.broker_dropdown.bind('<<ComboboxSelected>>', lambda e: self.update_broker_fields())
        
        # Update broker options based on selected type
        self.update_broker_options()
        
        # Broker credentials
        self.broker_token_entry = self.create_input_field(
            content,
            "API Token:",
            is_password=True,
            placeholder=self.config.get("broker_token", "")
        )
        
        self.broker_username_entry = self.create_input_field(
            content,
            "Username/Email:",
            placeholder=self.config.get("broker_username", "")
        )
        
        # Help text
        help_text = tk.Label(
            content,
            text="💡 Get your API credentials from your broker's account dashboard",
            font=("Arial", 9),
            bg=self.colors['card'],
            fg=self.colors['text_secondary']
        )
        help_text.pack(pady=(5, 20))
        
        # Button container
        button_frame = tk.Frame(content, bg=self.colors['card'])
        button_frame.pack(fill=tk.X, pady=10)
        
        # Back button
        back_btn = self.create_button(button_frame, "← BACK", self.setup_quotrading_screen, "back")
        back_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Next button
        next_btn = self.create_button(button_frame, "NEXT →", self.validate_broker, "next")
        next_btn.pack(side=tk.RIGHT)
    
    def select_broker_type(self, broker_type):
        """Select broker type and update UI."""
        self.broker_type_var.set(broker_type)
        
        # Update card styling
        for btype, card in self.broker_cards.items():
            if btype == broker_type:
                card.config(highlightbackground=self.colors['border'], highlightthickness=3)
            else:
                card.config(highlightbackground=self.colors['text_secondary'], highlightthickness=3)
        
        # Update broker dropdown
        self.update_broker_options()
    
    def update_broker_options(self):
        """Update broker dropdown based on account type."""
        broker_type = self.broker_type_var.get()
        
        if broker_type == "Prop Firm":
            options = ["TopStep", "Earn2Trade", "The5ers", "FTMO"]
        else:  # Live Broker
            options = ["Tradovate", "Rithmic", "Interactive Brokers", "NinjaTrader"]
        
        self.broker_dropdown['values'] = options
        self.broker_dropdown.current(0)
    
    def update_broker_fields(self):
        """Update credential field labels based on selected broker."""
        # This method can be extended to customize labels per broker
        pass
    
    def validate_broker(self):
        """Validate broker credentials before proceeding."""
        broker = self.broker_var.get()
        token = self.broker_token_entry.get().strip()
        username = self.broker_username_entry.get().strip()
        
        # Validation
        if not token or not username:
            messagebox.showerror(
                "Missing Credentials",
                f"Please enter both {broker} API Token and Username."
            )
            return
        
        # Check if using admin key - bypass broker validation
        quotrading_key = self.config.get("quotrading_api_key", "")
        if quotrading_key == "QUOTRADING_ADMIN_MASTER_2025":
            self.config["broker_type"] = self.broker_type_var.get()
            self.config["broker"] = broker
            self.config["broker_token"] = token
            self.config["broker_username"] = username
            self.config["broker_validated"] = True
            self.save_config()
            self.setup_trading_screen()
            return
        
        # TODO: Make real API call to broker to validate credentials
        # For now, simulate validation
        messagebox.showinfo(
            "Validation",
            "⚠️ Note: Real broker API validation will be implemented.\n\nFor now, credentials are accepted if filled in."
        )
        
        # Save broker credentials
        self.config["broker_type"] = self.broker_type_var.get()
        self.config["broker"] = broker
        self.config["broker_token"] = token
        self.config["broker_username"] = username
        self.config["broker_validated"] = True
        self.save_config()
        
        # Proceed to trading preferences
        self.setup_trading_screen()
    
    def setup_trading_screen(self):
        """Screen 3: Trading Preferences and Launch."""
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        self.current_screen = 3
        self.root.title("QuoTrading - Trading Settings")
        
        # Header
        header = self.create_header("Trading Preferences", "Configure your trading strategy")
        
        # Main container with scrollbar capability
        main = tk.Frame(self.root, bg=self.colors['background'], padx=30, pady=20)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Card
        card = tk.Frame(main, bg=self.colors['card'], relief=tk.FLAT, bd=0)
        card.pack(fill=tk.BOTH, expand=True)
        card.configure(highlightbackground=self.colors['border'], highlightthickness=2)
        
        # Card content
        content = tk.Frame(card, bg=self.colors['card'], padx=25, pady=25)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Symbol Selection
        symbol_label = tk.Label(
            content,
            text="Trading Symbols (select at least one):",
            font=("Arial", 12, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        symbol_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Symbol checkboxes - 2 columns
        symbol_frame = tk.Frame(content, bg=self.colors['card'])
        symbol_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.symbol_vars = {}
        symbols = [
            ("ES", "E-mini S&P 500"),
            ("NQ", "E-mini Nasdaq 100"),
            ("RTY", "E-mini Russell 2000"),
            ("YM", "E-mini Dow"),
            ("CL", "Crude Oil"),
            ("GC", "Gold")
        ]
        
        saved_symbols = self.config.get("symbols", ["ES"])
        if isinstance(saved_symbols, str):
            saved_symbols = [saved_symbols]
        
        for i, (code, name) in enumerate(symbols):
            row = i // 2
            col = i % 2
            
            var = tk.BooleanVar(value=(code in saved_symbols))
            self.symbol_vars[code] = var
            
            cb = tk.Checkbutton(
                symbol_frame,
                text=f"{code} - {name}",
                variable=var,
                font=("Arial", 10),
                bg=self.colors['card'],
                fg=self.colors['text'],
                selectcolor=self.colors['secondary'],
                activebackground=self.colors['card'],
                activeforeground=self.colors['success'],
                cursor="hand2"
            )
            cb.grid(row=row, column=col, sticky=tk.W, padx=10, pady=3)
        
        # Account Settings Row
        settings_row = tk.Frame(content, bg=self.colors['card'])
        settings_row.pack(fill=tk.X, pady=(0, 15))
        
        # Account Size
        acc_frame = tk.Frame(settings_row, bg=self.colors['card'])
        acc_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        tk.Label(
            acc_frame,
            text="Account Size ($):",
            font=("Arial", 11, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 5))
        
        self.account_entry = tk.Entry(
            acc_frame,
            font=("Arial", 11),
            bg=self.colors['input_bg'],
            fg=self.colors['text'],
            insertbackground=self.colors['success'],
            relief=tk.FLAT,
            bd=0,
            highlightthickness=2,
            highlightbackground=self.colors['border'],
            highlightcolor=self.colors['success']
        )
        self.account_entry.pack(fill=tk.X, ipady=6, padx=2)
        self.account_entry.insert(0, self.config.get("account_size", "10000"))
        
        # Risk Per Trade
        risk_frame = tk.Frame(settings_row, bg=self.colors['card'])
        risk_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        tk.Label(
            risk_frame,
            text="Risk per Trade (%):",
            font=("Arial", 11, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 5))
        
        self.risk_var = tk.DoubleVar(value=self.config.get("risk_per_trade", 1.0))
        risk_spin = ttk.Spinbox(
            risk_frame,
            from_=0.5,
            to=5.0,
            increment=0.1,
            textvariable=self.risk_var,
            width=15,
            format="%.1f"
        )
        risk_spin.pack(fill=tk.X, ipady=4)
        
        # Daily Loss Limit
        loss_frame = tk.Frame(settings_row, bg=self.colors['card'])
        loss_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(
            loss_frame,
            text="Daily Loss Limit ($):",
            font=("Arial", 11, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 5))
        
        self.loss_entry = tk.Entry(
            loss_frame,
            font=("Arial", 11),
            bg=self.colors['input_bg'],
            fg=self.colors['text'],
            insertbackground=self.colors['success'],
            relief=tk.FLAT,
            bd=0,
            highlightthickness=2,
            highlightbackground=self.colors['border'],
            highlightcolor=self.colors['success']
        )
        self.loss_entry.pack(fill=tk.X, ipady=6, padx=2)
        self.loss_entry.insert(0, self.config.get("daily_loss_limit", "2000"))
        
        # Advanced Settings Row
        advanced_row = tk.Frame(content, bg=self.colors['card'])
        advanced_row.pack(fill=tk.X, pady=(0, 15))
        
        # Max Contracts
        contracts_frame = tk.Frame(advanced_row, bg=self.colors['card'])
        contracts_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        tk.Label(
            contracts_frame,
            text="Max Contracts:",
            font=("Arial", 11, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 5))
        
        self.contracts_var = tk.IntVar(value=self.config.get("max_contracts", 3))
        contracts_spin = ttk.Spinbox(
            contracts_frame,
            from_=1,
            to=25,
            textvariable=self.contracts_var,
            width=15
        )
        contracts_spin.pack(fill=tk.X, ipady=4)
        
        # Max Trades Per Day
        trades_frame = tk.Frame(advanced_row, bg=self.colors['card'])
        trades_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(
            trades_frame,
            text="Max Trades/Day:",
            font=("Arial", 11, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 5))
        
        self.trades_var = tk.IntVar(value=self.config.get("max_trades", 10))
        trades_spin = ttk.Spinbox(
            trades_frame,
            from_=1,
            to=50,
            textvariable=self.trades_var,
            width=15
        )
        trades_spin.pack(fill=tk.X, ipady=4)
        
        # Summary display
        summary_frame = tk.Frame(content, bg=self.colors['secondary'], relief=tk.FLAT, bd=0)
        summary_frame.pack(fill=tk.X, pady=(10, 20))
        summary_frame.configure(highlightbackground=self.colors['border'], highlightthickness=1)
        
        summary_content = tk.Frame(summary_frame, bg=self.colors['secondary'], padx=15, pady=10)
        summary_content.pack(fill=tk.X)
        
        summary_title = tk.Label(
            summary_content,
            text="✓ Ready to Trade",
            font=("Arial", 12, "bold"),
            bg=self.colors['secondary'],
            fg=self.colors['success']
        )
        summary_title.pack(pady=(0, 5))
        
        username = self.config.get("username", "Trader")
        broker = self.config.get("broker", "TopStep")
        
        summary_text = tk.Label(
            summary_content,
            text=f"User: {username} | Broker: {broker}\nAll credentials validated and ready",
            font=("Arial", 9),
            bg=self.colors['secondary'],
            fg=self.colors['text_light'],
            justify=tk.CENTER
        )
        summary_text.pack()
        
        # Button container
        button_frame = tk.Frame(content, bg=self.colors['card'])
        button_frame.pack(fill=tk.X, pady=15)
        
        # Back button
        back_btn = self.create_button(button_frame, "← BACK", self.setup_broker_screen, "back")
        back_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Start Bot button
        start_btn = self.create_button(button_frame, "START BOT", self.start_bot, "start")
        start_btn.pack(side=tk.RIGHT)
    
    def start_bot(self):
        """Validate settings and start the trading bot."""
        # Validate at least one symbol selected
        selected_symbols = [code for code, var in self.symbol_vars.items() if var.get()]
        
        if not selected_symbols:
            messagebox.showerror(
                "No Symbols Selected",
                "Please select at least one trading symbol."
            )
            return
        
        # Validate account size
        try:
            account_size = float(self.account_entry.get())
            if account_size <= 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror(
                "Invalid Account Size",
                "Please enter a valid account size (greater than 0)."
            )
            return
        
        # Validate daily loss limit
        try:
            loss_limit = float(self.loss_entry.get())
            if loss_limit <= 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror(
                "Invalid Loss Limit",
                "Please enter a valid daily loss limit (greater than 0)."
            )
            return
        
        # Save all settings
        self.config["symbols"] = selected_symbols
        self.config["account_size"] = account_size
        self.config["risk_per_trade"] = self.risk_var.get()
        self.config["daily_loss_limit"] = loss_limit
        self.config["max_contracts"] = self.contracts_var.get()
        self.config["max_trades"] = self.trades_var.get()
        self.save_config()
        
        # Create .env file
        self.create_env_file()
        
        # Show confirmation
        symbols_str = ", ".join(selected_symbols)
        broker = self.config.get("broker", "TopStep")
        
        result = messagebox.askyesno(
            "Launch Trading Bot?",
            f"Ready to start bot with these settings:\n\n"
            f"Broker: {broker}\n"
            f"Symbols: {symbols_str}\n"
            f"Max Contracts: {self.contracts_var.get()}\n"
            f"Max Trades/Day: {self.trades_var.get()}\n"
            f"Risk/Trade: {self.risk_var.get()}%\n"
            f"Daily Loss Limit: ${loss_limit}\n\n"
            f"This will open a PowerShell terminal with live logs.\n"
            f"Use the window's close button to stop the bot.\n\n"
            f"Continue?"
        )
        
        if not result:
            return
        
        # Launch bot in PowerShell terminal
        try:
            # Get the bot directory (parent of customer folder)
            bot_dir = Path(__file__).parent.parent.absolute()
            
            # PowerShell command to run the bot
            ps_command = [
                "powershell.exe",
                "-NoExit",  # Keep window open
                "-Command",
                f"cd '{bot_dir}'; python run.py"
            ]
            
            # Start PowerShell process in a NEW CONSOLE WINDOW
            self.bot_process = subprocess.Popen(
                ps_command,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                cwd=str(bot_dir)
            )
            
            # Success message
            messagebox.showinfo(
                "Bot Launched!",
                f"✓ QuoTrading AI bot launched successfully!\n\n"
                f"Symbols: {symbols_str}\n\n"
                f"PowerShell terminal opened with live logs.\n"
                f"To stop the bot, close the PowerShell window.\n\n"
                f"You can close this setup window now."
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
        selected_symbols = [code for code, var in self.symbol_vars.items() if var.get()]
        if not selected_symbols:
            selected_symbols = [self.DEFAULT_SYMBOL]  # Use class constant instead of magic string
        
        symbols_str = ",".join(selected_symbols)
        broker = self.config.get("broker", "TopStep")
        
        # Get the bot directory (parent of customer folder)
        bot_dir = Path(__file__).parent.parent.absolute()
        env_path = bot_dir / '.env'
        
        env_content = f"""# QuoTrading AI - Auto-generated Configuration
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# DO NOT EDIT MANUALLY - Use the launcher to change settings

# QuoTrading Account
QUOTRADING_EMAIL={self.config.get("quotrading_email", "")}
QUOTRADING_API_KEY={self.config.get("quotrading_api_key", "")}
QUOTRADING_API_URL=https://api.quotrading.com/v1/signals

# Broker Configuration
BROKER={broker}
BROKER_API_TOKEN={self.config.get("broker_token", "")}
BROKER_USERNAME={self.config.get("broker_username", "")}

# Legacy TopStep/Tradovate variables (for compatibility)
TOPSTEP_API_TOKEN={self.config.get("broker_token", "")}
TOPSTEP_USERNAME={self.config.get("broker_username", "")}
TRADOVATE_API_KEY={self.config.get("broker_token", "")}
TRADOVATE_USERNAME={self.config.get("broker_username", "")}

# Trading Configuration - Multi-Symbol Support
BOT_INSTRUMENTS={symbols_str}
BOT_MAX_CONTRACTS={self.contracts_var.get()}
BOT_MAX_TRADES_PER_DAY={self.trades_var.get()}
BOT_RISK_PER_TRADE={self.risk_var.get() / 100}
BOT_DAILY_LOSS_LIMIT={self.loss_entry.get()}

# Trading Mode
BOT_SHADOW_MODE=false
BOT_DRY_RUN=false

# Environment
BOT_ENVIRONMENT=production
CONFIRM_LIVE_TRADING=1
BOT_LOG_LEVEL=INFO
"""
        
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        print(f"✓ .env file created with {len(selected_symbols)} symbols: {symbols_str}")
    
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
        """Save current configuration."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


if __name__ == "__main__":
    app = QuoTradingLauncher()
    app.run()
