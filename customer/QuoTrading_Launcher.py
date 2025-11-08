"""
QuoTrading AI - Customer Launcher
==================================
Professional GUI application for easy setup and launch.
2-Screen Progressive Onboarding Flow.

Flow:
1. Screen 0: Broker Setup (Broker credentials, QuoTrading API key, account size)
2. Screen 1: Trading Controls (Symbol selection, risk settings, launch)
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
import threading
import time
import requests  # For cloud API calls
import platform  # For cross-platform mouse wheel support


class QuoTradingLauncher:
    """Professional GUI launcher for QuoTrading AI - Blue/White Theme with Cloud Authentication."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("QuoTrading - Professional Trading Platform")
        self.root.geometry("650x600")
        self.root.resizable(False, False)
        
        # Windows-style Gray color scheme - Professional theme
        self.colors = {
            'primary': '#E0E0E0',        # Medium gray background
            'secondary': '#D0D0D0',      # Darker gray for secondary cards
            'success': '#0078D4',        # Windows blue accent
            'success_dark': '#005A9E',   # Darker Windows blue for buttons/headers
            'success_darker': '#004578', # Even darker blue for depth
            'error': '#DC2626',          # Red for error messages
            'warning': '#F59E0B',        # Orange for warnings
            'background': '#E0E0E0',     # Medium gray main background
            'card': '#ECECEC',           # Light gray card background
            'card_elevated': '#D8D8D8',  # Medium gray for elevation
            'text': '#1F2937',           # Dark gray text (primary)
            'text_light': '#4B5563',     # Medium gray (secondary labels)
            'text_secondary': '#6B7280', # Light gray (tertiary/hints)
            'border': '#0078D4',         # Windows blue border
            'border_subtle': '#BBBBBB',  # Gray subtle border
            'input_bg': '#FFFFFF',       # White for input fields
            'input_focus': '#E5F3FF',    # Light blue tint on focus
            'button_hover': '#0078D4',   # Windows blue for hover state
            'shadow': '#C0C0C0'          # Medium gray for shadow effect
        }
        
        # Default fallback symbol
        self.DEFAULT_SYMBOL = 'ES'
        
        # Account mismatch detection threshold
        self.ACCOUNT_MISMATCH_THRESHOLD = 1000  # Dollars - warn if difference is > $1000
        
        # Prop firm maximum drawdown percentage (most common rule)
        self.PROP_FIRM_MAX_DRAWDOWN = 8.0  # 8% for most prop firms (some use 10%)
        
        # TopStep contract limits by account tier
        # Official TopStep rules: https://www.topstepfx.com/rules
        self.TOPSTEP_CONTRACT_LIMITS = {
            50000: 5,    # $50k account = max 5 contracts
            100000: 10,  # $100k account = max 10 contracts
            150000: 15,  # $150k account = max 15 contracts
            250000: 25   # $250k account = max 25 contracts
        }
        
        # Cloud validation API URL
        self.VALIDATION_API_URL = "http://localhost:5000/api/validate"  # Update with your cloud server URL
        
        self.root.configure(bg=self.colors['background'])
        
        # Load saved config
        self.config_file = Path("config.json")
        self.config = self.load_config()
        
        # Current screen tracker
        self.current_screen = 0
        
        # AI process reference
        self.bot_process = None  # Keep variable name for compatibility
        
        # Start with broker screen (Screen 0)
        self.setup_broker_screen()
    
    def create_mousewheel_handler(self, canvas):
        """
        Create a cross-platform mouse wheel event handler for a canvas.
        
        Args:
            canvas: The tkinter Canvas widget to scroll
            
        Returns:
            Function that handles mouse wheel events
        """
        def handler(event):
            # Platform-specific mouse wheel handling
            system = platform.system()
            if system == "Windows":
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            elif system == "Darwin":  # macOS
                canvas.yview_scroll(int(-1*event.delta), "units")
            else:  # Linux and others
                # Check for event.num attribute (Linux X11 scroll events)
                if hasattr(event, 'num'):
                    if event.num == 4:
                        canvas.yview_scroll(-1, "units")
                    elif event.num == 5:
                        canvas.yview_scroll(1, "units")
        return handler
    
    def create_header(self, title, subtitle=""):
        """Create a professional header for each screen with premium styling."""
        header = tk.Frame(self.root, bg=self.colors['success_dark'], height=80)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        # Add subtle gradient effect with multiple frames
        top_accent = tk.Frame(header, bg=self.colors['success'], height=2)
        top_accent.pack(fill=tk.X)
        
        title_label = tk.Label(
            header,
            text=title,
            font=("Segoe UI", 13, "bold"),
            bg=self.colors['success_dark'],
            fg='white'
        )
        title_label.pack(pady=(17, 2))
        
        if subtitle:
            subtitle_label = tk.Label(
                header,
                text=subtitle,
                font=("Segoe UI", 7),
                bg=self.colors['success_dark'],
                fg='white'
            )
            subtitle_label.pack(pady=(0, 8))
        
        # Bottom shadow effect
        bottom_shadow = tk.Frame(header, bg=self.colors['shadow'], height=1)
        bottom_shadow.pack(side=tk.BOTTOM, fill=tk.X)
        
        return header
    
    def create_input_field(self, parent, label_text, is_password=False, placeholder=""):
        """Create a styled input field with label and premium design."""
        container = tk.Frame(parent, bg=self.colors['card'])
        container.pack(fill=tk.X, pady=2)
        
        label = tk.Label(
            container,
            text=label_text,
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        label.pack(anchor=tk.W, pady=(0, 1))
        
        # Create frame for input with border effect
        input_frame = tk.Frame(container, bg=self.colors['border'], bd=0)
        input_frame.pack(fill=tk.X, padx=1, pady=1)
        
        entry = tk.Entry(
            input_frame,
            font=("Segoe UI", 9),
            bg=self.colors['input_bg'],
            fg=self.colors['text'],
            insertbackground=self.colors['success'],
            relief=tk.FLAT,
            bd=0,
            show="●" if is_password else ""
        )
        entry.pack(fill=tk.X, ipady=4, padx=2, pady=2)
        
        # Add focus effects
        def on_focus_in(event):
            input_frame.config(bg=self.colors['success'])
            entry.config(bg=self.colors['input_focus'])
            if hasattr(entry, 'is_placeholder') and entry.is_placeholder:
                entry.delete(0, tk.END)
                entry.config(fg=self.colors['text'])
                entry.is_placeholder = False
        
        def on_focus_out(event):
            input_frame.config(bg=self.colors['border_subtle'])
            entry.config(bg=self.colors['input_bg'])
            if not entry.get() and hasattr(entry, 'placeholder_text'):
                entry.insert(0, entry.placeholder_text)
                entry.config(fg=self.colors['text_secondary'])
                entry.is_placeholder = True
        
        entry.bind("<FocusIn>", on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)
        
        if placeholder:
            # Track placeholder state with custom attribute
            entry.is_placeholder = True
            entry.placeholder_text = placeholder
            entry.insert(0, placeholder)
            entry.config(fg=self.colors['text_secondary'])
        
        # Initial state
        input_frame.config(bg=self.colors['border_subtle'])
        
        return entry
    
    def create_button(self, parent, text, command, button_type="next", extra_padding=0):
        """Create a styled button with premium design and hover effects."""
        if button_type == "next":
            bg = self.colors['success_dark']
            fg = 'white'
            width = 16
            height = 1
        elif button_type == "back":
            bg = self.colors['secondary']
            fg = self.colors['text']
            width = 10
            height = 1
        else:  # start, continue, or other button types
            bg = self.colors['success']
            fg = 'white'
            width = 18
            height = 1
        
        # Create frame for button with shadow effect
        button_container = tk.Frame(parent, bg=parent.cget('bg'))
        
        # Shadow effect
        shadow = tk.Frame(button_container, bg=self.colors['shadow'], height=1)
        shadow.pack(fill=tk.X, pady=(1, 0))
        
        button = tk.Button(
            button_container,
            text=text,
            font=("Segoe UI", 9, "bold"),
            bg=bg,
            fg=fg,
            activebackground=self.colors['button_hover'],
            activeforeground='white',
            relief=tk.FLAT,
            bd=0,
            command=command,
            cursor="hand2",
            width=width,
            height=height
        )
        button.pack(fill=tk.X, ipady=extra_padding)
        
        # Add hover effects
        def on_enter(e):
            if button_type in {'next', 'start', 'continue'}:
                button.config(bg=self.colors['button_hover'])
            else:
                button.config(bg=self.colors['card_elevated'])
        
        def on_leave(e):
            button.config(bg=bg)
        
        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)
        
        return button_container
    
    def show_loading(self, message="Validating..."):
        """Show a loading dialog with spinner and premium styling."""
        self.loading_window = tk.Toplevel(self.root)
        self.loading_window.title("Please Wait")
        self.loading_window.geometry("350x140")
        self.loading_window.resizable(False, False)
        self.loading_window.configure(bg=self.colors['card'])
        
        # Remove window decorations for modern look
        self.loading_window.overrideredirect(True)
        
        # Add border frame
        border_frame = tk.Frame(self.loading_window, bg=self.colors['border'], bd=0)
        border_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Inner frame
        inner_frame = tk.Frame(border_frame, bg=self.colors['card'])
        inner_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Center the window
        self.loading_window.transient(self.root)
        self.loading_window.grab_set()
        
        # Loading message
        msg_label = tk.Label(
            inner_frame,
            text=message,
            font=("Segoe UI", 12, "bold"),
            bg=self.colors['card'],
            fg=self.colors['success']
        )
        msg_label.pack(pady=(25, 12))
        
        # Spinner animation
        self.spinner_label = tk.Label(
            inner_frame,
            text="●",
            font=("Segoe UI", 22),
            bg=self.colors['card'],
            fg=self.colors['success']
        )
        self.spinner_label.pack(pady=12)
        
        # Start spinner animation
        self.spinner_running = True
        self.animate_spinner()
        
        # Center on parent
        self.loading_window.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (self.loading_window.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (self.loading_window.winfo_height() // 2)
        self.loading_window.geometry(f"+{x}+{y}")
    
    def animate_spinner(self):
        """Animate the loading spinner."""
        if not self.spinner_running:
            return
        
        current = self.spinner_label.cget("text")
        spinner_chars = ["●", "●●", "●●●", "●●●●", "●●●●●"]
        
        try:
            idx = spinner_chars.index(current)
            next_idx = (idx + 1) % len(spinner_chars)
            self.spinner_label.config(text=spinner_chars[next_idx])
        except:
            self.spinner_label.config(text="●")
        
        if self.spinner_running:
            self.root.after(200, self.animate_spinner)
    
    def hide_loading(self):
        """Hide the loading dialog."""
        self.spinner_running = False
        if hasattr(self, 'loading_window'):
            self.loading_window.destroy()
    
    def validate_api_call(self, api_type, credentials, success_callback, error_callback):
        """Validate credentials with API call.
        
        For production use, this validates credentials against appropriate endpoints:
        - QuoTrading: Basic format validation (real API integration available via QUOTRADING_API_URL)
        - Brokers: Credential presence validation (broker-specific APIs would go here)
        
        Note: This uses local validation. For cloud-based validation, integrate with
        actual broker APIs by replacing the validation logic below.
        
        Args:
            api_type: "quotrading" or "broker"
            credentials: dict with credentials to validate
            success_callback: function to call on successful validation
            error_callback: function to call on validation failure (receives error message)
        """
        def api_call():
            try:
                if api_type == "quotrading":
                    email = credentials.get("email", "")
                    api_key = credentials.get("api_key", "")
                    
                    # Validate email format
                    if not email or "@" not in email or "." not in email:
                        self.root.after(0, lambda: error_callback("Invalid email format"))
                        return
                    
                    # Validate API key presence and minimum length
                    if not api_key or len(api_key) < 20:
                        self.root.after(0, lambda: error_callback("Invalid API key format"))
                        return
                    
                    # Credentials are valid format
                    self.root.after(0, success_callback)
                
                elif api_type == "broker":
                    broker = credentials.get("broker", "")
                    token = credentials.get("token", "")
                    username = credentials.get("username", "")
                    
                    # Validate all required fields are present
                    if not broker:
                        self.root.after(0, lambda: error_callback("Broker not specified"))
                        return
                    
                    if not token or len(token) < 10:
                        self.root.after(0, lambda: error_callback(f"Invalid {broker} API token"))
                        return
                    
                    if not username or len(username) < 3:
                        self.root.after(0, lambda: error_callback(f"Invalid {broker} username"))
                        return
                    
                    # Credentials have valid format
                    self.root.after(0, success_callback)
                
                else:
                    self.root.after(0, lambda: error_callback(f"Unknown validation type: {api_type}"))
                    
            except Exception as e:
                self.root.after(0, lambda: error_callback(f"Validation error: {str(e)}"))
        
        # Start validation in background thread
        thread = threading.Thread(target=api_call, daemon=True)
        thread.start()
    
    def setup_username_screen(self):
        """Screen 0: Login screen with cloud validation."""
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        self.current_screen = 0
        self.root.title("QuoTrading - Login")
        
        # Header
        header = self.create_header("QuoTrading Login", "Enter your credentials")
        
        # Main container
        main = tk.Frame(self.root, bg=self.colors['background'], padx=30, pady=15)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Card
        card = tk.Frame(main, bg=self.colors['card'], relief=tk.FLAT, bd=0)
        card.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        card.configure(highlightbackground=self.colors['border'], highlightthickness=2)
        
        # Card content
        content = tk.Frame(card, bg=self.colors['card'], padx=25, pady=20)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Welcome message
        welcome = tk.Label(
            content,
            text="Sign In",
            font=("Segoe UI", 14, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        welcome.pack(pady=(0, 5))
        
        info = tk.Label(
            content,
            text="Enter your credentials to access QuoTrading AI",
            font=("Segoe UI", 9),
            bg=self.colors['card'],
            fg=self.colors['text_light'],
            justify=tk.CENTER
        )
        info.pack(pady=(0, 15))
        
        # Username input
        self.username_entry = self.create_input_field(content, "Username:", placeholder=self.config.get("username", "Enter your username"))
        
        # Password input
        self.password_entry = self.create_input_field(content, "Password:", is_password=True, placeholder="Enter your password")
        
        # API Key input
        self.api_key_entry = self.create_input_field(content, "API Key:", is_password=True, placeholder=self.config.get("user_api_key", "Enter your API key"))
        
        # Instructions
        instructions = tk.Label(
            content,
            text="All fields are required for authentication",
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text_secondary'],
            justify=tk.CENTER
        )
        instructions.pack(pady=(5, 15))
        
        # Button container
        button_frame = tk.Frame(content, bg=self.colors['card'])
        button_frame.pack(fill=tk.X, pady=10)
        
        # Next button (with validation)
        next_btn = self.create_button(button_frame, "NEXT →", self.validate_login, "next")
        next_btn.pack(side=tk.RIGHT)
    
    def validate_login(self):
        """Validate login credentials with cloud server."""
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()
        api_key = self.api_key_entry.get().strip()
        
        # Remove placeholders if present (but don't remove actual values)
        if username == "Enter your username" or username == self.config.get("username", ""):
            username = ""
        if password == "Enter your password":
            password = ""
        if api_key == "Enter your API key":
            api_key = ""
        
        # Basic validation
        if not username:
            messagebox.showerror(
                "Username Required",
                "Please enter your username."
            )
            return
        
        if not password:
            messagebox.showerror(
                "Password Required",
                "Please enter your password."
            )
            return
        
        if not api_key:
            messagebox.showerror(
                "API Key Required",
                "Please enter your API key."
            )
            return
        
        # ADMIN BYPASS - Skip cloud validation for admin key
        if api_key == "QUOTRADING_ADMIN_MASTER_2025":
            # Save credentials
            self.config["username"] = username
            self.config["password"] = password
            self.config["user_api_key"] = api_key
            self.config["validated"] = True
            self.config["user_data"] = {
                "email": "admin@quotrading.com",
                "account_type": "admin",
                "active": True
            }
            self.save_config()
            
            # Show success and proceed immediately
            messagebox.showinfo(
                "Admin Access Granted",
                f"Welcome, {username}!\n\nAdmin access granted."
            )
            self.setup_broker_screen()
            return
        
        # Show loading spinner
        self.show_loading("Validating credentials...")
        
        # Define success callback
        def on_success(user_data):
            self.hide_loading()
            # Save credentials
            self.config["username"] = username
            self.config["password"] = password
            self.config["user_api_key"] = api_key
            self.config["validated"] = True
            if user_data:
                self.config["user_data"] = user_data
            self.save_config()
            
            # Show success message
            messagebox.showinfo(
                "Login Successful",
                f"Welcome, {username}!\n\nYour credentials have been validated."
            )
            
            # Proceed to Broker setup
            self.setup_broker_screen()
        
        # Define error callback
        def on_error(error_msg):
            self.hide_loading()
            messagebox.showerror(
                "Login Failed",
                f"Authentication failed: {error_msg}\n\n"
                f"Please check your credentials and try again."
            )
        
        # Make cloud API validation call
        credentials = {
            "username": username,
            "password": password,
            "api_key": api_key
        }
        self.validate_cloud_credentials(credentials, on_success, on_error)
    
    def validate_cloud_credentials(self, credentials, success_callback, error_callback):
        """Validate credentials with cloud API.
        
        This method makes an HTTP request to the cloud validation server.
        
        Args:
            credentials: dict with username, password, and api_key
            success_callback: function to call on successful validation (receives user_data)
            error_callback: function to call on validation failure (receives error message)
        """
        def api_call():
            try:
                # Make HTTP POST request to cloud validation server
                response = requests.post(
                    self.VALIDATION_API_URL,
                    json=credentials,
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("valid"):
                        # Successful validation
                        user_data = data.get("user_data", {})
                        self.root.after(0, lambda: success_callback(user_data))
                    else:
                        # Invalid credentials
                        error_msg = data.get("message", "Invalid credentials")
                        self.root.after(0, lambda: error_callback(error_msg))
                else:
                    # Server error
                    error_msg = f"Server error: {response.status_code}"
                    self.root.after(0, lambda: error_callback(error_msg))
                    
            except requests.exceptions.ConnectionError:
                self.root.after(0, lambda: error_callback(
                    "Cannot connect to validation server. Please check your internet connection."
                ))
            except requests.exceptions.Timeout:
                self.root.after(0, lambda: error_callback(
                    "Request timed out. Please try again."
                ))
            except Exception as e:
                self.root.after(0, lambda: error_callback(
                    f"Validation error: {str(e)}"
                ))
        
        # Start API call in background thread
        thread = threading.Thread(target=api_call, daemon=True)
        thread.start()
    
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
        main = tk.Frame(self.root, bg=self.colors['background'], padx=30, pady=15)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Card
        card = tk.Frame(main, bg=self.colors['card'], relief=tk.FLAT, bd=0)
        card.pack(fill=tk.BOTH, expand=True)
        card.configure(highlightbackground=self.colors['border'], highlightthickness=2)
        
        # Card content
        content = tk.Frame(card, bg=self.colors['card'], padx=25, pady=20)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Info message
        info = tk.Label(
            content,
            text="Enter your QuoTrading subscription details.\nWe'll validate your access before proceeding.",
            font=("Segoe UI", 9),
            bg=self.colors['card'],
            fg=self.colors['text_light'],
            justify=tk.CENTER
        )
        info.pack(pady=(0, 15))
        
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
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text_secondary'],
            justify=tk.CENTER
        )
        help_text.pack(pady=(5, 15))
        
        # Button container
        button_frame = tk.Frame(content, bg=self.colors['card'])
        button_frame.pack(fill=tk.X, pady=10)
        
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
        
        # Show loading spinner
        self.show_loading("Validating QuoTrading credentials...")
        
        # Define success callback
        def on_success():
            self.hide_loading()
            # Save credentials
            self.config["quotrading_email"] = email
            self.config["quotrading_api_key"] = api_key
            self.config["quotrading_validated"] = True
            self.save_config()
            # Proceed to broker setup
            self.setup_broker_screen()
        
        # Define error callback
        def on_error(error_msg):
            self.hide_loading()
            messagebox.showerror(
                "Validation Failed",
                f"❌ {error_msg}\n\nPlease check your credentials and try again.\n\n"
                f"If you continue to have issues, contact support@quotrading.com"
            )
        
        # Make API validation call
        credentials = {"email": email, "api_key": api_key}
        self.validate_api_call("quotrading", credentials, on_success, on_error)
    
    def setup_broker_screen(self):
        """Screen 0: Broker Connection Setup with QuoTrading API Key and Account Size."""
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        self.current_screen = 0
        self.root.title("QuoTrading - Broker Setup")
        
        # Header
        header = self.create_header("QuoTrading AI", "Select your account type and broker")
        
        # Main container - no scrolling
        main = tk.Frame(self.root, bg=self.colors['background'], padx=10, pady=5)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Card
        card = tk.Frame(main, bg=self.colors['card'], relief=tk.FLAT, bd=0)
        card.pack(fill=tk.BOTH, expand=True)
        card.configure(highlightbackground=self.colors['border'], highlightthickness=2)
        
        # Card content
        content = tk.Frame(card, bg=self.colors['card'], padx=10, pady=2)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Info message
        info = tk.Label(
            content,
            text="Choose your broker type and enter credentials",
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text_light'],
            justify=tk.CENTER
        )
        info.pack(pady=(0, 2))
        
        # Broker Type Selection - Card-style buttons
        type_label = tk.Label(
            content,
            text="Account Type:",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        type_label.pack(pady=(0, 3))
        
        # Container for cards
        cards_container = tk.Frame(content, bg=self.colors['card'])
        cards_container.pack(fill=tk.X, pady=(0, 4))
        
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
                highlightthickness=2,
                highlightbackground=self.colors['border'] if self.broker_type_var.get() == btype else self.colors['text_secondary']
            )
            card_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=3)
            
            # Make card clickable
            def make_select(bt=btype):
                return lambda e: self.select_broker_type(bt)
            
            card_frame.bind("<Button-1>", make_select(btype))
            
            # Card content
            inner = tk.Frame(card_frame, bg=self.colors['secondary'])
            inner.pack(expand=True, fill=tk.BOTH, padx=4, pady=4)
            inner.bind("<Button-1>", make_select(btype))
            
            # Icon
            icon_label = tk.Label(
                inner,
                text=icon,
                font=("Segoe UI", 12),
                bg=self.colors['secondary'],
                fg=self.colors['text']
            )
            icon_label.pack()
            icon_label.bind("<Button-1>", make_select(btype))
            
            # Type name
            type_name = tk.Label(
                inner,
                text=btype,
                font=("Segoe UI", 8, "bold"),
                bg=self.colors['secondary'],
                fg=self.colors['text']
            )
            type_name.pack(pady=(2, 1))
            type_name.bind("<Button-1>", make_select(btype))
            
            # Description
            desc_label = tk.Label(
                inner,
                text=desc,
                font=("Segoe UI", 7),
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
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        broker_label.pack(anchor=tk.W, pady=(4, 2))
        
        self.broker_var = tk.StringVar(value=self.config.get("broker", "TopStep"))
        self.broker_dropdown = ttk.Combobox(
            content,
            textvariable=self.broker_var,
            state="readonly",
            font=("Segoe UI", 9),
            width=35
        )
        self.broker_dropdown.pack(fill=tk.X, pady=(0, 4))
        
        # Update broker options based on selected type
        self.update_broker_options()
        
        # Account Type Selection (dynamic label based on broker)
        self.account_type_label = tk.Label(
            content,
            text="Account Type:",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        self.account_type_label.pack(anchor=tk.W, pady=(8, 3))
        
        # Define TopStep account types with their rules
        self.topstep_account_types = {
            "Trading Combine $50K": {
                "size": "50000",
                "daily_loss": "2000",
                "total_drawdown": "3000",
                "profit_target": "3000",
                "description": "Daily Loss: $2,000 | Total DD: $3,000 | Target: $3,000"
            },
            "Trading Combine $100K": {
                "size": "100000",
                "daily_loss": "4000",
                "total_drawdown": "6000",
                "profit_target": "6000",
                "description": "Daily Loss: $4,000 | Total DD: $6,000 | Target: $6,000"
            },
            "Trading Combine $150K": {
                "size": "150000",
                "daily_loss": "6000",
                "total_drawdown": "9000",
                "profit_target": "9000",
                "description": "Daily Loss: $6,000 | Total DD: $9,000 | Target: $9,000"
            },
            "Trading Combine $250K": {
                "size": "250000",
                "daily_loss": "10000",
                "total_drawdown": "15000",
                "profit_target": "15000",
                "description": "Daily Loss: $10,000 | Total DD: $15,000 | Target: $15,000"
            },
            "Express Funded $50K": {
                "size": "50000",
                "daily_loss": "2000",
                "total_drawdown": "3000",
                "profit_target": "3000",
                "description": "Daily Loss: $2,000 | Total DD: $3,000 | Target: $3,000 (15-day eval)"
            },
            "Express Funded $100K": {
                "size": "100000",
                "daily_loss": "4000",
                "total_drawdown": "6000",
                "profit_target": "6000",
                "description": "Daily Loss: $4,000 | Total DD: $6,000 | Target: $6,000 (15-day eval)"
            },
            "Funded Account $50K": {
                "size": "50000",
                "daily_loss": "2000",
                "total_drawdown": "3000",
                "profit_target": "N/A",
                "description": "Daily Loss: $2,000 | Trailing Threshold: $3,000 | No target"
            },
            "Funded Account $100K": {
                "size": "100000",
                "daily_loss": "4000",
                "total_drawdown": "6000",
                "profit_target": "N/A",
                "description": "Daily Loss: $4,000 | Trailing Threshold: $6,000 | No target"
            },
            "Funded Account $150K": {
                "size": "150000",
                "daily_loss": "6000",
                "total_drawdown": "9000",
                "profit_target": "N/A",
                "description": "Daily Loss: $6,000 | Trailing Threshold: $9,000 | No target"
            }
        }
        
        # Define Tradovate account types
        self.tradovate_account_types = {
            "Live Account $5K": {
                "size": "5000",
                "daily_loss": "500",
                "total_drawdown": "1000",
                "profit_target": "N/A",
                "description": "Personal funded account - $5,000 starting balance"
            },
            "Live Account $10K": {
                "size": "10000",
                "daily_loss": "1000",
                "total_drawdown": "2000",
                "profit_target": "N/A",
                "description": "Personal funded account - $10,000 starting balance"
            },
            "Live Account $25K": {
                "size": "25000",
                "daily_loss": "2500",
                "total_drawdown": "5000",
                "profit_target": "N/A",
                "description": "Personal funded account - $25,000 starting balance"
            },
            "Live Account $50K": {
                "size": "50000",
                "daily_loss": "5000",
                "total_drawdown": "10000",
                "profit_target": "N/A",
                "description": "Personal funded account - $50,000 starting balance"
            },
            "Live Account $100K": {
                "size": "100000",
                "daily_loss": "10000",
                "total_drawdown": "20000",
                "profit_target": "N/A",
                "description": "Personal funded account - $100,000 starting balance"
            },
            "Live Account (Custom)": {
                "size": "10000",
                "daily_loss": "1000",
                "total_drawdown": "2000",
                "profit_target": "N/A",
                "description": "Custom account size - adjust values after selection"
            }
        }
        
        # Get current broker type to determine which account types to show
        broker_type = self.broker_type_var.get()
        if broker_type == "Prop Firm":
            current_account_types = self.topstep_account_types
            default_account_type = "Trading Combine $50K"
        else:  # Live Broker
            current_account_types = self.tradovate_account_types
            default_account_type = "Live Account $10K"
        
        # Get saved account type or default
        saved_account_type = self.config.get("account_type", default_account_type)
        # Ensure saved account type is valid for current broker, fallback to default if not
        if saved_account_type not in current_account_types:
            saved_account_type = default_account_type
        self.account_type_var = tk.StringVar(value=saved_account_type)
        
        # Store current account types for access in other methods
        self.current_account_types = current_account_types
        
        # Create styled dropdown frame
        dropdown_frame = tk.Frame(content, bg=self.colors['border'], bd=0)
        dropdown_frame.pack(fill=tk.X, padx=1, pady=1)
        
        # Create custom style for combobox to make it look better
        # Use unique style name to avoid conflicts with other widgets
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TopStepAccount.TCombobox',
                       fieldbackground=self.colors['input_bg'],
                       background=self.colors['success'],
                       foreground=self.colors['text'],
                       arrowcolor=self.colors['success'],
                       borderwidth=0,
                       relief='flat')
        style.map('TopStepAccount.TCombobox',
                 fieldbackground=[('readonly', self.colors['input_bg'])],
                 selectbackground=[('readonly', self.colors['input_focus'])],
                 selectforeground=[('readonly', self.colors['text'])])
        
        self.account_type_dropdown = ttk.Combobox(
            dropdown_frame,
            textvariable=self.account_type_var,
            state="readonly",
            font=("Segoe UI", 9),
            style='TopStepAccount.TCombobox',
            values=list(current_account_types.keys())
        )
        self.account_type_dropdown.pack(fill=tk.X, ipady=3, padx=2, pady=2)
        self.account_type_dropdown.bind("<<ComboboxSelected>>", self.update_account_type_info)
        
        # Account type info display
        self.account_info_display = tk.Label(
            content,
            text=current_account_types[saved_account_type]["description"],
            font=("Segoe UI", 7),
            bg=self.colors['card'],
            fg=self.colors['text_light'],
            wraplength=500,
            justify=tk.LEFT
        )
        self.account_info_display.pack(anchor=tk.W, pady=(2, 4))
        
        # Broker Username
        self.broker_username_entry = self.create_input_field(
            content,
            "Username/Email:",
            placeholder=""
        )
        # Load saved username if exists
        saved_username = self.config.get("broker_username", "")
        if saved_username:
            self.broker_username_entry.insert(0, saved_username)
        
        # Broker API Token
        self.broker_token_entry = self.create_input_field(
            content,
            "Broker API Key:",
            is_password=True,
            placeholder=""
        )
        # Load saved token if exists
        saved_token = self.config.get("broker_token", "")
        if saved_token:
            self.broker_token_entry.insert(0, saved_token)
        
        # QuoTrading API Key
        self.quotrading_api_key_entry = self.create_input_field(
            content,
            "QuoTrading API Key:",
            is_password=True,
            placeholder=""
        )
        # Load saved QuoTrading key if exists
        saved_quotrading_key = self.config.get("quotrading_api_key", "")
        if saved_quotrading_key:
            self.quotrading_api_key_entry.insert(0, saved_quotrading_key)
        
        # Remember credentials checkbox
        remember_frame = tk.Frame(content, bg=self.colors['card'])
        remember_frame.pack(fill=tk.X, pady=(6, 10))
        
        self.remember_credentials_var = tk.BooleanVar(value=self.config.get("remember_credentials", True))
        remember_cb = tk.Checkbutton(
            remember_frame,
            text="Save credentials",
            variable=self.remember_credentials_var,
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text'],
            selectcolor=self.colors['secondary'],
            activebackground=self.colors['card'],
            activeforeground=self.colors['success'],
            cursor="hand2"
        )
        remember_cb.pack(side=tk.LEFT, anchor=tk.W)
        
        # Login button on the RIGHT side of the same row
        login_btn = tk.Button(
            remember_frame,
            text="LOGIN",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['success_dark'],
            fg='white',
            activebackground=self.colors['button_hover'],
            activeforeground='white',
            relief=tk.FLAT,
            bd=0,
            command=self.validate_broker,
            cursor="hand2",
            width=16,
            height=1
        )
        login_btn.pack(side=tk.RIGHT, padx=5, ipady=6)
    
    def select_broker_type(self, broker_type):
        """Select broker type and update UI."""
        self.broker_type_var.set(broker_type)
        
        # Update card styling
        for btype, card in self.broker_cards.items():
            if btype == broker_type:
                card.config(highlightbackground=self.colors['border'], highlightthickness=2)
            else:
                card.config(highlightbackground=self.colors['text_secondary'], highlightthickness=2)
        
        # Update broker dropdown
        self.update_broker_options()
        
        # Update account types based on broker type
        if broker_type == "Prop Firm":
            self.current_account_types = self.topstep_account_types
            default_account_type = "Trading Combine $50K"
            self.account_type_label.config(text="TopStep Account Type:")
        else:  # Live Broker
            self.current_account_types = self.tradovate_account_types
            default_account_type = "Live Account $10K"
            self.account_type_label.config(text="Tradovate Account Type:")
        
        # Update dropdown values
        self.account_type_dropdown['values'] = list(self.current_account_types.keys())
        
        # Set default account type for this broker
        self.account_type_var.set(default_account_type)
        
        # Update info display
        self.update_account_type_info()
    
    def update_account_type_info(self, event=None):
        """Update the account type info display when selection changes."""
        selected_type = self.account_type_var.get()
        if selected_type in self.current_account_types:
            account_info = self.current_account_types[selected_type]
            self.account_info_display.config(text=account_info["description"])
    
    def update_broker_options(self):
        """Update broker dropdown based on account type."""
        broker_type = self.broker_type_var.get()
        
        if broker_type == "Prop Firm":
            options = ["TopStep"]
        else:  # Live Broker
            options = ["Tradovate"]
        
        self.broker_dropdown['values'] = options
        self.broker_dropdown.current(0)
    
    def validate_broker(self):
        """Validate broker credentials and QuoTrading API key before proceeding."""
        broker = self.broker_var.get()
        token = self.broker_token_entry.get().strip()
        username = self.broker_username_entry.get().strip()
        quotrading_api_key = self.quotrading_api_key_entry.get().strip()
        
        # Get selected account type (works for both TopStep and Tradovate)
        account_type = self.account_type_var.get()
        if account_type in self.current_account_types:
            account_info = self.current_account_types[account_type]
            account_size = account_info["size"]
        else:
            account_size = "10000"  # Default fallback
        
        # Check if using admin key FIRST - bypass all validation
        if quotrading_api_key == "QUOTRADING_ADMIN_MASTER_2025":
            self.config["broker_type"] = self.broker_type_var.get()
            self.config["broker"] = broker
            self.config["account_type"] = account_type
            self.config["account_size"] = account_size
            self.config["broker_validated"] = True
            
            # Only save credentials if "Remember" is checked
            if self.remember_credentials_var.get():
                self.config["broker_token"] = token
                self.config["broker_username"] = username if username else "admin"
                self.config["quotrading_api_key"] = quotrading_api_key
                self.config["remember_credentials"] = True
            else:
                # Clear saved credentials
                self.config["broker_token"] = ""
                self.config["broker_username"] = ""
                self.config["quotrading_api_key"] = ""
                self.config["remember_credentials"] = False
            
            self.save_config()
            self.setup_trading_screen()
            return
        
        # Remove placeholder if present
        if quotrading_api_key == self.config.get("quotrading_api_key", ""):
            quotrading_api_key = ""
        
        # Validation
        if not token or not username:
            messagebox.showerror(
                "Missing Credentials",
                f"Please enter both {broker} API Token and Username."
            )
            return
        
        if not quotrading_api_key:
            messagebox.showerror(
                "Missing API Key",
                "Please enter your QuoTrading API Key."
            )
            return
        
        # Show loading spinner
        self.show_loading(f"Validating {broker} credentials...")
        
        # Define success callback
        def on_success():
            self.hide_loading()
            # Only save credentials if "Remember" is checked
            self.config["broker_type"] = self.broker_type_var.get()
            self.config["broker"] = broker
            self.config["account_type"] = account_type
            self.config["account_size"] = account_size
            self.config["broker_validated"] = True
            
            if self.remember_credentials_var.get():
                self.config["broker_token"] = token
                self.config["broker_username"] = username
                self.config["quotrading_api_key"] = quotrading_api_key
                self.config["remember_credentials"] = True
            else:
                # Clear saved credentials
                self.config["broker_token"] = ""
                self.config["broker_username"] = ""
                self.config["quotrading_api_key"] = ""
                self.config["remember_credentials"] = False
            
            self.save_config()
            # Proceed to trading preferences
            self.setup_trading_screen()
        
        # Define error callback
        def on_error(error_msg):
            self.hide_loading()
            messagebox.showerror(
                "Validation Failed",
                f"❌ {error_msg}\n\nPlease check your {broker} credentials and try again.\n\n"
                f"Make sure you have:\n"
                f"• Valid API token from your {broker} account\n"
                f"• Correct username/email\n"
                f"• Active account status"
            )
        
        # Make API validation call
        credentials = {
            "broker": broker,
            "token": token,
            "username": username
        }
        self.validate_api_call("broker", credentials, on_success, on_error)
    
    
    def setup_trading_screen(self):
        """Screen 1: Trading Controls and Launch."""
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        self.current_screen = 1
        self.root.title("QuoTrading - Trading Controls")
        
        # Header
        header = self.create_header("Trading Controls", "Configure your trading strategy")
        
        # Main container - NO SCROLLBAR
        main = tk.Frame(self.root, bg=self.colors['background'], padx=10, pady=5)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Card
        card = tk.Frame(main, bg=self.colors['card'], relief=tk.FLAT, bd=0)
        card.pack(fill=tk.BOTH, expand=True)
        card.configure(highlightbackground=self.colors['border'], highlightthickness=2)
        
        # Card content - REDUCED PADDING
        content = tk.Frame(card, bg=self.colors['card'], padx=10, pady=4)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Account Fetch Section - COMPACT HORIZONTAL STYLE
        fetch_frame = tk.Frame(content, bg=self.colors['card'])
        fetch_frame.pack(fill=tk.X, pady=(0, 4))
        
        # Left side: Account dropdown
        account_select_frame = tk.Frame(fetch_frame, bg=self.colors['card'])
        account_select_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        tk.Label(
            account_select_frame,
            text="Select Account:",
            font=("Segoe UI", 7),
            bg=self.colors['card'],
            fg=self.colors['text_light']
        ).pack(anchor=tk.W)
        
        self.account_dropdown_var = tk.StringVar(value="Click 'Fetch Account Info' to load accounts")
        self.account_dropdown = ttk.Combobox(
            account_select_frame,
            textvariable=self.account_dropdown_var,
            state="readonly",
            font=("Segoe UI", 7),
            width=20,
            values=["Click 'Fetch Account Info' to load accounts"]
        )
        self.account_dropdown.pack(fill=tk.X)
        
        # Middle: Fetch button with label
        fetch_button_frame = tk.Frame(fetch_frame, bg=self.colors['card'])
        fetch_button_frame.pack(side=tk.LEFT, padx=5)
        
        important_label = tk.Label(
            fetch_button_frame,
            text="⚠️ IMPORTANT:",
            font=("Segoe UI", 7, "bold"),
            bg=self.colors['card'],
            fg=self.colors['error']
        )
        important_label.pack(anchor=tk.W)
        
        fetch_btn = self.create_button(fetch_button_frame, "Fetch Account Info", self.fetch_account_info, "next")
        fetch_btn.pack()
        
        # Right: Auto-adjust button
        auto_adjust_frame = tk.Frame(fetch_frame, bg=self.colors['card'])
        auto_adjust_frame.pack(side=tk.LEFT, padx=5)
        
        tk.Label(
            auto_adjust_frame,
            text="Quick Setup:",
            font=("Segoe UI", 7),
            bg=self.colors['card'],
            fg=self.colors['text_light']
        ).pack(anchor=tk.W)
        
        auto_adjust_btn = self.create_button(auto_adjust_frame, "Auto Configure", self.auto_adjust_parameters, "next")
        auto_adjust_btn.pack()
        
        # Info label below everything
        self.account_info_label = tk.Label(
            content,
            text="Fetching helps AI determine account type and provide best settings",
            font=("Segoe UI", 6),
            bg=self.colors['card'],
            fg=self.colors['text_secondary'],
            anchor=tk.W
        )
        self.account_info_label.pack(anchor=tk.W, pady=(0, 4))
        
        self.auto_adjust_info_label = self.account_info_label  # Reuse same label
        
        # Symbol Selection
        symbol_label = tk.Label(
            content,
            text="Trading Symbols (select at least one):",
            font=("Segoe UI", 8, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        symbol_label.pack(anchor=tk.W, pady=(0, 2))
        
        # Symbol checkboxes - 2 columns
        symbol_frame = tk.Frame(content, bg=self.colors['card'])
        symbol_frame.pack(fill=tk.X, pady=(0, 4))
        
        self.symbol_vars = {}
        symbols = [
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
                font=("Segoe UI", 7),
                bg=self.colors['card'],
                fg=self.colors['text'],
                selectcolor=self.colors['secondary'],
                activebackground=self.colors['card'],
                activeforeground=self.colors['success'],
                cursor="hand2"
            )
            cb.grid(row=row, column=col, sticky=tk.W, padx=4, pady=0)
        
        # ========================================
        # TRADING MODES - Next to Symbols
        # ========================================
        modes_section = tk.Frame(content, bg=self.colors['card'])
        modes_section.pack(fill=tk.X, pady=(6, 3))
        
        tk.Label(
            modes_section,
            text="Trading Modes:",
            font=("Segoe UI", 8, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 2))
        
        modes_row = tk.Frame(modes_section, bg=self.colors['card'])
        modes_row.pack(fill=tk.X)
        
        # Confidence Trading
        conf_mode_frame = tk.Frame(modes_row, bg=self.colors['card'])
        conf_mode_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.confidence_trading_var = tk.BooleanVar(value=self.config.get("confidence_trading", False))
        
        def on_confidence_trading_toggle():
            """Disable Recovery Mode if Confidence Trading is enabled."""
            if self.confidence_trading_var.get():
                # Confidence Trading enabled - disable Recovery Mode
                if self.recovery_mode_var.get():
                    messagebox.showinfo(
                        "Mode Conflict",
                        "Confidence Trading and Recovery Mode cannot be enabled at the same time.\n\n"
                        "Recovery Mode has been automatically disabled."
                    )
                    self.recovery_mode_var.set(False)
        
        tk.Checkbutton(
            conf_mode_frame,
            text="⚖️ Confidence Trading",
            variable=self.confidence_trading_var,
            command=on_confidence_trading_toggle,
            font=("Segoe UI", 7, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text'],
            selectcolor=self.colors['secondary'],
            activebackground=self.colors['card'],
            activeforeground=self.colors['success'],
            cursor="hand2"
        ).pack(anchor=tk.W)
        
        tk.Label(
            conf_mode_frame,
            text="Smart contract sizing based on account risk level",
            font=("Segoe UI", 6),
            bg=self.colors['card'],
            fg=self.colors['text_secondary']
        ).pack(anchor=tk.W)
        
        # Shadow Mode
        shadow_mode_frame = tk.Frame(modes_row, bg=self.colors['card'])
        shadow_mode_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.shadow_mode_var = tk.BooleanVar(value=self.config.get("shadow_mode", False))
        tk.Checkbutton(
            shadow_mode_frame,
            text="👁️ Shadow Mode",
            variable=self.shadow_mode_var,
            font=("Segoe UI", 7, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text'],
            selectcolor=self.colors['secondary'],
            activebackground=self.colors['card'],
            activeforeground=self.colors['success'],
            cursor="hand2"
        ).pack(anchor=tk.W)
        
        tk.Label(
            shadow_mode_frame,
            text="Watch signals without executing",
            font=("Segoe UI", 6),
            bg=self.colors['card'],
            fg=self.colors['text_secondary']
        ).pack(anchor=tk.W)
        
        # Recovery Mode
        recovery_mode_frame = tk.Frame(modes_row, bg=self.colors['card'])
        recovery_mode_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.recovery_mode_var = tk.BooleanVar(value=self.config.get("recovery_mode", False))
        
        def on_recovery_mode_toggle():
            """Show warning when enabling Recovery Mode."""
            if self.recovery_mode_var.get():
                # Check if Confidence Trading is enabled
                if self.confidence_trading_var.get():
                    messagebox.showinfo(
                        "Mode Conflict",
                        "Confidence Trading and Recovery Mode cannot be enabled at the same time.\n\n"
                        "Confidence Trading has been automatically disabled."
                    )
                    self.confidence_trading_var.set(False)
                
                # User is trying to enable it - show warning
                warning_msg = (
                    "⚠️ RECOVERY MODE WARNING ⚠️\n\n"
                    "Recovery Mode is designed for advanced traders who understand the risks.\n\n"
                    "What Recovery Mode Does:\n"
                    "• Automatically reduces contract size when your account is losing money\n"
                    "• Continues trading even AFTER hitting your daily loss limit\n"
                    "• Attempts to recover losses by taking smaller positions\n\n"
                    "IMPORTANT RISKS:\n"
                    "• Can exceed your daily loss limit significantly\n"
                    "• May violate broker/prop firm rules that require stopping at daily limit\n"
                    "• Could result in account termination or larger losses\n"
                    "• Not recommended for funded/prop accounts with strict rules\n\n"
                    "⚠️ Use Recovery Mode at your own risk ⚠️\n\n"
                    "Do you want to enable Recovery Mode?"
                )
                
                response = messagebox.askyesno(
                    "Recovery Mode Warning",
                    warning_msg,
                    icon='warning'
                )
                
                if not response:
                    # User clicked No - disable it
                    self.recovery_mode_var.set(False)
        
        recovery_checkbox = tk.Checkbutton(
            recovery_mode_frame,
            text="🔄 Recovery Mode",
            variable=self.recovery_mode_var,
            command=on_recovery_mode_toggle,
            font=("Segoe UI", 7, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text'],
            selectcolor=self.colors['secondary'],
            activebackground=self.colors['card'],
            activeforeground=self.colors['success'],
            cursor="hand2"
        )
        recovery_checkbox.pack(anchor=tk.W)
        
        tk.Label(
            recovery_mode_frame,
            text="Reduces contract size when losing, keeps trading past daily limit",
            font=("Segoe UI", 6),
            bg=self.colors['card'],
            fg=self.colors['text_secondary']
        ).pack(anchor=tk.W)
        
        # Account Settings Row - COMPACT
        settings_row = tk.Frame(content, bg=self.colors['card'])
        settings_row.pack(fill=tk.X, pady=(0, 3))
        
        # Account Size
        acc_frame = tk.Frame(settings_row, bg=self.colors['card'])
        acc_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        
        tk.Label(
            acc_frame,
            text="Account Size ($):",
            font=("Segoe UI", 7, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 1))
        
        self.account_entry = tk.Entry(
            acc_frame,
            font=("Segoe UI", 7),
            bg=self.colors['input_bg'],
            fg=self.colors['text'],
            insertbackground=self.colors['success'],
            relief=tk.FLAT,
            bd=0,
            highlightthickness=2,
            highlightbackground=self.colors['border'],
            highlightcolor=self.colors['success']
        )
        self.account_entry.pack(fill=tk.X, ipady=2, padx=2)
        self.account_entry.insert(0, self.config.get("account_size", "10000"))
        
        # Daily Loss Limit
        loss_frame = tk.Frame(settings_row, bg=self.colors['card'])
        loss_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(
            loss_frame,
            text="Daily Loss Limit ($):",
            font=("Segoe UI", 7, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 1))
        
        self.loss_entry = tk.Entry(
            loss_frame,
            font=("Segoe UI", 7),
            bg=self.colors['input_bg'],
            fg=self.colors['text'],
            insertbackground=self.colors['success'],
            relief=tk.FLAT,
            bd=0,
            highlightthickness=2,
            highlightbackground=self.colors['border'],
            highlightcolor=self.colors['success']
        )
        self.loss_entry.pack(fill=tk.X, ipady=2, padx=2)
        self.loss_entry.insert(0, self.config.get("daily_loss_limit", "2000"))
        
        # Advanced Settings Row - COMPACT
        advanced_row = tk.Frame(content, bg=self.colors['card'])
        advanced_row.pack(fill=tk.X, pady=(0, 3))
        
        # Max Contracts with account type awareness and enforcement
        contracts_frame = tk.Frame(advanced_row, bg=self.colors['card'])
        contracts_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        
        tk.Label(
            contracts_frame,
            text="Contracts Per Trade:",
            font=("Segoe UI", 7, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 1))
        
        # Get account type to set ENFORCED limits
        account_type = self.config.get("broker_type", "Prop Firm")
        
        # Enforce actual broker limits - exceeding these will cause trade rejection
        if account_type == "Prop Firm":
            max_contracts_allowed = 5  # Prop firm strict limit
        else:
            max_contracts_allowed = 25  # Live broker flexible limit
        
        # Store for validation
        self.max_contracts_allowed = max_contracts_allowed
        self.account_type = account_type
        
        self.contracts_var = tk.IntVar(value=min(self.config.get("max_contracts", 3), max_contracts_allowed))
        
        contracts_spin = ttk.Spinbox(
            contracts_frame,
            from_=1,
            to=max_contracts_allowed,  # Enforced based on account type
            textvariable=self.contracts_var,
            width=12
        )
        contracts_spin.pack(fill=tk.X, ipady=2)
        
        # Info label showing enforced contract limit
        contracts_info = tk.Label(
            contracts_frame,
            text=f"Max {max_contracts_allowed} for {account_type} (enforced)",
            font=("Segoe UI", 6),
            bg=self.colors['card'],
            fg=self.colors['text_secondary']
        )
        contracts_info.pack(anchor=tk.W, pady=(1, 0))
        
        # Max Trades Per Day
        trades_frame = tk.Frame(advanced_row, bg=self.colors['card'])
        trades_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(
            trades_frame,
            text="Max Trades/Day:",
            font=("Segoe UI", 7, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 1))
        
        self.trades_var = tk.IntVar(value=self.config.get("max_trades", 10))
        trades_spin = ttk.Spinbox(
            trades_frame,
            from_=1,
            to=50,
            textvariable=self.trades_var,
            width=12
        )
        trades_spin.pack(fill=tk.X, ipady=2)
        
        # ========================================
        # CONFIDENCE SLIDER - COMPACT
        # ========================================
        confidence_section = tk.Frame(content, bg=self.colors['card'])
        confidence_section.pack(fill=tk.X, pady=(2, 1))
        
        # Header with title
        conf_header = tk.Frame(confidence_section, bg=self.colors['card'])
        conf_header.pack(fill=tk.X, pady=(0, 1))
        
        # Title and description on same line
        tk.Label(
            conf_header,
            text="AI CONFIDENCE THRESHOLD",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(side=tk.LEFT)
        
        tk.Label(
            conf_header,
            text="  •  Higher = Fewer trades, safer  •  Lower = More trades, riskier",
            font=("Segoe UI", 7),
            bg=self.colors['card'],
            fg=self.colors['text_secondary']
        ).pack(side=tk.LEFT, padx=(4, 0))
        
        # Current value display with trading style
        self.confidence_var = tk.DoubleVar(value=self.config.get("confidence_threshold", 70.0))
        
        self.confidence_style_label = tk.Label(
            conf_header,
            text=self.get_trading_style(self.confidence_var.get()),
            font=("Segoe UI", 7, "bold"),
            bg=self.colors['card'],
            fg=self.get_style_color(self.confidence_var.get())
        )
        self.confidence_style_label.pack(side=tk.RIGHT, padx=(0, 8))
        
        self.confidence_display = tk.Label(
            conf_header,
            text=f"{self.confidence_var.get():.0f}%",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['card'],
            fg=self.get_style_color(self.confidence_var.get())
        )
        self.confidence_display.pack(side=tk.RIGHT)
        
        # Slider container - simple and clean
        slider_container = tk.Frame(confidence_section, bg=self.colors['card'])
        slider_container.pack(fill=tk.X, padx=15, pady=(3, 0))
        
        # Create slider with dynamic color
        self.confidence_slider = tk.Scale(
            slider_container,
            from_=10,
            to=100,
            resolution=5,
            variable=self.confidence_var,
            orient=tk.HORIZONTAL,
            bg=self.get_style_color(self.confidence_var.get()),
            fg=self.colors['text'],
            activebackground=self.get_style_color(self.confidence_var.get()),
            troughcolor=self.get_style_color(self.confidence_var.get()),
            highlightthickness=0,
            bd=0,
            length=500,
            showvalue=0,
            sliderlength=25,
            width=15,
            command=self.update_confidence_display
        )
        self.confidence_slider.pack(fill=tk.X)
        
        # Trade details below slider - with emojis and colored text
        trade_details = tk.Frame(confidence_section, bg=self.colors['card'])
        trade_details.pack(fill=tk.X, padx=15, pady=(0, 1))
        
        # Create frame for trade information
        self.trade_info_frame = tk.Frame(trade_details, bg=self.colors['card'])
        self.trade_info_frame.pack()
        
        # Initialize trade info display
        self.update_trade_info(self.confidence_var.get())
        
        # Summary display - COMPACT
        summary_frame = tk.Frame(content, bg=self.colors['card'])
        summary_frame.pack(fill=tk.X, pady=(0, 1))
        
        # Center container for launch button
        launch_container = tk.Frame(summary_frame, bg=self.colors['card'])
        launch_container.pack()
        
        launch_btn = self.create_button(launch_container, "LAUNCH AI", self.start_bot, "next")
        launch_btn.pack(pady=2, ipady=3)
    
    def _update_account_size_from_fetched(self, balance: float):
        """Helper method to update account size field with fetched balance."""
        self.config["account_size"] = str(int(balance))
        self.account_entry.delete(0, tk.END)
        self.account_entry.insert(0, str(int(balance)))
        self.save_config()
    
    def fetch_account_info(self):
        """Fetch account information from the broker using actual API."""
        print("\n[DEBUG] Fetch Account Info button clicked!")
        
        broker = self.config.get("broker", "TopStep")
        token = self.config.get("broker_token", "")
        username = self.config.get("broker_username", "")
        
        print(f"[DEBUG] Broker: {broker}")
        print(f"[DEBUG] Token exists: {bool(token)}")
        print(f"[DEBUG] Username: {username}")
        
        if not token or not username:
            print("[DEBUG] Missing credentials - showing error")
            messagebox.showerror(
                "Missing Credentials",
                "Please enter your broker credentials on the first screen."
            )
            return
        
        print(f"[DEBUG] Starting REAL API fetch for {broker}...")
        
        # Show loading spinner
        self.show_loading(f"Connecting to {broker} API...")
        
        def fetch_in_thread():
            print("[DEBUG] Inside fetch thread...")
            import traceback  # Import at top of thread function
            try:
                # Import broker interface (SDK already installed with AI)
                import sys
                from pathlib import Path
                src_path = Path(__file__).parent.parent / "src"
                if str(src_path) not in sys.path:
                    sys.path.insert(0, str(src_path))
                
                accounts = []
                
                # REAL API CALL ONLY - NO DEMO DATA
                if broker == "TopStep":
                    print("[DEBUG] Connecting to TopStep API...")
                    from broker_interface import TopStepBroker
                    
                    # Create broker instance with user's REAL credentials
                    print(f"[DEBUG] Creating TopStepBroker instance...")
                    ts_broker = TopStepBroker(api_token=token, username=username)
                    
                    # Connect to broker API
                    print("[DEBUG] Calling connect()...")
                    connected = ts_broker.connect()
                    
                    if connected:
                        print("[DEBUG] Connected successfully! Getting account balance...")
                        # Get actual account balance from TopStep
                        current_equity = ts_broker.get_account_equity()
                        print(f"[DEBUG] Equity retrieved: ${current_equity:,.2f}")
                        
                        # Determine starting balance vs current equity
                        # If this is first fetch, current equity IS the starting balance
                        # If subsequent fetch, use stored starting balance
                        stored_starting_balance = self.config.get("topstep_starting_balance")
                        
                        if stored_starting_balance:
                            # Subsequent fetch - user has profits/losses
                            starting_balance = stored_starting_balance
                            equity = current_equity
                            print(f"[DEBUG] Starting balance: ${starting_balance:,.2f}, Current equity: ${equity:,.2f}, P&L: ${equity - starting_balance:,.2f}")
                        else:
                            # First fetch - store this as starting balance
                            starting_balance = current_equity
                            equity = current_equity
                            self.config["topstep_starting_balance"] = starting_balance
                            self.save_config()
                            print(f"[DEBUG] First fetch - stored starting balance: ${starting_balance:,.2f}")
                        
                        # Create account info from REAL data
                        accounts = [{
                            "id": "TOPSTEP_MAIN",
                            "name": f"TopStep Account ({username})",
                            "balance": starting_balance,  # Starting balance (what prop firm gave)
                            "equity": equity,              # Current equity (balance + profits)
                            "type": "prop_firm"
                        }]
                        
                        # Disconnect
                        ts_broker.disconnect()
                        print("[DEBUG] Disconnected from TopStep")
                    else:
                        raise Exception("Failed to connect to TopStep API. Check your API token and username.")
                
                elif broker == "Tradovate":
                    print("[DEBUG] Tradovate not yet supported...")
                    raise Exception("Tradovate API integration coming soon. Please use TopStep for now.")
                
                else:
                    raise Exception(f"Unsupported broker: {broker}")
                
                # If no accounts fetched, raise error
                if not accounts:
                    raise Exception("No accounts retrieved from broker API")
                
                # Update UI on main thread
                def update_ui():
                    self.hide_loading()
                    
                    # Update account dropdown with fetched accounts
                    account_names = [f"{acc['name']} - ${acc['balance']:,.2f}" for acc in accounts]
                    self.account_dropdown['values'] = account_names
                    self.account_dropdown.current(0)
                    
                    # Store accounts data
                    self.config["accounts"] = accounts
                    self.config["selected_account"] = account_names[0]
                    self.config["fetched_account_balance"] = accounts[0]['balance']
                    self.config["fetched_account_type"] = accounts[0].get('type', 'live_broker')
                    self.save_config()
                    
                    # Check for mismatch between user input and fetched data
                    user_account_size = float(self.config.get("account_size", "0"))
                    fetched_balance = accounts[0]['balance']
                    
                    mismatch_warning = ""
                    if user_account_size > 0 and abs(user_account_size - fetched_balance) > self.ACCOUNT_MISMATCH_THRESHOLD:
                        mismatch_warning = (
                            f"\n\n⚠️ MISMATCH DETECTED:\n"
                            f"You entered: ${user_account_size:,.2f}\n"
                            f"Fetched account: ${fetched_balance:,.2f}\n\n"
                            f"Using fetched account data (more accurate)."
                        )
                        # Update account size to fetched value using helper method
                        self._update_account_size_from_fetched(fetched_balance)
                    
                    # Update info label
                    selected_acc = accounts[0]
                    self.account_info_label.config(
                        text=f"✓ Balance: ${selected_acc['balance']:,.2f} | Equity: ${selected_acc['equity']:,.2f} | Type: {selected_acc.get('type', 'Unknown')}",
                        fg=self.colors['success']
                    )
                    
                    messagebox.showinfo(
                        "Account Info Fetched",
                        f"Successfully retrieved account from {broker}.\n\n"
                        f"Selected: {selected_acc['name']}\n"
                        f"Balance: ${selected_acc['balance']:,.2f}\n"
                        f"Equity: ${selected_acc['equity']:,.2f}\n"
                        f"Type: {selected_acc.get('type', 'Unknown')}"
                        + (mismatch_warning if mismatch_warning else "")
                    )
                
                self.root.after(0, update_ui)
                
            except Exception as error:
                # Capture error in outer scope
                error_msg = str(error)
                error_traceback = traceback.format_exc()
                
                def show_error():
                    self.hide_loading()
                    
                    # Show user-friendly error
                    messagebox.showerror(
                        "Fetch Failed",
                        f"Failed to fetch account information:\n\n{error_msg}\n\n"
                        f"Please check:\n"
                        f"• API Token is valid and not expired\n"
                        f"• Username/Email matches your account\n"
                        f"• You have an active {broker} account\n\n"
                        f"Contact support if the issue persists."
                    )
                    
                    # Log to console for debugging
                    print(f"\n{'='*60}")
                    print(f"FETCH ACCOUNT ERROR - {broker}")
                    print(f"{'='*60}")
                    print(f"Token: {token[:15]}... (hidden)")
                    print(f"Username: {username}")
                    print(f"Error: {error_msg}")
                    print(f"\nFull traceback:")
                    print(error_traceback)
                    print(f"{'='*60}\n")
                    
                self.root.after(0, show_error)
        
        # Start fetch in background thread
        thread = threading.Thread(target=fetch_in_thread, daemon=True)
        thread.start()
    
    def auto_adjust_parameters(self):
        """Auto-adjust trading parameters based on FETCHED account data (prioritized over user input)."""
        # IMPORTANT: Prioritize fetched account data over user input
        # Check if account info has been fetched
        accounts = self.config.get("accounts", [])
        
        if not accounts:
            messagebox.showwarning(
                "No Account Info",
                "⚠️ IMPORTANT: Please fetch account information first using the 'Fetch Account Info' button.\n\n"
                "Fetching helps the bot:\n"
                "• Detect your account type (prop firm vs live broker)\n"
                "• Determine accurate account size\n"
                "• Provide optimal risk settings\n"
                "• Apply broker-specific rules"
            )
            return
        
        # Use SELECTED account from dropdown (user can have multiple accounts)
        selected_account_name = self.account_dropdown_var.get()
        
        # Find the matching account from fetched accounts list
        selected_account = None
        for acc in accounts:
            acc_display_name = f"{acc['name']} - ${acc['balance']:,.2f}"
            if acc_display_name == selected_account_name:
                selected_account = acc
                break
        
        # Fallback: If no match, use first account
        if not selected_account:
            selected_account = accounts[0]
            print(f"[WARNING] Selected account '{selected_account_name}' not found, using first account")
        
        # Extract balance, equity, and type from SELECTED account
        balance = selected_account.get("balance", 10000)       # Starting balance
        equity = selected_account.get("equity", balance)       # Current equity (with profits)
        account_type = selected_account.get("type", "live_broker")
        
        # Update account size with fetched data (overrides user input) using helper method
        self._update_account_size_from_fetched(balance)
        
        # Calculate drawdown percentage (how far from starting balance)
        drawdown_pct = ((balance - equity) / balance) * 100 if balance > 0 else 0
        drawdown_pct = max(0, drawdown_pct)  # Ensure non-negative
        
        # Sophisticated risk management based on ACCOUNT TYPE (from fetched data)
        if account_type == "prop_firm":
            # Prop firm rules: Be more conservative as drawdown increases
            # Most prop firms fail traders at 8-10% drawdown
            
            # Calculate distance to failure using configured max drawdown
            max_dd = self.PROP_FIRM_MAX_DRAWDOWN
            distance_to_failure = max_dd - drawdown_pct
            
            # Daily loss limit: Scale based on distance to failure
            # Use 2% rule as baseline for prop firms
            if distance_to_failure > 6:  # Safe zone (0-2% drawdown)
                daily_loss_pct = 0.02  # 2% of equity (standard prop firm rule)
            elif distance_to_failure > 4:  # Caution zone (2-4% drawdown)
                daily_loss_pct = 0.015   # 1.5% of equity
            elif distance_to_failure > 2:  # Warning zone (4-6% drawdown)
                daily_loss_pct = 0.01  # 1% of equity
            else:  # Danger zone (6-8% drawdown)
                daily_loss_pct = 0.005   # 0.5% of equity - very conservative
            
            daily_loss_limit = equity * daily_loss_pct
            
            # Contracts: Smart sizing based on DOLLAR BUFFER to failure
            # CRITICAL: Failure threshold is based on INITIAL/STARTING BALANCE, not current equity
            # Example: $50k starting account fails at $48k (96% of initial), even if equity is now $54k
            # 
            # Fresh $50k account: buffer = $50,000 - $48,000 = $2,000
            # Same account at $54k equity: buffer = $54,000 - $48,000 = $6,000 (more room with profits!)
            # Same account at $49k equity: buffer = $49,000 - $48,000 = $1,000 (danger!)
            #
            # IMPORTANT: Must account for:
            # 1. Risk per contract (~$300 avg on ES: slippage + spread + volatility)
            # 2. Trailing drawdown (profits can evaporate quickly)
            # 3. Multiple losing trades (need buffer for at least 6 bad trades)
            # 4. Daily loss limits (can't risk entire buffer in one day)
            
            failure_threshold = balance * 0.96  # Prop firms fail at ~4% drawdown from STARTING balance
            buffer_to_failure = equity - failure_threshold  # How far current equity is from failure
            
            # Risk per contract accounting for:
            # - Average loss per contract: $250 (4-5 ticks on ES)
            # - Slippage: $25 per contract
            # - Spread costs: $25 per contract  
            # - Volatility buffer: $50 (bad fills, gaps)
            # Total conservative estimate: $350 per contract
            risk_per_contract = 350
            
            # Safety factor: Want to survive at least 6 consecutive losing trades
            # This accounts for bad days and protects trailing drawdown
            safe_losing_trades = 6
            
            # Calculate max contracts based on buffer safety
            # Example: $6k buffer / ($350 × 6 trades) = 2.85 → 2 contracts
            max_safe_contracts = max(1, int(buffer_to_failure / (risk_per_contract * safe_losing_trades)))
            
            # Apply TopStep tier limits (can't exceed these even if math allows)
            tier_limit = 5  # Default for < $50k
            for tier_balance, limit in sorted(self.TOPSTEP_CONTRACT_LIMITS.items()):
                if balance >= tier_balance * 0.9:
                    tier_limit = limit
            
            # FINAL CONTRACT SIZING: Take minimum of calculated safe amount and tier limit
            # This respects both mathematical safety AND broker rules
            # Also accounts for trailing drawdown protection
            
            # Adjust for current drawdown severity (protects trailing profits)
            if drawdown_pct < 2:  # Very safe (fresh account or profits) - use full calculated amount
                base_contracts = min(max_safe_contracts, tier_limit)
            elif drawdown_pct < 5:  # Moderately safe - reduce by 40% (trailing drawdown risk)
                base_contracts = max(1, min(int(max_safe_contracts * 0.6), tier_limit))
            else:  # In drawdown - very conservative
                base_contracts = 1  # Single contract only
            
            # Additional safety: Don't let contracts exceed what daily loss limit can handle
            # If daily loss limit is $1000 and risk is $350/contract, max 2 contracts for one bad day
            # But allow multiple bad trades, so divide by 3 (assume 3 losing trades per day max)
            max_daily_risk_contracts = max(1, int(daily_loss_limit / (risk_per_contract * 3)))
            
            # Final decision: Take minimum of all constraints
            max_contracts = min(base_contracts, max_daily_risk_contracts, tier_limit)
            
            # Trades per day: Calculate based on contracts × trades interaction
            # Total daily risk = max_contracts × max_trades × risk_per_contract
            # Should not exceed daily loss limit
            # Formula: max_trades = daily_loss_limit / (max_contracts × risk_per_contract)
            
            safe_max_trades = max(3, int(daily_loss_limit / (max_contracts * risk_per_contract)))
            
            # Cap based on distance to failure
            if distance_to_failure > 5:  # Safe zone
                max_trades = min(safe_max_trades, 12)  # Never more than 12
            else:  # Near limits - be very selective
                max_trades = min(safe_max_trades, 8)   # Never more than 8
            
            # Absolute minimum: 3 trades (need some opportunities)
            max_trades = max(3, max_trades)
                
        else:  # Live Broker - More flexible but still strategic
            # Daily loss limit: 2-4% based on account size
            if equity < 25000:
                daily_loss_limit = equity * 0.02  # 2%
            elif equity < 50000:
                daily_loss_limit = equity * 0.025  # 2.5%
            elif equity < 100000:
                daily_loss_limit = equity * 0.03  # 3%
            else:
                daily_loss_limit = equity * 0.035  # 3.5%
            
            # Contracts: Strategic based on equity
            max_contracts = max(1, min(10, int(equity / 20000)))
            
            # Trades: More flexible with live brokers
            if equity < 50000:
                max_trades = 15
            elif equity < 100000:
                max_trades = 18
            else:
                max_trades = 20
        
        # Intelligently set max drawdown based on account type and prop firm rules
        if account_type == "Prop Firm":
            # Most prop firms enforce 10% max drawdown
            # Set to 8% for safety buffer (strategic, not maxing out)
            recommended_max_drawdown = 8.0
        else:  # Live Broker
            # Live brokers are more flexible
            # Set based on equity size
            if equity < 50000:
                recommended_max_drawdown = 12.0
            elif equity < 100000:
                recommended_max_drawdown = 15.0
            else:
                recommended_max_drawdown = 18.0
        
        # Apply the calculated settings
        self.loss_entry.delete(0, tk.END)
        self.loss_entry.insert(0, f"{daily_loss_limit:.2f}")
        self.contracts_var.set(max_contracts)
        self.trades_var.set(max_trades)
        
        # Update info label with comprehensive feedback
        if account_type == "Prop Firm":
            self.auto_adjust_info_label.config(
                text=f"✓ Optimized for ${equity:,.2f} equity ({drawdown_pct:.1f}% current drawdown) - {max_contracts} contracts, ${daily_loss_limit:.0f} daily limit, {max_trades} trades/day",
                fg=self.colors['success']
            )
        else:
            self.auto_adjust_info_label.config(
                text=f"✓ Optimized for ${equity:,.2f} equity - {max_contracts} contracts, ${daily_loss_limit:.0f} daily limit, {max_trades} trades/day",
                fg=self.colors['success']
            )
    
    def start_bot(self):
        """Validate settings and start the trading AI."""
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
        self.config["daily_loss_limit"] = loss_limit
        self.config["max_contracts"] = self.contracts_var.get()
        self.config["max_trades"] = self.trades_var.get()
        self.config["confidence_threshold"] = self.confidence_var.get()
        self.config["shadow_mode"] = self.shadow_mode_var.get()
        self.config["confidence_trading"] = self.confidence_trading_var.get()
        self.config["selected_account"] = self.account_dropdown_var.get()
        
        # Save selected account ID for bot to use
        selected_account_name = self.account_dropdown_var.get()
        accounts = self.config.get("accounts", [])
        for acc in accounts:
            acc_display_name = f"{acc['name']} - ${acc['balance']:,.2f}"
            if acc_display_name == selected_account_name:
                self.config["selected_account_id"] = acc.get("id", "UNKNOWN")
                break
        
        self.config["recovery_mode"] = self.recovery_mode_var.get()
        self.save_config()
        
        # Create .env file
        self.create_env_file()
        
        # Show confirmation with new settings
        symbols_str = ", ".join(selected_symbols)
        broker = self.config.get("broker", "TopStep")
        
        confirmation_text = f"Ready to start AI with these settings:\n\n"
        confirmation_text += f"Broker: {broker}\n"
        confirmation_text += f"Account: {self.account_dropdown_var.get()}\n"
        confirmation_text += f"Symbols: {symbols_str}\n"
        confirmation_text += f"Contracts Per Trade: {self.contracts_var.get()}\n"
        confirmation_text += f"Daily Loss Limit: ${loss_limit}\n"
        confirmation_text += f"Max Trades/Day: {self.trades_var.get()}\n"
        confirmation_text += f"Confidence Threshold: {self.confidence_var.get()}%\n"
        
        # Only show enabled features
        if self.shadow_mode_var.get():
            confirmation_text += f"\n✓ Shadow Mode: ON (paper trading - no real trades)\n"
        
        if self.confidence_trading_var.get():
            confirmation_text += f"✓ Confidence Trading: ENABLED\n"
            confirmation_text += f"  → Auto-adjusts contracts + confidence based on performance\n"
        
        if self.recovery_mode_var.get():
            confirmation_text += f"✓ Recovery Mode: ENABLED\n"
            confirmation_text += f"  → Scales down when approaching daily limits\n"
        
        confirmation_text += f"\nThis will open a PowerShell terminal with live logs.\n"
        confirmation_text += f"Use the window's close button to stop the bot.\n\n"
        confirmation_text += f"Continue?"
        
        result = messagebox.askyesno(
            "Launch Trading Bot?",
            confirmation_text
        )
        
        if not result:
            return
        
        # Launch AI in PowerShell terminal
        try:
            # Get the AI directory (parent of customer folder)
            bot_dir = Path(__file__).parent.parent.absolute()
            
            # PowerShell command to run the AI
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
        """Create .env file from GUI settings.
        
        Security Note: This intentionally writes credentials to .env file, which is the
        standard configuration method for this application. The .env file is in .gitignore
        and should never be committed to version control. Users are responsible for
        securing their .env file on their local system.
        """
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
QUOTRADING_API_KEY={self.config.get("quotrading_api_key", "")}
ACCOUNT_SIZE={self.config.get("account_size", 50000)}

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
# Bot stays on but will NOT execute trades after reaching max (resets daily after market maintenance)
BOT_DAILY_LOSS_LIMIT={self.loss_entry.get()}
# Bot stays on but will NOT execute trades if this limit (in dollars) is hit (resets daily after market maintenance)

# AI/Confidence Settings
BOT_CONFIDENCE_THRESHOLD={self.confidence_var.get()}
# Bot only takes signals above this confidence threshold (user's minimum)
BOT_CONFIDENCE_TRADING={'true' if self.confidence_trading_var.get() else 'false'}
# When approaching daily limit (80%+): SCALES DOWN (higher confidence thresholds + fewer contracts)
# When limit HIT (100%): STOPS trading until daily reset at 6 PM ET
# When moving away from limit: Returns to user's initial settings

# Recovery Mode (All Account Types)
BOT_RECOVERY_MODE={'true' if self.recovery_mode_var.get() else 'false'}
# When approaching daily limit (80%+): SCALES DOWN (higher confidence thresholds + fewer contracts)
# When limit HIT (100%): CONTINUES trading with scaled-down settings (does NOT stop)
# When moving away from limit: Returns to user's initial settings

# Trading Mode
BOT_SHADOW_MODE={'true' if self.shadow_mode_var.get() else 'false'}
BOT_DRY_RUN={'true' if self.shadow_mode_var.get() else 'false'}

# Account Selection
SELECTED_ACCOUNT={self.config.get("selected_account", "Default Account")}
SELECTED_ACCOUNT_ID={self.config.get("selected_account_id", "UNKNOWN")}
# The bot will use this specific account ID when connecting to the broker

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
    
    def _blend_colors(self, color1, color2, ratio):
        """Blend two hex colors together.
        
        Args:
            color1: First hex color (e.g., '#FF0000')
            color2: Second hex color (e.g., '#00FF00')
            ratio: Blend ratio from 0.0 (all color1) to 1.0 (all color2)
        
        Returns:
            Blended hex color string
        """
        # Convert hex to RGB
        r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
        r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
        
        # Blend
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        
        # Convert back to hex
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def get_trading_style(self, value):
        """Get trading style name based on confidence value."""
        if value <= 40:
            return "⚡ VERY AGGRESSIVE"
        elif value <= 60:
            return "⚡ AGGRESSIVE"
        elif value <= 75:
            return "⚖️ BALANCED"
        elif value <= 85:
            return "🛡️ CONSERVATIVE"
        else:
            return "🛡️ VERY CONSERVATIVE"
    
    def get_style_color(self, value):
        """Get color based on confidence value - matches threshold zones."""
        if value <= 40:
            return "#DC2626"  # Red - Aggressive zone
        elif value <= 65:
            return "#F59E0B"  # Orange - Moderate zone
        elif value <= 85:
            return "#10B981"  # Green - Balanced zone
        else:
            return "#3B82F6"  # Blue - Conservative zone
    
    def get_trade_info(self, value):
        """Get trade activity information with color based on confidence value."""
        if value <= 30:
            return ("Maximum Trades", "Highest Activity", "Maximum Risk", "#DC2626")
        elif value <= 40:
            return ("Very High Trades", "Very Active", "High Risk", "#DC2626")
        elif value <= 55:
            return ("High Trade Volume", "Active Trading", "Elevated Risk", "#F59E0B")
        elif value <= 65:
            return ("Good Trade Volume", "Moderate Activity", "Balanced Risk", "#F59E0B")
        elif value <= 75:
            return ("Moderate Trades", "Selective", "Controlled Risk", "#10B981")
        elif value <= 85:
            return ("Fewer Trades", "Conservative", "Lower Risk", "#10B981")
        else:
            return ("Minimal Trades", "Very Conservative", "Lowest Risk", "#3B82F6")
    
    def update_trade_info(self, value):
        """Update trade info labels with colored text."""
        # Get trade info data
        trades, activity, risk, color = self.get_trade_info(value)
        
        # Clear existing labels
        for widget in self.trade_info_frame.winfo_children():
            widget.destroy()
        
        # Create three colored labels with emojis and separators
        labels_data = [
            ("📊", trades),
            ("⚡", activity),
            ("⚠️", risk)
        ]
        
        for i, (emoji, text) in enumerate(labels_data):
            # Create label with emoji and text
            label = tk.Label(
                self.trade_info_frame,
                text=f"{emoji} {text}",
                font=("Segoe UI", 10, "bold"),
                bg=self.colors['card'],
                fg=color
            )
            label.pack(side=tk.LEFT, padx=10)
            
            # Add separator bullet between labels (but not after last one)
            if i < len(labels_data) - 1:
                separator = tk.Label(
                    self.trade_info_frame,
                    text="•",
                    font=("Segoe UI", 10),
                    bg=self.colors['card'],
                    fg="#666666"
                )
                separator.pack(side=tk.LEFT, padx=6)
    
    def update_confidence_display(self, value):
        """Update confidence display when slider moves."""
        conf_value = float(value)
        
        # Update percentage display
        self.confidence_display.config(text=f"{conf_value:.0f}%")
        
        # Update style label
        style_text = self.get_trading_style(conf_value)
        self.confidence_style_label.config(text=style_text)
        
        # Update colors to match threshold zone
        color = self.get_style_color(conf_value)
        self.confidence_display.config(fg=color)
        self.confidence_style_label.config(fg=color)
        
        # Update slider thumb color to match zone (both when idle and when clicked)
        self.confidence_slider.config(bg=color, activebackground=color, troughcolor=color)
        
        # Update trade info with colored text
        self.update_trade_info(conf_value)
    
    def save_config(self):
        """Save current configuration.
        
        Security Note: config.json may contain sensitive data (API keys, passwords). 
        It is included in .gitignore to prevent accidental commits. Users should secure
        this file on their local system.
        
        Password Storage: The password is stored in plain text when "remember credentials"
        is enabled. This is for convenience in a local desktop application context where
        the user has physical access to the machine. For production deployments, consider
        implementing password hashing or using OS keyring services.
        """
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


if __name__ == "__main__":
    app = QuoTradingLauncher()
    app.run()
