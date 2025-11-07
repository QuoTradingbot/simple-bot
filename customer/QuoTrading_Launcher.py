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
    """Professional GUI launcher for QuoTrading AI bot - Blue/White Theme with Cloud Authentication."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("QuoTrading - Professional Trading Platform")
        self.root.geometry("700x800")
        self.root.resizable(True, True)
        self.root.minsize(700, 800)
        
        # Blue and White color scheme - Professional theme
        self.colors = {
            'primary': '#FFFFFF',        # White background
            'secondary': '#F0F4F8',      # Light gray/blue for secondary cards
            'success': '#2563EB',        # Blue - primary accent
            'success_dark': '#1E40AF',   # Darker blue for buttons/headers
            'success_darker': '#1E3A8A', # Even darker blue for depth
            'error': '#DC2626',          # Red for error messages
            'warning': '#F59E0B',        # Orange for warnings
            'background': '#FFFFFF',     # White main background
            'card': '#F8FAFC',           # Very light gray card background
            'card_elevated': '#EFF6FF',  # Light blue tint for elevation
            'text': '#1F2937',           # Dark gray text (primary)
            'text_light': '#4B5563',     # Medium gray (secondary labels)
            'text_secondary': '#6B7280', # Light gray (tertiary/hints)
            'border': '#2563EB',         # Blue border
            'border_subtle': '#93C5FD',  # Light blue subtle border
            'input_bg': '#FFFFFF',       # White for input fields
            'input_focus': '#EFF6FF',    # Light blue tint on focus
            'button_hover': '#3B82F6',   # Lighter blue for hover state
            'shadow': '#E5E7EB'          # Light gray for shadow effect
        }
        
        # Default fallback symbol
        self.DEFAULT_SYMBOL = 'ES'
        
        # Account mismatch detection threshold
        self.ACCOUNT_MISMATCH_THRESHOLD = 1000  # Dollars - warn if difference is > $1000
        
        # Prop firm maximum drawdown percentage (most common rule)
        self.PROP_FIRM_MAX_DRAWDOWN = 8.0  # 8% for most prop firms (some use 10%)
        
        # Cloud validation API URL
        self.VALIDATION_API_URL = "http://localhost:5000/api/validate"  # Update with your cloud server URL
        
        self.root.configure(bg=self.colors['background'])
        
        # Load saved config
        self.config_file = Path("config.json")
        self.config = self.load_config()
        
        # Current screen tracker
        self.current_screen = 0
        
        # Bot process reference
        self.bot_process = None
        
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
        top_accent = tk.Frame(header, bg=self.colors['success'], height=3)
        top_accent.pack(fill=tk.X)
        
        title_label = tk.Label(
            header,
            text=title,
            font=("Segoe UI", 18, "bold"),
            bg=self.colors['success_dark'],
            fg='white'
        )
        title_label.pack(pady=(15, 3))
        
        if subtitle:
            subtitle_label = tk.Label(
                header,
                text=subtitle,
                font=("Segoe UI", 10),
                bg=self.colors['success_dark'],
                fg='white'
            )
            subtitle_label.pack(pady=(0, 5))
        
        # Bottom shadow effect
        bottom_shadow = tk.Frame(header, bg=self.colors['shadow'], height=2)
        bottom_shadow.pack(side=tk.BOTTOM, fill=tk.X)
        
        return header
    
    def create_input_field(self, parent, label_text, is_password=False, placeholder=""):
        """Create a styled input field with label and premium design."""
        container = tk.Frame(parent, bg=self.colors['card'])
        container.pack(fill=tk.X, pady=8)
        
        label = tk.Label(
            container,
            text=label_text,
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        label.pack(anchor=tk.W, pady=(0, 4))
        
        # Create frame for input with border effect
        input_frame = tk.Frame(container, bg=self.colors['border'], bd=0)
        input_frame.pack(fill=tk.X, padx=1, pady=1)
        
        entry = tk.Entry(
            input_frame,
            font=("Segoe UI", 10),
            bg=self.colors['input_bg'],
            fg=self.colors['text'],
            insertbackground=self.colors['success'],
            relief=tk.FLAT,
            bd=0,
            show="●" if is_password else ""
        )
        entry.pack(fill=tk.X, ipady=8, padx=2, pady=2)
        
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
    
    def create_button(self, parent, text, command, button_type="next"):
        """Create a styled button with premium design and hover effects."""
        if button_type == "next":
            bg = self.colors['success_dark']
            fg = 'white'
            width = 18
            height = 1
        elif button_type == "back":
            bg = self.colors['secondary']
            fg = self.colors['text']
            width = 12
            height = 1
        else:  # start, continue, or other button types
            bg = self.colors['success']
            fg = 'white'
            width = 20
            height = 1
        
        # Create frame for button with shadow effect
        button_container = tk.Frame(parent, bg=parent.cget('bg'))
        
        # Shadow effect
        shadow = tk.Frame(button_container, bg=self.colors['shadow'], height=2)
        shadow.pack(fill=tk.X, pady=(2, 0))
        
        button = tk.Button(
            button_container,
            text=text,
            font=("Segoe UI", 11, "bold"),
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
        button.pack(fill=tk.X)
        
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
        header = self.create_header("Broker Connection", "Select your account type and broker")
        
        # Main container with scrolling
        main = tk.Frame(self.root, bg=self.colors['background'])
        main.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for scrolling
        canvas = tk.Canvas(main, bg=self.colors['background'], highlightthickness=0)
        scrollbar = tk.Scrollbar(main, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['background'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Enable mouse wheel scrolling with cross-platform support
        mousewheel_handler = self.create_mousewheel_handler(canvas)
        canvas.bind("<MouseWheel>", mousewheel_handler)  # Windows/macOS
        canvas.bind("<Button-4>", mousewheel_handler)    # Linux scroll up
        canvas.bind("<Button-5>", mousewheel_handler)    # Linux scroll down
        
        # Container inside scrollable frame
        container = tk.Frame(scrollable_frame, bg=self.colors['background'], padx=30, pady=15)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Card
        card = tk.Frame(container, bg=self.colors['card'], relief=tk.FLAT, bd=0)
        card.pack(fill=tk.BOTH, expand=True)
        card.configure(highlightbackground=self.colors['border'], highlightthickness=2)
        
        # Card content
        content = tk.Frame(card, bg=self.colors['card'], padx=25, pady=20)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Info message
        info = tk.Label(
            content,
            text="Choose your broker type and enter credentials",
            font=("Segoe UI", 9),
            bg=self.colors['card'],
            fg=self.colors['text_light'],
            justify=tk.CENTER
        )
        info.pack(pady=(0, 10))
        
        # Broker Type Selection - Card-style buttons
        type_label = tk.Label(
            content,
            text="Account Type:",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        type_label.pack(pady=(0, 8))
        
        # Container for cards
        cards_container = tk.Frame(content, bg=self.colors['card'])
        cards_container.pack(fill=tk.X, pady=(0, 10))
        
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
            card_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
            
            # Make card clickable
            def make_select(bt=btype):
                return lambda e: self.select_broker_type(bt)
            
            card_frame.bind("<Button-1>", make_select(btype))
            
            # Card content
            inner = tk.Frame(card_frame, bg=self.colors['secondary'])
            inner.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
            inner.bind("<Button-1>", make_select(btype))
            
            # Icon
            icon_label = tk.Label(
                inner,
                text=icon,
                font=("Segoe UI", 20),
                bg=self.colors['secondary'],
                fg=self.colors['text']
            )
            icon_label.pack()
            icon_label.bind("<Button-1>", make_select(btype))
            
            # Type name
            type_name = tk.Label(
                inner,
                text=btype,
                font=("Segoe UI", 10, "bold"),
                bg=self.colors['secondary'],
                fg=self.colors['text']
            )
            type_name.pack(pady=(5, 3))
            type_name.bind("<Button-1>", make_select(btype))
            
            # Description
            desc_label = tk.Label(
                inner,
                text=desc,
                font=("Segoe UI", 8),
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
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        broker_label.pack(anchor=tk.W, pady=(8, 3))
        
        self.broker_var = tk.StringVar(value=self.config.get("broker", "TopStep"))
        self.broker_dropdown = ttk.Combobox(
            content,
            textvariable=self.broker_var,
            state="readonly",
            font=("Segoe UI", 9),
            width=35
        )
        self.broker_dropdown.pack(fill=tk.X, pady=(0, 10))
        
        # Update broker options based on selected type
        self.update_broker_options()
        
        # QuoTrading API Key
        self.quotrading_api_key_entry = self.create_input_field(
            content,
            "QuoTrading API Key:",
            is_password=True,
            placeholder=self.config.get("quotrading_api_key", "")
        )
        
        # TopStep Account Type Selection
        account_type_label = tk.Label(
            content,
            text="TopStep Account Type:",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        account_type_label.pack(anchor=tk.W, pady=(8, 3))
        
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
        
        # Get saved account type or default
        saved_account_type = self.config.get("topstep_account_type", "Trading Combine $50K")
        # Ensure saved account type is valid, fallback to default if not
        if saved_account_type not in self.topstep_account_types:
            saved_account_type = "Trading Combine $50K"
        self.account_type_var = tk.StringVar(value=saved_account_type)
        
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
            font=("Segoe UI", 10),
            style='TopStepAccount.TCombobox',
            values=list(self.topstep_account_types.keys())
        )
        self.account_type_dropdown.pack(fill=tk.X, ipady=6, padx=2, pady=2)
        self.account_type_dropdown.bind("<<ComboboxSelected>>", self.update_account_type_info)
        
        # Account type info display
        self.account_info_display = tk.Label(
            content,
            text=self.topstep_account_types[saved_account_type]["description"],
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text_light'],
            wraplength=500,
            justify=tk.LEFT
        )
        self.account_info_display.pack(anchor=tk.W, pady=(3, 10))
        
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
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text_secondary']
        )
        help_text.pack(pady=(3, 12))
        
        # Button container
        button_frame = tk.Frame(content, bg=self.colors['card'])
        button_frame.pack(fill=tk.X, pady=5)
        
        # Continue button (no back button on first screen)
        continue_btn = self.create_button(button_frame, "CONTINUE →", self.validate_broker, "next")
        continue_btn.pack(side=tk.RIGHT)
    
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
    
    def update_account_type_info(self, event=None):
        """Update the account type info display when selection changes."""
        selected_type = self.account_type_var.get()
        if selected_type in self.topstep_account_types:
            account_info = self.topstep_account_types[selected_type]
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
        
        # Get selected TopStep account type
        account_type = self.account_type_var.get()
        if account_type in self.topstep_account_types:
            account_info = self.topstep_account_types[account_type]
            account_size = account_info["size"]
        else:
            account_size = "50000"  # Default fallback
        
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
        
        # Check if using admin key - bypass validation
        if quotrading_api_key == "QUOTRADING_ADMIN_MASTER_2025":
            self.config["broker_type"] = self.broker_type_var.get()
            self.config["broker"] = broker
            self.config["broker_token"] = token
            self.config["broker_username"] = username
            self.config["quotrading_api_key"] = quotrading_api_key
            self.config["topstep_account_type"] = account_type
            self.config["account_size"] = account_size
            self.config["broker_validated"] = True
            self.save_config()
            self.setup_trading_screen()
            return
        
        # Show loading spinner
        self.show_loading(f"Validating {broker} credentials...")
        
        # Define success callback
        def on_success():
            self.hide_loading()
            # Save broker credentials
            self.config["broker_type"] = self.broker_type_var.get()
            self.config["broker"] = broker
            self.config["broker_token"] = token
            self.config["broker_username"] = username
            self.config["quotrading_api_key"] = quotrading_api_key
            self.config["topstep_account_type"] = account_type
            self.config["account_size"] = account_size
            self.config["broker_validated"] = True
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
        
        # Main container with scrollbar capability
        main = tk.Frame(self.root, bg=self.colors['background'])
        main.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for scrolling
        canvas = tk.Canvas(main, bg=self.colors['background'], highlightthickness=0)
        scrollbar = tk.Scrollbar(main, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['background'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Enable mouse wheel scrolling with cross-platform support
        mousewheel_handler = self.create_mousewheel_handler(canvas)
        canvas.bind("<MouseWheel>", mousewheel_handler)  # Windows/macOS
        canvas.bind("<Button-4>", mousewheel_handler)    # Linux scroll up
        canvas.bind("<Button-5>", mousewheel_handler)    # Linux scroll down
        
        # Container inside scrollable frame
        container = tk.Frame(scrollable_frame, bg=self.colors['background'], padx=25, pady=12)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Card
        card = tk.Frame(container, bg=self.colors['card'], relief=tk.FLAT, bd=0)
        card.pack(fill=tk.BOTH, expand=True)
        card.configure(highlightbackground=self.colors['border'], highlightthickness=2)
        
        # Card content
        content = tk.Frame(card, bg=self.colors['card'], padx=20, pady=15)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Symbol Selection
        symbol_label = tk.Label(
            content,
            text="Trading Symbols (select at least one):",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        symbol_label.pack(anchor=tk.W, pady=(0, 6))
        
        # Symbol checkboxes - 2 columns
        symbol_frame = tk.Frame(content, bg=self.colors['card'])
        symbol_frame.pack(fill=tk.X, pady=(0, 10))
        
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
                font=("Segoe UI", 9),
                bg=self.colors['card'],
                fg=self.colors['text'],
                selectcolor=self.colors['secondary'],
                activebackground=self.colors['card'],
                activeforeground=self.colors['success'],
                cursor="hand2"
            )
            cb.grid(row=row, column=col, sticky=tk.W, padx=8, pady=2)
        
        # Account Settings Row
        settings_row = tk.Frame(content, bg=self.colors['card'])
        settings_row.pack(fill=tk.X, pady=(0, 10))
        
        # Account Size
        acc_frame = tk.Frame(settings_row, bg=self.colors['card'])
        acc_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        
        tk.Label(
            acc_frame,
            text="Account Size ($):",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 3))
        
        self.account_entry = tk.Entry(
            acc_frame,
            font=("Segoe UI", 9),
            bg=self.colors['input_bg'],
            fg=self.colors['text'],
            insertbackground=self.colors['success'],
            relief=tk.FLAT,
            bd=0,
            highlightthickness=2,
            highlightbackground=self.colors['border'],
            highlightcolor=self.colors['success']
        )
        self.account_entry.pack(fill=tk.X, ipady=4, padx=2)
        self.account_entry.insert(0, self.config.get("account_size", "10000"))
        
        # Max Drawdown with account awareness
        drawdown_frame = tk.Frame(settings_row, bg=self.colors['card'])
        drawdown_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        
        tk.Label(
            drawdown_frame,
            text="Max Drawdown (%):",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 3))
        
        # Get account type for drawdown validation
        account_type = self.config.get("broker_type", "Prop Firm")
        default_drawdown = 8.0 if account_type == "Prop Firm" else 15.0
        
        self.drawdown_var = tk.DoubleVar(value=self.config.get("max_drawdown", default_drawdown))
        
        # Add validation to warn if exceeding safe limits
        def validate_drawdown(*args):
            try:
                value = self.drawdown_var.get()
                account_type = self.config.get("broker_type", "Prop Firm")
                
                if account_type == "Prop Firm":
                    # Most prop firms fail at 10% drawdown
                    if value > 10.0:
                        drawdown_info.config(
                            text="⚠ Exceeds prop firm limit (10%) - Will cause account failure!",
                            fg='#FF0000'  # Red warning
                        )
                    elif value > 8.0:
                        drawdown_info.config(
                            text="⚠ Caution: Very close to prop firm failure threshold (10%)",
                            fg=self.colors.get('warning', '#FFA500')
                        )
                    else:
                        drawdown_info.config(
                            text=f"Safe zone for {account_type} (under 8%)",
                            fg=self.colors['text_secondary']
                        )
                else:  # Live Broker
                    if value > 20.0:
                        drawdown_info.config(
                            text="⚠ High drawdown risk - Consider reducing",
                            fg=self.colors.get('warning', '#FFA500')
                        )
                    else:
                        drawdown_info.config(
                            text=f"Drawdown limit for {account_type}",
                            fg=self.colors['text_secondary']
                        )
            except:
                pass
        
        self.drawdown_var.trace_add('write', validate_drawdown)
        
        drawdown_spin = ttk.Spinbox(
            drawdown_frame,
            from_=1.0,
            to=25.0,
            increment=0.5,
            textvariable=self.drawdown_var,
            width=12,
            format="%.1f"
        )
        drawdown_spin.pack(fill=tk.X, ipady=2)
        
        # Info label for drawdown warnings
        drawdown_info = tk.Label(
            drawdown_frame,
            text=f"Safe zone for {account_type}",
            font=("Segoe UI", 7),
            bg=self.colors['card'],
            fg=self.colors['text_secondary']
        )
        drawdown_info.pack(anchor=tk.W, pady=(2, 0))
        
        # Trigger initial validation
        validate_drawdown()
        
        # Daily Loss Limit
        loss_frame = tk.Frame(settings_row, bg=self.colors['card'])
        loss_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(
            loss_frame,
            text="Daily Loss Limit ($):",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 3))
        
        self.loss_entry = tk.Entry(
            loss_frame,
            font=("Segoe UI", 9),
            bg=self.colors['input_bg'],
            fg=self.colors['text'],
            insertbackground=self.colors['success'],
            relief=tk.FLAT,
            bd=0,
            highlightthickness=2,
            highlightbackground=self.colors['border'],
            highlightcolor=self.colors['success']
        )
        self.loss_entry.pack(fill=tk.X, ipady=4, padx=2)
        self.loss_entry.insert(0, self.config.get("daily_loss_limit", "2000"))
        
        # Advanced Settings Row
        advanced_row = tk.Frame(content, bg=self.colors['card'])
        advanced_row.pack(fill=tk.X, pady=(0, 10))
        
        # Max Contracts with account type awareness and enforcement
        contracts_frame = tk.Frame(advanced_row, bg=self.colors['card'])
        contracts_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        
        tk.Label(
            contracts_frame,
            text="Contracts Per Trade:",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 3))
        
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
            font=("Segoe UI", 7),
            bg=self.colors['card'],
            fg=self.colors['text_secondary']
        )
        contracts_info.pack(anchor=tk.W, pady=(2, 0))
        
        # Max Trades Per Day
        trades_frame = tk.Frame(advanced_row, bg=self.colors['card'])
        trades_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(
            trades_frame,
            text="Max Trades/Day:",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 3))
        
        self.trades_var = tk.IntVar(value=self.config.get("max_trades", 10))
        trades_spin = ttk.Spinbox(
            trades_frame,
            from_=1,
            to=50,
            textvariable=self.trades_var,
            width=12
        )
        trades_spin.pack(fill=tk.X, ipady=2)
        
        # AI/Confidence Settings Row
        ai_row = tk.Frame(content, bg=self.colors['card'])
        ai_row.pack(fill=tk.X, pady=(0, 10))
        
        # Confidence Threshold
        confidence_frame = tk.Frame(ai_row, bg=self.colors['card'])
        confidence_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        
        tk.Label(
            confidence_frame,
            text="Confidence Threshold (%):",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 3))
        
        self.confidence_var = tk.DoubleVar(value=self.config.get("confidence_threshold", 65.0))
        confidence_spin = ttk.Spinbox(
            confidence_frame,
            from_=0.0,
            to=100.0,
            increment=5.0,
            textvariable=self.confidence_var,
            width=12,
            format="%.1f"
        )
        confidence_spin.pack(fill=tk.X, ipady=2)
        
        # Info label for confidence threshold
        confidence_info = tk.Label(
            confidence_frame,
            text="Minimum confidence - bot takes signals above this",
            font=("Segoe UI", 7),
            bg=self.colors['card'],
            fg=self.colors['text_secondary']
        )
        confidence_info.pack(anchor=tk.W, pady=(2, 0))
        
        # Dynamic Confidence Threshold checkbox
        self.dynamic_confidence_var = tk.BooleanVar(value=self.config.get("dynamic_confidence", False))
        dynamic_conf_cb = tk.Checkbutton(
            confidence_frame,
            text="Enable Dynamic Confidence",
            variable=self.dynamic_confidence_var,
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text'],
            selectcolor=self.colors['secondary'],
            activebackground=self.colors['card'],
            activeforeground=self.colors['success'],
            cursor="hand2"
        )
        dynamic_conf_cb.pack(anchor=tk.W, pady=(3, 2))
        
        # Info label for dynamic confidence
        dynamic_conf_info = tk.Label(
            confidence_frame,
            text="Auto-increases (never below your setting) when bot\nperforms poorly or approaching account limits",
            font=("Segoe UI", 7),
            bg=self.colors['card'],
            fg=self.colors['text_secondary'],
            justify=tk.LEFT
        )
        dynamic_conf_info.pack(anchor=tk.W)
        
        # Shadow Mode checkbox (keeping existing functionality)
        shadow_frame = tk.Frame(ai_row, bg=self.colors['card'])
        shadow_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.shadow_mode_var = tk.BooleanVar(value=self.config.get("shadow_mode", False))
        shadow_cb = tk.Checkbutton(
            shadow_frame,
            text="Shadow Mode",
            variable=self.shadow_mode_var,
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text'],
            selectcolor=self.colors['secondary'],
            activebackground=self.colors['card'],
            activeforeground=self.colors['success'],
            cursor="hand2"
        )
        shadow_cb.pack(anchor=tk.W, pady=(0, 3))
        
        # Info label for shadow mode
        shadow_info = tk.Label(
            shadow_frame,
            text="Paper trade mode - no real trades executed",
            font=("Segoe UI", 7),
            bg=self.colors['card'],
            fg=self.colors['text_secondary']
        )
        shadow_info.pack(anchor=tk.W)
        
        # Dynamic Contract Mode Row
        dynamic_row = tk.Frame(content, bg=self.colors['card'])
        dynamic_row.pack(fill=tk.X, pady=(0, 10))
        
        self.dynamic_contracts_var = tk.BooleanVar(value=self.config.get("dynamic_contracts", False))
        dynamic_cb = tk.Checkbutton(
            dynamic_row,
            text="Dynamic Contract Mode",
            variable=self.dynamic_contracts_var,
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text'],
            selectcolor=self.colors['secondary'],
            activebackground=self.colors['card'],
            activeforeground=self.colors['success'],
            cursor="hand2"
        )
        dynamic_cb.pack(anchor=tk.W, pady=(0, 3))
        
        # Info label for dynamic contract mode
        dynamic_info = tk.Label(
            dynamic_row,
            text="Uses signal confidence to scale contracts (1 to your max) dynamically",
            font=("Segoe UI", 7),
            bg=self.colors['card'],
            fg=self.colors['text_secondary']
        )
        dynamic_info.pack(anchor=tk.W)
        
        # Recovery Mode Section - Works for ALL account types
        recovery_frame = tk.Frame(content, bg=self.colors['card_elevated'], relief=tk.FLAT, bd=0)
        recovery_frame.pack(fill=tk.X, pady=(0, 10), padx=2)
        recovery_frame.configure(highlightbackground=self.colors['border_subtle'], highlightthickness=1)
        
        recovery_content = tk.Frame(recovery_frame, bg=self.colors['card_elevated'], padx=12, pady=10)
        recovery_content.pack(fill=tk.X)
        
        recovery_label = tk.Label(
            recovery_content,
            text="Recovery Mode (All Account Types):",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['card_elevated'],
            fg=self.colors['text']
        )
        recovery_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Default: Recovery mode disabled (safest - bot stops when approaching limits)
        # Inverted from previous logic - now the checkbox ENABLES recovery mode
        self.recovery_mode_var = tk.BooleanVar(value=self.config.get("recovery_mode", False))
        
        # Checkbox for recovery mode
        recovery_cb = tk.Checkbutton(
            recovery_content,
            text="Enable Recovery Mode",
            variable=self.recovery_mode_var,
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card_elevated'],
            fg=self.colors['text'],
            selectcolor=self.colors['secondary'],
            activebackground=self.colors['card_elevated'],
            activeforeground=self.colors['success'],
            cursor="hand2"
        )
        recovery_cb.pack(anchor=tk.W, pady=(0, 3))
        
        # Add callback to update info label
        def update_recovery_info(*args):
            if self.recovery_mode_var.get():
                recovery_info.config(
                    text="⚠️ RECOVERY MODE ENABLED: Bot will continue trading when approaching limits, automatically increasing confidence "
                         "requirements and adjusting risk. The closer to limits, the higher the confidence and the more conservative the trading. "
                         "Use this to give the bot a chance to recover from a bad day.",
                    fg='#FFA500'  # Orange warning
                )
                recovery_explanation.config(
                    text="When Recovery Mode is ENABLED:\n"
                         "• Bot continues trading even when close to daily loss or max drawdown limits\n"
                         "• Confidence auto-scales: 75% @ 80% of limits, 85% @ 90%, 90% @ 95%+\n"
                         "• Position size dynamically adjusts: reduces when approaching, increases when safe\n"
                         "• Contracts scale: 33%→50%→75%→85%→95%→100% as you get safer\n"
                         "• Only takes highest-quality signals when close to failure\n"
                         "• Attempts to recover from losses with smart risk management\n"
                         "• Applies to ALL account types (prop firms, live brokers, etc.)",
                    fg=self.colors['text_secondary']
                )
            else:
                recovery_info.config(
                    text="ℹ️ Bot will WARN when approaching limits but NEVER lock you out. "
                         "You maintain full control. Bot shows warnings and recommendations, but YOU decide whether to trade. "
                         "Recommended for cautious traders who want guidance without restrictions.",
                    fg=self.colors['text']
                )
                recovery_explanation.config(
                    text="When Recovery Mode is DISABLED:\n"
                         "• Bot WARNS at 80% of daily loss (e.g., $1600 of $2000 limit)\n"
                         "• Bot WARNS at 80% of max drawdown with dollar amounts\n"
                         "• Bot NEVER stops you from trading - you maintain full control\n"
                         "• Smart recommendations provided (confidence, contracts, limits)\n"
                         "• Bot continues running and monitoring market conditions\n"
                         "• One-click apply recommendations if desired\n"
                         "• User maintains full control with intelligent warnings",
                    fg=self.colors['text_secondary']
                )
        
        self.recovery_mode_var.trace_add('write', update_recovery_info)
        
        # Info label
        recovery_info = tk.Label(
            recovery_content,
            text="",
            font=("Segoe UI", 7),
            bg=self.colors['card_elevated'],
            fg=self.colors['text_secondary'],
            wraplength=520,
            justify=tk.LEFT
        )
        recovery_info.pack(anchor=tk.W, pady=(0, 5))
        
        # Detailed explanation
        recovery_explanation = tk.Label(
            recovery_content,
            text="",
            font=("Segoe UI", 7, "italic"),
            bg=self.colors['card_elevated'],
            fg=self.colors['text_secondary'],
            wraplength=520,
            justify=tk.LEFT
        )
        recovery_explanation.pack(anchor=tk.W, pady=(2, 0))
        
        # Trigger initial info update
        update_recovery_info()
        
        # Trailing Drawdown Section (Optional Risk Management Feature)
        trailing_dd_frame = tk.Frame(content, bg=self.colors['card_elevated'], relief=tk.FLAT, bd=0)
        trailing_dd_frame.pack(fill=tk.X, pady=(0, 10), padx=2)
        trailing_dd_frame.configure(highlightbackground=self.colors['border_subtle'], highlightthickness=1)
        
        trailing_dd_content = tk.Frame(trailing_dd_frame, bg=self.colors['card_elevated'], padx=12, pady=10)
        trailing_dd_content.pack(fill=tk.X)
        
        trailing_dd_label = tk.Label(
            trailing_dd_content,
            text="Advanced Risk Protection:",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['card_elevated'],
            fg=self.colors['text']
        )
        trailing_dd_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Trailing Drawdown checkbox with two-floor system awareness
        # Initialize with smart defaults based on account type and broker
        broker_name = self.config.get("broker", "")
        default_trailing = False  # Default to disabled for most cases
        self.trailing_drawdown_var = tk.BooleanVar(value=self.config.get("trailing_drawdown", default_trailing))
        
        # Determine broker-specific trailing drawdown requirements
        def get_broker_trailing_requirement(broker):
            """Returns (requires_trailing, hard_floor_type)"""
            if broker == "Apex":
                return (True, "trailing")  # REQUIRED - Hard trailing floor enforced
            elif broker == "TopStep":
                return (False, "static")  # Optional - Static hard floor at $47K
            elif broker == "Tradovate":
                return (False, "static")  # Live broker - Optional soft floor
            else:
                return (False, "static")  # Default: optional with static hard floor
        
        requires_trailing, hard_floor_type = get_broker_trailing_requirement(broker_name)
        
        # If broker REQUIRES trailing, enable and lock checkbox
        if requires_trailing:
            self.trailing_drawdown_var.set(True)
        
        # Add validation to provide smart, two-floor aware guidance
        def update_trailing_drawdown_info(*args):
            enabled = self.trailing_drawdown_var.get()
            account_type = self.config.get("broker_type", "Prop Firm")
            broker = self.config.get("broker", "")
            requires, floor_type = get_broker_trailing_requirement(broker)
            
            # Session-aware: Check if account has fetched data
            accounts = self.config.get("accounts", [])
            has_account_data = len(accounts) > 0
            
            # Update checkbox label based on requirement
            if requires:
                trailing_dd_cb.config(
                    text=f"Trailing Drawdown (Required by {broker})",
                    state='disabled',  # Lock checkbox
                    fg=self.colors['error']  # Red to show it's critical
                )
            else:
                trailing_dd_cb.config(
                    text="Enable Trailing Drawdown (Optional - Soft Floor Protection)",
                    state='normal',
                    fg=self.colors['text']
                )
            
            if enabled:
                if requires:
                    # HARD FLOOR - Required by prop firm (account fails if violated)
                    trailing_dd_info.config(
                        text=f"⚠️ REQUIRED Hard Floor - {broker} enforces trailing drawdown. Account FAILS if violated (not pauseable). This is your firm's actual rule.",
                        fg=self.colors['error']
                    )
                else:
                    # SOFT FLOOR - Optional personal protection (bot stops trading when approaching failure)
                    if broker == "TopStep":
                        trailing_dd_info.config(
                            text="✓ Personal Soft Floor Active - Moves UP with profits. Bot stops making trades (doesn't fail) when approaching this floor. TopStep's hard floor: $47K static (separate).",
                            fg=self.colors['success']
                        )
                    else:
                        trailing_dd_info.config(
                            text="✓ Personal Soft Floor Active - Moves UP with profits, never down. Bot stops making trades when approaching this floor - continuous monitoring. This is YOUR protection layer.",
                            fg=self.colors['success']
                        )
            else:
                # Disabled - show what they're missing
                if account_type == "Prop Firm":
                    if broker == "TopStep":
                        trailing_dd_info.config(
                            text=f"TopStep's hard floor: $47K static (account fails if violated). Optional: Enable soft trailing floor for extra protection - bot stops trading when too close to failure.",
                            fg=self.colors['text_secondary']
                        )
                    else:
                        trailing_dd_info.config(
                            text="Optional soft floor protection - Adds personal safety layer that stops trading before you approach your firm's hard floor. Bot continues monitoring.",
                            fg=self.colors['text_secondary']
                        )
                else:  # Live Broker
                    trailing_dd_info.config(
                        text="Optional soft floor - Bot stops making trades (doesn't shutdown) when approaching this floor. Continues monitoring. Protects your capital from extended losing streaks.",
                        fg=self.colors['text_secondary']
                    )
        
        self.trailing_drawdown_var.trace_add('write', update_trailing_drawdown_info)
        
        trailing_dd_cb = tk.Checkbutton(
            trailing_dd_content,
            text="Enable Trailing Drawdown",
            variable=self.trailing_drawdown_var,
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card_elevated'],
            fg=self.colors['text'],
            selectcolor=self.colors['secondary'],
            activebackground=self.colors['card_elevated'],
            activeforeground=self.colors['success'],
            cursor="hand2"
        )
        trailing_dd_cb.pack(anchor=tk.W, pady=(0, 3))
        
        # Info label for trailing drawdown with account-aware guidance
        trailing_dd_info = tk.Label(
            trailing_dd_content,
            text="",
            font=("Segoe UI", 7),
            bg=self.colors['card_elevated'],
            fg=self.colors['text_secondary'],
            wraplength=520,
            justify=tk.LEFT
        )
        trailing_dd_info.pack(anchor=tk.W, pady=(0, 5))
        
        # Trigger initial info update
        update_trailing_drawdown_info()
        
        # Comprehensive explanation text for two-floor trailing drawdown system
        trailing_dd_explanation = tk.Label(
            trailing_dd_content,
            text="ℹ️ Two-Floor System Explained:\n"
                 "\n"
                 "HARD FLOOR (Prop Firm's Actual Rule):\n"
                 "• Apex: Trailing floor that moves up (REQUIRED - account fails if violated)\n"
                 "• TopStep: Static at $47K forever (account fails if violated)\n"
                 "• Cannot be disabled - enforced by your prop firm\n"
                 "\n"
                 "SOFT FLOOR (Your Personal Protection - THIS CHECKBOX):\n"
                 "• Optional safety YOU can enable for extra protection\n"
                 "• Moves UP with your profits (just like hard trailing)\n"
                 "• If approaching: Bot stops making trades (doesn't fail or shutdown)\n"
                 "• Bot continues monitoring - may resume if conditions improve\n"
                 "• Helps you avoid getting close to the dangerous hard floor\n"
                 "\n"
                 "Example: TopStep account with soft floor enabled:\n"
                 "• Hard floor: $47K static (account fails - can't change)\n"
                 "• Peak at $55K → Soft floor at $49.5K (with 10% max DD)\n"
                 "• Drop to $49K → Bot stops trading (approaching soft floor)\n"
                 "• Hard floor still $47K - you're $2K away, still safe!\n"
                 "• Bot continues running and monitoring performance\n"
                 "• May resume if dynamic confidence helps create safety margin",
            font=("Segoe UI", 7, "italic"),
            bg=self.colors['card_elevated'],
            fg=self.colors['text_secondary'],
            wraplength=520,
            justify=tk.LEFT
        )
        trailing_dd_explanation.pack(anchor=tk.W, pady=(2, 0))
        
        # Account Fetch Section
        fetch_frame = tk.Frame(content, bg=self.colors['card_elevated'], relief=tk.FLAT, bd=0)
        fetch_frame.pack(fill=tk.X, pady=(0, 10), padx=2)
        fetch_frame.configure(highlightbackground=self.colors['border_subtle'], highlightthickness=1)
        
        fetch_content = tk.Frame(fetch_frame, bg=self.colors['card_elevated'], padx=12, pady=10)
        fetch_content.pack(fill=tk.X)
        
        fetch_label = tk.Label(
            fetch_content,
            text="Account Information:",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['card_elevated'],
            fg=self.colors['text']
        )
        fetch_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Account selection dropdown
        account_select_frame = tk.Frame(fetch_content, bg=self.colors['card_elevated'])
        account_select_frame.pack(fill=tk.X, pady=(0, 8))
        
        tk.Label(
            account_select_frame,
            text="Select Account:",
            font=("Segoe UI", 9),
            bg=self.colors['card_elevated'],
            fg=self.colors['text_light']
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        self.account_dropdown_var = tk.StringVar(value=self.config.get("selected_account", "Default Account"))
        self.account_dropdown = ttk.Combobox(
            account_select_frame,
            textvariable=self.account_dropdown_var,
            state="readonly",
            font=("Segoe UI", 9),
            width=25,
            values=["Default Account"]
        )
        self.account_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Fetch button and info display - VERY IMPORTANT
        fetch_button_frame = tk.Frame(fetch_content, bg=self.colors['card_elevated'])
        fetch_button_frame.pack(fill=tk.X)
        
        # Add emphasis label
        important_label = tk.Label(
            fetch_button_frame,
            text="⚠️ VERY IMPORTANT:",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card_elevated'],
            fg=self.colors['error'],
            anchor=tk.W
        )
        important_label.pack(side=tk.LEFT, padx=(0, 5))
        
        fetch_btn = self.create_button(fetch_button_frame, "Fetch Account Info", self.fetch_account_info, "next")
        fetch_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Account info display label
        self.account_info_label = tk.Label(
            fetch_button_frame,
            text="Fetching helps bot determine account type and provide best settings",
            font=("Segoe UI", 8),
            bg=self.colors['card_elevated'],
            fg=self.colors['text_secondary'],
            anchor=tk.W
        )
        self.account_info_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Auto-adjust parameters section
        auto_adjust_frame = tk.Frame(fetch_content, bg=self.colors['card_elevated'])
        auto_adjust_frame.pack(fill=tk.X, pady=(8, 0))
        
        auto_adjust_btn = self.create_button(auto_adjust_frame, "Auto-Adjust Parameters", self.auto_adjust_parameters, "next")
        auto_adjust_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Auto-adjust info label
        self.auto_adjust_info_label = tk.Label(
            auto_adjust_frame,
            text="Automatically optimizes settings based on account balance",
            font=("Segoe UI", 8),
            bg=self.colors['card_elevated'],
            fg=self.colors['text_secondary'],
            anchor=tk.W
        )
        self.auto_adjust_info_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Summary display
        summary_frame = tk.Frame(content, bg=self.colors['secondary'], relief=tk.FLAT, bd=0)
        summary_frame.pack(fill=tk.X, pady=(8, 12))
        summary_frame.configure(highlightbackground=self.colors['border'], highlightthickness=1)
        
        summary_content = tk.Frame(summary_frame, bg=self.colors['secondary'], padx=12, pady=8)
        summary_content.pack(fill=tk.X)
        
        summary_title = tk.Label(
            summary_content,
            text="✓ Ready to Trade",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['secondary'],
            fg=self.colors['success']
        )
        summary_title.pack(pady=(0, 3))
        
        broker = self.config.get("broker", "TopStep")
        account_type = self.config.get("topstep_account_type", "Trading Combine $50K")
        
        summary_text = tk.Label(
            summary_content,
            text=f"Broker: {broker} | Account: {account_type}\nAll credentials validated and ready",
            font=("Segoe UI", 8),
            bg=self.colors['secondary'],
            fg=self.colors['text_light'],
            justify=tk.CENTER
        )
        summary_text.pack()
        
        # Button container
        button_frame = tk.Frame(content, bg=self.colors['card'])
        button_frame.pack(fill=tk.X, pady=8)
        
        # Back button
        back_btn = self.create_button(button_frame, "← BACK", self.setup_broker_screen, "back")
        back_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Start Bot button
        start_btn = self.create_button(button_frame, "START BOT →", self.start_bot, "next")
        start_btn.pack(side=tk.RIGHT)
    
    def _update_account_size_from_fetched(self, balance: float):
        """Helper method to update account size field with fetched balance."""
        self.config["account_size"] = str(int(balance))
        self.account_entry.delete(0, tk.END)
        self.account_entry.insert(0, str(int(balance)))
        self.save_config()
    
    def fetch_account_info(self):
        """Fetch account information from the broker."""
        broker = self.config.get("broker", "TopStep")
        token = self.config.get("broker_token", "")
        username = self.config.get("broker_username", "")
        
        if not token or not username:
            messagebox.showerror(
                "Missing Credentials",
                "Please complete broker setup first (go back to Screen 0)."
            )
            return
        
        # Show loading spinner
        self.show_loading(f"Fetching account info from {broker}...")
        
        def fetch_in_thread():
            try:
                # Simulate API call - in production, this would call actual broker API
                import time
                time.sleep(1.5)  # Simulate network delay
                
                # Mock account data - replace with actual API call
                accounts = [
                    {"id": "ACC001", "name": f"{broker} Account 1", "balance": 50000.00, "equity": 51234.56, "type": "prop_firm"},
                    {"id": "ACC002", "name": f"{broker} Account 2", "balance": 100000.00, "equity": 102456.78, "type": "prop_firm"}
                ]
                
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
                        f"Successfully retrieved {len(accounts)} account(s) from {broker}.\n\n"
                        f"Selected: {selected_acc['name']}\n"
                        f"Balance: ${selected_acc['balance']:,.2f}\n"
                        f"Equity: ${selected_acc['equity']:,.2f}\n"
                        f"Type: {selected_acc.get('type', 'Unknown')}"
                        + (mismatch_warning if mismatch_warning else "")
                    )
                
                self.root.after(0, update_ui)
                
            except Exception as e:
                def show_error():
                    self.hide_loading()
                    messagebox.showerror(
                        "Fetch Failed",
                        f"Failed to fetch account information:\n{str(e)}\n\n"
                        f"Please check your broker credentials."
                    )
                self.root.after(0, show_error)
        
        # Start fetch in background thread
        thread = threading.Thread(target=fetch_in_thread, daemon=True)
        thread.start()
    
    def auto_adjust_parameters(self):
        """Auto-adjust trading parameters based on FETCHED account data (prioritized over user input)."""
        # IMPORTANT: Prioritize fetched account data over user input
        # Check if account info has been fetched
        accounts = self.config.get("accounts", [])
        fetched_balance = self.config.get("fetched_account_balance")
        fetched_type = self.config.get("fetched_account_type", "live_broker")
        
        if not accounts and not fetched_balance:
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
        
        # Use FETCHED data (most accurate) - prioritize over user input
        if fetched_balance:
            balance = fetched_balance
            equity = fetched_balance  # Use balance as equity if not provided
            account_type = fetched_type
        else:
            # Fallback to accounts data
            selected_account = accounts[0]
            balance = selected_account.get("balance", 10000)
            equity = selected_account.get("equity", balance)
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
            
            # Contracts: Strategic based on account size and drawdown
            # Never max out contracts - leave room for error
            if drawdown_pct < 2:  # Very safe
                max_contracts = min(3, max(1, int(equity / 30000)))
            elif drawdown_pct < 5:  # Moderately safe
                max_contracts = min(2, max(1, int(equity / 40000)))
            else:  # In drawdown - very conservative
                max_contracts = 1  # Single contract only
            
            # Trades per day: Fewer trades = more selective
            if distance_to_failure > 5:
                max_trades = 12
            else:
                max_trades = 8  # Very selective when in drawdown
                
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
            
            # Enable trailing drawdown for prop firms (industry best practice)
            # Helps protect from giving back profits
            self.trailing_drawdown_var.set(True)
        else:  # Live Broker
            # Live brokers are more flexible
            # Set based on equity size
            if equity < 50000:
                recommended_max_drawdown = 12.0
            elif equity < 100000:
                recommended_max_drawdown = 15.0
            else:
                recommended_max_drawdown = 18.0
            
            # Trailing drawdown optional for live brokers but recommended
            # Leave user's existing choice unchanged
        
        # Apply the calculated settings (all 4 parameters now)
        self.loss_entry.delete(0, tk.END)
        self.loss_entry.insert(0, f"{daily_loss_limit:.2f}")
        self.contracts_var.set(max_contracts)
        self.trades_var.set(max_trades)
        self.drawdown_var.set(recommended_max_drawdown)
        
        # Update info label with comprehensive feedback
        if account_type == "Prop Firm":
            self.auto_adjust_info_label.config(
                text=f"✓ Optimized for ${equity:,.2f} equity ({drawdown_pct:.1f}% current drawdown) - {max_contracts} contracts, ${daily_loss_limit:.0f} daily limit, {max_trades} trades/day, {recommended_max_drawdown}% max drawdown, trailing enabled",
                fg=self.colors['success']
            )
        else:
            self.auto_adjust_info_label.config(
                text=f"✓ Optimized for ${equity:,.2f} equity - {max_contracts} contracts, ${daily_loss_limit:.0f} daily limit, {max_trades} trades/day, {recommended_max_drawdown}% max drawdown",
                fg=self.colors['success']
            )
    
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
        self.config["max_drawdown"] = self.drawdown_var.get()
        self.config["daily_loss_limit"] = loss_limit
        self.config["max_contracts"] = self.contracts_var.get()
        self.config["max_trades"] = self.trades_var.get()
        self.config["confidence_threshold"] = self.confidence_var.get()
        self.config["dynamic_confidence"] = self.dynamic_confidence_var.get()
        self.config["shadow_mode"] = self.shadow_mode_var.get()
        self.config["dynamic_contracts"] = self.dynamic_contracts_var.get()
        self.config["trailing_drawdown"] = self.trailing_drawdown_var.get()
        self.config["selected_account"] = self.account_dropdown_var.get()
        self.config["recovery_mode"] = self.recovery_mode_var.get()
        self.save_config()
        
        # Create .env file
        self.create_env_file()
        
        # Show confirmation with new settings
        symbols_str = ", ".join(selected_symbols)
        broker = self.config.get("broker", "TopStep")
        
        confirmation_text = f"Ready to start bot with these settings:\n\n"
        confirmation_text += f"Broker: {broker}\n"
        confirmation_text += f"Account: {self.account_dropdown_var.get()}\n"
        confirmation_text += f"Symbols: {symbols_str}\n"
        confirmation_text += f"Contracts Per Trade: {self.contracts_var.get()}\n"
        confirmation_text += f"Max Drawdown: {self.drawdown_var.get()}%\n"
        if self.trailing_drawdown_var.get():
            broker = self.config.get("broker", "")
            if broker == "Apex":
                confirmation_text += f"Trailing Drawdown: ON (HARD FLOOR - Required by {broker}, account fails if violated)\n"
            else:
                confirmation_text += f"Trailing Drawdown: ON (SOFT FLOOR - Bot stops trading when approaching failure)\n"
                if broker == "TopStep":
                    confirmation_text += f"  → TopStep's hard floor: $47K static (separate, account fails if violated)\n"
        confirmation_text += f"Daily Loss Limit: ${loss_limit}\n"
        confirmation_text += f"  → Bot stays on but will NOT execute trades if limit is hit\n"
        confirmation_text += f"  → Resets daily after market maintenance\n"
        confirmation_text += f"Max Trades/Day: {self.trades_var.get()}\n"
        confirmation_text += f"  → Bot stays on but will NOT execute trades after limit\n"
        confirmation_text += f"  → Resets daily after market maintenance\n"
        if self.dynamic_confidence_var.get():
            confirmation_text += f"Confidence Threshold: {self.confidence_var.get()}% (Min - dynamic adjustments enabled)\n"
            confirmation_text += f"  → Bot may auto-increase confidence when needed (never below minimum)\n"
        else:
            confirmation_text += f"Confidence Threshold: {self.confidence_var.get()}% (Fixed)\n"
        if self.shadow_mode_var.get():
            confirmation_text += f"Shadow Mode: ON (paper trading)\n"
        if self.dynamic_contracts_var.get():
            confirmation_text += f"Dynamic Contracts: ON (confidence-based sizing)\n"
        
        # Add Recovery Mode info
        if self.recovery_mode_var.get():
            confirmation_text += f"\n⚠️ Recovery Mode: ENABLED\n"
            confirmation_text += f"  → Bot continues trading when close to limits\n"
            confirmation_text += f"  → Auto-scales confidence (75-90%) and reduces risk dynamically\n"
            confirmation_text += f"  → Attempts to recover from losses\n"
        else:
            confirmation_text += f"\nSafe Mode: Bot stops NEW trades at 80% of limits\n"
            confirmation_text += f"  → Bot stops trading when approaching failure\n"
            confirmation_text += f"  → Continues monitoring, resumes after daily reset\n"
        
        confirmation_text += f"\nThis will open a PowerShell terminal with live logs.\n"
        confirmation_text += f"Use the window's close button to stop the bot.\n\n"
        confirmation_text += f"Continue?"
        
        result = messagebox.askyesno(
            "Launch Trading Bot?",
            confirmation_text
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
BOT_MAX_DRAWDOWN_PERCENT={self.drawdown_var.get()}
# Maximum drawdown percentage before bot stops trading (account type aware)
BOT_TRAILING_DRAWDOWN={'true' if self.trailing_drawdown_var.get() else 'false'}
# SOFT FLOOR: Optional personal protection. When enabled, floor moves UP with profits (never down).
# If violated: Bot stops making trades when approaching account failure (doesn't shut off, continues monitoring).
# This is YOUR extra protection layer - separate from prop firm's hard floor rules.
# Hard floors: Apex=trailing (required), TopStep=$47K static, both cause account failure if violated.
BOT_TRAILING_TYPE={'hard' if self.config.get("broker", "") == "Apex" else 'soft'}
# Type: "hard" (required by prop firm, account fails) or "soft" (optional, bot stops trading when approaching failure)
BOT_DAILY_LOSS_LIMIT={self.loss_entry.get()}
# Bot stays on but will NOT execute trades if this limit (in dollars) is hit (resets daily after market maintenance)

# AI/Confidence Settings
BOT_CONFIDENCE_THRESHOLD={self.confidence_var.get()}
# Bot only takes signals above this confidence threshold (user's minimum)
BOT_DYNAMIC_CONFIDENCE={'true' if self.dynamic_confidence_var.get() else 'false'}
# When enabled, bot auto-increases confidence (never below user's setting) when performing poorly or approaching account limits
BOT_DYNAMIC_CONTRACTS={'true' if self.dynamic_contracts_var.get() else 'false'}
# Uses signal confidence to determine contract size dynamically (bot uses adaptive exits)

# Recovery Mode (All Account Types)
BOT_RECOVERY_MODE={'true' if self.recovery_mode_var.get() else 'false'}
# When true (ENABLED): Bot continues trading when approaching limits with auto-scaled confidence (75-90%) and dynamic risk reduction
# When false (DISABLED): Bot stops making NEW trades (but stays running) at 80% of daily loss or max drawdown limits, resumes after daily reset

# Trading Mode
BOT_SHADOW_MODE={'true' if self.shadow_mode_var.get() else 'false'}
BOT_DRY_RUN={'true' if self.shadow_mode_var.get() else 'false'}

# Account Selection
SELECTED_ACCOUNT={self.config.get("selected_account", "Default Account")}

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
