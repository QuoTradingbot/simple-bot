"""
QuoTrading AI - Customer Launcher
==================================
Professional GUI application for easy setup and launch.
2-Screen Progressive Onboarding Flow.

Flow:
1. Screen 0: Broker Setup (Broker credentials, QuoTrading API key, account size)
2. Screen 1: Trading Controls (Symbol selection, risk settings, launch)

HYBRID ARCHITECTURE:
- USE_CLOUD_SIGNALS = False: Local mode (testing, development)
- USE_CLOUD_SIGNALS = True: Production mode (cloud ML/RL brain)
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
import psutil  # For process checking (stale lock detection)
import hashlib  # For device fingerprint generation
import getpass  # For username in device fingerprint
import uuid  # For machine ID in device fingerprint

# ========================================
# CONFIGURATION
# ========================================

# Toggle between local and cloud signal generation
USE_CLOUD_SIGNALS = True  # Set to True for production (cloud ML/RL)

# Cloud API endpoints - Azure deployment
# The bot (src/quotrading_engine.py) handles all cloud ML/RL communication
CLOUD_API_BASE_URL = os.getenv("QUOTRADING_API_URL", "https://quotrading-flask-api.azurewebsites.net")
CLOUD_SIGNAL_POLL_INTERVAL = 5  # Seconds between signal polls

# Multi-symbol launch configuration
# Delay between launching each symbol to prevent session race conditions
MULTI_SYMBOL_LAUNCH_DELAY_SECONDS = 3


def get_device_fingerprint() -> str:
    """
    Generate a unique device fingerprint for session locking.
    One session per machine/user - shared between launcher and bot.
    
    Components:
    - Machine ID (MAC address via uuid.getnode)
    - Username (from getpass)
    - Platform name (Windows/Linux/Darwin)
    
    Returns:
        Unique device fingerprint (hashed for privacy)
    
    Security Note: Launcher creates session, bot reconnects to same session.
    This prevents multiple instances while allowing launcher → bot handoff.
    PID is NOT included so launcher and bot share the same fingerprint.
    """
    # Get platform-specific machine ID
    try:
        machine_id = str(uuid.getnode())  # MAC address as unique ID
    except:
        machine_id = "unknown"
    
    # Get username
    try:
        username = getpass.getuser()
    except:
        username = "unknown"
    
    # Get platform info
    platform_name = platform.system()  # Windows, Darwin (Mac), Linux
    
    # Combine all components WITHOUT PID - one session per machine/user, not per process
    # This allows launcher and bot to share the same session
    fingerprint_raw = f"{machine_id}:{username}:{platform_name}"
    
    # Hash for privacy (don't send raw MAC address to server)
    fingerprint_hash = hashlib.sha256(fingerprint_raw.encode()).hexdigest()[:16]
    
    return fingerprint_hash


def check_launcher_lock(api_key: str) -> tuple[bool, dict]:
    """
    Check if another launcher instance is using this API key on this machine.
    
    Returns:
        (is_locked, lock_info) - is_locked=True if another launcher is active
    """
    locks_dir = Path("locks")
    locks_dir.mkdir(exist_ok=True)
    
    # Hash the API key for filename (don't store raw key in filename)
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    lock_file = locks_dir / f"launcher_{key_hash}.lock"
    
    if not lock_file.exists():
        return False, {}
    
    try:
        with open(lock_file, 'r') as f:
            lock_data = json.load(f)
        
        # Check if the process is still running (stale lock detection)
        pid = lock_data.get("pid")
        if pid and psutil.pid_exists(pid):
            try:
                proc = psutil.Process(pid)
                # Check if it's actually a Python process (launcher)
                if proc.is_running() and 'python' in proc.name().lower():
                    return True, lock_data
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Stale lock - remove it
        lock_file.unlink()
        return False, {}
    except Exception:
        return False, {}


def create_launcher_lock(api_key: str) -> bool:
    """
    Create a lock file for this launcher instance.
    
    Returns:
        True if lock created successfully
    """
    locks_dir = Path("locks")
    locks_dir.mkdir(exist_ok=True)
    
    # Hash the API key for filename
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    lock_file = locks_dir / f"launcher_{key_hash}.lock"
    
    lock_data = {
        "api_key_hash": key_hash,
        "pid": os.getpid(),
        "created_at": datetime.now().isoformat(),
        "device_fingerprint": get_device_fingerprint()
    }
    
    try:
        with open(lock_file, 'w') as f:
            json.dump(lock_data, f, indent=2)
        return True
    except Exception:
        return False


def release_launcher_lock(api_key: str) -> bool:
    """
    Release the launcher lock for this API key.
    
    Returns:
        True if lock released successfully
    """
    locks_dir = Path("locks")
    if not locks_dir.exists():
        return True
    
    # Hash the API key for filename
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    lock_file = locks_dir / f"launcher_{key_hash}.lock"
    
    if lock_file.exists():
        try:
            # Only remove if it's our lock
            with open(lock_file, 'r') as f:
                lock_data = json.load(f)
            
            if lock_data.get("pid") == os.getpid():
                lock_file.unlink()
                return True
        except Exception:
            pass
    
    return True


class QuoTradingLauncher:
    """Professional GUI launcher for QuoTrading AI - Blue/White Theme with Cloud Authentication."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("QuoTrading - Professional Trading Platform")
        self.root.geometry("650x600")
        self.root.resizable(False, False)
        
        # Windows-style Gray color scheme - Professional theme with improved contrast
        self.light_theme_colors = {
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
            'text': '#1A1A1A',           # Almost black text (primary) - improved contrast
            'text_light': '#3A3A3A',     # Dark gray (secondary labels) - improved
            'text_secondary': '#5A5A5A', # Medium gray (tertiary/hints) - improved
            'border': '#0078D4',         # Windows blue border
            'border_subtle': '#BBBBBB',  # Gray subtle border
            'input_bg': '#FFFFFF',       # White for input fields
            'input_focus': '#E5F3FF',    # Light blue tint on focus
            'button_hover': '#0078D4',   # Windows blue for hover state
            'shadow': '#C0C0C0'          # Medium gray for shadow effect
        }
        
        # Dark theme colors - Modern black/blue with better contrast
        self.dark_theme_colors = {
            'primary': '#1E1E1E',        # Black background
            'secondary': '#252525',      # Slightly lighter black
            'success': '#0078D4',        # Windows blue accent
            'success_dark': '#0078D4',   # Same blue for buttons/headers
            'success_darker': '#005A9E', # Darker blue for depth
            'error': '#DC2626',          # Red for error messages
            'warning': '#F59E0B',        # Orange for warnings
            'background': '#1E1E1E',     # Black main background
            'card': '#2D2D2D',           # Dark gray card background
            'card_elevated': '#3A3A3A',  # Lighter gray for elevation
            'text': '#FFFFFF',           # White text (primary) - improved contrast
            'text_light': '#D0D0D0',     # Light gray (secondary labels) - improved
            'text_secondary': '#A0A0A0', # Medium gray (tertiary/hints) - improved
            'border': '#0078D4',         # Windows blue border
            'border_subtle': '#404040',  # Dark gray subtle border
            'input_bg': '#2D2D2D',       # Dark input fields
            'input_focus': '#1A3A52',    # Dark blue tint on focus
            'button_hover': '#0078D4',   # Windows blue for hover state
            'shadow': '#000000'          # Black for shadow effect
        }
        
        # Set initial colors (will be overridden by theme loading)
        self.colors = self.light_theme_colors.copy()
        
        # Default fallback symbol
        self.DEFAULT_SYMBOL = 'ES'
        
        self.root.configure(bg=self.colors['background'])
        
        # Load saved config
        self.config_file = Path("config.json")
        self.config = self.load_config()
        
        # Apply saved theme
        saved_theme = self.config.get("theme", "light")
        if saved_theme == "dark":
            self.colors = self.dark_theme_colors.copy()
        else:
            self.colors = self.light_theme_colors.copy()
        self.root.configure(bg=self.colors['background'])
        
        # Current screen tracker
        self.current_screen = 0
        
        # AI process reference
        self.bot_process = None  # Keep variable name for compatibility
        
        # Track current API key for lock management
        self.current_api_key = None
        
        # Countdown state (initialized here to avoid dynamic attribute creation)
        self.countdown_cancelled = False
        
        # License timer label (will be created when trading screen is shown)
        self.license_timer_label = None
        
        # Register cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start with broker screen (Screen 0)
        self.setup_broker_screen()
    
    def on_closing(self):
        """Handle launcher window closing."""
        # Release launcher lock if we have one
        if self.current_api_key:
            release_launcher_lock(self.current_api_key)
        
        # Destroy the window
        self.root.destroy()
    
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
    
    def validate_license_key(self, api_key, success_callback, error_callback):
        """Validate QuoTrading license key with cloud API.
        
        Args:
            api_key: The QuoTrading license key to validate
            success_callback: Function to call on successful validation (receives license data)
            error_callback: Function to call on validation failure (receives error message)
        """
        import requests
        import os
        
        def validate_in_thread():
            try:
                api_url = os.getenv("QUOTRADING_API_URL", "https://quotrading-flask-api.azurewebsites.net")
                
                # Call cloud API to validate license - server validates key, checks status and expiration
                # Include device fingerprint for session locking
                # Use /api/validate-license with check_only=True to check for active bot sessions
                response = requests.post(
                    f"{api_url}/api/validate-license",
                    json={
                        "license_key": api_key,
                        "device_fingerprint": get_device_fingerprint(),  # For session locking
                        "check_only": True  # Don't create session, just validate and check conflicts
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    license_data = response.json()
                    
                    # Check if license is actually valid
                    if not license_data.get("license_valid", False):
                        error_msg = license_data.get("message", "License validation failed")
                        self.root.after(0, lambda: error_callback(error_msg))
                        return
                    
                    # Get license type from response (now directly from API)
                    license_type = license_data.get("license_type", "standard")
                    
                    # Get expiry date and calculate days/hours until expiration
                    expiry_date_str = license_data.get("expiry_date")
                    days_until_expiration = None
                    hours_until_expiration = None
                    
                    if expiry_date_str:
                        try:
                            expiry_date = datetime.fromisoformat(expiry_date_str.replace('Z', '+00:00'))
                            time_until_expiration = expiry_date - datetime.now(expiry_date.tzinfo)
                            days_until_expiration = time_until_expiration.days
                            hours_until_expiration = time_until_expiration.total_seconds() / 3600
                        except Exception:
                            pass
                    
                    # Server-side subscription configuration (no hardcoded limits)
                    # These values come from license_type in database
                    self.config["license_type"] = license_type
                    self.config["license_expiration"] = expiry_date_str
                    self.config["days_until_expiration"] = days_until_expiration
                    self.config["hours_until_expiration"] = hours_until_expiration
                    self.save_config()
                    
                    self.root.after(0, lambda: success_callback(license_data))
                    
                elif response.status_code == 403:
                    # Check if it's a session conflict
                    error_data = response.json()
                    if error_data.get("session_conflict"):
                        # Session conflict - bot is already running with this API key
                        # Get the server's detailed message
                        server_message = error_data.get("message", "License already in use")
                        estimated_wait = error_data.get("estimated_wait_seconds", 60)
                        
                        error_msg = (
                            "⚠️ API KEY ALREADY IN USE ⚠️\n\n"
                            f"{server_message}\n\n"
                            "Only ONE instance can run per API key at a time.\n\n"
                        )
                        
                        if estimated_wait > 0:
                            error_msg += (
                                f"Please wait approximately {estimated_wait} seconds\n"
                                "for the current session to expire, or stop the\n"
                                "running instance manually.\n\n"
                            )
                        
                        error_msg += (
                            "If you recently stopped the bot and can't login,\n"
                            "please wait up to 60 seconds for the session to clear."
                        )
                    else:
                        error_msg = error_data.get("message", "License validation failed - Access Forbidden")
                    self.root.after(0, lambda: error_callback(error_msg))
                    
                else:
                    error_data = response.json()
                    error_msg = error_data.get("message", "License validation failed")
                    self.root.after(0, lambda: error_callback(error_msg))
            
            except requests.exceptions.Timeout:
                self.root.after(0, lambda: error_callback("Connection timeout - please check your internet connection"))
            except requests.exceptions.ConnectionError:
                self.root.after(0, lambda: error_callback("Cannot connect to QuoTrading servers - please check your internet"))
            except Exception as e:
                self.root.after(0, lambda: error_callback(f"API error: {str(e)}"))
        
        # Start validation in background thread
        thread = threading.Thread(target=validate_in_thread, daemon=True)
        thread.start()
    
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
        
        # QuoTrading License Key
        self.quotrading_api_key_entry = self.create_input_field(
            content,
            "QuoTrading License Key:",
            is_password=True,
            placeholder=""
        )
        # Load saved QuoTrading license key if exists
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
        """Validate broker credentials by connecting to API and pre-loading accounts."""
        broker = self.broker_var.get()
        token = self.broker_token_entry.get().strip()
        username = self.broker_username_entry.get().strip()
        quotrading_api_key = self.quotrading_api_key_entry.get().strip()
        
        # Get account size - will be replaced by real data from broker anyway
        account_size = "50000"  # Default starting point, real data will override
        
        # Validation
        if not token or not username:
            messagebox.showerror(
                "Missing Credentials",
                f"Please enter both {broker} API Token and Username."
            )
            return
        
        # Require QuoTrading license key
        if not quotrading_api_key:
            messagebox.showerror(
                "Missing License Key",
                "Please enter your QuoTrading License Key."
            )
            return
        
        # CHECK LAUNCHER LOCK - Prevent multiple launchers with same API key on this machine
        is_locked, lock_info = check_launcher_lock(quotrading_api_key)
        if is_locked:
            pid = lock_info.get("pid", "unknown")
            created_at = lock_info.get("created_at", "unknown time")
            messagebox.showerror(
                "License Already in Use",
                f"⚠️ ANOTHER LAUNCHER INSTANCE IS ALREADY RUNNING ⚠️\n\n"
                f"This license key is currently in use by another\n"
                f"launcher instance on this computer.\n\n"
                f"Process ID: {pid}\n"
                f"Started: {created_at}\n\n"
                f"You can only run ONE launcher instance per license key.\n\n"
                f"To fix this:\n"
                f"1. Close the other launcher window\n"
                f"2. Or use a different license key"
            )
            return
        
        # Validate license key with Azure server (server will check if it's admin or regular)
        self.show_loading("Validating license...")
        
        def on_license_success(license_data):
            self.hide_loading()
            # Create launcher lock after successful validation
            if create_launcher_lock(quotrading_api_key):
                self.current_api_key = quotrading_api_key
            # License valid - proceed with broker validation
            self._continue_broker_validation(broker, token, username, quotrading_api_key, account_size)
        
        def on_license_error(error_msg):
            self.hide_loading()
            messagebox.showerror(
                "License Validation Failed",
                f"Invalid license key: {error_msg}\n\n"
                f"Please check your license key and try again.\n\n"
                f"If you need a license, visit:\n"
                f"https://quotrading.com/subscribe"
            )
        
        # Validate license with Azure (server handles admin vs regular keys)
        self.validate_license_key(quotrading_api_key, on_license_success, on_license_error)
    
    
    def _continue_broker_validation(self, broker, token, username, quotrading_api_key, account_size):
        """Continue broker validation after license is confirmed valid."""
        # Validate by actually connecting and fetching accounts
        def validate_in_thread():
            import traceback
            try:
                # Import broker interface
                import sys
                from pathlib import Path
                src_path = Path(__file__).parent.parent / "src"
                if str(src_path) not in sys.path:
                    sys.path.insert(0, str(src_path))
                
                accounts = []
                
                # Connect to broker API
                if broker == "TopStep":
                    from broker_interface import TopStepBroker
                    
                    ts_broker = TopStepBroker(api_token=token, username=username)
                    connected = ts_broker.connect()
                    
                    if connected:
                        # Get account info from SDK client property (list_accounts has bugs)
                        # The account_info property is populated automatically after connect()
                        try:
                            account_info = ts_broker.sdk_client.account_info
                            
                            if account_info:
                                # Extract real account details from TopStep
                                # Use the account name (e.g., "50KTC-V2-398684-33989413") as the display ID
                                account_name = getattr(account_info, 'name', f'TopStep Account')
                                account_id = account_name  # Use name as ID for display
                                internal_id = str(getattr(account_info, 'id', 'TOPSTEP_MAIN'))  # Keep numeric ID for storage
                                current_balance = float(getattr(account_info, 'balance', 0))
                                is_simulated = getattr(account_info, 'simulated', True)
                                acc_type = 'prop_firm' if is_simulated else 'live'
                                
                                # Check if we have a stored starting balance for this account (use internal ID)
                                stored_balances = self.config.get("topstep_starting_balances", {})
                                if internal_id in stored_balances:
                                    starting_balance = stored_balances[internal_id]
                                else:
                                    # First time seeing this account - store its starting balance
                                    starting_balance = current_balance
                                    if "topstep_starting_balances" not in self.config:
                                        self.config["topstep_starting_balances"] = {}
                                    self.config["topstep_starting_balances"][internal_id] = starting_balance
                                
                                # Always show current balance in GUI (not cached starting balance)
                                accounts = [{
                                    "id": account_id,  # Display name
                                    "name": account_name,
                                    "balance": current_balance,  # Show current balance
                                    "equity": current_balance,   # Equity is current balance
                                    "starting_balance": starting_balance,  # Track starting for P&L
                                    "type": acc_type
                                }]
                            else:
                                raise Exception("No account info available")
                        except Exception as e:
                            # Fallback if account_info is not available
                            current_equity = ts_broker.get_account_equity()
                            stored_starting_balance = self.config.get("topstep_starting_balance")
                            if stored_starting_balance:
                                starting_balance = stored_starting_balance
                            else:
                                starting_balance = current_equity
                                self.config["topstep_starting_balance"] = starting_balance
                            
                            # Always show current balance in GUI (not cached starting balance)
                            accounts = [{
                                "id": "TOPSTEP_MAIN",
                                "name": f"TopStep Account ({username})",
                                "balance": current_equity,  # Show current balance
                                "equity": current_equity,   # Equity is current balance
                                "starting_balance": starting_balance,  # Track starting for P&L
                                "type": "prop_firm"
                            }]
                        
                        ts_broker.disconnect()
                    else:
                        # Connection failed - always require valid broker connection
                        raise Exception("Failed to connect to TopStep API. Check your API token and username.")
                
                elif broker == "Tradovate":
                    raise Exception("Tradovate API integration coming soon. Please use TopStep for now.")
                else:
                    raise Exception(f"Unsupported broker: {broker}")
                
                if not accounts:
                    raise Exception("No accounts retrieved from broker API")
                
                # Success - update UI on main thread
                def on_success_ui():
                    self.hide_loading()
                    
                    # Validate account was fetched successfully
                    actual_current_balance = accounts[0]['balance']
                    
                    # Save config
                    self.config["broker_type"] = self.broker_type_var.get()
                    self.config["broker"] = broker
                    self.config["account_size"] = str(int(actual_current_balance))  # Use current balance from broker
                    self.config["broker_validated"] = True
                    self.config["accounts"] = accounts
                    self.config["fetched_account_balance"] = accounts[0]['balance']
                    self.config["fetched_account_type"] = accounts[0].get('type', 'live_broker')
                    
                    if self.remember_credentials_var.get():
                        self.config["broker_token"] = token
                        self.config["broker_username"] = username
                        self.config["quotrading_api_key"] = quotrading_api_key
                        self.config["remember_credentials"] = True
                    else:
                        self.config["broker_token"] = ""
                        self.config["broker_username"] = ""
                        self.config["quotrading_api_key"] = ""
                        self.config["remember_credentials"] = False
                    
                    self.save_config()
                    
                    
                    # Proceed to trading screen with accounts pre-loaded
                    self.setup_trading_screen()
                
                self.root.after(0, on_success_ui)
                
            except Exception as error:
                error_msg = str(error)
                error_traceback = traceback.format_exc()
                
                def on_error_ui():
                    self.hide_loading()
                    messagebox.showerror(
                        "Login Failed",
                        f"❌ Failed to connect to {broker}:\n\n{error_msg}\n\n"
                        f"Please check your credentials:\n"
                        f"• Valid API token from your {broker} account\n"
                        f"• Correct username/email\n"
                        f"• Active account status\n\n"
                        f"Contact support if the issue persists."
                    )
                
                self.root.after(0, on_error_ui)
        
        # Show loading spinner
        self.show_loading(f"Connecting to {broker} API...")
        self.root.update_idletasks()  # Non-blocking UI update
        
        # Start validation in background thread
        thread = threading.Thread(target=validate_in_thread, daemon=True)
        thread.start()
    
    
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
        
        # Account Fetch Section - COMPACT HORIZONTAL STYLE with main background
        fetch_frame = tk.Frame(content, bg=self.colors['background'])
        fetch_frame.pack(fill=tk.X, pady=(0, 4))
        
        # Left side: Account dropdown
        account_select_frame = tk.Frame(fetch_frame, bg=self.colors['background'])
        account_select_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        tk.Label(
            account_select_frame,
            text="Select Account:",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['background'],
            fg=self.colors['text']
        ).pack(anchor=tk.W)
        
        # Check if accounts were already fetched during login
        pre_loaded_accounts = self.config.get("accounts", [])
        if pre_loaded_accounts:
            # Show account ID with exact balance (not rounded)
            account_names = [f"{acc['id']} - ${acc['balance']:,.2f}" for acc in pre_loaded_accounts]
            default_value = account_names[0]
        else:
            account_names = ["No accounts loaded"]
            default_value = account_names[0]
        
        self.account_dropdown_var = tk.StringVar(value=default_value)
        self.account_dropdown = ttk.Combobox(
            account_select_frame,
            textvariable=self.account_dropdown_var,
            state="readonly",
            font=("Segoe UI", 9),
            width=20,
            values=account_names
        )
        self.account_dropdown.pack(fill=tk.X)
        self.account_dropdown.bind("<<ComboboxSelected>>", self.on_account_selected)
        
        # Middle: Ping button with label
        fetch_button_frame = tk.Frame(fetch_frame, bg=self.colors['background'])
        fetch_button_frame.pack(side=tk.LEFT, padx=5)
        
        sync_label = tk.Label(
            fetch_button_frame,
            text="🔄 Test Connection:",
            font=("Segoe UI", 8, "bold"),
            bg=self.colors['background'],
            fg=self.colors['success']
        )
        sync_label.pack(anchor=tk.W)
        
        # Button to ping RL server
        button_text = "Ping Server"
        fetch_btn = self.create_button(fetch_button_frame, button_text, self.fetch_account_info, "next")
        fetch_btn.pack()
        
        # Right: Auto-adjust button
        auto_adjust_frame = tk.Frame(fetch_frame, bg=self.colors['background'])
        auto_adjust_frame.pack(side=tk.LEFT, padx=5)
        
        tk.Label(
            auto_adjust_frame,
            text="Quick Setup:",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['background'],
            fg=self.colors['text']
        ).pack(anchor=tk.W)
        
        auto_adjust_btn = self.create_button(auto_adjust_frame, "Auto Configure", self.auto_adjust_parameters, "next")
        auto_adjust_btn.pack()
        
        # Info label below everything
        if pre_loaded_accounts:
            selected_acc = pre_loaded_accounts[0]
            info_text = f"✓ Balance: ${selected_acc['balance']:,.2f} | Equity: ${selected_acc['equity']:,.2f} | Type: {selected_acc.get('type', 'Unknown')}"
            info_color = self.colors['success']
        else:
            info_text = "Helps AI auto-configure optimal settings for your account"
            info_color = self.colors['text_light']
        
        self.account_info_label = tk.Label(
            content,
            text=info_text,
            font=("Segoe UI", 7, "bold"),
            bg=self.colors['card'],
            fg=info_color,
            anchor=tk.W
        )
        self.account_info_label.pack(anchor=tk.W, pady=(0, 4))
        
        self.auto_adjust_info_label = self.account_info_label  # Reuse same label
        
        # ========================================
        # SYMBOLS AND MODES SIDE BY SIDE
        # ========================================
        symbols_modes_container = tk.Frame(content, bg=self.colors['card'])
        symbols_modes_container.pack(fill=tk.X, pady=(0, 4))
        
        # Left side - Symbol Selection
        symbol_section = tk.Frame(symbols_modes_container, bg=self.colors['card'])
        symbol_section.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        symbol_label = tk.Label(
            symbol_section,
            text="Trading Symbols (select at least one):",
            font=("Segoe UI", 8, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        symbol_label.pack(anchor=tk.W, pady=(0, 2))
        
        # Symbol checkboxes - 2 columns
        symbol_frame = tk.Frame(symbol_section, bg=self.colors['card'])
        symbol_frame.pack(fill=tk.X, pady=(0, 0))
        
        self.symbol_vars = {}
        # Primary trading symbols in GUI (ES, MES, MNQ, NQ)
        # Other symbols (YM, RTY, CL, GC, NG, 6E, ZN, MBTX) are kept in experiences/ for future use
        symbols = [
            ("ES", "E-mini S&P 500"),
            ("MES", "Micro E-mini S&P 500"),
            ("MNQ", "Micro E-mini Nasdaq 100"),
            ("NQ", "E-mini Nasdaq 100")
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
                font=("Segoe UI", 8, "bold"),
                bg=self.colors['card'],
                fg=self.colors['text'],
                selectcolor=self.colors['secondary'],
                activebackground=self.colors['card'],
                activeforeground=self.colors['success'],
                cursor="hand2"
            )
            cb.grid(row=row, column=col, sticky=tk.W, padx=4, pady=1)
        
        # Right side - Trading Modes
        modes_section = tk.Frame(symbols_modes_container, bg=self.colors['card'])
        modes_section.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Shadow Mode
        shadow_mode_frame = tk.Frame(modes_section, bg=self.colors['card'])
        shadow_mode_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.shadow_mode_var = tk.BooleanVar(value=self.config.get("shadow_mode", False))
        tk.Checkbutton(
            shadow_mode_frame,
            text="👁 Shadow Mode",
            variable=self.shadow_mode_var,
            font=("Segoe UI", 8, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text'],
            selectcolor=self.colors['secondary'],
            activebackground=self.colors['card'],
            activeforeground=self.colors['success'],
            cursor="hand2"
        ).pack(anchor=tk.W)
        
        tk.Label(
            shadow_mode_frame,
            text="View signals without auto-trading",
            font=("Segoe UI", 7, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text_light']
        ).pack(anchor=tk.W, padx=(20, 0))
        
        
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
            highlightcolor=self.colors['success'],
            readonlybackground=self.colors['input_bg'],  # Match theme in readonly state
            state='readonly'  # Make account size field readonly - shows selected account balance
        )
        self.account_entry.pack(fill=tk.X, ipady=2, padx=2)
        # Use textvariable for readonly field - format with cents
        account_size_val = self.config.get("account_size", "10000")
        try:
            formatted_account = f"{float(account_size_val):.2f}"
        except (ValueError, TypeError):
            formatted_account = "10000.00"
        self.account_var = tk.StringVar(value=formatted_account)
        self.account_entry.configure(textvariable=self.account_var)
        
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
        
        # Second Settings Row for Max Loss Per Trade
        settings_row_2 = tk.Frame(content, bg=self.colors['card'])
        settings_row_2.pack(fill=tk.X, pady=(3, 3))
        
        # Max Loss Per Trade
        max_loss_trade_frame = tk.Frame(settings_row_2, bg=self.colors['card'])
        max_loss_trade_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        
        tk.Label(
            max_loss_trade_frame,
            text="Max Loss Per Trade ($):",
            font=("Segoe UI", 7, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 1))
        
        self.max_loss_per_trade_entry = tk.Entry(
            max_loss_trade_frame,
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
        self.max_loss_per_trade_entry.pack(fill=tk.X, ipady=2, padx=2)
        # Default value matches DEFAULT_MAX_STOP_LOSS_DOLLARS in src/config.py
        self.max_loss_per_trade_entry.insert(0, self.config.get("max_loss_per_trade", "200"))
        
        # Info label for max loss per trade
        tk.Label(
            max_loss_trade_frame,
            text="Position closes if trade loses this amount",
            font=("Segoe UI", 6),
            bg=self.colors['card'],
            fg=self.colors['text_light']
        ).pack(anchor=tk.W, pady=(1, 0))
        
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
        # Max contracts - user configurable, no account type restrictions
        self.max_contracts_allowed = 25  # General limit for safety
        
        self.contracts_var = tk.IntVar(value=min(self.config.get("max_contracts", 3), self.max_contracts_allowed))
        
        contracts_spin = ttk.Spinbox(
            contracts_frame,
            from_=1,
            to=self.max_contracts_allowed,
            textvariable=self.contracts_var,
            width=12
        )
        contracts_spin.pack(fill=tk.X, ipady=2)
        
        # Info label
        contracts_info = tk.Label(
            contracts_frame,
            text=f"Max {self.max_contracts_allowed} contracts (safety limit)",
            font=("Segoe UI", 7, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text_light']
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
        confidence_section.pack(fill=tk.X, pady=(0, 0))
        
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
            font=("Segoe UI", 7, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text_light']
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
        
        # Bottom row with Settings icon and Launch button
        bottom_row = tk.Frame(summary_frame, bg=self.colors['card'])
        bottom_row.pack(fill=tk.X)
        
        # Settings icon (far left) with label
        settings_container = tk.Frame(bottom_row, bg=self.colors['card'])
        settings_container.pack(side=tk.LEFT, anchor=tk.W, padx=(5, 0))
        
        settings_btn = tk.Button(
            settings_container,
            text="⚙️",
            command=self.show_settings_dialog,
            font=("Segoe UI", 14),
            bg=self.colors['card'],
            fg=self.colors['text'],
            cursor="hand2",
            relief=tk.FLAT,
            bd=0,
            padx=8,
            pady=2
        )
        settings_btn.pack()
        
        tk.Label(
            settings_container,
            text="Settings",
            font=("Segoe UI", 7, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text_light']
        ).pack()
        
        # Launch button (center)
        launch_container = tk.Frame(bottom_row, bg=self.colors['card'])
        launch_container.pack(expand=True, padx=(0, 60))
        launch_btn = self.create_button(launch_container, "LAUNCH AI", self.start_bot, "next")
        launch_btn.pack(pady=2, ipady=3)
        
        # License expiration timer (bottom right)
        timer_container = tk.Frame(bottom_row, bg=self.colors['card'])
        timer_container.pack(side=tk.RIGHT, anchor=tk.E, padx=(0, 5))
        
        tk.Label(
            timer_container,
            text="⏱️ License Expires:",
            font=("Segoe UI", 7, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text_light']
        ).pack()
        
        self.license_timer_label = tk.Label(
            timer_container,
            text="Loading...",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card'],
            fg=self.colors['success']
        )
        self.license_timer_label.pack()
        
        # Start the license timer update
        self.update_license_timer()
    
    def on_account_selected(self, event=None):
        """Update account size field when user selects a different account from dropdown."""
        selected_display = self.account_dropdown_var.get()
        accounts = self.config.get("accounts", [])
        
        if not accounts or "No accounts" in selected_display:
            return
        
        # Parse account ID from display format: "AccountID - $50,000.00"
        try:
            # Extract account ID (everything before " - $")
            account_id = selected_display.split(' - $')[0].strip()
            
            # Find account by ID
            selected_account = next((acc for acc in accounts if acc['id'] == account_id), None)
            
            if selected_account:
                # Update account size field with selected account's balance (show cents)
                balance = selected_account['balance']
                try:
                    self.account_var.set(f"{float(balance):.2f}")
                except (ValueError, TypeError):
                    self.account_var.set("10000.00")
                
                # Update info label
                info_text = f"✓ {selected_account['id']} | Balance: ${selected_account['balance']:,.2f}"
                self.account_info_label.config(text=info_text, fg=self.colors['success'])
        except Exception as e:
            pass
    
    def update_license_timer(self):
        """Update the license expiration timer display."""
        # Check if timer label exists and is still valid
        if not self.license_timer_label or not self.license_timer_label.winfo_exists():
            return  # Timer label not created or already destroyed
        
        # Get license expiration from config
        license_expiration = self.config.get("license_expiration")
        
        if not license_expiration:
            self.license_timer_label.config(text="No expiration data", fg=self.colors['text_secondary'])
            return
        
        try:
            # Parse the expiration datetime (format: "2025-12-31T23:59:59" or similar)
            # Handle different possible formats
            if isinstance(license_expiration, str):
                # Try ISO format first
                try:
                    expiration_dt = datetime.fromisoformat(license_expiration.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    # Try other common formats
                    try:
                        expiration_dt = datetime.strptime(license_expiration, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        expiration_dt = datetime.strptime(license_expiration, "%Y-%m-%dT%H:%M:%S")
            else:
                # Already a datetime object
                expiration_dt = license_expiration
            
            # Calculate time remaining
            now = datetime.now()
            time_remaining = expiration_dt - now
            
            # Check if expired
            if time_remaining.total_seconds() <= 0:
                self.license_timer_label.config(text="EXPIRED", fg=self.colors['error'])
                return
            
            # Calculate days, hours, minutes, seconds
            total_seconds = int(time_remaining.total_seconds())
            days = total_seconds // 86400
            hours = (total_seconds % 86400) // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            
            # Format the display
            if days > 0:
                timer_text = f"{days}d {hours}h {minutes}m {seconds}s"
            elif hours > 0:
                timer_text = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                timer_text = f"{minutes}m {seconds}s"
            else:
                timer_text = f"{seconds}s"
            
            # Color based on time remaining
            if days > 7:
                color = self.colors['success']  # Green - plenty of time
            elif days > 1:
                color = self.colors['warning']  # Orange - getting close
            else:
                color = self.colors['error']  # Red - expiring soon
            
            self.license_timer_label.config(text=timer_text, fg=color)
            
        except (ValueError, AttributeError, TypeError) as e:
            self.license_timer_label.config(text="Error", fg=self.colors['error'])
        
        # Schedule next update in 1 second, but only if we're still on trading screen
        # Check if label still exists and current_screen is still 1 (trading screen)
        if (self.license_timer_label and 
            self.license_timer_label.winfo_exists() and 
            self.current_screen == 1):
            self.root.after(1000, self.update_license_timer)
    
    
    def fetch_account_info(self):
        """Ping server to verify connectivity."""
        
        # Show loading spinner
        self.show_loading("Testing server connection...")
        
        def test_connection_thread():
            """Ping server to test connectivity."""
            import traceback
            try:
                import requests
                
                # Get server URL
                server_url = CLOUD_API_BASE_URL
                health_endpoint = f"{server_url}/health"
                
                # Ping with timeout
                response = requests.get(health_endpoint, timeout=10)
                
                if response.status_code == 200:
                    
                    # Get response data if available
                    try:
                        data = response.json()
                        status = data.get("status", "online")
                        version = data.get("version", "unknown")
                    except:
                        status = "online"
                        version = "unknown"
                    
                    # Update UI on main thread
                    def show_success():
                        self.hide_loading()
                        
                        # Update info label to show connection success
                        self.account_info_label.config(
                            text=f"✓ Server ping successful! Connection is working.",
                            fg=self.colors['success']
                        )
                        
                        messagebox.showinfo(
                            "Ping Successful",
                            f"✓ Connection Test Passed!\n\n"
                            f"Server: QuoTrading Server\n"
                            f"URL: {server_url}\n"
                            f"Status: {status.upper()}\n"
                            f"Version: {version}\n\n"
                            f"Your connection to the server is working properly."
                        )
                    
                    self.root.after(0, show_success)
                else:
                    raise Exception(f"Server returned status code {response.status_code}")
                
            except requests.exceptions.Timeout:
                error_msg = "Timeout - Server not responding"
                
                def show_error():
                    self.hide_loading()
                    messagebox.showerror(
                        "Ping Failed",
                        f"❌ Connection Failed\n\n{error_msg}\n\n"
                        f"Server: {server_url}\n\n"
                        f"Please check:\n"
                        f"• Internet connection\n"
                        f"• Firewall settings\n"
                        f"• Server status\n\n"
                        f"Contact support if issue persists."
                    )
                
                self.root.after(0, show_error)
                
            except requests.exceptions.ConnectionError:
                error_msg = "Cannot connect - network error"
                
                def show_error():
                    self.hide_loading()
                    messagebox.showerror(
                        "Ping Failed",
                        f"❌ Connection Failed\n\n{error_msg}\n\n"
                        f"Server: {server_url}\n\n"
                        f"Please check:\n"
                        f"• Internet connection\n"
                        f"• Firewall settings\n"
                        f"• Server status\n\n"
                        f"Contact support if issue persists."
                    )
                
                self.root.after(0, show_error)
                
            except Exception as error:
                error_msg = str(error)
                
                def show_error():
                    self.hide_loading()
                    messagebox.showerror(
                        "Ping Failed",
                        f"❌ Connection Failed\n\n{error_msg}\n\n"
                        f"Server: {server_url}\n\n"
                        f"Contact support if issue persists."
                    )
                
                self.root.after(0, show_error)
        
        # Start ping in background thread
        thread = threading.Thread(target=test_connection_thread, daemon=True)
        thread.start()
    
    def auto_adjust_parameters(self):
        """Universal smart auto-configure for ALL account types.
        
        Uses intelligent math that works for prop firms and live brokers alike.
        Max loss per trade values are rounded to nearest $50 for clean numbers.
        """
        # Get account size from user input or selected account
        try:
            account_size = float(self.account_var.get() or "10000")
        except ValueError:
            messagebox.showwarning(
                "Invalid Account Size",
                "Please enter a valid account size before using Auto Configure."
            )
            return
        
        if account_size <= 0:
            messagebox.showwarning(
                "Invalid Account Size",
                "Account size must be greater than 0."
            )
            return
        
        # UNIVERSAL SMART FORMULA - Works for ALL account types
        # Key principle: Simple percentage-based with clean $50 increments
        
        # Max loss per trade: Based on account size, rounded to nearest $50
        # This is the main risk parameter - everything else scales from this
        
        if account_size <= 10000:
            # Small accounts: $50-$100 per trade
            max_loss_per_trade = 50
        elif account_size <= 25000:
            # Growing accounts: $150 per trade
            max_loss_per_trade = 150
        elif account_size <= 50000:
            # Medium accounts: $300 per trade
            max_loss_per_trade = 300
        elif account_size <= 100000:
            # Large accounts: $450 per trade
            max_loss_per_trade = 450
        elif account_size <= 150000:
            # Very large accounts: $600 per trade
            max_loss_per_trade = 600
        else:
            # Huge accounts: Cap at $750 per trade
            max_loss_per_trade = 750
        
        # DAILY LOSS LIMIT: Based on max_loss_per_trade with buffer
        # Allow 3-4 losing trades before hitting daily limit
        # This gives reasonable protection without being too restrictive
        TRADES_BEFORE_DAILY_LIMIT = 3.5
        daily_loss_limit = max_loss_per_trade * TRADES_BEFORE_DAILY_LIMIT
        
        # Round daily limit to nearest $50 for clean numbers
        daily_loss_limit = round(daily_loss_limit / 50) * 50
        
        # MAX CONTRACTS: Conservative scaling based on max loss per trade
        # 1 contract per $300 of max loss per trade
        DOLLARS_PER_CONTRACT = 300
        max_contracts = min(self.max_contracts_allowed, max(1, int(max_loss_per_trade / DOLLARS_PER_CONTRACT)))
        
        # MAX TRADES PER DAY: Based on account size
        # Smaller accounts = fewer trades
        if account_size <= 25000:
            max_trades = 4
        elif account_size <= 50000:
            max_trades = 5
        elif account_size <= 100000:
            max_trades = 6
        else:
            max_trades = 7
        
        # Apply the calculated settings
        self.loss_entry.delete(0, tk.END)
        self.loss_entry.insert(0, f"{daily_loss_limit:.0f}")
        
        self.max_loss_per_trade_entry.delete(0, tk.END)
        self.max_loss_per_trade_entry.insert(0, f"{max_loss_per_trade:.0f}")
        
        self.contracts_var.set(max_contracts)
        self.trades_var.set(max_trades)
        
        # Update info label with universal feedback
        loss_trades_buffer = daily_loss_limit / max_loss_per_trade if max_loss_per_trade > 0 else 0
        self.auto_adjust_info_label.config(
            text=f"✓ Smart config: {max_contracts} contracts • ${max_loss_per_trade:.0f}/trade • {max_trades} trades/day • ${daily_loss_limit:.0f} daily limit • {loss_trades_buffer:.1f}x buffer",
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
            account_size = float(self.account_var.get())
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
        
        # Validate max loss per trade
        try:
            max_loss_per_trade = float(self.max_loss_per_trade_entry.get())
            if max_loss_per_trade <= 0:
                raise ValueError("Max loss per trade must be greater than 0")
        except ValueError as e:
            messagebox.showerror(
                "Invalid Max Loss Per Trade",
                "Please enter a valid numeric value greater than 0 for max loss per trade."
            )
            return
        
        # Validate that daily loss limit is at least as large as max loss per trade
        if loss_limit < max_loss_per_trade:
            messagebox.showerror(
                "Invalid Risk Settings",
                f"Daily Loss Limit (${loss_limit:.0f}) must be greater than or equal to Max Loss Per Trade (${max_loss_per_trade:.0f}).\n\n"
                f"Your daily limit should allow for at least one full trade at your max loss per trade setting.\n\n"
                f"Please increase Daily Loss Limit to at least ${max_loss_per_trade:.0f} or decrease Max Loss Per Trade."
            )
            return
        
        # Save all settings
        self.config["symbols"] = selected_symbols
        self.config["account_size"] = account_size
        self.config["daily_loss_limit"] = loss_limit
        self.config["max_loss_per_trade"] = max_loss_per_trade
        self.config["max_contracts"] = self.contracts_var.get()
        self.config["max_trades"] = self.trades_var.get()
        self.config["confidence_threshold"] = self.confidence_var.get()
        self.config["shadow_mode"] = self.shadow_mode_var.get()
        self.config["selected_account"] = self.account_dropdown_var.get()
        
        # Get selected account ID - parse from dropdown display format
        selected_display = self.account_dropdown_var.get()
        selected_account_id = None
        accounts = self.config.get("accounts", [])
        
        # Parse account ID from display format: "AccountID - $50,000.00"
        try:
            account_id = selected_display.split(' - $')[0].strip()
            
            # Find matching account
            for acc in accounts:
                if acc['id'] == account_id:
                    selected_account_id = acc.get("id")
                    self.config["selected_account_id"] = selected_account_id
                    break
        except:
            pass
        
        # If no match and we have accounts, use the first one
        if not selected_account_id and accounts:
            selected_account_id = accounts[0].get("id")
            self.config["selected_account_id"] = selected_account_id
        
        # CHECK INSTANCE LOCK - Prevent duplicate trading on same account
        if selected_account_id:
            is_locked, lock_info = self.check_account_lock(selected_account_id)
            if is_locked:
                broker_username = lock_info.get("broker_username", "unknown")
                created_at = lock_info.get("created_at", "unknown time")
                # Use the display string from dropdown for account name
                selected_account_display = self.account_dropdown_var.get()
                messagebox.showerror(
                    "Account Already Trading",
                    f"❌ This account is already being traded!\n\n"
                    f"Account: {selected_account_display}\n"
                    f"Broker: {broker_username}\n"
                    f"Started: {created_at}\n\n"
                    f"You cannot run multiple bots on the same trading account.\n"
                    f"This prevents duplicate orders and position conflicts.\n\n"
                    f"To trade this account:\n"
                    f"1. Stop the other bot instance first\n"
                    f"2. Or select a different account"
                )
                return
        
        # Auto-enable alerts if email is configured
        self.config["alerts_enabled"] = bool(self.config.get("alert_email") and self.config.get("alert_email_password"))
        
        self.save_config()
        
        # Create .env file
        self.create_env_file()
        
        # Show countdown dialog before launching (no confirmation popup needed)
        self.show_countdown_and_launch(selected_symbols, selected_account_id, loss_limit)
    
    def show_countdown_and_launch(self, selected_symbols, selected_account_id, loss_limit):
        """Show 8-second countdown with settings display and cancel option."""
        # Create countdown dialog
        countdown_dialog = tk.Toplevel(self.root)
        countdown_dialog.title("Launching QuoTrading AI")
        countdown_dialog.geometry("500x450")
        countdown_dialog.resizable(False, False)
        countdown_dialog.configure(bg=self.colors['card'])
        
        # Make it modal
        countdown_dialog.transient(self.root)
        countdown_dialog.grab_set()
        
        # Keep window decorations for better UX (user can still close if needed)
        # This is safer than overrideredirect(True) which could trap users
        
        # Center the dialog
        countdown_dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (countdown_dialog.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (countdown_dialog.winfo_height() // 2)
        countdown_dialog.geometry(f"+{x}+{y}")
        
        # Main content frame
        inner_frame = tk.Frame(countdown_dialog, bg=self.colors['card'])
        inner_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        header_label = tk.Label(
            inner_frame,
            text="🚀 LAUNCHING IN...",
            font=("Segoe UI", 16, "bold"),
            bg=self.colors['card'],
            fg=self.colors['success']
        )
        header_label.pack(pady=(20, 10))
        
        # Countdown display
        countdown_label = tk.Label(
            inner_frame,
            text="8",
            font=("Segoe UI", 48, "bold"),
            bg=self.colors['card'],
            fg=self.colors['success']
        )
        countdown_label.pack(pady=10)
        
        # Settings display
        settings_frame = tk.Frame(inner_frame, bg=self.colors['secondary'], relief=tk.FLAT)
        settings_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(
            settings_frame,
            text="📋 Your Trading Configuration:",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['secondary'],
            fg=self.colors['text']
        ).pack(pady=(10, 5))
        
        # Display all settings
        symbols_str = ", ".join(selected_symbols)
        broker = self.config.get("broker", "TopStep")
        account = self.account_dropdown_var.get()
        contracts = self.contracts_var.get()
        max_trades = self.trades_var.get()
        confidence = self.confidence_var.get()
        shadow_mode = "ON" if self.shadow_mode_var.get() else "OFF"
        max_loss_per_trade = self.config.get("max_loss_per_trade", 200)
        
        settings_text = f"""
Broker: {broker}
Account: {account}
Symbols: {symbols_str}
Contracts Per Trade: {contracts}
Max Loss Per Trade: ${max_loss_per_trade}
Daily Loss Limit: ${loss_limit}
Max Trades/Day: {max_trades}
Confidence Threshold: {confidence}%
Shadow Mode: {shadow_mode}
        """
        
        settings_label = tk.Label(
            settings_frame,
            text=settings_text.strip(),
            font=("Segoe UI", 9),
            bg=self.colors['secondary'],
            fg=self.colors['text'],
            justify=tk.LEFT
        )
        settings_label.pack(pady=(5, 10))
        
        # Cancel button
        self.countdown_cancelled = False
        
        def cancel_launch():
            self.countdown_cancelled = True
            countdown_dialog.destroy()
        
        cancel_btn = tk.Button(
            inner_frame,
            text="  CANCEL  ",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['error'],
            fg='white',
            activebackground='#B91C1C',
            activeforeground='white',
            relief=tk.FLAT,
            bd=0,
            command=cancel_launch,
            cursor="hand2",
            padx=30,
            pady=10
        )
        cancel_btn.pack(pady=(15, 20))
        
        # Countdown logic
        countdown_value = 8  # Use simple variable with nonlocal
        
        def update_countdown():
            nonlocal countdown_value  # Use nonlocal instead of list workaround
            
            if self.countdown_cancelled:
                return
            
            if countdown_value > 0:
                countdown_label.config(text=str(countdown_value))
                countdown_value -= 1
                countdown_dialog.after(1000, update_countdown)
            else:
                # Countdown finished - launch bot
                countdown_dialog.destroy()
                self.launch_bot_process(selected_symbols, selected_account_id)
        
        # Start countdown
        update_countdown()
    
    def launch_bot_process(self, selected_symbols, selected_account_id):
        """Launch bot processes after countdown completes.
        
        Multi-Symbol Architecture:
        - Spawns a SEPARATE PowerShell window for each selected symbol
        - Each window runs independently with its own:
          * Symbol-specific RL data (experiences/{symbol}/signal_experience.json)
          * Independent connection to broker
          * Own heartbeat to server
        - All windows share the same API key (server handles multiple sessions)
        - Account locks prevent duplicate trading on same account
        """
        symbols_str = ", ".join(selected_symbols)
        
        # Launch AI in PowerShell terminal(s)
        try:
            # Get the AI directory (parent of launcher folder)
            bot_dir = Path(__file__).parent.parent.absolute()
            
            # Track all launched processes
            self.bot_processes = []
            
            # Launch a SEPARATE PowerShell window for each symbol
            # MULTI-SYMBOL FIX: Add small delay between launches to avoid session race conditions
            # Each bot creates its own symbol-specific session with the server
            for i, symbol in enumerate(selected_symbols):
                # PowerShell command to run the QuoTrading AI bot with symbol argument
                # Each window gets its own symbol via command-line argument
                ps_command = [
                    "powershell.exe",
                    "-NoExit",  # Keep window open
                    "-Command",
                    f"$host.UI.RawUI.WindowTitle = 'QuoTrading AI - {symbol}'; cd '{bot_dir}'; python src/quotrading_engine.py {symbol}"
                ]
                
                # Start PowerShell process in a NEW CONSOLE WINDOW
                process = subprocess.Popen(
                    ps_command,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                    cwd=str(bot_dir)
                )
                
                self.bot_processes.append((symbol, process))
                
                # MULTI-SYMBOL FIX: Wait between launching each symbol
                # This ensures each bot completes its session registration before the next starts
                # Prevents race conditions where multiple bots try to create sessions simultaneously
                if i < len(selected_symbols) - 1:  # Don't wait after the last symbol
                    time.sleep(MULTI_SYMBOL_LAUNCH_DELAY_SECONDS)
            
            # CREATE ACCOUNT LOCK with ALL bot PIDs
            # Lock tracks all processes so stale lock detection works properly
            if selected_account_id and self.bot_processes:
                all_pids = [proc.pid for symbol, proc in self.bot_processes]
                self.create_account_lock(selected_account_id, all_pids)
            
            # Keep backward compatibility - store first process
            self.bot_process = self.bot_processes[0][1] if self.bot_processes else None
            
            # Close the GUI immediately and release launcher lock
            # Each bot will maintain its own runtime session via heartbeats
            if self.current_api_key:
                release_launcher_lock(self.current_api_key)
            self.root.destroy()
            
        except Exception as e:
            messagebox.showerror(
                "Launch Error",
                f"Failed to launch bot:\n{str(e)}\n\n"
                f"Make sure Python is installed and src/quotrading_engine.py exists."
            )
    
    def show_settings_dialog(self):
        """Show comprehensive settings dialog with tabs."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Settings")
        dialog.geometry("600x700")
        dialog.configure(bg=self.colors['background'])
        dialog.resizable(False, False)
        
        # Center the dialog
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Header
        header_frame = tk.Frame(dialog, bg=self.colors['success_dark'], height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame,
            text="⚙️ Settings",
            font=("Segoe UI", 14, "bold"),
            bg=self.colors['success_dark'],
            fg='white'
        ).pack(pady=15)
        
        # Tab container
        tab_container = tk.Frame(dialog, bg=self.colors['background'])
        tab_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Tab buttons
        tab_button_frame = tk.Frame(tab_container, bg=self.colors['background'])
        tab_button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Tab content frames
        theme_tab = tk.Frame(tab_container, bg=self.colors['card'])
        alerts_tab = tk.Frame(tab_container, bg=self.colors['card'])
        
        def show_tab(tab_name):
            """Switch between tabs."""
            # Hide all tabs
            theme_tab.pack_forget()
            alerts_tab.pack_forget()
            
            # Update button colors
            for btn in tab_button_frame.winfo_children():
                btn.configure(bg=self.colors['card_elevated'], fg=self.colors['text'])
            
            # Show selected tab and highlight button
            if tab_name == "theme":
                theme_tab.pack(fill=tk.BOTH, expand=True)
                theme_btn.configure(bg=self.colors['success_dark'], fg='white')
            elif tab_name == "alerts":
                alerts_tab.pack(fill=tk.BOTH, expand=True)
                alerts_btn.configure(bg=self.colors['success_dark'], fg='white')
        
        # Tab buttons
        theme_btn = tk.Button(
            tab_button_frame,
            text="🎨 Theme",
            command=lambda: show_tab("theme"),
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['success_dark'],
            fg='white',
            cursor="hand2",
            relief=tk.FLAT,
            padx=20,
            pady=8
        )
        theme_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        alerts_btn = tk.Button(
            tab_button_frame,
            text="🔔 Alerts",
            command=lambda: show_tab("alerts"),
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card_elevated'],
            fg=self.colors['text'],
            cursor="hand2",
            relief=tk.FLAT,
            padx=20,
            pady=8
        )
        alerts_btn.pack(side=tk.LEFT)
        
        # === THEME TAB CONTENT ===
        self._build_theme_tab(theme_tab, dialog)
        
        # === ALERTS TAB CONTENT ===
        self._build_alerts_tab(alerts_tab, dialog)
        
        # Show theme tab by default
        show_tab("theme")
        
        # Close button
        close_btn = tk.Button(
            dialog,
            text="Close",
            command=dialog.destroy,
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card_elevated'],
            fg=self.colors['text'],
            cursor="hand2",
            relief=tk.FLAT,
            padx=30,
            pady=10
        )
        close_btn.pack(pady=(0, 20))
    
    def _build_theme_tab(self, parent, dialog):
        """Build theme settings tab content."""
        tk.Label(
            parent,
            text="Choose Your Theme",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(pady=(20, 10))
        
        tk.Label(
            parent,
            text="Select a color scheme for the application",
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text_secondary']
        ).pack(pady=(0, 20))
        
        # Theme selection buttons
        theme_buttons = tk.Frame(parent, bg=self.colors['card'])
        theme_buttons.pack(pady=20)
        
        current_theme = self.config.get("theme", "light")
        
        def apply_theme(theme_name):
            """Apply selected theme and reload UI immediately."""
            self.config["theme"] = theme_name
            self.save_config()
            
            # Update colors
            if theme_name == "dark":
                self.colors = self.dark_theme_colors.copy()
            else:
                self.colors = self.light_theme_colors.copy()
            
            # Close settings dialog
            dialog.destroy()
            
            # Reload the current screen
            current_screen = self.current_screen
            
            # Clear all widgets
            for widget in self.root.winfo_children():
                widget.destroy()
            
            # Reapply background
            self.root.configure(bg=self.colors['background'])
            
            # Recreate the appropriate screen
            if current_screen == 0:
                self.setup_broker_screen()
            elif current_screen == 1:
                self.setup_trading_screen()
            
            # Show success message
            messagebox.showinfo(
                "Theme Applied",
                f"{theme_name.title()} theme has been applied successfully!"
            )
        
        # Light Theme Button
        light_frame = tk.Frame(theme_buttons, bg='white', bd=2, relief=tk.RIDGE)
        light_frame.pack(side=tk.LEFT, padx=10)
        
        light_preview = tk.Frame(light_frame, bg='#E0E0E0', width=150, height=100)
        light_preview.pack(padx=10, pady=10)
        light_preview.pack_propagate(False)
        
        tk.Label(
            light_preview,
            text="Light Theme",
            font=("Segoe UI", 10, "bold"),
            bg='#E0E0E0',
            fg='#1F2937'
        ).pack(expand=True)
        
        tk.Button(
            light_frame,
            text="✓ Select" if current_theme == "light" else "Select",
            command=lambda: apply_theme("light"),
            font=("Segoe UI", 8, "bold" if current_theme == "light" else "normal"),
            bg='#0078D4' if current_theme == "light" else '#E0E0E0',
            fg='white' if current_theme == "light" else '#1F2937',
            cursor="hand2",
            relief=tk.FLAT,
            padx=15,
            pady=5
        ).pack(pady=(0, 10))
        
        # Dark Theme Button
        dark_frame = tk.Frame(theme_buttons, bg='#1F2937', bd=2, relief=tk.RIDGE)
        dark_frame.pack(side=tk.LEFT, padx=10)
        
        dark_preview = tk.Frame(dark_frame, bg='#1a1a2e', width=150, height=100)
        dark_preview.pack(padx=10, pady=10)
        dark_preview.pack_propagate(False)
        
        tk.Label(
            dark_preview,
            text="Dark Theme",
            font=("Segoe UI", 10, "bold"),
            bg='#1a1a2e',
            fg='#E0E0E0'
        ).pack(expand=True)
        
        tk.Button(
            dark_frame,
            text="✓ Select" if current_theme == "dark" else "Select",
            command=lambda: apply_theme("dark"),
            font=("Segoe UI", 8, "bold" if current_theme == "dark" else "normal"),
            bg='#0078D4' if current_theme == "dark" else '#2D3748',
            fg='white',
            cursor="hand2",
            relief=tk.FLAT,
            padx=15,
            pady=5
        ).pack(pady=(0, 10))
        
        # Current theme indicator
        tk.Label(
            parent,
            text=f"Current Theme: {current_theme.title()}",
            font=("Segoe UI", 9),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(pady=(30, 10))
    
    def _build_alerts_tab(self, parent, settings_dialog):
        """Build alerts settings tab content."""
        # Scrollable content
        canvas = tk.Canvas(parent, bg=self.colors['card'], highlightthickness=0)
        scrollbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        content = tk.Frame(canvas, bg=self.colors['card'])
        
        content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Email Provider Section
        tk.Label(
            content,
            text="📨 Email Provider:",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 5))
        
        provider_var = tk.StringVar(value=self.config.get("smtp_provider", "gmail"))
        provider_frame = tk.Frame(content, bg=self.colors['card'])
        provider_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(
            provider_frame,
            text="Provider:",
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(side=tk.LEFT)
        
        provider_dropdown = ttk.Combobox(
            provider_frame,
            textvariable=provider_var,
            values=["gmail", "outlook", "yahoo", "office365", "custom"],
            font=("Segoe UI", 9),
            state="readonly",
            width=15
        )
        provider_dropdown.pack(side=tk.LEFT, padx=(5, 0))
        
        # Primary Email Section
        tk.Label(
            content,
            text="✉️ Primary Email Account:",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(10, 5))
        
        tk.Label(
            content,
            text="Your Email Address:",
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W)
        
        email_var = tk.StringVar(value=self.config.get("alert_email", ""))
        email_entry = tk.Entry(
            content,
            textvariable=email_var,
            font=("Segoe UI", 9),
            bg='white',
            fg=self.colors['text']
        )
        email_entry.pack(fill=tk.X, pady=(2, 10))
        
        tk.Label(
            content,
            text="Email Password / App Password:",
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W)
        
        email_pass_var = tk.StringVar(value=self.config.get("alert_email_password", ""))
        email_pass_entry = tk.Entry(
            content,
            textvariable=email_pass_var,
            font=("Segoe UI", 9),
            bg='white',
            fg=self.colors['text'],
            show="•"
        )
        email_pass_entry.pack(fill=tk.X, pady=(2, 5))
        
        # Provider-specific instructions
        instruction_text = tk.StringVar()
        instruction_label = tk.Label(
            content,
            textvariable=instruction_text,
            font=("Segoe UI", 7),
            bg=self.colors['card'],
            fg=self.colors['text_secondary'],
            wraplength=480,
            justify=tk.LEFT
        )
        instruction_label.pack(anchor=tk.W, pady=(0, 10))
        
        def update_instructions(*args):
            provider = provider_var.get()
            if provider == "gmail":
                instruction_text.set("💡 Gmail: Google Account → Security → 2-Step Verification → App Passwords → Generate")
            elif provider == "outlook":
                instruction_text.set("💡 Outlook/Hotmail: Use your regular email password (enable 2FA if required)")
            elif provider == "yahoo":
                instruction_text.set("💡 Yahoo: Account Settings → Security → Generate App Password")
            elif provider == "office365":
                instruction_text.set("💡 Office 365: Use your regular email password")
            elif provider == "custom":
                instruction_text.set("💡 Custom: Enter your SMTP server details below")
                custom_smtp_frame.pack(fill=tk.X, pady=(5, 10), after=instruction_label)
                return
            # Hide custom SMTP fields for non-custom providers
            custom_smtp_frame.pack_forget()
        
        provider_var.trace("w", update_instructions)
        
        # Custom SMTP Settings (hidden by default)
        custom_smtp_frame = tk.Frame(content, bg=self.colors['card'])
        
        tk.Label(
            custom_smtp_frame,
            text="SMTP Server:",
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W)
        
        smtp_server_var = tk.StringVar(value=self.config.get("smtp_server", ""))
        smtp_server_entry = tk.Entry(
            custom_smtp_frame,
            textvariable=smtp_server_var,
            font=("Segoe UI", 9),
            bg='white',
            fg=self.colors['text']
        )
        smtp_server_entry.pack(fill=tk.X, pady=(2, 5))
        
        port_frame = tk.Frame(custom_smtp_frame, bg=self.colors['card'])
        port_frame.pack(fill=tk.X, pady=(0, 5))
        
        tk.Label(
            port_frame,
            text="Port:",
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(side=tk.LEFT)
        
        smtp_port_var = tk.StringVar(value=str(self.config.get("smtp_port", "587")))
        smtp_port_entry = tk.Entry(
            port_frame,
            textvariable=smtp_port_var,
            font=("Segoe UI", 9),
            bg='white',
            fg=self.colors['text'],
            width=10
        )
        smtp_port_entry.pack(side=tk.LEFT, padx=(5, 20))
        
        smtp_tls_var = tk.BooleanVar(value=self.config.get("smtp_tls", True))
        tk.Checkbutton(
            port_frame,
            text="Use TLS/STARTTLS",
            variable=smtp_tls_var,
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text'],
            selectcolor='white'
        ).pack(side=tk.LEFT)
        
        # Additional Email Recipients
        tk.Label(
            content,
            text="👥 Additional Email Recipients (Optional):",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(15, 5))
        
        tk.Label(
            content,
            text="Add extra email addresses to receive alerts (comma-separated):",
            font=("Segoe UI", 7),
            bg=self.colors['card'],
            fg=self.colors['text_secondary']
        ).pack(anchor=tk.W)
        
        additional_emails_var = tk.StringVar(
            value=", ".join(self.config.get("additional_alert_emails", []))
        )
        additional_emails_entry = tk.Entry(
            content,
            textvariable=additional_emails_var,
            font=("Segoe UI", 9),
            bg='white',
            fg=self.colors['text']
        )
        additional_emails_entry.pack(fill=tk.X, pady=(2, 5))
        
        tk.Label(
            content,
            text="💡 Example: partner@email.com, manager@company.com",
            font=("Segoe UI", 7),
            bg=self.colors['card'],
            fg=self.colors['text_secondary']
        ).pack(anchor=tk.W, pady=(0, 15))
        
        # SMS Section
        tk.Label(
            content,
            text="📱 SMS Notifications (Optional - Free via Email-to-SMS):",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 5))
        
        tk.Label(
            content,
            text="Phone Number (10 digits, no spaces):",
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W)
        
        phone_var = tk.StringVar(value=self.config.get("alert_phone", ""))
        phone_entry = tk.Entry(
            content,
            textvariable=phone_var,
            font=("Segoe UI", 9),
            bg='white',
            fg=self.colors['text']
        )
        phone_entry.pack(fill=tk.X, pady=(2, 10))
        
        tk.Label(
            content,
            text="Carrier:",
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W)
        
        carrier_var = tk.StringVar(value=self.config.get("alert_carrier", "verizon"))
        carrier_dropdown = ttk.Combobox(
            content,
            textvariable=carrier_var,
            values=[
                "verizon", "att", "t-mobile", "sprint",
                "boost", "cricket", "metro-pcs", "us-cellular",
                "virgin", "google-fi", "xfinity", "mint",
                "republic", "ting",
                "rogers", "bell", "telus"
            ],
            font=("Segoe UI", 9),
            state="readonly"
        )
        carrier_dropdown.pack(fill=tk.X, pady=(2, 15))
        
        # Initialize instructions
        update_instructions()
        
        # Save Button
        save_btn = tk.Button(
            content,
            text="💾 Save Alert Settings",
            command=lambda: self._save_alerts_settings(
                settings_dialog,
                provider_var, email_var, email_pass_var, additional_emails_var,
                smtp_server_var, smtp_port_var, smtp_tls_var,
                phone_var, carrier_var
            ),
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['success_dark'],
            fg='white',
            cursor="hand2",
            relief=tk.FLAT,
            padx=30,
            pady=10
        )
        save_btn.pack(pady=(20, 0))
    
    def _save_alerts_settings(self, dialog, provider_var, email_var, email_pass_var,
                             additional_emails_var, smtp_server_var, smtp_port_var,
                             smtp_tls_var, phone_var, carrier_var):
        """Save alert settings from the Alerts tab."""
        # Parse additional emails
        additional_emails = []
        if additional_emails_var.get().strip():
            additional_emails = [
                email.strip() 
                for email in additional_emails_var.get().split(",") 
                if email.strip()
            ]
        
        self.config["smtp_provider"] = provider_var.get()
        self.config["alert_email"] = email_var.get()
        self.config["alert_email_password"] = email_pass_var.get()
        self.config["additional_alert_emails"] = additional_emails
        self.config["smtp_server"] = smtp_server_var.get()
        self.config["smtp_port"] = int(smtp_port_var.get()) if smtp_port_var.get().isdigit() else 587
        self.config["smtp_tls"] = smtp_tls_var.get()
        self.config["alert_phone"] = phone_var.get()
        self.config["alert_carrier"] = carrier_var.get()
        
        # Auto-enable alerts if email is configured, disable if not
        self.config["alerts_enabled"] = bool(email_var.get() and email_pass_var.get())
        
        self.save_config()
        messagebox.showinfo("Alert Settings Saved", "Your alert settings have been saved successfully!")
        dialog.destroy()
    
    def show_alerts_config_dialog(self):
        """Show dialog to configure email/SMS alert settings."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Configure Alerts")
        dialog.geometry("550x650")
        dialog.configure(bg=self.colors['background'])
        dialog.resizable(False, False)
        
        # Center the dialog
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Header
        header_frame = tk.Frame(dialog, bg=self.colors['success_dark'], height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame,
            text="📧 Alert Configuration",
            font=("Segoe UI", 14, "bold"),
            bg=self.colors['success_dark'],
            fg='white'
        ).pack(pady=15)
        
        # Scrollable content
        canvas = tk.Canvas(dialog, bg=self.colors['card'], highlightthickness=0)
        scrollbar = tk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        content = tk.Frame(canvas, bg=self.colors['card'])
        
        content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=20, pady=20)
        scrollbar.pack(side="right", fill="y")
        
        # Email Provider Section
        tk.Label(
            content,
            text="📨 Email Provider:",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 5))
        
        provider_var = tk.StringVar(value=self.config.get("smtp_provider", "gmail"))
        provider_frame = tk.Frame(content, bg=self.colors['card'])
        provider_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(
            provider_frame,
            text="Provider:",
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(side=tk.LEFT)
        
        provider_dropdown = ttk.Combobox(
            provider_frame,
            textvariable=provider_var,
            values=["gmail", "outlook", "yahoo", "office365", "custom"],
            font=("Segoe UI", 9),
            state="readonly",
            width=15
        )
        provider_dropdown.pack(side=tk.LEFT, padx=(5, 0))
        
        # Primary Email Section
        tk.Label(
            content,
            text="✉️ Primary Email Account:",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(10, 5))
        
        tk.Label(
            content,
            text="Your Email Address:",
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W)
        
        email_var = tk.StringVar(value=self.config.get("alert_email", ""))
        email_entry = tk.Entry(
            content,
            textvariable=email_var,
            font=("Segoe UI", 9),
            bg='white',
            fg=self.colors['text']
        )
        email_entry.pack(fill=tk.X, pady=(2, 10))
        
        tk.Label(
            content,
            text="Email Password / App Password:",
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W)
        
        email_pass_var = tk.StringVar(value=self.config.get("alert_email_password", ""))
        email_pass_entry = tk.Entry(
            content,
            textvariable=email_pass_var,
            font=("Segoe UI", 9),
            bg='white',
            fg=self.colors['text'],
            show="•"
        )
        email_pass_entry.pack(fill=tk.X, pady=(2, 5))
        
        # Provider-specific instructions
        instruction_text = tk.StringVar()
        instruction_label = tk.Label(
            content,
            textvariable=instruction_text,
            font=("Segoe UI", 7),
            bg=self.colors['card'],
            fg=self.colors['text_secondary'],
            wraplength=480,
            justify=tk.LEFT
        )
        instruction_label.pack(anchor=tk.W, pady=(0, 10))
        
        def update_instructions(*args):
            provider = provider_var.get()
            if provider == "gmail":
                instruction_text.set("💡 Gmail: Google Account → Security → 2-Step Verification → App Passwords → Generate")
            elif provider == "outlook":
                instruction_text.set("💡 Outlook/Hotmail: Use your regular email password (enable 2FA if required)")
            elif provider == "yahoo":
                instruction_text.set("💡 Yahoo: Account Settings → Security → Generate App Password")
            elif provider == "office365":
                instruction_text.set("💡 Office 365: Use your regular email password")
            elif provider == "custom":
                instruction_text.set("💡 Custom: Enter your SMTP server details below")
                custom_smtp_frame.pack(fill=tk.X, pady=(5, 10), after=instruction_label)
                return
            # Hide custom SMTP fields for non-custom providers
            custom_smtp_frame.pack_forget()
        
        provider_var.trace("w", update_instructions)
        
        # Custom SMTP Settings (hidden by default)
        custom_smtp_frame = tk.Frame(content, bg=self.colors['card'])
        
        tk.Label(
            custom_smtp_frame,
            text="SMTP Server:",
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W)
        
        smtp_server_var = tk.StringVar(value=self.config.get("smtp_server", ""))
        smtp_server_entry = tk.Entry(
            custom_smtp_frame,
            textvariable=smtp_server_var,
            font=("Segoe UI", 9),
            bg='white',
            fg=self.colors['text']
        )
        smtp_server_entry.pack(fill=tk.X, pady=(2, 5))
        
        port_frame = tk.Frame(custom_smtp_frame, bg=self.colors['card'])
        port_frame.pack(fill=tk.X, pady=(0, 5))
        
        tk.Label(
            port_frame,
            text="Port:",
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(side=tk.LEFT)
        
        smtp_port_var = tk.StringVar(value=str(self.config.get("smtp_port", "587")))
        smtp_port_entry = tk.Entry(
            port_frame,
            textvariable=smtp_port_var,
            font=("Segoe UI", 9),
            bg='white',
            fg=self.colors['text'],
            width=10
        )
        smtp_port_entry.pack(side=tk.LEFT, padx=(5, 20))
        
        smtp_tls_var = tk.BooleanVar(value=self.config.get("smtp_tls", True))
        tk.Checkbutton(
            port_frame,
            text="Use TLS/STARTTLS",
            variable=smtp_tls_var,
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text'],
            selectcolor='white'
        ).pack(side=tk.LEFT)
        
        # Additional Email Recipients
        tk.Label(
            content,
            text="👥 Additional Email Recipients (Optional):",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(15, 5))
        
        tk.Label(
            content,
            text="Add extra email addresses to receive alerts (comma-separated):",
            font=("Segoe UI", 7),
            bg=self.colors['card'],
            fg=self.colors['text_secondary']
        ).pack(anchor=tk.W)
        
        additional_emails_var = tk.StringVar(
            value=", ".join(self.config.get("additional_alert_emails", []))
        )
        additional_emails_entry = tk.Entry(
            content,
            textvariable=additional_emails_var,
            font=("Segoe UI", 9),
            bg='white',
            fg=self.colors['text']
        )
        additional_emails_entry.pack(fill=tk.X, pady=(2, 5))
        
        tk.Label(
            content,
            text="💡 Example: partner@email.com, manager@company.com",
            font=("Segoe UI", 7),
            bg=self.colors['card'],
            fg=self.colors['text_secondary']
        ).pack(anchor=tk.W, pady=(0, 15))
        
        # SMS Section
        tk.Label(
            content,
            text="📱 SMS Notifications (Optional - Free via Email-to-SMS):",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W, pady=(0, 5))
        
        tk.Label(
            content,
            text="Phone Number (10 digits, no spaces):",
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W)
        
        phone_var = tk.StringVar(value=self.config.get("alert_phone", ""))
        phone_entry = tk.Entry(
            content,
            textvariable=phone_var,
            font=("Segoe UI", 9),
            bg='white',
            fg=self.colors['text']
        )
        phone_entry.pack(fill=tk.X, pady=(2, 10))
        
        tk.Label(
            content,
            text="Carrier:",
            font=("Segoe UI", 8),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(anchor=tk.W)
        
        carrier_var = tk.StringVar(value=self.config.get("alert_carrier", "verizon"))
        carrier_dropdown = ttk.Combobox(
            content,
            textvariable=carrier_var,
            values=[
                "verizon", "att", "t-mobile", "sprint",
                "boost", "cricket", "metro-pcs", "us-cellular",
                "virgin", "google-fi", "xfinity", "mint",
                "republic", "ting",
                "rogers", "bell", "telus"
            ],
            font=("Segoe UI", 9),
            state="readonly"
        )
        carrier_dropdown.pack(fill=tk.X, pady=(2, 15))
        
        # Initialize instructions
        update_instructions()
        
        # Buttons
        button_frame = tk.Frame(content, bg=self.colors['card'])
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        def save_and_close():
            # Parse additional emails
            additional_emails = []
            if additional_emails_var.get().strip():
                additional_emails = [
                    email.strip() 
                    for email in additional_emails_var.get().split(",") 
                    if email.strip()
                ]
            
            self.config["smtp_provider"] = provider_var.get()
            self.config["alert_email"] = email_var.get()
            self.config["alert_email_password"] = email_pass_var.get()
            self.config["additional_alert_emails"] = additional_emails
            self.config["smtp_server"] = smtp_server_var.get()
            self.config["smtp_port"] = int(smtp_port_var.get()) if smtp_port_var.get().isdigit() else 587
            self.config["smtp_tls"] = smtp_tls_var.get()
            self.config["alert_phone"] = phone_var.get()
            self.config["alert_carrier"] = carrier_var.get()
            
            # Auto-enable alerts if email is configured, disable if not
            self.config["alerts_enabled"] = bool(email_var.get() and email_pass_var.get())
            
            self.save_config()
            dialog.destroy()
        
        def cancel():
            # User cancelled
            dialog.destroy()
        
        tk.Button(
            button_frame,
            text="Save Configuration",
            command=save_and_close,
            font=("Segoe UI", 9, "bold"),
            bg=self.colors['success_dark'],
            fg='white',
            cursor="hand2",
            relief=tk.FLAT,
            padx=20,
            pady=8
        ).pack(side=tk.RIGHT, padx=(5, 0))
        
        tk.Button(
            button_frame,
            text="Cancel",
            command=cancel,
            font=("Segoe UI", 9),
            bg=self.colors['card_elevated'],
            fg=self.colors['text'],
            cursor="hand2",
            relief=tk.FLAT,
            padx=20,
            pady=8
        ).pack(side=tk.RIGHT)
    
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

# QuoTrading License (Required - contact support@quotrading.com to purchase)
QUOTRADING_LICENSE_KEY={self.config.get("quotrading_api_key", "")}
QUOTRADING_API_KEY={self.config.get("quotrading_api_key", "")}
QUOTRADING_API_URL=https://quotrading-flask-api.azurewebsites.net
ACCOUNT_SIZE={self.config.get("account_size", 50000)}

# Broker Configuration
BROKER={broker}
BROKER_API_TOKEN={self.config.get("broker_token", "")}
BROKER_USERNAME={self.config.get("broker_username", "")}

# Trading Configuration - Multi-Symbol Support
BOT_INSTRUMENTS={symbols_str}
BOT_MAX_CONTRACTS={self.contracts_var.get()}
BOT_MAX_TRADES_PER_DAY={self.trades_var.get()}
# Bot stays on but will NOT execute trades after reaching max (resets daily after market maintenance)
BOT_DAILY_LOSS_LIMIT={self.loss_entry.get()}
# Bot stays on but will NOT execute trades if this limit (in dollars) is hit (resets daily after market maintenance)
BOT_MAX_LOSS_PER_TRADE={self.config.get("max_loss_per_trade", 200)}
# Position closes automatically if a single trade loses this amount

# AI/Confidence Settings
BOT_CONFIDENCE_THRESHOLD={self.confidence_var.get()}
# Bot only takes signals above this confidence threshold (user's minimum)

# Trading Mode (Shadow Trading / Shadow Mode)
BOT_SHADOW_MODE={'true' if self.shadow_mode_var.get() else 'false'}
# When true: Bot provides signals only, no automatic trade execution (manual trading)

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
    
    def check_account_lock(self, account_id):
        """Check if an account is already being traded in another instance.
        
        Multi-symbol aware: checks all PIDs in the lock file.
        Lock is considered active if ANY of the tracked processes is still running.
        """
        locks_dir = Path("locks")
        locks_dir.mkdir(exist_ok=True)
        
        lock_file = locks_dir / f"account_{account_id}.lock"
        
        if not lock_file.exists():
            return False, None
        
        try:
            with open(lock_file, 'r') as f:
                lock_data = json.load(f)
            
            # Check if ANY process is still running (multi-symbol aware)
            # First check the pids list (multi-symbol mode)
            pids = lock_data.get("pids", [])
            if not pids:
                # Fallback to single pid for backward compatibility
                single_pid = lock_data.get("pid")
                if single_pid:
                    pids = [single_pid]
            
            # Lock is active if ANY tracked process is still running
            for pid in pids:
                if pid and psutil.pid_exists(pid):
                    return True, lock_data
            
            # All processes terminated - stale lock, remove it
            lock_file.unlink()
            return False, None
        except:
            return False, None
    
    def create_account_lock(self, account_id, bot_pids):
        """Create a lock file for an account being traded.
        
        Args:
            account_id: The trading account ID to lock
            bot_pids: Single PID (int) or list of PIDs for multi-symbol mode
        """
        locks_dir = Path("locks")
        locks_dir.mkdir(exist_ok=True)
        
        lock_file = locks_dir / f"account_{account_id}.lock"
        
        # Support both single PID and list of PIDs for multi-symbol
        if isinstance(bot_pids, int):
            pid_list = [bot_pids]
        else:
            pid_list = list(bot_pids)
        
        lock_data = {
            "account_id": account_id,
            "pid": pid_list[0] if pid_list else None,  # Primary PID for backward compatibility
            "pids": pid_list,  # All PIDs for multi-symbol mode
            "created_at": datetime.now().isoformat(),
            "broker_username": self.config.get("topstep_username", "unknown")
        }
        
        try:
            with open(lock_file, 'w') as f:
                json.dump(lock_data, f, indent=2)
            return True
        except Exception as e:
            return False
    
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
