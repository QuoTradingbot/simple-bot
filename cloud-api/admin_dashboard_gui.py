"""
QuoTrading Admin Dashboard - GUI Application
Launch this to view all users, subscriptions, and revenue
"""

import tkinter as tk
from tkinter import ttk, messagebox
import requests
import os
from datetime import datetime
import threading

# Configuration - supports both Render and Azure deployments
# Set QUOTRADING_API_URL environment variable to override default
# Examples:
#   - Render: https://quotrading-api.onrender.com
#   - Azure: https://quotrading-api.azurewebsites.net
#   - Local: http://localhost:8000
API_URL = os.getenv("QUOTRADING_API_URL", "https://quotrading-api.onrender.com")
ADMIN_KEY = "QUOTRADING_ADMIN_MASTER_2025"

class AdminDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("QuoTrading Admin Dashboard")
        self.root.geometry("1000x700")
        self.root.configure(bg="#1a1a2e")
        
        # Colors
        self.colors = {
            'bg': '#1a1a2e',
            'card': '#16213e',
            'accent': '#0f3460',
            'success': '#00d4ff',
            'warning': '#ffa500',
            'danger': '#ff4444',
            'text': '#ffffff',
            'text_secondary': '#a0a0a0'
        }
        
        self.setup_ui()
        self.load_dashboard()
    
    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg=self.colors['accent'], height=80)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        title = tk.Label(
            header,
            text="📊 QuoTrading Admin Dashboard",
            font=("Segoe UI", 24, "bold"),
            bg=self.colors['accent'],
            fg=self.colors['success']
        )
        title.pack(pady=20)
        
        # Main container
        main = tk.Frame(self.root, bg=self.colors['bg'])
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Stats cards
        stats_frame = tk.Frame(main, bg=self.colors['bg'])
        stats_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Create stat cards
        self.total_users_label = self.create_stat_card(stats_frame, "Total Users", "0", 0)
        self.active_subs_label = self.create_stat_card(stats_frame, "Active ($200/mo)", "0", 1)
        self.revenue_label = self.create_stat_card(stats_frame, "Monthly Revenue", "$0", 2)
        self.past_due_label = self.create_stat_card(stats_frame, "Past Due", "0", 3)
        
        # Users table
        table_frame = tk.Frame(main, bg=self.colors['card'])
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        table_title = tk.Label(
            table_frame,
            text="👥 All Users",
            font=("Segoe UI", 14, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        table_title.pack(anchor=tk.W, padx=15, pady=10)
        
        # Action buttons
        action_frame = tk.Frame(table_frame, bg=self.colors['card'])
        action_frame.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        register_btn = tk.Button(
            action_frame,
            text="➕ Register New User",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['success'],
            fg=self.colors['bg'],
            bd=0,
            padx=15,
            pady=8,
            command=self.register_user,
            cursor="hand2"
        )
        register_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        cancel_sub_btn = tk.Button(
            action_frame,
            text="🚫 Cancel Subscription",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['danger'],
            fg=self.colors['text'],
            bd=0,
            padx=15,
            pady=8,
            command=self.cancel_subscription,
            cursor="hand2"
        )
        cancel_sub_btn.pack(side=tk.LEFT)
        
        # Create treeview
        tree_frame = tk.Frame(table_frame, bg=self.colors['card'])
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Treeview
        self.tree = ttk.Treeview(
            tree_frame,
            columns=("Email", "Status", "Tier", "Logins", "Last Login"),
            show="headings",
            yscrollcommand=scrollbar.set
        )
        scrollbar.config(command=self.tree.yview)
        
        # Columns
        self.tree.heading("Email", text="Email")
        self.tree.heading("Status", text="Status")
        self.tree.heading("Tier", text="Tier")
        self.tree.heading("Logins", text="Total Logins")
        self.tree.heading("Last Login", text="Last Login")
        
        self.tree.column("Email", width=250)
        self.tree.column("Status", width=100)
        self.tree.column("Tier", width=100)
        self.tree.column("Logins", width=100)
        self.tree.column("Last Login", width=200)
        
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Refresh button
        refresh_btn = tk.Button(
            main,
            text="🔄 Refresh Dashboard",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors['success'],
            fg=self.colors['bg'],
            activebackground="#00b8e6",
            bd=0,
            padx=20,
            pady=10,
            command=self.load_dashboard,
            cursor="hand2"
        )
        refresh_btn.pack(pady=(20, 0))
    
    def create_stat_card(self, parent, title, value, col):
        card = tk.Frame(parent, bg=self.colors['card'], relief=tk.FLAT, bd=1)
        card.grid(row=0, column=col, padx=10, sticky="ew")
        parent.columnconfigure(col, weight=1)
        
        title_label = tk.Label(
            card,
            text=title,
            font=("Segoe UI", 10),
            bg=self.colors['card'],
            fg=self.colors['text_secondary']
        )
        title_label.pack(pady=(15, 5))
        
        value_label = tk.Label(
            card,
            text=value,
            font=("Segoe UI", 20, "bold"),
            bg=self.colors['card'],
            fg=self.colors['success']
        )
        value_label.pack(pady=(0, 15))
        
        return value_label
    
    def create_tier_label(self, parent, title, value, col):
        frame = tk.Frame(parent, bg=self.colors['card'])
        frame.grid(row=0, column=col, padx=20, sticky="w")
        parent.columnconfigure(col, weight=1)
        
        title_label = tk.Label(
            frame,
            text=title,
            font=("Segoe UI", 11),
            bg=self.colors['card'],
            fg=self.colors['text_secondary']
        )
        title_label.pack(side=tk.LEFT)
        
        value_label = tk.Label(
            frame,
            text=value,
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['card'],
            fg=self.colors['success']
        )
        value_label.pack(side=tk.LEFT, padx=(10, 0))
        
        return value_label
    
    def register_user(self):
        """Register a new user"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Register New User")
        dialog.geometry("400x150")
        dialog.configure(bg=self.colors['card'])
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(
            dialog,
            text="Enter new user's email:",
            font=("Segoe UI", 11),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(pady=(20, 10))
        
        email_entry = tk.Entry(
            dialog,
            font=("Segoe UI", 11),
            bg=self.colors['bg'],
            fg=self.colors['text'],
            insertbackground=self.colors['text'],
            width=30
        )
        email_entry.pack(pady=10)
        email_entry.focus()
        
        def submit():
            email = email_entry.get().strip()
            if not email:
                messagebox.showerror("Error", "Please enter an email")
                return
            
            def register():
                try:
                    response = requests.post(
                        f"{API_URL}/api/v1/users/register",
                        json={"email": email},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        api_key = data.get("api_key", "")
                        dialog.destroy()
                        messagebox.showinfo(
                            "User Registered!",
                            f"Email: {email}\n\nAPI Key:\n{api_key}\n\nUser can now login with this key!"
                        )
                        self.load_dashboard()
                    else:
                        messagebox.showerror("Error", f"Registration failed: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    messagebox.showerror("Error", f"Registration failed:\n{str(e)}")
            
            threading.Thread(target=register, daemon=True).start()
        
        tk.Button(
            dialog,
            text="Register",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['success'],
            fg=self.colors['bg'],
            bd=0,
            padx=20,
            pady=8,
            command=submit,
            cursor="hand2"
        ).pack(pady=10)
    
    def cancel_subscription(self):
        """Cancel a user's subscription"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a user from the table")
            return
        
        item = self.tree.item(selected[0])
        email = item['values'][0]
        
        if messagebox.askyesno("Confirm", f"Cancel subscription for:\n{email}?"):
            messagebox.showinfo("Info", "Subscription cancellation will be implemented when Stripe is configured.\n\nFor now, you can manually update in Stripe dashboard.")
    
    def load_dashboard(self):
        """Load dashboard data in background thread"""
        def fetch_data():
            try:
                response = requests.get(
                    f"{API_URL}/api/v1/admin/dashboard",
                    headers={"X-Admin-Key": ADMIN_KEY},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self.root.after(0, lambda: self.update_ui(data))
                else:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error",
                        f"Failed to load dashboard: {response.status_code}"
                    ))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Connection Error",
                    f"Could not connect to API:\n{str(e)}"
                ))
        
        # Show loading
        self.total_users_label.config(text="Loading...")
        thread = threading.Thread(target=fetch_data, daemon=True)
        thread.start()
    
    def update_ui(self, data):
        """Update UI with dashboard data"""
        summary = data["summary"]
        users = data["users"]
        
        # Update stats
        self.total_users_label.config(text=str(summary["total_users"]))
        self.active_subs_label.config(text=str(summary["active_subscriptions"]))
        self.revenue_label.config(text=summary["monthly_revenue"])
        self.past_due_label.config(text=str(summary["past_due"]))
        
        # Update users table
        self.tree.delete(*self.tree.get_children())
        for user in users:
            last_login = user["last_login"]
            if last_login:
                last_login = datetime.fromisoformat(last_login.replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M")
            else:
                last_login = "Never"
            
            # Color code by status
            tags = ()
            if user["status"] == "active":
                tags = ("active",)
            elif user["status"] == "past_due":
                tags = ("warning",)
            elif user["status"] == "canceled":
                tags = ("danger",)
            
            self.tree.insert(
                "",
                tk.END,
                values=(
                    user["email"],
                    user["status"],
                    user["tier"],
                    user["total_logins"],
                    last_login
                ),
                tags=tags
            )
        
        # Configure tags
        self.tree.tag_configure("active", foreground="#00ff00")
        self.tree.tag_configure("warning", foreground="#ffa500")
        self.tree.tag_configure("danger", foreground="#ff4444")

if __name__ == "__main__":
    root = tk.Tk()
    app = AdminDashboard(root)
    root.mainloop()
