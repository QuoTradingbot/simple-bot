"""
QuoTrading API - Admin CLI Tool
Manage users, subscriptions, and test API endpoints
"""

import requests
import json
import os
from datetime import datetime

# Configuration - supports both Render and Azure deployments
# Set QUOTRADING_API_URL environment variable to override default
# Examples:
#   - Render: https://quotrading-api.onrender.com
#   - Azure: https://quotrading-api.azurewebsites.net
#   - Local: http://localhost:8000
API_URL = os.getenv("QUOTRADING_API_URL", "https://quotrading-api.onrender.com")

def print_response(response):
    """Pretty print API response"""
    try:
        data = response.json()
        print(json.dumps(data, indent=2, default=str))
    except:
        print(response.text)

def register_user(email):
    """Register a new user"""
    print(f"\n📝 Registering user: {email}")
    response = requests.post(
        f"{API_URL}/api/v1/users/register",
        json={"email": email}
    )
    print(f"Status: {response.status_code}")
    print_response(response)
    return response

def validate_license(email, api_key):
    """Validate a user's license"""
    print(f"\n🔐 Validating license for: {email}")
    response = requests.post(
        f"{API_URL}/api/v1/license/validate",
        json={"email": email, "api_key": api_key}
    )
    print(f"Status: {response.status_code}")
    print_response(response)
    return response

def get_user_info(email):
    """Get user information"""
    print(f"\n👤 Getting info for: {email}")
    response = requests.get(f"{API_URL}/api/v1/users/{email}")
    print(f"Status: {response.status_code}")
    print_response(response)
    return response

def test_admin_key():
    """Test admin master key"""
    print("\n🔑 Testing admin master key")
    response = requests.post(
        f"{API_URL}/api/v1/license/validate",
        json={
            "email": "admin@quotrading.com",
            "api_key": "QUOTRADING_ADMIN_MASTER_2025"
        }
    )
    print(f"Status: {response.status_code}")
    print_response(response)
    return response

def health_check():
    """Check API health"""
    print("\n💚 Checking API health")
    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")
    print_response(response)
    return response

def view_dashboard():
    """View complete admin dashboard"""
    print("\n📊 Loading Admin Dashboard")
    response = requests.get(
        f"{API_URL}/api/v1/admin/dashboard",
        headers={"X-Admin-Key": "QUOTRADING_ADMIN_MASTER_2025"}
    )
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        print("\n" + "="*60)
        print("📈 QUOTRADING DASHBOARD")
        print("="*60)
        
        # Summary
        summary = data["summary"]
        print("\n📊 SUMMARY:")
        print(f"  Total Users: {summary['total_users']}")
        print(f"  Active Subscriptions: {summary['active_subscriptions']}")
        print(f"  Inactive Users: {summary['inactive_users']}")
        print(f"  Past Due: {summary['past_due']}")
        print(f"  Canceled: {summary['canceled']}")
        print(f"  💰 Monthly Revenue: {summary['monthly_revenue']}")
        
        # Tier breakdown
        tiers = data["tier_breakdown"]
        print("\n🎯 TIER BREAKDOWN:")
        print(f"  Basic ($99/mo): {tiers['basic']} users")
        print(f"  Pro ($199/mo): {tiers['pro']} users")
        print(f"  Enterprise ($499/mo): {tiers['enterprise']} users")
        
        # Expiring soon
        expiring = data["expiring_soon"]
        if expiring:
            print(f"\n⚠️  EXPIRING SOON ({len(expiring)}):")
            for user in expiring:
                print(f"  - {user['email']} ({user['tier']}) expires {user['expires']}")
        else:
            print("\n✅ No subscriptions expiring in next 7 days")
        
        # Users list
        users = data["users"]
        if users:
            print(f"\n👥 ALL USERS ({len(users)}):")
            for user in users[:10]:  # Show first 10
                print(f"  - {user['email']}")
                print(f"    Status: {user['status']} | Tier: {user['tier']}")
                print(f"    Logins: {user['total_logins']} | Last: {user['last_login']}")
            
            if len(users) > 10:
                print(f"  ... and {len(users) - 10} more users")
        
        print("\n" + "="*60)
    else:
        print_response(response)
    
    return response

def main_menu():
    """Interactive menu"""
    while True:
        print("\n" + "="*50)
        print("QuoTrading API - Admin Tool")
        print("="*50)
        print("1. Health Check")
        print("2. Test Admin Key")
        print("3. Register New User")
        print("4. Validate License")
        print("5. Get User Info")
        print("6. 📊 View Dashboard (ALL USERS & STATS)")
        print("7. Run Full Test Suite")
        print("0. Exit")
        print("="*50)
        
        choice = input("\nEnter choice: ").strip()
        
        if choice == "1":
            health_check()
        
        elif choice == "2":
            test_admin_key()
        
        elif choice == "3":
            email = input("Enter email: ").strip()
            register_user(email)
        
        elif choice == "4":
            email = input("Enter email: ").strip()
            api_key = input("Enter API key: ").strip()
            validate_license(email, api_key)
        
        elif choice == "5":
            email = input("Enter email: ").strip()
            get_user_info(email)
        
        elif choice == "6":
            view_dashboard()
        
        elif choice == "7":
            run_test_suite()
        
        elif choice == "0":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("\n❌ Invalid choice!")

def run_test_suite():
    """Run comprehensive test suite"""
    print("\n" + "="*50)
    print("🧪 Running Test Suite")
    print("="*50)
    
    # Test 1: Health check
    print("\n[1/4] Health Check")
    health_check()
    
    # Test 2: Admin key
    print("\n[2/4] Admin Key Validation")
    test_admin_key()
    
    # Test 3: Register test user
    print("\n[3/4] User Registration")
    test_email = f"test_{datetime.now().timestamp()}@example.com"
    reg_response = register_user(test_email)
    
    if reg_response.status_code == 200:
        data = reg_response.json()
        test_api_key = data.get("api_key")
        
        # Test 4: Validate new user (should fail - no subscription)
        print("\n[4/4] License Validation (should fail - no subscription)")
        validate_license(test_email, test_api_key)
    
    print("\n" + "="*50)
    print("✅ Test Suite Complete!")
    print("="*50)

if __name__ == "__main__":
    print(f"\nAPI URL: {API_URL}")
    print("Note: Make sure your API is deployed and running!")
    
    # Quick test mode
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            run_test_suite()
        elif sys.argv[1] == "health":
            health_check()
        elif sys.argv[1] == "admin":
            test_admin_key()
        elif sys.argv[1] == "dashboard":
            view_dashboard()
    else:
        main_menu()
