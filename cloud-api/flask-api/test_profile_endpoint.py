"""
Test the /api/profile endpoint
Run this script to test the user profile endpoint functionality
"""
import requests
import json
import os

# API base URL
BASE_URL = os.getenv("QUOTRADING_API_URL", "https://quotrading-flask-api.azurewebsites.net")
# BASE_URL = "http://localhost:5000"  # Uncomment for local testing

# Test license key from environment variable (for security)
TEST_LICENSE_KEY = os.getenv("TEST_LICENSE_KEY", "TEST-LICENSE-KEY-123")

def test_profile_endpoint():
    """Test the /api/profile endpoint with various scenarios"""
    
    print("=" * 60)
    print("Testing /api/profile endpoint")
    print("=" * 60)
    
    # Test 1: Get profile with query parameter
    print("\n1. Testing with query parameter...")
    url = f"{BASE_URL}/api/profile?license_key={TEST_LICENSE_KEY}"
    try:
        response = requests.get(url)
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success!")
            print(f"   Profile: {json.dumps(data, indent=2)}")
        else:
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Get profile with Authorization header
    print("\n2. Testing with Authorization header...")
    url = f"{BASE_URL}/api/profile"
    headers = {"Authorization": f"Bearer {TEST_LICENSE_KEY}"}
    try:
        response = requests.get(url, headers=headers)
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success!")
            print(f"   Email (masked): {data['profile']['email']}")
            print(f"   Total Trades: {data['trading_stats']['total_trades']}")
            print(f"   Total PnL: ${data['trading_stats']['total_pnl']:.2f}")
            print(f"   Win Rate: {data['trading_stats']['win_rate_percent']:.2f}%")
        else:
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: No license key (should fail)
    print("\n3. Testing without license key (should fail)...")
    url = f"{BASE_URL}/api/profile"
    try:
        response = requests.get(url)
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.text}")
        if response.status_code == 400:
            print(f"   ✅ Correctly rejected request without license key")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Invalid license key (should fail)
    print("\n4. Testing with invalid license key (should fail)...")
    url = f"{BASE_URL}/api/profile?license_key=INVALID-KEY-999"
    try:
        response = requests.get(url)
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.text}")
        if response.status_code == 401:
            print(f"   ✅ Correctly rejected invalid license key")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Profile endpoint tests complete!")
    print("=" * 60)

def display_profile_summary(license_key):
    """Display a formatted profile summary for a user"""
    url = f"{BASE_URL}/api/profile?license_key={license_key}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            profile = data['profile']
            stats = data['trading_stats']
            activity = data['recent_activity']
            
            print("\n" + "=" * 60)
            print("USER PROFILE SUMMARY")
            print("=" * 60)
            
            print("\n📋 Account Information:")
            print(f"   Account ID: {profile['account_id']}")
            print(f"   Email: {profile['email']}")
            print(f"   License Type: {profile['license_type']}")
            print(f"   Status: {profile['license_status']}")
            print(f"   Expires: {profile['license_expiration']}")
            print(f"   Days Until Expiration: {profile['days_until_expiration']}")
            print(f"   Account Age: {profile['account_age_days']} days")
            print(f"   Online: {'🟢 Yes' if profile['is_online'] else '🔴 No'}")
            
            print("\n📊 Trading Statistics:")
            print(f"   Total Trades: {stats['total_trades']}")
            print(f"   Total PnL: ${stats['total_pnl']:.2f}")
            print(f"   Avg PnL/Trade: ${stats['avg_pnl_per_trade']:.2f}")
            print(f"   Win Rate: {stats['win_rate_percent']:.2f}%")
            print(f"   Winning Trades: {stats['winning_trades']}")
            print(f"   Losing Trades: {stats['losing_trades']}")
            print(f"   Best Trade: ${stats['best_trade']:.2f}")
            print(f"   Worst Trade: ${stats['worst_trade']:.2f}")
            
            print("\n🔄 Recent Activity:")
            print(f"   API Calls Today: {activity['api_calls_today']}")
            print(f"   API Calls Total: {activity['api_calls_total']}")
            print(f"   Last Heartbeat: {activity['last_heartbeat']}")
            print(f"   Current Device: {activity['current_device']}")
            print(f"   Symbols Traded: {', '.join(activity['symbols_traded']) if activity['symbols_traded'] else 'None'}")
            
            print("\n" + "=" * 60)
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Run the tests
    test_profile_endpoint()
    
    # If you have a valid license key, uncomment the line below to see a formatted profile
    # display_profile_summary("YOUR-LICENSE-KEY-HERE")
