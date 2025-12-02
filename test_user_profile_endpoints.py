"""
Test script for user profile endpoints

IMPORTANT: This test script is for local development/testing only.
Set environment variables before running:
  - API_BASE_URL: The API endpoint (default: http://localhost:5000)
  - TEST_LICENSE_KEY: A valid test license key from your test database

DO NOT run this against production without proper safeguards.
"""
import requests
import json
import os

# Test configuration
BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5000")
# Use environment variable or a placeholder that won't exist in production
TEST_LICENSE_KEY = os.getenv("TEST_LICENSE_KEY", "TEST-XXXX-YYYY-ZZZZ-DOES-NOT-EXIST")

def test_get_profile():
    """Test GET /api/user/profile"""
    print("\n=== Testing GET /api/user/profile ===")
    
    # Test with Authorization header
    headers = {
        "Authorization": f"Bearer {TEST_LICENSE_KEY}",
        "Content-Type": "application/json"
    }
    
    response = requests.get(f"{BASE_URL}/api/user/profile", headers=headers)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test with query parameter
    response = requests.get(f"{BASE_URL}/api/user/profile?license_key={TEST_LICENSE_KEY}")
    print(f"\nWith query param - Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test without authentication
    response = requests.get(f"{BASE_URL}/api/user/profile")
    print(f"\nWithout auth - Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_update_profile():
    """Test PUT /api/user/profile"""
    print("\n=== Testing PUT /api/user/profile ===")
    
    # Test updating email
    headers = {
        "Authorization": f"Bearer {TEST_LICENSE_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "email": "newemail@example.com"
    }
    
    response = requests.put(f"{BASE_URL}/api/user/profile", headers=headers, json=payload)
    print(f"Update email - Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test updating metadata
    payload = {
        "metadata": {
            "theme": "dark",
            "notifications": True,
            "timezone": "America/New_York"
        }
    }
    
    response = requests.put(f"{BASE_URL}/api/user/profile", headers=headers, json=payload)
    print(f"\nUpdate metadata - Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test with invalid email
    payload = {
        "email": "invalid-email"
    }
    
    response = requests.put(f"{BASE_URL}/api/user/profile", headers=headers, json=payload)
    print(f"\nInvalid email - Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test without authentication
    payload = {
        "email": "test@example.com"
    }
    
    response = requests.put(f"{BASE_URL}/api/user/profile", json=payload)
    print(f"\nWithout auth - Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_hello_endpoint():
    """Test if new endpoints are documented in /api/hello"""
    print("\n=== Testing GET /api/hello ===")
    
    response = requests.get(f"{BASE_URL}/api/hello")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    print("User Profile Endpoints Test Suite")
    print("=" * 50)
    
    # Note: These tests require the Flask app to be running
    # and a test database with a valid license key
    
    try:
        test_hello_endpoint()
        test_get_profile()
        test_update_profile()
        
        print("\n" + "=" * 50)
        print("Tests completed!")
        print("\nNote: Some tests may fail if the Flask app is not running")
        print("or if the test license key doesn't exist in the database.")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to Flask API")
        print(f"Make sure the API is running at {BASE_URL}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
