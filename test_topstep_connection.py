#!/usr/bin/env python3
"""
Quick TopStep Connection Test
Tests your TopStep API credentials without starting the full trading bot.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("TopStep Connection Test")
print("=" * 60)

# Check if credentials are set
api_token = os.getenv("TOPSTEP_API_TOKEN")
username = os.getenv("TOPSTEP_USERNAME")

print("\n1. Checking credentials...")
if not api_token or api_token == "your_api_token_here":
    print("   ❌ TOPSTEP_API_TOKEN not set in .env file")
    print("   Please edit .env and set your API token")
    sys.exit(1)
else:
    print(f"   ✅ API Token found: {api_token[:10]}...")

if not username or username == "your_email@example.com":
    print("   ❌ TOPSTEP_USERNAME not set in .env file")
    print("   Please edit .env and set your username/email")
    sys.exit(1)
else:
    print(f"   ✅ Username found: {username}")

# Check if TopStep SDK is installed
print("\n2. Checking TopStep SDK installation...")
try:
    import project_x
    print(f"   ✅ project-x-py installed (version {project_x.__version__})")
except ImportError:
    print("   ❌ TopStep SDK (project-x-py) not installed!")
    print("   Run: pip install project-x-py signalrcore")
    sys.exit(1)

try:
    import signalrcore
    print(f"   ✅ signalrcore installed")
except ImportError:
    print("   ❌ signalrcore not installed!")
    print("   Run: pip install signalrcore")
    sys.exit(1)

# Try to connect to TopStep
print("\n3. Testing connection to TopStep...")
try:
    from project_x import TopStepXClient
    
    # Create client
    client = TopStepXClient(api_token=api_token)
    print("   ✅ Client created")
    
    # Authenticate
    print("   → Authenticating with TopStep...")
    auth_response = client.authenticate(username=username)
    
    if auth_response and auth_response.get('success'):
        print("   ✅ Authentication successful!")
        
        # Get session token
        session_token = auth_response.get('sessionToken')
        if session_token:
            print(f"   ✅ Session token received: {session_token[:20]}...")
        
        # Try to get accounts
        print("\n4. Fetching account information...")
        accounts = client.get_accounts()
        
        if accounts:
            print(f"   ✅ Found {len(accounts)} account(s):")
            for i, account in enumerate(accounts, 1):
                account_id = account.get('id', 'Unknown')
                account_balance = account.get('balance', 0)
                account_type = account.get('type', 'Unknown')
                print(f"      Account {i}: {account_id}")
                print(f"         Type: {account_type}")
                print(f"         Balance: ${account_balance:,.2f}")
        else:
            print("   ⚠️ No accounts found")
        
        print("\n" + "=" * 60)
        print("✅ CONNECTION TEST SUCCESSFUL!")
        print("=" * 60)
        print("\nYou can now run the bot:")
        print("  python src/main.py --dry-run")
        
    else:
        print("   ❌ Authentication failed!")
        print(f"   Response: {auth_response}")
        sys.exit(1)
        
except Exception as e:
    print(f"   ❌ Connection failed: {e}")
    print("\nTroubleshooting:")
    print("  1. Verify your API token is correct")
    print("  2. Verify your username/email is correct")
    print("  3. Check if your TopStep account is active")
    print("  4. Try generating a new API token from TopStep dashboard")
    sys.exit(1)
