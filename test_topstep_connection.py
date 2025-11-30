"""
Quick TopStep Connection Test
Tests if your TopStep credentials work and connection is stable
"""
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from broker_interface import TopStepBroker
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_connection():
    """Test TopStep connection with your credentials"""
    print("=" * 60)
    print("🔌 TopStep Connection Test")
    print("=" * 60)
    
    # Get credentials from user
    print("\nEnter your TopStep credentials:")
    username = input("Username/Email: ").strip()
    api_token = input("API Token: ").strip()
    
    if not username or not api_token:
        print("❌ Error: Username and API token are required!")
        return False
    
    print(f"\n📡 Testing connection...")
    print(f"Username: {username}")
    print(f"API Token: {api_token[:10]}..." + "*" * 20)
    print()
    
    # Create broker instance
    try:
        broker = TopStepBroker(
            api_token=api_token,
            username=username,
            instrument="ES"  # Test with ES
        )
        
        # Test connection
        print("🔄 Connecting to TopStep SDK...")
        connected = broker.connect(max_retries=3)
        
        if connected:
            print("\n" + "=" * 60)
            print("✅ CONNECTION SUCCESSFUL!")
            print("=" * 60)
            
            # Get account info
            try:
                equity = broker.get_account_equity()
                print(f"\n💰 Account Balance: ${equity:,.2f}")
                
                # Test health check
                print("\n🏥 Testing connection health check...")
                is_healthy = broker.verify_connection()
                if is_healthy:
                    print("✅ Health check passed - connection is stable")
                else:
                    print("⚠️  Health check warning - connection may be unstable")
                
            except Exception as e:
                print(f"⚠️  Warning: Could not get account info: {e}")
            
            # Disconnect
            print("\n🔌 Disconnecting...")
            broker.disconnect()
            print("✅ Disconnected successfully")
            
            print("\n" + "=" * 60)
            print("🎉 TEST PASSED - Your TopStep connection works!")
            print("=" * 60)
            return True
            
        else:
            print("\n" + "=" * 60)
            print("❌ CONNECTION FAILED")
            print("=" * 60)
            print("\nPossible issues:")
            print("1. Invalid API token")
            print("2. Invalid username/email")
            print("3. Network/firewall blocking connection")
            print("4. TopStep API is down")
            return False
            
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ ERROR DURING TEST")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = test_connection()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test cancelled by user")
        sys.exit(1)
