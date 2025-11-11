"""
Restart Azure cloud API to load the updated 6,880 signal experiences with all 13 features
"""
import subprocess
import time

print("🔄 Restarting Azure cloud API to load updated experiences...")
print("   Container: quotrading-signals")
print("   Resource Group: quotrading-rg")
print()

try:
    # Restart the container
    result = subprocess.run([
        "az", "containerapp", "restart",
        "--name", "quotrading-signals",
        "--resource-group", "quotrading-rg"
    ], capture_output=True, text=True, timeout=60)
    
    if result.returncode == 0:
        print("✅ Container restart initiated!")
        print("\n⏳ Waiting 30 seconds for API to come back online...")
        time.sleep(30)
        
        print("\n📊 Checking API health...")
        import requests
        try:
            response = requests.get("https://quotrading-signals.kindsky-7c6ec7cb.eastus.azurecontainerapps.io/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API is healthy!")
                print(f"   Signal experiences: {data.get('total_signal_experiences', 0):,}")
                print(f"   Exit experiences: {data.get('total_exit_experiences', 0):,}")
                print(f"   Total: {data.get('total_experiences', 0):,}")
            else:
                print(f"⚠️  API returned status {response.status_code}")
        except Exception as e:
            print(f"⚠️  Could not check health: {e}")
    else:
        print(f"❌ Restart failed: {result.stderr}")
        
except subprocess.TimeoutExpired:
    print("❌ Restart command timed out")
except Exception as e:
    print(f"❌ Error: {e}")
