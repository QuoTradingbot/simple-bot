import requests
import time

print("Waiting 30 seconds for API to restart...")
time.sleep(30)

print("\nChecking cloud API health...")
r = requests.get('https://quotrading-signals.icymeadow-86b2969e.eastus.azurecontainerapps.io/health')
data = r.json()

print(f"✅ Cloud API Status:")
print(f"   Signal experiences: {data.get('total_signal_experiences', 0):,}")
print(f"   Exit experiences: {data.get('total_exit_experiences', 0):,}")
print(f"   Total: {data.get('total_experiences', 0):,}")
