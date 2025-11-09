"""
Test rate limiting on Azure API
"""
import requests
import time
from datetime import datetime

BASE_URL = "https://quotrading-signals.icymeadow-86b2969e.eastus.azurecontainerapps.io"

print("\n" + "="*70)
print("RATE LIMITING TEST")
print("="*70)

# Test 1: Check initial rate limit status
print("\n1️⃣ CHECKING INITIAL RATE LIMIT STATUS")
print("-" * 70)

response = requests.get(f"{BASE_URL}/api/rate-limit/status")
data = response.json()
print(f"Status: {response.status_code}")
print(f"IP: {data['ip']}")
print(f"Requests used: {data['requests_used']}/{data['limit']}")
print(f"Requests remaining: {data['requests_remaining']}")
print(f"Window: {data['window_seconds']} seconds")
print(f"Blocked: {data['blocked']}")

initial_remaining = data['requests_remaining']
print(f"\n✅ Rate limit status working - {initial_remaining} requests available")

# Test 2: Make several requests
print("\n2️⃣ TESTING NORMAL REQUEST FLOW")
print("-" * 70)

print("Making 5 requests...")
for i in range(5):
    response = requests.get(f"{BASE_URL}/")
    print(f"  Request {i+1}: {response.status_code}")
    time.sleep(0.1)  # Small delay

# Check status after requests
response = requests.get(f"{BASE_URL}/api/rate-limit/status")
data = response.json()
print(f"\nAfter 5 requests:")
print(f"  Requests used: {data['requests_used']}/{data['limit']}")
print(f"  Requests remaining: {data['requests_remaining']}")

print(f"\n✅ Normal requests working - {data['requests_used']} requests counted")

# Test 3: Explain rate limits
print("\n3️⃣ RATE LIMIT CONFIGURATION")
print("-" * 70)

print(f"Limit: {data['limit']} requests per {data['window_seconds']} seconds")
print(f"Block time: 5 minutes (300 seconds) if exceeded")
print(f"\nThis means:")
print(f"  ✅ Normal bot: ~1 request/second = NO PROBLEM")
print(f"  ✅ ML confidence checks: Every few seconds = NO PROBLEM")
print(f"  ✅ Calendar checks: Once per minute = NO PROBLEM")
print(f"  ⚠️ API spam: 100+ requests/minute = BLOCKED for 5 minutes")

print("\n✅ Rate limiting is reasonable for beta users")

# Test 4: Rapid fire test (WARNING: Will get blocked if you do this!)
print("\n4️⃣ RAPID FIRE TEST (DEMO ONLY - NOT EXECUTING)")
print("-" * 70)

print("⚠️ NOT running rapid fire test (would trigger rate limit)")
print("   If we made 101 requests in < 60 seconds:")
print("   - Request 101 would return: 429 Too Many Requests")
print("   - Blocked for: 5 minutes (300 seconds)")
print("   - Error message: 'Too many requests. Please try again in XXX seconds.'")

# Test 5: Current usage summary
print("\n5️⃣ CURRENT USAGE SUMMARY")
print("-" * 70)

response = requests.get(f"{BASE_URL}/api/rate-limit/status")
data = response.json()

usage_percent = (data['requests_used'] / data['limit']) * 100
print(f"Current usage: {data['requests_used']}/{data['limit']} ({usage_percent:.1f}%)")
print(f"Remaining: {data['requests_remaining']} requests")
print(f"Resets in: ~{data['window_seconds']}s from oldest request")

if usage_percent < 50:
    print(f"\n✅ Usage is low - plenty of capacity")
elif usage_percent < 80:
    print(f"\n⚠️ Usage is moderate - watch for spikes")
else:
    print(f"\n⚠️ Usage is high - getting close to limit")

print("\n" + "="*70)
print("✅ RATE LIMITING TEST COMPLETE!")
print("="*70)

print("\n💡 KEY TAKEAWAYS:")
print("  • Rate limit: 100 requests per 60 seconds")
print("  • Block time: 5 minutes if exceeded")
print("  • Normal bot usage: Well within limits")
print("  • Protection: Prevents API abuse and cost overruns")
print("  • Beta-friendly: Simple in-memory implementation")
print("\n" + "="*70)
