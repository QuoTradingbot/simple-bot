"""
Test Cloud RL Brain Response Speed
Measures decision latency from Azure Flask API
"""

import requests
import time
import statistics

# Cloud API endpoint
CLOUD_API_URL = "https://quotrading-flask-api.azurewebsites.net"

def test_cloud_rl_speed():
    print("Testing Cloud RL Brain Speed...")
    print(f"Endpoint: {CLOUD_API_URL}\n")
    
    # Test payload matching cloud API format
    test_payload = {
        "license_key": "TEST_LICENSE",  # Use test license or real one
        "state": {
            "rsi": 45.0,
            "vwap_distance": -0.5,
            "atr": 2.5,
            "hour": 10,
            "trend": 1,
            "tick_distance_from_vwap": -2.0,
            "symbol": "ES",
            "regime": "NORMAL",
            "side": "LONG",
            "spread_ticks": 1.0
        }
    }
    
    # Warm-up request (first request is always slower)
    print("Warming up connection...")
    try:
        response = requests.post(
            f"{CLOUD_API_URL}/api/rl/analyze-signal",
            json=test_payload,
            timeout=10
        )
        print(f"✅ Warmup complete: {response.status_code}\n")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Run 100 test queries
    print("Running 100 decision queries...\n")
    times = []
    
    for i in range(100):
        start = time.time()
        try:
            response = requests.post(
                f"{CLOUD_API_URL}/api/rl/analyze-signal",
                json=test_payload,
                timeout=10
            )
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)
            
            if i == 0:
                # Show first result
                result = response.json()
                print(f"Sample result: {result}\n")
        except Exception as e:
            print(f"❌ Request {i+1} failed: {e}")
    
    if times:
        avg = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        p95 = sorted(times)[int(len(times) * 0.95)]
        
        print(f"📊 Cloud RL Decision Performance:")
        print(f"   Average: {avg:.2f}ms")
        print(f"   Min:     {min_time:.2f}ms")
        print(f"   Max:     {max_time:.2f}ms")
        print(f"   P95:     {p95:.2f}ms")
        print(f"\n   Total requests: {len(times)}")
        print(f"   Success rate: {len(times)/100*100:.0f}%")
        
        # Compare to local (from previous test)
        print(f"\n🔍 Comparison:")
        print(f"   Local RL:  12.32ms average")
        print(f"   Cloud RL:  {avg:.2f}ms average")
        print(f"   Overhead:  {avg - 12.32:.2f}ms (network + API)")

if __name__ == "__main__":
    test_cloud_rl_speed()
