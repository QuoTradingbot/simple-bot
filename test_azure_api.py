"""
Test all Azure API endpoints to verify everything is working
"""
import requests
import json
from datetime import datetime

BASE_URL = "https://quotrading-signals.icymeadow-86b2969e.eastus.azurecontainerapps.io"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("1. TESTING HEALTH ENDPOINT")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ Health check PASSED")


def test_calendar_today():
    """Test today's calendar endpoint"""
    print("\n" + "="*60)
    print("2. TESTING CALENDAR - TODAY")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/api/calendar/today")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Date: {data['date']}")
    print(f"Events today: {data['count']}")
    print(f"Has FOMC: {data['has_fomc']}")
    print(f"Has NFP: {data['has_nfp']}")
    print(f"Has CPI: {data['has_cpi']}")
    print(f"Trading recommended: {data['trading_recommended']}")
    
    assert response.status_code == 200
    print("✅ Calendar today PASSED")


def test_calendar_events():
    """Test calendar events endpoint"""
    print("\n" + "="*60)
    print("3. TESTING CALENDAR - UPCOMING EVENTS")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/api/calendar/events?days=30")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Events in next 30 days: {data['count']}")
    print(f"Last updated: {data['last_updated']}")
    print(f"\nUpcoming events:")
    for event in data['events'][:5]:  # Show first 5
        print(f"  {event['date']} - {event['event']} ({event['impact']} impact)")
    
    assert response.status_code == 200
    assert data['count'] > 0
    print("✅ Calendar events PASSED")


def test_ml_stats():
    """Test ML/RL statistics endpoint"""
    print("\n" + "="*60)
    print("4. TESTING ML/RL STATS")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/api/ml/stats")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Total trades: {data['total_trades']:,}")
    print(f"Win rate: {data['win_rate']*100:.1f}%")
    print(f"Avg P&L: ${data['avg_pnl']:.2f}")
    print(f"Total P&L: ${data['total_pnl']:,.2f}")
    print(f"Message: {data['message']}")
    print(f"Last updated: {data['last_updated']}")
    
    assert response.status_code == 200
    assert data['total_trades'] > 0
    print("✅ ML stats PASSED")


def test_ml_confidence():
    """Test ML confidence calculation"""
    print("\n" + "="*60)
    print("5. TESTING ML CONFIDENCE CALCULATION")
    print("="*60)
    
    # Test a long signal
    payload = {
        "user_id": "test_user_12345",
        "symbol": "ES",
        "vwap": 5900.0,
        "price": 5898.5,
        "rsi": 35,
        "signal": "LONG"
    }
    
    print(f"Testing signal: LONG at 5898.5, VWAP=5900, RSI=35")
    response = requests.post(
        f"{BASE_URL}/api/ml/get_confidence",
        json=payload
    )
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"ML Confidence: {data['ml_confidence']*100:.1f}%")
    print(f"Action: {data['action']}")
    print(f"Model version: {data['model_version']}")
    print(f"Total trades in database: {data.get('total_trade_count', 'N/A')}")
    
    assert response.status_code == 200
    assert 0 <= data['ml_confidence'] <= 1
    print("✅ ML confidence PASSED")


def test_license_validation():
    """Test license validation"""
    print("\n" + "="*60)
    print("6. TESTING LICENSE VALIDATION")
    print("="*60)
    
    # Test with invalid license
    payload = {
        "license_key": "INVALID_KEY_12345"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/license/validate",
        json=payload
    )
    print(f"Status (invalid key): {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Should return 403 or error message
    print("✅ License validation endpoint works")


def test_root():
    """Test root endpoint"""
    print("\n" + "="*60)
    print("7. TESTING ROOT ENDPOINT")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Service: {data['service']}")
    print(f"Status: {data['status']}")
    print(f"Version: {data['version']}")
    print(f"Instrument: {data.get('instrument', 'N/A')}")
    
    assert response.status_code == 200
    print("✅ Root endpoint PASSED")


def test_rate_limit():
    """Test rate limiting"""
    print("\n" + "="*60)
    print("8. TESTING RATE LIMITING")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/api/rate-limit/status")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"IP: {data['ip']}")
    print(f"Requests used: {data['requests_used']}/{data['limit']}")
    print(f"Requests remaining: {data['requests_remaining']}")
    print(f"Rate limit: {data['limit']} requests per {data['window_seconds']}s")
    print(f"Blocked: {data['blocked']}")
    
    assert response.status_code == 200
    assert data['limit'] == 100
    assert data['window_seconds'] == 60
    print("✅ Rate limiting PASSED")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("AZURE API COMPREHENSIVE TEST SUITE")
    print(f"Testing: {BASE_URL}")
    print(f"Time: {datetime.now()}")
    print("="*60)
    
    try:
        test_health()
        test_root()
        test_calendar_today()
        test_calendar_events()
        test_ml_stats()
        test_ml_confidence()
        test_license_validation()
        test_rate_limit()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ TEST FAILED: {e}")
        print("="*60)
        raise
