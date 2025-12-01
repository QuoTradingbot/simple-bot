"""
Integration Test - Session Management End-to-End
================================================
This test demonstrates the complete session management flow including:
1. Device fingerprint consistency
2. Session creation and validation
3. Stale session auto-clearing

Note: This is a DEMONSTRATION test showing the logic flow.
Actual integration testing requires a running Flask server and database.
"""

import hashlib
import uuid
import getpass
import platform
from datetime import datetime, timedelta


class MockSessionManager:
    """Mock session manager to demonstrate session logic"""
    
    def __init__(self):
        self.sessions = {}  # license_key -> (device_fp, last_heartbeat)
        self.SESSION_TIMEOUT_SECONDS = 120
        
    def get_device_fingerprint(self) -> str:
        """Generate device fingerprint (same logic as production)"""
        try:
            machine_id = str(uuid.getnode())
        except:
            machine_id = "unknown"
        
        try:
            username = getpass.getuser()
        except:
            username = "unknown"
        
        platform_name = platform.system()
        fingerprint_raw = f"{machine_id}:{username}:{platform_name}"
        fingerprint_hash = hashlib.sha256(fingerprint_raw.encode()).hexdigest()[:16]
        
        return fingerprint_hash
    
    def auto_clear_stale_sessions(self, license_key: str) -> bool:
        """Auto-clear stale sessions (same logic as production)"""
        if license_key in self.sessions:
            device_fp, last_heartbeat = self.sessions[license_key]
            if last_heartbeat:
                time_since = datetime.now() - last_heartbeat
                if time_since > timedelta(seconds=self.SESSION_TIMEOUT_SECONDS):
                    print(f"  🧹 Auto-clearing stale session (last seen {int(time_since.total_seconds())}s ago)")
                    del self.sessions[license_key]
                    return True
        return False
    
    def validate_and_create_session(self, license_key: str, device_fp: str) -> tuple[bool, str]:
        """Validate license and create/update session - matches production behavior"""
        # Get current session info FIRST (before any clearing)
        if license_key in self.sessions:
            stored_fp, last_heartbeat = self.sessions[license_key]
            
            # If it's the SAME device, always allow reconnection
            if stored_fp == device_fp:
                # Same device reconnecting - allow immediately
                print(f"  ✅ Same device {device_fp[:8]}... reconnecting - allowing")
                self.sessions[license_key] = (device_fp, datetime.now())
                return True, "Session updated (same device)"
            
            # Different device - NEVER auto-clear, always check if active
            time_since = datetime.now() - last_heartbeat
            if time_since < timedelta(seconds=self.SESSION_TIMEOUT_SECONDS):
                # Active session on different device - BLOCK
                print(f"  ❌ Different device, active session (last seen {int(time_since.total_seconds())}s ago)")
                return False, "LICENSE ALREADY IN USE"
            else:
                # Stale session on different device (>120s) - allow takeover
                print(f"  🧹 Stale session from different device (last seen {int(time_since.total_seconds())}s ago) - allowing takeover")
        
        # Create new session
        print(f"  ✅ Creating new session")
        self.sessions[license_key] = (device_fp, datetime.now())
        return True, "Session created"
    
    def release_session(self, license_key: str, device_fp: str) -> bool:
        """Release session (called on shutdown)"""
        if license_key in self.sessions:
            stored_fp, _ = self.sessions[license_key]
            if stored_fp == device_fp:
                del self.sessions[license_key]
                return True
        return False


def test_scenario_1_fresh_login():
    """Test: Fresh login, no previous session"""
    print("\n" + "="*70)
    print("SCENARIO 1: Fresh Login (No Previous Session)")
    print("="*70)
    
    mgr = MockSessionManager()
    license_key = "TEST-KEY-001"
    device_fp = mgr.get_device_fingerprint()
    
    print(f"1. User logs in with license key: {license_key[:8]}...")
    print(f"2. Device fingerprint: {device_fp}")
    
    # Launcher validates and creates session
    success, msg = mgr.validate_and_create_session(license_key, device_fp)
    print(f"3. Launcher validation: {msg}")
    
    # Bot starts with SAME fingerprint
    print(f"4. Bot starts with same fingerprint: {device_fp}")
    success, msg = mgr.validate_and_create_session(license_key, device_fp)
    print(f"5. Bot validation: {msg}")
    
    assert success, "Bot should be able to start"
    print("✅ PASS: Smooth launcher → bot transition\n")


def test_scenario_2_crash_recovery():
    """Test: Bot crashed, immediate relaunch"""
    print("="*70)
    print("SCENARIO 2: Bot Crashed, Immediate Relaunch (Same Device)")
    print("="*70)
    
    mgr = MockSessionManager()
    license_key = "TEST-KEY-002"
    device_fp = mgr.get_device_fingerprint()
    
    print(f"1. Bot was running, created session")
    mgr.sessions[license_key] = (device_fp, datetime.now() - timedelta(seconds=5))
    
    print(f"2. Bot crashed 5 seconds ago (session still in DB)")
    print(f"3. User immediately relaunches")
    
    # Launcher validates - should allow (same device)
    success, msg = mgr.validate_and_create_session(license_key, device_fp)
    print(f"4. Launcher validation: {msg}")
    
    assert success, "Should allow relaunch (same device)"
    print("✅ PASS: Instant relaunch on same device\n")


def test_scenario_3_stale_session():
    """Test: Bot crashed 5 minutes ago, auto-clear"""
    print("="*70)
    print("SCENARIO 3: Bot Crashed 5 Minutes Ago (Auto-Clear)")
    print("="*70)
    
    mgr = MockSessionManager()
    license_key = "TEST-KEY-003"
    device_fp = mgr.get_device_fingerprint()
    
    print(f"1. Bot crashed 5 minutes ago")
    mgr.sessions[license_key] = (device_fp, datetime.now() - timedelta(seconds=300))
    
    print(f"2. User tries to login")
    
    # Should auto-clear stale session and create new one
    success, msg = mgr.validate_and_create_session(license_key, device_fp)
    print(f"3. Validation result: {msg}")
    
    assert success, "Should auto-clear stale session"
    assert license_key in mgr.sessions, "Should create new session"
    print("✅ PASS: Auto-cleared stale session, instant login\n")


def test_scenario_4_concurrent_login():
    """Test: Concurrent login from different device (blocked)"""
    print("="*70)
    print("SCENARIO 4: Concurrent Login Different Device (Security)")
    print("="*70)
    
    mgr = MockSessionManager()
    license_key = "TEST-KEY-004"
    
    # Device A running
    device_a_fp = "abc123device_a"
    print(f"1. Device A running bot")
    mgr.sessions[license_key] = (device_a_fp, datetime.now() - timedelta(seconds=10))
    
    # Device B tries to login
    device_b_fp = "xyz789device_b"
    print(f"2. Device B tries to login (different device)")
    
    success, msg = mgr.validate_and_create_session(license_key, device_b_fp)
    print(f"3. Validation result: {msg}")
    
    assert not success, "Should block concurrent login"
    assert "ALREADY IN USE" in msg, "Should show clear error"
    print("✅ PASS: Blocked concurrent login (security maintained)\n")


def test_scenario_5_clean_shutdown():
    """Test: Clean shutdown releases session"""
    print("="*70)
    print("SCENARIO 5: Clean Shutdown (Session Released)")
    print("="*70)
    
    mgr = MockSessionManager()
    license_key = "TEST-KEY-005"
    device_fp = mgr.get_device_fingerprint()
    
    print(f"1. Bot running")
    mgr.sessions[license_key] = (device_fp, datetime.now())
    
    print(f"2. Bot shutdown (calls release_session)")
    released = mgr.release_session(license_key, device_fp)
    
    assert released, "Should release session"
    assert license_key not in mgr.sessions, "Session should be cleared"
    
    print(f"3. Session released: {released}")
    print(f"4. User relaunches immediately")
    
    success, msg = mgr.validate_and_create_session(license_key, device_fp)
    print(f"5. Validation result: {msg}")
    
    assert success, "Should allow instant relaunch"
    print("✅ PASS: Clean shutdown, instant relaunch\n")


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "="*70)
    print("SESSION MANAGEMENT INTEGRATION TESTS")
    print("="*70)
    
    tests = [
        ("Fresh Login", test_scenario_1_fresh_login),
        ("Crash Recovery (Same Device)", test_scenario_2_crash_recovery),
        ("Stale Session Auto-Clear", test_scenario_3_stale_session),
        ("Concurrent Login Block", test_scenario_4_concurrent_login),
        ("Clean Shutdown", test_scenario_5_clean_shutdown),
    ]
    
    passed = 0
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAIL: {name} - {e}\n")
    
    print("="*70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print("="*70)
    
    if passed == len(tests):
        print("\n✅ ALL INTEGRATION TESTS PASSED")
        print("\nKey Improvements Demonstrated:")
        print("  ✅ Same device can reconnect instantly (no wait)")
        print("  ✅ Stale sessions auto-cleared (no 2-min wait)")
        print("  ✅ Concurrent logins blocked (security maintained)")
        print("  ✅ Clean shutdowns release immediately")
        print("  ✅ Launcher and bot share same session")
        return 0
    else:
        print(f"\n❌ {len(tests) - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run_all_tests())
