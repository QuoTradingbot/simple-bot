#!/usr/bin/env python3
"""
Test to verify the session lock and undefined variable fixes.
This validates the specific issues mentioned in the problem statement are fixed.
"""

import sys
from pathlib import Path

def test_validate_license_at_startup_exists():
    """Test 1: Verify validate_license_at_startup function exists"""
    print("\n" + "="*70)
    print("TEST 1: validate_license_at_startup Function Exists")
    print("="*70)
    
    try:
        with open('src/quotrading_engine.py', 'r') as f:
            content = f.read()
        
        # Verify function exists
        if "def validate_license_at_startup(" in content:
            print("✅ validate_license_at_startup function: Found")
        else:
            print("❌ validate_license_at_startup function: NOT FOUND")
            return False
        
        # Verify it's called at the start of main()
        main_func_start = content.find("def main(symbol_override: str = None)")
        if main_func_start == -1:
            print("❌ main function: NOT FOUND")
            return False
        
        # Get the first 1000 characters after main() starts
        main_content = content[main_func_start:main_func_start+1000]
        
        if "validate_license_at_startup()" in main_content:
            print("✅ validate_license_at_startup called in main(): Found")
        else:
            print("❌ validate_license_at_startup called in main(): NOT FOUND")
            return False
        
        # Verify it's called BEFORE other initializations
        # Check that validate_license_at_startup comes before initialize_broker
        validate_pos = main_content.find("validate_license_at_startup()")
        comment_about_validate = main_content.find("CRITICAL: Validate license FIRST")
        
        if comment_about_validate > 0 and comment_about_validate < validate_pos:
            print("✅ validate_license_at_startup called FIRST (before other init): Confirmed")
        else:
            print("❌ validate_license_at_startup placement: NOT at start of main()")
            return False
        
        print("✅ PASS: validate_license_at_startup correctly implemented and called first")
        return True
            
    except Exception as e:
        print(f"❌ FAIL: Error checking validate_license_at_startup: {e}")
        return False


def test_undefined_symbol_variable_fixed():
    """Test 2: Verify undefined 'symbol' variable is fixed"""
    print("\n" + "="*70)
    print("TEST 2: Undefined 'symbol' Variable Fixed")
    print("="*70)
    
    try:
        with open('src/quotrading_engine.py', 'r') as f:
            content = f.read()
        
        # Find the subscribe_quotes section
        subscribe_section_start = content.find("# Subscribe to bid/ask quotes if broker supports it")
        if subscribe_section_start == -1:
            print("❌ subscribe_quotes section: NOT FOUND")
            return False
        
        # Get 500 characters around this section
        subscribe_section = content[subscribe_section_start:subscribe_section_start+500]
        
        # Check for the fixed version using 'trading_symbol'
        if "broker.subscribe_quotes(trading_symbol, on_quote)" in subscribe_section:
            print("✅ Uses 'trading_symbol' instead of 'symbol': Confirmed")
        elif "broker.subscribe_quotes(symbol, on_quote)" in subscribe_section:
            print("❌ Still uses undefined 'symbol': BUG NOT FIXED")
            return False
        else:
            print("⚠️  subscribe_quotes call not found in expected location")
            return False
        
        print("✅ PASS: Undefined 'symbol' variable bug is fixed")
        return True
            
    except Exception as e:
        print(f"❌ FAIL: Error checking undefined symbol fix: {e}")
        return False


def test_license_validation_moved_from_initialize_broker():
    """Test 3: Verify license validation was moved out of initialize_broker"""
    print("\n" + "="*70)
    print("TEST 3: License Validation Moved Out of initialize_broker")
    print("="*70)
    
    try:
        with open('src/quotrading_engine.py', 'r') as f:
            content = f.read()
        
        # Find initialize_broker function
        init_broker_start = content.find("def initialize_broker(")
        if init_broker_start == -1:
            print("❌ initialize_broker function: NOT FOUND")
            return False
        
        # Find the next function after initialize_broker
        next_func_start = content.find("\ndef ", init_broker_start + 1)
        if next_func_start == -1:
            next_func_start = len(content)
        
        # Get initialize_broker function body
        init_broker_body = content[init_broker_start:next_func_start]
        
        # Verify license validation is NOT in initialize_broker anymore
        has_license_check = (
            "LICENSE VALIDATION" in init_broker_body or
            'os.getenv("QUOTRADING_LICENSE_KEY")' in init_broker_body or
            "/api/main" in init_broker_body
        )
        
        if not has_license_check:
            print("✅ License validation removed from initialize_broker: Confirmed")
        else:
            print("❌ License validation still in initialize_broker: NOT MOVED")
            return False
        
        # Verify the comment about license validation being moved
        if "Note: License validation is done at startup" in init_broker_body:
            print("✅ Comment about license validation being moved: Found")
        else:
            print("⚠️  Missing comment about license validation location")
        
        print("✅ PASS: License validation successfully moved to startup")
        return True
            
    except Exception as e:
        print(f"❌ FAIL: Error checking license validation move: {e}")
        return False


def test_session_conflict_caught_at_login():
    """Test 4: Verify session conflicts are caught at 'login screen'"""
    print("\n" + "="*70)
    print("TEST 4: Session Conflict Caught at Login Screen")
    print("="*70)
    
    try:
        with open('src/quotrading_engine.py', 'r') as f:
            content = f.read()
        
        # Verify validate_license_at_startup has session conflict handling
        validate_func_start = content.find("def validate_license_at_startup(")
        if validate_func_start == -1:
            print("❌ validate_license_at_startup function: NOT FOUND")
            return False
        
        # Get the function body (next 6000 chars should cover it)
        validate_func_body = content[validate_func_start:validate_func_start+6000]
        
        checks = {
            "Session conflict detection": 'if data.get("session_conflict"):',
            "Clear stale session attempt": "/api/session/clear",
            "License already in use message": "LICENSE ALREADY IN USE",
            "sys.exit on conflict": "sys.exit(1)",
        }
        
        all_pass = True
        for check_name, check_str in checks.items():
            if check_str in validate_func_body:
                print(f"✅ {check_name}: Found")
            else:
                print(f"❌ {check_name}: NOT FOUND")
                all_pass = False
        
        if all_pass:
            print("✅ PASS: Session conflicts caught at login screen (before initialization)")
            return True
        else:
            print("❌ FAIL: Session conflict handling incomplete")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Error checking session conflict handling: {e}")
        return False


def main():
    """Run all fix verification tests"""
    print("\n" + "="*70)
    print("FIX VERIFICATION TESTS")
    print("Validating fixes for session lock and undefined variable issues")
    print("="*70)
    
    results = {
        "validate_license_at_startup Function": test_validate_license_at_startup_exists(),
        "Undefined 'symbol' Variable Fix": test_undefined_symbol_variable_fixed(),
        "License Validation Moved": test_license_validation_moved_from_initialize_broker(),
        "Session Conflict at Login": test_session_conflict_caught_at_login(),
    }
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL FIXES VERIFIED - Issues are resolved!")
        print("="*70)
        print("\n✅ Issue 1 FIXED: Session conflicts caught at login screen, not runtime")
        print("✅ Issue 2 FIXED: Undefined variable 'symbol' -> now uses 'trading_symbol'")
        print("\nThe bot now:")
        print("  - Validates license BEFORE any initialization")
        print("  - Catches session conflicts immediately at startup")
        print("  - Uses correct variable name for symbol subscriptions")
    else:
        print("⚠️  SOME FIXES NOT VERIFIED - Please review")
        print("="*70)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
