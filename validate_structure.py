#!/usr/bin/env python3
"""
Validation script to verify the new project structure.
Tests that both production and dev environments are properly set up.
"""

import sys
import os

def test_production_structure():
    """Test that production files are in place and importable"""
    print("Testing production structure (src/)...")
    
    # Check key files exist
    required_files = [
        'src/main.py',
        'src/config.py',
        'src/quotrading_engine.py',
        'src/signal_confidence.py',
        'src/regime_detection.py',
        'src/__init__.py'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"  ❌ Missing: {file}")
            return False
        else:
            print(f"  ✅ Found: {file}")
    
    # Check that backtesting.py is NOT in src/
    if os.path.exists('src/backtesting.py'):
        print(f"  ❌ ERROR: backtesting.py should not be in src/")
        return False
    else:
        print(f"  ✅ Confirmed: backtesting.py removed from src/")
    
    return True

def test_dev_structure():
    """Test that dev files are in place"""
    print("\nTesting dev structure (dev/)...")
    
    required_files = [
        'dev/run_backtest.py',
        'dev/backtesting.py',
        'dev/__init__.py',
        'dev/README.md'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"  ❌ Missing: {file}")
            return False
        else:
            print(f"  ✅ Found: {file}")
    
    return True

def test_documentation():
    """Test that documentation is in place"""
    print("\nTesting documentation...")
    
    required_files = [
        'README.md',
        'dev/README.md'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"  ❌ Missing: {file}")
            return False
        else:
            # Check file size to ensure it's not empty
            size = os.path.getsize(file)
            if size < 100:
                print(f"  ⚠️  {file} seems too small ({size} bytes)")
            else:
                print(f"  ✅ Found: {file} ({size} bytes)")
    
    return True

def test_imports():
    """Test that basic imports work"""
    print("\nTesting imports...")
    
    # Test that we can import from src
    sys.path.insert(0, 'src')
    
    try:
        # Test config import
        from config import BotConfiguration
        print("  ✅ Can import from config")
    except ImportError as e:
        print(f"  ❌ Cannot import from config: {e}")
        return False
    
    # Test that dev can import backtesting
    sys.path.insert(0, 'dev')
    
    try:
        from backtesting import BacktestConfig, BacktestEngine
        print("  ✅ Can import from dev/backtesting")
    except ImportError as e:
        print(f"  ❌ Cannot import from dev/backtesting: {e}")
        return False
    
    return True

def main():
    """Run all validation tests"""
    print("="*60)
    print("VWAP Bounce Bot - Structure Validation")
    print("="*60)
    
    tests = [
        ("Production Structure", test_production_structure),
        ("Dev Structure", test_dev_structure),
        ("Documentation", test_documentation),
        ("Import Tests", test_imports)
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n  ❌ {name} failed with exception: {e}")
            results[name] = False
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All validation tests passed!")
        print("\nProject structure is correct:")
        print("  • src/ contains production trading bot")
        print("  • dev/ contains backtesting environment")
        print("  • Documentation is in place")
        print("  • Imports are working")
        return 0
    else:
        print("\n⚠️  Some validation tests failed. Please review the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
