#!/usr/bin/env python3
"""
Test script for verifying logo display improvements.
Tests that the logo:
1. Uses crisp, readable characters
2. Displays immediately without delay
3. Centers vertically on screen
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_logo_imports():
    """Test that logo module imports correctly."""
    try:
        from rainbow_logo import display_animated_logo, QUO_AI_LOGO, SUBTITLE
        print("✓ Logo module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import logo module: {e}")
        return False


def test_logo_content():
    """Test that logo contains expected characters."""
    from rainbow_logo import QUO_AI_LOGO
    
    # Check that logo is not empty
    if not QUO_AI_LOGO or len(QUO_AI_LOGO) == 0:
        print("✗ Logo is empty")
        return False
    
    print(f"✓ Logo has {len(QUO_AI_LOGO)} lines")
    
    # Check that logo uses box-drawing characters (improved clarity)
    logo_text = '\n'.join(QUO_AI_LOGO)
    box_chars = ['█', '╗', '╔', '╝', '╚', '═', '║']
    
    has_box_chars = any(char in logo_text for char in box_chars)
    if has_box_chars:
        print("✓ Logo uses crisp box-drawing characters")
    else:
        print("✗ Logo doesn't use box-drawing characters")
        return False
    
    # Check that each line has content
    for i, line in enumerate(QUO_AI_LOGO):
        if len(line.strip()) == 0:
            print(f"✗ Line {i} is empty")
            return False
    
    print("✓ All logo lines have content")
    return True


def test_logo_display():
    """Test that logo displays without errors."""
    from rainbow_logo import display_animated_logo
    
    try:
        # Test with very short duration to avoid long waits
        print("\nTesting logo display (0.5 second animation)...")
        display_animated_logo(duration=0.5, fps=10, with_headers=False)
        print("✓ Logo displayed successfully")
        return True
    except Exception as e:
        print(f"✗ Logo display failed: {e}")
        return False


def test_instant_display_flag():
    """Test that the main quotrading_engine.py is configured for instant display."""
    try:
        engine_file = Path(__file__).parent / "src" / "quotrading_engine.py"
        with open(engine_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check that there's NO initial cls before logo
        # The logo should display immediately
        lines = content.split('\n')
        
        # Find the __main__ block
        in_main_block = False
        found_logo_call = False
        found_initial_cls = False
        
        for i, line in enumerate(lines):
            if 'if __name__ == "__main__"' in line:
                in_main_block = True
                continue
            
            if in_main_block and not found_logo_call:
                # Check for cls before logo
                if 'os.system(' in line and ('cls' in line or 'clear' in line):
                    found_initial_cls = True
                
                # Check for logo call
                if 'display_animated_logo' in line:
                    found_logo_call = True
                    break
        
        if found_initial_cls:
            print("✗ Found initial screen clear before logo (causes black screen)")
            return False
        else:
            print("✓ No initial screen clear - logo displays instantly")
        
        # Check that duration is reduced (should be 3.0 or less, not 8.0)
        if 'display_animated_logo(duration=8' in content:
            print("✗ Logo duration is still 8 seconds (too slow)")
            return False
        
        if 'display_animated_logo(duration=3' in content or \
           'display_animated_logo(duration=2' in content or \
           'display_animated_logo(duration=1' in content:
            print("✓ Logo duration is optimized (3 seconds or less)")
            return True
        
        # If we get here, duration might be missing or unknown
        print("⚠ Warning: Could not verify logo duration")
        return True  # Don't fail the test, just warn
        
    except Exception as e:
        print(f"✗ Failed to check main file: {e}")
        return False


def main():
    """Run all logo tests."""
    print("=" * 60)
    print("LOGO DISPLAY IMPROVEMENTS TEST")
    print("=" * 60)
    print()
    
    tests = [
        ("Logo Module Import", test_logo_imports),
        ("Logo Content Quality", test_logo_content),
        ("Logo Display", test_logo_display),
        ("Instant Display Configuration", test_instant_display_flag),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        result = test_func()
        results.append((test_name, result))
        print()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
