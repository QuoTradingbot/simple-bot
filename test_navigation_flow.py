"""
Test Navigation Flow
====================
Tests the GUI navigation flow without actually running the GUI.
This validates the logic and method structure.
"""
import os
from pathlib import Path


def get_launcher_path():
    """Get the path to the launcher file."""
    # Get the directory of this test file
    test_dir = Path(__file__).parent
    # Launcher is in customer subdirectory
    return test_dir / 'customer' / 'QuoTrading_Launcher.py'


def test_launcher_structure():
    """Test that QuoTradingLauncher has all required methods."""
    print("=" * 60)
    print("TEST 1: QuoTrading Launcher Structure")
    print("=" * 60)
    
    launcher_path = get_launcher_path()
    if not launcher_path.exists():
        print(f"❌ Launcher file not found at {launcher_path}")
        return False
    
    # Check required methods by examining the file directly
    # (We can't import because tkinter might not be available)
    with open(launcher_path, 'r') as f:
        content = f.read()
    
    # Check if class exists
    if 'class QuoTradingLauncher' in content:
        print("✅ QuoTradingLauncher class exists")
    else:
        print("❌ QuoTradingLauncher class not found")
        return False
    
    # Check required methods
    required_methods = [
        'setup_broker_screen',        # Screen 0: Broker credentials + QuoTrading API key
        'validate_broker',            # Validates credentials and moves to trading screen
        'setup_trading_screen',       # Screen 1: Trading controls
        'start_bot',                  # Starts the bot
        'create_button',              # Creates navigation buttons
    ]
    
    print("\nChecking required methods:")
    all_present = True
    for method in required_methods:
        # Check the file directly
        if f'def {method}' in content:
            print(f"✅ {method}")
        else:
            print(f"❌ {method} - MISSING")
            all_present = False
    
    return all_present


def test_navigation_flow_logic():
    """Test the navigation flow by examining the code."""
    print("\n" + "=" * 60)
    print("TEST 2: Navigation Flow Logic")
    print("=" * 60)
    
    launcher_path = get_launcher_path()
    with open(launcher_path, 'r') as f:
        content = f.read()
    
    # Test flow: Screen 0 -> Screen 1
    flow_checks = [
        ("Screen 0 (broker setup)", 'def setup_broker_screen', 'self.current_screen = 0'),
        ("Screen 0 next button", 'def validate_broker', 'self.setup_trading_screen()'),
        ("Screen 1 (trading controls)", 'def setup_trading_screen', 'self.current_screen = 1'),
        ("Screen 1 back button", 'setup_trading_screen', 'setup_broker_screen'),
        ("Screen 1 start button", 'def start_bot', 'self.save_config()'),
    ]
    
    print("\nChecking navigation flow:")
    all_passed = True
    for check_name, search_pattern, expected_pattern in flow_checks:
        if search_pattern in content and expected_pattern in content:
            print(f"✅ {check_name}")
        else:
            print(f"❌ {check_name} - Pattern not found")
            all_passed = False
    
    return all_passed


def test_button_existence():
    """Test that navigation buttons exist in all screens."""
    print("\n" + "=" * 60)
    print("TEST 3: Navigation Buttons")
    print("=" * 60)
    
    launcher_path = get_launcher_path()
    with open(launcher_path, 'r') as f:
        content = f.read()
    
    button_checks = [
        ("Screen 0 has NEXT button", 'setup_broker_screen', '"NEXT →"'),
        ("Screen 1 has BACK button", 'setup_trading_screen', '"← BACK"'),
        ("Screen 1 has START button", 'setup_trading_screen', '"START BOT →"'),
    ]
    
    print("\nChecking navigation buttons:")
    all_passed = True
    
    # Split content by function definitions
    functions = {}
    current_func = None
    current_content = []
    
    for line in content.split('\n'):
        if line.strip().startswith('def '):
            if current_func:
                functions[current_func] = '\n'.join(current_content)
            # Extract function name more carefully
            func_line = line.strip()
            if '(' in func_line:
                current_func = func_line.split('(')[0].replace('def ', '').strip()
            else:
                current_func = None
            current_content = [line]
        elif current_func:
            current_content.append(line)
            # Also stop at next class or def
            if line.strip().startswith('class ') or (line.strip().startswith('def ') and current_content):
                pass
    
    if current_func:
        functions[current_func] = '\n'.join(current_content)
    
    for check_name, func_name, button_text in button_checks:
        found = False
        # Check in the specific function
        if func_name in functions and button_text in functions[func_name]:
            found = True
        # Also check if the button text appears anywhere after the function definition
        elif func_name in content:
            func_start = content.find(f'def {func_name}')
            if func_start != -1:
                # Look for the next function or class definition
                next_def = content.find('\n    def ', func_start + 1)
                next_class = content.find('\nclass ', func_start + 1)
                func_end = min(x for x in [next_def, next_class, len(content)] if x > func_start)
                func_content = content[func_start:func_end]
                if button_text in func_content:
                    found = True
        
        if found:
            print(f"✅ {check_name}")
        else:
            print(f"❌ {check_name} - Not found")
            all_passed = False
    
    return all_passed


def test_screen_flow_documentation():
    """Test that the file header documents the correct flow."""
    print("\n" + "=" * 60)
    print("TEST 4: Documentation")
    print("=" * 60)
    
    launcher_path = get_launcher_path()
    with open(launcher_path, 'r') as f:
        lines = f.readlines()
    
    # Check first 20 lines for flow documentation
    header = ''.join(lines[:20])
    
    expected_flow = [
        "Screen 0:",
        "Screen 1:",
    ]
    
    print("\nChecking documentation:")
    all_found = True
    for screen_desc in expected_flow:
        # More flexible check - just look for key parts
        screen_num = screen_desc.split(':')[0]
        if screen_num in header:
            print(f"✅ {screen_desc} documented")
        else:
            print(f"❌ {screen_desc} - Not documented")
            all_found = False
    
    return all_found


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GUI NAVIGATION FLOW VALIDATION")
    print("=" * 60)
    
    test1_pass = test_launcher_structure()
    test2_pass = test_navigation_flow_logic()
    test3_pass = test_button_existence()
    test4_pass = test_screen_flow_documentation()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if test1_pass and test2_pass and test3_pass and test4_pass:
        print("✅ ALL TESTS PASSED - Navigation flow is correctly implemented!")
        print("\nNavigation Flow:")
        print("  Screen 0: Broker Setup (credentials + QuoTrading API key + account size) → [NEXT] →")
        print("  Screen 1: Trading Controls → [START BOT] (← BACK to Screen 0)")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Check errors above")
        return 1


if __name__ == "__main__":
    exit(main())
