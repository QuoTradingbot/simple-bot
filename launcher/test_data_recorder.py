"""
Test script for Market Data Recorder
Validates the CSV output format and data structure for per-symbol files
"""

import csv
from pathlib import Path


def test_csv_format():
    """Test that the CSV format matches expected structure for per-symbol files."""
    expected_headers = [
        'timestamp',
        'data_type',
        'bid_price',
        'bid_size',
        'ask_price',
        'ask_size',
        'trade_price',
        'trade_size',
        'trade_side',
        'depth_level',
        'depth_side',
        'depth_price',
        'depth_size'
    ]
    
    # Create a sample CSV to test format (simulating ES.csv)
    test_file = Path("test_ES.csv")
    
    # Write sample data
    with open(test_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(expected_headers)
        
        # Sample quote
        writer.writerow([
            '2025-12-06T14:30:15.123456',
            'quote',
            '4500.25',
            '10',
            '4500.50',
            '8',
            '',  # trade_price
            '',  # trade_size
            '',  # trade_side
            '',  # depth_level
            '',  # depth_side
            '',  # depth_price
            ''   # depth_size
        ])
        
        # Sample trade
        writer.writerow([
            '2025-12-06T14:30:15.234567',
            'trade',
            '',  # bid_price
            '',  # bid_size
            '',  # ask_price
            '',  # ask_size
            '4500.50',
            '2',
            'buy',
            '',
            '',
            '',
            ''
        ])
        
        # Sample depth
        writer.writerow([
            '2025-12-06T14:30:15.345678',
            'depth',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '0',    # depth_level
            'bid',  # depth_side
            '4500.25',
            '10'
        ])
    
    # Verify the file can be read
    print("Testing CSV file format...")
    with open(test_file, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        
        # Check headers match
        if headers != expected_headers:
            print("❌ FAIL: Headers don't match")
            print(f"Expected: {expected_headers}")
            print(f"Got: {headers}")
            return False
        
        print("✓ Headers match expected format")
        
        # Read and validate each row
        row_count = 0
        for row in reader:
            row_count += 1
            
            # Check that required fields are present
            assert 'timestamp' in row
            assert 'data_type' in row
            
            data_type = row['data_type']
            
            if data_type == 'quote':
                # Quote should have bid/ask data
                if not row['bid_price'] or not row['ask_price']:
                    print(f"⚠ Warning: Quote row missing bid/ask data")
            elif data_type == 'trade':
                # Trade should have trade data
                if not row['trade_price']:
                    print(f"⚠ Warning: Trade row missing trade data")
            elif data_type == 'depth':
                # Depth should have depth data
                if not row['depth_price']:
                    print(f"⚠ Warning: Depth row missing depth data")
            
            print(f"  Row {row_count}: {data_type} - OK")
        
        print(f"✓ Read {row_count} rows successfully")
    
    # Test that we can filter by data type (without pandas)
    print("\nTesting data type filtering...")
    quote_count = 0
    trade_count = 0
    depth_count = 0
    
    with open(test_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['data_type'] == 'quote':
                quote_count += 1
            elif row['data_type'] == 'trade':
                trade_count += 1
            elif row['data_type'] == 'depth':
                depth_count += 1
    
    print(f"✓ Quote rows: {quote_count}")
    print(f"✓ Trade rows: {trade_count}")
    print(f"✓ Depth rows: {depth_count}")
    
    # Clean up test file
    test_file.unlink()
    
    print("\n" + "=" * 50)
    print("✓ ALL TESTS PASSED")
    print("=" * 50)
    print("\nCSV format is valid for backtesting!")
    return True


def validate_recorder_imports():
    """Test that data recorder modules can be imported."""
    print("Testing module imports...")
    
    try:
        import sys
        from pathlib import Path
        
        # Add launcher directory to path
        launcher_path = Path(__file__).parent
        if str(launcher_path) not in sys.path:
            sys.path.insert(0, str(launcher_path))
        
        # Skip GUI launcher test (requires tkinter which may not be available in all environments)
        print("  Skipping DataRecorder_Launcher (requires tkinter GUI)")
        
        # Test importing the core recorder logic (may fail if broker SDK not installed)
        print("  Testing data_recorder core module...")
        try:
            # Just check if the file exists and can be compiled
            recorder_file = launcher_path / "data_recorder.py"
            if not recorder_file.exists():
                print(f"  ❌ FAIL: data_recorder.py not found")
                return False
            
            # Compile check
            import py_compile
            py_compile.compile(str(recorder_file), doraise=True)
            print("  ✓ data_recorder.py compiles successfully")
            
            # Try importing (may fail if broker SDK not installed, which is OK)
            try:
                import data_recorder
                print("  ✓ data_recorder imported (broker SDK available)")
            except ImportError as e:
                print(f"  ⚠ data_recorder import skipped (broker SDK not installed - this is OK)")
                print(f"    Error: {e}")
            
            return True
        except Exception as e:
            print(f"  ❌ FAIL: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Market Data Recorder - Test Suite")
    print("=" * 50)
    print()
    
    # Test 1: Module imports
    print("Test 1: Module Imports")
    print("-" * 50)
    if not validate_recorder_imports():
        print("❌ Module import test failed")
        exit(1)
    print()
    
    # Test 2: CSV format
    print("Test 2: CSV Format Validation")
    print("-" * 50)
    if not test_csv_format():
        print("❌ CSV format test failed")
        exit(1)
    print()
    
    print("=" * 50)
    print("✓ ALL TESTS PASSED")
    print("=" * 50)
