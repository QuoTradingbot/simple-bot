#!/usr/bin/env python3
"""
Comprehensive Experience Deduplication Script
==============================================
Removes truly identical experiences from RL experience files.

Checks ALL significant fields to ensure only exact duplicates are removed:
- timestamp, symbol, price, pnl, duration, took_trade
- regime, rsi, vwap_distance, atr, exit_reason
- mfe, mae, returns, volume_ratio, etc.

Usage:
    python scripts/deduplicate_experiences.py                    # Process all symbols
    python scripts/deduplicate_experiences.py --symbol ES        # Process specific symbol
    python scripts/deduplicate_experiences.py --dry-run          # Preview without changes
"""

import json
import os
import sys
import argparse
import hashlib
from pathlib import Path

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def experience_hash(exp):
    """
    Create a hash of an experience based on ALL significant fields.
    Only experiences with identical hashes are considered duplicates.
    
    Args:
        exp: Experience dictionary
        
    Returns:
        Hash string representing the experience
    """
    # ALL fields that make an experience unique
    key_fields = [
        'timestamp', 'symbol', 'price', 'pnl', 'duration', 'took_trade',
        'regime', 'volatility_regime', 'rsi', 'vwap_distance', 'vwap_slope',
        'atr', 'atr_slope', 'macd_hist', 'stoch_k',
        'volume_ratio', 'volume_slope', 'hour', 'session',
        'exit_reason', 'mfe', 'mae', 'returns',
        'order_type_used', 'entry_slippage_ticks', 'exploration_rate'
    ]
    
    values = []
    for field in key_fields:
        if field in exp:
            val = exp[field]
            # Round floats to 6 decimal places to avoid precision issues
            if isinstance(val, float):
                val = round(val, 6)
            values.append(str(val))
        else:
            values.append('')  # Empty string for missing fields
    
    # Create hash from concatenated values
    key_string = '|'.join(values)
    return hashlib.md5(key_string.encode()).hexdigest()


def deduplicate_file(filepath, dry_run=False):
    """
    Remove exact duplicates from a single experience file.
    
    Args:
        filepath: Path to experience JSON file
        dry_run: If True, don't save changes (just report)
        
    Returns:
        Tuple of (original_count, unique_count, duplicates_removed)
    """
    if not os.path.exists(filepath):
        print(f"⚠️  File not found: {filepath}")
        return 0, 0, 0
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle both wrapped and unwrapped formats
        if isinstance(data, dict) and 'experiences' in data:
            experiences = data['experiences']
            is_wrapped = True
        else:
            experiences = data
            is_wrapped = False
        
        original_count = len(experiences)
        
        # Use hash set for O(1) duplicate detection
        seen_hashes = set()
        unique_experiences = []
        duplicates = 0
        
        for exp in experiences:
            exp_hash = experience_hash(exp)
            
            if exp_hash not in seen_hashes:
                seen_hashes.add(exp_hash)
                unique_experiences.append(exp)
            else:
                duplicates += 1
        
        unique_count = len(unique_experiences)
        
        # Save if not dry run and duplicates found
        if not dry_run and duplicates > 0:
            if is_wrapped:
                data['experiences'] = unique_experiences
            else:
                data = unique_experiences
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        return original_count, unique_count, duplicates
        
    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return 0, 0, 0


def main():
    parser = argparse.ArgumentParser(
        description='Deduplicate RL experience files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all symbols
  python scripts/deduplicate_experiences.py
  
  # Process specific symbol
  python scripts/deduplicate_experiences.py --symbol ES
  
  # Dry run (preview without changes)
  python scripts/deduplicate_experiences.py --dry-run
        """
    )
    parser.add_argument('--symbol', type=str, help='Specific symbol to process (e.g., ES, MES, NQ)')
    parser.add_argument('--dry-run', action='store_true', help='Preview without making changes')
    
    args = parser.parse_args()
    
    experiences_dir = PROJECT_ROOT / 'experiences'
    
    if not experiences_dir.exists():
        print(f"❌ Experiences directory not found: {experiences_dir}")
        return 1
    
    print("=" * 80)
    print("EXPERIENCE DEDUPLICATION")
    print("=" * 80)
    if args.dry_run:
        print("DRY RUN MODE - No changes will be saved")
    print()
    
    # Get list of symbols to process
    if args.symbol:
        symbols = [args.symbol]
    else:
        symbols = [d.name for d in experiences_dir.iterdir() if d.is_dir()]
    
    total_original = 0
    total_unique = 0
    total_duplicates = 0
    
    for symbol in symbols:
        exp_file = experiences_dir / symbol / 'signal_experience.json'
        
        if not exp_file.exists():
            print(f"⚠️  {symbol}: No experience file found")
            continue
        
        original, unique, dups = deduplicate_file(exp_file, dry_run=args.dry_run)
        
        total_original += original
        total_unique += unique
        total_duplicates += dups
        
        if dups > 0:
            status = "DRY RUN - Would remove" if args.dry_run else "✅ Removed"
            print(f"{symbol:6} {status:20} {dups:4} duplicates ({original} → {unique})")
        else:
            print(f"{symbol:6} ✅ Clean              {original:4} experiences (no duplicates)")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total original:   {total_original}")
    print(f"Total unique:     {total_unique}")
    print(f"Duplicates found: {total_duplicates}")
    
    if args.dry_run and total_duplicates > 0:
        print()
        print("Run without --dry-run to remove duplicates")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
