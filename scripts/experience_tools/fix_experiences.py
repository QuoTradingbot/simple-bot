#!/usr/bin/env python3
"""Fix corrupted signal_experience.json file"""
import json
import os
from pathlib import Path

def fix_experience_file():
    # Get project root (2 levels up from this file)
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    filepath = PROJECT_ROOT / 'experiences' / 'ES' / 'signal_experience.json'
    
    # Read the file
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Try to parse as JSON first
    try:
        data = json.loads(content)
        if isinstance(data, dict) and 'experiences' in data:
            experiences = data['experiences']
            print(f"File is valid JSON with {len(experiences)} experiences")
            return experiences
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
    
    # Manual extraction
    experiences = []
    lines = content.split('\n')
    current_obj = []
    brace_count = 0
    in_object = False
    
    for line in lines:
        # Start of new object
        if '"timestamp"' in line and brace_count == 0:
            if current_obj:
                # Try to parse previous object
                obj_str = '\n'.join(current_obj).strip().rstrip(',')
                try:
                    obj = json.loads(obj_str)
                    experiences.append(obj)
                except:
                    pass
            current_obj = ['{']  # Start new object
            in_object = True
            brace_count = 1
        
        if in_object:
            if '{' not in line or '"timestamp"' not in line:
                current_obj.append(line)
            brace_count += line.count('{') - line.count('}')
            
            if brace_count == 0:
                obj_str = '\n'.join(current_obj).strip().rstrip(',')
                try:
                    obj = json.loads(obj_str)
                    experiences.append(obj)
                except Exception as e:
                    print(f"Failed to parse object: {e}")
                current_obj = []
                in_object = False
    
    # Check for duplicates
    unique_experiences = []
    seen = set()
    duplicates = 0
    
    for exp in experiences:
        key = (exp['timestamp'], exp['symbol'], exp['pnl'], exp['exit_reason'])
        if key not in seen:
            seen.add(key)
            unique_experiences.append(exp)
        else:
            duplicates += 1
    
    print(f"\nResults:")
    print(f"Total experiences found: {len(experiences)}")
    print(f"Unique experiences: {len(unique_experiences)}")
    print(f"Duplicates removed: {duplicates}")
    
    if unique_experiences:
        print(f"First: {unique_experiences[0]['timestamp']}")
        print(f"Last: {unique_experiences[-1]['timestamp']}")
        
        # Create backup
        backup_path = filepath + '.backup'
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"\nBackup saved to: {backup_path}")
        
        # Save fixed file
        fixed_data = {
            "experiences": unique_experiences,
            "stats": {
                "total_signals": len(unique_experiences),
                "taken": len(unique_experiences),
                "skipped": 0,
                "take_rate": 100.0,
                "recent_pnl": sum(e['pnl'] for e in unique_experiences[-10:]) if len(unique_experiences) >= 10 else sum(e['pnl'] for e in unique_experiences),
                "recent_win_rate": sum(1 for e in unique_experiences[-10:] if e['pnl'] > 0) / min(10, len(unique_experiences)) * 100 if unique_experiences else 0.0,
                "exploration_rate": 100.0
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(fixed_data, f, indent=2)
        
        print(f"Fixed file saved with {len(unique_experiences)} experiences")
        return unique_experiences
    
    return []

if __name__ == '__main__':
    fix_experience_file()
