#!/usr/bin/env python3
"""
Build QuoTrading Launcher EXE for customer distribution
Uses PyInstaller to create standalone executable with all dependencies
"""

import PyInstaller.__main__
import sys
from pathlib import Path

# Get project root
ROOT = Path(__file__).parent

def build_exe():
    """Build standalone EXE with PyInstaller"""
    
    import os
    
    PyInstaller.__main__.run([
        str(ROOT / 'launcher' / 'QuoTrading_Launcher.py'),  # Main script
        '--name=QuoTrading',                                 # EXE name
        '--onefile',                                         # Single EXE file
        '--windowed',                                        # No console window (GUI only)
        '--icon=NONE',                                       # Add icon later if you have one
        
        # Add source code files only (no data directory)
        f'--add-data={ROOT / "src"}{os.pathsep}{os.path.join(".", "src")}',  # Bundle src/ folder
        # NOTE: data/ directory is NOT bundled - RL is cloud-based for production
        
        # Hidden imports (modules PyInstaller might miss)
        '--hidden-import=tkinter',
        '--hidden-import=tkinter.ttk',
        '--hidden-import=aiohttp',
        '--hidden-import=asyncio',
        '--hidden-import=websockets',
        '--hidden-import=pandas',
        '--hidden-import=numpy',
        
        # Output directory
        f'--distpath={ROOT / "dist"}',
        f'--workpath={ROOT / "build"}',
        f'--specpath={ROOT}',
        
        # Clean build
        '--clean',
        '--noconfirm',
    ])
    
    print("\n" + "="*60)
    print("✅ EXE Build Complete!")
    print("="*60)
    print(f"📦 Output: {ROOT / 'dist' / 'QuoTrading.exe'}")
    print(f"📏 Size: {(ROOT / 'dist' / 'QuoTrading.exe').stat().st_size / 1024 / 1024:.1f} MB")
    print("\nCustomers can run this single EXE file - no Python needed!")
    print("="*60)

if __name__ == "__main__":
    import os
    build_exe()
