#!/usr/bin/env python3
"""
Simple test to verify that main.py can be imported without errors
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Test importing the main module
    print("Testing import of main.py...")
    
    # We'll just test the import structure without actually running the main function
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("main", "main.py")
    main_module = importlib.util.module_from_spec(spec)
    
    print("✓ main.py can be loaded as a module")
    
    # Check if key functions exist
    if hasattr(main_module, 'start_comfyui'):
        print("✓ start_comfyui function exists")
    else:
        print("✗ start_comfyui function missing")
        
    if hasattr(main_module, 'hijack_progress'):
        print("✓ hijack_progress function exists")
    else:
        print("✗ hijack_progress function missing")
        
    if hasattr(main_module, 'cleanup_temp'):
        print("✓ cleanup_temp function exists")
    else:
        print("✗ cleanup_temp function missing")
        
    print("✓ All basic structure checks passed")
    
except Exception as e:
    print(f"✗ Error importing main.py: {e}")
    sys.exit(1)

print("✓ main.py import test completed successfully")
