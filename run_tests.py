#!/usr/bin/env python
"""
Simple test runner for the memristor neural network framework.

Run with: python run_tests.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == '__main__':
    try:
        import pytest
    except ImportError:
        print("pytest not installed. Install with: pip install pytest")
        sys.exit(1)
    
    # Run tests
    test_dir = os.path.join(os.path.dirname(__file__), 'tests')
    exit_code = pytest.main([test_dir, '-v'])
    sys.exit(exit_code)


