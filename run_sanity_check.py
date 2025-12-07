#!/usr/bin/env python
"""Simple test runner for sanity check"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.sanity_check import sanity_check_layer

if __name__ == '__main__':
    print("Running sanity check...")
    print()
    success, stats = sanity_check_layer(verbose=True)
    print()
    if success:
        print("✓ ALL CHECKS PASSED - Numerical correctness verified!")
        sys.exit(0)
    else:
        print("✗ SOME CHECKS FAILED - Please review the output above")
        sys.exit(1)

