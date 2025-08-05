#!/usr/bin/env python3
"""
Sample Data Download Script

Downloads historical data for backtesting.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    print("Sample data download script")
    print("This script will be implemented in Phase 2: Data Acquisition")
    print("Will download historical data from IBKR API for backtesting")

if __name__ == "__main__":
    main()
