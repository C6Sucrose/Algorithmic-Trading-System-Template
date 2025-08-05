#!/usr/bin/env python3
"""
Quick system verification script
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    print("=" * 50)
    print("ALGORITHMIC TRADING SYSTEM - QUICK CHECK")
    print("=" * 50)
    
    try:
        # Test configuration manager
        from config_manager import config
        print("✓ Config manager loaded successfully")
        
        # Test IBKR configuration
        ibkr = config.get_ibkr_config()
        print(f"✓ IBKR config: {ibkr['host']}:{ibkr['port']}")
        
        # Test risk configuration
        risk = config.get_risk_config()
        print(f"✓ Risk target: EUR {risk['daily_profit_target']}")
        
        # Test strategy configuration
        mean_rev = config.get_strategy_config('mean_reversion')
        if mean_rev:
            print(f"✓ Mean reversion strategy configured")
        
        pairs = config.get_strategy_config('pairs_trading')
        if pairs:
            print(f"✓ Pairs trading strategy configured")
        
        print("=" * 50)
        print("SUCCESS: Environment setup complete!")
        print("Ready to begin Phase 2: Data Management")
        print("=" * 50)
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
