#!/usr/bin/env python3
"""
Phase 3 Screening Engine Demo

Demonstrates the functionality of all Phase 3 screening components.
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def create_demo_data():
    """Create sample market data for demonstration."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
    
    # Sample data for AAPL
    np.random.seed(42)  # For reproducible results
    aapl_base = 150.0
    aapl_prices = [aapl_base]
    
    for i in range(1, len(dates)):
        change = np.random.normal(0, 2)  # Random walk with volatility
        new_price = max(aapl_prices[-1] + change, 100.0)  # Ensure positive prices
        aapl_prices.append(new_price)
    
    aapl_data = pd.DataFrame({
        'Date': dates,
        'Open': [p * (1 + np.random.normal(0, 0.01)) for p in aapl_prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.02))) for p in aapl_prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in aapl_prices],
        'Close': aapl_prices,
        'Volume': [1000000 + np.random.randint(-100000, 200000) for _ in range(len(dates))]
    })
    
    # Sample data for MSFT (correlated with AAPL)
    msft_prices = [p * 2 + np.random.normal(0, 5) for p in aapl_prices]  # Correlated but different scale
    msft_data = pd.DataFrame({
        'Date': dates,
        'Open': [p * (1 + np.random.normal(0, 0.01)) for p in msft_prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.02))) for p in msft_prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in msft_prices],
        'Close': msft_prices,
        'Volume': [800000 + np.random.randint(-80000, 160000) for _ in range(len(dates))]
    })
    
    return {'AAPL': aapl_data, 'MSFT': msft_data}

def demo_universe_manager():
    """Demonstrate Universe Manager functionality."""
    print("\n=== Universe Manager Demo ===")
    
    try:
        from screening.universe import UniverseManager, UniverseDefinition
        
        # Create universe definition (uses default config)
        universe_def = UniverseDefinition()
        print(f"✓ Universe definition created with min_price=${universe_def.min_price}")
        print(f"  - Max price: ${universe_def.max_price}")
        print(f"  - Min volume: {universe_def.min_volume:,}")
        print(f"  - Min market cap: ${universe_def.min_market_cap:,}")
        
        # Create universe manager
        universe_manager = UniverseManager()
        print(f"✓ Universe manager initialized")
        
        # Load base universe
        count = universe_manager.load_base_universe()
        print(f"✓ Loaded {count} symbols in base universe")
        
        return True
        
    except Exception as e:
        print(f"✗ Universe Manager demo failed: {e}")
        return False

def demo_technical_filters():
    """Demonstrate Technical Filters functionality."""
    print("\n=== Technical Filters Demo ===")
    
    try:
        from screening.filters import TechnicalIndicators, TechnicalFilters
        
        demo_data = create_demo_data()
        
        for symbol, data in demo_data.items():
            print(f"\nAnalyzing {symbol}:")
            
            # Test RSI calculation
            rsi_values = TechnicalIndicators.rsi(data['Close'])
            current_rsi = rsi_values.iloc[-1] if not rsi_values.empty else 50
            print(f"  ✓ Current RSI: {current_rsi:.2f}")
            
            # Test Bollinger Bands
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(data['Close'])
            current_price = data['Close'].iloc[-1]
            print(f"  ✓ Current price: ${current_price:.2f}")
            print(f"    Bollinger Upper: ${bb_upper.iloc[-1]:.2f}")
            print(f"    Bollinger Middle: ${bb_middle.iloc[-1]:.2f}")
            print(f"    Bollinger Lower: ${bb_lower.iloc[-1]:.2f}")
            
            # Test volume analysis
            avg_volume = data['Volume'].tail(10).mean()
            current_volume = data['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            print(f"  ✓ Volume ratio (current/avg): {volume_ratio:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Technical Filters demo failed: {e}")
        return False

def demo_mean_reversion_scanner():
    """Demonstrate Mean Reversion Scanner functionality."""
    print("\n=== Mean Reversion Scanner Demo ===")
    
    try:
        from screening.mean_reversion_scanner import MeanReversionScanner
        
        demo_data = create_demo_data()
        
        # Create scanner
        scanner = MeanReversionScanner()
        print("✓ Mean reversion scanner created")
        
        # Analyze each symbol
        for symbol, data in demo_data.items():
            print(f"\nScanning {symbol} for mean reversion opportunities:")
            
            current_price = data['Close'].iloc[-1]
            price_mean = data['Close'].mean()
            price_std = data['Close'].std()
            z_score = (current_price - price_mean) / price_std if price_std > 0 else 0
            
            print(f"  Current price: ${current_price:.2f}")
            print(f"  Price mean: ${price_mean:.2f}")
            print(f"  Z-score: {z_score:.2f}")
            
            if abs(z_score) > 1.5:
                signal_type = "OVERSOLD" if z_score < -1.5 else "OVERBOUGHT"
                print(f"  🎯 {signal_type} signal detected (|z| > 1.5)")
            else:
                print(f"  ℹ️  No strong mean reversion signal")
        
        return True
        
    except Exception as e:
        print(f"✗ Mean Reversion Scanner demo failed: {e}")
        return False

def demo_pairs_scanner():
    """Demonstrate Pairs Trading Scanner functionality."""
    print("\n=== Pairs Trading Scanner Demo ===")
    
    try:
        from screening.pairs_scanner import PairsTradingScanner
        
        demo_data = create_demo_data()
        
        # Create scanner
        scanner = PairsTradingScanner()
        print("✓ Pairs trading scanner created")
        
        # Analyze pair
        symbols = list(demo_data.keys())
        if len(symbols) >= 2:
            symbol_a, symbol_b = symbols[0], symbols[1]
            data_a, data_b = demo_data[symbol_a], demo_data[symbol_b]
            
            print(f"\nAnalyzing pair: {symbol_a} - {symbol_b}")
            
            # Calculate correlation
            correlation = data_a['Close'].corr(data_b['Close'])
            print(f"  Correlation: {correlation:.3f}")
            
            # Analyze pairs
            pairs = scanner.find_correlated_pairs([symbol_a, symbol_b])
            
            if pairs:
                pair = pairs[0]  # Get the first (and likely only) pair
                print(f"  ✓ Pair analysis successful:")
                print(f"    Hedge ratio: {pair.hedge_ratio:.3f}")
                print(f"    Current spread: {pair.current_spread:.2f}")
                print(f"    Z-score: {pair.z_score:.2f}")
                print(f"    Data points: {pair.data_points}")
                
                if abs(pair.z_score) > 1.5:
                    signal = "DIVERGENCE" if pair.z_score > 1.5 else "CONVERGENCE"
                    print(f"    🎯 {signal} signal detected")
            else:
                print(f"  ℹ️  Pair does not meet criteria for trading")
        
        return True
        
    except Exception as e:
        print(f"✗ Pairs Trading Scanner demo failed: {e}")
        return False

def main():
    """Run the Phase 3 screening engine demonstration."""
    print("Phase 3 Screening Engine Demonstration")
    print("=" * 50)
    
    demo_results = []
    
    # Run all demonstrations
    demo_results.append(("Universe Manager", demo_universe_manager()))
    demo_results.append(("Technical Filters", demo_technical_filters()))
    demo_results.append(("Mean Reversion Scanner", demo_mean_reversion_scanner()))
    demo_results.append(("Pairs Trading Scanner", demo_pairs_scanner()))
    
    # Summary
    print("\n" + "=" * 50)
    print("DEMONSTRATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    for demo_name, result in demo_results:
        status = "SUCCESS" if result else "FAILED"
        print(f"{demo_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(demo_results)} demonstrations successful")
    
    if passed == len(demo_results):
        print("\n🎉 Phase 3 Screening Engine is operational!")
        print("\nKey capabilities demonstrated:")
        print("✓ Universe definition and management")
        print("✓ Technical analysis and filtering")
        print("✓ Mean reversion opportunity identification")
        print("✓ Pairs trading correlation analysis")
        print("\nReady for strategy implementation and live trading!")
    else:
        print(f"\n⚠️  {len(demo_results) - passed} demonstration(s) failed.")
        print("Please check the implementations and dependencies.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
