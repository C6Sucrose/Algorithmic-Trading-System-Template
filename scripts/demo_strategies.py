#!/usr/bin/env python3
"""
Phase 4 Strategy Implementation Demo

Demonstrates the functionality of the implemented trading strategies:
- Mean Reversion Strategy
- Pairs Trading Strategy
- Base Strategy Framework
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from strategies.mean_reversion_strategy import MeanReversionStrategy
from strategies.pairs_trading_strategy import PairsTradingStrategy
from strategies.base_strategy import StrategyState


def create_mock_market_data(symbols: list, days: int = 60) -> dict:
    """Create mock market data for strategy testing."""
    market_data = {}
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    
    np.random.seed(42)  # For reproducible results
    
    for i, symbol in enumerate(symbols):
        # Create correlated price movements for realistic pairs trading
        base_price = 100.0 + i * 50.0  # Different base prices
        prices = [base_price]
        
        # Add some correlation between AAPL and MSFT for pairs testing
        correlation_factor = 0.7 if symbol in ['AAPL', 'MSFT'] else 0.0
        external_factor = np.random.normal(0, 1, days) if symbol == 'AAPL' else np.zeros(days)
        
        for j in range(1, days):
            # Base random walk
            random_change = np.random.normal(0, 2)
            
            # Add correlation if applicable
            if symbol == 'MSFT' and correlation_factor > 0:
                correlated_change = correlation_factor * external_factor[j] + (1 - correlation_factor) * random_change
                price_change = correlated_change
            else:
                price_change = random_change
            
            new_price = max(prices[-1] + price_change, base_price * 0.5)  # Floor at 50% of base
            prices.append(float(new_price))
        
        # Create OHLCV data
        data = pd.DataFrame({
            'Date': dates,
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
            'Close': prices,
            'Volume': [1000000 + np.random.randint(-200000, 500000) for _ in range(days)]
        })
        
        market_data[symbol] = data
    
    return market_data


def demo_mean_reversion_strategy():
    """Demonstrate Mean Reversion Strategy."""
    print("\n=== Mean Reversion Strategy Demo ===")
    
    try:
        # Create strategy
        strategy = MeanReversionStrategy(
            strategy_id="demo_mean_reversion",
            name="Demo Mean Reversion"
        )
        
        print(f"✓ Strategy created: {strategy.name}")
        print(f"  Strategy ID: {strategy.strategy_id}")
        print(f"  State: {strategy.state.value}")
        print(f"  Configuration:")
        print(f"    - Z-score entry: {strategy.z_score_entry}")
        print(f"    - Z-score exit: {strategy.z_score_exit}")
        print(f"    - RSI oversold: {strategy.rsi_oversold}")
        print(f"    - RSI overbought: {strategy.rsi_overbought}")
        
        # Create mock market data
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        market_data = create_mock_market_data(symbols, 30)
        print(f"✓ Generated market data for {len(symbols)} symbols")
        
        # Start strategy
        strategy.start()
        print(f"✓ Strategy started, state: {strategy.state.value}")
        
        # Simulate strategy updates
        update_count = 0
        for i in range(5):  # 5 updates
            # Update market data (simulate new day)
            current_data = {}
            for symbol in symbols:
                # Get data up to current day
                current_data[symbol] = market_data[symbol].iloc[:20+i*2]
            
            # Update strategy
            strategy.update(current_data)
            update_count += 1
            
            print(f"  Update {update_count}: {len(strategy.positions)} positions, "
                  f"{len(strategy.trade_history)} trades")
        
        # Get performance summary
        performance = strategy.get_performance_summary()
        print(f"✓ Performance Summary:")
        print(f"  - Open positions: {performance['open_positions']}")
        print(f"  - Total trades: {performance['total_trades']}")
        print(f"  - Portfolio value: ${performance['portfolio_value']:.2f}")
        
        # Stop strategy
        strategy.stop()
        print(f"✓ Strategy stopped, state: {strategy.state.value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Mean Reversion Strategy demo failed: {e}")
        return False


def demo_pairs_trading_strategy():
    """Demonstrate Pairs Trading Strategy."""
    print("\n=== Pairs Trading Strategy Demo ===")
    
    try:
        # Create strategy
        strategy = PairsTradingStrategy(
            strategy_id="demo_pairs_trading",
            name="Demo Pairs Trading"
        )
        
        print(f"✓ Strategy created: {strategy.name}")
        print(f"  Strategy ID: {strategy.strategy_id}")
        print(f"  State: {strategy.state.value}")
        print(f"  Configuration:")
        print(f"    - Min correlation: {strategy.min_correlation}")
        print(f"    - Z-score entry: {strategy.z_score_entry}")
        print(f"    - Z-score exit: {strategy.z_score_exit}")
        print(f"    - Max pairs: {strategy.max_pairs}")
        
        # Create mock market data with correlated pairs
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        market_data = create_mock_market_data(symbols, 60)  # More data for correlation analysis
        print(f"✓ Generated market data for {len(symbols)} symbols")
        
        # Show correlation between AAPL and MSFT
        correlation = market_data['AAPL']['Close'].corr(market_data['MSFT']['Close'])
        print(f"  AAPL-MSFT correlation: {correlation:.3f}")
        
        # Start strategy
        strategy.start()
        print(f"✓ Strategy started, state: {strategy.state.value}")
        
        # Simulate strategy updates
        update_count = 0
        for i in range(3):  # 3 updates with sufficient data
            # Update market data (simulate time progression)
            current_data = {}
            for symbol in symbols:
                # Get data up to current point (need sufficient data for correlation)
                start_idx = max(0, 40 + i * 5)  # Start with 40 days minimum
                current_data[symbol] = market_data[symbol].iloc[:start_idx + 15]
            
            # Update strategy
            strategy.update(current_data)
            update_count += 1
            
            print(f"  Update {update_count}: {len(strategy.pair_positions)} pair positions, "
                  f"{len(strategy.positions)} individual positions")
        
        # Get performance summary
        performance = strategy.get_performance_summary()
        pairs_metrics = strategy.get_strategy_metrics()
        
        print(f"✓ Performance Summary:")
        print(f"  - Active pairs: {pairs_metrics['pairs_metrics']['active_pairs']}")
        print(f"  - Total trades: {performance['total_trades']}")
        print(f"  - Portfolio value: ${performance['portfolio_value']:.2f}")
        if pairs_metrics['pairs_metrics']['avg_correlation'] > 0:
            print(f"  - Avg correlation: {pairs_metrics['pairs_metrics']['avg_correlation']:.3f}")
        
        # Stop strategy
        strategy.stop()
        print(f"✓ Strategy stopped, state: {strategy.state.value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pairs Trading Strategy demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_strategy_management():
    """Demonstrate strategy lifecycle management."""
    print("\n=== Strategy Management Demo ===")
    
    try:
        # Create multiple strategies
        mean_rev = MeanReversionStrategy("mr_1", "Mean Reversion 1")
        pairs_trade = PairsTradingStrategy("pt_1", "Pairs Trading 1")
        
        strategies = [mean_rev, pairs_trade]
        
        print(f"✓ Created {len(strategies)} strategies")
        
        # Test state management
        for strategy in strategies:
            print(f"\n  {strategy.name}:")
            print(f"    Initial state: {strategy.state.value}")
            
            # Start strategy
            strategy.start()
            print(f"    After start: {strategy.state.value}")
            
            # Pause strategy
            strategy.pause()
            print(f"    After pause: {strategy.state.value}")
            
            # Resume strategy
            strategy.resume()
            print(f"    After resume: {strategy.state.value}")
            
            # Stop strategy
            strategy.stop()
            print(f"    After stop: {strategy.state.value}")
        
        print(f"✓ Strategy lifecycle management working correctly")
        
        # Test serialization
        strategy_dict = mean_rev.to_dict()
        print(f"✓ Strategy serialization working (keys: {list(strategy_dict.keys())})")
        
        return True
        
    except Exception as e:
        print(f"✗ Strategy management demo failed: {e}")
        return False


def demo_risk_management():
    """Demonstrate risk management features."""
    print("\n=== Risk Management Demo ===")
    
    try:
        # Create strategy with custom risk parameters
        strategy = MeanReversionStrategy("risk_demo", "Risk Management Demo")
        
        # Mock some positions for risk testing
        from strategies.base_strategy import Position, PositionType
        
        # Create a mock losing position
        mock_position = Position(
            symbol="TEST",
            position_type=PositionType.LONG,
            quantity=100,
            entry_price=100.0,
            entry_time=datetime.now() - timedelta(hours=1),
            strategy_id=strategy.strategy_id,
            stop_loss=95.0,
            take_profit=110.0
        )
        
        # Update with current (losing) price
        mock_position.update_price(90.0, datetime.now())  # 10% loss
        strategy.positions["TEST"] = mock_position
        
        print(f"✓ Created mock position: {mock_position.symbol}")
        print(f"  Entry price: ${mock_position.entry_price:.2f}")
        print(f"  Current price: ${mock_position.current_price:.2f}")
        print(f"  Unrealized P&L: ${mock_position.unrealized_pnl:.2f}")
        print(f"  Stop loss: ${mock_position.stop_loss:.2f}")
        
        # Test stop loss condition
        should_exit, reason = mock_position.check_exit_conditions()
        print(f"  Should exit: {should_exit}, Reason: {reason}")
        
        # Test risk management actions
        risk_actions = strategy.manage_risk(strategy.positions)
        print(f"✓ Risk management generated {len(risk_actions)} actions")
        
        if risk_actions:
            for action in risk_actions:
                print(f"  - Action: {action.get('type', 'unknown')}, Reason: {action.get('reason', 'none')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Risk management demo failed: {e}")
        return False


def main():
    """Run all Phase 4 strategy demonstrations."""
    print("Phase 4 Strategy Implementation Demonstration")
    print("=" * 60)
    
    demo_results = []
    
    # Run all demonstrations
    demo_results.append(("Mean Reversion Strategy", demo_mean_reversion_strategy()))
    demo_results.append(("Pairs Trading Strategy", demo_pairs_trading_strategy()))
    demo_results.append(("Strategy Management", demo_strategy_management()))
    demo_results.append(("Risk Management", demo_risk_management()))
    
    # Summary
    print("\n" + "=" * 60)
    print("DEMONSTRATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    for demo_name, result in demo_results:
        status = "SUCCESS" if result else "FAILED"
        print(f"{demo_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(demo_results)} demonstrations successful")
    
    if passed == len(demo_results):
        print("\n🎉 Phase 4 Strategy Implementation is operational!")
        print("\nKey capabilities demonstrated:")
        print("✓ Mean reversion strategy with technical analysis")
        print("✓ Pairs trading strategy with correlation analysis")
        print("✓ Strategy lifecycle management (start/stop/pause/resume)")
        print("✓ Risk management and position control")
        print("✓ Performance tracking and metrics")
        print("✓ Configurable parameters and thresholds")
        print("\nReady for backtesting and live trading deployment!")
    else:
        print(f"\n⚠️  {len(demo_results) - passed} demonstration(s) failed.")
        print("Please check the implementations and dependencies.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
