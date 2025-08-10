#!/usr/bin/env python3
"""
Order Management System (OMS) Demonstration

This demo showcases the comprehensive capabilities of our Order Management System
including order placement, risk controls, position tracking, and execution monitoring.
"""

import logging
import time
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Add the src directory to the path
sys.path.append('../src')

from src.order_management import (
    OrderManager, BrokerInterface, MockBrokerInterface, Order, Fill, 
    OrderStatus, OrderType, PositionManager, RiskController, 
    ExecutionEngine, ExecutionAlgorithm
)
from src.order_management.risk_controls import RiskLimits
from src.strategies.base_strategy import TradeDirection


class OMSDemo:
    """Comprehensive OMS demonstration."""
    
    def __init__(self):
        """Initialize the OMS demo."""
        self.broker = None
        self.order_manager = None
        self.position_manager = None
        self.risk_controller = None
        self.execution_engine = None
        
        # Demo state
        self.running = False
        self.orders_placed = []
        
    def setup_oms(self):
        """Set up the Order Management System components."""
        logger.info("Setting up Order Management System...")
        
        # Initialize broker interface
        self.broker = MockBrokerInterface({})
        logger.info("✓ Broker interface initialized")
        
        # Initialize risk controller with demo limits
        risk_limits = RiskLimits(
            max_position_size=1000.0,
            max_daily_loss=5000.0,
            max_portfolio_exposure=50000.0,
            max_single_stock_exposure=10000.0,
            max_drawdown_percent=10.0
        )
        self.risk_controller = RiskController(limits=risk_limits)
        logger.info("✓ Risk controller initialized with limits")
        
        # Initialize position manager
        initial_capital = 100000.0
        self.position_manager = PositionManager(initial_capital=initial_capital)
        logger.info(f"✓ Position manager initialized with ${initial_capital:,.2f} capital")
        
        # Initialize order manager
        self.order_manager = OrderManager(broker=self.broker)
        logger.info("✓ Order manager initialized")
        
        # Initialize execution engine
        self.execution_engine = ExecutionEngine(
            broker=self.broker,
            order_manager=self.order_manager,
            position_manager=self.position_manager,
            risk_controller=self.risk_controller
        )
        logger.info("✓ Execution engine initialized")
        
        # Set up callbacks
        self.order_manager.add_fill_callback(self._on_fill)
        self.execution_engine.add_execution_callback(self._on_execution_complete)
        
        logger.info("🚀 Order Management System setup complete!\n")
    
    def start_oms(self):
        """Start the OMS components."""
        logger.info("Starting OMS components...")
        
        success = True
        
        if not self.order_manager.start():
            logger.error("Failed to start Order Manager")
            success = False
        
        if not self.execution_engine.start():
            logger.error("Failed to start Execution Engine")
            success = False
        
        if success:
            logger.info("✓ All OMS components started successfully\n")
            self.running = True
        else:
            logger.error("❌ Failed to start some OMS components")
        
        return success
    
    def stop_oms(self):
        """Stop the OMS components."""
        logger.info("\nStopping OMS components...")
        
        self.running = False
        
        if self.execution_engine:
            self.execution_engine.stop()
        
        if self.order_manager:
            self.order_manager.stop()
        
        logger.info("✓ OMS components stopped")
    
    def simulate_market_data(self):
        """Simulate market data updates."""
        market_data = {
            'AAPL': {'bid': 150.25, 'ask': 150.30, 'last': 150.28, 'volume': 1000},
            'MSFT': {'bid': 280.50, 'ask': 280.55, 'last': 280.52, 'volume': 500},
            'GOOGL': {'bid': 2500.00, 'ask': 2500.10, 'last': 2500.05, 'volume': 200},
            'TSLA': {'bid': 200.75, 'ask': 200.85, 'last': 200.80, 'volume': 800}
        }
        
        for symbol, data in market_data.items():
            self.position_manager.update_market_data(symbol, data)
            self.execution_engine.update_market_data(symbol, data)
        
        logger.info("📊 Market data updated for all symbols")
    
    def demo_immediate_execution(self):
        """Demonstrate immediate order execution."""
        logger.info("=== IMMEDIATE EXECUTION DEMO ===")
        
        # Place a market buy order
        execution_id = self.execution_engine.execute_order(
            symbol='AAPL',
            direction=TradeDirection.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            algorithm=ExecutionAlgorithm.IMMEDIATE,
            strategy_id='demo_strategy'
        )
        
        if execution_id:
            logger.info(f"✓ Immediate execution queued: {execution_id}")
            time.sleep(2)  # Allow time for execution
            
            # Check execution status
            status = self.execution_engine.get_execution_status(execution_id)
            if status:
                logger.info(f"Execution status: {status['status']}")
        else:
            logger.error("❌ Failed to queue immediate execution")
    
    def demo_limit_orders(self):
        """Demonstrate limit order placement."""
        logger.info("\n=== LIMIT ORDER DEMO ===")
        
        # Place a limit buy order below market
        order_id = self.order_manager.place_order(
            symbol='MSFT',
            direction=TradeDirection.BUY,
            quantity=50,
            order_type=OrderType.LIMIT,
            price=280.00,  # Below current ask of 280.55
            strategy_id='demo_strategy'
        )
        
        if order_id:
            logger.info(f"✓ Limit buy order placed: {order_id} (MSFT 50 @ $280.00)")
            self.orders_placed.append(order_id)
            
            # Check order status
            time.sleep(1)
            orders = self.order_manager.get_orders()
            if order_id in orders:
                order = orders[order_id]
                logger.info(f"Order status: {order.status.value}")
        else:
            logger.error("❌ Failed to place limit order")
    
    def demo_risk_controls(self):
        """Demonstrate risk control validation."""
        logger.info("\n=== RISK CONTROLS DEMO ===")
        
        # Try to place an order that exceeds position limits
        logger.info("Testing position size limit...")
        order_id = self.order_manager.place_order(
            symbol='GOOGL',
            direction=TradeDirection.BUY,
            quantity=50,  # This should exceed max position size based on price
            order_type=OrderType.MARKET,
            strategy_id='demo_strategy'
        )
        
        if order_id:
            logger.info(f"✓ Large order placed: {order_id}")
            self.orders_placed.append(order_id)
        else:
            logger.info("✓ Large order rejected by risk controls (as expected)")
        
        # Display current risk metrics
        risk_metrics = self.risk_controller.get_risk_metrics()
        logger.info("Current risk metrics:")
        for metric, value in risk_metrics.items():
            logger.info(f"  {metric}: {value}")
    
    def demo_position_tracking(self):
        """Demonstrate position tracking capabilities."""
        logger.info("\n=== POSITION TRACKING DEMO ===")
        
        # Get current portfolio
        portfolio = self.position_manager.get_portfolio()
        logger.info(f"Portfolio value: ${portfolio['total_value']:,.2f}")
        logger.info(f"Cash available: ${portfolio['cash']:,.2f}")
        logger.info(f"Total P&L: ${portfolio['total_pnl']:,.2f}")
        
        # Show positions
        positions = self.position_manager.get_positions()
        if positions:
            logger.info("Current positions:")
            for symbol, position in positions.items():
                logger.info(f"  {symbol}: {position['quantity']} shares, "
                          f"P&L: ${position['unrealized_pnl']:,.2f}")
        else:
            logger.info("No current positions")
    
    def demo_execution_algorithms(self):
        """Demonstrate different execution algorithms."""
        logger.info("\n=== EXECUTION ALGORITHMS DEMO ===")
        
        # TWAP execution
        logger.info("Testing TWAP execution algorithm...")
        execution_id = self.execution_engine.execute_order(
            symbol='TSLA',
            direction=TradeDirection.BUY,
            quantity=200,
            algorithm=ExecutionAlgorithm.TWAP,
            strategy_id='demo_strategy',
            duration_minutes=30,
            num_slices=5
        )
        
        if execution_id:
            logger.info(f"✓ TWAP execution queued: {execution_id}")
        
        # Iceberg execution
        logger.info("Testing Iceberg execution algorithm...")
        execution_id = self.execution_engine.execute_order(
            symbol='AAPL',
            direction=TradeDirection.SELL,
            quantity=500,
            algorithm=ExecutionAlgorithm.ICEBERG,
            strategy_id='demo_strategy',
            visible_quantity=100
        )
        
        if execution_id:
            logger.info(f"✓ Iceberg execution queued: {execution_id}")
        
        # Smart execution
        logger.info("Testing Smart execution algorithm...")
        execution_id = self.execution_engine.execute_order(
            symbol='MSFT',
            direction=TradeDirection.BUY,
            quantity=75,
            algorithm=ExecutionAlgorithm.SMART,
            strategy_id='demo_strategy'
        )
        
        if execution_id:
            logger.info(f"✓ Smart execution queued: {execution_id}")
    
    def demo_performance_monitoring(self):
        """Demonstrate performance monitoring capabilities."""
        logger.info("\n=== PERFORMANCE MONITORING DEMO ===")
        
        # Order manager statistics
        order_stats = self.order_manager.get_statistics()
        logger.info("Order Manager Statistics:")
        for stat, value in order_stats.items():
            if isinstance(value, float):
                logger.info(f"  {stat}: {value:.4f}")
            else:
                logger.info(f"  {stat}: {value}")
        
        # Execution engine statistics
        exec_stats = self.execution_engine.get_execution_statistics()
        logger.info("\nExecution Engine Statistics:")
        for stat, value in exec_stats.items():
            if isinstance(value, float):
                logger.info(f"  {stat}: {value:.4f}")
            else:
                logger.info(f"  {stat}: {value}")
        
        # Position manager portfolio metrics
        portfolio_metrics = self.position_manager.get_portfolio_metrics()
        logger.info("\nPortfolio Metrics:")
        for metric, value in portfolio_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")
    
    def run_comprehensive_demo(self):
        """Run comprehensive OMS demonstration."""
        logger.info("🎯 Starting Comprehensive Order Management System Demo")
        logger.info("=" * 60)
        
        try:
            # Setup
            self.setup_oms()
            
            if not self.start_oms():
                return False
            
            # Simulate market data
            self.simulate_market_data()
            
            # Run demonstrations
            self.demo_immediate_execution()
            time.sleep(2)
            
            self.demo_limit_orders()
            time.sleep(2)
            
            self.demo_risk_controls()
            time.sleep(2)
            
            self.demo_position_tracking()
            time.sleep(2)
            
            self.demo_execution_algorithms()
            time.sleep(3)  # Allow time for executions
            
            self.demo_performance_monitoring()
            
            # Wait a bit to see final results
            time.sleep(3)
            
            # Final portfolio status
            logger.info("\n=== FINAL PORTFOLIO STATUS ===")
            self.demo_position_tracking()
            
            logger.info("\n🎉 Order Management System demonstration completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {str(e)}")
            return False
        finally:
            self.stop_oms()
    
    def _on_fill(self, order_id: str, fill: Fill):
        """Handle order fill callback."""
        logger.info(f"🔄 Order filled: {order_id} - {fill.quantity} @ ${fill.price:.2f}")
    
    def _on_execution_complete(self, execution_id: str, success: bool, message: str):
        """Handle execution completion callback."""
        status = "✅ Success" if success else "❌ Failed"
        logger.info(f"🎯 Execution complete: {execution_id} - {status}: {message}")


def signal_handler(signum, frame):
    """Handle shutdown signal."""
    logger.info("\n🛑 Shutdown signal received")
    sys.exit(0)


def main():
    """Main demo execution."""
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the demo
    demo = OMSDemo()
    success = demo.run_comprehensive_demo()
    
    if success:
        logger.info("\n✅ Demo completed successfully")
        return 0
    else:
        logger.error("\n❌ Demo failed")
        return 1


if __name__ == '__main__':
    exit(main())
