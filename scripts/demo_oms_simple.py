#!/usr/bin/env python3
"""
Order Management System (OMS) Demo - Simplified Version

A focused demonstration of the Order Management System core capabilities.
"""

import logging
import time
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Add the src directory to the path
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from order_management import (
    OrderManager, MockBrokerInterface, Order, Fill, 
    OrderStatus, OrderType, PositionManager, RiskController, 
    ExecutionEngine, ExecutionAlgorithm
)
from order_management.risk_controls import RiskLimits
from strategies.base_strategy import TradeDirection


def demo_basic_order_placement():
    """Demonstrate basic order placement and management."""
    logger.info("=== BASIC ORDER PLACEMENT DEMO ===")
    
    # Initialize components
    broker = MockBrokerInterface({})
    order_manager = OrderManager(broker=broker)
    
    # Start order manager
    order_manager.start()
    
    try:
        # Place a market order
        order_id = order_manager.place_order(
            symbol='AAPL',
            direction=TradeDirection.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            strategy_id='demo_strategy'
        )
        
        if order_id:
            logger.info(f"✓ Market order placed: {order_id}")
            
            # Wait a moment for processing
            time.sleep(1)
            
            # Check order status
            orders = order_manager.get_orders()
            if order_id in orders:
                order = orders[order_id]
                logger.info(f"Order status: {order.status.value}")
        else:
            logger.error("❌ Failed to place market order")
        
        # Place a limit order
        limit_order_id = order_manager.place_order(
            symbol='MSFT',
            direction=TradeDirection.BUY,
            quantity=50,
            order_type=OrderType.LIMIT,
            price=280.00,
            strategy_id='demo_strategy'
        )
        
        if limit_order_id:
            logger.info(f"✓ Limit order placed: {limit_order_id}")
        
        # Show order statistics
        stats = order_manager.get_statistics()
        logger.info("Order Manager Statistics:")
        for stat, value in stats.items():
            logger.info(f"  {stat}: {value}")
            
    finally:
        order_manager.stop()
    
    logger.info("Basic order placement demo completed\n")


def demo_position_management():
    """Demonstrate position management capabilities."""
    logger.info("=== POSITION MANAGEMENT DEMO ===")
    
    # Initialize position manager
    initial_capital = 100000.0
    position_manager = PositionManager(initial_capital=initial_capital)
    
    logger.info(f"Initial capital: ${initial_capital:,.2f}")
    
    # Simulate some market data
    market_data = {
        'AAPL': {'bid': 150.25, 'ask': 150.30, 'last': 150.28},
        'MSFT': {'bid': 280.50, 'ask': 280.55, 'last': 280.52}
    }
    
    for symbol, data in market_data.items():
        position_manager.update_market_data(symbol, data)
    
    # Simulate opening a position
    fill_aapl = Fill(
        order_id='order_1',
        symbol='AAPL',
        quantity=100,
        price=150.28,
        timestamp=datetime.now(),
        direction=TradeDirection.BUY
    )
    
    position_manager.update_position(fill_aapl, 'demo_strategy')
    logger.info("✓ Opened AAPL position: 100 shares @ $150.28")
    
    # Get positions
    positions = position_manager.get_positions()
    logger.info(f"Current positions: {len(positions)}")
    
    for position in positions:
        logger.info(f"  {position.symbol}: {position.quantity} shares, "
                   f"Avg Price: ${position.average_price:.2f}")
    
    # Get portfolio metrics
    portfolio_value = position_manager.calculate_portfolio_value()
    logger.info(f"Portfolio value: ${portfolio_value:,.2f}")
    
    logger.info("Position management demo completed\n")


def demo_risk_controls():
    """Demonstrate risk control capabilities."""
    logger.info("=== RISK CONTROLS DEMO ===")
    
    # Create risk limits
    risk_limits = RiskLimits(
        max_position_size=10000.0,
        max_daily_loss=0.02,  # 2%
        max_portfolio_exposure=0.95  # 95%
    )
    
    risk_controller = RiskController(limits=risk_limits)
    logger.info("✓ Risk controller initialized")
    
    # Test order validation
    test_order = Order(
        order_id='test_order',
        symbol='AAPL',
        direction=TradeDirection.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
        status=OrderStatus.PENDING,
        timestamp=datetime.now()
    )
    
    # Validate the order
    validation_result = risk_controller.validate_order(test_order, portfolio_value=50000.0)
    
    logger.info(f"Order validation result: {validation_result.is_valid}")
    if not validation_result.is_valid:
        logger.info(f"Rejection reason: {validation_result.reason}")
    
    logger.info("Risk controls demo completed\n")


def demo_execution_engine():
    """Demonstrate execution engine capabilities."""
    logger.info("=== EXECUTION ENGINE DEMO ===")
    
    # Initialize components
    broker = MockBrokerInterface({})
    order_manager = OrderManager(broker=broker)
    position_manager = PositionManager(initial_capital=100000.0)
    
    execution_engine = ExecutionEngine(
        broker=broker,
        order_manager=order_manager,
        position_manager=position_manager
    )
    
    # Start components
    order_manager.start()
    execution_engine.start()
    
    try:
        # Update market data
        market_data = {'bid': 150.25, 'ask': 150.30, 'last': 150.28}
        execution_engine.update_market_data('AAPL', market_data)
        
        # Execute immediate order
        execution_id = execution_engine.execute_order(
            symbol='AAPL',
            direction=TradeDirection.BUY,
            quantity=100,
            algorithm=ExecutionAlgorithm.IMMEDIATE,
            strategy_id='demo_strategy'
        )
        
        if execution_id:
            logger.info(f"✓ Execution queued: {execution_id}")
            
            # Wait for execution
            time.sleep(2)
            
            # Check execution status
            status = execution_engine.get_execution_status(execution_id)
            if status:
                logger.info(f"Execution status: {status['status']}")
        
        # Get execution statistics
        stats = execution_engine.get_execution_statistics()
        logger.info("Execution Statistics:")
        for stat, value in stats.items():
            if isinstance(value, float):
                logger.info(f"  {stat}: {value:.4f}")
            else:
                logger.info(f"  {stat}: {value}")
                
    finally:
        execution_engine.stop()
        order_manager.stop()
    
    logger.info("Execution engine demo completed\n")


def main():
    """Run all OMS demonstrations."""
    logger.info("🎯 Starting Order Management System Demo")
    logger.info("=" * 50)
    
    try:
        demo_basic_order_placement()
        demo_position_management()
        demo_risk_controls()
        demo_execution_engine()
        
        logger.info("🎉 All OMS demonstrations completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
