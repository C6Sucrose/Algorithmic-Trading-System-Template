"""
Order Manager for Algorithmic Trading System

Central order management system that coordinates order placement,
tracking, and risk management across all trading strategies.
"""

import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict
import uuid

from .broker_interface import BrokerInterface, Order, Fill, OrderStatus, OrderType, TimeInForce
from .risk_controls import RiskController
from ..strategies.base_strategy import TradeDirection


class OrderManager:
    """
    Central order management system.
    
    Coordinates all order placement, tracking, and management across
    multiple strategies with comprehensive risk controls and monitoring.
    """
    
    def __init__(
        self,
        broker: BrokerInterface,
        risk_controller: Optional[RiskController] = None
    ):
        """
        Initialize Order Manager.
        
        Args:
            broker: Broker interface for order execution
            risk_controller: Risk controller for order validation
        """
        self.broker = broker
        self.risk_controller = risk_controller
        self.logger = logging.getLogger(__name__)
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self.order_history: List[Order] = []
        
        # Strategy tracking
        self.strategy_orders: Dict[str, List[str]] = defaultdict(list)
        self.strategy_fills: Dict[str, List[Fill]] = defaultdict(list)
        
        # Performance tracking
        self.total_orders_placed = 0
        self.total_orders_filled = 0
        self.total_orders_cancelled = 0
        self.total_commission_paid = 0.0
        
        # Threading for async operations
        self.lock = threading.Lock()
        self.active = False
        
        # Set up broker callbacks
        self.broker.set_callbacks(
            order_update_callback=self._handle_order_update,
            fill_callback=self._handle_fill,
            error_callback=self._handle_broker_error
        )
        
        # Event callbacks
        self.order_callbacks: List[Callable[[Order], None]] = []
        self.fill_callbacks: List[Callable[[Fill], None]] = []
        self.error_callbacks: List[Callable[[str, Exception], None]] = []
    
    def start(self) -> bool:
        """
        Start the order manager.
        
        Returns:
            True if startup successful
        """
        try:
            if not self.broker.connect():
                self.logger.error("Failed to connect to broker")
                return False
            
            self.active = True
            self.logger.info("Order Manager started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Order Manager: {str(e)}")
            return False
    
    def stop(self) -> None:
        """Stop the order manager and disconnect from broker."""
        self.active = False
        
        # Cancel all active orders
        self.cancel_all_orders()
        
        # Disconnect from broker
        self.broker.disconnect()
        
        self.logger.info("Order Manager stopped")
    
    def place_order(
        self,
        symbol: str,
        direction: TradeDirection,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        strategy_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Place a new order.
        
        Args:
            symbol: Trading symbol
            direction: Buy or sell direction
            quantity: Number of shares/units
            order_type: Type of order (market, limit, etc.)
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Time in force
            strategy_id: Strategy placing the order
            
        Returns:
            Order ID if successful, None if failed
        """
        if not self.active:
            self.logger.warning("Order Manager not active")
            return None
        
        try:
            # Generate unique order ID
            order_id = self._generate_order_id()
            
            # Create order object
            order = Order(
                order_id=order_id,
                symbol=symbol,
                direction=direction,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                strategy_id=strategy_id,
                created_time=datetime.now()
            )
            
            # Validate order with risk controller
            if self.risk_controller:
                validation_result = self.risk_controller.validate_order(order)
                if not validation_result.is_valid:
                    self.logger.warning(f"Order rejected by risk controller: {validation_result.reason}")
                    return None
            
            # Place order with broker
            with self.lock:
                if self.broker.place_order(order):
                    self.orders[order_id] = order
                    self.strategy_orders[strategy_id or 'unknown'].append(order_id)
                    self.total_orders_placed += 1
                    
                    self.logger.info(f"Order placed: {order_id} {symbol} {direction.value} {quantity}")
                    return order_id
                else:
                    self.logger.error(f"Failed to place order with broker: {order_id}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order to cancel
            
        Returns:
            True if cancellation successful
        """
        try:
            with self.lock:
                if order_id not in self.orders:
                    self.logger.warning(f"Order not found for cancellation: {order_id}")
                    return False
                
                order = self.orders[order_id]
                if not order.is_active:
                    self.logger.warning(f"Order not active for cancellation: {order_id}")
                    return False
                
                if self.broker.cancel_order(order_id):
                    self.total_orders_cancelled += 1
                    self.logger.info(f"Order cancelled: {order_id}")
                    return True
                else:
                    self.logger.error(f"Failed to cancel order with broker: {order_id}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return False
    
    def modify_order(
        self,
        order_id: str,
        **kwargs
    ) -> bool:
        """
        Modify an existing order.
        
        Args:
            order_id: Order to modify
            **kwargs: Parameters to update
            
        Returns:
            True if modification successful
        """
        try:
            with self.lock:
                if order_id not in self.orders:
                    self.logger.warning(f"Order not found for modification: {order_id}")
                    return False
                
                order = self.orders[order_id]
                if not order.is_active:
                    self.logger.warning(f"Order not active for modification: {order_id}")
                    return False
                
                # Validate modifications with risk controller
                if self.risk_controller:
                    # Create a copy with modifications for validation
                    temp_order = Order(**order.to_dict())
                    for key, value in kwargs.items():
                        if hasattr(temp_order, key):
                            setattr(temp_order, key, value)
                    
                    validation_result = self.risk_controller.validate_order(temp_order)
                    if not validation_result.is_valid:
                        self.logger.warning(f"Order modification rejected: {validation_result.reason}")
                        return False
                
                if self.broker.modify_order(order_id, **kwargs):
                    self.logger.info(f"Order modified: {order_id}")
                    return True
                else:
                    self.logger.error(f"Failed to modify order with broker: {order_id}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error modifying order {order_id}: {str(e)}")
            return False
    
    def cancel_all_orders(self, strategy_id: Optional[str] = None) -> int:
        """
        Cancel all orders or all orders for a specific strategy.
        
        Args:
            strategy_id: Optional strategy ID to filter orders
            
        Returns:
            Number of orders cancelled
        """
        cancelled_count = 0
        
        try:
            with self.lock:
                orders_to_cancel = []
                
                if strategy_id:
                    # Cancel orders for specific strategy
                    for order_id in self.strategy_orders.get(strategy_id, []):
                        if order_id in self.orders and self.orders[order_id].is_active:
                            orders_to_cancel.append(order_id)
                else:
                    # Cancel all active orders
                    for order_id, order in self.orders.items():
                        if order.is_active:
                            orders_to_cancel.append(order_id)
                
                for order_id in orders_to_cancel:
                    if self.cancel_order(order_id):
                        cancelled_count += 1
                
                self.logger.info(f"Cancelled {cancelled_count} orders")
                
        except Exception as e:
            self.logger.error(f"Error cancelling orders: {str(e)}")
        
        return cancelled_count
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get current status of an order.
        
        Args:
            order_id: Order to check
            
        Returns:
            Order object or None if not found
        """
        return self.orders.get(order_id)
    
    def get_orders(
        self,
        strategy_id: Optional[str] = None,
        status_filter: Optional[OrderStatus] = None
    ) -> List[Order]:
        """
        Get orders with optional filtering.
        
        Args:
            strategy_id: Filter by strategy ID
            status_filter: Filter by order status
            
        Returns:
            List of matching orders
        """
        with self.lock:
            orders = list(self.orders.values())
            
            if strategy_id:
                orders = [o for o in orders if o.strategy_id == strategy_id]
            
            if status_filter:
                orders = [o for o in orders if o.status == status_filter]
            
            return orders
    
    def get_fills(self, strategy_id: Optional[str] = None) -> List[Fill]:
        """
        Get fills with optional strategy filtering.
        
        Args:
            strategy_id: Filter by strategy ID
            
        Returns:
            List of fills
        """
        if strategy_id:
            return self.strategy_fills.get(strategy_id, [])
        return self.fills.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get order manager performance summary."""
        with self.lock:
            active_orders = sum(1 for o in self.orders.values() if o.is_active)
            
            return {
                'total_orders_placed': self.total_orders_placed,
                'total_orders_filled': self.total_orders_filled,
                'total_orders_cancelled': self.total_orders_cancelled,
                'active_orders': active_orders,
                'total_fills': len(self.fills),
                'total_commission_paid': self.total_commission_paid,
                'fill_rate': self.total_orders_filled / max(1, self.total_orders_placed) * 100,
                'strategies_active': len([s for s in self.strategy_orders.keys() if s != 'unknown'])
            }
    
    def add_order_callback(self, callback: Callable[[Order], None]):
        """Add callback for order updates."""
        self.order_callbacks.append(callback)
    
    def add_fill_callback(self, callback: Callable[[Fill], None]):
        """Add callback for fill updates."""
        self.fill_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[str, Exception], None]):
        """Add callback for error notifications."""
        self.error_callbacks.append(callback)
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        return f"ORD_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def _handle_order_update(self, order: Order):
        """Handle order status updates from broker."""
        with self.lock:
            self.orders[order.order_id] = order
            
            # Update statistics
            if order.status == OrderStatus.FILLED:
                self.total_orders_filled += 1
            elif order.status == OrderStatus.CANCELLED:
                self.total_orders_cancelled += 1
            
            # Move to history if order is complete
            if not order.is_active and order not in self.order_history:
                self.order_history.append(order)
        
        # Notify callbacks
        for callback in self.order_callbacks:
            try:
                callback(order)
            except Exception as e:
                self.logger.error(f"Error in order callback: {str(e)}")
    
    def _handle_fill(self, fill: Fill):
        """Handle fill notifications from broker."""
        with self.lock:
            self.fills.append(fill)
            self.total_commission_paid += fill.commission
            
            # Update strategy tracking
            if fill.order_id in self.orders:
                order = self.orders[fill.order_id]
                if order.strategy_id:
                    self.strategy_fills[order.strategy_id].append(fill)
        
        # Notify callbacks
        for callback in self.fill_callbacks:
            try:
                callback(fill)
            except Exception as e:
                self.logger.error(f"Error in fill callback: {str(e)}")
        
        self.logger.info(f"Fill received: {fill.symbol} {fill.quantity} @ {fill.price}")
    
    def _handle_broker_error(self, context: str, error: Exception):
        """Handle broker error notifications."""
        self.logger.error(f"Broker error in {context}: {str(error)}")
        
        # Notify callbacks
        for callback in self.error_callbacks:
            try:
                callback(context, error)
            except Exception as e:
                self.logger.error(f"Error in error callback: {str(e)}")
