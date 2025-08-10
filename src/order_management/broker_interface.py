"""
Broker Interface for Order Management System

Provides abstraction layer for broker connectivity and order execution.
Supports multiple broker implementations with a common interface.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass

from ..strategies.base_strategy import TradeDirection


class OrderType(Enum):
    """Types of orders that can be placed."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(Enum):
    """Status of order execution."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    """Time in force for orders."""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    symbol: str
    direction: TradeDirection
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    strategy_id: Optional[str] = None
    created_time: Optional[datetime] = None
    
    # Order state
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0
    
    def __post_init__(self):
        """Initialize order after creation."""
        if self.created_time is None:
            self.created_time = datetime.now()
        if self.remaining_quantity == 0.0:
            self.remaining_quantity = self.quantity
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
    
    def update_fill(self, fill_quantity: float, fill_price: float, commission: float = 0.0):
        """Update order with fill information."""
        self.filled_quantity += fill_quantity
        self.remaining_quantity = max(0, self.quantity - self.filled_quantity)
        
        # Update average fill price
        if self.filled_quantity > 0:
            total_value = (self.average_fill_price * (self.filled_quantity - fill_quantity)) + (fill_price * fill_quantity)
            self.average_fill_price = total_value / self.filled_quantity
        
        self.commission += commission
        
        # Update status
        if self.remaining_quantity == 0:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'direction': self.direction.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'strategy_id': self.strategy_id,
            'created_time': self.created_time,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'average_fill_price': self.average_fill_price,
            'commission': self.commission
        }


@dataclass
class Fill:
    """Represents a trade fill."""
    fill_id: str
    order_id: str
    symbol: str
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    execution_id: Optional[str] = None


class BrokerInterface(ABC):
    """
    Abstract base class for broker interfaces.
    
    Defines the common interface that all broker implementations must support.
    This allows the OMS to work with different brokers (IBKR, Alpaca, etc.).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize broker interface.
        
        Args:
            config: Broker-specific configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.connected = False
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        
        # Callback functions
        self.order_update_callback: Optional[Callable[[Order], None]] = None
        self.fill_callback: Optional[Callable[[Fill], None]] = None
        self.error_callback: Optional[Callable[[str, Exception], None]] = None
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the broker.
        
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the broker."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to broker."""
        pass
    
    @abstractmethod
    def place_order(self, order: Order) -> bool:
        """
        Place an order with the broker.
        
        Args:
            order: Order to place
            
        Returns:
            True if order placement successful
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order to cancel
            
        Returns:
            True if cancellation successful
        """
        pass
    
    @abstractmethod
    def modify_order(self, order_id: str, **kwargs) -> bool:
        """
        Modify an existing order.
        
        Args:
            order_id: Order to modify
            **kwargs: Order parameters to update
            
        Returns:
            True if modification successful
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get current status of an order.
        
        Args:
            order_id: Order to check
            
        Returns:
            Order object with current status or None if not found
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current positions from broker.
        
        Returns:
            Dictionary of positions by symbol
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dictionary with account details
        """
        pass
    
    @abstractmethod
    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current market data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            
        Returns:
            Market data dictionary or None if not available
        """
        pass
    
    def set_callbacks(
        self,
        order_update_callback: Optional[Callable[[Order], None]] = None,
        fill_callback: Optional[Callable[[Fill], None]] = None,
        error_callback: Optional[Callable[[str, Exception], None]] = None
    ):
        """Set callback functions for broker events."""
        self.order_update_callback = order_update_callback
        self.fill_callback = fill_callback
        self.error_callback = error_callback
    
    def _handle_order_update(self, order: Order):
        """Handle order status updates."""
        self.orders[order.order_id] = order
        if self.order_update_callback:
            self.order_update_callback(order)
    
    def _handle_fill(self, fill: Fill):
        """Handle trade fills."""
        self.fills.append(fill)
        if self.fill_callback:
            self.fill_callback(fill)
    
    def _handle_error(self, context: str, error: Exception):
        """Handle broker errors."""
        self.logger.error(f"Broker error in {context}: {str(error)}")
        if self.error_callback:
            self.error_callback(context, error)


class MockBrokerInterface(BrokerInterface):
    """
    Mock broker interface for testing and development.
    
    Simulates broker behavior without actual market connectivity.
    Useful for testing the OMS functionality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize mock broker."""
        super().__init__(config)
        self.next_order_id = 1
        self.simulated_prices: Dict[str, float] = {}
        
    def connect(self) -> bool:
        """Simulate broker connection."""
        self.logger.info("Mock broker connecting...")
        self.connected = True
        return True
    
    def disconnect(self) -> None:
        """Simulate broker disconnection."""
        self.logger.info("Mock broker disconnecting...")
        self.connected = False
    
    def is_connected(self) -> bool:
        """Check connection status."""
        return self.connected
    
    def place_order(self, order: Order) -> bool:
        """Simulate order placement."""
        if not self.connected:
            return False
        
        # Assign order ID if not set
        if not order.order_id:
            order.order_id = f"MOCK_{self.next_order_id}"
            self.next_order_id += 1
        
        # Update order status
        order.status = OrderStatus.SUBMITTED
        self.orders[order.order_id] = order
        
        self.logger.info(f"Mock order placed: {order.order_id} {order.symbol} {order.direction.value} {order.quantity}")
        
        # Simulate immediate fill for market orders
        if order.order_type == OrderType.MARKET:
            self._simulate_fill(order)
        
        self._handle_order_update(order)
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """Simulate order cancellation."""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.is_active:
                order.status = OrderStatus.CANCELLED
                self._handle_order_update(order)
                self.logger.info(f"Mock order cancelled: {order_id}")
                return True
        return False
    
    def modify_order(self, order_id: str, **kwargs) -> bool:
        """Simulate order modification."""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.is_active:
                # Update order parameters
                for key, value in kwargs.items():
                    if hasattr(order, key):
                        setattr(order, key, value)
                
                self._handle_order_update(order)
                self.logger.info(f"Mock order modified: {order_id}")
                return True
        return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status."""
        return self.orders.get(order_id)
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get mock positions."""
        # Return empty positions for simplicity
        return {}
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get mock account info."""
        return {
            'account_id': 'MOCK_ACCOUNT',
            'buying_power': 100000.0,
            'total_cash': 100000.0,
            'portfolio_value': 100000.0,
            'day_trades': 0,
            'currency': 'USD'
        }
    
    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get mock market data."""
        # Return mock price or generate one
        price = self.simulated_prices.get(symbol, 100.0)
        
        return {
            'symbol': symbol,
            'price': price,
            'bid': price - 0.01,
            'ask': price + 0.01,
            'volume': 1000000,
            'timestamp': datetime.now()
        }
    
    def _simulate_fill(self, order: Order):
        """Simulate order fill."""
        if order.price:
            fill_price = order.price
        else:
            market_data = self.get_market_data(order.symbol)
            fill_price = market_data['price'] if market_data else 100.0
        
        fill = Fill(
            fill_id=f"FILL_{len(self.fills) + 1}",
            order_id=order.order_id,
            symbol=order.symbol,
            quantity=order.quantity,
            price=fill_price,
            timestamp=datetime.now(),
            commission=1.0  # Mock commission
        )
        
        order.update_fill(fill.quantity, fill.price, fill.commission)
        self._handle_fill(fill)
        self._handle_order_update(order)
