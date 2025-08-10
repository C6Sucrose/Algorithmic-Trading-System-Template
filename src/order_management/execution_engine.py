"""
Execution Engine for Order Management System

Coordinates order execution, monitoring, and management with
intelligent routing and execution algorithms.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from queue import Queue, Empty

from .broker_interface import BrokerInterface, Order, Fill, OrderStatus, OrderType
from .order_manager import OrderManager
from .position_manager import PositionManager
from .risk_controls import RiskController
from ..strategies.base_strategy import TradeDirection


class ExecutionAlgorithm(Enum):
    """Execution algorithm types."""
    IMMEDIATE = "immediate"
    TWAP = "twap"  # Time Weighted Average Price
    VWAP = "vwap"  # Volume Weighted Average Price
    ICEBERG = "iceberg"  # Large order slicing
    SMART = "smart"  # Intelligent routing


class ExecutionEngine:
    """
    Advanced execution engine for algorithmic trading.
    
    Provides intelligent order execution with multiple algorithms,
    real-time monitoring, and automated management capabilities.
    """
    
    def __init__(
        self,
        broker: BrokerInterface,
        order_manager: OrderManager,
        position_manager: PositionManager,
        risk_controller: Optional[RiskController] = None
    ):
        """
        Initialize Execution Engine.
        
        Args:
            broker: Broker interface
            order_manager: Order manager
            position_manager: Position manager
            risk_controller: Risk controller
        """
        self.broker = broker
        self.order_manager = order_manager
        self.position_manager = position_manager
        self.risk_controller = risk_controller
        self.logger = logging.getLogger(__name__)
        
        # Execution state
        self.running = False
        self.execution_thread: Optional[threading.Thread] = None
        
        # Execution queue
        self.execution_queue: Queue = Queue()
        self.pending_executions: Dict[str, Dict[str, Any]] = {}
        
        # Market data for execution decisions
        self.market_data: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'avg_execution_time': 0.0,
            'slippage_total': 0.0
        }
        
        # Callbacks
        self.execution_callbacks: List[Callable[[str, bool, str], None]] = []
    
    def start(self) -> bool:
        """
        Start the execution engine.
        
        Returns:
            True if startup successful
        """
        try:
            if self.running:
                self.logger.warning("Execution engine already running")
                return True
            
            self.running = True
            self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
            self.execution_thread.start()
            
            self.logger.info("Execution Engine started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Execution Engine: {str(e)}")
            return False
    
    def stop(self):
        """Stop the execution engine."""
        self.running = False
        
        if self.execution_thread:
            self.execution_thread.join(timeout=5.0)
        
        self.logger.info("Execution Engine stopped")
    
    def execute_order(
        self,
        symbol: str,
        direction: TradeDirection,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        algorithm: ExecutionAlgorithm = ExecutionAlgorithm.IMMEDIATE,
        strategy_id: Optional[str] = None,
        **execution_params
    ) -> Optional[str]:
        """
        Execute an order with specified algorithm.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction
            quantity: Quantity to trade
            order_type: Order type
            price: Limit price (if applicable)
            algorithm: Execution algorithm
            strategy_id: Strategy ID
            **execution_params: Algorithm-specific parameters
            
        Returns:
            Execution ID if successful
        """
        try:
            execution_id = f"EXEC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}"
            
            execution_request = {
                'execution_id': execution_id,
                'symbol': symbol,
                'direction': direction,
                'quantity': quantity,
                'order_type': order_type,
                'price': price,
                'algorithm': algorithm,
                'strategy_id': strategy_id,
                'params': execution_params,
                'created_time': datetime.now(),
                'status': 'pending'
            }
            
            # Add to execution queue
            self.execution_queue.put(execution_request)
            self.pending_executions[execution_id] = execution_request
            
            self.logger.info(f"Execution queued: {execution_id} {symbol} {direction.value} {quantity}")
            return execution_id
            
        except Exception as e:
            self.logger.error(f"Error queuing execution: {str(e)}")
            return None
    
    def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel a pending execution.
        
        Args:
            execution_id: Execution to cancel
            
        Returns:
            True if cancellation successful
        """
        try:
            if execution_id in self.pending_executions:
                execution = self.pending_executions[execution_id]
                execution['status'] = 'cancelled'
                
                # Cancel any associated orders
                if 'order_ids' in execution:
                    for order_id in execution['order_ids']:
                        self.order_manager.cancel_order(order_id)
                
                self.logger.info(f"Execution cancelled: {execution_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error cancelling execution {execution_id}: {str(e)}")
            return False
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of an execution.
        
        Args:
            execution_id: Execution to check
            
        Returns:
            Execution status dictionary
        """
        return self.pending_executions.get(execution_id)
    
    def update_market_data(self, symbol: str, market_data: Dict[str, Any]):
        """
        Update market data for execution decisions.
        
        Args:
            symbol: Symbol to update
            market_data: Market data
        """
        self.market_data[symbol] = market_data
        
        # Update position manager as well
        self.position_manager.update_market_data(symbol, market_data)
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution performance statistics."""
        return self.execution_stats.copy()
    
    def add_execution_callback(self, callback: Callable[[str, bool, str], None]):
        """
        Add callback for execution completion.
        
        Args:
            callback: Function(execution_id, success, message)
        """
        self.execution_callbacks.append(callback)
    
    def _execution_loop(self):
        """Main execution loop running in separate thread."""
        self.logger.info("Execution loop started")
        
        while self.running:
            try:
                # Get execution request from queue
                execution_request = self.execution_queue.get(timeout=1.0)
                
                # Process the execution
                self._process_execution(execution_request)
                
            except Empty:
                # No execution requests, continue
                continue
            except Exception as e:
                self.logger.error(f"Error in execution loop: {str(e)}")
                time.sleep(1.0)
        
        self.logger.info("Execution loop stopped")
    
    def _process_execution(self, execution_request: Dict[str, Any]):
        """Process a single execution request."""
        execution_id = execution_request['execution_id']
        
        try:
            self.execution_stats['total_executions'] += 1
            start_time = time.time()
            
            # Update status
            execution_request['status'] = 'executing'
            
            # Choose execution algorithm
            algorithm = execution_request['algorithm']
            
            if algorithm == ExecutionAlgorithm.IMMEDIATE:
                success = self._execute_immediate(execution_request)
            elif algorithm == ExecutionAlgorithm.TWAP:
                success = self._execute_twap(execution_request)
            elif algorithm == ExecutionAlgorithm.ICEBERG:
                success = self._execute_iceberg(execution_request)
            else:
                success = self._execute_smart(execution_request)
            
            # Update statistics
            execution_time = time.time() - start_time
            self.execution_stats['avg_execution_time'] = (
                (self.execution_stats['avg_execution_time'] * (self.execution_stats['total_executions'] - 1) + execution_time) /
                self.execution_stats['total_executions']
            )
            
            if success:
                self.execution_stats['successful_executions'] += 1
                execution_request['status'] = 'completed'
                message = "Execution completed successfully"
            else:
                self.execution_stats['failed_executions'] += 1
                execution_request['status'] = 'failed'
                message = "Execution failed"
            
            # Notify callbacks
            self._notify_execution_callbacks(execution_id, success, message)
            
        except Exception as e:
            self.logger.error(f"Error processing execution {execution_id}: {str(e)}")
            execution_request['status'] = 'failed'
            self.execution_stats['failed_executions'] += 1
            self._notify_execution_callbacks(execution_id, False, str(e))
    
    def _execute_immediate(self, execution_request: Dict[str, Any]) -> bool:
        """Execute order immediately."""
        try:
            order_id = self.order_manager.place_order(
                symbol=execution_request['symbol'],
                direction=execution_request['direction'],
                quantity=execution_request['quantity'],
                order_type=execution_request['order_type'],
                price=execution_request['price'],
                strategy_id=execution_request['strategy_id']
            )
            
            if order_id:
                execution_request['order_ids'] = [order_id]
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in immediate execution: {str(e)}")
            return False
    
    def _execute_twap(self, execution_request: Dict[str, Any]) -> bool:
        """Execute order using Time Weighted Average Price algorithm."""
        try:
            # TWAP parameters
            params = execution_request.get('params', {})
            duration_minutes = params.get('duration_minutes', 60)
            num_slices = params.get('num_slices', 10)
            
            total_quantity = execution_request['quantity']
            slice_quantity = total_quantity / num_slices
            slice_interval = (duration_minutes * 60) / num_slices
            
            order_ids = []
            
            # Place first slice immediately
            order_id = self.order_manager.place_order(
                symbol=execution_request['symbol'],
                direction=execution_request['direction'],
                quantity=slice_quantity,
                order_type=OrderType.MARKET,
                strategy_id=execution_request['strategy_id']
            )
            
            if order_id:
                order_ids.append(order_id)
                execution_request['order_ids'] = order_ids
                execution_request['twap_progress'] = {
                    'slices_remaining': num_slices - 1,
                    'next_slice_time': datetime.now() + timedelta(seconds=slice_interval)
                }
                
                # Note: In a real implementation, you'd schedule the remaining slices
                # For this demo, we'll just place the first slice
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in TWAP execution: {str(e)}")
            return False
    
    def _execute_iceberg(self, execution_request: Dict[str, Any]) -> bool:
        """Execute large order using iceberg algorithm."""
        try:
            # Iceberg parameters
            params = execution_request.get('params', {})
            visible_quantity = params.get('visible_quantity', 100)
            
            total_quantity = execution_request['quantity']
            
            # Place first visible slice
            first_slice = min(visible_quantity, total_quantity)
            
            order_id = self.order_manager.place_order(
                symbol=execution_request['symbol'],
                direction=execution_request['direction'],
                quantity=first_slice,
                order_type=execution_request['order_type'],
                price=execution_request['price'],
                strategy_id=execution_request['strategy_id']
            )
            
            if order_id:
                execution_request['order_ids'] = [order_id]
                execution_request['iceberg_progress'] = {
                    'quantity_remaining': total_quantity - first_slice,
                    'visible_quantity': visible_quantity
                }
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in iceberg execution: {str(e)}")
            return False
    
    def _execute_smart(self, execution_request: Dict[str, Any]) -> bool:
        """Execute order using smart routing algorithm."""
        try:
            symbol = execution_request['symbol']
            
            # Get current market data for smart decisions
            if symbol in self.market_data:
                market_data = self.market_data[symbol]
                spread = market_data.get('ask', 0) - market_data.get('bid', 0)
                
                # Choose order type based on market conditions
                if spread > 0.05:  # Wide spread - use limit order
                    if execution_request['direction'] == TradeDirection.BUY:
                        price = market_data.get('bid', 0) + spread * 0.3
                    else:
                        price = market_data.get('ask', 0) - spread * 0.3
                    
                    order_type = OrderType.LIMIT
                else:
                    # Tight spread - use market order
                    price = None
                    order_type = OrderType.MARKET
            else:
                # No market data - default to market order
                price = execution_request['price']
                order_type = execution_request['order_type']
            
            order_id = self.order_manager.place_order(
                symbol=execution_request['symbol'],
                direction=execution_request['direction'],
                quantity=execution_request['quantity'],
                order_type=order_type,
                price=price,
                strategy_id=execution_request['strategy_id']
            )
            
            if order_id:
                execution_request['order_ids'] = [order_id]
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in smart execution: {str(e)}")
            return False
    
    def _notify_execution_callbacks(self, execution_id: str, success: bool, message: str):
        """Notify execution callbacks."""
        for callback in self.execution_callbacks:
            try:
                callback(execution_id, success, message)
            except Exception as e:
                self.logger.error(f"Error in execution callback: {str(e)}")
