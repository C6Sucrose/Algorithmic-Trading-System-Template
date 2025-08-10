"""
Position Manager for Order Management System

Tracks and manages all trading positions with real-time updates,
P&L calculation, and portfolio monitoring capabilities.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict
import threading

from .broker_interface import Fill, Order
from ..strategies.base_strategy import Position, PositionType, TradeDirection


class PositionManager:
    """
    Comprehensive position tracking and management system.
    
    Maintains real-time position state, calculates P&L, and provides
    portfolio analytics across all strategies and symbols.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize Position Manager.
        
        Args:
            initial_capital: Starting capital amount
        """
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # Portfolio state
        self.cash = initial_capital
        self.total_portfolio_value = initial_capital
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        # Strategy tracking
        self.strategy_positions: Dict[str, List[str]] = defaultdict(list)
        self.strategy_pnl: Dict[str, float] = defaultdict(float)
        
        # Performance tracking
        self.peak_portfolio_value = initial_capital
        self.total_trades = 0
        self.total_commission_paid = 0.0
        
        # Threading for thread-safe operations
        self.lock = threading.Lock()
        
        # Market data cache for P&L calculations
        self.market_data_cache: Dict[str, Dict[str, Any]] = {}
    
    def process_fill(self, fill: Fill, order: Order):
        """
        Process a trade fill and update positions.
        
        Args:
            fill: Fill information
            order: Original order that was filled
        """
        with self.lock:
            try:
                symbol = fill.symbol
                
                # Update cash
                trade_value = fill.quantity * fill.price
                if order.direction in [TradeDirection.BUY, TradeDirection.BUY_TO_COVER]:
                    self.cash -= trade_value + fill.commission
                else:  # SELL, SELL_SHORT
                    self.cash += trade_value - fill.commission
                
                self.total_commission_paid += fill.commission
                
                # Update or create position
                if symbol in self.positions:
                    self._update_existing_position(fill, order)
                else:
                    self._create_new_position(fill, order)
                
                # Update strategy tracking
                if order.strategy_id:
                    if symbol not in self.strategy_positions[order.strategy_id]:
                        self.strategy_positions[order.strategy_id].append(symbol)
                
                self.total_trades += 1
                
                self.logger.info(f"Position updated: {symbol} fill {fill.quantity} @ {fill.price}")
                
            except Exception as e:
                self.logger.error(f"Error processing fill: {str(e)}")
    
    def update_market_data(self, symbol: str, market_data: Dict[str, Any]):
        """
        Update market data for position valuation.
        
        Args:
            symbol: Symbol to update
            market_data: Current market data
        """
        with self.lock:
            self.market_data_cache[symbol] = market_data
            
            # Update position if it exists
            if symbol in self.positions:
                position = self.positions[symbol]
                current_price = market_data.get('price', position.current_price)
                position.update_price(current_price, datetime.now())
        
        # Recalculate portfolio value
        self._recalculate_portfolio_value()
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_time: datetime,
        reason: str = "manual"
    ) -> Optional[float]:
        """
        Close a position and calculate realized P&L.
        
        Args:
            symbol: Symbol to close
            exit_price: Exit price
            exit_time: Exit timestamp
            reason: Reason for closing
            
        Returns:
            Realized P&L or None if position not found
        """
        with self.lock:
            if symbol not in self.positions:
                self.logger.warning(f"No position found to close: {symbol}")
                return None
            
            position = self.positions[symbol]
            
            # Calculate realized P&L
            realized_pnl = position.close_position(exit_price, exit_time, reason)
            self.realized_pnl += realized_pnl
            
            # Update strategy P&L
            if position.strategy_id:
                self.strategy_pnl[position.strategy_id] += realized_pnl
            
            # Move to closed positions
            self.closed_positions.append(position)
            del self.positions[symbol]
            
            # Update cash with proceeds
            proceeds = position.quantity * exit_price
            self.cash += proceeds
            
            self.logger.info(f"Position closed: {symbol} P&L: ${realized_pnl:.2f}")
            
            return realized_pnl
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Position object or None if not found
        """
        return self.positions.get(symbol)
    
    def get_positions(
        self,
        strategy_id: Optional[str] = None,
        position_type: Optional[PositionType] = None
    ) -> List[Position]:
        """
        Get positions with optional filtering.
        
        Args:
            strategy_id: Filter by strategy ID
            position_type: Filter by position type
            
        Returns:
            List of matching positions
        """
        with self.lock:
            positions = list(self.positions.values())
            
            if strategy_id:
                positions = [p for p in positions if p.strategy_id == strategy_id]
            
            if position_type:
                positions = [p for p in positions if p.position_type == position_type]
            
            return positions
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        with self.lock:
            # Calculate current metrics
            self._recalculate_portfolio_value()
            
            # Unrealized P&L
            total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
            
            # Total P&L
            total_pnl = self.realized_pnl + total_unrealized
            
            # Performance metrics
            total_return = (self.total_portfolio_value - self.initial_capital) / self.initial_capital
            
            # Drawdown calculation
            current_drawdown = (self.peak_portfolio_value - self.total_portfolio_value) / self.peak_portfolio_value
            
            return {
                'initial_capital': self.initial_capital,
                'cash': self.cash,
                'total_portfolio_value': self.total_portfolio_value,
                'positions_value': sum(
                    pos.quantity * pos.current_price for pos in self.positions.values()
                ),
                'num_positions': len(self.positions),
                'unrealized_pnl': total_unrealized,
                'realized_pnl': self.realized_pnl,
                'total_pnl': total_pnl,
                'total_return_pct': total_return * 100,
                'peak_portfolio_value': self.peak_portfolio_value,
                'current_drawdown_pct': current_drawdown * 100,
                'total_trades': self.total_trades,
                'total_commission_paid': self.total_commission_paid,
                'active_strategies': len(self.strategy_positions),
                'closed_positions': len(self.closed_positions)
            }
    
    def get_strategy_summary(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get summary for a specific strategy.
        
        Args:
            strategy_id: Strategy to summarize
            
        Returns:
            Strategy performance summary
        """
        with self.lock:
            strategy_positions = [
                pos for pos in self.positions.values()
                if pos.strategy_id == strategy_id
            ]
            
            # Calculate unrealized P&L for strategy
            strategy_unrealized = sum(pos.unrealized_pnl for pos in strategy_positions)
            
            # Total P&L (realized + unrealized)
            total_strategy_pnl = self.strategy_pnl[strategy_id] + strategy_unrealized
            
            # Position value
            positions_value = sum(
                pos.quantity * pos.current_price for pos in strategy_positions
            )
            
            return {
                'strategy_id': strategy_id,
                'num_positions': len(strategy_positions),
                'positions_value': positions_value,
                'unrealized_pnl': strategy_unrealized,
                'realized_pnl': self.strategy_pnl[strategy_id],
                'total_pnl': total_strategy_pnl,
                'symbols': [pos.symbol for pos in strategy_positions]
            }
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Calculate portfolio risk metrics."""
        with self.lock:
            positions_list = list(self.positions.values())
            
            if not positions_list:
                return {
                    'portfolio_beta': 0.0,
                    'concentration_risk': 0.0,
                    'largest_position_pct': 0.0,
                    'position_correlation': 0.0
                }
            
            # Calculate position concentrations
            position_values = [pos.quantity * pos.current_price for pos in positions_list]
            total_positions_value = sum(position_values)
            
            concentrations = [val / max(1, total_positions_value) for val in position_values]
            largest_position_pct = max(concentrations) if concentrations else 0.0
            
            # Concentration risk (Herfindahl Index)
            concentration_risk = sum(c**2 for c in concentrations)
            
            return {
                'concentration_risk': concentration_risk,
                'largest_position_pct': largest_position_pct * 100,
                'num_positions': len(positions_list),
                'avg_position_size': total_positions_value / len(positions_list) if positions_list else 0,
                'total_exposure': total_positions_value,
                'cash_percentage': self.cash / max(1, self.total_portfolio_value) * 100
            }
    
    def _create_new_position(self, fill: Fill, order: Order):
        """Create a new position from a fill."""
        # Determine position type
        if order.direction in [TradeDirection.BUY, TradeDirection.BUY_TO_COVER]:
            position_type = PositionType.LONG
            quantity = fill.quantity
        else:  # SELL_SHORT
            position_type = PositionType.SHORT
            quantity = -fill.quantity  # Negative for short positions
        
        position = Position(
            symbol=fill.symbol,
            position_type=position_type,
            quantity=quantity,
            entry_price=fill.price,
            entry_time=fill.timestamp,
            strategy_id=order.strategy_id or "unknown"
        )
        
        self.positions[fill.symbol] = position
        
        self.logger.info(f"New position created: {fill.symbol} {position_type.value} {quantity}")
    
    def _update_existing_position(self, fill: Fill, order: Order):
        """Update an existing position with a new fill."""
        position = self.positions[fill.symbol]
        
        if order.direction in [TradeDirection.BUY, TradeDirection.BUY_TO_COVER]:
            # Adding to long position or covering short
            if position.position_type == PositionType.LONG:
                # Adding to long position - weighted average price
                total_cost = (position.quantity * position.entry_price) + (fill.quantity * fill.price)
                new_quantity = position.quantity + fill.quantity
                position.entry_price = total_cost / new_quantity
                position.quantity = new_quantity
            else:
                # Covering short position
                position.quantity += fill.quantity
                if position.quantity >= 0:
                    # Position closed or flipped to long
                    if position.quantity == 0:
                        # Position closed exactly
                        self.close_position(fill.symbol, fill.price, fill.timestamp, "cover")
                    else:
                        # Flipped to long
                        position.position_type = PositionType.LONG
                        position.entry_price = fill.price
        
        else:  # SELL, SELL_SHORT
            if position.position_type == PositionType.LONG:
                # Reducing long position
                position.quantity -= fill.quantity
                if position.quantity <= 0:
                    if position.quantity == 0:
                        # Position closed exactly
                        self.close_position(fill.symbol, fill.price, fill.timestamp, "sell")
                    else:
                        # Flipped to short
                        position.position_type = PositionType.SHORT
                        position.quantity = -position.quantity
                        position.entry_price = fill.price
            else:
                # Adding to short position
                total_proceeds = (abs(position.quantity) * position.entry_price) + (fill.quantity * fill.price)
                new_quantity = abs(position.quantity) + fill.quantity
                position.entry_price = total_proceeds / new_quantity
                position.quantity = -new_quantity  # Keep negative for short
    
    def _recalculate_portfolio_value(self):
        """Recalculate total portfolio value."""
        positions_value = 0.0
        unrealized_pnl = 0.0
        
        for position in self.positions.values():
            if position.symbol in self.market_data_cache:
                current_price = self.market_data_cache[position.symbol]['price']
                position.update_price(current_price, datetime.now())
            
            position_value = position.quantity * position.current_price
            positions_value += position_value
            unrealized_pnl += position.unrealized_pnl
        
        self.total_portfolio_value = self.cash + positions_value
        self.unrealized_pnl = unrealized_pnl
        
        # Update peak value for drawdown calculation
        if self.total_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.total_portfolio_value
