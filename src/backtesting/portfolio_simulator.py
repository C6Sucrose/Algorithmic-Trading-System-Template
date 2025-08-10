"""
Portfolio Simulator for Backtesting

Simulates portfolio behavior during backtesting with realistic
position tracking, cash management, and execution modeling.
"""

from typing import Dict, List, Optional
from datetime import datetime
import logging

from ..strategies.base_strategy import Position, PositionType
from .backtesting_engine import BacktestTrade


class PortfolioSimulator:
    """
    Simulates portfolio state during backtesting.
    
    Tracks cash, positions, and portfolio value with realistic
    execution modeling including commissions and slippage.
    """
    
    def __init__(self, initial_capital: float):
        """Initialize portfolio simulator."""
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[BacktestTrade] = []
        self.logger = logging.getLogger(__name__)
        
    def execute_trade(self, trade: BacktestTrade) -> bool:
        """
        Execute a trade and update portfolio state.
        
        Args:
            trade: BacktestTrade to execute
            
        Returns:
            True if trade was executed successfully
        """
        total_cost = trade.quantity * trade.entry_price + trade.commission
        
        if total_cost > self.cash:
            self.logger.warning(f"Insufficient funds for trade: ${total_cost:.2f} > ${self.cash:.2f}")
            return False
        
        # Update cash
        self.cash -= total_cost
        
        # Create or update position
        if trade.symbol in self.positions:
            # Add to existing position
            existing_pos = self.positions[trade.symbol]
            total_quantity = existing_pos.quantity + trade.quantity
            weighted_price = (
                (existing_pos.quantity * existing_pos.entry_price) +
                (trade.quantity * trade.entry_price)
            ) / total_quantity
            existing_pos.quantity = total_quantity
            existing_pos.entry_price = weighted_price
        else:
            # Create new position
            position = Position(
                symbol=trade.symbol,
                position_type=PositionType.LONG,  # Simplified for demo
                quantity=trade.quantity,
                entry_price=trade.entry_price,
                entry_time=trade.entry_time,
                strategy_id=trade.strategy_id
            )
            self.positions[trade.symbol] = position
        
        self.trade_history.append(trade)
        return True
    
    def close_position(self, symbol: str, exit_price: float, exit_time: datetime) -> Optional[float]:
        """
        Close a position and update cash.
        
        Args:
            symbol: Symbol to close
            exit_price: Exit price
            exit_time: Exit timestamp
            
        Returns:
            Realized P&L or None if position not found
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Calculate proceeds
        proceeds = position.quantity * exit_price
        
        # Calculate P&L
        pnl = position.close_position(exit_price, exit_time, "manual")
        
        # Update cash
        self.cash += proceeds
        
        # Remove position
        del self.positions[symbol]
        
        return pnl
    
    def calculate_portfolio_value(self, market_data: Dict[str, Dict]) -> float:
        """Calculate current portfolio value."""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]['price']
                position_value = position.quantity * current_price
                total_value += position_value
        
        return total_value
    
    def get_position_pnl(self, symbol: str, current_price: float) -> float:
        """Calculate unrealized P&L for a position."""
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        return (current_price - position.entry_price) * position.quantity
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary."""
        return {
            'cash': self.cash,
            'num_positions': len(self.positions),
            'total_trades': len(self.trade_history),
            'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()}
        }
