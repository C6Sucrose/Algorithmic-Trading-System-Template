"""
Base Strategy Framework

Defines the foundation for all trading strategies in the AlgoB system.
Provides core abstractions for positions, trades, and strategy lifecycle management.
"""

import sys
from pathlib import Path
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import pandas as pd
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config_manager import config


class PositionType(Enum):
    """Types of trading positions."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class PositionStatus(Enum):
    """Status of trading positions."""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"
    PENDING = "pending"


class TradeDirection(Enum):
    """Direction of trade execution."""
    BUY = "buy"
    SELL = "sell"
    BUY_TO_COVER = "buy_to_cover"
    SELL_SHORT = "sell_short"


class StrategyState(Enum):
    """States of strategy execution."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class Position:
    """Represents a trading position."""
    
    def __init__(
        self,
        symbol: str,
        position_type: PositionType,
        quantity: float,
        entry_price: float,
        entry_time: datetime,
        strategy_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        self.symbol = symbol
        self.position_type = position_type
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.strategy_id = strategy_id
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # Position tracking
        self.status = PositionStatus.OPEN
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.last_update = entry_time
        
        # Risk metrics
        self.max_unrealized_pnl = 0.0
        self.min_unrealized_pnl = 0.0
        self.duration = timedelta(0)
    
    def update_price(self, current_price: float, timestamp: datetime) -> None:
        """Update position with current market price."""
        self.current_price = current_price
        self.last_update = timestamp
        self.duration = timestamp - self.entry_time
        
        # Calculate unrealized P&L
        if self.position_type == PositionType.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        elif self.position_type == PositionType.SHORT:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
        
        # Track P&L extremes
        self.max_unrealized_pnl = max(self.max_unrealized_pnl, self.unrealized_pnl)
        self.min_unrealized_pnl = min(self.min_unrealized_pnl, self.unrealized_pnl)
    
    def check_exit_conditions(self) -> Tuple[bool, str]:
        """
        Check if position should be closed based on stop loss or take profit.
        
        Returns:
            Tuple of (should_exit, reason)
        """
        if self.stop_loss is not None:
            if self.position_type == PositionType.LONG and self.current_price <= self.stop_loss:
                return True, "stop_loss"
            elif self.position_type == PositionType.SHORT and self.current_price >= self.stop_loss:
                return True, "stop_loss"
        
        if self.take_profit is not None:
            if self.position_type == PositionType.LONG and self.current_price >= self.take_profit:
                return True, "take_profit"
            elif self.position_type == PositionType.SHORT and self.current_price <= self.take_profit:
                return True, "take_profit"
        
        return False, "none"
    
    def close_position(self, exit_price: float, exit_time: datetime, reason: str = "manual") -> float:
        """
        Close the position and calculate realized P&L.
        
        Returns:
            Realized P&L from the position
        """
        self.status = PositionStatus.CLOSED
        
        # Calculate final P&L
        if self.position_type == PositionType.LONG:
            self.realized_pnl = (exit_price - self.entry_price) * self.quantity
        elif self.position_type == PositionType.SHORT:
            self.realized_pnl = (self.entry_price - exit_price) * self.quantity
        
        self.duration = exit_time - self.entry_time
        return self.realized_pnl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'position_type': self.position_type.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'status': self.status.value,
            'strategy_id': self.strategy_id,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'duration_seconds': self.duration.total_seconds()
        }


class Trade:
    """Represents a completed trade."""
    
    def __init__(
        self,
        symbol: str,
        direction: TradeDirection,
        quantity: float,
        price: float,
        timestamp: datetime,
        strategy_id: str,
        commission: float = 0.0
    ):
        self.symbol = symbol
        self.direction = direction
        self.quantity = quantity
        self.price = price
        self.timestamp = timestamp
        self.strategy_id = strategy_id
        self.commission = commission
        
        # Trade metadata
        self.trade_id = f"{strategy_id}_{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        self.notional_value = abs(quantity * price)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'direction': self.direction.value,
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp,
            'strategy_id': self.strategy_id,
            'commission': self.commission,
            'notional_value': self.notional_value
        }


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(
        self,
        strategy_id: str,
        name: str,
        universe_manager=None,
        data_manager=None,
        config_section: str = "STRATEGY"
    ):
        """
        Initialize base strategy.
        
        Args:
            strategy_id: Unique identifier for the strategy
            name: Human-readable strategy name
            universe_manager: Universe manager for symbol filtering
            data_manager: Data manager for market data access
            config_section: Configuration section name
        """
        self.strategy_id = strategy_id
        self.name = name
        self.universe_manager = universe_manager
        self.data_manager = data_manager
        
        # Configuration
        self.config = config.get(config_section, {})
        self.max_positions = self.config.get('MAX_POSITIONS', 10)
        self.max_portfolio_value = self.config.get('MAX_PORTFOLIO_VALUE', 100000)
        self.position_size_percent = self.config.get('POSITION_SIZE_PERCENT', 0.05)  # 5% per position
        
        # Strategy state
        self.state = StrategyState.INACTIVE
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        self.performance_metrics: Dict[str, float] = {}
        
        # Timing
        self.start_time: Optional[datetime] = None
        self.last_update: Optional[datetime] = None
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.{self.strategy_id}")
        
        # Initialize performance tracking
        self._initialize_performance_tracking()
    
    def _initialize_performance_tracking(self) -> None:
        """Initialize performance tracking metrics."""
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0
        }
    
    @abstractmethod
    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on market data.
        
        Args:
            market_data: Dictionary of symbol -> OHLCV dataframes
            
        Returns:
            List of trading signals
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, symbol: str, signal: Dict[str, Any]) -> float:
        """
        Calculate appropriate position size for a trading signal.
        
        Args:
            symbol: Stock symbol
            signal: Trading signal information
            
        Returns:
            Position size (number of shares)
        """
        pass
    
    @abstractmethod
    def manage_risk(self, positions: Dict[str, Position]) -> List[Dict[str, Any]]:
        """
        Manage risk for existing positions.
        
        Args:
            positions: Current positions
            
        Returns:
            List of risk management actions
        """
        pass
    
    def start(self) -> None:
        """Start strategy execution."""
        if self.state == StrategyState.INACTIVE:
            self.state = StrategyState.ACTIVE
            self.start_time = datetime.now()
            self.logger.info(f"Strategy {self.name} started")
        else:
            self.logger.warning(f"Strategy {self.name} is already running")
    
    def stop(self) -> None:
        """Stop strategy execution."""
        self.state = StrategyState.STOPPED
        self.logger.info(f"Strategy {self.name} stopped")
        
        # Close all open positions
        self._close_all_positions("strategy_stopped")
    
    def pause(self) -> None:
        """Pause strategy execution."""
        if self.state == StrategyState.ACTIVE:
            self.state = StrategyState.PAUSED
            self.logger.info(f"Strategy {self.name} paused")
    
    def resume(self) -> None:
        """Resume strategy execution."""
        if self.state == StrategyState.PAUSED:
            self.state = StrategyState.ACTIVE
            self.logger.info(f"Strategy {self.name} resumed")
    
    def update(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """
        Main strategy update method called on each market data update.
        
        Args:
            market_data: Current market data
        """
        if self.state != StrategyState.ACTIVE:
            return
        
        try:
            self.last_update = datetime.now()
            
            # Update existing positions
            self._update_positions(market_data)
            
            # Generate new signals
            signals = self.generate_signals(market_data)
            
            # Execute signals
            for signal in signals:
                self._execute_signal(signal, market_data)
            
            # Manage risk
            risk_actions = self.manage_risk(self.positions)
            for action in risk_actions:
                self._execute_risk_action(action)
            
            # Update performance metrics
            self._update_performance_metrics()
            
        except Exception as e:
            self.logger.error(f"Error updating strategy {self.name}: {e}")
            self.state = StrategyState.ERROR
    
    def _update_positions(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """Update all positions with current market data."""
        current_time = datetime.now()
        
        for symbol, position in self.positions.items():
            if symbol in market_data and not market_data[symbol].empty:
                current_price = market_data[symbol]['Close'].iloc[-1]
                position.update_price(current_price, current_time)
                
                # Check exit conditions
                should_exit, reason = position.check_exit_conditions()
                if should_exit:
                    self._close_position(symbol, reason)
    
    def _execute_signal(self, signal: Dict[str, Any], market_data: Dict[str, pd.DataFrame]) -> None:
        """Execute a trading signal."""
        symbol = signal['symbol']
        action = signal['action']  # 'buy', 'sell', 'buy_short', 'cover'
        
        # Check if we already have a position
        if symbol in self.positions and action in ['buy', 'sell_short']:
            self.logger.debug(f"Already have position in {symbol}, skipping signal")
            return
        
        # Check position limits
        if len(self.positions) >= self.max_positions and action in ['buy', 'sell_short']:
            self.logger.debug(f"Maximum positions reached, skipping signal for {symbol}")
            return
        
        # Calculate position size
        position_size = self.calculate_position_size(symbol, signal)
        
        if position_size == 0:
            return
        
        # Get current price
        if symbol not in market_data or market_data[symbol].empty:
            self.logger.warning(f"No market data for {symbol}, skipping signal")
            return
        
        current_price = market_data[symbol]['Close'].iloc[-1]
        current_time = datetime.now()
        
        # Execute the trade
        if action == 'buy':
            self._open_long_position(symbol, position_size, current_price, current_time, signal)
        elif action == 'sell_short':
            self._open_short_position(symbol, position_size, current_price, current_time, signal)
        elif action == 'sell' and symbol in self.positions:
            self._close_position(symbol, "signal_exit")
        elif action == 'cover' and symbol in self.positions:
            self._close_position(symbol, "signal_exit")
    
    def _open_long_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        signal: Dict[str, Any]
    ) -> None:
        """Open a long position."""
        position = Position(
            symbol=symbol,
            position_type=PositionType.LONG,
            quantity=quantity,
            entry_price=price,
            entry_time=timestamp,
            strategy_id=self.strategy_id,
            stop_loss=signal.get('stop_loss'),
            take_profit=signal.get('take_profit')
        )
        
        self.positions[symbol] = position
        
        # Record the trade
        trade = Trade(
            symbol=symbol,
            direction=TradeDirection.BUY,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            strategy_id=self.strategy_id
        )
        self.trade_history.append(trade)
        
        self.logger.info(f"Opened long position: {symbol} x{quantity} @ ${price:.2f}")
    
    def _open_short_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        signal: Dict[str, Any]
    ) -> None:
        """Open a short position."""
        position = Position(
            symbol=symbol,
            position_type=PositionType.SHORT,
            quantity=quantity,
            entry_price=price,
            entry_time=timestamp,
            strategy_id=self.strategy_id,
            stop_loss=signal.get('stop_loss'),
            take_profit=signal.get('take_profit')
        )
        
        self.positions[symbol] = position
        
        # Record the trade
        trade = Trade(
            symbol=symbol,
            direction=TradeDirection.SELL_SHORT,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            strategy_id=self.strategy_id
        )
        self.trade_history.append(trade)
        
        self.logger.info(f"Opened short position: {symbol} x{quantity} @ ${price:.2f}")
    
    def _close_position(self, symbol: str, reason: str) -> None:
        """Close an existing position."""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        current_time = datetime.now()
        exit_price = position.current_price
        
        # Close the position
        realized_pnl = position.close_position(exit_price, current_time, reason)
        
        # Record the trade
        if position.position_type == PositionType.LONG:
            trade_direction = TradeDirection.SELL
        else:
            trade_direction = TradeDirection.BUY_TO_COVER
        
        trade = Trade(
            symbol=symbol,
            direction=trade_direction,
            quantity=position.quantity,
            price=exit_price,
            timestamp=current_time,
            strategy_id=self.strategy_id
        )
        self.trade_history.append(trade)
        
        # Remove from active positions
        del self.positions[symbol]
        
        self.logger.info(f"Closed position: {symbol} @ ${exit_price:.2f}, P&L: ${realized_pnl:.2f}, Reason: {reason}")
    
    def _close_all_positions(self, reason: str) -> None:
        """Close all open positions."""
        symbols_to_close = list(self.positions.keys())
        for symbol in symbols_to_close:
            self._close_position(symbol, reason)
    
    def _execute_risk_action(self, action: Dict[str, Any]) -> None:
        """Execute a risk management action."""
        action_type = action.get('type')
        symbol = action.get('symbol')
        
        if action_type == 'close_position' and symbol in self.positions:
            self._close_position(symbol, action.get('reason', 'risk_management'))
        elif action_type == 'adjust_stop_loss' and symbol in self.positions:
            new_stop_loss = action.get('new_stop_loss')
            self.positions[symbol].stop_loss = new_stop_loss
            self.logger.info(f"Adjusted stop loss for {symbol}: ${new_stop_loss:.2f}")
    
    def _update_performance_metrics(self) -> None:
        """Update strategy performance metrics."""
        if not self.trade_history:
            return
        
        # Calculate basic metrics
        closed_trades = []
        pnl_values = []
        
        # Group trades into complete round trips
        trade_pairs = {}
        for trade in self.trade_history:
            symbol = trade.symbol
            if symbol not in trade_pairs:
                trade_pairs[symbol] = []
            trade_pairs[symbol].append(trade)
        
        # Calculate P&L for each completed round trip
        for symbol, trades in trade_pairs.items():
            if len(trades) >= 2:
                # Simple approach: pair entry and exit trades
                for i in range(0, len(trades) - 1, 2):
                    entry_trade = trades[i]
                    exit_trade = trades[i + 1]
                    
                    if entry_trade.direction in [TradeDirection.BUY, TradeDirection.SELL_SHORT]:
                        if entry_trade.direction == TradeDirection.BUY:
                            pnl = (exit_trade.price - entry_trade.price) * entry_trade.quantity
                        else:
                            pnl = (entry_trade.price - exit_trade.price) * entry_trade.quantity
                        
                        pnl_values.append(pnl)
                        if pnl > 0:
                            closed_trades.append('win')
                        else:
                            closed_trades.append('loss')
        
        # Update metrics
        if pnl_values:
            self.performance_metrics['total_trades'] = len(pnl_values)
            self.performance_metrics['winning_trades'] = sum(1 for pnl in pnl_values if pnl > 0)
            self.performance_metrics['losing_trades'] = sum(1 for pnl in pnl_values if pnl <= 0)
            self.performance_metrics['total_pnl'] = sum(pnl_values)
            self.performance_metrics['win_rate'] = self.performance_metrics['winning_trades'] / len(pnl_values)
            
            winning_trades = [pnl for pnl in pnl_values if pnl > 0]
            losing_trades = [pnl for pnl in pnl_values if pnl <= 0]
            
            if winning_trades:
                self.performance_metrics['avg_win'] = sum(winning_trades) / len(winning_trades)
            if losing_trades:
                self.performance_metrics['avg_loss'] = sum(losing_trades) / len(losing_trades)
    
    def get_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        total_value = 0.0
        for position in self.positions.values():
            total_value += position.current_price * position.quantity
        return total_value
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'strategy_id': self.strategy_id,
            'name': self.name,
            'state': self.state.value,
            'start_time': self.start_time,
            'last_update': self.last_update,
            'open_positions': len(self.positions),
            'total_trades': len(self.trade_history),
            'portfolio_value': self.get_portfolio_value(),
            'performance_metrics': self.performance_metrics.copy()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary."""
        return {
            'strategy_id': self.strategy_id,
            'name': self.name,
            'state': self.state.value,
            'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
            'trade_history': [trade.to_dict() for trade in self.trade_history],
            'performance_metrics': self.performance_metrics.copy()
        }
