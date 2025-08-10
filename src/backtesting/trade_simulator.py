"""
Trade Simulator for Backtesting

Handles trade execution simulation with realistic market conditions
including slippage, commissions, and market impact modeling.
"""

import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

from .backtesting_engine import BacktestTrade, BacktestConfig
from ..strategies.base_strategy import TradeDirection


class TradeSimulator:
    """
    Simulates realistic trade execution during backtesting.
    
    Models market impact, slippage, and timing effects to provide
    realistic execution prices and costs.
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize trade simulator.
        
        Args:
            config: Backtest configuration with execution parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def simulate_trade_execution(
        self,
        symbol: str,
        direction: TradeDirection,
        quantity: float,
        target_price: float,
        timestamp: datetime,
        strategy_id: str
    ) -> Optional[BacktestTrade]:
        """
        Simulate the execution of a trade.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction (BUY/SELL)
            quantity: Quantity to trade
            target_price: Target execution price
            timestamp: Trade timestamp
            strategy_id: Strategy identifier
            
        Returns:
            BacktestTrade if execution successful, None otherwise
        """
        # Check minimum trade size
        notional_value = quantity * target_price
        if notional_value < self.config.min_trade_size:
            self.logger.warning(f"Trade size ${notional_value:.2f} below minimum ${self.config.min_trade_size}")
            return None
        
        # Check maximum position size
        if notional_value > self.config.max_position_size:
            self.logger.warning(f"Trade size ${notional_value:.2f} exceeds maximum ${self.config.max_position_size}")
            return None
        
        # Calculate execution price with slippage
        execution_price = self._apply_slippage(target_price, direction, quantity)
        
        # Apply market impact (simplified model)
        execution_price = self._apply_market_impact(execution_price, quantity, notional_value)
        
        # Create trade record
        trade = BacktestTrade(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=execution_price,
            exit_price=None,
            entry_time=timestamp,
            exit_time=None,
            strategy_id=strategy_id,
            commission=self.config.commission_per_trade
        )
        
        self.logger.debug(f"Executed trade: {symbol} {direction.value} {quantity:.0f} @ ${execution_price:.2f}")
        
        return trade
    
    def _apply_slippage(self, price: float, direction: TradeDirection, quantity: float) -> float:
        """
        Apply slippage to execution price.
        
        Args:
            price: Target price
            direction: Trade direction
            quantity: Trade quantity
            
        Returns:
            Price with slippage applied
        """
        # Convert basis points to decimal
        slippage_rate = self.config.slippage_bps / 10000
        
        # Random slippage within configured range
        random_factor = random.uniform(0.5, 1.5)  # 50% to 150% of expected slippage
        slippage = price * slippage_rate * random_factor
        
        # Apply slippage direction (buy higher, sell lower)
        if direction in [TradeDirection.BUY, TradeDirection.BUY_TO_COVER]:
            return price + slippage
        else:  # SELL, SELL_SHORT
            return price - slippage
    
    def _apply_market_impact(self, price: float, quantity: float, notional_value: float) -> float:
        """
        Apply market impact to execution price.
        
        Simplified model: larger trades have more impact.
        
        Args:
            price: Current price
            quantity: Trade quantity
            notional_value: Trade notional value
            
        Returns:
            Price with market impact applied
        """
        # Simple market impact model based on trade size
        if notional_value > 50000:  # Large trade
            impact_rate = 0.0005  # 5 basis points
        elif notional_value > 10000:  # Medium trade
            impact_rate = 0.0002  # 2 basis points
        else:  # Small trade
            impact_rate = 0.0001  # 1 basis point
        
        impact = price * impact_rate
        
        # Market impact always goes against the trader
        return price + impact
    
    def simulate_order_fill(
        self,
        trade: BacktestTrade,
        market_data: Dict[str, Dict],
        current_time: datetime
    ) -> bool:
        """
        Simulate whether an order would be filled given market conditions.
        
        Args:
            trade: Trade to simulate
            market_data: Current market data
            current_time: Current timestamp
            
        Returns:
            True if order would be filled
        """
        if trade.symbol not in market_data:
            return False
        
        symbol_data = market_data[trade.symbol]
        current_price = symbol_data['price']
        
        # Simple fill logic based on price and volume
        # In reality, this would be much more sophisticated
        
        # Check if price is reasonable (within 5% of market)
        price_diff = abs(trade.entry_price - current_price) / current_price
        if price_diff > 0.05:  # 5% threshold
            return False
        
        # Check volume (simplified)
        if 'volume' in symbol_data:
            daily_volume = symbol_data['volume']
            if trade.quantity > daily_volume * 0.01:  # Can't be more than 1% of daily volume
                return False
        
        return True
    
    def calculate_transaction_costs(self, trade: BacktestTrade) -> float:
        """
        Calculate total transaction costs for a trade.
        
        Args:
            trade: Trade to calculate costs for
            
        Returns:
            Total transaction costs
        """
        costs = trade.commission
        
        # Add any additional fees (spread, regulatory, etc.)
        # For now, just commission
        
        return costs
