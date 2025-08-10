"""
Core backtesting engine for strategy validation and performance analysis.

This module provides the main backtesting infrastructure that simulates
trading strategies against historical data with realistic execution modeling.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from ..strategies.base_strategy import BaseStrategy, Position, Trade, StrategyState, PositionType, TradeDirection
from ..data_management.historical_provider import HistoricalDataManager


@dataclass
class BacktestTrade:
    """Trade record for backtesting with P&L tracking."""
    symbol: str
    direction: TradeDirection
    quantity: float
    entry_price: float
    exit_price: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    strategy_id: str
    commission: float = 0.0
    pnl: float = 0.0
    
    @property
    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.exit_price is not None and self.exit_time is not None
    
    def close_trade(self, exit_price: float, exit_time: datetime) -> float:
        """Close the trade and calculate P&L."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        
        if self.direction in [TradeDirection.BUY]:
            self.pnl = (exit_price - self.entry_price) * self.quantity - self.commission
        else:  # SELL/SHORT
            self.pnl = (self.entry_price - exit_price) * self.quantity - self.commission
            
        return self.pnl


class BacktestStatus(Enum):
    """Backtest execution status."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BacktestConfig:
    """Configuration for backtest execution."""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission_per_trade: float = 1.0
    slippage_bps: float = 5.0  # Basis points
    min_trade_size: float = 100.0
    max_position_size: float = 10000.0
    benchmark_symbol: str = "SPY"
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if self.commission_per_trade < 0:
            raise ValueError("Commission cannot be negative")


@dataclass
class BacktestResult:
    """Results from backtest execution."""
    config: BacktestConfig
    status: BacktestStatus
    start_time: datetime
    end_time: Optional[datetime]
    strategy_results: Dict[str, Any]
    portfolio_history: pd.DataFrame
    trade_history: List[BacktestTrade]
    performance_metrics: Dict[str, float]
    error_message: Optional[str] = None
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get backtest execution duration."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def total_return(self) -> float:
        """Calculate total return percentage."""
        if self.portfolio_history.empty:
            return 0.0
        initial_value = self.portfolio_history['portfolio_value'].iloc[0]
        final_value = self.portfolio_history['portfolio_value'].iloc[-1]
        return (final_value - initial_value) / initial_value * 100
    
    @property
    def total_trades(self) -> int:
        """Get total number of trades executed."""
        return len(self.trade_history)


class BacktestingEngine:
    """
    Main backtesting engine that orchestrates strategy testing.
    
    This engine simulates trading strategies against historical data,
    accounting for realistic market conditions including slippage,
    commissions, and position sizing constraints.
    """
    
    def __init__(self, data_manager: HistoricalDataManager):
        """
        Initialize the backtesting engine.
        
        Args:
            data_manager: DataManager instance for historical data access
        """
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        self.current_backtest: Optional[BacktestResult] = None
        
    def run_backtest(
        self,
        strategy: BaseStrategy,
        config: BacktestConfig,
        symbols: List[str]
    ) -> BacktestResult:
        """
        Execute a complete backtest for a strategy.
        
        Args:
            strategy: Trading strategy to test
            config: Backtest configuration
            symbols: List of symbols to include in backtest
            
        Returns:
            BacktestResult with performance metrics and trade history
        """
        self.logger.info(f"Starting backtest for strategy: {strategy.name}")
        
        # Initialize backtest result
        result = BacktestResult(
            config=config,
            status=BacktestStatus.NOT_STARTED,
            start_time=datetime.now(),
            end_time=None,
            strategy_results={},
            portfolio_history=pd.DataFrame(),
            trade_history=[],
            performance_metrics={}
        )
        
        self.current_backtest = result
        
        try:
            result.status = BacktestStatus.RUNNING
            
            # Get historical data
            self.logger.info(f"Loading historical data for {len(symbols)} symbols")
            historical_data = self._load_historical_data(symbols, config)
            
            if historical_data.empty:
                raise ValueError("No historical data available for backtest period")
            
            # Initialize portfolio simulator
            portfolio_sim = PortfolioSimulator(config.initial_capital)
            
            # Initialize strategy
            strategy.start()
            
            # Run simulation
            portfolio_history, trade_history = self._run_simulation(
                strategy, historical_data, config, portfolio_sim
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                portfolio_history, trade_history, config
            )
            
            # Populate results
            result.portfolio_history = portfolio_history
            result.trade_history = trade_history
            result.performance_metrics = performance_metrics
            result.strategy_results = strategy.get_performance_summary()
            result.status = BacktestStatus.COMPLETED
            result.end_time = datetime.now()
            
            self.logger.info(f"Backtest completed successfully")
            self.logger.info(f"Total return: {result.total_return:.2f}%")
            self.logger.info(f"Total trades: {result.total_trades}")
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            result.status = BacktestStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
        
        finally:
            strategy.stop()
        
        return result
    
    def _load_historical_data(
        self, 
        symbols: List[str], 
        config: BacktestConfig
    ) -> pd.DataFrame:
        """Load and prepare historical data for backtesting."""
        try:
            # For demo purposes, generate synthetic data
            # In production, this would load real historical data
            return self._generate_synthetic_data(symbols, config)
            
        except Exception as e:
            self.logger.error(f"Failed to load historical data: {str(e)}")
            return pd.DataFrame()
    
    def _generate_synthetic_data(
        self, 
        symbols: List[str], 
        config: BacktestConfig
    ) -> pd.DataFrame:
        """Generate synthetic market data for backtesting demo."""
        date_range = pd.date_range(
            start=config.start_date,
            end=config.end_date,
            freq='D'
        )
        
        data_frames = []
        
        for symbol in symbols:
            # Generate realistic price series
            np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
            
            initial_price = 100.0 + np.random.normal(0, 20)
            returns = np.random.normal(0.0005, 0.02, len(date_range))  # Daily returns
            prices = [initial_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # Create OHLCV data
            symbol_data = pd.DataFrame({
                'date': date_range,
                'symbol': symbol,
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'close': prices,
                'volume': np.random.randint(10000, 1000000, len(date_range))
            })
            
            # Ensure high >= close >= low
            symbol_data['high'] = symbol_data[['open', 'close', 'high']].max(axis=1)
            symbol_data['low'] = symbol_data[['open', 'close', 'low']].min(axis=1)
            
            data_frames.append(symbol_data)
        
        combined_data = pd.concat(data_frames, ignore_index=True)
        return combined_data.sort_values(['date', 'symbol']).reset_index(drop=True)
    
    def _run_simulation(
        self,
        strategy: BaseStrategy,
        historical_data: pd.DataFrame,
        config: BacktestConfig,
        portfolio_sim: 'PortfolioSimulator'
            ) -> Tuple[pd.DataFrame, List[BacktestTrade]]:
        """Run the main simulation loop."""
        portfolio_history = []
        all_trades = []
        
        # Group data by date for daily simulation
        dates = sorted(historical_data['date'].unique())
        
        for i, current_date in enumerate(dates):
            day_data = historical_data[historical_data['date'] == current_date]
            
            # Update strategy with current market data
            market_data = {}
            for _, row in day_data.iterrows():
                market_data[row['symbol']] = {
                    'price': row['close'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'volume': row['volume']
                }
            
            # Strategy update (this would generate signals)
            strategy.update(market_data)
            
            # Process any new trades from strategy
            new_trades = self._process_strategy_signals(
                strategy, market_data, config, portfolio_sim
            )
            all_trades.extend(new_trades)
            
            # Update portfolio value
            portfolio_value = portfolio_sim.calculate_portfolio_value(market_data)
            
            # Record portfolio state
            portfolio_history.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': portfolio_sim.cash,
                'positions_value': portfolio_value - portfolio_sim.cash,
                'num_positions': len(portfolio_sim.positions),
                'num_trades_today': len(new_trades)
            })
            
            # Log progress periodically
            if i % 30 == 0:  # Every 30 days
                self.logger.info(f"Simulation progress: {current_date.strftime('%Y-%m-%d')}, "
                               f"Portfolio: ${portfolio_value:,.2f}")
        
        return pd.DataFrame(portfolio_history), all_trades
    
    def _process_strategy_signals(
        self,
        strategy: BaseStrategy,
        market_data: Dict[str, Dict],
        config: BacktestConfig,
        portfolio_sim: 'PortfolioSimulator'
    ) -> List[BacktestTrade]:
        """Process trading signals from strategy and execute trades."""
        new_trades = []
        
        # Get pending positions that should be opened/closed
        # This is a simplified implementation - in practice would be more complex
        
        # For demo, randomly generate some trades based on strategy state
        if np.random.random() < 0.05:  # 5% chance of trade per day
            symbol = np.random.choice(list(market_data.keys()))
            price = market_data[symbol]['price']
            
            # Apply slippage and commission
            slippage = price * (config.slippage_bps / 10000)
            execution_price = price + slippage
            
            quantity = min(config.max_position_size / execution_price, 100)
            
            if quantity >= config.min_trade_size / execution_price:
                # Create a backtest trade
                trade = BacktestTrade(
                    symbol=symbol,
                    direction=TradeDirection.BUY,
                    quantity=quantity,
                    entry_price=execution_price,
                    exit_price=None,
                    entry_time=datetime.now(),
                    exit_time=None,
                    strategy_id=strategy.strategy_id,
                    commission=config.commission_per_trade
                )
                
                # Execute trade in portfolio simulator
                if portfolio_sim.execute_trade(trade):
                    new_trades.append(trade)
        
        return new_trades
    
    def _calculate_performance_metrics(
        self,
        portfolio_history: pd.DataFrame,
        trade_history: List[BacktestTrade],
        config: BacktestConfig
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if portfolio_history.empty:
            return {}
        
        # Basic return metrics
        initial_value = config.initial_capital
        final_value = portfolio_history['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Daily returns for advanced metrics
        portfolio_history['daily_return'] = portfolio_history['portfolio_value'].pct_change()
        daily_returns = portfolio_history['daily_return'].dropna()
        
        # Calculate metrics
        metrics = {
            'total_return': total_return * 100,
            'total_return_abs': final_value - initial_value,
            'annualized_return': self._annualized_return(daily_returns),
            'volatility': daily_returns.std() * np.sqrt(252) * 100,
            'sharpe_ratio': self._calculate_sharpe_ratio(daily_returns),
            'max_drawdown': self._calculate_max_drawdown(portfolio_history['portfolio_value']),
            'calmar_ratio': self._calculate_calmar_ratio(daily_returns, portfolio_history['portfolio_value']),
            'win_rate': self._calculate_win_rate(trade_history),
            'profit_factor': self._calculate_profit_factor(trade_history),
            'avg_trade_return': self._calculate_avg_trade_return(trade_history),
            'num_trades': len(trade_history),
            'trading_days': len(portfolio_history),
            'final_portfolio_value': final_value
        }
        
        return metrics
    
    def _annualized_return(self, daily_returns: pd.Series) -> float:
        """Calculate annualized return."""
        if len(daily_returns) == 0:
            return 0.0
        mean_daily_return = daily_returns.mean()
        return (1 + mean_daily_return) ** 252 - 1
    
    def _calculate_sharpe_ratio(self, daily_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(daily_returns) == 0 or daily_returns.std() == 0:
            return 0.0
        
        excess_returns = daily_returns - (risk_free_rate / 252)
        return (excess_returns.mean() / daily_returns.std()) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown percentage."""
        if len(portfolio_values) == 0:
            return 0.0
        
        cumulative_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - cumulative_max) / cumulative_max
        return abs(drawdown.min()) * 100
    
    def _calculate_calmar_ratio(self, daily_returns: pd.Series, portfolio_values: pd.Series) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        annual_return = self._annualized_return(daily_returns)
        max_dd = self._calculate_max_drawdown(portfolio_values)
        
        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0.0
        return annual_return / (max_dd / 100)
    
    def _calculate_win_rate(self, trades: List[BacktestTrade]) -> float:
        """Calculate percentage of winning trades."""
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if trade.pnl > 0)
        return (winning_trades / len(trades)) * 100
    
    def _calculate_profit_factor(self, trades: List[BacktestTrade]) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if not trades:
            return 0.0
        
        gross_profit = sum(trade.pnl for trade in trades if trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in trades if trade.pnl < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 1.0
        return gross_profit / gross_loss
    
    def _calculate_avg_trade_return(self, trades: List[BacktestTrade]) -> float:
        """Calculate average return per trade."""
        if not trades:
            return 0.0
        
        total_pnl = sum(trade.pnl for trade in trades)
        return total_pnl / len(trades)


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
        
    def execute_trade(self, trade: BacktestTrade) -> bool:
        """
        Execute a trade and update portfolio state.
        
        Args:
            trade: Trade to execute
            
        Returns:
            True if trade was executed successfully
        """
        total_cost = trade.quantity * trade.entry_price + trade.commission
        
        if total_cost > self.cash:
            return False  # Insufficient funds
        
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
