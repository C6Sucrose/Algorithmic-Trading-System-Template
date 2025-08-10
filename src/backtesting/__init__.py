"""
Backtesting framework for algorithmic trading strategies.

This module provides comprehensive backtesting capabilities including:
- Historical data processing
- Strategy performance evaluation
- Risk metrics calculation
- Trade analysis and reporting
- Portfolio simulation
"""

from .backtesting_engine import BacktestingEngine
from .portfolio_simulator import PortfolioSimulator
from .performance_analyzer import PerformanceAnalyzer
from .trade_simulator import TradeSimulator
from .data_handler import BacktestDataHandler

__all__ = [
    'BacktestingEngine',
    'PortfolioSimulator', 
    'PerformanceAnalyzer',
    'TradeSimulator',
    'BacktestDataHandler'
]
