"""
Strategies Package

Contains all trading strategy implementations for the AlgoB system.
Provides base classes and specific strategy implementations.
"""

from .base_strategy import BaseStrategy, Position, Trade, StrategyState
from .mean_reversion_strategy import MeanReversionStrategy
from .pairs_trading_strategy import PairsTradingStrategy

__all__ = [
    'BaseStrategy',
    'Position', 
    'Trade',
    'StrategyState',
    'MeanReversionStrategy',
    'PairsTradingStrategy'
]
