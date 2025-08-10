"""
Data Handler for Backtesting

Manages historical data loading, preprocessing, and serving
for backtesting operations with efficient data access patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

from ..data_management.historical_provider import HistoricalDataManager
from .backtesting_engine import BacktestConfig


class BacktestDataHandler:
    """
    Handles data operations for backtesting.
    
    Manages historical data loading, preprocessing, and efficient
    data serving for strategy backtesting operations.
    """
    
    def __init__(self, historical_manager: Optional[HistoricalDataManager] = None):
        """
        Initialize data handler.
        
        Args:
            historical_manager: Optional historical data manager
        """
        self.historical_manager = historical_manager
        self.logger = logging.getLogger(__name__)
        self._data_cache: Dict[str, pd.DataFrame] = {}
    
    def load_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1D"
    ) -> pd.DataFrame:
        """
        Load historical data for backtesting.
        
        Args:
            symbols: List of symbols to load
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe (1D, 1H, etc.)
            
        Returns:
            Combined DataFrame with historical data
        """
        self.logger.info(f"Loading data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # For demo purposes, generate synthetic data
        # In production, this would load real historical data
        combined_data = self._generate_synthetic_market_data(symbols, start_date, end_date)
        
        # Cache the data
        cache_key = f"{','.join(symbols)}_{start_date}_{end_date}_{timeframe}"
        self._data_cache[cache_key] = combined_data
        
        self.logger.info(f"Loaded {len(combined_data)} rows of data")
        return combined_data
    
    def _generate_synthetic_market_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate realistic synthetic market data for backtesting."""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        all_data = []
        
        for symbol in symbols:
            # Set seed for consistent data per symbol
            np.random.seed(hash(symbol) % 2**32)
            
            # Generate base parameters
            initial_price = 50 + np.random.normal(0, 30)
            if initial_price <= 0:
                initial_price = 50
            
            # Generate price series with realistic characteristics
            volatility = 0.15 + np.random.uniform(-0.05, 0.05)  # Annual volatility
            drift = 0.08 + np.random.uniform(-0.04, 0.04)  # Annual drift
            
            daily_vol = volatility / np.sqrt(252)
            daily_drift = drift / 252
            
            prices = [initial_price]
            volumes = []
            
            for i in range(len(date_range)):
                # Generate return with mean reversion
                random_return = np.random.normal(daily_drift, daily_vol)
                
                # Add some mean reversion
                if len(prices) > 20:
                    recent_returns = np.diff(np.log(prices[-20:]))
                    avg_recent_return = np.mean(recent_returns)
                    mean_reversion = -0.1 * avg_recent_return  # 10% mean reversion
                    random_return += mean_reversion
                
                new_price = prices[-1] * np.exp(random_return)
                prices.append(new_price)
                
                # Generate volume (higher volume on larger price moves)
                price_change = abs(random_return)
                base_volume = 100000 + np.random.normal(0, 50000)
                volume_multiplier = 1 + price_change * 5  # Higher volume on big moves
                volume = max(int(base_volume * volume_multiplier), 1000)
                volumes.append(volume)
            
            # Remove the extra initial price
            prices = prices[1:]
            
            # Create OHLCV data
            symbol_data = []
            for i, (date, close_price, volume) in enumerate(zip(date_range, prices, volumes)):
                # Generate realistic OHLC from close price
                high_factor = 1 + abs(np.random.normal(0, 0.01))
                low_factor = 1 - abs(np.random.normal(0, 0.01))
                
                if i == 0:
                    open_price = close_price * np.random.uniform(0.99, 1.01)
                else:
                    # Open near previous close with gap
                    gap = np.random.normal(0, 0.005)
                    open_price = prices[i-1] * (1 + gap)
                
                high_price = max(open_price, close_price) * high_factor
                low_price = min(open_price, close_price) * low_factor
                
                symbol_data.append({
                    'date': date,
                    'symbol': symbol,
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close_price, 2),
                    'volume': volume
                })
            
            all_data.extend(symbol_data)
        
        # Create DataFrame and ensure proper sorting
        df = pd.DataFrame(all_data)
        df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
        
        return df
    
    def get_market_data_for_date(
        self,
        data: pd.DataFrame,
        target_date: datetime,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get market data for a specific date.
        
        Args:
            data: Historical data DataFrame
            target_date: Target date
            symbols: Optional list of symbols to filter
            
        Returns:
            Dictionary of market data by symbol
        """
        # Filter data for the target date
        date_data = data[data['date'] == target_date]
        
        if symbols:
            date_data = date_data[date_data['symbol'].isin(symbols)]
        
        # Convert to dictionary format
        market_data = {}
        for _, row in date_data.iterrows():
            market_data[row['symbol']] = {
                'price': row['close'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'volume': row['volume'],
                'date': row['date']
            }
        
        return market_data
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for backtesting.
        
        Args:
            data: Raw historical data
            
        Returns:
            Preprocessed data
        """
        # Create a copy to avoid modifying original
        processed_data = data.copy()
        
        # Ensure date column is datetime
        if 'date' in processed_data.columns:
            processed_data['date'] = pd.to_datetime(processed_data['date'])
        
        # Sort by date and symbol
        processed_data = processed_data.sort_values(['date', 'symbol']).reset_index(drop=True)
        
        # Add technical indicators
        processed_data = self._add_technical_indicators(processed_data)
        
        # Forward fill missing values (but be careful with look-ahead bias)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in processed_data.columns:
                processed_data[col] = processed_data.groupby('symbol')[col].fillna(method='ffill')
        
        return processed_data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to the data."""
        result_data = []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            
            if len(symbol_data) > 0:
                # Simple Moving Averages
                symbol_data['sma_20'] = symbol_data['close'].rolling(window=20).mean()
                symbol_data['sma_50'] = symbol_data['close'].rolling(window=50).mean()
                
                # Returns
                symbol_data['daily_return'] = symbol_data['close'].pct_change()
                
                # Volatility (20-day rolling)
                symbol_data['volatility_20'] = symbol_data['daily_return'].rolling(window=20).std()
                
                # High-Low spread
                symbol_data['hl_spread'] = (symbol_data['high'] - symbol_data['low']) / symbol_data['close']
                
            result_data.append(symbol_data)
        
        return pd.concat(result_data, ignore_index=True)
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for the data.
        
        Args:
            data: Historical data DataFrame
            
        Returns:
            Summary statistics
        """
        if data.empty:
            return {}
        
        summary = {
            'total_rows': len(data),
            'symbols': data['symbol'].nunique(),
            'symbol_list': data['symbol'].unique().tolist(),
            'date_range': {
                'start': data['date'].min(),
                'end': data['date'].max()
            },
            'trading_days': data['date'].nunique(),
            'data_completeness': {
                symbol: (data[data['symbol'] == symbol].shape[0] / data['date'].nunique()) * 100
                for symbol in data['symbol'].unique()
            }
        }
        
        return summary
