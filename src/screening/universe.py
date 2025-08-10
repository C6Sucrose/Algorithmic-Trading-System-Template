"""
Universe Manager

Manages the tradeable universe definition and filtering.
Handles stock selection criteria and maintains the active universe.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Any
import pandas as pd
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config_manager import config
from src.data_management.data_core import DataStorage
from src.data_management.historical_provider import HistoricalDataManager


class UniverseDefinition:
    """Defines criteria for the tradeable universe."""
    
    def __init__(self):
        """Initialize universe definition with config values."""
        self.min_price = config.get('UNIVERSE', {}).get('MIN_PRICE', 5.0)
        self.max_price = config.get('UNIVERSE', {}).get('MAX_PRICE', 500.0)
        self.min_market_cap = config.get('UNIVERSE', {}).get('MIN_MARKET_CAP', 100000000)  # $100M
        self.min_volume = config.get('UNIVERSE', {}).get('MIN_DAILY_VOLUME', 1000000)  # 1M shares
        self.min_liquidity = config.get('UNIVERSE', {}).get('MIN_LIQUIDITY', 1000000)  # $1M daily
        self.max_bid_ask_spread = config.get('UNIVERSE', {}).get('MAX_BID_ASK_SPREAD', 0.001)  # 0.1%
        self.exchanges = config.get('UNIVERSE', {}).get('EXCHANGES', ['NYSE', 'NASDAQ'])
        self.excluded_sectors = config.get('UNIVERSE', {}).get('EXCLUDED_SECTORS', [])
        self.excluded_symbols = config.get('UNIVERSE', {}).get('EXCLUDED_SYMBOLS', [])
        
        # Data quality requirements
        self.min_data_days = config.get('UNIVERSE', {}).get('MIN_DATA_DAYS', 252)  # 1 year
        self.max_data_gaps = config.get('UNIVERSE', {}).get('MAX_DATA_GAPS', 10)  # Max missing days
        self.min_data_quality = config.get('UNIVERSE', {}).get('MIN_DATA_QUALITY', 0.95)  # 95%
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert universe definition to dictionary."""
        return {
            'min_price': self.min_price,
            'max_price': self.max_price,
            'min_market_cap': self.min_market_cap,
            'min_volume': self.min_volume,
            'min_liquidity': self.min_liquidity,
            'max_bid_ask_spread': self.max_bid_ask_spread,
            'exchanges': self.exchanges,
            'excluded_sectors': self.excluded_sectors,
            'excluded_symbols': self.excluded_symbols,
            'min_data_days': self.min_data_days,
            'max_data_gaps': self.max_data_gaps,
            'min_data_quality': self.min_data_quality
        }


class UniverseManager:
    """Manages the tradeable universe and applies filtering criteria."""
    
    def __init__(self, data_manager: Optional[HistoricalDataManager] = None):
        """
        Initialize universe manager.
        
        Args:
            data_manager: Historical data manager for data access
        """
        self.data_manager = data_manager or HistoricalDataManager()
        self.storage = self.data_manager.storage
        self.definition = UniverseDefinition()
        
        # Universe state
        self.base_universe: Set[str] = set()
        self.filtered_universe: Set[str] = set()
        self.universe_metrics: Dict[str, Dict[str, Any]] = {}
        self.last_update: Optional[datetime] = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def load_base_universe(self, symbols: Optional[List[str]] = None) -> int:
        """
        Load base universe from predefined list or discovery.
        
        Args:
            symbols: Optional list of symbols to use as base universe
            
        Returns:
            Number of symbols loaded
        """
        if symbols:
            # Use provided symbols
            self.base_universe = set(symbols)
            self.logger.info(f"Loaded {len(symbols)} symbols from provided list")
        else:
            # Use symbols from available data
            available_symbols = self.data_manager.get_available_symbols()
            if available_symbols:
                self.base_universe = set(available_symbols)
                self.logger.info(f"Loaded {len(available_symbols)} symbols from available data")
            else:
                # Fallback to predefined universe
                default_symbols = self._get_default_universe()
                self.base_universe = set(default_symbols)
                self.logger.info(f"Loaded {len(default_symbols)} symbols from default universe")
        
        return len(self.base_universe)
    
    def _get_default_universe(self) -> List[str]:
        """Get default tradeable universe (major stocks)."""
        # S&P 500 core components and major tech stocks
        return [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'ADBE',
            'CRM', 'ORCL', 'INTC', 'AMD', 'PYPL', 'UBER', 'SHOP', 'SQ', 'TWLO', 'ZOOM',
            
            # Finance
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK.B', 'AXP', 'USB', 'PNC',
            'COF', 'SCHW', 'BLK', 'SPGI', 'ICE', 'CME', 'MCO', 'AON', 'MMC', 'AJG',
            
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'ABT', 'TMO', 'DHR', 'BMY', 'ABBV', 'MRK', 'LLY',
            'GILD', 'AMGN', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'MRNA', 'BNTX', 'JNJ', 'CVS',
            
            # Consumer
            'AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'WMT', 'COST', 'LOW', 'TJX',
            'DIS', 'CMCSA', 'VZ', 'T', 'NFLX', 'ROKU', 'SPOT', 'PINS', 'SNAP', 'TWTR',
            
            # Industrial
            'BA', 'CAT', 'MMM', 'HON', 'UPS', 'FDX', 'GE', 'LMT', 'RTX', 'NOC',
            'DE', 'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'DOV', 'XYL', 'FTV', 'CARR',
            
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX', 'OXY', 'BKR',
            
            # Materials
            'LIN', 'APD', 'SHW', 'FCX', 'NEM', 'ECL', 'DD', 'DOW', 'PPG', 'IFF'
        ]
    
    def apply_filters(self, symbols: Optional[Set[str]] = None) -> int:
        """
        Apply universe filtering criteria.
        
        Args:
            symbols: Optional set of symbols to filter (uses base_universe if None)
            
        Returns:
            Number of symbols remaining after filtering
        """
        symbols_to_filter = symbols or self.base_universe.copy()
        self.filtered_universe.clear()
        self.universe_metrics.clear()
        
        self.logger.info(f"Applying filters to {len(symbols_to_filter)} symbols")
        
        for symbol in symbols_to_filter:
            if self._passes_filters(symbol):
                self.filtered_universe.add(symbol)
        
        self.last_update = datetime.now()
        
        self.logger.info(f"Filtering complete: {len(self.filtered_universe)}/{len(symbols_to_filter)} symbols passed")
        
        return len(self.filtered_universe)
    
    def _passes_filters(self, symbol: str) -> bool:
        """
        Check if a symbol passes all universe filters.
        
        Args:
            symbol: Stock symbol to check
            
        Returns:
            True if symbol passes all filters
        """
        # Skip excluded symbols
        if symbol in self.definition.excluded_symbols:
            return False
        
        # Get historical data for analysis
        data = self.data_manager.get_symbol_data(symbol)
        if data is None or data.empty:
            self.logger.debug(f"{symbol}: No data available")
            return False
        
        # Initialize metrics for this symbol
        metrics = {}
        
        # Check data quality and quantity
        if not self._check_data_quality(symbol, data, metrics):
            return False
        
        # Check price filters
        if not self._check_price_filters(symbol, data, metrics):
            return False
        
        # Check volume and liquidity filters
        if not self._check_liquidity_filters(symbol, data, metrics):
            return False
        
        # Store metrics for successful symbols
        self.universe_metrics[symbol] = metrics
        
        return True
    
    def _check_data_quality(self, symbol: str, data: pd.DataFrame, metrics: Dict) -> bool:
        """Check data quality requirements."""
        # Check minimum data days
        data_days = len(data)
        metrics['data_days'] = data_days
        
        if data_days < self.definition.min_data_days:
            self.logger.debug(f"{symbol}: Insufficient data days ({data_days} < {self.definition.min_data_days})")
            return False
        
        # Check for data gaps (missing days)
        if len(data) > 1:
            date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
            weekdays = date_range[date_range.weekday < 5]  # Only weekdays
            missing_days = len(weekdays) - len(data)
            metrics['missing_days'] = missing_days
            
            if missing_days > self.definition.max_data_gaps:
                self.logger.debug(f"{symbol}: Too many data gaps ({missing_days} > {self.definition.max_data_gaps})")
                return False
        
        # Check data completeness (no NaN values in key columns)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col in data.columns:
                null_count = data[col].isnull().sum()
                if null_count > 0:
                    null_pct = null_count / len(data)
                    if null_pct > (1 - self.definition.min_data_quality):
                        self.logger.debug(f"{symbol}: Too many null values in {col} ({null_pct:.1%})")
                        return False
        
        return True
    
    def _check_price_filters(self, symbol: str, data: pd.DataFrame, metrics: Dict) -> bool:
        """Check price-based filters."""
        recent_data = data.tail(20)  # Last 20 days
        avg_price = recent_data['close'].mean()
        metrics['avg_price'] = avg_price
        
        # Price range check
        if avg_price < self.definition.min_price or avg_price > self.definition.max_price:
            self.logger.debug(f"{symbol}: Price outside range (${avg_price:.2f})")
            return False
        
        return True
    
    def _check_liquidity_filters(self, symbol: str, data: pd.DataFrame, metrics: Dict) -> bool:
        """Check volume and liquidity filters."""
        recent_data = data.tail(20)  # Last 20 days
        
        # Average daily volume
        avg_volume = recent_data['volume'].mean()
        metrics['avg_volume'] = avg_volume
        
        if avg_volume < self.definition.min_volume:
            self.logger.debug(f"{symbol}: Low volume ({avg_volume:,.0f} < {self.definition.min_volume:,.0f})")
            return False
        
        # Average daily liquidity (price * volume)
        avg_liquidity = (recent_data['close'] * recent_data['volume']).mean()
        metrics['avg_liquidity'] = avg_liquidity
        
        if avg_liquidity < self.definition.min_liquidity:
            self.logger.debug(f"{symbol}: Low liquidity (${avg_liquidity:,.0f} < ${self.definition.min_liquidity:,.0f})")
            return False
        
        return True
    
    def get_universe(self, apply_filters: bool = True) -> Set[str]:
        """
        Get the current tradeable universe.
        
        Args:
            apply_filters: Whether to return filtered or base universe
            
        Returns:
            Set of stock symbols in universe
        """
        if apply_filters:
            if not self.filtered_universe:
                self.apply_filters()
            return self.filtered_universe.copy()
        else:
            return self.base_universe.copy()
    
    def get_universe_metrics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get universe metrics.
        
        Args:
            symbol: Optional specific symbol to get metrics for
            
        Returns:
            Dictionary of metrics
        """
        if symbol:
            return self.universe_metrics.get(symbol, {})
        else:
            return {
                'total_base_symbols': len(self.base_universe),
                'total_filtered_symbols': len(self.filtered_universe),
                'filter_pass_rate': len(self.filtered_universe) / len(self.base_universe) if self.base_universe else 0,
                'last_update': self.last_update,
                'definition': self.definition.to_dict(),
                'symbol_metrics': self.universe_metrics
            }
    
    def refresh_universe(self, force_reload: bool = False) -> int:
        """
        Refresh the universe by reapplying filters.
        
        Args:
            force_reload: Whether to reload base universe from data
            
        Returns:
            Number of symbols in filtered universe
        """
        if force_reload:
            self.load_base_universe()
        
        return self.apply_filters()
    
    def get_universe_summary(self) -> Dict[str, Any]:
        """Get a summary of the current universe."""
        if not self.filtered_universe:
            self.apply_filters()
        
        # Calculate summary statistics
        if self.universe_metrics:
            prices = [metrics.get('avg_price', 0) for metrics in self.universe_metrics.values()]
            volumes = [metrics.get('avg_volume', 0) for metrics in self.universe_metrics.values()]
            liquidities = [metrics.get('avg_liquidity', 0) for metrics in self.universe_metrics.values()]
            
            price_stats = {
                'min': min(prices) if prices else 0,
                'max': max(prices) if prices else 0,
                'mean': sum(prices) / len(prices) if prices else 0
            }
            
            volume_stats = {
                'min': min(volumes) if volumes else 0,
                'max': max(volumes) if volumes else 0,
                'mean': sum(volumes) / len(volumes) if volumes else 0
            }
            
            liquidity_stats = {
                'min': min(liquidities) if liquidities else 0,
                'max': max(liquidities) if liquidities else 0,
                'mean': sum(liquidities) / len(liquidities) if liquidities else 0
            }
        else:
            price_stats = volume_stats = liquidity_stats = {'min': 0, 'max': 0, 'mean': 0}
        
        return {
            'total_symbols': len(self.filtered_universe),
            'symbols': sorted(list(self.filtered_universe)),
            'price_stats': price_stats,
            'volume_stats': volume_stats,
            'liquidity_stats': liquidity_stats,
            'last_update': self.last_update,
            'filter_criteria': self.definition.to_dict()
        }
