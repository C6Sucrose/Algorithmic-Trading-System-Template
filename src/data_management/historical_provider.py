"""
Historical Data Provider

Downloads and manages historical market data for backtesting.
Supports multiple data sources with fallback options.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
import pandas as pd
import requests
import time
import logging

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config_manager import config
from src.data_management.data_core import DataStorage, DataValidator, MarketData


class YahooFinanceProvider:
    """Yahoo Finance data provider (free tier)."""
    
    def __init__(self):
        self.base_url = "https://query1.finance.yahoo.com/v7/finance/download"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.rate_limit_delay = 1.0  # seconds between requests
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Implement rate limiting to avoid being blocked."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def download_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Download historical data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval ('1d', '1wk', '1mo')
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        self._rate_limit()
        
        try:
            # Convert dates to Unix timestamps
            start_ts = int(start_date.timestamp())
            end_ts = int(end_date.timestamp())
            
            # Build URL
            url = f"{self.base_url}/{symbol}"
            params = {
                'period1': start_ts,
                'period2': end_ts,
                'interval': interval,
                'events': 'history',
                'includeAdjustedClose': 'true'
            }
            
            # Make request
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            # Parse CSV response
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            # Clean and standardize column names
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Rename columns to match our standard format
            column_mapping = {
                'adj_close': 'adj_close',  # Keep adjusted close
                'volume': 'volume'
            }
            df = df.rename(columns=column_mapping)
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                raise ValueError(f"Missing required columns: {missing}")
            
            # Sort by date
            df = df.sort_index()
            
            logging.info(f"Downloaded {len(df)} records for {symbol} from {start_date.date()} to {end_date.date()}")
            
            return df
            
        except Exception as e:
            logging.error(f"Failed to download data for {symbol}: {e}")
            return None


class AlphaVantageProvider:
    """Alpha Vantage data provider (requires API key)."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.get('ALPHA_VANTAGE_API_KEY')
        self.base_url = "https://www.alphavantage.co/query"
        self.session = requests.Session()
        self.rate_limit_delay = 12.0  # 5 requests per minute for free tier
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Implement rate limiting for Alpha Vantage API."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def download_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        output_size: str = "full"
    ) -> Optional[pd.DataFrame]:
        """
        Download historical data from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (will get all available data and filter)
            end_date: End date
            output_size: 'compact' (100 points) or 'full' (20+ years)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not self.api_key:
            logging.warning("Alpha Vantage API key not configured")
            return None
        
        self._rate_limit()
        
        try:
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'outputsize': output_size,
                'apikey': self.api_key
            }
            
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for error messages
            if 'Error Message' in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            
            if 'Note' in data:
                raise ValueError(f"API Limit: {data['Note']}")
            
            # Extract time series data
            time_series_key = 'Time Series (Daily)'
            if time_series_key not in data:
                raise ValueError("Time series data not found in response")
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns
            df.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume', 'dividend', 'split']
            
            # Convert to numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            logging.info(f"Downloaded {len(df)} records for {symbol} from Alpha Vantage")
            
            return df
            
        except Exception as e:
            logging.error(f"Failed to download data for {symbol} from Alpha Vantage: {e}")
            return None


class HistoricalDataManager:
    """Manages historical data download, storage, and retrieval."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize historical data manager.
        
        Args:
            storage_path: Path for data storage (uses config default if None)
        """
        self.storage_path = storage_path or Path(config.get('DATA_STORAGE_PATH', './data'))
        self.storage = DataStorage(self.storage_path)
        self.validator = DataValidator()
        
        # Initialize data providers
        self.yahoo_provider = YahooFinanceProvider()
        self.alpha_vantage_provider = AlphaVantageProvider()
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config.get('LOG_LEVEL', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def download_symbol_data(
        self, 
        symbol: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        provider: str = 'yahoo',
        force_update: bool = False
    ) -> bool:
        """
        Download historical data for a symbol.
        
        Args:
            symbol: Stock symbol to download
            start_date: Start date (defaults to 1 year ago)
            end_date: End date (defaults to today)
            provider: Data provider ('yahoo' or 'alphavantage')
            force_update: Force re-download even if data exists
            
        Returns:
            True if successful, False otherwise
        """
        # Set default dates
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=config.get('HISTORICAL_DATA_DAYS', 252))
        
        # Check if data already exists
        if not force_update:
            existing_data = self.storage.load_historical_data(symbol)
            if existing_data is not None and not existing_data.empty:
                # Check if we have recent enough data
                latest_date = existing_data.index.max()
                if latest_date >= end_date - timedelta(days=7):  # Within a week
                    self.logger.info(f"Recent data exists for {symbol}, skipping download")
                    return True
        
        # Download data using specified provider
        if provider.lower() == 'yahoo':
            data = self.yahoo_provider.download_historical_data(symbol, start_date, end_date)
        elif provider.lower() == 'alphavantage':
            data = self.alpha_vantage_provider.download_historical_data(symbol, start_date, end_date)
        else:
            self.logger.error(f"Unknown provider: {provider}")
            return False
        
        if data is None or data.empty:
            self.logger.error(f"No data downloaded for {symbol}")
            return False
        
        # Validate data quality
        quality_report = self.validator.validate_market_data(data, symbol)
        self.logger.info(f"Data quality for {symbol}: {quality_report.quality_score:.2%}")
        
        if not quality_report.is_valid():
            self.logger.warning(f"Data quality issues for {symbol}: {quality_report.issues}")
            # Clean the data
            data = self.validator.clean_data(data)
            if data.empty:
                self.logger.error(f"Data cleaning resulted in empty dataset for {symbol}")
                return False
        
        # Save data
        success = self.storage.save_data(symbol, data, data_type='historical')
        if success:
            self.logger.info(f"Successfully saved {len(data)} records for {symbol}")
        else:
            self.logger.error(f"Failed to save data for {symbol}")
        
        return success
    
    def download_multiple_symbols(
        self, 
        symbols: List[str], 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        provider: str = 'yahoo',
        max_workers: int = 3
    ) -> Dict[str, bool]:
        """
        Download historical data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date  
            provider: Data provider
            max_workers: Maximum concurrent downloads (keep low to avoid rate limits)
            
        Returns:
            Dictionary mapping symbol to success status
        """
        results = {}
        
        # Sequential download to respect rate limits
        for symbol in symbols:
            self.logger.info(f"Downloading data for {symbol} ({symbols.index(symbol)+1}/{len(symbols)})")
            results[symbol] = self.download_symbol_data(
                symbol, start_date, end_date, provider
            )
            
            # Small delay between symbols
            time.sleep(0.5)
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        self.logger.info(f"Download complete: {successful}/{len(symbols)} symbols successful")
        
        return results
    
    def get_symbol_data(
        self, 
        symbol: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Filter start date
            end_date: Filter end date
            
        Returns:
            DataFrame with historical data or None
        """
        data = self.storage.load_historical_data(symbol)
        
        if data is None:
            return None
        
        # Apply date filters if specified
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        return data if not data.empty else None
    
    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with stored data."""
        return self.storage.get_available_symbols()
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of stored data."""
        symbols = self.get_available_symbols()
        summary = {
            'total_symbols': len(symbols),
            'symbols': [],
            'total_size_mb': 0
        }
        
        for symbol in symbols:
            info = self.storage.get_data_info(symbol)
            if info:
                summary['symbols'].append(info)
                summary['total_size_mb'] += info['file_size_mb']
        
        return summary
