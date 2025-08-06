"""
Data Management Module - Core Components

This module handles historical and real-time market data acquisition, storage, and validation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
import pandas as pd
import numpy as np


@dataclass
class MarketData:
    """Market data structure for consistent data handling."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'bid': self.bid,
            'ask': self.ask,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size
        }
    
    def to_pandas_row(self) -> pd.Series:
        """Convert to pandas Series for DataFrame integration."""
        return pd.Series(self.to_dict())


@dataclass 
class DataQualityReport:
    """Data quality validation report."""
    symbol: str
    start_date: datetime
    end_date: datetime
    total_records: int
    missing_records: int
    outliers_detected: int
    quality_score: float
    issues: List[str]
    
    def is_valid(self, min_quality_score: float = 0.95) -> bool:
        """Check if data quality meets minimum standards."""
        return self.quality_score >= min_quality_score


class DataValidator:
    """Data quality validation and cleaning."""
    
    def __init__(self):
        self.outlier_threshold = 3.0  # Standard deviations for outlier detection
        
    def validate_market_data(self, data: pd.DataFrame, symbol: str) -> DataQualityReport:
        """
        Validate market data quality.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol being validated
            
        Returns:
            DataQualityReport with validation results
        """
        if data.empty:
            return DataQualityReport(
                symbol=symbol,
                start_date=datetime.now(),
                end_date=datetime.now(),
                total_records=0,
                missing_records=0,
                outliers_detected=0,
                quality_score=0.0,
                issues=["No data provided"]
            )
        
        issues = []
        total_records = len(data)
        
        # Check for missing values
        missing_records = data.isnull().sum().sum()
        if missing_records > 0:
            issues.append(f"Missing values detected: {missing_records}")
        
        # Check for negative prices or volumes
        price_cols = ['open', 'high', 'low', 'close']
        negative_prices = (data[price_cols] < 0).sum().sum()
        if negative_prices > 0:
            issues.append(f"Negative prices detected: {negative_prices}")
        
        negative_volume = (data['volume'] < 0).sum()
        if negative_volume > 0:
            issues.append(f"Negative volume detected: {negative_volume}")
        
        # Check for zero volume (suspicious)
        zero_volume = (data['volume'] == 0).sum()
        if zero_volume > 0:
            issues.append(f"Zero volume detected: {zero_volume}")
        
        # Check OHLC relationships
        invalid_ohlc = self._validate_ohlc_relationships(data)
        if invalid_ohlc > 0:
            issues.append(f"Invalid OHLC relationships: {invalid_ohlc}")
        
        # Detect outliers
        outliers = self._detect_outliers(data)
        if outliers > 0:
            issues.append(f"Price outliers detected: {outliers}")
        
        # Check for gaps in timestamp sequence
        if 'timestamp' in data.columns:
            gaps = self._check_timestamp_gaps(data)
            if gaps > 0:
                issues.append(f"Timestamp gaps detected: {gaps}")
        
        # Calculate quality score
        total_issues = missing_records + negative_prices + negative_volume + invalid_ohlc + outliers
        quality_score = max(0.0, 1.0 - (total_issues / total_records))
        
        return DataQualityReport(
            symbol=symbol,
            start_date=data.index.min() if hasattr(data.index, 'min') else datetime.now(),
            end_date=data.index.max() if hasattr(data.index, 'max') else datetime.now(),
            total_records=total_records,
            missing_records=missing_records,
            outliers_detected=outliers,
            quality_score=quality_score,
            issues=issues
        )
    
    def _validate_ohlc_relationships(self, data: pd.DataFrame) -> int:
        """Validate that High >= Open,Close,Low and Low <= Open,Close,High."""
        invalid_high = ((data['high'] < data['open']) | 
                       (data['high'] < data['close']) | 
                       (data['high'] < data['low'])).sum()
        
        invalid_low = ((data['low'] > data['open']) | 
                      (data['low'] > data['close']) | 
                      (data['low'] > data['high'])).sum()
        
        return invalid_high + invalid_low
    
    def _detect_outliers(self, data: pd.DataFrame) -> int:
        """Detect price outliers using z-score method."""
        price_cols = ['open', 'high', 'low', 'close']
        outliers = 0
        
        for col in price_cols:
            if col in data.columns:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outliers += (z_scores > self.outlier_threshold).sum()
        
        return outliers
    
    def _check_timestamp_gaps(self, data: pd.DataFrame) -> int:
        """Check for unusual gaps in timestamp sequence."""
        if 'timestamp' not in data.columns or len(data) < 2:
            return 0
        
        # Sort by timestamp
        sorted_data = data.sort_values('timestamp')
        time_diffs = sorted_data['timestamp'].diff().dt.total_seconds()
        
        # Detect gaps larger than expected (e.g., more than 1 week for daily data)
        expected_gap = 24 * 3600  # 1 day in seconds
        large_gaps = (time_diffs > expected_gap * 7).sum()  # Gaps > 1 week
        
        return large_gaps
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess market data.
        
        Args:
            data: Raw market data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        cleaned = data.copy()
        
        # Remove rows with negative prices or volume
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in cleaned.columns:
                cleaned = cleaned[cleaned[col] >= 0]
        
        if 'volume' in cleaned.columns:
            cleaned = cleaned[cleaned['volume'] >= 0]
        
        # Remove rows with invalid OHLC relationships
        cleaned = cleaned[
            (cleaned['high'] >= cleaned['open']) &
            (cleaned['high'] >= cleaned['close']) &
            (cleaned['high'] >= cleaned['low']) &
            (cleaned['low'] <= cleaned['open']) &
            (cleaned['low'] <= cleaned['close']) &
            (cleaned['low'] <= cleaned['high'])
        ]
        
        # Forward fill missing values (updated pandas syntax)
        cleaned = cleaned.ffill()
        
        # Drop remaining NaN rows
        cleaned = cleaned.dropna()
        
        return cleaned


class DataStorage:
    """Data storage and retrieval using HDF5 and CSV formats."""
    
    def __init__(self, storage_path: Path):
        """
        Initialize data storage.
        
        Args:
            storage_path: Base path for data storage
        """
        self.storage_path = Path(storage_path)
        self.historical_path = self.storage_path / "historical"
        self.real_time_path = self.storage_path / "real_time"
        self.processed_path = self.storage_path / "processed"
        
        # Create directories if they don't exist
        for path in [self.historical_path, self.real_time_path, self.processed_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def save_data(self, symbol: str, data: pd.DataFrame, data_type: str = 'historical', file_format: str = 'hdf5') -> bool:
        """
        Save historical data for a symbol.
        
        Args:
            symbol: Stock symbol
            data: Historical OHLCV data
            data_type: Type of data ('historical', 'processed', etc.)
            file_format: Storage format ('hdf5' or 'csv')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if file_format.lower() == 'hdf5':
                file_path = self.historical_path / f"{symbol}.h5"
                data.to_hdf(file_path, key='data', mode='w', format='table')
            elif file_format.lower() == 'csv':
                file_path = self.historical_path / f"{symbol}.csv"
                data.to_csv(file_path, index=True)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            
            return True
        except Exception as e:
            print(f"Error saving historical data for {symbol}: {e}")
            return False
    
    def load_historical_data(self, symbol: str, format: str = 'hdf5') -> Optional[pd.DataFrame]:
        """
        Load historical data for a symbol.
        
        Args:
            symbol: Stock symbol
            format: Storage format ('hdf5' or 'csv')
            
        Returns:
            DataFrame with historical data or None if not found
        """
        try:
            if format.lower() == 'hdf5':
                file_path = self.historical_path / f"{symbol}.h5"
                if file_path.exists():
                    data = pd.read_hdf(file_path, key='data')
                    return data if isinstance(data, pd.DataFrame) else None
            elif format.lower() == 'csv':
                file_path = self.historical_path / f"{symbol}.csv"
                if file_path.exists():
                    return pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            return None
        except Exception as e:
            print(f"Error loading historical data for {symbol}: {e}")
            return None
    
    def append_real_time_data(self, symbol: str, data: MarketData) -> bool:
        """
        Append real-time data point to storage.
        
        Args:
            symbol: Stock symbol
            data: Real-time market data point
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self.real_time_path / f"{symbol}_realtime.csv"
            
            # Convert to DataFrame row
            df_row = pd.DataFrame([data.to_dict()])
            df_row.set_index('timestamp', inplace=True)
            
            # Append to file
            if file_path.exists():
                df_row.to_csv(file_path, mode='a', header=False)
            else:
                df_row.to_csv(file_path, mode='w', header=True)
            
            return True
        except Exception as e:
            print(f"Error appending real-time data for {symbol}: {e}")
            return False
    
    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with stored historical data."""
        symbols = []
        
        # Check HDF5 files
        for file_path in self.historical_path.glob("*.h5"):
            symbols.append(file_path.stem)
        
        # Check CSV files
        for file_path in self.historical_path.glob("*.csv"):
            symbols.append(file_path.stem)
        
        return sorted(list(set(symbols)))
    
    def get_data_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get information about stored data for a symbol."""
        data = self.load_historical_data(symbol)
        if data is None:
            return None
        
        return {
            'symbol': symbol,
            'start_date': data.index.min(),
            'end_date': data.index.max(),
            'records': len(data),
            'columns': list(data.columns),
            'file_size_mb': self._get_file_size(symbol)
        }
    
    def _get_file_size(self, symbol: str) -> float:
        """Get file size in MB for a symbol's data."""
        h5_path = self.historical_path / f"{symbol}.h5"
        csv_path = self.historical_path / f"{symbol}.csv"
        
        size = 0
        if h5_path.exists():
            size += h5_path.stat().st_size
        if csv_path.exists():
            size += csv_path.stat().st_size
        
        return size / (1024 * 1024)  # Convert to MB
