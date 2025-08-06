"""
Real-Time Data Provider

Handles real-time market data streaming and processing.
Supports IBKR API and other real-time data sources.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
import pandas as pd
import numpy as np
import threading
import queue
import time
import logging
from abc import ABC, abstractmethod

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config_manager import config
from src.data_management.data_core import DataStorage, MarketData


class DataStreamHandler(ABC):
    """Abstract base class for real-time data stream handlers."""
    
    def __init__(self):
        """Initialize the data stream handler."""
        self.subscribed_symbols = set()
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to data source."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from data source."""
        pass
    
    @abstractmethod
    def subscribe_symbol(self, symbol: str) -> bool:
        """Subscribe to real-time data for a symbol."""
        pass
    
    @abstractmethod
    def unsubscribe_symbol(self, symbol: str) -> bool:
        """Unsubscribe from real-time data for a symbol."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to data source."""
        pass


class MockDataStreamHandler(DataStreamHandler):
    """Mock data stream handler for testing and development."""
    
    def __init__(self):
        super().__init__()
        self.connected = False
        self.data_queue = queue.Queue()
        self.streaming_thread = None
        self.stop_streaming = threading.Event()
        
    def connect(self) -> bool:
        """Simulate connection to data source."""
        self.connected = True
        logging.info("Mock data stream connected")
        return True
    
    def disconnect(self) -> None:
        """Simulate disconnection from data source."""
        self.stop_streaming.set()
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=5)
        self.connected = False
        self.subscribed_symbols.clear()
        logging.info("Mock data stream disconnected")
    
    def subscribe_symbol(self, symbol: str) -> bool:
        """Subscribe to mock data for a symbol."""
        if not self.connected:
            return False
        
        self.subscribed_symbols.add(symbol)
        
        # Start streaming thread if not already running
        if not self.streaming_thread or not self.streaming_thread.is_alive():
            self.stop_streaming.clear()
            self.streaming_thread = threading.Thread(
                target=self._generate_mock_data,
                daemon=True
            )
            self.streaming_thread.start()
        
        logging.info(f"Subscribed to mock data for {symbol}")
        return True
    
    def unsubscribe_symbol(self, symbol: str) -> bool:
        """Unsubscribe from mock data for a symbol."""
        self.subscribed_symbols.discard(symbol)
        logging.info(f"Unsubscribed from mock data for {symbol}")
        return True
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self.connected
    
    def get_data(self, timeout: float = 1.0) -> Optional[MarketData]:
        """Get next data point from queue."""
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _generate_mock_data(self):
        """Generate mock real-time data."""
        base_prices = {symbol: 100.0 for symbol in self.subscribed_symbols}
        
        while not self.stop_streaming.is_set() and self.subscribed_symbols:
            for symbol in list(self.subscribed_symbols):  # Copy to avoid modification during iteration
                # Generate realistic price movement
                last_price = base_prices.get(symbol, 100.0)
                price_change = (np.random.randn() * 0.01 + 0.0001) * last_price  # Small random walk
                new_price = max(0.01, last_price + price_change)  # Ensure positive price
                base_prices[symbol] = new_price
                
                # Create market data
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    open=new_price,
                    high=new_price * (1 + abs(np.random.randn() * 0.005)),
                    low=new_price * (1 - abs(np.random.randn() * 0.005)),
                    close=new_price,
                    volume=int(np.random.randint(1000, 10000)),
                    bid=new_price * 0.999,
                    ask=new_price * 1.001,
                    bid_size=int(np.random.randint(100, 1000)),
                    ask_size=int(np.random.randint(100, 1000))
                )
                
                self.data_queue.put(market_data)
            
            # Wait before next update
            time.sleep(1.0)  # 1 second intervals


class IBKRDataStreamHandler(DataStreamHandler):
    """IBKR API data stream handler (placeholder for future implementation)."""
    
    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1):
        super().__init__()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connected = False
        
        # Note: Actual IBKR implementation would require ib_insync package
        logging.warning("IBKR data stream handler not fully implemented yet")
    
    def connect(self) -> bool:
        """Connect to IBKR TWS/Gateway."""
        try:
            # Placeholder for IBKR connection
            # In full implementation:
            # from ib_insync import IB
            # self.ib = IB()
            # self.ib.connect(self.host, self.port, clientId=self.client_id)
            
            self.connected = True
            logging.info(f"IBKR connection simulated (would connect to {self.host}:{self.port})")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to IBKR: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        # Placeholder for IBKR disconnection
        self.connected = False
        self.subscribed_symbols.clear()
        logging.info("IBKR disconnected")
    
    def subscribe_symbol(self, symbol: str) -> bool:
        """Subscribe to real-time data for a symbol."""
        if not self.connected:
            return False
        
        # Placeholder for IBKR subscription
        self.subscribed_symbols.add(symbol)
        logging.info(f"IBKR subscription simulated for {symbol}")
        return True
    
    def unsubscribe_symbol(self, symbol: str) -> bool:
        """Unsubscribe from real-time data."""
        self.subscribed_symbols.discard(symbol)
        logging.info(f"IBKR unsubscription simulated for {symbol}")
        return True
    
    def is_connected(self) -> bool:
        """Check connection status."""
        return self.connected


class RealTimeDataManager:
    """Manages real-time market data streams and processing."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize real-time data manager.
        
        Args:
            storage_path: Path for data storage
        """
        self.storage_path = storage_path or Path(config.get('DATA_STORAGE_PATH', './data'))
        self.storage = DataStorage(self.storage_path)
        
        # Data stream handler
        self.stream_handler: Optional[DataStreamHandler] = None
        
        # Data processing
        self.data_callbacks: List[Callable[[MarketData], None]] = []
        self.processing_thread: Optional[threading.Thread] = None
        self.stop_processing = threading.Event()
        
        # Buffering for batch processing
        self.data_buffer: Dict[str, List[MarketData]] = {}
        self.buffer_size = 100  # Number of data points to buffer before saving
        self.buffer_timeout = 60  # Seconds before forcing buffer flush
        self.last_flush_time = datetime.now()
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config.get('LOG_LEVEL', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_stream(self, stream_type: str = 'mock') -> bool:
        """
        Initialize data stream handler.
        
        Args:
            stream_type: Type of stream ('mock' or 'ibkr')
            
        Returns:
            True if successful, False otherwise
        """
        if stream_type.lower() == 'mock':
            self.stream_handler = MockDataStreamHandler()
        elif stream_type.lower() == 'ibkr':
            ibkr_config = config.get_ibkr_config()
            self.stream_handler = IBKRDataStreamHandler(
                host=ibkr_config['host'],
                port=ibkr_config['port'],
                client_id=ibkr_config['clientId']
            )
        else:
            self.logger.error(f"Unknown stream type: {stream_type}")
            return False
        
        # Connect to stream
        if self.stream_handler.connect():
            self.logger.info(f"Real-time data stream initialized: {stream_type}")
            return True
        else:
            self.logger.error(f"Failed to initialize {stream_type} stream")
            return False
    
    def start_streaming(self, symbols: List[str]) -> bool:
        """
        Start real-time data streaming for symbols.
        
        Args:
            symbols: List of symbols to stream
            
        Returns:
            True if successful, False otherwise
        """
        if not self.stream_handler or not self.stream_handler.is_connected():
            self.logger.error("Stream handler not initialized or connected")
            return False
        
        # Subscribe to symbols
        for symbol in symbols:
            if not self.stream_handler.subscribe_symbol(symbol):
                self.logger.error(f"Failed to subscribe to {symbol}")
                return False
        
        # Start processing thread
        self.stop_processing.clear()
        self.processing_thread = threading.Thread(
            target=self._process_data_stream,
            daemon=True
        )
        self.processing_thread.start()
        
        self.logger.info(f"Started streaming for {len(symbols)} symbols")
        return True
    
    def stop_streaming(self) -> None:
        """Stop real-time data streaming."""
        self.stop_processing.set()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
        
        if self.stream_handler:
            self.stream_handler.disconnect()
        
        # Flush any remaining buffered data
        self._flush_all_buffers()
        
        self.logger.info("Real-time streaming stopped")
    
    def add_data_callback(self, callback: Callable[[MarketData], None]) -> None:
        """
        Add callback function for real-time data processing.
        
        Args:
            callback: Function to call with each MarketData point
        """
        self.data_callbacks.append(callback)
    
    def remove_data_callback(self, callback: Callable[[MarketData], None]) -> None:
        """Remove a data callback."""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
    
    def _process_data_stream(self) -> None:
        """Process incoming real-time data."""
        while not self.stop_processing.is_set():
            if isinstance(self.stream_handler, MockDataStreamHandler):
                # Get data from mock handler
                data = self.stream_handler.get_data(timeout=1.0)
                if data:
                    self._handle_market_data(data)
            else:
                # For other handlers, implement specific data retrieval
                time.sleep(0.1)
            
            # Check if buffer needs flushing
            current_time = datetime.now()
            if (current_time - self.last_flush_time).total_seconds() > self.buffer_timeout:
                self._flush_all_buffers()
    
    def _handle_market_data(self, data: MarketData) -> None:
        """Handle incoming market data point."""
        # Call registered callbacks
        for callback in self.data_callbacks:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Error in data callback: {e}")
        
        # Add to buffer for storage
        symbol = data.symbol
        if symbol not in self.data_buffer:
            self.data_buffer[symbol] = []
        
        self.data_buffer[symbol].append(data)
        
        # Flush buffer if it's full
        if len(self.data_buffer[symbol]) >= self.buffer_size:
            self._flush_buffer(symbol)
    
    def _flush_buffer(self, symbol: str) -> None:
        """Flush buffer for a specific symbol."""
        if symbol not in self.data_buffer or not self.data_buffer[symbol]:
            return
        
        # Save buffered data
        for data_point in self.data_buffer[symbol]:
            self.storage.append_real_time_data(symbol, data_point)
        
        # Clear buffer
        count = len(self.data_buffer[symbol])
        self.data_buffer[symbol] = []
        
        self.logger.debug(f"Flushed {count} data points for {symbol}")
    
    def _flush_all_buffers(self) -> None:
        """Flush all symbol buffers."""
        for symbol in list(self.data_buffer.keys()):
            self._flush_buffer(symbol)
        
        self.last_flush_time = datetime.now()
        self.logger.debug("Flushed all data buffers")
    
    def get_streaming_status(self) -> Dict[str, Any]:
        """Get current streaming status."""
        return {
            'connected': self.stream_handler.is_connected() if self.stream_handler else False,
            'processing': self.processing_thread.is_alive() if self.processing_thread else False,
            'subscribed_symbols': list(self.stream_handler.subscribed_symbols) if self.stream_handler else [],
            'buffer_sizes': {symbol: len(data_list) for symbol, data_list in self.data_buffer.items()},
            'callbacks_registered': len(self.data_callbacks)
        }
