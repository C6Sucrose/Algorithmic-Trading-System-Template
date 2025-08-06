"""
Tests for Real-Time Data Provider

Test real-time data streaming, storage, and processing functionality.
"""

import unittest
import threading
import time
from pathlib import Path
from datetime import datetime
import tempfile
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_management.realtime_provider import (
    RealTimeDataManager,
    MockDataStreamHandler,
    IBKRDataStreamHandler
)
from src.data_management.data_core import MarketData


class TestMockDataStreamHandler(unittest.TestCase):
    """Test cases for MockDataStreamHandler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = MockDataStreamHandler()
    
    def tearDown(self):
        """Clean up after tests."""
        if self.handler.is_connected():
            self.handler.disconnect()
    
    def test_connection(self):
        """Test connection and disconnection."""
        # Initially not connected
        self.assertFalse(self.handler.is_connected())
        
        # Connect
        result = self.handler.connect()
        self.assertTrue(result)
        self.assertTrue(self.handler.is_connected())
        
        # Disconnect
        self.handler.disconnect()
        self.assertFalse(self.handler.is_connected())
    
    def test_symbol_subscription(self):
        """Test symbol subscription and unsubscription."""
        # Connect first
        self.handler.connect()
        
        # Subscribe to symbol
        result = self.handler.subscribe_symbol('AAPL')
        self.assertTrue(result)
        self.assertIn('AAPL', self.handler.subscribed_symbols)
        
        # Subscribe to another symbol
        result = self.handler.subscribe_symbol('GOOGL')
        self.assertTrue(result)
        self.assertIn('GOOGL', self.handler.subscribed_symbols)
        
        # Unsubscribe
        result = self.handler.unsubscribe_symbol('AAPL')
        self.assertTrue(result)
        self.assertNotIn('AAPL', self.handler.subscribed_symbols)
        self.assertIn('GOOGL', self.handler.subscribed_symbols)
    
    def test_subscription_without_connection(self):
        """Test that subscription fails when not connected."""
        result = self.handler.subscribe_symbol('AAPL')
        self.assertFalse(result)
    
    def test_mock_data_generation(self):
        """Test mock data generation."""
        self.handler.connect()
        self.handler.subscribe_symbol('AAPL')
        
        # Wait for some data to be generated
        time.sleep(2)
        
        # Get data from queue
        data_points = []
        while True:
            data = self.handler.get_data(timeout=0.1)
            if data is None:
                break
            data_points.append(data)
        
        # Should have received some data
        self.assertGreater(len(data_points), 0)
        
        # Check data structure
        for data in data_points:
            self.assertIsInstance(data, MarketData)
            self.assertEqual(data.symbol, 'AAPL')
            self.assertGreater(data.close, 0)
            self.assertGreater(data.volume, 0)
            self.assertGreaterEqual(data.high, data.low)


class TestIBKRDataStreamHandler(unittest.TestCase):
    """Test cases for IBKRDataStreamHandler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = IBKRDataStreamHandler()
    
    def tearDown(self):
        """Clean up after tests."""
        if self.handler.is_connected():
            self.handler.disconnect()
    
    def test_initialization(self):
        """Test handler initialization."""
        # Default parameters
        self.assertEqual(self.handler.host, '127.0.0.1')
        self.assertEqual(self.handler.port, 7497)
        self.assertEqual(self.handler.client_id, 1)
        self.assertFalse(self.handler.is_connected())
        
        # Custom parameters
        handler = IBKRDataStreamHandler(host='localhost', port=7496, client_id=2)
        self.assertEqual(handler.host, 'localhost')
        self.assertEqual(handler.port, 7496)
        self.assertEqual(handler.client_id, 2)
    
    def test_simulated_connection(self):
        """Test simulated IBKR connection."""
        # Currently just simulated
        result = self.handler.connect()
        self.assertTrue(result)
        self.assertTrue(self.handler.is_connected())
        
        # Disconnect
        self.handler.disconnect()
        self.assertFalse(self.handler.is_connected())
    
    def test_symbol_subscription(self):
        """Test symbol subscription (simulated)."""
        self.handler.connect()
        
        result = self.handler.subscribe_symbol('AAPL')
        self.assertTrue(result)
        self.assertIn('AAPL', self.handler.subscribed_symbols)
        
        result = self.handler.unsubscribe_symbol('AAPL')
        self.assertTrue(result)
        self.assertNotIn('AAPL', self.handler.subscribed_symbols)


class TestRealTimeDataManager(unittest.TestCase):
    """Test cases for RealTimeDataManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir)
        self.manager = RealTimeDataManager(storage_path=self.storage_path)
    
    def tearDown(self):
        """Clean up after tests."""
        self.manager.stop_streaming()
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_stream_initialization(self):
        """Test stream handler initialization."""
        # Mock stream
        result = self.manager.initialize_stream('mock')
        self.assertTrue(result)
        self.assertIsNotNone(self.manager.stream_handler)
        
        # IBKR stream
        result = self.manager.initialize_stream('ibkr')
        self.assertTrue(result)
        self.assertIsNotNone(self.manager.stream_handler)
        
        # Invalid stream type
        result = self.manager.initialize_stream('invalid')
        self.assertFalse(result)
    
    def test_streaming_workflow(self):
        """Test complete streaming workflow."""
        # Initialize mock stream
        self.manager.initialize_stream('mock')
        
        # Start streaming
        symbols = ['AAPL', 'GOOGL']
        result = self.manager.start_streaming(symbols)
        self.assertTrue(result)
        
        # Check status
        status = self.manager.get_streaming_status()
        self.assertTrue(status['connected'])
        self.assertTrue(status['processing'])
        self.assertEqual(set(status['subscribed_symbols']), set(symbols))
        
        # Let it run for a bit
        time.sleep(2)
        
        # Stop streaming
        self.manager.stop_streaming()
        
        # Check final status
        status = self.manager.get_streaming_status()
        self.assertFalse(status['connected'])
        self.assertFalse(status['processing'])
    
    def test_data_callbacks(self):
        """Test data callback functionality."""
        received_data = []
        
        def data_callback(data: MarketData):
            received_data.append(data)
        
        # Add callback
        self.manager.add_data_callback(data_callback)
        self.assertEqual(len(self.manager.data_callbacks), 1)
        
        # Initialize and start streaming
        self.manager.initialize_stream('mock')
        self.manager.start_streaming(['AAPL'])
        
        # Wait for data
        time.sleep(2)
        
        # Stop streaming
        self.manager.stop_streaming()
        
        # Check that we received data through callback
        self.assertGreater(len(received_data), 0)
        
        # Remove callback
        self.manager.remove_data_callback(data_callback)
        self.assertEqual(len(self.manager.data_callbacks), 0)
    
    def test_streaming_without_initialization(self):
        """Test that streaming fails without stream initialization."""
        result = self.manager.start_streaming(['AAPL'])
        self.assertFalse(result)
    
    def test_buffer_management(self):
        """Test data buffering and flushing."""
        # Set small buffer size for testing
        self.manager.buffer_size = 2
        
        # Initialize mock stream
        self.manager.initialize_stream('mock')
        self.manager.start_streaming(['AAPL'])
        
        # Wait for some data to accumulate
        time.sleep(3)
        
        # Check buffer status
        status = self.manager.get_streaming_status()
        buffer_info = status['buffer_sizes']
        
        # Stop streaming (which flushes buffers)
        self.manager.stop_streaming()
        
        # Verify real-time data files were created
        rt_path = self.storage_path / 'real_time'
        self.assertTrue(rt_path.exists())
        
        # Check for CSV files (from the existing append_real_time_data method)
        csv_files = list(rt_path.glob('*.csv'))
        self.assertGreater(len(csv_files), 0)


class TestDataIntegration(unittest.TestCase):
    """Integration tests for data management components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_data_flow(self):
        """Test complete data flow from streaming to storage."""
        # Create manager
        manager = RealTimeDataManager(storage_path=self.storage_path)
        
        # Track processed data
        processed_data = []
        
        def process_data(data: MarketData):
            processed_data.append(data)
            # Could add additional processing here
        
        # Add processing callback
        manager.add_data_callback(process_data)
        
        # Initialize and start streaming
        manager.initialize_stream('mock')
        manager.start_streaming(['AAPL', 'MSFT'])
        
        # Let it run
        time.sleep(3)
        
        # Stop streaming
        manager.stop_streaming()
        
        # Verify data was processed
        self.assertGreater(len(processed_data), 0)
        
        # Verify data contains both symbols
        symbols_processed = {data.symbol for data in processed_data}
        self.assertIn('AAPL', symbols_processed)
        self.assertIn('MSFT', symbols_processed)
        
        # Verify data quality
        for data in processed_data:
            self.assertIsInstance(data.timestamp, datetime)
            self.assertGreater(data.close, 0)
            self.assertGreater(data.volume, 0)
            self.assertGreaterEqual(data.high, data.low)
            self.assertGreaterEqual(data.ask, data.bid)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Suppress debug logs during testing
    
    unittest.main()
