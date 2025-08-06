#!/usr/bin/env python3
"""
Data Management Integration Demo

Demonstrates the complete data management system including:
- Historical data download
- Real-time data streaming
- Data validation and storage
"""

import sys
from pathlib import Path
import time
import logging
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_manager import config
from src.data_management.historical_provider import HistoricalDataManager
from src.data_management.realtime_provider import RealTimeDataManager
from src.data_management.data_core import DataValidator, DataStorage


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('data_management_demo.log')
        ]
    )


def demonstrate_historical_data():
    """Demonstrate historical data download and validation."""
    print("\n" + "="*60)
    print("HISTORICAL DATA DEMONSTRATION")
    print("="*60)
    
    # Initialize managers
    storage_path = Path('./data')
    historical_manager = HistoricalDataManager(storage_path)
    validator = DataValidator()
    
    # Test symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Download historical data
    print(f"\\nDownloading historical data for {len(symbols)} symbols...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Last 30 days
    
    for symbol in symbols:
        print(f"\\nProcessing {symbol}...")
        
        try:
            # Download data
            success = historical_manager.download_symbol_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                provider='yahoo'
            )
            
            if success:
                # Load the downloaded data
                data = historical_manager.storage.load_historical_data(symbol)
                
                if data is not None and not data.empty:
                    print(f"  ✓ Downloaded {len(data)} records")
                    
                    # Validate data quality
                    quality_report = validator.validate_market_data(data, symbol)
                    print(f"  ✓ Quality score: {quality_report.quality_score:.2%}")
                    
                    if quality_report.issues:
                        print(f"  ⚠ Issues found: {', '.join(quality_report.issues)}")
                    
                    # Display sample data
                    print(f"  📊 Sample data (last 3 days):")
                    print(data.tail(3).to_string())
                    
                else:
                    print(f"  ❌ No data loaded for {symbol}")
            else:
                print(f"  ❌ Download failed for {symbol}")
                
        except Exception as e:
            print(f"  ❌ Error processing {symbol}: {e}")
    
    return historical_manager


def demonstrate_realtime_data():
    """Demonstrate real-time data streaming."""
    print("\\n" + "="*60)
    print("REAL-TIME DATA DEMONSTRATION")
    print("="*60)
    
    # Initialize real-time manager
    storage_path = Path('./data')
    rt_manager = RealTimeDataManager(storage_path)
    
    # Data collection for demonstration
    collected_data = []
    
    def data_collector(market_data):
        """Collect data for demonstration."""
        collected_data.append(market_data)
        print(f"  📈 {market_data.symbol}: ${market_data.close:.2f} "
              f"(Vol: {market_data.volume:,})")
    
    # Add callback
    rt_manager.add_data_callback(data_collector)
    
    # Initialize mock stream
    print("\\nInitializing mock data stream...")
    if rt_manager.initialize_stream('mock'):
        print("  ✓ Mock stream initialized")
        
        # Start streaming
        symbols = ['AAPL', 'MSFT']
        print(f"\\nStarting real-time streaming for {symbols}...")
        
        if rt_manager.start_streaming(symbols):
            print("  ✓ Streaming started")
            
            # Let it run for demonstration
            print("\\n📡 Real-time data feed (10 seconds):")
            time.sleep(10)
            
            # Get status
            status = rt_manager.get_streaming_status()
            print(f"\\n📊 Streaming Status:")
            print(f"  Connected: {status['connected']}")
            print(f"  Processing: {status['processing']}")
            print(f"  Subscribed symbols: {status['subscribed_symbols']}")
            print(f"  Buffer sizes: {status['buffer_sizes']}")
            print(f"  Data points collected: {len(collected_data)}")
            
            # Stop streaming
            print("\\n🛑 Stopping stream...")
            rt_manager.stop_streaming()
            print("  ✓ Stream stopped")
            
        else:
            print("  ❌ Failed to start streaming")
    else:
        print("  ❌ Failed to initialize stream")
    
    return rt_manager, collected_data


def demonstrate_data_storage():
    """Demonstrate data storage capabilities."""
    print("\\n" + "="*60)
    print("DATA STORAGE DEMONSTRATION")
    print("="*60)
    
    storage_path = Path('./data')
    storage = DataStorage(storage_path)
    
    # Check available data
    print("\\n📁 Available historical data files:")
    symbols = storage.get_available_symbols()
    if symbols:
        for symbol in symbols[:5]:  # Show first 5
            print(f"  📄 {symbol}")
            
            # Load and show summary
            data = storage.load_historical_data(symbol)
            if data is not None:
                print(f"      Records: {len(data)}, Period: {data.index.min()} to {data.index.max()}")
    else:
        print("  📭 No historical data files found")
    
    # Check real-time data storage
    print("\\n📁 Real-time data storage:")
    rt_path = storage_path / 'real_time'
    if rt_path.exists():
        rt_files = list(rt_path.glob('*.csv'))
        if rt_files:
            print(f"  📄 {len(rt_files)} real-time data files found")
            for file in rt_files[:3]:  # Show first 3
                print(f"      {file.name}")
        else:
            print("  📭 No real-time data files found yet")
    else:
        print("  📭 Real-time data directory not created yet")


def demonstrate_configuration():
    """Demonstrate configuration management."""
    print("\\n" + "="*60)
    print("CONFIGURATION DEMONSTRATION")
    print("="*60)
    
    print("\\n⚙️ Current Configuration:")
    
    # Data configuration
    data_config = config.get('DATA', {})
    print(f"  📊 Data Storage Path: {data_config.get('STORAGE_PATH', './data')}")
    print(f"  🕐 Update Frequency: {data_config.get('UPDATE_FREQUENCY_MINUTES', 60)} minutes")
    
    # Risk configuration
    risk_config = config.get_risk_config()
    print(f"  💰 Max Position Size: ${risk_config['max_position_size']:,}")
    print(f"  📉 Stop Loss: {risk_config['stop_loss_pct']:.1%}")
    
    # IBKR configuration (simulated)
    ibkr_config = config.get_ibkr_config()
    print(f"  🔌 IBKR Host: {ibkr_config['host']}:{ibkr_config['port']}")
    print(f"  🆔 Client ID: {ibkr_config['clientId']}")


def main():
    """Main demonstration function."""
    print("🚀 ALGORITHMIC TRADING SYSTEM - DATA MANAGEMENT DEMO")
    print("="*60)
    
    # Setup
    setup_logging()
    
    try:
        # Configuration
        demonstrate_configuration()
        
        # Historical data
        hist_manager = demonstrate_historical_data()
        
        # Real-time data
        rt_manager, collected_data = demonstrate_realtime_data()
        
        # Storage overview
        demonstrate_data_storage()
        
        # Summary
        print("\\n" + "="*60)
        print("DEMONSTRATION SUMMARY")
        print("="*60)
        print("✅ Historical data download and validation")
        print("✅ Real-time data streaming and processing")
        print("✅ Data quality validation and scoring")
        print("✅ Multi-format data storage (HDF5/CSV)")
        print("✅ Configuration management")
        print("✅ Error handling and logging")
        
        if collected_data:
            print(f"\\n📈 Real-time data summary:")
            symbols = set(d.symbol for d in collected_data)
            for symbol in symbols:
                symbol_data = [d for d in collected_data if d.symbol == symbol]
                avg_price = sum(d.close for d in symbol_data) / len(symbol_data)
                print(f"  {symbol}: {len(symbol_data)} updates, avg price: ${avg_price:.2f}")
        
        print("\\n🎉 Data Management Phase 2 - COMPLETE!")
        print("\\n🔜 Next: Phase 3 - Screening Engine Implementation")
        
    except Exception as e:
        logging.error(f"Demonstration failed: {e}")
        print(f"\\n❌ Error during demonstration: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
