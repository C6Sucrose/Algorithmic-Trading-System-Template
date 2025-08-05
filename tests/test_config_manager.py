"""Test configuration manager functionality."""

import pytest
import tempfile
import yaml
from pathlib import Path
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config_manager import ConfigManager


class TestConfigManager:
    """Test cases for ConfigManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_config = {
            'DAILY_PROFIT_TARGET': 50.0,
            'INITIAL_CAPITAL': 10000.0,
            'MAX_DAILY_DRAWDOWN': 0.03,
            'IBKR_HOST': '127.0.0.1',
            'IBKR_PORT': 7497,
            'IBKR_CLIENT_ID': 1,
            'MEAN_REVERSION': {
                'BOLLINGER_PERIOD': 20,
                'RSI_PERIOD': 14
            }
        }
    
    def test_config_loading(self):
        """Test configuration loading from YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name
        
        try:
            config_manager = ConfigManager(config_path=config_path)
            assert config_manager.get('DAILY_PROFIT_TARGET') == 50.0
            assert config_manager.get('IBKR_HOST') == '127.0.0.1'
        finally:
            Path(config_path).unlink()
    
    def test_nested_config_access(self):
        """Test accessing nested configuration values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name
        
        try:
            config_manager = ConfigManager(config_path=config_path)
            assert config_manager.get('MEAN_REVERSION.BOLLINGER_PERIOD') == 20
            assert config_manager.get('MEAN_REVERSION.RSI_PERIOD') == 14
        finally:
            Path(config_path).unlink()
    
    def test_default_values(self):
        """Test default value handling."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name
        
        try:
            config_manager = ConfigManager(config_path=config_path)
            assert config_manager.get('NON_EXISTENT_KEY', 'default') == 'default'
            assert config_manager.get('NESTED.NON_EXISTENT', 42) == 42
        finally:
            Path(config_path).unlink()
    
    def test_ibkr_config_getter(self):
        """Test IBKR configuration getter."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name
        
        try:
            config_manager = ConfigManager(config_path=config_path)
            ibkr_config = config_manager.get_ibkr_config()
            
            assert ibkr_config['host'] == '127.0.0.1'
            assert ibkr_config['port'] == 7497
            assert ibkr_config['clientId'] == 1
        finally:
            Path(config_path).unlink()
    
    def test_risk_config_getter(self):
        """Test risk configuration getter."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name
        
        try:
            config_manager = ConfigManager(config_path=config_path)
            risk_config = config_manager.get_risk_config()
            
            assert risk_config['daily_profit_target'] == 50.0
            assert risk_config['initial_capital'] == 10000.0
            assert risk_config['max_daily_drawdown'] == 0.03
        finally:
            Path(config_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
