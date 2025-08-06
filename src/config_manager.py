"""Configuration Management Module

Handles loading and validation of system configuration from YAML files and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class ConfigManager:
    """Centralized configuration management for the trading system."""
    
    def __init__(self, config_path: Optional[str] = None, env_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file
            env_file: Path to the .env file
        """
        self.project_root = Path(__file__).parent.parent
        self.config_path = Path(config_path) if config_path else self.project_root / "config" / "config.yaml"
        self.env_file = Path(env_file) if env_file else self.project_root / "config" / ".env"
        
        # Load environment variables
        if self.env_file.exists():
            load_dotenv(self.env_file)
        
        # Load configuration if file exists
        self._config = self._load_config() if self.config_path.exists() else {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Override with environment variables where applicable
            self._apply_env_overrides(config)
            
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {self.config_path}: {e}")
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> None:
        """Apply environment variable overrides to configuration."""
        # IBKR Configuration
        ibkr_host = os.getenv('IBKR_HOST')
        if ibkr_host:
            config['IBKR_HOST'] = ibkr_host
        
        ibkr_port = os.getenv('IBKR_PORT')
        if ibkr_port:
            config['IBKR_PORT'] = int(ibkr_port)
        
        ibkr_client_id = os.getenv('IBKR_CLIENT_ID')
        if ibkr_client_id:
            config['IBKR_CLIENT_ID'] = int(ibkr_client_id)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_ibkr_config(self) -> Dict[str, Any]:
        """Get IBKR connection configuration."""
        return {
            'host': self.get('IBKR_HOST', '127.0.0.1'),
            'port': self.get('IBKR_PORT', 7497),
            'clientId': self.get('IBKR_CLIENT_ID', 1),
            'timeout': self.get('IBKR_TIMEOUT', 10)
        }
    
    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration."""
        return {
            'daily_profit_target': self.get('DAILY_PROFIT_TARGET', 50.0),
            'initial_capital': self.get('INITIAL_CAPITAL', 10000.0),
            'max_daily_drawdown': self.get('MAX_DAILY_DRAWDOWN', 0.03),
            'max_weekly_drawdown': self.get('MAX_WEEKLY_DRAWDOWN', 0.05),
            'max_monthly_drawdown': self.get('MAX_MONTHLY_DRAWDOWN', 0.10),
            'max_risk_per_trade': self.get('MAX_RISK_PER_TRADE', 0.01),
            'max_position_size': self.get('MAX_POSITION_SIZE', 0.05),
            'max_concurrent_positions': self.get('MAX_CONCURRENT_POSITIONS', 20),
            'stop_loss_pct': self.get('MEAN_REVERSION', {}).get('STOP_LOSS_PCT', 0.02)
        }
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get configuration for a specific strategy."""
        return self.get(strategy_name.upper(), {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data management configuration."""
        return {
            'storage_path': Path(self.get('DATA_STORAGE_PATH', './data')),
            'log_path': Path(self.get('LOG_STORAGE_PATH', './logs')),
            'historical_days': self.get('HISTORICAL_DATA_DAYS', 252)
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring and alerting configuration."""
        return {
            'log_level': self.get('LOG_LEVEL', 'INFO'),
            'enable_dashboard': self.get('ENABLE_DASHBOARD', True),
            'dashboard_host': self.get('DASHBOARD_HOST', '0.0.0.0'),
            'dashboard_port': self.get('DASHBOARD_PORT', 5000),
            'email_alerts': self.get('EMAIL_ALERTS', False),
            'telegram_alerts': self.get('TELEGRAM_ALERTS', False)
        }
    
    def validate_config(self) -> bool:
        """Validate the loaded configuration."""
        required_keys = [
            'DAILY_PROFIT_TARGET',
            'INITIAL_CAPITAL',
            'MAX_DAILY_DRAWDOWN',
            'IBKR_HOST',
            'IBKR_PORT'
        ]
        
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
        return True
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary."""
        return self._config.copy()


# Global configuration instance
config = ConfigManager()
