#!/usr/bin/env python3
"""
Project Initialization Script

This script initializes the algorithmic trading system by:
- Validating the environment
- Setting up necessary directories
- Creating initial configuration files
- Running basic system checks
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from config_manager import config
except ImportError as e:
    print(f"Error importing config_manager: {e}")
    print("Make sure you've installed the required dependencies: pip install -r requirements.txt")
    sys.exit(1)


def setup_logging():
    """Set up basic logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "initialization.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def validate_environment():
    """Validate the Python environment and dependencies."""
    logger = logging.getLogger(__name__)
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 9):
        logger.error(f"Python 3.9+ required. Current version: {python_version}")
        return False
    
    logger.info(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check required packages
    required_packages = ['pandas', 'numpy', 'yaml', 'dotenv']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Install with: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All required packages are installed")
    return True


def create_directory_structure():
    """Create necessary directory structure."""
    logger = logging.getLogger(__name__)
    
    directories = [
        "data/historical",
        "data/real_time",
        "data/processed",
        "data/backtest_results",
        "logs/trading",
        "logs/system", 
        "logs/errors",
        "logs/performance",
        "tests/unit",
        "tests/integration",
        "tests/data",
        "scripts/data_download",
        "scripts/backtest",
        "scripts/deployment"
    ]
    
    created_dirs = []
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(directory)
    
    if created_dirs:
        logger.info(f"📁 Created directories: {', '.join(created_dirs)}")
    else:
        logger.info("📁 All directories already exist")
    
    return True


def setup_configuration():
    """Set up configuration files."""
    logger = logging.getLogger(__name__)
    
    try:
        # Validate configuration
        config.validate_config()
        logger.info("✅ Configuration validation passed")
        
        # Create .env file if it doesn't exist
        env_file = Path("config/.env")
        env_template = Path("config/.env.template")
        
        if not env_file.exists() and env_template.exists():
            import shutil
            shutil.copy(env_template, env_file)
            logger.warning("⚠️  Created .env file from template. Please edit with your credentials.")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration setup failed: {e}")
        return False


def create_sample_scripts():
    """Create sample scripts for common tasks."""
    logger = logging.getLogger(__name__)
    
    # Sample data download script
    sample_download_script = '''#!/usr/bin/env python3
"""
Sample Data Download Script

Downloads historical data for backtesting.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    print("📊 Sample data download script")
    print("This script will be implemented in Phase 2: Data Acquisition")
    print("Will download historical data from IBKR API for backtesting")

if __name__ == "__main__":
    main()
'''
    
    script_path = Path("scripts/data_download/download_historical.py")
    if not script_path.exists():
        with open(script_path, 'w') as f:
            f.write(sample_download_script)
        logger.info("📝 Created sample data download script")
    
    return True


def run_system_checks():
    """Run basic system checks."""
    logger = logging.getLogger(__name__)
    
    checks_passed = 0
    total_checks = 4
    
    # Check 1: Configuration loading
    try:
        ibkr_config = config.get_ibkr_config()
        risk_config = config.get_risk_config()
        logger.info("✅ Configuration loading successful")
        checks_passed += 1
    except Exception as e:
        logger.error(f"❌ Configuration loading failed: {e}")
    
    # Check 2: Directory permissions
    try:
        test_file = Path("logs/test_write.tmp")
        test_file.write_text("test")
        test_file.unlink()
        logger.info("✅ Directory write permissions OK")
        checks_passed += 1
    except Exception as e:
        logger.error(f"❌ Directory write permissions failed: {e}")
    
    # Check 3: Data directory structure
    try:
        required_dirs = ["data", "logs", "config", "src"]
        missing_dirs = [d for d in required_dirs if not Path(d).exists()]
        if not missing_dirs:
            logger.info("✅ Directory structure validation passed")
            checks_passed += 1
        else:
            logger.error(f"❌ Missing directories: {missing_dirs}")
    except Exception as e:
        logger.error(f"❌ Directory structure check failed: {e}")
    
    # Check 4: Import system modules
    try:
        from config_manager import ConfigManager
        logger.info("✅ System modules import successfully")
        checks_passed += 1
    except Exception as e:
        logger.error(f"❌ System modules import failed: {e}")
    
    logger.info(f"System checks: {checks_passed}/{total_checks} passed")
    return checks_passed == total_checks


def main():
    """Main initialization function."""
    print("🚀 Initializing Algorithmic Trading System...")
    print("=" * 50)
    
    # Set up logging
    logger = setup_logging()
    logger.info(f"Starting system initialization at {datetime.now()}")
    
    # Run initialization steps
    steps = [
        ("Validating environment", validate_environment),
        ("Creating directory structure", create_directory_structure),
        ("Setting up configuration", setup_configuration),
        ("Creating sample scripts", create_sample_scripts),
        ("Running system checks", run_system_checks)
    ]
    
    failed_steps = []
    for step_name, step_func in steps:
        logger.info(f"🔄 {step_name}...")
        try:
            if not step_func():
                failed_steps.append(step_name)
        except Exception as e:
            logger.error(f"❌ {step_name} failed with exception: {e}")
            failed_steps.append(step_name)
    
    # Summary
    print("\n" + "=" * 50)
    if not failed_steps:
        logger.info("🎉 System initialization completed successfully!")
        print("\nNext steps:")
        print("1. Edit config/.env with your IBKR credentials")
        print("2. Review config/config.yaml for trading parameters")
        print("3. Run: python -m pytest tests/ (when tests are implemented)")
        print("4. Start implementing Phase 2: Data Acquisition")
    else:
        logger.error(f"❌ Initialization failed. Failed steps: {failed_steps}")
        print("\nPlease fix the above issues and run the script again.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
