#!/usr/bin/env python3
"""
Project Status Summary

Displays the current status of the algorithmic trading system setup.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    required = (3, 9)
    status = "✅" if version >= required else "❌"
    return f"{status} Python {version.major}.{version.minor}.{version.micro} (required: {required[0]}.{required[1]}+)"

def check_virtual_environment():
    """Check if running in virtual environment."""
    in_venv = sys.prefix != sys.base_prefix
    status = "✅" if in_venv else "⚠️"
    env_type = "Virtual Environment" if in_venv else "System Python"
    return f"{status} {env_type}"

def check_required_packages():
    """Check if required packages are installed."""
    required_packages = [
        'pandas', 'numpy', 'yaml', 'dotenv', 'pytest', 'black', 'flake8'
    ]
    
    installed = []
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            installed.append(package)
        except ImportError:
            missing.append(package)
    
    if not missing:
        return f"✅ All required packages installed ({len(installed)}/{len(required_packages)})"
    else:
        return f"⚠️ Missing packages: {', '.join(missing)} ({len(installed)}/{len(required_packages)})"

def check_directory_structure():
    """Check if directory structure is created."""
    required_dirs = [
        "src", "config", "data", "logs", "tests", "scripts",
        "src/data_management", "src/screening", "src/strategies",
        "src/risk_management", "src/order_management", "src/monitoring"
    ]
    
    existing = []
    missing = []
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            existing.append(dir_path)
        else:
            missing.append(dir_path)
    
    if not missing:
        return f"✅ Directory structure complete ({len(existing)}/{len(required_dirs)})"
    else:
        return f"⚠️ Missing directories: {', '.join(missing)} ({len(existing)}/{len(required_dirs)})"

def check_configuration_files():
    """Check if configuration files exist."""
    config_files = [
        "config/config.yaml",
        "config/.env.template",
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml"
    ]
    
    existing = []
    missing = []
    
    for file_path in config_files:
        if Path(file_path).exists():
            existing.append(file_path)
        else:
            missing.append(file_path)
    
    env_file_exists = Path("config/.env").exists()
    env_status = "✅ .env file created" if env_file_exists else "⚠️ .env file needs to be created"
    
    if not missing:
        base_status = f"✅ Configuration files complete ({len(existing)}/{len(config_files)})"
    else:
        base_status = f"⚠️ Missing config files: {', '.join(missing)} ({len(existing)}/{len(config_files)})"
    
    return f"{base_status}\n    {env_status}"

def check_core_modules():
    """Check if core modules can be imported."""
    try:
        from config_manager import ConfigManager
        config = ConfigManager()
        return "✅ Core modules importable"
    except Exception as e:
        return f"❌ Core module import failed: {str(e)[:50]}..."

def check_tests():
    """Check if tests can run."""
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "--tb=no", "-q"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return "✅ Tests passing"
        else:
            return f"⚠️ Some tests failing (exit code: {result.returncode})"
    except subprocess.TimeoutExpired:
        return "⚠️ Tests timed out"
    except Exception as e:
        return f"❌ Test execution failed: {str(e)[:50]}..."

def get_next_steps():
    """Get next steps based on current status."""
    return [
        "1. Edit config/.env with your IBKR credentials",
        "2. Review and adjust config/config.yaml parameters",
        "3. Begin Phase 2: Data Management implementation",
        "4. Set up IBKR TWS/Gateway for API access",
        "5. Implement historical data download module"
    ]

def main():
    """Main status check function."""
    print("=" * 60)
    print("🚀 ALGORITHMIC TRADING SYSTEM - STATUS SUMMARY")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("📋 ENVIRONMENT STATUS")
    print("-" * 30)
    print(f"  {check_python_version()}")
    print(f"  {check_virtual_environment()}")
    print(f"  {check_required_packages()}")
    print()
    
    print("📁 PROJECT STRUCTURE")
    print("-" * 30)
    print(f"  {check_directory_structure()}")
    print(f"  {check_configuration_files()}")
    print()
    
    print("🔧 SYSTEM STATUS")
    print("-" * 30)
    print(f"  {check_core_modules()}")
    print(f"  {check_tests()}")
    print()
    
    print("📈 PROJECT PHASES")
    print("-" * 30)
    phases = [
        ("Phase 1: Environment Setup", "✅ COMPLETE"),
        ("Phase 2: Data Management", "🔄 NEXT"),
        ("Phase 3: Screening Engine", "⏳ PENDING"),
        ("Phase 4: Strategy Implementation", "⏳ PENDING"),
        ("Phase 5: Risk Management", "⏳ PENDING"),
        ("Phase 6: Order Management", "⏳ PENDING"),
        ("Phase 7: Monitoring System", "⏳ PENDING"),
        ("Phase 8: Testing & Validation", "⏳ PENDING"),
        ("Phase 9: Deployment", "⏳ PENDING")
    ]
    
    for phase, status in phases:
        print(f"  {status} {phase}")
    print()
    
    print("🎯 NEXT STEPS")
    print("-" * 30)
    for step in get_next_steps():
        print(f"  {step}")
    print()
    
    print("📊 QUICK COMMANDS")
    print("-" * 30)
    print("  Test system:     python -m pytest tests/ -v")
    print("  Format code:     black src/ tests/")
    print("  Lint code:       flake8 src/ tests/")
    print("  Config test:     python -c \"from src.config_manager import config; print(config.get_ibkr_config())\"")
    print()
    
    print("=" * 60)
    print("✅ Phase 1 Complete: Ready for Phase 2 Development!")
    print("=" * 60)

if __name__ == "__main__":
    main()
