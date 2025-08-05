#!/usr/bin/env python3
"""
Setup script for the Algorithmic Trading System

This script sets up the development environment including:
- Virtual environment creation
- Package installation
- Directory structure validation
- Configuration file setup
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(command, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running command: {command}")
        print(f"Error output: {result.stderr}")
        sys.exit(1)
    return result


def setup_environment():
    """Set up the development environment."""
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("🚀 Setting up Algorithmic Trading System environment...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 9):
        print("❌ Python 3.9+ is required. Current version:", python_version)
        sys.exit(1)
    print(f"✅ Python version: {python_version.major}.{python_version.minor}")
    
    # Create virtual environment if it doesn't exist
    venv_path = project_root / "venv"
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        run_command(f"{sys.executable} -m venv venv")
    else:
        print("✅ Virtual environment already exists")
    
    # Determine activation script based on OS
    if os.name == 'nt':  # Windows
        activate_script = venv_path / "Scripts" / "activate"
        pip_executable = venv_path / "Scripts" / "pip"
    else:  # Unix/Linux/macOS
        activate_script = venv_path / "bin" / "activate"
        pip_executable = venv_path / "bin" / "pip"
    
    # Install requirements
    print("📚 Installing requirements...")
    run_command(f"{pip_executable} install --upgrade pip")
    run_command(f"{pip_executable} install -r requirements-dev.txt")
    
    # Create .env file from template if it doesn't exist
    env_file = project_root / "config" / ".env"
    env_template = project_root / "config" / ".env.template"
    if not env_file.exists() and env_template.exists():
        print("🔧 Creating .env file from template...")
        shutil.copy(env_template, env_file)
        print("⚠️  Please edit config/.env with your actual credentials")
    
    # Create necessary directories
    directories = [
        "data/historical",
        "data/real_time", 
        "data/processed",
        "logs/trading",
        "logs/system",
        "logs/errors",
        "tests/unit",
        "tests/integration",
        "tests/data"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("📁 Created necessary directories")
    
    # Validate installation
    print("🔍 Validating installation...")
    try:
        run_command(f"{pip_executable} list")
        print("✅ All packages installed successfully")
    except:
        print("❌ Some packages may have failed to install")
    
    print("\n🎉 Environment setup complete!")
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    if os.name == 'nt':
        print("   .\\venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Edit config/.env with your IBKR credentials")
    print("3. Run tests: python -m pytest tests/")
    print("4. Start development!")


if __name__ == "__main__":
    setup_environment()
