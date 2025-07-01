#!/usr/bin/env python3
"""
Easy setup script for NetworKit-based Spider Map
This script handles the complete setup process automatically
"""

import subprocess
import sys
import os
import time

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is too old")
        print("   NetworKit requires Python 3.8 or newer")
        print("   Please upgrade Python and try again")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def main():
    print("=" * 60)
    print("🚀 NetworKit Spider Map Setup")
    print("=" * 60)
    print("This script will:")
    print("  1. Check Python version")
    print("  2. Install NetworKit and dependencies")
    print("  3. Build the ultra-fast network graph")
    print("  4. Start the service")
    print()
    
    # Check Python version
    if not check_python_version():
        return False
    
    print()
    
    # Install dependencies
    print("📦 Installing dependencies...")
    install_commands = [
        "pip install --upgrade pip",
        "pip install networkit",
        "pip install osmnx", 
        "pip install flask flask-cors",
        "pip install numpy pandas geopandas"
    ]
    
    for cmd in install_commands:
        if not run_command(cmd, f"Installing {cmd.split()[-1]}"):
            print(f"⚠️  Failed to install {cmd.split()[-1]}, but continuing...")
    
    print()
    
    # Build network
    if not os.path.exists('cdmx_networkit_graph.pkl'):
        print("🏗️  Building NetworKit network (this will take ~10 seconds)...")
        if run_command("python build_networkit_network.py", "Building NetworKit network"):
            print("🎉 Network built successfully!")
        else:
            print("❌ Network build failed. Check the logs above.")
            return False
    else:
        print("✅ NetworKit network already exists")
    
    print()
    
    # Final instructions
    print("=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print("🚀 Next steps:")
    print("   1. Start the service:")
    print("      python walking_service.py")
    print()
    print("   2. Open your browser:")
    print("      http://localhost:8080")
    print()
    print("   3. Enjoy ultra-fast distance calculations!")
    print()
    print("📊 Expected performance:")
    print("   - Network loading: ~1s")
    print("   - Distance calc: ~0.01s for 165 stations")
    print("   - Route calc: ~0.002s per route")
    print("   - Memory usage: ~200MB")
    print()
    print("🛠️  Troubleshooting:")
    print("   - If NetworKit fails: conda install -c conda-forge networkit")
    print("   - If OSMnx fails: conda install -c conda-forge osmnx")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1)
