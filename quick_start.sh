#!/bin/bash

# Quick Start Script for Walk-Forward Ensemble ML Trading System
# This script sets up the environment and runs a demo

echo "🚀 Walk-Forward Ensemble ML Trading System - Quick Start"
echo "========================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $python_version detected. Python 3.8+ is required."
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data models results reports logs cache

# Run demo
echo "🎯 Running demonstration..."
python3 example.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Quick start completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Review the generated files:"
    echo "   - sample_data.csv (sample dataset)"
    echo "   - demo_performance_report.txt (performance analysis)"
    echo ""
    echo "2. Run the full system with your data:"
    echo "   python3 main.py --data-file your_data.csv --ensemble-method Voting"
    echo ""
    echo "3. Explore different ensemble methods:"
    echo "   python3 main.py --data-file your_data.csv --ensemble-method Stacking"
    echo "   python3 main.py --data-file your_data.csv --ensemble-method Blending"
    echo ""
    echo "📖 For more information, see README.md"
else
    echo "❌ Demo failed. Check the error messages above."
    exit 1
fi
