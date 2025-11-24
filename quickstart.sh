#!/bin/bash
# Quick start script for Posture Trainer

set -e

echo "=================================================="
echo "Posture Trainer - Quick Start"
echo "=================================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check if in virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment: $VIRTUAL_ENV"
else
    echo "⚠ Warning: Not in a virtual environment"
    echo "  Recommended: python -m venv venv && source venv/bin/activate"
    echo ""
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# Run tests
echo ""
echo "Running tests..."
python test_dual_models.py
test_result=$?

if [ $test_result -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✓ Setup complete!"
    echo "=================================================="
    echo ""
    echo "Available commands:"
    echo ""
    echo "  1. Start Streamlit app (recommended):"
    echo "     streamlit run app.py"
    echo ""
    echo "  2. Start desktop app (legacy):"
    echo "     python main.py"
    echo ""
    echo "  3. Docker CPU deployment:"
    echo "     docker-compose --profile cpu up -d"
    echo ""
    echo "  4. Docker GPU deployment:"
    echo "     docker-compose --profile gpu up -d"
    echo ""
    echo "=================================================="
    echo ""
    echo "Choose option 1 to start: streamlit run app.py"
    echo ""
else
    echo ""
    echo "✗ Tests failed. Please check the output above."
    exit 1
fi
