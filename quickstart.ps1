# Quick start script for Posture Trainer (Windows)

Write-Host "=================================================="
Write-Host "Posture Trainer - Quick Start"
Write-Host "=================================================="
Write-Host ""

# Check Python version
$pythonVersion = python --version 2>&1
Write-Host "✓ Python version: $pythonVersion"

# Check if in virtual environment
if ($env:VIRTUAL_ENV) {
    Write-Host "✓ Virtual environment: $env:VIRTUAL_ENV"
} else {
    Write-Host "⚠ Warning: Not in a virtual environment" -ForegroundColor Yellow
    Write-Host "  Recommended: python -m venv venv && .\venv\Scripts\Activate.ps1"
    Write-Host ""
}

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies..."
pip install -q -r requirements.txt
Write-Host "✓ Dependencies installed"

# Run tests
Write-Host ""
Write-Host "Running tests..."
python test_dual_models.py
$testResult = $LASTEXITCODE

if ($testResult -eq 0) {
    Write-Host ""
    Write-Host "=================================================="
    Write-Host "✓ Setup complete!" -ForegroundColor Green
    Write-Host "=================================================="
    Write-Host ""
    Write-Host "Available commands:"
    Write-Host ""
    Write-Host "  1. Start Streamlit app (recommended):"
    Write-Host "     streamlit run app.py" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  2. Start desktop app (legacy):"
    Write-Host "     python main.py" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  3. Docker CPU deployment:"
    Write-Host "     docker-compose --profile cpu up -d" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  4. Docker GPU deployment:"
    Write-Host "     docker-compose --profile gpu up -d" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "=================================================="
    Write-Host ""
    Write-Host "Choose option 1 to start: streamlit run app.py" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "✗ Tests failed. Please check the output above." -ForegroundColor Red
    exit 1
}
