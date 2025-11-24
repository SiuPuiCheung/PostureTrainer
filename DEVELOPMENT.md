# Development Setup Guide

## Environment Setup

### 1. Create Virtual Environment

```powershell
# Navigate to project root
cd c:\Users\cheun\Downloads\posture_trainer

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip
```

### 2. Install Dependencies

```powershell
# Install all dependencies including dev tools
python -m pip install -r requirements.txt

# OR install as editable package
python -m pip install -e ".[dev]"
```

### 3. Verify Installation

```powershell
# Run verification script
python scripts\verify_installation.py

# Expected output: All checks should pass ✅
```

## Project Structure Overview

```
posture_trainer/
├── src/                    # Source code (production)
│   ├── core/              # Analysis & detection algorithms
│   ├── data/              # Input capture & processing
│   └── utils/             # Shared utilities
├── tests/                 # Unit & integration tests
├── scripts/               # Helper scripts
├── notebooks/             # Jupyter notebooks (exploration)
├── config/                # Configuration files
├── models/                # Model artifacts (if any)
└── output/                # Generated results (gitignored)
```

## Running the Application

### Basic Run

```powershell
# From project root
python main.py
```

### Run with Custom Config

```powershell
# Modify config/config.yaml first, then:
python main.py
```

### Run in Docker

```powershell
# Build image
docker build -t posture-trainer .

# Launch container (CPU)
docker run --rm -p 8501:8501 -v ${PWD}/output:/app/output posture-trainer

# Launch with GPU acceleration (requires NVIDIA runtime)
docker run --rm -p 8501:8501 --gpus all -v ${PWD}/output:/app/output posture-trainer
```

The container caches Ultralytics pose weights under `/app/models`; mount this directory if you need persistence between runs.

## Development Workflow

### 1. Code Style

```powershell
# Format code with Black
black src/ tests/

# Check style with flake8
flake8 src/ tests/ --max-line-length=100

# Type checking with mypy
mypy src/
```

### 2. Testing

```powershell
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_analysis.py -v

# Run specific test
pytest tests/test_analysis.py::TestGeometry::test_calculate_angle_right_angle -v
```

### 3. Adding New Features

#### Add New Analysis Type

1. **Edit** `src/core/pose_analysis.py`:
   ```python
   def my_new_analysis(landmarks, mp_pose) -> Tuple:
       # Compute angles
       return (angle1, angle2, ...)
   ```

2. **Edit** `src/core/pose_detection.py`:
   ```python
   def my_new_detection(image, results, angles) -> np.ndarray:
       # Draw annotations
       return image
   ```

3. **Edit** `config/config.yaml`:
   ```yaml
   analysis:
     types:
       - id: 7
         name: "My New Analysis"
         analysis_func: "my_new_analysis"
         detection_func: "my_new_detection"
         label_index: 6
   
   body_labels:
     my_new_analysis:
       - "Joint 1"
       - "Joint 2"
   ```

4. **Test**:
   ```powershell
   python main.py
   # Select "My New Analysis" from GUI
   ```

#### Add New Utility Function

1. **Add to appropriate module** (`src/utils/`):
   ```python
   def my_utility(param1, param2):
       """Docstring with clear description."""
       # Implementation
       return result
   ```

2. **Export in** `src/utils/__init__.py`:
   ```python
   from .module import my_utility
   __all__ = [..., 'my_utility']
   ```

3. **Add test** in `tests/`:
   ```python
   def test_my_utility():
       result = my_utility(test_input1, test_input2)
       assert result == expected_output
   ```

### 4. Configuration Changes

Edit `config/config.yaml` for:
- Model confidence thresholds
- Output paths
- GUI settings
- Report formatting
- Body labels

**No code changes required** for configuration adjustments!

## Debugging

### Enable Verbose Logging

Add to `main.py`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Debug Specific Frame

```python
# In process_frame(), add:
cv2.imwrite(f"debug_frame_{frame_count}.jpg", annotated_image)
```

### Check Landmark Detection

```python
# In process_frame(), after pose.process():
if results.pose_landmarks:
    print(f"Detected {len(results.pose_landmarks.landmark)} landmarks")
```

## Common Issues

### ImportError: No module named 'src'

**Solution**: Ensure you're running from project root:
```powershell
cd c:\Users\cheun\Downloads\posture_trainer
python main.py
```

### Camera Not Found

**Solution**: Check camera permissions and index:
```python
# In capture.py, try different camera indices:
cap = cv2.VideoCapture(0)  # Try 0, 1, 2, etc.
```

### Config Not Loading

**Solution**: Verify config path:
```powershell
# Check config exists
Test-Path config\config.yaml

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"
```

## Git Workflow

### Initial Commit

```powershell
git init
git add .
git commit -m "Initial commit: MLOps structure for posture trainer"
```

### Feature Branch

```powershell
git checkout -b feature/new-analysis-type
# Make changes
git add .
git commit -m "Add new analysis type for X"
git checkout main
git merge feature/new-analysis-type
```

## Package Distribution

### Build Package

```powershell
python setup.py sdist bdist_wheel
```

### Install from Source

```powershell
python -m pip install -e .
```

### Uninstall

```powershell
# Uninstall
python -m pip uninstall posture_trainer
```

## Performance Profiling

```powershell
# Profile main script
python -m cProfile -o profile.stats main.py

# View results
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
```

## CI/CD Setup (Future)

Create `.github/workflows/test.yml`:
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt
      - run: pytest --cov=src
```

## Documentation

### Generate API Docs

```powershell
# Install sphinx
pip install sphinx sphinx-rtd-theme

# Generate docs
sphinx-quickstart docs
sphinx-apidoc -o docs/source src/
cd docs
make html
```

## Resources

- **MediaPipe Docs**: https://google.github.io/mediapipe/solutions/pose
- **OpenCV Docs**: https://docs.opencv.org/
- **MLOps Best Practices**: https://ml-ops.org/
- **Python Packaging**: https://packaging.python.org/

## Contact

For questions or issues, refer to the project README.md or copilot-instructions.md.
