# Posture Trainer

AI-powered posture evaluation tool using MediaPipe for real-time pose estimation and analysis.

## Features

- **Modern Web Interface**: Streamlit-based UI accessible from any browser
- **6 Analysis Modes**: Front angle, side angle, balance (back/test/front/side)
- **Multiple Input Sources**: Live camera, video files, or static images
- **Real-time Processing**: MediaPipe pose detection with configurable confidence
- **Interactive Metrics**: Live angle measurements with min/max/average displays
- **Automated Reports**: PDF reports with angle plots and statistics
- **Easy Downloads**: One-click download for annotated results and reports
- **Configuration-Driven**: All settings externalized to YAML
- **Modular Architecture**: Clean separation of concerns for maintainability

## Project Structure

```
posture_trainer/
├── app.py                  # Streamlit web interface (RECOMMENDED)
├── main.py                 # Desktop tkinter interface (legacy)
├── config/
│   └── config.yaml          # Configuration file for model, paths, analysis types
├── src/
│   ├── core/
│   │   ├── pose_analysis.py    # Analysis functions for computing joint angles
│   │   └── pose_detection.py   # Detection/annotation functions for visualization
│   ├── data/
│   │   └── capture.py          # Video/image capture and input handling (tkinter)
│   ├── utils/
│   │   ├── config_loader.py    # Configuration management
│   │   ├── geometry.py         # Geometric calculations (angles, points)
│   │   ├── report.py           # PDF report generation
│   │   └── visualization.py    # Drawing utilities (landmarks, connections, labels)
│   └── __init__.py
├── tests/                   # Unit tests
├── notebooks/              # Jupyter notebooks for exploration
├── models/                 # Model artifacts (if any)
├── image/                  # GUI assets (bg.png)
├── output/                 # Generated results and reports
├── requirements.txt        # Python dependencies
└── setup.py               # Package installation

```

## Installation

### Quick Start

```powershell
# Clone the repository
git clone <repository-url>
cd posture_trainer

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt

# Or install as package
python -m pip install -e .
```

### Development Installation

```powershell
# Install with dev dependencies
python -m pip install -e ".[dev]"
```

## Quick Start

```powershell
# Install dependencies
python -m pip install -r requirements.txt

# Run Streamlit web application (RECOMMENDED)
streamlit run app.py

# Or run desktop application (legacy)
python main.py
```

## Usage

### Streamlit Web Interface (Recommended)

```powershell
streamlit run app.py
```

**Features:**
- 🎨 Modern web UI accessible from any browser
- 📊 Real-time angle metrics with min/max/average
- 📥 One-click downloads for results and PDF reports
- 🎯 Three input modes: Image upload, Video file, or Webcam
- ⚙️ Adjustable confidence slider in sidebar

**Workflow:**
1. **Sidebar**: Select analysis type and input source
2. **Upload/Capture**: Upload file or start webcam
3. **Process**: Click button to analyze
4. **View Results**: See annotated output with live metrics
5. **Download**: Get annotated media and PDF report

### Desktop Application (Legacy)

```powershell
python main.py
```

**Workflow:**
1. **Select Analysis Type**: Choose from 6 analysis modes via GUI
2. **Select Input Source**: Image file, video file, or live camera
3. **Processing**: Real-time pose detection and annotation
4. **Stop/Proceed**: Use control GUI to stop analysis anytime
5. **Report Generation**: Automatic PDF report with plots and statistics

### Configuration

Edit `config/config.yaml` to customize:
- Model confidence thresholds
- Output paths
- GUI settings
- Body joint labels
- Report formatting

Example configuration:
```yaml
model:
  video:
    detection_confidence: 0.9
    tracking_confidence: 0.9

paths:
  output_dir: "output"
  assets_dir: "image"
```

## Development

### Running Tests

```powershell
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

### Code Quality

```powershell
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Analysis Types

1. **Create analysis function** in `src/core/pose_analysis.py`:
   ```python
   def new_analysis(landmarks, mp_pose) -> Tuple:
       # Compute angles from landmarks
       return (angle1, angle2, ...)
   ```

2. **Create detection function** in `src/core/pose_detection.py`:
   ```python
   def new_detection(image, results, angles) -> np.ndarray:
       # Annotate image with angles
       return image
   ```

3. **Register in config** (`config/config.yaml`):
   ```yaml
   analysis:
     types:
       - id: 7
         name: "New Analysis"
         analysis_func: "new_analysis"
         detection_func: "new_detection"
         label_index: 6
   ```

4. **Add body labels** in config:
   ```yaml
   body_labels:
     new_analysis:
       - "Joint 1"
       - "Joint 2"
   ```

## Architecture

### Pipeline Flow

```
1. CaptureManager.initialize_capture()
   ↓ (cap, is_image, frame_rate, anal_func, detect_func, analysis_choice)
2. setup_output_writer()
   ↓ (out)
3. Mediapipe Pose Context
   ↓
4. Frame Loop:
   - process_frame(frame, pose, anal_func, detect_func)
     → anal_func(landmarks) → angles
     → detect_func(image, results, angles) → annotated_image
   - Append angles to DataFrame
   - Write to output
   ↓
5. ReportGenerator.generate_report()
   → PDF with plots and statistics
```

### Key Components

- **Config**: Centralized YAML-based configuration
- **CaptureManager**: GUI dialogs and video/image capture initialization
- **Analysis Functions**: Compute joint angles from landmarks (signature: `landmarks, mp_pose -> tuple`)
- **Detection Functions**: Annotate images with analysis results (signature: `image, results, angles -> image`)
- **ReportGenerator**: Create PDF reports with matplotlib

## Dependencies

- `opencv-python`: Video/image processing
- `mediapipe`: Pose estimation
- `numpy`, `pandas`: Data manipulation
- `matplotlib`: Plotting and reports
- `pyyaml`: Configuration management
- `tkinter`: GUI (usually included with Python)

## Troubleshooting

### No GUI Background Image
- Ensure `image/bg.png` exists
- Code falls back gracefully if missing

### Camera Not Found
- Check camera permissions
- Verify camera index (default: 0)

### Import Errors
- Ensure you're running from project root
- Check Python path includes `src/`

## License

MIT License

Copyright (c) 2025 Siu Pui Cheung

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Author

Siu Pui Cheung

## Version

0.1.0
