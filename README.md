# Posture Trainer

AI-powered posture evaluation tool using dual pose detection models (MediaPipe and YOLOv11) for real-time pose estimation and analysis.

## Features

- **Dual Model Support**: Choose between MediaPipe (CPU-optimized) or YOLOv11 (GPU-optimized)
- **Modern Web Interface**: Streamlit-based UI accessible from any browser
- **6 Analysis Modes**: Front angle, side angle, balance (back/test/front/side)
- **Multiple Input Sources**: Live camera, video files, or static images
- **Real-time Processing**: Configurable pose detection with adjustable confidence
- **GPU Acceleration**: Optional CUDA support for faster processing
- **Interactive Metrics**: Live angle measurements with min/max/average displays
- **Automated Reports**: PDF reports with angle plots and statistics
- **Easy Downloads**: One-click download for annotated results and reports
- **Configuration-Driven**: All settings externalized to YAML
- **Docker Support**: Containerized deployment with CPU and GPU variants
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

### Model Selection

The application supports two pose detection models:

1. **MediaPipe** (Default)
   - Fast and efficient on CPU
   - No additional downloads required
   - Good for real-time webcam processing
   - Lower accuracy but faster inference

2. **YOLOv11-pose**
   - State-of-the-art accuracy
   - GPU-accelerated (CUDA support)
   - Automatic model download (~6-50MB)
   - Better for batch processing

Select your preferred model in the sidebar under "Pose Detection Model".

### Streamlit Web Interface (Recommended)

```powershell
streamlit run app.py
```

**Features:**
- 🎨 Modern web UI accessible from any browser
- 🤖 Dual model support (MediaPipe / YOLOv11)
- 🚀 GPU acceleration toggle
- 📊 Real-time angle metrics with min/max/average
- 📥 One-click downloads for results and PDF reports
- 🎯 Three input modes: Image upload, Video file, or Webcam
- ⚙️ Adjustable confidence slider in sidebar

**Workflow:**
1. **Sidebar**: Select analysis type, pose model, and input source
2. **Model Settings**: Choose between CPU or GPU acceleration
3. **Upload/Capture**: Upload file or start webcam
4. **Process**: Click button to analyze
5. **View Results**: See annotated output with live metrics
6. **Download**: Get annotated media and PDF report

### Docker Deployment

Docker provides an isolated, reproducible environment for running Posture Trainer.

#### CPU-only Deployment

```bash
# Build and run CPU version
docker-compose --profile cpu up -d

# Or build manually
docker build --target cpu -t posture-trainer:cpu .
docker run -p 8501:8501 -v ./output:/app/output posture-trainer:cpu
```

#### GPU-accelerated Deployment

**Prerequisites:**
- NVIDIA GPU with CUDA support
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) installed

```bash
# Build and run GPU version
docker-compose --profile gpu up -d

# Or build manually
docker build --target gpu -t posture-trainer:gpu .
docker run --gpus all -p 8501:8501 -v ./output:/app/output posture-trainer:gpu
```

#### Access the Application

Once running, open your browser to:
```
http://localhost:8501
```

#### Stop the Container

```bash
# Using docker-compose
docker-compose --profile cpu down    # or --profile gpu

# Using docker directly
docker stop posture-trainer-cpu      # or posture-trainer-gpu
```

#### Volume Mounts

The Docker setup automatically mounts:
- `./output` - For saving processed results and reports
- `./config` - For configuration file access

#### Environment Variables

Customize behavior with environment variables:

```bash
docker run -p 8501:8501 \
  -e STREAMLIT_SERVER_PORT=8501 \
  -e STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
  -v ./output:/app/output \
  posture-trainer:cpu
```

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
- Default model type (MediaPipe or YOLOv11)
- Model confidence thresholds
- GPU acceleration settings
- YOLOv11 model size
- Output paths
- GUI settings
- Body joint labels
- Report formatting

Example configuration:
```yaml
model:
  # Default model type: 'mediapipe' or 'yolov11'
  default_type: "mediapipe"
  
  # Model confidence thresholds
  video:
    detection_confidence: 0.9
    tracking_confidence: 0.9
  
  # GPU acceleration (auto-detected if available)
  use_gpu: true
  
  # YOLOv11 specific settings
  yolov11:
    model_size: "n"  # n=nano, s=small, m=medium, l=large, x=xlarge

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
