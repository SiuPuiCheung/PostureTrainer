# Posture Trainer - Dual Model Feature Guide

## Overview

This document describes the dual model architecture implementation in Posture Trainer, allowing users to choose between MediaPipe and YOLOv11 for pose detection.

## Features Implemented

### 1. Model Selection Architecture

```
src/core/
├── base_pose_model.py      # Abstract base class
├── mediapipe_model.py      # MediaPipe wrapper
├── yolo_model.py           # YOLOv11 wrapper
└── model_factory.py        # Factory pattern for model creation
```

**Key Design Principles:**
- Unified interface via `BasePoseModel` abstract class
- Factory pattern for easy model instantiation
- Consistent 33-landmark format (MediaPipe standard)
- Context manager support for resource cleanup

### 2. User Interface Enhancements

The Streamlit sidebar now includes:

#### Model Selection (Section 3)
```
📍 Pose Detection Model
└─ Dropdown: MediaPipe (Fast, CPU-optimized) / YOLOv11 (Accurate, GPU-optimized)
```

#### GPU Acceleration Toggle
```
└─ Checkbox: 🚀 Use GPU Acceleration
   ├─ Auto-detected when CUDA is available
   ├─ Shows status: ✓ GPU enabled / ℹ️ Using CPU / ⚠️ GPU not available
   └─ Works with both MediaPipe and YOLOv11
```

#### Auto-download Notice
```
For YOLOv11: 🔄 Model will be downloaded automatically on first use (~6-50MB)
```

### 3. Model Comparison

| Feature | MediaPipe | YOLOv11 |
|---------|-----------|---------|
| **Speed** | Fast | Medium-Fast |
| **Accuracy** | Good | Excellent |
| **Hardware** | CPU-optimized | GPU-optimized |
| **Model Size** | ~20MB (auto-download) | 6-50MB (depends on variant) |
| **Keypoints** | 33 landmarks | 17 keypoints (mapped to 33) |
| **Best For** | Real-time webcam, CPU-only systems | Batch processing, high accuracy needs |

### 4. Configuration

New configuration options in `config/config.yaml`:

```yaml
model:
  # Default model type: 'mediapipe' or 'yolov11'
  default_type: "mediapipe"
  
  # Model confidence thresholds
  image:
    detection_confidence: 0.6
    tracking_confidence: 0.6
  video:
    detection_confidence: 0.9
    tracking_confidence: 0.9
  
  # GPU acceleration (auto-detected if available)
  use_gpu: true
  
  # YOLOv11 specific settings
  yolov11:
    model_size: "n"  # n=nano, s=small, m=medium, l=large, x=xlarge
```

### 5. Docker Support

Two deployment options:

#### CPU-Only Container
```bash
docker-compose --profile cpu up -d
```
- Smaller image size
- Works on any hardware
- Uses MediaPipe by default
- Can still use YOLOv11 (slower on CPU)

#### GPU-Accelerated Container
```bash
docker-compose --profile gpu up -d
```
- Requires nvidia-docker
- Full GPU acceleration for YOLOv11
- Faster video processing
- Recommended for production use

### 6. Code Changes Summary

#### app.py Updates

**New imports:**
```python
from src.core.model_factory import create_pose_model, get_available_models, check_gpu_available
```

**Session state additions:**
```python
st.session_state.model_type = "mediapipe"
st.session_state.use_gpu = check_gpu_available()
st.session_state.available_models = get_available_models()
```

**Updated processing functions:**
- `process_frame()` - Now accepts `model_type` parameter
- `process_image()` - Adds `model_type` and `use_gpu` parameters
- `process_video()` - Adds `model_type` and `use_gpu` parameters
- `process_webcam()` - Adds `model_type` and `use_gpu` parameters

All functions now use the factory pattern:
```python
if model_type == "mediapipe":
    pose_context = mp.solutions.pose.Pose(...)
else:
    pose_context = create_pose_model(model_type, confidence=confidence, use_gpu=use_gpu)
```

#### requirements.txt Updates

New dependencies:
```
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
```

### 7. Landmark Mapping

YOLOv11 provides 17 keypoints, which are mapped to MediaPipe's 33 landmarks:

```python
YOLO_TO_MEDIAPIPE_MAP = {
    0: 0,   # nose -> nose
    1: 2,   # left_eye -> left_eye
    2: 5,   # right_eye -> right_eye
    3: 7,   # left_ear -> left_ear
    4: 8,   # right_ear -> right_ear
    5: 11,  # left_shoulder -> left_shoulder
    6: 12,  # right_shoulder -> right_shoulder
    # ... (see yolo_model.py for complete mapping)
}
```

Missing landmarks are filled with zero visibility, ensuring compatibility with existing analysis functions.

## Usage Examples

### Command Line (Python)

```python
from src.core.model_factory import create_pose_model
import cv2

# Create MediaPipe model
with create_pose_model("mediapipe", confidence=0.7) as model:
    image = cv2.imread("pose.jpg")
    results = model.process(image)
    if results.pose_landmarks:
        print(f"Detected {len(results.pose_landmarks)} landmarks")

# Create YOLOv11 model with GPU
with create_pose_model("yolov11", confidence=0.7, use_gpu=True) as model:
    image = cv2.imread("pose.jpg")
    results = model.process(image)
    if results.pose_landmarks:
        print(f"Detected {len(results.pose_landmarks)} landmarks")
```

### Streamlit Web App

1. Start the app: `streamlit run app.py`
2. In the sidebar:
   - Section 1: Choose analysis type (e.g., "Front Angle")
   - Section 2: Choose input source (e.g., "Image")
   - Section 3: Select model (MediaPipe or YOLOv11)
   - Toggle GPU if available
   - Section 4: Adjust confidence threshold
3. Upload image/video or start webcam
4. View results and download reports

### Docker Deployment

```bash
# CPU-only (default MediaPipe)
docker-compose --profile cpu up -d

# Access at http://localhost:8501

# GPU-accelerated (can use YOLOv11)
docker-compose --profile gpu up -d

# Logs
docker-compose logs -f posture-trainer-cpu
```

## Testing

Run the test suite:
```bash
python test_dual_models.py
```

Tests include:
- ✓ Import verification
- ✓ Model factory functionality
- ✓ Configuration loading
- ✓ PoseLandmark class
- ✓ PoseResults class
- ✓ YOLOv11 availability check

## Performance Recommendations

### For Real-time Webcam
- **Use MediaPipe** - Faster on CPU
- Lower confidence threshold (0.5-0.7)
- Resolution: 640x480 or 1280x720

### For Batch Video Processing
- **Use YOLOv11 with GPU** - Higher accuracy
- Higher confidence threshold (0.7-0.9)
- Resolution: 1920x1080 or higher

### For Production Deployment
- **Docker GPU variant** - Better resource management
- Mount volumes for persistent output
- Set appropriate confidence based on use case

## Troubleshooting

### YOLOv11 not available
**Symptom:** Only MediaPipe shows in dropdown

**Solution:** Install YOLOv11 dependencies:
```bash
pip install ultralytics torch torchvision
```

### GPU not detected
**Symptom:** "GPU not available, using CPU" warning

**Solution:**
1. Check CUDA installation: `nvidia-smi`
2. Install CUDA-enabled PyTorch: https://pytorch.org/get-started/locally/
3. For Docker: Use `--profile gpu` and ensure nvidia-docker is installed

### Model download fails
**Symptom:** Network error when downloading models

**Solution:**
- Check internet connection
- For YOLOv11: Models cached in `~/.cache/ultralytics/`
- For MediaPipe: Models cached in `~/.local/lib/python*/site-packages/mediapipe/modules/`

## Future Enhancements

Potential improvements:
1. Add more pose models (MMPose, OpenPose)
2. Model ensemble for higher accuracy
3. Custom model training pipeline
4. Real-time performance metrics display
5. Model benchmarking tool

## References

- MediaPipe Pose: https://google.github.io/mediapipe/solutions/pose.html
- Ultralytics YOLOv11: https://docs.ultralytics.com/models/yolo11/
- Docker GPU Support: https://docs.docker.com/config/containers/resource_constraints/#gpu
