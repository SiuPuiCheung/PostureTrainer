# Posture Trainer Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Interface (Streamlit)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Sidebar    │→ │  Main Area   │→ │   Metrics    │         │
│  │  - Analysis  │  │  - Upload    │  │  & Downloads │         │
│  │  - Input     │  │  - Display   │  │              │         │
│  │  - Settings  │  │  - Controls  │  │              │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Orchestrator (app.py / main.py)                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  app.py (Streamlit):                                     │  │
│  │  1. Display UI with sidebar controls                     │  │
│  │  2. Handle file uploads / webcam capture                 │  │
│  │  3. Process frames with MediaPipe                        │  │
│  │  4. Display real-time metrics                            │  │
│  │  5. Generate & offer downloads                           │  │
│  │                                                           │  │
│  │  main.py (Desktop - tkinter):                            │  │
│  │  1. Initialize CaptureManager                            │  │
│  │  2. Setup Output Writer                                  │  │
│  │  3. Start Control GUI Thread                             │  │
│  │  4. Initialize MediaPipe Pose                            │  │
│  │  5. Frame Processing Loop                                │  │
│  │  6. Generate Report                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Processing Pipeline                          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   Capture    │→ │   Analyze    │→ │    Detect    │        │
│  │   Frame      │  │   Pose       │  │  & Annotate  │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│         ↓                  ↓                  ↓                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ cv2.VideoCapture│ │ MediaPipe   │  │  Annotated   │        │
│  │    / Image    │  │  Landmarks   │  │    Image     │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Core Modules                               │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  src/core/pose_analysis.py                             │   │
│  │  ├─ front_angle_analysis(landmarks) → angles           │   │
│  │  ├─ side_angle_analysis(landmarks) → angles            │   │
│  │  ├─ balance_back_analysis(landmarks) → angles          │   │
│  │  ├─ balance_test_analysis(landmarks) → deviations      │   │
│  │  ├─ balance_front_analysis(landmarks) → angles         │   │
│  │  └─ balance_side_analysis(landmarks) → angles          │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  src/core/pose_detection.py                            │   │
│  │  ├─ front_angle_detection(img, results, angles) → img  │   │
│  │  ├─ side_angle_detection(img, results, angles) → img   │   │
│  │  ├─ balance_back_detection(img, results, angles) → img │   │
│  │  ├─ balance_test_detection(img, results, devs) → img   │   │
│  │  ├─ balance_front_detection(img, results, angles) → img│   │
│  │  └─ balance_side_detection(img, results, angles) → img │   │
│  └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Utility Modules                              │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │  geometry.py     │  │ visualization.py │                   │
│  │  ├─ get_point    │  │ ├─ draw_colored_ │                   │
│  │  │    ()          │  │ │    connection │                   │
│  │  └─ calculate_   │  │ ├─ draw_landmarks│                   │
│  │      angle()     │  │ └─ draw_labeled_ │                   │
│  │                  │  │      box()       │                   │
│  └──────────────────┘  └──────────────────┘                   │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │ config_loader.py │  │   report.py      │                   │
│  │  └─ Config class │  │  └─ ReportGen    │                   │
│  │     (YAML mgmt)  │  │     (PDF gen)    │                   │
│  └──────────────────┘  └──────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Data Management                               │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  src/data/capture.py                                   │   │
│  │  ├─ CaptureManager (GUI + capture setup)              │   │
│  │  └─ setup_output_writer()                             │   │
│  └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Configuration                                │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  config/config.yaml                                    │   │
│  │  ├─ model (confidence thresholds)                     │   │
│  │  ├─ paths (output, assets)                            │   │
│  │  ├─ analysis (types, functions, labels)               │   │
│  │  ├─ gui (layout, version)                             │   │
│  │  └─ report (formatting)                               │   │
│  └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Outputs                                    │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │  Annotated       │  │   PDF Report     │                   │
│  │  Video/Image     │  │  (plots, stats)  │                   │
│  │  output/Result_  │  │  output/Report_  │                   │
│  │  YYYYMMDD_HHMMSS │  │  YYYYMMDD_HHMMSS │                   │
│  └──────────────────┘  └──────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Input (Camera/Video/Image)
  ↓
Frame Capture (cv2.VideoCapture)
  ↓
RGB Conversion
  ↓
MediaPipe Pose Detection
  ↓
Landmark Extraction (33 points)
  ↓
┌────────────────────────────────┐
│  Analysis Function              │
│  (compute angles from landmarks)│
│  → returns tuple of angles      │
└────────────────────────────────┘
  ↓
┌────────────────────────────────┐
│  Detection Function             │
│  (annotate image with angles)   │
│  → returns annotated image      │
└────────────────────────────────┘
  ↓
Display + Save Frame
  ↓
Accumulate Angles in DataFrame
  ↓
(Repeat for all frames)
  ↓
Generate PDF Report
```

## Module Dependencies

```
main.py
  ├─ src.data.capture (CaptureManager, setup_output_writer)
  ├─ src.utils.config_loader (Config)
  ├─ src.utils.report (ReportGenerator)
  ├─ src.core.pose_analysis (6 analysis functions)
  └─ src.core.pose_detection (6 detection functions)

src.core.pose_analysis
  └─ src.utils.geometry (get_point, calculate_angle)

src.core.pose_detection
  └─ src.utils.visualization (draw_colored_connection, etc.)

src.data.capture
  ├─ src.utils.config_loader (Config)
  ├─ src.core.pose_analysis (analysis functions)
  └─ src.core.pose_detection (detection functions)

src.utils.report
  └─ src.utils.config_loader (Config)

src.utils.config_loader
  └─ config/config.yaml
```

## Configuration Flow

```
config/config.yaml
  ↓
Config class loads YAML
  ↓
Provides properties:
  ├─ model_config
  ├─ paths
  ├─ analysis_types
  ├─ body_labels
  ├─ gui_config
  └─ report_config
  ↓
Used by:
  ├─ main.py (model confidence, paths)
  ├─ CaptureManager (gui config, analysis types)
  └─ ReportGenerator (report config, body labels)
```

## Threading Model

### Streamlit (app.py)
```
Single Thread (Streamlit event loop)
  ├─ UI rendering
  ├─ File upload handling
  ├─ Frame processing (synchronous)
  ├─ Real-time display updates
  └─ Session state management
```

### Desktop (main.py)
```
Main Thread
  ├─ Video Capture Loop
  ├─ Frame Processing (MediaPipe)
  ├─ Display Window (cv2.imshow)
  └─ File Writing

Background Thread (daemon)
  └─ Control GUI (tkinter)
       ├─ Stop Button
       └─ Proceed Button
           → sets global stop_evaluation flag
```

## Extension Points

### Adding New Analysis Type
1. `src/core/pose_analysis.py` → new function
2. `src/core/pose_detection.py` → new function
3. `config/config.yaml` → register in analysis.types
4. `config/config.yaml` → add body_labels entry

### Modifying Visualization
- Edit `src/utils/visualization.py`
- Reused by all detection functions

### Changing Report Format
- Edit `src/utils/report.py`
- Config options in `config/config.yaml`

### Adding New Input Source
- Edit `src/data/capture.py`
- Update CaptureManager.initialize_capture()
