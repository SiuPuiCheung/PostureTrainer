# Copilot / AI agent instructions for Posture Trainer

This repository is an AI-powered posture evaluation tool using MediaPipe. The project follows MLOps best practices with modular architecture, configuration-driven design, and clear separation of concerns.

## Architecture Overview

**Entrypoints:** 
- `app.py` — Streamlit web interface (recommended)
- `main.py` — Desktop tkinter interface (legacy)

**Core libs:** `opencv-python` (`cv2`), `mediapipe`, `numpy`, `pandas`, `matplotlib`, `pyyaml`, `streamlit`, `pillow`

**Artifacts:** Results written to `output/Result_<timestamp>.(jpg|mp4)` and `output/Report_<timestamp>.pdf`

**Configuration:** All settings centralized in `config/config.yaml` (model confidence, paths, analysis types, GUI settings, body labels)

## Project Structure

```
app.py              # Streamlit web interface (RECOMMENDED)
main.py             # Desktop tkinter interface (legacy)

src/
├── core/           # Analysis and detection logic
│   ├── pose_analysis.py    # 6 analysis functions computing joint angles
│   └── pose_detection.py   # 6 detection functions annotating images
├── data/
│   └── capture.py          # CaptureManager (tkinter GUI, used by main.py)
└── utils/
    ├── config_loader.py    # Config class for YAML settings
    ├── geometry.py         # get_point(), calculate_angle()
    ├── visualization.py    # draw_colored_connection(), draw_landmarks(), draw_labeled_box()
    └── report.py           # ReportGenerator for PDF creation
```

## Pipeline Dataflow

### Streamlit (app.py)
1. **Sidebar UI** → user selects analysis type, input source, confidence level
2. **File upload / webcam capture** → Streamlit handles input
3. **process_image() / process_video() / process_webcam()** → with `mp.solutions.pose.Pose` context:
   - **process_frame()** → `pose.process()` → `anal_func(landmarks)` → angles → `detect_func(image, results, angles)` → annotated_image
   - Angles appended to `joint_angles_df` DataFrame
4. **display_metrics()** → shows real-time angle measurements in columns
5. **ReportGenerator.generate_report()** → writes PDF with time-series plots and statistics
6. **Download buttons** → user downloads annotated results and PDF reports

### Desktop (main.py)
1. **CaptureManager.initialize_capture()** → shows GUI dialogs, returns `(cap, is_image, frame_rate, anal_func, detect_func, analysis_choice)`
2. **setup_output_writer()** → creates VideoWriter or image path
3. **Main loop in run_estimation()** → with `mp.solutions.pose.Pose` context:
   - **process_frame()** → `pose.process()` → `anal_func(landmarks)` → angles → `detect_func(image, results, angles)` → annotated_image
   - Angles appended to `joint_angles_df` DataFrame
4. **ReportGenerator.generate_report()** → writes PDF with time-series plots and statistics
5. **Control GUI (show_stop_gui)** → threaded, toggles `stop_evaluation` flag

## Project-Specific Conventions

### Analysis/Detection Function Pairs
Each analysis mode has paired functions in `src/core/`:
- **Analysis**: `def foo_analysis(landmarks, mp_pose) -> Tuple` — computes angles/metrics from landmarks
- **Detection**: `def foo_detection(image, results, angles) -> np.ndarray` — annotates image with visual overlays

Example: `front_angle_analysis` / `front_angle_detection`

### Adding New Analysis Types (4 steps)

1. **Create analysis function** in `src/core/pose_analysis.py`:
   ```python
   def new_analysis(landmarks, mp_pose) -> Tuple:
       # Use get_point() and calculate_angle() from utils.geometry
       return (angle1, angle2, ...)
   ```

2. **Create detection function** in `src/core/pose_detection.py`:
   ```python
   def new_detection(image, results, angles) -> np.ndarray:
       # Use draw_colored_connection(), draw_landmarks(), draw_labeled_box()
       return image
   ```

3. **Register in config/config.yaml**:
   ```yaml
   analysis:
     types:
       - id: 7
         name: "New Analysis"
         analysis_func: "new_analysis"
         detection_func: "new_detection"
         label_index: 6
   ```

4. **Add body labels** in `config/config.yaml`:
   ```yaml
   body_labels:
     new_analysis:
       - "Joint 1 Name"
       - "Joint 2 Name"
   ```

### Landmark & Geometry Helpers
- **Always use** `get_point(landmarks, landmark_enum)` from `src/utils/geometry.py` for consistent indexing
- **Always use** `calculate_angle(a, b, c)` for angle computation
- `landmarks_to_draw = list(mp.solutions.pose.PoseLandmark)` is the canonical landmark list

### Visualization Helpers
Reuse these from `src/utils/visualization.py` for consistency:
- `draw_colored_connection(image, results, start_idx, end_idx, color, thickness)`
- `draw_landmarks(image, results, landmark_indices, color, radius)`
- `draw_labeled_box(image, results, joint_landmarks, angles, padding, ...)`

### Configuration Management
- **Load config:** `config = Config()` (defaults to `config/config.yaml`)
- **Access settings:** `config.model_config`, `config.paths`, `config.analysis_types`, `config.body_labels`, `config.gui_config`, `config.report_config`
- **Get labels by index:** `config.get_body_labels_by_index(analysis_choice)`

### State Management

**Streamlit (app.py):**
- `st.session_state`: manages all application state
- `st.session_state.config`: Config instance
- `st.session_state.processed`: processing status flag
- `st.session_state.joint_angles_df`: accumulated angle data
- `st.session_state.analysis_choice`: selected analysis mode
- No threading required — Streamlit handles UI reactivity

**Desktop (main.py):**
- `stop_evaluation`: toggled by threaded GUI to halt processing
- `joint_angles_df`: accumulates angle data across frames
- `run`: controls outer loop for multiple analysis sessions
- **Critical:** Avoid blocking the main thread — `show_stop_gui()` runs in a separate thread and must remain responsive.

## Integration Points & External Assumptions

- **GUI assets:** `image/bg.png` — code gracefully handles missing file with `os.path.exists()` check
- **PyInstaller packaging:** Code checks `getattr(sys, 'frozen', False)` and `sys._MEIPASS` for bundled resources
- **Camera access:** `cv2.VideoCapture(0)` for live feed; tests should mock or skip hardware checks
- **Output directory:** `output/` auto-created if missing via `os.makedirs(output_dir, exist_ok=True)`

## Build / Run / Debug Commands

### Quick Start
```powershell
# Install dependencies
python -m pip install -r requirements.txt

# Run Streamlit web application (RECOMMENDED)
streamlit run app.py

# Or run desktop application
python main.py
```

### Development
```powershell
# Install with dev dependencies
python -m pip install -e ".[dev]"

# Run tests
pytest

# Code quality
black src/ tests/
flake8 src/
mypy src/
```

### Legacy Notebook
Original implementation in `evaluator.ipynb` — can still run interactively:
```powershell
jupyter notebook evaluator.ipynb
```

## Testing Strategy

1. **Unit tests** in `tests/`:
   - Test analysis functions with mock landmarks
   - Test geometry calculations (angles, points)
   - Test config loading

2. **Integration tests**:
   - Test `process_frame()` with static test image
   - Verify analysis → detection pipeline
   - Check report generation with synthetic data

3. **Smoke test**:
   ```python
   from src.core.pose_analysis import front_angle_analysis
   from src.utils.config_loader import Config
   # Create mock landmarks and verify return shape
   ```

## What to Avoid

- **Don't** modify `config/config.yaml` analysis types without updating both `pose_analysis.py` and `pose_detection.py`
- **Don't** add blocking operations in main thread (use threading for long-running tasks)
- **Don't** hardcode paths, thresholds, or labels — use `config/config.yaml`
- **Don't** bypass helper functions like `get_point()` or `calculate_angle()` — maintain consistency
- **Don't** assume `image/bg.png` exists — code must handle missing assets

## Key Files for Context

- `app.py` — Streamlit web interface, modern UI implementation
- `main.py` — desktop pipeline orchestration (legacy)
- `src/core/pose_analysis.py` — all 6 analysis functions, see signature patterns
- `src/core/pose_detection.py` — all 6 detection functions, see annotation patterns
- `config/config.yaml` — configuration schema, understand settings structure
- `src/utils/config_loader.py` — Config class API
- `README.md` — comprehensive usage guide and architecture docs

## Quick Reference: Adding Features

| Task | Files to Edit |
|------|---------------|
| New analysis mode | `pose_analysis.py`, `pose_detection.py`, `config.yaml` |
| Change model confidence | `config.yaml` (model section) |
| Modify GUI layout | `config.yaml` (gui section) |
| Change output paths | `config.yaml` (paths section) |
| Add visualization style | `src/utils/visualization.py` |
| Extend report format | `src/utils/report.py` |

**Note:** The original `evaluator.ipynb` notebook is kept for reference but the production code now follows the modular `src/` structure.
