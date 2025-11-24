"""Base classes and helpers for pose estimation backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

AnalysisFunc = Callable[[Any, Any], Tuple[Any, ...]]
DetectionFunc = Callable[[np.ndarray, Any, Tuple[Any, ...]], np.ndarray]


@dataclass
class NormalizedLandmark:
    """Minimal landmark representation independent of MediaPipe."""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0


@dataclass
class PoseLandmarks:
    """Container mimicking MediaPipe landmark structure."""

    landmark: List[NormalizedLandmark]


class PoseResults:
    """Simple results wrapper exposing pose_landmarks attribute."""

    def __init__(self, landmarks: Optional[List[NormalizedLandmark]] = None):
        self.pose_landmarks = PoseLandmarks(landmarks) if landmarks else None

    def __bool__(self) -> bool:
        return self.pose_landmarks is not None


class BasePoseModel:
    """Base context manager for pose estimation backends."""

    def __init__(
        self,
        *,
        detection_conf: float,
        tracking_conf: Optional[float] = None,
        model_config: Optional[dict] = None,
        device: Optional[str] = None,
        config: Optional[object] = None,
    ) -> None:
        self.detection_conf = float(detection_conf)
        self.tracking_conf = float(tracking_conf) if tracking_conf is not None else None
        self.model_config = model_config or {}
        self.device = device
        self.config = config

    def __enter__(self) -> "BasePoseModel":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def process_frame(
        self,
        frame: np.ndarray,
        anal_func: AnalysisFunc,
        detect_func: DetectionFunc,
    ) -> Tuple[np.ndarray, Tuple[Any, ...], bool]:
        """Evaluate a frame and return annotated output plus angles."""
        raise NotImplementedError

    def close(self) -> None:
        """Release backend resources."""
        return None