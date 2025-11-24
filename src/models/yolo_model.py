"""YOLOv11-based pose estimation backend using Ultralytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

try:  # pragma: no cover - optional dependency at import time
    import torch
except Exception:  # pragma: no cover - torch is optional for CPU inference
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency at import time
    from ultralytics import YOLO
except Exception:  # pragma: no cover - handled during model construction
    YOLO = None  # type: ignore

from .base import BasePoseModel, NormalizedLandmark, PoseResults, AnalysisFunc, DetectionFunc


class YOLOv11PoseModel(BasePoseModel):
    """Pose estimation using Ultralytics YOLOv11 pose weights."""

    MODEL_ID = "yolo11"
    DISPLAY_NAME = "YOLOv11 Pose"

    # Map MediaPipe-style landmark indices to YOLO keypoint indices.
    _POSTURE_TO_YOLO: Dict[int, int] = {
        0: 0,
        1: 1,
        2: 1,
        3: 1,
        4: 2,
        5: 2,
        6: 2,
        7: 3,
        8: 4,
        9: 0,
        10: 0,
        11: 5,
        12: 6,
        13: 7,
        14: 8,
        15: 9,
        16: 10,
        17: 9,
        18: 10,
        19: 9,
        20: 10,
        21: 9,
        22: 10,
        23: 11,
        24: 12,
        25: 13,
        26: 14,
        27: 15,
        28: 16,
        29: 15,
        30: 16,
        31: 15,
        32: 16,
    }

    def __enter__(self) -> "YOLOv11PoseModel":
        if YOLO is None:
            raise RuntimeError("Ultralytics package is required for YOLO models. Install 'ultralytics'.")

        weights_path = self._resolve_weights_path()
        self._model = YOLO(weights_path)
        self._device = self._resolve_device(self.device)
        return self

    def process_frame(
        self,
        frame: np.ndarray,
        anal_func: AnalysisFunc,
        detect_func: DetectionFunc,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, ...], bool]:
        prediction = self._model.predict(
            source=frame,
            conf=self.detection_conf,
            verbose=False,
            device=self._device,
        )

        if not prediction:
            return frame.copy(), tuple(), False

        result = prediction[0]
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
            return frame.copy(), tuple(), False

        instance_index = self._select_instance(keypoints)
        if instance_index is None:
            return frame.copy(), tuple(), False

        xy = keypoints.xy[instance_index]
        conf = keypoints.conf[instance_index] if keypoints.conf is not None else None

        if hasattr(xy, "cpu"):
            xy = xy.cpu().numpy()
        else:
            xy = np.asarray(xy)

        if conf is not None:
            if hasattr(conf, "cpu"):
                conf = conf.cpu().numpy()
            else:
                conf = np.asarray(conf)

        pose_results = self._build_results(frame.shape, xy, conf)
        if not pose_results.pose_landmarks:
            return frame.copy(), tuple(), False

        angles = anal_func(pose_results.pose_landmarks.landmark, mp.solutions.pose)
        annotated = detect_func(frame.copy(), pose_results, angles)
        return annotated, tuple(angles), True

    def close(self) -> None:
        self._model = None
        self._device = None

    def _resolve_weights_path(self) -> str:
        weights_name = self.model_config.get("weights", "yolo11n-pose.pt")
        weights_dir = self._weights_directory()
        candidate = weights_dir / weights_name
        if candidate.exists():
            return str(candidate)
        return weights_name

    def _weights_directory(self) -> Path:
        base_dir = self.config.get_pose_weights_dir() if hasattr(self.config, "get_pose_weights_dir") else "models"
        base_path = Path(base_dir)
        if not base_path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            base_path = project_root / base_path
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path

    @staticmethod
    def _resolve_device(device: Optional[str]) -> Optional[str]:
        if not device or device.lower() in {"auto", "default"}:
            return None
        if device.lower() == "cpu":
            return "cpu"
        if device.lower() == "gpu":
            if torch is not None and torch.cuda.is_available():  # pragma: no cover - hardware specific
                return "cuda:0"
            return "cpu"
        return device

    @staticmethod
    def _select_instance(keypoints: Any) -> Optional[int]:
        xy = keypoints.xy
        if xy is None or len(xy) == 0:
            return None
        if len(xy) == 1:
            return 0
        conf = keypoints.conf
        if conf is not None and len(conf) == len(xy):
            if hasattr(conf, "cpu"):
                conf = conf.cpu().numpy()
            else:
                conf = np.asarray(conf)
            scores = conf.mean(axis=1)
            return int(np.argmax(scores))
        return 0

    def _build_results(
        self,
        frame_shape: Tuple[int, int, int],
        keypoints_xy: np.ndarray,
        keypoint_conf: Optional[np.ndarray],
    ) -> PoseResults:
        height, width = frame_shape[:2]
        landmarks: List[NormalizedLandmark] = [
            NormalizedLandmark(x=0.0, y=0.0, z=0.0, visibility=0.0) for _ in range(33)
        ]

        for posture_idx, yolo_idx in self._POSTURE_TO_YOLO.items():
            if yolo_idx >= len(keypoints_xy):
                continue
            x_px, y_px = keypoints_xy[yolo_idx]
            visibility = 1.0
            if keypoint_conf is not None and yolo_idx < len(keypoint_conf):
                visibility = float(np.clip(keypoint_conf[yolo_idx], 0.0, 1.0))
            landmarks[posture_idx] = NormalizedLandmark(
                x=float(np.clip(x_px / width, 0.0, 1.0)),
                y=float(np.clip(y_px / height, 0.0, 1.0)),
                visibility=visibility,
            )

        return PoseResults(landmarks)
