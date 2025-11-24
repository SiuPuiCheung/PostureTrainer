"""Pose estimation backends for MediaPipe and YOLOv11."""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

try:  # pragma: no cover - torch imported lazily
    import torch
except Exception:  # pragma: no cover - torch optional in import time
    torch = None

try:  # pragma: no cover - ultralytics may be optional at runtime
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - handled during estimator creation
    YOLO = None  # type: ignore

AnalysisFunc = Any
DetectionFunc = Any
PoseResults = Tuple[np.ndarray, Tuple[Any, ...], bool]


@dataclass(frozen=True)
class PoseModelConfig:
    """Runtime configuration for a pose model option."""

    id: str
    name: str
    type: str
    weights: Optional[str] = None
    devices: Optional[Iterable[str]] = None
    extra: Optional[Dict[str, Any]] = None


class PoseEstimatorError(RuntimeError):
    """Raised when a pose estimator cannot be constructed."""


class BasePoseEstimator:
    """Abstract base class for pose estimators."""

    def __init__(self, detection_conf: float, tracking_conf: Optional[float] = None):
        self.detection_conf = float(detection_conf)
        self.tracking_conf = float(tracking_conf if tracking_conf is not None else detection_conf)

    def __enter__(self) -> "BasePoseEstimator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # pragma: no cover - resource cleanup
        self.close()

    def process(
        self,
        frame: np.ndarray,
        anal_func: AnalysisFunc,
        detect_func: DetectionFunc,
    ) -> PoseResults:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - optional override
        """Release underlying resources."""


class MediaPipePoseEstimator(BasePoseEstimator):
    """MediaPipe pose backend wrapper."""

    def __init__(self, detection_conf: float, tracking_conf: Optional[float] = None):
        super().__init__(detection_conf, tracking_conf)
        self._pose = mp.solutions.pose.Pose(
            min_detection_confidence=self.detection_conf,
            min_tracking_confidence=self.tracking_conf,
        )

    def process(
        self,
        frame: np.ndarray,
        anal_func: AnalysisFunc,
        detect_func: DetectionFunc,
    ) -> PoseResults:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(image_rgb)

        if results.pose_landmarks:
            angles = anal_func(results.pose_landmarks.landmark, mp.solutions.pose)
            annotated = detect_func(frame.copy(), results, angles)
            return annotated, tuple(angles), True

        return frame.copy(), tuple(), False

    def close(self) -> None:
        if self._pose:
            self._pose.close()
            self._pose = None


class YOLOPoseEstimator(BasePoseEstimator):
    """YOLOv11 pose backend using Ultralytics models."""

    _YOLO_TO_MP_MAP: Dict[int, int] = {
        mp.solutions.pose.PoseLandmark.NOSE.value: 0,
        mp.solutions.pose.PoseLandmark.LEFT_EYE.value: 1,
        mp.solutions.pose.PoseLandmark.RIGHT_EYE.value: 2,
        mp.solutions.pose.PoseLandmark.LEFT_EAR.value: 3,
        mp.solutions.pose.PoseLandmark.RIGHT_EAR.value: 4,
        mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value: 5,
        mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value: 6,
        mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value: 7,
        mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value: 8,
        mp.solutions.pose.PoseLandmark.LEFT_WRIST.value: 9,
        mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value: 10,
        mp.solutions.pose.PoseLandmark.LEFT_HIP.value: 11,
        mp.solutions.pose.PoseLandmark.RIGHT_HIP.value: 12,
        mp.solutions.pose.PoseLandmark.LEFT_KNEE.value: 13,
        mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value: 14,
        mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value: 15,
        mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value: 16,
        mp.solutions.pose.PoseLandmark.LEFT_HEEL.value: 15,
        mp.solutions.pose.PoseLandmark.RIGHT_HEEL.value: 16,
        mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX.value: 15,
        mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX.value: 16,
    }

    def __init__(
        self,
        weights_path: str,
        detection_conf: float,
        *,
        tracking_conf: Optional[float] = None,
        device: Optional[str] = None,
    ):
        if YOLO is None:
            raise PoseEstimatorError(
                "Ultralytics YOLO package is not installed. Install 'ultralytics' to use YOLO models."
            )

        super().__init__(detection_conf, tracking_conf)
        self._model = YOLO(weights_path)
        self._device = self._resolve_device(device)

    def process(
        self,
        frame: np.ndarray,
        anal_func: AnalysisFunc,
        detect_func: DetectionFunc,
    ) -> PoseResults:
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

        mp_results = self._build_mediapipe_results(frame.shape, xy, conf)
        angles = anal_func(mp_results.pose_landmarks.landmark, mp.solutions.pose)
        annotated = detect_func(frame.copy(), mp_results, angles)
        return annotated, tuple(angles), True

    @staticmethod
    def _resolve_device(device: Optional[str]) -> Optional[str]:
        if not device or device.lower() in {"auto", "default"}:
            return None
        if device.lower() == "cpu":
            return "cpu"
        if device.lower() == "gpu":
            if torch is not None and torch.cuda.is_available():  # pragma: no cover - hardware dependent
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

    def _build_mediapipe_results(
        self,
        frame_shape: Sequence[int],
        keypoints_xy: np.ndarray,
        keypoint_conf: Optional[np.ndarray],
    ) -> SimpleNamespace:
        height, width = frame_shape[:2]
        mp_landmarks = [
            landmark_pb2.NormalizedLandmark(x=0.0, y=0.0, z=0.0, visibility=0.0)
            for _ in range(len(mp.solutions.pose.PoseLandmark))
        ]

        for mp_index, yolo_index in self._YOLO_TO_MP_MAP.items():
            if yolo_index >= len(keypoints_xy):
                continue
            x_px, y_px = keypoints_xy[yolo_index]
            landmark = mp_landmarks[mp_index]
            landmark.x = float(np.clip(x_px / width, 0.0, 1.0))
            landmark.y = float(np.clip(y_px / height, 0.0, 1.0))
            if keypoint_conf is not None and yolo_index < len(keypoint_conf):
                landmark.visibility = float(np.clip(keypoint_conf[yolo_index], 0.0, 1.0))
            else:
                landmark.visibility = 1.0

        pose_landmarks = SimpleNamespace(landmark=mp_landmarks)
        return SimpleNamespace(pose_landmarks=pose_landmarks)

    def close(self) -> None:  # pragma: no cover - no explicit release required
        """Ultralytics models manage their own resources."""


def _coerce_option(option: Dict[str, Any]) -> PoseModelConfig:
    if 'id' not in option or 'type' not in option:
        raise PoseEstimatorError("Each pose model option must define 'id' and 'type'.")
    devices = option.get('devices')
    if devices is not None:
        if isinstance(devices, str):
            devices = [devices]
        elif not isinstance(devices, Iterable):
            raise PoseEstimatorError(
                "Pose model 'devices' must be an iterable of strings if provided."
            )
    return PoseModelConfig(
        id=str(option['id']),
        name=str(option.get('name', option['id'])),
        type=str(option['type']).lower(),
        weights=option.get('weights'),
        devices=list(devices) if devices is not None else None,
        extra={k: v for k, v in option.items() if k not in {'id', 'name', 'type', 'weights', 'devices'}},
    )


def _resolve_weights_path(weights: Optional[str], config: Any) -> Optional[str]:
    if not weights:
        return None
    if os.path.isabs(weights):
        return weights
    weights_dir = getattr(config, 'get_pose_weights_dir', None)
    if callable(weights_dir):
        base_dir = weights_dir()
    else:
        base_dir = 'models'
    os.makedirs(base_dir, exist_ok=True)
    candidate = os.path.join(base_dir, weights)
    if os.path.exists(candidate):
        return candidate
    return weights


def create_pose_estimator(
    model_id: str,
    config: Any,
    *,
    detection_conf: float,
    tracking_conf: Optional[float] = None,
    device: Optional[str] = None,
) -> BasePoseEstimator:
    """Factory that builds a pose estimator for the requested model id."""
    options = []
    if hasattr(config, 'get_pose_model_options'):
        options = config.get_pose_model_options()
    elif hasattr(config, 'pose_models'):
        pose_cfg = config.pose_models
        if isinstance(pose_cfg, dict):
            options = pose_cfg.get('options', [])

    selected_option = None
    for option in options:
        if isinstance(option, dict) and option.get('id') == model_id:
            selected_option = _coerce_option(option)
            break

    if selected_option is None:
        raise PoseEstimatorError(f"Pose model '{model_id}' is not defined in configuration.")

    if device is None and hasattr(config, 'get_default_pose_device'):
        device = config.get_default_pose_device()

    if selected_option.devices and device not in {None, 'auto'}:
        normalized = device.lower()
        allowed = {d.lower() for d in selected_option.devices}
        if normalized not in allowed:
            raise PoseEstimatorError(
                f"Device '{device}' is not supported by pose model '{selected_option.id}'."
            )

    if selected_option.type == 'mediapipe':
        return MediaPipePoseEstimator(detection_conf, tracking_conf)

    if selected_option.type == 'yolo':
        weights_path = _resolve_weights_path(selected_option.weights, config)
        if weights_path is None:
            raise PoseEstimatorError(
                f"YOLO pose model '{selected_option.id}' requires a 'weights' path in the configuration."
            )
        return YOLOPoseEstimator(
            weights_path=weights_path,
            detection_conf=detection_conf,
            tracking_conf=tracking_conf,
            device=device,
        )

    raise PoseEstimatorError(
        f"Unsupported pose model type '{selected_option.type}' for model '{selected_option.id}'."
    )
