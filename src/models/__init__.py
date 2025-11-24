"""Pose estimation model registry and helpers."""

from __future__ import annotations

from typing import Dict, Type

from .base import BasePoseModel
from .mediapipe_model import MediaPipePoseModel
from .yolo_model import YOLOv11PoseModel

# Registry of available pose estimation backends
MODEL_REGISTRY: Dict[str, Type[BasePoseModel]] = {
    "mediapipe": MediaPipePoseModel,
    "yolo11": YOLOv11PoseModel,
}


def get_model_registry() -> Dict[str, Type[BasePoseModel]]:
    """Return mapping of model identifiers to their classes."""
    return MODEL_REGISTRY.copy()


def create_pose_model(
    model_id: str,
    *,
    detection_conf: float,
    tracking_conf: float | None = None,
    model_config: dict | None = None,
    device: str | None = None,
    config: object | None = None,
) -> BasePoseModel:
    """Instantiate a pose model by identifier."""
    model_cls = MODEL_REGISTRY.get(model_id)
    if model_cls is None:
        raise ValueError(f"Pose model '{model_id}' is not registered")
    return model_cls(
        detection_conf=detection_conf,
        tracking_conf=tracking_conf,
        model_config=model_config,
        device=device,
        config=config,
    )
