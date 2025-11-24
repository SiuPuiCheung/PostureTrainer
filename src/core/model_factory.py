"""Factory for creating pose detection models."""

# Copyright (c) 2025 Siu Pui Cheung
# Licensed under the MIT License

from .base_pose_model import BasePoseModel
from .mediapipe_model import MediaPipePoseModel


def create_pose_model(
    model_type: str = "mediapipe", confidence: float = 0.5, use_gpu: bool = True, **kwargs
) -> BasePoseModel:
    """
    Factory function to create pose detection models.

    Args:
        model_type: Type of model ('mediapipe' or 'yolov11')
        confidence: Detection confidence threshold (0-1)
        use_gpu: Whether to use GPU acceleration if available
        **kwargs: Additional model-specific arguments
            - model_size: For YOLOv11, model size ('n', 's', 'm', 'l', 'x')

    Returns:
        BasePoseModel instance

    Raises:
        ValueError: If model_type is not supported
        ImportError: If required dependencies are not installed
    """
    model_type = model_type.lower()

    if model_type == "mediapipe":
        return MediaPipePoseModel(confidence=confidence, use_gpu=use_gpu)

    elif model_type == "yolov11" or model_type == "yolo":
        try:
            from .yolo_model import YOLOv11PoseModel

            model_size = kwargs.get("model_size", "n")
            return YOLOv11PoseModel(confidence=confidence, use_gpu=use_gpu, model_size=model_size)
        except ImportError as e:
            raise ImportError(
                f"YOLOv11 dependencies not available: {e}\n"
                "Install with: pip install ultralytics torch torchvision"
            )

    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. " "Supported types: 'mediapipe', 'yolov11'"
        )


def get_available_models() -> list:
    """
    Get list of available pose detection models.

    Returns:
        List of available model names
    """
    models = ["mediapipe"]

    try:
        from .yolo_model import YOLO_AVAILABLE

        if YOLO_AVAILABLE:
            models.append("yolov11")
    except ImportError:
        pass

    return models


def check_gpu_available() -> bool:
    """
    Check if GPU acceleration is available.

    Returns:
        True if CUDA is available, False otherwise
    """
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
