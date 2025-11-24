"""Abstract base class for pose detection models."""
# Copyright (c) 2025 Siu Pui Cheung
# Licensed under the MIT License

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import numpy as np


class PoseLandmark:
    """Unified pose landmark representation."""
    
    def __init__(self, x: float, y: float, z: float, visibility: float = 1.0):
        """
        Initialize pose landmark.
        
        Args:
            x: X coordinate (normalized 0-1)
            y: Y coordinate (normalized 0-1)
            z: Z coordinate (depth)
            visibility: Visibility score (0-1)
        """
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class PoseResults:
    """Unified pose detection results."""
    
    def __init__(self, landmarks: Optional[list] = None, annotated_image: Optional[np.ndarray] = None):
        """
        Initialize pose results.
        
        Args:
            landmarks: List of PoseLandmark objects (33 landmarks for body pose)
            annotated_image: Annotated image with pose overlay
        """
        self.pose_landmarks = landmarks
        self.annotated_image = annotated_image


class BasePoseModel(ABC):
    """Abstract base class for pose detection models."""
    
    # MediaPipe pose landmark indices (33 landmarks)
    # This is the standard we'll use across all models
    LANDMARK_NAMES = [
        'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER',
        'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
        'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
        'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
        'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
        'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
        'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
        'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
        'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
    ]
    
    def __init__(self, confidence: float = 0.5, use_gpu: bool = True):
        """
        Initialize pose model.
        
        Args:
            confidence: Detection confidence threshold (0-1)
            use_gpu: Whether to use GPU acceleration if available
        """
        self.confidence = confidence
        self.use_gpu = use_gpu
        self._initialize_model()
    
    @abstractmethod
    def _initialize_model(self):
        """Initialize the specific model implementation."""
        pass
    
    @abstractmethod
    def process(self, image: np.ndarray) -> PoseResults:
        """
        Process an image and detect pose landmarks.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            PoseResults object with landmarks and annotated image
        """
        pass
    
    @abstractmethod
    def close(self):
        """Clean up model resources."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
