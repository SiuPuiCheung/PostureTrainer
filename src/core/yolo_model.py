"""YOLOv11 pose detection model wrapper."""
# Copyright (c) 2025 Siu Pui Cheung
# Licensed under the MIT License

import cv2
import numpy as np
import torch
from typing import Optional
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from .base_pose_model import BasePoseModel, PoseLandmark, PoseResults


class YOLOv11PoseModel(BasePoseModel):
    """YOLOv11 implementation of pose detection."""
    
    # YOLO pose keypoint indices to MediaPipe landmark mapping
    # YOLO has 17 keypoints, we map them to MediaPipe's 33 landmarks
    YOLO_TO_MEDIAPIPE_MAP = {
        0: 0,   # nose -> nose
        1: 2,   # left_eye -> left_eye
        2: 5,   # right_eye -> right_eye
        3: 7,   # left_ear -> left_ear
        4: 8,   # right_ear -> right_ear
        5: 11,  # left_shoulder -> left_shoulder
        6: 12,  # right_shoulder -> right_shoulder
        7: 13,  # left_elbow -> left_elbow
        8: 14,  # right_elbow -> right_elbow
        9: 15,  # left_wrist -> left_wrist
        10: 16, # right_wrist -> right_wrist
        11: 23, # left_hip -> left_hip
        12: 24, # right_hip -> right_hip
        13: 25, # left_knee -> left_knee
        14: 26, # right_knee -> right_knee
        15: 27, # left_ankle -> left_ankle
        16: 28, # right_ankle -> right_ankle
    }
    
    def __init__(self, confidence: float = 0.5, use_gpu: bool = True, model_size: str = "n"):
        """
        Initialize YOLOv11 pose model.
        
        Args:
            confidence: Detection confidence threshold (0-1)
            use_gpu: Whether to use GPU acceleration if available
            model_size: Model size ('n', 's', 'm', 'l', 'x') - n=nano, s=small, m=medium, l=large, x=xlarge
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics is not installed. Install it with: pip install ultralytics"
            )
        
        self.model_size = model_size
        self.model = None
        super().__init__(confidence, use_gpu)
    
    def _initialize_model(self):
        """Initialize YOLOv11 Pose model."""
        # Determine device
        if self.use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        # Load YOLOv11 pose model (will auto-download if not present)
        model_name = f"yolo11{self.model_size}-pose.pt"
        print(f"Loading YOLOv11 pose model: {model_name} on {self.device}")
        
        try:
            self.model = YOLO(model_name)
            self.model.to(self.device)
            print(f"✓ Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise
    
    def process(self, image: np.ndarray) -> PoseResults:
        """
        Process an image and detect pose landmarks.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            PoseResults object with landmarks
        """
        # Run inference
        results = self.model(image, conf=self.confidence, verbose=False)
        
        if len(results) == 0 or results[0].keypoints is None:
            return PoseResults(landmarks=None)
        
        # Get keypoints from first detected person
        keypoints = results[0].keypoints
        if keypoints.xy is None or len(keypoints.xy) == 0:
            return PoseResults(landmarks=None)
        
        # Get image dimensions for normalization
        height, width = image.shape[:2]
        
        # Extract keypoints (xy coordinates and confidence)
        kpts_xy = keypoints.xy[0].cpu().numpy()  # Shape: (17, 2)
        kpts_conf = keypoints.conf[0].cpu().numpy() if keypoints.conf is not None else np.ones(17)  # Shape: (17,)
        
        # Create 33 landmarks (MediaPipe format), initialize with zeros
        landmarks = [PoseLandmark(0, 0, 0, 0) for _ in range(33)]
        
        # Map YOLO keypoints to MediaPipe landmarks
        for yolo_idx, mp_idx in self.YOLO_TO_MEDIAPIPE_MAP.items():
            if yolo_idx < len(kpts_xy):
                x, y = kpts_xy[yolo_idx]
                conf = float(kpts_conf[yolo_idx])
                
                # Normalize coordinates to 0-1 range
                x_norm = float(x / width)
                y_norm = float(y / height)
                
                landmarks[mp_idx] = PoseLandmark(
                    x=x_norm,
                    y=y_norm,
                    z=0.0,  # YOLO doesn't provide depth
                    visibility=conf
                )
        
        # Interpolate missing landmarks for better compatibility
        # For now, we mark missing ones with low visibility
        # You could add interpolation logic here if needed
        
        return PoseResults(landmarks=landmarks)
    
    def get_annotated_image(self, image: np.ndarray) -> np.ndarray:
        """
        Get image annotated with YOLO pose overlay.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Annotated image
        """
        results = self.model(image, conf=self.confidence, verbose=False)
        if len(results) > 0:
            return results[0].plot()
        return image
    
    def close(self):
        """Clean up YOLO resources."""
        if self.model is not None:
            # Clear CUDA cache if using GPU
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            self.model = None
