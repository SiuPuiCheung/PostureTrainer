"""MediaPipe pose detection model wrapper."""

# Copyright (c) 2025 Siu Pui Cheung
# Licensed under the MIT License

import cv2
import numpy as np
import mediapipe as mp

from .base_pose_model import BasePoseModel, PoseLandmark, PoseResults


class MediaPipePoseModel(BasePoseModel):
    """MediaPipe implementation of pose detection."""

    def __init__(self, confidence: float = 0.5, use_gpu: bool = True):
        """
        Initialize MediaPipe pose model.

        Args:
            confidence: Detection confidence threshold (0-1)
            use_gpu: Whether to use GPU acceleration (MediaPipe uses CPU by default)
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        super().__init__(confidence, use_gpu)

    def _initialize_model(self):
        """Initialize MediaPipe Pose model."""
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # 0, 1, or 2. Higher = more accurate but slower
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=self.confidence,
            min_tracking_confidence=self.confidence,
        )

    def process(self, image: np.ndarray) -> PoseResults:
        """
        Process an image and detect pose landmarks.

        Args:
            image: Input image (BGR format)

        Returns:
            PoseResults object with landmarks
        """
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image
        results = self.pose.process(image_rgb)

        # Convert landmarks to unified format
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append(
                    PoseLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z, visibility=landmark.visibility
                    )
                )
            return PoseResults(landmarks=landmarks)

        return PoseResults(landmarks=None)

    def get_raw_results(self, image: np.ndarray):
        """
        Get raw MediaPipe results for backward compatibility.

        Args:
            image: Input image (BGR format)

        Returns:
            MediaPipe pose results object
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.pose.process(image_rgb)

    def close(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, "pose"):
            self.pose.close()
