"""MediaPipe-based pose estimation backend."""

from __future__ import annotations

from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np

from .base import BasePoseModel, AnalysisFunc, DetectionFunc


class MediaPipePoseModel(BasePoseModel):
    """Pose estimation using MediaPipe's lightweight pose solution."""

    MODEL_ID = "mediapipe"
    DISPLAY_NAME = "MediaPipe Pose"

    def __enter__(self) -> "MediaPipePoseModel":
        tracking_conf = self.tracking_conf if self.tracking_conf is not None else self.detection_conf
        self._pose = mp.solutions.pose.Pose(
            min_detection_confidence=self.detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        return self

    def process_frame(
        self,
        frame: np.ndarray,
        anal_func: AnalysisFunc,
        detect_func: DetectionFunc,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, ...], bool]:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(image_rgb)

        if results.pose_landmarks:
            angles = anal_func(results.pose_landmarks.landmark, mp.solutions.pose)
            annotated = detect_func(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), results, angles)
            return annotated, tuple(angles), True

        return frame.copy(), tuple(), False

    def close(self) -> None:
        if hasattr(self, "_pose") and self._pose:
            self._pose.close()
            self._pose = None
