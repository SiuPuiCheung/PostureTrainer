"""Utility functions for pose estimation calculations."""
# Copyright (c) 2025 Siu Pui Cheung
# Licensed under the MIT License

import numpy as np
from typing import Tuple


def get_point(landmarks, landmark) -> Tuple[float, float]:
    """
    Retrieve the (x, y) coordinates of a specified landmark.
    
    Args:
        landmarks: The list or array of landmarks from which coordinates are extracted.
        landmark: The specific landmark (enum or index) for which coordinates are requested.
    
    Returns:
        A tuple (x, y) representing the coordinates of the specified landmark.
    """
    return landmarks[landmark.value].x, landmarks[landmark.value].y


def calculate_angle(a: Tuple[float, float], 
                   b: Tuple[float, float], 
                   c: Tuple[float, float]) -> float:
    """
    Calculate the angle formed by three points.
    
    This function is essential for posture analysis, where calculating the angles 
    between joints (represented by points) is necessary to determine posture correctness.
    
    Args:
        a, b, c: Coordinates of the points forming the angle (each as an (x, y) tuple).
    
    Returns:
        The calculated angle in degrees, normalized to the range [0, 180].
    """
    # Convert points to numpy arrays for easier mathematical operations
    a, b, c = np.array(a), np.array(b), np.array(c)
    
    # Calculate the radians between the points
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    
    # Convert radians to degrees and normalize the angle
    angle = np.degrees(radians)
    angle = (angle + 360) % 360
    if angle > 180:
        angle -= 360
    
    return abs(angle)
