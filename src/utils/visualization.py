"""Visualization utilities for pose detection."""
# Copyright (c) 2025 Siu Pui Cheung
# Licensed under the MIT License

import cv2
import numpy as np
from typing import List, Tuple


def draw_colored_connection(image: np.ndarray, 
                           results, 
                           start_idx: int, 
                           end_idx: int, 
                           color: Tuple[int, int, int] = (255, 0, 0), 
                           thickness: int = 2) -> None:
    """
    Draw a colored line between two specified landmarks on the image.
    
    Args:
        image: The image on which to draw.
        results: The pose detection results containing landmarks.
        start_idx: The index of the start landmark.
        end_idx: The index of the end landmark.
        color: The color of the line (default red).
        thickness: The thickness of the line (default 2).
    """
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        start, end = landmarks[start_idx], landmarks[end_idx]
        cv2.line(
            image, 
            (int(start.x * image.shape[1]), int(start.y * image.shape[0])),
            (int(end.x * image.shape[1]), int(end.y * image.shape[0])), 
            color, 
            thickness
        )


def draw_landmarks(image: np.ndarray, 
                  results, 
                  landmark_indices: List[int], 
                  color: Tuple[int, int, int] = (255, 255, 255), 
                  radius: int = 3) -> None:
    """
    Draw circles on specified landmarks.
    
    Args:
        image: The image on which to draw.
        results: The pose detection results containing landmarks.
        landmark_indices: Indices of the landmarks to draw.
        color: The color of the circles (default white).
        radius: The radius of the circles (default 3).
    """
    for idx in landmark_indices:
        landmark_point = results.pose_landmarks.landmark[idx]
        pos = (int(landmark_point.x * image.shape[1]), int(landmark_point.y * image.shape[0]))
        cv2.circle(image, pos, radius, color, -1)


def draw_labeled_box(image: np.ndarray, 
                    results, 
                    joint_landmarks: List[int], 
                    angles: List[float], 
                    padding: int = 3, 
                    font_scale: float = 0.35, 
                    font_thickness: int = 1,
                    box_color: Tuple[int, int, int] = (255, 255, 255), 
                    text_color: Tuple[int, int, int] = (139, 0, 0), 
                    edge_color: Tuple[int, int, int] = (230, 216, 173)) -> None:
    """
    Draw labeled boxes with angle values near specified joints.
    
    Args:
        image: The image on which to draw.
        results: The pose detection results containing landmarks.
        joint_landmarks: Indices of the joints near which to draw labeled boxes.
        angles: List of angle values corresponding to the joints.
        padding: Padding for text box.
        font_scale: Font scale for text.
        font_thickness: Font thickness for text.
        box_color: Background color of the box.
        text_color: Color of the text.
        edge_color: Color of the box's edge.
    """
    for joint_index, angle in enumerate(angles):
        joint = results.pose_landmarks.landmark[joint_landmarks[joint_index]]
        angle_text = f"{round(angle)}"
        pos = (int(joint.x * image.shape[1]) + 10, int(joint.y * image.shape[0]))
        
        text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        box_start = (pos[0] - padding, pos[1] + padding)
        box_end = (pos[0] + text_size[0] + padding, pos[1] - text_size[1] - padding)
        
        cv2.rectangle(image, box_start, box_end, box_color, cv2.FILLED)
        cv2.rectangle(image, box_start, box_end, edge_color, 1)
        cv2.putText(image, angle_text, (pos[0], pos[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 
                   font_thickness, cv2.LINE_AA)
