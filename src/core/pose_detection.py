"""Pose detection and annotation functions for visualizing analysis results."""

import cv2
import numpy as np
import mediapipe as mp
from ..utils.visualization import draw_colored_connection, draw_landmarks, draw_labeled_box

# Initialize mediapipe pose references
mp_pose = mp.solutions.pose
landmarks_to_draw = list(mp_pose.PoseLandmark)


def front_angle_detection(image: np.ndarray, results, angles: tuple) -> np.ndarray:
    """
    Annotate front-view image with pose analysis results.
    
    Args:
        image: The image to annotate.
        results: Pose detection results containing landmarks.
        angles: Tuple of computed angles from analysis.
    
    Returns:
        Annotated image with landmarks, angles, and posture information.
    """
    if results.pose_landmarks:
        # Draw connections
        connect_idx = [(23, 25), (25, 31), (27, 31), (24, 26), (26, 32), (28, 32), 
                      (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), 
                      (11, 12), (23, 24)]
        colors = [(255, 0, 0)] * 12 + [(0, 0, 255)] * 4
        
        for (p1, p2), color in zip(connect_idx, colors):
            draw_colored_connection(image, results, landmarks_to_draw[p1], 
                                  landmarks_to_draw[p2], color=color)

        # Draw landmarks
        landmark_idx = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 31, 32, 27, 28]
        draw_landmarks(image, results, landmark_idx)
        
        # Draw angle labels
        joint_landmarks = [landmarks_to_draw[idx].value for idx in landmark_idx]
        draw_labeled_box(image, results, joint_landmarks, angles[:-3])

        # Draw central line and deviation
        central_vertical_line_x, shoulder_mid_x = angles[-1], angles[-2]
        central_vertical_line_pixel_x = int(central_vertical_line_x * image.shape[1])
        cv2.line(image, (central_vertical_line_pixel_x, 0), 
                (central_vertical_line_pixel_x, image.shape[0]), (255, 255, 0), 1)

        # Draw middle dot
        middle_dot_x = int(shoulder_mid_x * image.shape[1])
        middle_dot_y = int((results.pose_landmarks.landmark[landmarks_to_draw[landmark_idx[0]]].y +
                           results.pose_landmarks.landmark[landmarks_to_draw[landmark_idx[1]]].y) / 2 * image.shape[0])
        cv2.circle(image, (middle_dot_x, middle_dot_y), 3, (0, 255, 0), -1)

        # Draw deviation text
        shoulder_midpoint_pos = (middle_dot_x, middle_dot_y)
        deviation_text = 'Left' if (shoulder_mid_x - central_vertical_line_x) > 0 else 'Right' if (shoulder_mid_x - central_vertical_line_x) < 0 else 'Centered'
        deviation_full_text = f'Dev: {deviation_text}'
        
        deviation_text_size = cv2.getTextSize(deviation_full_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)[0]
        deviation_box_start = (central_vertical_line_pixel_x + 5, shoulder_midpoint_pos[1] + 15 - deviation_text_size[1] - 2)
        deviation_box_end = (deviation_box_start[0] + deviation_text_size[0] + 4, shoulder_midpoint_pos[1] + 15 + 2)
        
        cv2.rectangle(image, deviation_box_start, deviation_box_end, (255, 255, 255), cv2.FILLED)
        cv2.rectangle(image, deviation_box_start, deviation_box_end, (230, 216, 173), 1)
        cv2.putText(image, deviation_full_text, 
                   (deviation_box_start[0], deviation_box_start[1] + deviation_text_size[1] + 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (139, 0, 0), 1)

    return image


def side_angle_detection(image: np.ndarray, results, angles: tuple) -> np.ndarray:
    """
    Annotate side-view image with pose analysis results.
    
    Args:
        image: The image to annotate.
        results: Pose detection results containing landmarks.
        angles: Tuple of computed angles from analysis.
    
    Returns:
        Annotated image with landmarks and angles.
    """
    if results.pose_landmarks:
        # Draw connections
        connect_idx = [(7, 11), (8, 12), (11, 23), (25, 27), (12, 24), (26, 28), 
                      (11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (27, 31), 
                      (24, 26), (28, 32)]
        
        for i, (p1, p2) in enumerate(connect_idx):
            color = (0, 165, 255) if i < 2 else ((0, 0, 255) if i < 6 else (255, 0, 0))
            draw_colored_connection(image, results, landmarks_to_draw[p1], 
                                  landmarks_to_draw[p2], color=color)

        # Draw landmarks
        landmark_idx = [11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 7, 8, 15, 16, 31, 32]
        draw_landmarks(image, results, landmark_idx)
        
        joint_landmarks = [landmarks_to_draw[idx].value for idx in landmark_idx]
        draw_labeled_box(image, results, joint_landmarks, angles[:-4])

        # Find pose direction
        left_foot_index_x = results.pose_landmarks.landmark[31].x
        right_foot_index_x = results.pose_landmarks.landmark[32].x
        left_ankle_x = results.pose_landmarks.landmark[27].x
        right_ankle_x = results.pose_landmarks.landmark[28].x
        
        direction = 1 if left_foot_index_x < left_ankle_x or right_foot_index_x < right_ankle_x else -1

        # Draw additional angle boxes for torso and ankle
        landmark_idx_extra = [23, 24, 27, 28]
        joint_landmarks_extra = [landmarks_to_draw[idx].value for idx in landmark_idx_extra]
        
        for joint_index, angle in enumerate(angles[-4:]):
            joint_landmark = results.pose_landmarks.landmark[joint_landmarks_extra[joint_index]]
            angle_text = f"{round(angle)}"
            text_x = int(joint_landmark.x * image.shape[1]) - (90 * direction)
            text_y = int(joint_landmark.y * image.shape[0])
            
            text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            box_start = (text_x - 2, text_y + 2)
            box_end = (text_x + text_size[0] + 2, text_y - text_size[1] - 2)
            
            cv2.rectangle(image, box_start, box_end, (255, 255, 255), cv2.FILLED)
            cv2.rectangle(image, box_start, box_end, (230, 216, 173), 1)
            cv2.putText(image, angle_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (139, 0, 0), 1, cv2.LINE_AA)

    return image


def balance_back_detection(image: np.ndarray, results, angles: tuple) -> np.ndarray:
    """
    Annotate back-view balance image with pose analysis results.
    
    Args:
        image: The image to annotate.
        results: Pose detection results containing landmarks.
        angles: Tuple of computed angles from analysis.
    
    Returns:
        Annotated image with landmarks and angles.
    """
    if results.pose_landmarks:
        landmark_idx = [11, 12, 13, 14, 23, 24]
        
        # Draw connections
        draw_colored_connection(image, results, landmarks_to_draw[landmark_idx[0]], 
                              landmarks_to_draw[landmark_idx[1]])
        draw_colored_connection(image, results, landmarks_to_draw[landmark_idx[2]], 
                              landmarks_to_draw[landmark_idx[3]])
        draw_colored_connection(image, results, landmarks_to_draw[landmark_idx[4]], 
                              landmarks_to_draw[landmark_idx[5]])
        draw_colored_connection(image, results, landmarks_to_draw[landmark_idx[0]], 
                              landmarks_to_draw[landmark_idx[4]], color=(0, 0, 255))
        draw_colored_connection(image, results, landmarks_to_draw[landmark_idx[1]], 
                              landmarks_to_draw[landmark_idx[5]], color=(0, 0, 255))

        draw_landmarks(image, results, landmark_idx)
        
        joint_landmarks = [landmarks_to_draw[idx].value for idx in landmark_idx[::2]]
        draw_labeled_box(image, results, joint_landmarks, angles)

    return image


def balance_test_detection(image: np.ndarray, results, angles: tuple) -> np.ndarray:
    """
    Annotate balance test image with pose analysis results.
    
    Args:
        image: The image to annotate.
        results: Pose detection results containing landmarks.
        angles: Tuple of computed deviation metrics.
    
    Returns:
        Annotated image with landmarks, vertical line, and deviation information.
    """
    should_dev, hip_dev, central_vertical_line_x, shoulder_mid_x = angles

    if results.pose_landmarks:
        landmark_idx = [11, 12, 23, 24]
        
        # Draw connections
        draw_colored_connection(image, results, landmarks_to_draw[landmark_idx[0]], 
                              landmarks_to_draw[landmark_idx[1]])
        draw_colored_connection(image, results, landmarks_to_draw[landmark_idx[2]], 
                              landmarks_to_draw[landmark_idx[3]], color=(0, 0, 255))
        draw_colored_connection(image, results, landmarks_to_draw[landmark_idx[0]], 
                              landmarks_to_draw[landmark_idx[2]], color=(0, 0, 255))
        draw_colored_connection(image, results, landmarks_to_draw[landmark_idx[1]], 
                              landmarks_to_draw[landmark_idx[3]], color=(0, 0, 255))

        # Draw central vertical line
        central_vertical_line_pixel_x = int(central_vertical_line_x * image.shape[1])
        cv2.line(image, (central_vertical_line_pixel_x, 0), 
                (central_vertical_line_pixel_x, image.shape[0]), (255, 255, 0), 1)
        
        # Draw middle dot
        middle_dot_x = int(shoulder_mid_x * image.shape[1])
        middle_dot_y = int((results.pose_landmarks.landmark[landmarks_to_draw[landmark_idx[0]]].y +
                           results.pose_landmarks.landmark[landmarks_to_draw[landmark_idx[1]]].y) / 2 * image.shape[0])
        
        draw_landmarks(image, results, landmark_idx)
        cv2.circle(image, (middle_dot_x, middle_dot_y), 3, (0, 255, 0), -1)

        shoulder_midpoint_pos = (middle_dot_x, middle_dot_y)
        
        # Draw ratio and deviation text
        shoulder_diff_text = f'Ratio: {round(should_dev, 3)}'
        deviation_text = 'Left' if should_dev > 0 else 'Right' if should_dev < 0 else 'Centered'
        deviation_full_text = f'Dev: {deviation_text}'
        
        shoulder_diff_text_size = cv2.getTextSize(shoulder_diff_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
        deviation_text_size = cv2.getTextSize(deviation_full_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
        
        # Shoulder Diff Box
        shoulder_diff_box_start = (shoulder_midpoint_pos[0] + 5, 
                                  shoulder_midpoint_pos[1] - 5 - shoulder_diff_text_size[1] - 2)
        shoulder_diff_box_end = (shoulder_diff_box_start[0] + shoulder_diff_text_size[0] + 4, 
                                shoulder_midpoint_pos[1] - 5 + 2)
        cv2.rectangle(image, shoulder_diff_box_start, shoulder_diff_box_end, (255, 255, 255), cv2.FILLED)
        cv2.rectangle(image, shoulder_diff_box_start, shoulder_diff_box_end, (230, 216, 173), 1)
        
        # Deviation Box
        deviation_box_start = (central_vertical_line_pixel_x + 5, 
                             shoulder_midpoint_pos[1] + 15 - deviation_text_size[1] - 2)
        deviation_box_end = (deviation_box_start[0] + deviation_text_size[0] + 4, 
                            shoulder_midpoint_pos[1] + 15 + 2)
        cv2.rectangle(image, deviation_box_start, deviation_box_end, (255, 255, 255), cv2.FILLED)
        cv2.rectangle(image, deviation_box_start, deviation_box_end, (230, 216, 173), 1)
        
        # Put text
        cv2.putText(image, shoulder_diff_text, 
                   (shoulder_diff_box_start[0], shoulder_diff_box_start[1] + shoulder_diff_text_size[1] + 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (139, 0, 0), 1)
        cv2.putText(image, deviation_full_text, 
                   (deviation_box_start[0], deviation_box_start[1] + deviation_text_size[1] + 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (139, 0, 0), 1)

    return image


def balance_front_detection(image: np.ndarray, results, angles: tuple) -> np.ndarray:
    """
    Annotate front-view balance image with pose analysis results.
    
    Args:
        image: The image to annotate.
        results: Pose detection results containing landmarks.
        angles: Tuple of computed angles from analysis.
    
    Returns:
        Annotated image with landmarks and angles.
    """
    if results.pose_landmarks:
        landmark_idx = [7, 8, 11, 12, 23, 24, 25, 26, 27, 28]
        
        # Draw horizontal connections
        for i in range(0, len(landmark_idx), 2):
            draw_colored_connection(image, results, landmarks_to_draw[landmark_idx[i]], 
                                  landmarks_to_draw[landmark_idx[i+1]])
        
        # Draw vertical connections (left side)
        for i in range(2, len(landmark_idx)-2, 2):
            draw_colored_connection(image, results, landmarks_to_draw[landmark_idx[i]], 
                                  landmarks_to_draw[landmark_idx[i+2]], color=(0, 0, 255))
        
        # Draw vertical connections (right side)
        for i in range(3, len(landmark_idx)-2, 2):
            draw_colored_connection(image, results, landmarks_to_draw[landmark_idx[i]], 
                                  landmarks_to_draw[landmark_idx[i+2]], color=(0, 0, 255))

        draw_colored_connection(image, results, landmarks_to_draw[landmark_idx[0]], 
                              landmarks_to_draw[landmark_idx[2]], color=(0, 0, 255))
        draw_colored_connection(image, results, landmarks_to_draw[landmark_idx[1]], 
                              landmarks_to_draw[landmark_idx[3]], color=(0, 0, 255))
        
        draw_landmarks(image, results, landmark_idx)
        
        joint_landmarks = [landmarks_to_draw[idx].value for idx in landmark_idx[::2]]
        draw_labeled_box(image, results, joint_landmarks, angles)

    return image


def balance_side_detection(image: np.ndarray, results, angles: tuple) -> np.ndarray:
    """
    Annotate side-view balance image with pose analysis results.
    
    Args:
        image: The image to annotate.
        results: Pose detection results containing landmarks.
        angles: Tuple of computed angles from analysis.
    
    Returns:
        Annotated image with neck angle and posture information.
    """
    if results.pose_landmarks:
        landmark_idx = [7, 11, 23, 25, 27]
        
        # Draw connections
        draw_colored_connection(image, results, landmarks_to_draw[7], landmarks_to_draw[11])
        draw_colored_connection(image, results, landmarks_to_draw[11], 
                              landmarks_to_draw[23], color=(0, 0, 255))
        draw_colored_connection(image, results, landmarks_to_draw[23], landmarks_to_draw[25])
        draw_colored_connection(image, results, landmarks_to_draw[25], 
                              landmarks_to_draw[27], color=(0, 0, 255))

        # Draw vertical line through left hip
        left_hip_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_hip_x = int(left_hip_landmark.x * image.shape[1])
        cv2.line(image, (left_hip_x, 0), (left_hip_x, image.shape[0]), (255, 255, 0), thickness=1)

        draw_landmarks(image, results, landmark_idx)
        
        joint_landmarks = [landmarks_to_draw[11].value, landmarks_to_draw[25].value]
        draw_labeled_box(image, results, joint_landmarks, angles)

    return image
