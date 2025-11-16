"""Pose analysis functions for different posture evaluation scenarios."""

from typing import Tuple, List
import mediapipe as mp
from ..utils.geometry import get_point, calculate_angle

# Initialize mediapipe pose references
mp_pose = mp.solutions.pose
pose_landmark = mp.solutions.pose.PoseLandmark
landmarks_to_draw = list(mp_pose.PoseLandmark)


def front_angle_analysis(landmarks, mp_pose) -> Tuple:
    """
    Analyze front-view squat posture based on landmarks.
    
    Args:
        landmarks: List of detected pose landmarks.
        mp_pose: Mediapipe pose module for landmark references.
    
    Returns:
        Tuple containing angles for shoulders, elbows, wrists, hips, knees, 
        ankles, and shoulder deviation metrics.
    """
    # Get points
    left_shoulder, left_elbow, left_wrist, left_index = map(
        lambda lm: get_point(landmarks, lm),
        [landmarks_to_draw[11], landmarks_to_draw[13], landmarks_to_draw[15], landmarks_to_draw[19]]
    )
    left_hip, left_knee, left_ankle, left_foot_index = map(
        lambda lm: get_point(landmarks, lm),
        [landmarks_to_draw[23], landmarks_to_draw[25], landmarks_to_draw[27], landmarks_to_draw[31]]
    )
    
    right_shoulder, right_elbow, right_wrist, right_index = map(
        lambda lm: get_point(landmarks, lm),
        [landmarks_to_draw[12], landmarks_to_draw[14], landmarks_to_draw[16], landmarks_to_draw[20]]
    )
    right_hip, right_knee, right_ankle, right_foot_index = map(
        lambda lm: get_point(landmarks, lm),
        [landmarks_to_draw[24], landmarks_to_draw[26], landmarks_to_draw[28], landmarks_to_draw[32]]
    )

    # Calculate angles
    left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
    right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
    
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    left_wrist_angle = calculate_angle(left_index, left_wrist, left_elbow)
    right_wrist_angle = calculate_angle(right_index, right_wrist, right_elbow)
    
    left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    
    left_ankle_angle = calculate_angle(left_ankle, left_foot_index, left_knee)
    right_ankle_angle = calculate_angle(right_ankle, right_foot_index, right_knee)
    
    # Calculate midpoints and deviations
    shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
    central_vertical_line_x = (left_hip[0] + right_hip[0]) / 2
    shoulder_dev = (shoulder_mid_x - central_vertical_line_x) / abs(left_shoulder[0] - right_shoulder[0])
    
    return (left_shoulder_angle, right_shoulder_angle, left_elbow_angle, right_elbow_angle, 
            left_wrist_angle, right_wrist_angle, left_hip_angle, right_hip_angle, 
            left_knee_angle, right_knee_angle, left_ankle_angle, right_ankle_angle, 
            shoulder_dev, shoulder_mid_x, central_vertical_line_x)


def side_angle_analysis(landmarks, mp_pose) -> Tuple:
    """
    Analyze side-view squat posture based on landmarks.
    
    Args:
        landmarks: List of detected pose landmarks.
        mp_pose: Mediapipe pose module for landmark references.
    
    Returns:
        Tuple containing angles for shoulders, elbows, hips, knees, ankles, 
        neck, and torso angles.
    """
    # Get points
    left_shoulder, left_elbow, left_wrist = map(
        lambda lm: get_point(landmarks, lm),
        [landmarks_to_draw[11], landmarks_to_draw[13], landmarks_to_draw[15]]
    )
    left_hip, left_knee, left_ankle, left_foot_index, left_ear = map(
        lambda lm: get_point(landmarks, lm),
        [landmarks_to_draw[23], landmarks_to_draw[25], landmarks_to_draw[27], 
         landmarks_to_draw[31], landmarks_to_draw[7]]
    )
    
    right_shoulder, right_elbow, right_wrist = map(
        lambda lm: get_point(landmarks, lm),
        [landmarks_to_draw[12], landmarks_to_draw[14], landmarks_to_draw[16]]
    )
    right_hip, right_knee, right_ankle, right_foot_index, right_ear = map(
        lambda lm: get_point(landmarks, lm),
        [landmarks_to_draw[24], landmarks_to_draw[26], landmarks_to_draw[28], 
         landmarks_to_draw[32], landmarks_to_draw[8]]
    )

    # Calculate angles
    left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
    right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
    
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    left_hip_angle = calculate_angle(left_shoulder, left_hip, [left_hip[0], 0])
    right_hip_angle = calculate_angle(right_shoulder, right_hip, [right_hip[0], 0])
    
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    
    left_ankle_angle = calculate_angle(left_knee, left_ankle, [left_ankle[0], 0])
    right_ankle_angle = calculate_angle(right_knee, right_ankle, [right_ankle[0], 0])
    
    left_neck_angle = calculate_angle([left_shoulder[0], 0], left_shoulder, left_ear)
    right_neck_angle = calculate_angle([right_shoulder[0], 0], right_shoulder, right_ear)
    
    left_torso_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_torso_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    
    left_ankle_f_angle = calculate_angle(left_knee, left_ankle, left_foot_index)
    right_ankle_f_angle = calculate_angle(right_knee, right_ankle, right_foot_index)
    
    return (left_shoulder_angle, right_shoulder_angle, left_elbow_angle, right_elbow_angle,
            left_hip_angle, right_hip_angle, left_knee_angle, right_knee_angle, 
            left_ankle_angle, right_ankle_angle, left_neck_angle, right_neck_angle, 
            left_torso_angle, right_torso_angle, left_ankle_f_angle, right_ankle_f_angle)


def balance_back_analysis(landmarks, mp_pose) -> Tuple:
    """
    Analyze back-view balance posture based on landmarks.
    
    Args:
        landmarks: List of detected pose landmarks.
        mp_pose: Mediapipe pose module for landmark references.
    
    Returns:
        Tuple containing angle differences for shoulders, elbows, and hips.
    """
    # Get points
    left_shoulder, left_hip, left_elbow = map(
        lambda lm: get_point(landmarks, lm), 
        [pose_landmark.LEFT_SHOULDER, pose_landmark.LEFT_HIP, pose_landmark.LEFT_ELBOW]
    )
    right_shoulder, right_hip, right_elbow = map(
        lambda lm: get_point(landmarks, lm), 
        [pose_landmark.RIGHT_SHOULDER, pose_landmark.RIGHT_HIP, pose_landmark.RIGHT_ELBOW]
    )

    # Calculate angles
    left_shoulder_angle = calculate_angle(right_shoulder, left_shoulder, [left_shoulder[0], 0])
    right_shoulder_angle = calculate_angle(left_shoulder, right_shoulder, [right_shoulder[0], 0])
    
    left_hip_angle = calculate_angle(right_hip, left_hip, [left_hip[0], 0])
    right_hip_angle = calculate_angle(left_hip, right_hip, [right_hip[0], 0])
    
    left_elbow_angle = calculate_angle(right_elbow, left_elbow, [left_elbow[0], 0])
    right_elbow_angle = calculate_angle(left_elbow, right_elbow, [right_elbow[0], 0])
    
    return (left_shoulder_angle - right_shoulder_angle, 
            left_elbow_angle - right_elbow_angle, 
            left_hip_angle - right_hip_angle)


def balance_test_analysis(landmarks, mp_pose) -> Tuple:
    """
    Analyze balance test posture during knee-raising exercises.
    
    Args:
        landmarks: List of detected pose landmarks.
        mp_pose: Mediapipe pose module for landmark references.
    
    Returns:
        Tuple containing shoulder and hip deviation metrics.
    """
    # Get points
    left_shoulder, left_hip, left_heel = map(
        lambda lm: get_point(landmarks, lm), 
        [pose_landmark.LEFT_SHOULDER, pose_landmark.LEFT_HIP, pose_landmark.LEFT_HEEL]
    )
    right_shoulder, right_hip, right_heel = map(
        lambda lm: get_point(landmarks, lm), 
        [pose_landmark.RIGHT_SHOULDER, pose_landmark.RIGHT_HIP, pose_landmark.RIGHT_HEEL]
    )

    # Calculate midpoints
    shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
    hip_mid_x = (left_hip[0] + right_hip[0]) / 2
    central_vertical_line_x = (left_heel[0] + right_heel[0]) / 2
    
    shoulder_dev = (shoulder_mid_x - central_vertical_line_x) / abs(left_shoulder[0] - right_shoulder[0])
    hip_dev = (hip_mid_x - central_vertical_line_x) / abs(left_hip[0] - right_hip[0])
    
    return (shoulder_dev, hip_dev, central_vertical_line_x, shoulder_mid_x)


def balance_front_analysis(landmarks, mp_pose) -> Tuple:
    """
    Analyze front-view standing balance posture.
    
    Args:
        landmarks: List of detected pose landmarks.
        mp_pose: Mediapipe pose module for landmark references.
    
    Returns:
        Tuple containing angle differences for ears, shoulders, hips, knees, and ankles.
    """
    # Get points
    left_ear, left_shoulder, left_hip, left_knee, left_ankle = map(
        lambda lm: get_point(landmarks, lm),
        [mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.LEFT_SHOULDER,
         mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE,
         mp_pose.PoseLandmark.LEFT_ANKLE]
    )
    
    right_ear, right_shoulder, right_hip, right_knee, right_ankle = map(
        lambda lm: get_point(landmarks, lm),
        [mp_pose.PoseLandmark.RIGHT_EAR, mp_pose.PoseLandmark.RIGHT_SHOULDER,
         mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE,
         mp_pose.PoseLandmark.RIGHT_ANKLE]
    )

    # Calculate angles
    left_ear_angle = calculate_angle([left_ear[0], 0], left_ear, right_ear)
    right_ear_angle = calculate_angle([right_ear[0], 0], right_ear, left_ear)
    
    left_shoulder_angle = calculate_angle([left_shoulder[0], 0], left_shoulder, right_shoulder)
    right_shoulder_angle = calculate_angle([right_shoulder[0], 0], right_shoulder, left_shoulder)
    
    left_hip_angle = calculate_angle([left_hip[0], 0], left_hip, right_hip)
    right_hip_angle = calculate_angle([right_hip[0], 0], right_hip, left_hip)
    
    left_knee_angle = calculate_angle([left_knee[0], 0], left_knee, right_knee)
    right_knee_angle = calculate_angle([right_knee[0], 0], right_knee, left_knee)
    
    left_ankle_angle = calculate_angle([left_ankle[0], 0], left_ankle, right_ankle)
    right_ankle_angle = calculate_angle([right_ankle[0], 0], right_ankle, left_ankle)
    
    return (left_ear_angle - right_ear_angle, 
            left_shoulder_angle - right_shoulder_angle, 
            left_hip_angle - right_hip_angle, 
            left_knee_angle - right_knee_angle, 
            left_ankle_angle - right_ankle_angle)


def balance_side_analysis(landmarks, mp_pose) -> Tuple:
    """
    Analyze side-view standing balance posture.
    
    Args:
        landmarks: List of detected pose landmarks.
        mp_pose: Mediapipe pose module for landmark references.
    
    Returns:
        Tuple containing shoulder and knee angles.
    """
    # Get points
    left_ear, left_shoulder, left_hip, left_knee, left_ankle = map(
        lambda lm: get_point(landmarks, lm),
        [mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.LEFT_SHOULDER,
         mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, 
         mp_pose.PoseLandmark.LEFT_ANKLE]
    )

    # Calculate angles
    left_shoulder_angle = calculate_angle([left_shoulder[0], 0], left_shoulder, left_ear)
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    
    return (left_shoulder_angle, left_knee_angle)
