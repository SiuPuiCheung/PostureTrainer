"""Main pipeline orchestration for pose evaluation."""

import cv2
import pandas as pd
import mediapipe as mp
import threading
import tkinter as tk
from datetime import datetime
from typing import Tuple
import numpy as np

from src.utils.config_loader import Config
from src.data.capture import CaptureManager, setup_output_writer
from src.utils.report import ReportGenerator


# Global control variables
stop_evaluation = False
run = True
joint_angles_df = pd.DataFrame()


def process_frame(frame: np.ndarray, 
                 pose, 
                 anal_func, 
                 detect_func) -> Tuple[np.ndarray, dict]:
    """
    Process a single frame to detect pose landmarks, analyze posture, and annotate.
    
    Args:
        frame: Input frame from video or camera.
        pose: Pose estimation model.
        anal_func: Function to analyze pose and compute angles.
        detect_func: Function to draw annotations on frame.
    
    Returns:
        Tuple containing annotated image and angles data dictionary.
    """
    # Convert frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    if results.pose_landmarks:
        # Compute angles from pose landmarks
        angles_data = anal_func(results.pose_landmarks.landmark, mp.solutions.pose)
        # Annotate frame with results
        annotated_image = detect_func(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 
                                      results, angles_data)
        return annotated_image, angles_data
    
    # No landmarks detected
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), {}


def show_stop_gui():
    """
    Create a GUI with Stop/Proceed button to control the analysis process.
    
    This provides a graphical interface to stop the ongoing analysis at any point.
    """
    global stop_evaluation
    
    def on_button_click():
        global stop_evaluation
        if button['text'] == 'Stop':
            stop_evaluation = True
            button.config(text='Proceed')
        else:
            root.destroy()
    
    # Initialize control GUI window
    root = tk.Tk()
    root.title("Analysis Control")
    tk.Label(root, text="Click 'Stop' to stop analysis \nClick 'Proceed' to complete analysis.").pack(pady=20)
    
    button = tk.Button(root, text="Stop", command=on_button_click)
    button.pack(pady=10)
    
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    
    return stop_evaluation


def run_estimation(config: Config):
    """
    Orchestrate the posture analysis process from initialization to report generation.
    
    This function handles setup of video capture, initiates posture analysis,
    and generates a report upon completion.
    
    Args:
        config: Configuration object containing all settings.
    """
    global stop_evaluation, joint_angles_df
    
    # Reset control variables and data storage
    stop_evaluation = False
    joint_angles_df = pd.DataFrame()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Initialize capture manager and get capture settings
    capture_mgr = CaptureManager(config)
    cap, is_image, _, frame_rate, anal_func, detect_func, analysis_choice = \
        capture_mgr.initialize_capture()
    
    if not cap:
        return

    # Setup output writer
    output_dir = config.paths['output_dir']
    out = setup_output_writer(cap, is_image, timestamp, output_dir)

    # Start control GUI in separate thread
    gui_thread = threading.Thread(target=show_stop_gui, daemon=True)
    gui_thread.start()

    # Get model confidence settings
    model_conf = config.model_config
    d_conf = model_conf['image']['detection_confidence'] if is_image else \
             model_conf['video']['detection_confidence']
    t_conf = model_conf['image']['tracking_confidence'] if is_image else \
             model_conf['video']['tracking_confidence']

    # Initialize Mediapipe pose detection
    with mp.solutions.pose.Pose(min_detection_confidence=d_conf,
                                min_tracking_confidence=t_conf) as pose:
        while cap.isOpened() and not stop_evaluation:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame for pose detection and annotation
            processed_frame, angles_data = process_frame(frame, pose, anal_func, detect_func)
            joint_angles_df = pd.concat([joint_angles_df, pd.DataFrame([angles_data])], 
                                       ignore_index=True)
            
            cv2.imshow('Mediapipe Feed', processed_frame)
            
            # Save processed frame/image
            if is_image:
                cv2.imwrite(out, processed_frame)
                break
            elif out:
                out.write(processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup resources
    cap.release()
    cv2.destroyAllWindows()
    if out and not is_image:
        out.release()
    gui_thread.join()

    # Generate report
    report_gen = ReportGenerator(config)
    report_gen.generate_report(joint_angles_df, frame_rate, analysis_choice, 
                               timestamp, output_dir)


def main():
    """
    Main function that repeatedly runs the posture estimation process.
    
    This loop allows continuous operation, processing multiple images or videos
    and generating reports for each until the program is explicitly terminated.
    """
    # Load configuration
    config = Config()
    
    global run
    while run:
        run_estimation(config)


if __name__ == "__main__":
    main()
