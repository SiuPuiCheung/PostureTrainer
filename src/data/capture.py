"""Data capture and input handling for pose evaluation."""

import cv2
import os
import sys
import tkinter as tk
from tkinter import filedialog, PhotoImage
from typing import Tuple, Optional, Dict, Any
from ..utils.config_loader import Config


class CaptureManager:
    """Manages video/image capture initialization and GUI interactions."""
    
    def __init__(self, config: Config):
        """
        Initialize capture manager with configuration.
        
        Args:
            config: Configuration object containing GUI and analysis settings.
        """
        self.config = config
        self.gui_config = config.gui_config
    
    def show_gui(self, options: Dict[str, Any]) -> Optional[Any]:
        """
        Create a graphical user interface dialog with buttons based on provided options.
        
        Args:
            options: Dictionary where keys are button text and values are return values.
        
        Returns:
            The value associated with the selected button, or None if no selection.
        """
        # Check if application is frozen
        if getattr(sys, 'frozen', False):
            application_path = sys._MEIPASS
        else:
            application_path = os.getcwd()

        bg_image_path = os.path.join(application_path, self.config.paths['bg_image'])
        
        # Initialize dialog window
        dialog = tk.Toplevel()
        dialog.title(self.gui_config['title'])

        # Load and place background image if it exists
        if os.path.exists(bg_image_path):
            bg_image = PhotoImage(file=bg_image_path).subsample(2, 2)
            tk.Label(dialog, image=bg_image).place(x=0, y=0, relwidth=1, relheight=1)
            dialog.bg_image = bg_image
            version_label = f"© 2024 E.C.| {self.gui_config['version']}"
            tk.Label(dialog, text=version_label, font=("Helvetica", 5)).place(
                x=0, y=bg_image.height() - 10
            )
            dialog.geometry(f"{bg_image.width()}x{bg_image.height()}")
        else:
            dialog.geometry("600x400")
        
        dialog.grab_set()
        result = [None]

        # Define layout parameters
        btn_width = self.gui_config['button_width']
        btn_height = self.gui_config['button_height']
        space_x = self.gui_config['space_x']
        space_y = self.gui_config['space_y']
        num_cols = self.gui_config['num_columns']
        
        total_width = (btn_width * num_cols) + (space_x * (num_cols - 1))
        start_x = (600 - total_width) / 2 if not os.path.exists(bg_image_path) else \
                  (bg_image.width() - total_width) / 2
        start_y = 100 if not os.path.exists(bg_image_path) else \
                  (bg_image.height() / 4) if hasattr(dialog, 'bg_image') else 100

        def make_selection(value):
            result[0] = value
            dialog.destroy()

        # Create buttons
        for index, (text, value) in enumerate(options.items()):
            x = start_x + (index % num_cols) * (btn_width + space_x)
            y = start_y + (index // num_cols) * (btn_height + space_y)
            tk.Button(
                dialog, 
                text=text, 
                command=lambda v=value: make_selection(v),
                font=("Helvetica", 10, "bold")
            ).place(x=x, y=y, width=btn_width, height=btn_height)

        dialog.wait_window()
        return result[0]
    
    def initialize_capture(self) -> Tuple:
        """
        Initialize video capture based on user selection for analysis type and input source.
        
        Returns:
            Tuple containing:
            - cap: Video capture object
            - is_image: Boolean indicating if source is an image
            - path: Path to file (if applicable)
            - frame_rate: Frame rate of video
            - anal_func: Analysis function
            - detect_func: Detection function
            - analysis_choice: Analysis choice index
            - pose_model_id: Selected pose model identifier
            - device_id: Selected compute device identifier
        """
        # Initialize hidden tkinter window
        root = tk.Tk()
        root.withdraw()

        # Get analysis types from config
        analysis_types = self.config.analysis_types
        a_opts = {atype['name']: atype['id'] for atype in analysis_types}
        i_opts = {"Image": 1, "Video file": 2, "Camera": 3}

        pose_options_cfg = self.config.get_pose_model_options()
        if pose_options_cfg:
            m_opts = {
                option.get('name', option['id']): option['id']
                for option in pose_options_cfg if 'id' in option
            }
        else:
            m_opts = {"MediaPipe Pose": "mediapipe"}

        # Get user selections
        a_choice = self.show_gui(a_opts)
        if not a_choice:
            sys.exit(0)

        pose_choice = self.show_gui(m_opts)
        if not pose_choice:
            sys.exit(0)

        selected_model_cfg = next(
            (option for option in pose_options_cfg if option.get('id') == pose_choice),
            {'devices': ['cpu'], 'id': 'mediapipe'}
        ) if pose_options_cfg else {'devices': ['cpu'], 'id': 'mediapipe'}

        device_buttons: Dict[str, str]
        devices_cfg = [str(dev).lower() for dev in selected_model_cfg.get('devices', [])]
        if 'gpu' in devices_cfg:
            device_buttons = {
                "Auto (CPU/GPU)": "auto",
                "GPU (CUDA)": "gpu",
                "CPU only": "cpu",
            }
        else:
            device_buttons = {
                "CPU only": "cpu",
            }

        device_choice = self.show_gui(device_buttons)
        if not device_choice:
            sys.exit(0)
        
        # Get analysis and detection functions
        from ..core.pose_analysis import (
            front_angle_analysis, side_angle_analysis, balance_back_analysis,
            balance_test_analysis, balance_front_analysis, balance_side_analysis
        )
        from ..core.pose_detection import (
            front_angle_detection, side_angle_detection, balance_back_detection,
            balance_test_detection, balance_front_detection, balance_side_detection
        )
        
        analysis_funcs = {
            1: front_angle_analysis, 2: side_angle_analysis, 3: balance_back_analysis,
            4: balance_test_analysis, 5: balance_front_analysis, 6: balance_side_analysis
        }
        detection_funcs = {
            1: front_angle_detection, 2: side_angle_detection, 3: balance_back_detection,
            4: balance_test_detection, 5: balance_front_detection, 6: balance_side_detection
        }
        
        anal_func = analysis_funcs[a_choice]
        detect_func = detection_funcs[a_choice]
        
        i_choice = self.show_gui(i_opts)
        if not i_choice:
            sys.exit(0)

        # Determine file type filter
        file_type = [("Image files", "*.jpg *.jpeg *.png")] if i_choice == 1 else \
                   [("Video files", "*.mp4 *.mov *.avi")]
        
        path = None
        if i_choice != 3:
            path = filedialog.askopenfilename(title="Select the file", filetypes=file_type)
            if not path:
                sys.exit(0)

        # Setup video capture
        cap = cv2.VideoCapture(0 if i_choice == 3 else path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
        frame_rate = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0

        return (
            cap,
            i_choice == 1,
            path,
            frame_rate,
            anal_func,
            detect_func,
            a_choice - 1,
            pose_choice,
            device_choice,
        )


def setup_output_writer(cap, is_image: bool, timestamp: str, output_dir: str = "output"):
    """
    Set up a writer object to save processed video or image output.
    
    Args:
        cap: Video capture object.
        is_image: Boolean indicating if input is an image.
        timestamp: Timestamp string for file naming.
        output_dir: Output directory path.
    
    Returns:
        For images, returns output file path. For videos, returns VideoWriter object.
    """
    # Initialize hidden tkinter window
    root = tk.Tk()
    root.withdraw()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file path
    extension = '.jpg' if is_image else '.mp4'
    out_path = f"{output_dir}/Result_{timestamp}{extension}"
    
    # Create VideoWriter for videos
    if not is_image:
        return cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*'MP4V'),
            cap.get(cv2.CAP_PROP_FPS),
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        )
    
    return out_path
