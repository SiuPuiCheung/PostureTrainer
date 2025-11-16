"""Streamlit web application for pose evaluation."""

import streamlit as st
import cv2
import pandas as pd
import mediapipe as mp
import numpy as np
from datetime import datetime
from PIL import Image
import tempfile
import os

from src.utils.config_loader import Config
from src.utils.report import ReportGenerator
from src.core import pose_analysis, pose_detection


# Page configuration
st.set_page_config(
    page_title="Posture Trainer",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 5px;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .analysis-card {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'config' not in st.session_state:
        st.session_state.config = Config()
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'joint_angles_df' not in st.session_state:
        st.session_state.joint_angles_df = None
    if 'analysis_choice' not in st.session_state:
        st.session_state.analysis_choice = 0


def get_analysis_functions(analysis_id):
    """Get analysis and detection functions by ID."""
    funcs_map = {
        1: (pose_analysis.front_angle_analysis, pose_detection.front_angle_detection),
        2: (pose_analysis.side_angle_analysis, pose_detection.side_angle_detection),
        3: (pose_analysis.balance_back_analysis, pose_detection.balance_back_detection),
        4: (pose_analysis.balance_test_analysis, pose_detection.balance_test_detection),
        5: (pose_analysis.balance_front_analysis, pose_detection.balance_front_detection),
        6: (pose_analysis.balance_side_analysis, pose_detection.balance_side_detection),
    }
    return funcs_map.get(analysis_id, (None, None))


def process_frame(frame, pose, anal_func, detect_func):
    """Process a single frame for pose detection."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    if results.pose_landmarks:
        angles_data = anal_func(results.pose_landmarks.landmark, mp.solutions.pose)
        annotated_image = detect_func(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results, angles_data)
        return annotated_image, angles_data, True
    
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), {}, False


def process_image(uploaded_file, analysis_id, confidence):
    """Process uploaded image."""
    config = st.session_state.config
    anal_func, detect_func = get_analysis_functions(analysis_id)
    
    if anal_func is None:
        st.error("Invalid analysis type selected")
        return None, None, None
    
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Process with MediaPipe
    with mp.solutions.pose.Pose(
        min_detection_confidence=confidence,
        min_tracking_confidence=confidence
    ) as pose:
        annotated_image, angles_data, detected = process_frame(frame, pose, anal_func, detect_func)
        
        if detected:
            joint_angles_df = pd.DataFrame([angles_data])
            return annotated_image, joint_angles_df, True
        else:
            return annotated_image, None, False


def process_video(uploaded_file, analysis_id, confidence, progress_bar, status_text):
    """Process uploaded video."""
    config = st.session_state.config
    anal_func, detect_func = get_analysis_functions(analysis_id)
    
    if anal_func is None:
        st.error("Invalid analysis type selected")
        return None, None, None, None
    
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    tfile.close()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file")
        return None, None, None, None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video with H264 codec for better browser compatibility
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H264 codec
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
    
    # Fallback to mp4v if avc1 fails
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
    
    joint_angles_list = []
    frame_count = 0
    
    with mp.solutions.pose.Pose(
        min_detection_confidence=confidence,
        min_tracking_confidence=confidence
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_image, angles_data, detected = process_frame(frame, pose, anal_func, detect_func)
            out.write(annotated_image)
            
            if detected:
                joint_angles_list.append(angles_data)
            
            frame_count += 1
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    os.unlink(video_path)
    
    joint_angles_df = pd.DataFrame(joint_angles_list) if joint_angles_list else None
    return output_path, joint_angles_df, frame_rate, True


def process_webcam(analysis_id, confidence, stop_button_placeholder):
    """Process webcam feed."""
    anal_func, detect_func = get_analysis_functions(analysis_id)
    
    if anal_func is None:
        st.error("Invalid analysis type selected")
        return None, None
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access webcam")
        return None, None
    
    # Set high resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30
    stframe = st.empty()
    joint_angles_list = []
    
    stop_button = stop_button_placeholder.button("⏹ Stop Recording", key="stop_webcam")
    
    with mp.solutions.pose.Pose(
        min_detection_confidence=confidence,
        min_tracking_confidence=confidence
    ) as pose:
        frame_count = 0
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_image, angles_data, detected = process_frame(frame, pose, anal_func, detect_func)
            
            # Display frame
            stframe.image(annotated_image, channels="BGR", use_column_width=True)
            
            if detected:
                joint_angles_list.append(angles_data)
            
            frame_count += 1
            stop_button = stop_button_placeholder.button("⏹ Stop Recording", key=f"stop_webcam_{frame_count}")
    
    cap.release()
    joint_angles_df = pd.DataFrame(joint_angles_list) if joint_angles_list else None
    return joint_angles_df, frame_rate


def display_metrics(joint_angles_df, analysis_choice):
    """Display angle metrics in columns."""
    if joint_angles_df is None or joint_angles_df.empty:
        st.warning("No pose detected in the input")
        return
    
    config = st.session_state.config
    labels = config.get_body_labels_by_index(analysis_choice)
    
    st.subheader("📊 Angle Measurements")
    
    # Get relevant columns (exclude metadata columns)
    angle_columns = joint_angles_df.columns[:len(labels)]
    
    # Display metrics in columns
    num_cols = min(4, len(angle_columns))
    cols = st.columns(num_cols)
    
    for idx, col_name in enumerate(angle_columns):
        with cols[idx % num_cols]:
            mean_angle = joint_angles_df[col_name].mean()
            max_angle = joint_angles_df[col_name].max()
            min_angle = joint_angles_df[col_name].min()
            
            st.markdown(f"""
                <div class="analysis-card">
                    <h4>{labels[idx] if idx < len(labels) else col_name}</h4>
                    <p><b>Average:</b> {mean_angle:.1f}°</p>
                    <p><b>Max:</b> {max_angle:.1f}°</p>
                    <p><b>Min:</b> {min_angle:.1f}°</p>
                </div>
            """, unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    initialize_session_state()
    config = st.session_state.config
    
    # Header
    st.title("🧘 Posture Trainer")
    st.markdown("**Analyze your posture with AI-powered pose estimation**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Analysis type selection
        st.subheader("1️⃣ Select Analysis Type")
        analysis_types = config.analysis_types
        analysis_names = [atype['name'] for atype in analysis_types]
        analysis_descriptions = {
            "Front Angle": "Analyze front-facing squat posture",
            "Side Angle": "Analyze side-view squat posture",
            "Balance Back": "Analyze back-view balance",
            "Balance Test": "Analyze balance during movement",
            "Balance Front": "Analyze front-view standing balance",
            "Balance Side": "Analyze side-view standing balance"
        }
        
        selected_analysis = st.selectbox(
            "Choose analysis mode",
            analysis_names,
            help="Select the type of posture analysis"
        )
        
        analysis_id = next(atype['id'] for atype in analysis_types if atype['name'] == selected_analysis)
        st.session_state.analysis_choice = analysis_id - 1
        
        st.info(analysis_descriptions.get(selected_analysis, ""))
        
        # Input source selection
        st.subheader("2️⃣ Select Input Source")
        input_source = st.radio(
            "Choose input type",
            ["📷 Image", "🎥 Video File", "📹 Webcam"],
            help="Select your input source"
        )
        
        # Model confidence
        st.subheader("3️⃣ Model Settings")
        confidence = st.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Higher values = more accurate but may miss some poses"
        )
        
        st.markdown("---")
        st.markdown("### 📖 Instructions")
        st.markdown("""
        1. Select analysis type
        2. Choose input source
        3. Upload/start capture
        4. View results & download report
        """)
    
    # Main content
    if input_source == "📷 Image":
        st.header("📷 Image Analysis")
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image showing the full body"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Analyzed Result")
                with st.spinner("Analyzing posture..."):
                    annotated_image, joint_angles_df, detected = process_image(
                        uploaded_file, analysis_id, confidence
                    )
                
                if detected:
                    st.image(annotated_image, channels="BGR", use_column_width=True)
                    st.success("✅ Pose detected successfully!")
                else:
                    st.image(annotated_image, channels="BGR", use_column_width=True)
                    st.warning("⚠️ No pose detected. Try a different image or adjust confidence.")
            
            if detected and joint_angles_df is not None:
                st.markdown("---")
                display_metrics(joint_angles_df, st.session_state.analysis_choice)
                
                # Download options
                st.markdown("---")
                st.subheader("💾 Download Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Save annotated image
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_path = f"output/Result_{timestamp}.jpg"
                    os.makedirs("output", exist_ok=True)
                    cv2.imwrite(output_path, annotated_image)
                    
                    with open(output_path, "rb") as f:
                        st.download_button(
                            "📥 Download Annotated Image",
                            f,
                            file_name=f"Result_{timestamp}.jpg",
                            mime="image/jpeg"
                        )
                
                with col2:
                    # Generate and offer PDF report
                    report_gen = ReportGenerator(config)
                    report_path = f"output/Report_{timestamp}.pdf"
                    
                    # Add frame rate for single image (use 1)
                    report_gen.generate_report(
                        joint_angles_df, 1.0, st.session_state.analysis_choice,
                        timestamp, "output"
                    )
                    
                    with open(report_path, "rb") as f:
                        st.download_button(
                            "📄 Download PDF Report",
                            f,
                            file_name=f"Report_{timestamp}.pdf",
                            mime="application/pdf"
                        )
    
    elif input_source == "🎥 Video File":
        st.header("🎥 Video Analysis")
        uploaded_file = st.file_uploader(
            "Upload a video",
            type=['mp4', 'mov', 'avi'],
            help="Upload a video showing the full body"
        )
        
        if uploaded_file is not None:
            # Display the original video
            st.video(uploaded_file)
            
            if st.button("🚀 Start Analysis", key="process_video"):
                # Reset file pointer to beginning for processing
                uploaded_file.seek(0)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Processing video..."):
                    output_path, joint_angles_df, frame_rate, detected = process_video(
                        uploaded_file, analysis_id, confidence, progress_bar, status_text
                    )
                
                progress_bar.empty()
                status_text.empty()
                
                if detected and output_path:
                    st.success("✅ Video processed successfully!")
                    
                    st.subheader("Processed Video")
                    st.video(output_path)
                    
                    if joint_angles_df is not None:
                        st.markdown("---")
                        display_metrics(joint_angles_df, st.session_state.analysis_choice)
                        
                        # Download options
                        st.markdown("---")
                        st.subheader("💾 Download Results")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            with open(output_path, "rb") as f:
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                st.download_button(
                                    "📥 Download Processed Video",
                                    f,
                                    file_name=f"Result_{timestamp}.mp4",
                                    mime="video/mp4"
                                )
                        
                        with col2:
                            # Generate PDF report
                            report_gen = ReportGenerator(config)
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            report_path = f"output/Report_{timestamp}.pdf"
                            os.makedirs("output", exist_ok=True)
                            
                            report_gen.generate_report(
                                joint_angles_df, frame_rate, st.session_state.analysis_choice,
                                timestamp, "output"
                            )
                            
                            with open(report_path, "rb") as f:
                                st.download_button(
                                    "📄 Download PDF Report",
                                    f,
                                    file_name=f"Report_{timestamp}.pdf",
                                    mime="application/pdf"
                                )
                else:
                    st.error("❌ Failed to process video or no pose detected")
    
    else:  # Webcam
        st.header("📹 Webcam Analysis")
        st.info("Click 'Start Recording' to begin capturing from your webcam. Click 'Stop Recording' when finished.")
        
        if st.button("🎬 Start Recording", key="start_webcam"):
            stop_placeholder = st.empty()
            
            with st.spinner("Recording from webcam..."):
                joint_angles_df, frame_rate = process_webcam(
                    analysis_id, confidence, stop_placeholder
                )
            
            if joint_angles_df is not None and not joint_angles_df.empty:
                st.success("✅ Recording completed!")
                
                display_metrics(joint_angles_df, st.session_state.analysis_choice)
                
                # Generate and offer report
                st.markdown("---")
                st.subheader("💾 Download Report")
                
                report_gen = ReportGenerator(config)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                report_path = f"output/Report_{timestamp}.pdf"
                os.makedirs("output", exist_ok=True)
                
                report_gen.generate_report(
                    joint_angles_df, frame_rate, st.session_state.analysis_choice,
                    timestamp, "output"
                )
                
                with open(report_path, "rb") as f:
                    st.download_button(
                        "📄 Download PDF Report",
                        f,
                        file_name=f"Report_{timestamp}.pdf",
                        mime="application/pdf"
                    )
            else:
                st.warning("⚠️ No pose data recorded")


if __name__ == "__main__":
    main()
