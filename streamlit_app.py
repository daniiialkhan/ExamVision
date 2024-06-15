import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import utils.detection as detection
import utils.post_processing as post_processing  # Ensure you have this module

# Load YOLO model
model_path = 'yolov8/trained_model.pt'
model = YOLO(model_path)

# Class names (adjust as per your dataset's class names)
class_names = ['student', 'cheating_behavior']

# Streamlit UI
st.title('Live Object Detection for Exam Cheating Detection')

# Start video capture
video_capture = cv2.VideoCapture(0)

# Check if video capture opened successfully
if not video_capture.isOpened():
    st.error("Error: Could not open video stream. Check webcam connection and permissions.")
else:
    stframe = st.empty()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            st.error("Failed to capture video.")
            break

        # Perform detection
        results = model.predict(frame, device='cpu')  # Ensure you're using the correct device

        # Convert YOLO results to usable detections
        detections = post_processing.convert_yolo_results_to_detections(results, frame.shape)

        # Filter detections
        detections = post_processing.filter_detections(detections, confidence_threshold=0.5)

        # Draw bounding boxes and labels
        annotated_frame = post_processing.draw_boxes(frame, detections, class_names)

        # Convert annotated frame to RGB for Streamlit display
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display the annotated frame
        stframe.image(annotated_frame, channels='RGB')

    # Release the capture when done
    video_capture.release()
    cv2.destroyAllWindows()
