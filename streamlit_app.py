import streamlit as st
import cv2
from ultralytics import YOLO
import utils.detection as detection
from io import BytesIO
import tempfile
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

# Define class names mapping
class_names = {0: 'cheating', 1: 'good', 2: 'good'}

# Model paths
model_paths = {
    "YOLOv8 Standard": 'yolov8/trained_model.pt',
    "YOLOv8 OBB": 'yolov8/yolov8-obb_trained.pt'
}

# Load YOLO models
models = {
    "YOLOv8 Standard": YOLO(model_paths["YOLOv8 Standard"]),
    "YOLOv8 OBB": YOLO(model_paths["YOLOv8 OBB"])
}

# Streamlit UI
st.sidebar.title('Exam Cheating Detection')

# Page selection
page = st.sidebar.radio("Select Page", ('Detection', 'Comparison'))

# Selection between live video and file upload
source = st.sidebar.radio("Select video source", ('Live Video', 'Upload MP4 File'))

# Initialize placeholders for displaying class counts
cheating_placeholder = st.sidebar.empty()
non_cheating_placeholder = st.sidebar.empty()
# background_placeholder = st.sidebar.empty()

def update_class_counts(results, class_names, model_selection):
    counts = {class_name: 0 for class_name in class_names.values()}
    if results is None:
        return counts
    for result in results:
        if model_selection == "YOLOv8 Standard":
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                confidence = float(box.conf)
                if confidence < 0.5:
                    continue
                class_id = int(box.cls)
                class_name = class_names.get(class_id, 'Unknown')
                if class_name in counts:
                    counts[class_name] += 1
        elif model_selection == "YOLOv8 OBB":
            obbs = result.obb
            if obbs is None:
                continue
            for obb in obbs:
                confidence = float(obb.conf)
                if confidence < 0.5:
                    continue
                class_id = int(obb.cls)
                class_name = class_names.get(class_id, 'Unknown')
                if class_name in counts:
                    counts[class_name] += 1
    return counts

def display_metrics(counts):
    cheating_placeholder.metric(label='cheating', value=counts['cheating'])
    non_cheating_placeholder.metric(label='good', value=counts['good'])

def save_frame_with_timestamp(frame, confidence):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"cheating_detections/cheating_{timestamp}_{confidence:.2f}.jpg"
    cv2.imwrite(filename, frame)
    st.write(f"Saved frame as: {filename}")

def obb_to_vertices(cx, cy, w, h, theta):
    """Convert OBB to vertices for drawing with cv2.polylines."""
    theta_rad = np.radians(theta)
    R = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])
    corners = np.array([
        [-w / 2, -h / 2],
        [w / 2, -h / 2],
        [w / 2, h / 2],
        [-w / 2, h / 2]
    ])
    vertices = np.dot(corners, R.T) + [cx, cy]
    return vertices.astype(int)

def draw_obb_with_label(image, vertices, label, color, thickness=2):
    vertices = vertices.reshape((-1, 1, 2))
    cv2.polylines(image, [vertices], isClosed=True, color=color, thickness=thickness)
    top_vertex = tuple(vertices[vertices[:, :, 1].argmin()][0])
    label_pos = (top_vertex[0], top_vertex[1] - 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    label_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    cv2.rectangle(image, 
                  (label_pos[0], label_pos[1] - label_size[1] - 2), 
                  (label_pos[0] + label_size[0], label_pos[1] + 2), 
                  color, cv2.FILLED)
    cv2.putText(image, label, label_pos, font, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)

class_colors = {
    'good': (0, 255, 0),  # Green
    'normal': (0, 255, 255),  # Yellow
    'cheating': (0, 0, 255)  # Red
}

def process_video_capture(video_capture, stframe, model, model_selection):
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        results = detection.detect_objects(model, frame)
        if results is None:
            continue
        counts = update_class_counts(results, class_names, model_selection)
        display_metrics(counts)
        for result in results:
            if model_selection == "YOLOv8 Standard":
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                for box in boxes:
                    confidence = float(box.conf)
                    if confidence < 0.5:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls)
                    class_name = class_names.get(class_id, 'Unknown')
                    color = class_colors.get(class_name.lower(), (0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f'{class_name} {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    if class_name == 'cheating' and confidence >= 0.85:
                        save_frame_with_timestamp(frame, confidence)
            elif model_selection == "YOLOv8 OBB":
                obbs = result.obb
                if obbs is None or len(obbs) == 0:
                    continue
                for obb in obbs:
                    confidence = float(obb.conf)
                    if confidence < 0.5:
                        continue
                    class_id = int(obb.cls)
                    class_name = class_names.get(class_id, 'Unknown')
                    label = f'{class_name} {confidence:.2f}'
                    color = class_colors.get(class_name.lower(), (0, 255, 0))
                    cx, cy, w, h, theta = obb.xywhr[0]
                    vertices = obb_to_vertices(cx, cy, w, h, theta)
                    draw_obb_with_label(frame, vertices, label, color)
                    if class_name == 'cheating' and confidence >= 0.85:
                        save_frame_with_timestamp(frame, confidence)
        annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_frame, channels='RGB')
    return st.button("Replay Video", key=f'replay-{time.time()}')

def detection_page():
    model_selection = st.sidebar.selectbox("Select YOLO model", ("YOLOv8 Standard", "YOLOv8 OBB"))
    model = models[model_selection]
    if source == 'Live Video':
        video_capture = cv2.VideoCapture(0)
        stframe = st.empty()
        process_video_capture(video_capture, stframe, model, model_selection)
        video_capture.release()
        cv2.destroyAllWindows()
    elif source == 'Upload MP4 File':
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
                tmpfile.write(uploaded_file.read())
                tmpfile_path = tmpfile.name
            stframe = st.empty()
            def process_uploaded_video():
                video_capture = cv2.VideoCapture(tmpfile_path)
                replay_button = process_video_capture(video_capture, stframe, model, model_selection)
                video_capture.release()
                cv2.destroyAllWindows()
                return replay_button
            while True:
                replay = process_uploaded_video()
                if not replay:
                    break

# Load training metrics
standard_df = pd.read_csv("yolov8\data\standard_model_training.csv")
standard_df.columns = standard_df.columns.str.strip()
obb_df = pd.read_csv("yolov8\data-obb\obb_model_training.csv")
obb_df.columns = obb_df.columns.str.strip()

# Calculate F1 scores
standard_df['F1'] = 2 * (standard_df['metrics/precision(B)'] * standard_df['metrics/recall(B)']) / \
                    (standard_df['metrics/precision(B)'] + standard_df['metrics/recall(B)'])
standard_epochs = standard_df['epoch']
standard_precision = standard_df['metrics/precision(B)']
standard_recall = standard_df['metrics/recall(B)']
standard_f1 = standard_df['F1']

obb_df['F1'] = 2 * (obb_df['metrics/precision(B)'] * obb_df['metrics/recall(B)']) / \
               (obb_df['metrics/precision(B)'] + obb_df['metrics/recall(B)'])
obb_epochs = obb_df['epoch']
obb_precision = obb_df['metrics/precision(B)']
obb_recall = obb_df['metrics/recall(B)']
obb_f1 = obb_df['F1']

def plot_metrics_comparison():
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].plot(standard_epochs, standard_precision, label='YOLOv8 Standard', color='blue')
    ax[0].plot(obb_epochs, obb_precision, label='YOLOv8 OBB', color='orange')
    ax[0].set_title('Precision')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Precision')
    ax[0].legend()
    
    ax[1].plot(standard_epochs, standard_recall, label='YOLOv8 Standard', color='blue')
    ax[1].plot(obb_epochs, obb_recall, label='YOLOv8 OBB', color='orange')
    ax[1].set_title('Recall')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Recall')
    ax[1].legend()
    
    ax[2].plot(standard_epochs, standard_f1, label='YOLOv8 Standard', color='blue')
    ax[2].plot(obb_epochs, obb_f1, label='YOLOv8 OBB', color='orange')
    ax[2].set_title('F1 Score')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('F1 Score')
    ax[2].legend()
    
    st.pyplot(fig)

def plot_confusion_matrix():
    standard_cm = np.array([
        [0.66, 0.06],
        [0.09, 0.86]
    ])
    obb_cm = np.array([
        [0.75, 0.11],
        [0.07, 0.81]
    ])
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    sns.heatmap(standard_cm, annot=True, fmt='.2f', cmap='Blues', ax=axes[0], 
                xticklabels=['cheating', 'good'], yticklabels=['cheating', 'good'])
    axes[0].set_title('Confusion Matrix for YOLOv8 Standard')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    sns.heatmap(obb_cm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1], 
                xticklabels=['cheating', 'good'], yticklabels=['cheating', 'good'])
    axes[1].set_title('Confusion Matrix for YOLOv8 OBB')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    st.pyplot(fig)

def comparison_page():
    plot_metrics_comparison()
    plot_confusion_matrix()

# Display appropriate page based on selection
if page == 'Detection':
    detection_page()
elif page == 'Comparison':
    comparison_page()
