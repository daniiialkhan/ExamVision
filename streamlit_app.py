import streamlit as st
import cv2
from ultralytics import YOLO
import utils.detection as detection
import tempfile
import time
import numpy as np

# Define class names mapping
class_names = {0: 'cheating', 1: 'good', 2: 'normal'}
# names: ['cheating', 'good', 'normal']

# Dropdown for model selection
model_selection = st.sidebar.selectbox(
    "Select YOLO model",
    ("YOLOv8 Standard", "YOLOv8 OBB")
)

# Model paths
model_paths = {
    "YOLOv8 Standard": 'yolov8/trained_model.pt',
    "YOLOv8 OBB": 'yolov8/yolov8-obb_trained.pt'
}

# Load YOLO model based on selection
model_path = model_paths[model_selection]
model = YOLO(model_path)

# Streamlit UI
st.title('Live Object Detection for Exam Cheating Detection')

# Selection between live video and file upload
source = st.radio("Select video source", ('Live Video', 'Upload MP4 File'))

# Initialize placeholders for displaying class counts
cheating_placeholder = st.sidebar.empty()
non_cheating_placeholder = st.sidebar.empty()
background_placeholder = st.sidebar.empty()

def update_class_counts(results, class_names):
    counts = {class_name: 0 for class_name in class_names.values()}
    
    if results is None:
        print("No results to update counts.")
        return counts
    
    for result in results:
        if model_selection == "YOLOv8 Standard":
            boxes = result.boxes
            
            if boxes is None:
                print("No boxes found in result.")
                continue
            
            for box in boxes:
                class_id = int(box.cls)
                class_name = class_names.get(class_id, 'Unknown')
                if class_name in counts:
                    counts[class_name] += 1
        
        elif model_selection == "YOLOv8 OBB":
            obbs = result.obb
            
            if obbs is None:
                print("No OBBs found in result.")
                continue
            
            for obb in obbs:
                # cx, cy, w, h, angle = obb
                # You may need to adjust how you interpret the OBB parameters based on your model output
                # Here, assuming 'angle' is in radians and you have (cx, cy, w, h)
                # Perform your logic here to count objects or handle OBBs
                
                class_id = int(obb.cls)  # Assuming 'obb.cls' exists in your result
                class_name = class_names.get(class_id, 'Unknown')
                if class_name in counts:
                    counts[class_name] += 1
    
    return counts

def display_metrics(counts):
    cheating_placeholder.metric(label='cheating', value=counts['cheating'])
    non_cheating_placeholder.metric(label='good', value=counts['good'])
    background_placeholder.metric(label='normal', value=counts['normal'])

def process_video_capture(video_capture, stframe):
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Perform detection
        results = detection.detect_objects(model, frame)
        print("Got results: ", results)

        # Check if results is None or empty
        if results is None:
            print("No results returned from detection function")
            continue
        elif len(results) == 0:
            print("No objects detected in the frame")
            continue

        # Update class counts
        counts = update_class_counts(results, class_names)
        display_metrics(counts)

        # Draw boxes and labels on the frame
        for result in results:
            if model_selection == "YOLOv8 Standard":
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    print("No boxes found in result")
                    continue
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls)
                    confidence = float(box.conf)

                    # Map class id to class name
                    class_name = class_names.get(class_id, 'Unknown')

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Put label with confidence
                    label = f'{class_name} {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            elif model_selection == "YOLOv8 OBB":
                obbs = result.obb
                if obbs is None or len(obbs) == 0:
                    print("No OBBs found in result")
                    continue
                for obb in obbs:
                    # cx, cy, w, h, angle = obb.xyxy
                    x1, y1, x2, y2 = map(int, obb.xyxy[0])
                    class_id = int(obb.cls)
                    # class_name = class_names.get(class_id, 'Unknown')
                    confidence = float(obb.conf)
                    # Map class id to class name
                    class_name = class_names.get(class_id, 'Unknown')

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Put label with confidence
                    label = f'{class_name} {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # if class_name in counts:
                    #     counts[class_name] += 1
                    # angle_deg = angle * 180.0 / np.pi  # Convert angle to degrees
                    # box = cv2.boxPoints(((cx, cy), (w, h), angle_deg))
                    # box = np.int0(box)

                    # # Draw rotated bounding box
                    # cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

                    # Optionally, put label with confidence
                    # confidence = result.conf
                    # label = f'Confidence: {confidence:.2f}'
                    # cv2.putText(frame, label, (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert frame to RGB for Streamlit display
        annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the annotated frame
        stframe.image(annotated_frame, channels='RGB')

    # At the end of the video, ask if the user wants to replay it
    return st.button("Replay Video", key=f'replay-{time.time()}')  # Unique key

if source == 'Live Video':
    # Start video capture
    video_capture = cv2.VideoCapture(0)
    stframe = st.empty()

    process_video_capture(video_capture, stframe)

    # Release the capture when done
    video_capture.release()
    cv2.destroyAllWindows()

elif source == 'Upload MP4 File':
    # File uploader
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            tmpfile.write(uploaded_file.read())
            tmpfile_path = tmpfile.name
        
        stframe = st.empty()

        # Function to process the video capture
        def process_uploaded_video():
            video_capture = cv2.VideoCapture(tmpfile_path)
            replay_button = process_video_capture(video_capture, stframe)
            video_capture.release()
            cv2.destroyAllWindows()
            return replay_button

        # Process and replay the video as needed
        while True:
            replay = process_uploaded_video()
            if not replay:
                break
            time.sleep(1)
