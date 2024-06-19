import streamlit as st
import cv2
from ultralytics import YOLO
import utils.detection as detection
import tempfile
import time

# Define class names mapping
class_names = {0: 'cheating', 1: 'good', 2: 'normal'}
# names: ['cheating', 'good', 'normal']

# Load YOLO model
model_path = 'yolov8/trained_model.pt'
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
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)
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

        # Update class counts
        counts = update_class_counts(results, class_names)
        display_metrics(counts)

        # Draw boxes and labels on the frame
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                confidence = float(box.conf)

                # Map class id to class name
                class_name = class_names.get(class_id, 'Unknown')

                # Skip drawing for the Phone class (commented out class id)
                # if class_name == 'Phone':
                    # continue

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Put label with confidence
                label = f'{class_name} {confidence:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
