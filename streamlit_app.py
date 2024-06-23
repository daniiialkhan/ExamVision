import streamlit as st
import cv2
from ultralytics import YOLO
import utils.detection as detection
from io import BytesIO
import tempfile
import time
import numpy as np

# Define class names mapping
class_names = {0: 'cheating', 1: 'good', 2: 'normal'}
# names: ['cheating', 'good', 'normal']

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
background_placeholder = st.sidebar.empty()

def update_class_counts(results, class_names, model_selection):
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
                class_id = int(obb.cls)
                class_name = class_names.get(class_id, 'Unknown')
                if class_name in counts:
                    counts[class_name] += 1
    
    return counts

def display_metrics(counts):
    cheating_placeholder.metric(label='cheating', value=counts['cheating'])
    non_cheating_placeholder.metric(label='good', value=counts['good'])
    background_placeholder.metric(label='normal', value=counts['normal'])

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
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    class_name = class_names.get(class_id, 'Unknown')
                    color = class_colors.get(class_name.lower(), (0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f'{class_name} {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif model_selection == "YOLOv8 OBB":
                obbs = result.obb
                if obbs is None or len(obbs) == 0:
                    continue
                for obb in obbs:
                    class_id = int(obb.cls)
                    confidence = float(obb.conf)
                    class_name = class_names.get(class_id, 'Unknown')
                    label = f'{class_name} {confidence:.2f}'
                    color = class_colors.get(class_name.lower(), (0, 255, 0))
                    cx, cy, w, h, theta = obb.xywhr[0]
                    vertices = obb_to_vertices(cx, cy, w, h, theta)
                    draw_obb_with_label(frame, vertices, label, color)

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
                time.sleep(1)

# def comparison_page():
#     if source == 'Upload MP4 File':
#         uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
#         if uploaded_file is not None:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
#                 tmpfile.write(uploaded_file.read())
#                 tmpfile_path = tmpfile.name
            
#             stframe_standard = st.empty()
#             stframe_obb = st.empty()

#             def process_and_compare_video():
#                 video_capture = cv2.VideoCapture(tmpfile_path)
#                 while video_capture.isOpened():
#                     ret, frame = video_capture.read()
#                     if not ret:
#                         break

#                     frame_copy_standard = frame.copy()
#                     frame_copy_obb = frame.copy()

#                     results_standard = detection.detect_objects(models["YOLOv8 Standard"], frame_copy_standard)
#                     results_obb = detection.detect_objects(models["YOLOv8 OBB"], frame_copy_obb)

#                     if results_standard:
#                         for result in results_standard:
#                             boxes = result.boxes
#                             if boxes:
#                                 for box in boxes:
#                                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                                     class_id = int(box.cls)
#                                     confidence = float(box.conf)
#                                     class_name = class_names.get(class_id, 'Unknown')
#                                     cv2.rectangle(frame_copy_standard, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                                     label = f'{class_name} {confidence:.2f}'
#                                     cv2.putText(frame_copy_standard, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#                     if results_obb:
#                         for result in results_obb:
#                             obbs = result.obb
#                             if obbs:
#                                 for obb in obbs:
#                                     class_id = int(obb.cls)
#                                     confidence = float(obb.conf)
#                                     class_name = class_names.get(class_id, 'Unknown')
#                                     label = f'{class_name} {confidence:.2f}'
#                                     cx, cy, w, h, theta = obb.xywhr[0]
#                                     vertices = obb_to_vertices(cx, cy, w, h, theta)
#                                     draw_obb_with_label(frame_copy_obb, vertices, label)

#                     annotated_frame_standard = cv2.cvtColor(frame_copy_standard, cv2.COLOR_BGR2RGB)
#                     annotated_frame_obb = cv2.cvtColor(frame_copy_obb, cv2.COLOR_BGR2RGB)

#                     stframe_standard.image(annotated_frame_standard, caption='YOLOv8 Standard', channels='RGB')
#                     stframe_obb.image(annotated_frame_obb, caption='YOLOv8 OBB', channels='RGB')

#                 video_capture.release()
#                 cv2.destroyAllWindows()

#             process_and_compare_video()
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Load CSV data for standard_model_training
standard_model_training = 'yolov8/data/standard_model_training.csv'
standard_df = pd.read_csv(standard_model_training)
print ("standard_df: ", standard_df)
print("Column names in standard_df:", standard_df.columns.tolist())
standard_df.columns = standard_df.columns.str.strip()
print("Column names in standard_df:", standard_df.columns.tolist())

# Load CSV data for obb_model_training
obb_model_training = 'yolov8/data-obb/obb_model_training.csv'
obb_df = pd.read_csv(obb_model_training)
print ("obb_df: ", obb_df)
print("Column names in obb_df:", obb_df.columns.tolist())
obb_df.columns = obb_df.columns.str.strip()
print("Column names in obb_df:", obb_df.columns.tolist())

# Calculate F1 score if not in CSV
standard_df['F1'] = 2 * (standard_df['metrics/precision(B)'] * standard_df['metrics/recall(B)']) / \
                   (standard_df['metrics/precision(B)'] + standard_df['metrics/recall(B)'])
# standard_df['F1'] = 5
# Extract metrics for plotting
standard_epochs = standard_df['epoch']
standard_precision = standard_df['metrics/precision(B)']
standard_recall = standard_df['metrics/recall(B)']
standard_f1 = standard_df['F1']

# Calculate F1 score if not in CSV
obb_df['F1'] = 2 * (obb_df['metrics/precision(B)'] * obb_df['metrics/recall(B)']) / \
                   (obb_df['metrics/precision(B)'] + obb_df['metrics/recall(B)'])
# obb_df['F1'] = 5
# Extract metrics for plotting
obb_epochs = obb_df['epoch']
obb_precision = obb_df['metrics/precision(B)']
obb_recall = obb_df['metrics/recall(B)']
obb_f1 = obb_df['F1']


# def plot_metrics(epochs, precision, recall, f1):
#     fig, axs = plt.subplots(3, 1, figsize=(10, 15))

#     # Precision-Recall Curve
#     axs[0].plot(recall, precision, label='Precision-Recall')
#     axs[0].set_xlabel('Recall')
#     axs[0].set_ylabel('Precision')
#     axs[0].set_title('Precision-Recall Curve')
#     axs[0].legend()
#     # Precision and Recall over epochs
#     axs[1].plot(epochs, precision, label='Precision', color='b')
#     axs[1].plot(epochs, recall, label='Recall', color='r')
#     axs[1].set_xlabel('Epoch')
#     axs[1].set_ylabel('Score')
#     axs[1].set_title('Precision and Recall over Epochs')
#     axs[1].legend()
#     # F1 Score over epochs
#     axs[2].plot(epochs, f1, label='F1 Score', color='g')
#     axs[2].set_xlabel('Epoch')
#     axs[2].set_ylabel('F1 Score')
#     axs[2].set_title('F1 Score over Epochs')
#     axs[2].legend()
#     fig.tight_layout()
    
#     # Convert plot to PNG image and display
#     buf = BytesIO()
#     plt.savefig(buf, format="png")
#     st.image(buf, caption='Training Metrics', use_column_width=True)
#     plt.close(fig)

def plot_metrics(epochs, precision, recall, f1):

	# Assuming epochs, precision, recall, f1 are already defined as lists or numpy arrays
	# Convert them into a pandas DataFrame
	data = pd.DataFrame({
		'Epoch': epochs,
		'Precision': precision,
		'Recall': recall,
		'F1 Score': f1
	})

	# Precision-Recall Curve (assuming precision and recall are of the same length and directly related)
	pr_data = pd.DataFrame({
		'Recall': recall,
		'Precision': precision
	})
	st.caption('Precision-Recall Curve')
	st.line_chart(pr_data, width=0, height=300, use_container_width=True)
	st.write("\n\n")
	# Precision and Recall over Epochs
	prec_recall_data = data[['Epoch', 'Precision', 'Recall']].set_index('Epoch')
	st.caption('Precision and Recall over Epochs')
	st.line_chart(prec_recall_data, width=0, height=300, use_container_width=True)
	st.write("\n\n")
	# F1 Score over Epochs
	f1_data = data[['Epoch', 'F1 Score']].set_index('Epoch')
	st.caption('F1 Score over Epochs')
	st.line_chart(f1_data, width=0, height=300, use_container_width=True)

def plot_metrics(epochs, precision_yolov8, recall_yolov8, precision_yolov8_obb, recall_yolov8_obb):
    # Create a DataFrame for precision and recall data
    data = pd.DataFrame({
        'Epoch': epochs,
        'Precision YOLOv8': precision_yolov8,
        'Recall YOLOv8': recall_yolov8,
        'Precision YOLOv8-OBB': precision_yolov8_obb,
        'Recall YOLOv8-OBB': recall_yolov8_obb
    }).melt(id_vars=['Epoch'], var_name='Metric', value_name='Value')
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=data, x='Epoch', y='Value', hue='Metric', style='Metric', markers=True, dashes=False, ax=ax)
    plt.title('Model Comparison: YOLOv8 vs YOLOv8-OBB')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend(title='Metrics', title_fontsize='13', fontsize='10', loc='upper left')
    plt.tight_layout()
    
    st.pyplot(fig)
    
def plot_metrics(epochs, precision_yolov8, recall_yolov8, f1_yolov8, precision_yolov8_obb, recall_yolov8_obb, f1_yolov8_obb):
    # Extend the DataFrame to include F1 scores
    data = pd.DataFrame({
        'Epoch': epochs,
        'Precision YOLOv8': precision_yolov8,
        'Recall YOLOv8': recall_yolov8,
        'F1 YOLOv8': f1_yolov8,
        'Precision YOLOv8-OBB': precision_yolov8_obb,
        'Recall YOLOv8-OBB': recall_yolov8_obb,
        'F1 YOLOv8-OBB': f1_yolov8_obb
    }).melt(id_vars=['Epoch'], var_name='Metric', value_name='Value')
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=data, x='Epoch', y='Value', hue='Metric', style='Metric', markers=True, dashes=False, ax=ax)
    plt.title('Model Comparison: YOLOv8 vs YOLOv8-OBB')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend(title='Metrics', title_fontsize='13', fontsize='10', loc='upper left')
    plt.tight_layout()
    
    st.pyplot(fig)

def plot_individual_metrics(epochs, standard_precision, standard_recall, standard_f1, obb_precision, obb_recall, obb_f1):
    metrics_data = {
        'Precision': {'YOLOv8': standard_precision, 'YOLOv8-OBB': obb_precision},
        'Recall': {'YOLOv8': standard_recall, 'YOLOv8-OBB': obb_recall},
        'F1 Score': {'YOLOv8': standard_f1, 'YOLOv8-OBB': obb_f1}
    }

    for metric_name, metric_values in metrics_data.items():
        data = pd.DataFrame({
            'Epoch': epochs,
            f'{metric_name} YOLOv8': metric_values['YOLOv8'],
            f'{metric_name} YOLOv8-OBB': metric_values['YOLOv8-OBB']
        }).melt(id_vars=['Epoch'], var_name='Model', value_name=metric_name)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=data, x='Epoch', y=metric_name, hue='Model', style='Model', markers=True, dashes=False, ax=ax)
        plt.title(f'Model Comparison: {metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.legend(title='Model', title_fontsize='13', fontsize='10', loc='upper left')
        plt.tight_layout()

        st.pyplot(fig)
        
def plot_confusion_matrix(standard_cm, obb_cm, labels):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # standard_cm = np.nan_to_num(standard_cm).astype(int)
    # obb_cm = np.nan_to_num(obb_cm).astype(int)
    print("standard_cm: ", standard_cm)
    print("obb_cm: ", obb_cm)
    sns.heatmap(standard_cm, annot=True, fmt="0.2f", ax=axs[0], cmap='Blues', cbar=False)
    axs[0].set_title('Standard Model Confusion Matrix')
    axs[0].set_xticklabels(labels)
    axs[0].set_yticklabels(labels)

    sns.heatmap(obb_cm, annot=True, fmt="0.2f", ax=axs[1], cmap='Oranges', cbar=False)
    axs[1].set_title('OBB Model Confusion Matrix')
    axs[1].set_xticklabels(labels)
    axs[1].set_yticklabels(labels)

    plt.tight_layout()
    st.pyplot(fig)

def comparison_page():
    if source == 'Upload MP4 File':
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
                tmpfile.write(uploaded_file.read())
                tmpfile_path = tmpfile.name
            
            stframe_standard = st.empty()
            stframe_obb = st.empty()
            
            # Display metrics plots
            # plot_metrics(epochs, precision, recall, f1) # yahan change
            # plot_metrics(standard_epochs, standard_precision, standard_recall, standard_f1, obb_precision, obb_recall, obb_f1)
            plot_individual_metrics(standard_epochs, standard_precision, standard_recall, standard_f1, obb_precision, obb_recall, obb_f1)
            
            # plot_confusion_matrix()
            standard_cm = np.array([[0.66, 0.06], [0.09, 0.86]]) 
            obb_cm = np.array([[0.75, 0.11], [0.07, 0.81]]) 
            labels = ['cheating', 'good']
            plot_confusion_matrix(standard_cm, obb_cm, labels)


            def process_and_compare_video():
                video_capture = cv2.VideoCapture(tmpfile_path)
                while video_capture.isOpened():
                    ret, frame = video_capture.read()
                    if not ret:
                        break

                    frame_copy_standard = frame.copy()
                    frame_copy_obb = frame.copy()

                    results_standard = detection.detect_objects(models["YOLOv8 Standard"], frame_copy_standard)
                    results_obb = detection.detect_objects(models["YOLOv8 OBB"], frame_copy_obb)

                    if results_standard:
                        for result in results_standard:
                            boxes = result.boxes
                            if boxes:
                                for box in boxes:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    class_id = int(box.cls)
                                    confidence = float(box.conf)
                                    class_name = class_names.get(class_id, 'Unknown')
                                    color = class_colors.get(class_name.lower(), (0, 255, 0))
                                    cv2.rectangle(frame_copy_standard, (x1, y1), (x2, y2), color, 2)
                                    label = f'{class_name} {confidence:.2f}'
                                    cv2.putText(frame_copy_standard, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if results_obb:
                        for result in results_obb:
                            obbs = result.obb
                            if obbs:
                                for obb in obbs:
                                    class_id = int(obb.cls)
                                    confidence = float(obb.conf)
                                    class_name = class_names.get(class_id, 'Unknown')
                                    label = f'{class_name} {confidence:.2f}'
                                    color = class_colors.get(class_name.lower(), (0, 255, 0))
                                    cx, cy, w, h, theta = obb.xywhr[0]
                                    vertices = obb_to_vertices(cx, cy, w, h, theta)
                                    draw_obb_with_label(frame_copy_obb, vertices, label, color)

                    annotated_frame_standard = cv2.cvtColor(frame_copy_standard, cv2.COLOR_BGR2RGB)
                    annotated_frame_obb = cv2.cvtColor(frame_copy_obb, cv2.COLOR_BGR2RGB)

                    stframe_standard.image(annotated_frame_standard, caption='YOLOv8 Standard', channels='RGB')
                    stframe_obb.image(annotated_frame_obb, caption='YOLOv8 OBB', channels='RGB')

                video_capture.release()
                cv2.destroyAllWindows()

            process_and_compare_video()
            
            


if page == 'Detection':
    detection_page()
elif page == 'Comparison':
    comparison_page()
