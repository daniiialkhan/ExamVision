from ultralytics import YOLO

# Define path to YOLOv8 model and dataset configuration
yolo_model_path = 'yolov8/models/yolov8n.pt'  # You can use different YOLOv8 models like yolov8n, yolov8s, etc.
dataset_config_path = 'yolov8/data/data.yaml'

# Load YOLO model
model = YOLO(yolo_model_path)

# Train the model
model.train(data=dataset_config_path, epochs=50, imgsz=640)
