# post_processing.py

import cv2

def convert_yolo_results_to_detections(results, image_shape):
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            detections.append((x1, y1, x2, y2, confidence, class_id))
    return detections

def filter_detections(detections, confidence_threshold):
    return [d for d in detections if d[4] >= confidence_threshold]

def draw_boxes(image, detections, class_names):
    for (x1, y1, x2, y2, confidence, class_id) in detections:
        label = f"{class_names[class_id]}: {confidence:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return image
