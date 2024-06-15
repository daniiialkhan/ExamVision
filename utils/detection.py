import cv2

def detect_objects(model, frame):
    results = model(frame)
    return results
