import cv2

def detect_objects(model, frame):
    results = model(frame)
    # print("[DETECTION.PY] Results: ", results)
    return results
