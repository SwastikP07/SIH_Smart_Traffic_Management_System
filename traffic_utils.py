import cv2
from ultralytics import YOLO

# ------------------------------
# Load YOLOv11 model
# ------------------------------
def load_yolo_model(model_path="yolov11n.pt"):
    """Load YOLOv11 model (ensure .pt file is in the project folder)."""
    return YOLO(model_path)

# ------------------------------
# Process frame and return counts + annotated frame
# ------------------------------
def process_frame(frame, model, conf=0.25):
    results = model(frame, conf=conf, verbose=False)[0]  # YOLOv11 inference
    annotated_frame = results.plot()  # draw detections

    counts = {"car": 0, "truck": 0, "bus": 0, "motorbike": 0}
    for r in results.boxes.data.cpu().numpy():
        cls_id = int(r[5])
        if cls_id == 2:        # car
            counts["car"] += 1
        elif cls_id == 5:      # bus
            counts["bus"] += 1
        elif cls_id == 7:      # truck
            counts["truck"] += 1
        elif cls_id == 3:      # motorbike
            counts["motorbike"] += 1

    return counts, annotated_frame

# ------------------------------
# Allocate green light times
# ------------------------------
def allocate_green_time(counts_per_direction):
    total = sum(counts_per_direction.values())
    cycle_time = 60
    min_green = 5

    if total == 0:
        return {k: cycle_time // 4 for k in counts_per_direction}

    green_times = {}
    for direction, count in counts_per_direction.items():
        share = (count / total) * cycle_time
        green_times[direction] = max(min_green, int(share))

    diff = cycle_time - sum(green_times.values())
    if diff != 0:
        max_dir = max(green_times, key=green_times.get)
        green_times[max_dir] += diff

    return green_times
