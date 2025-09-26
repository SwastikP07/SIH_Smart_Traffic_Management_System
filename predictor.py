import cv2
from ultralytics import YOLO

# ------------------------------
# Load YOLOv11 model
# ------------------------------
def load_yolo_model(model_path="yolov11n.pt"):
    return YOLO(model_path)

# ------------------------------
# Process frame and return counts + annotated frame
# ------------------------------
def process_frame(frame, model, conf=0.25, annotate=True):
    """
    Run YOLO on a frame and return counts + annotated frame.
    Only detects cars, trucks, buses, motorbikes.
    """
    # Run YOLO only on selected classes for speed
    results = model(frame, conf=conf, classes=[2, 3, 5, 7])[0]

    # Draw boxes only if needed
    annotated_frame = results.plot() if annotate else frame.copy()

    # Vehicle counts
    counts = {"car": 0, "truck": 0, "bus": 0, "motorbike": 0}
    for r in results.boxes.data.cpu().numpy():
        cls_id = int(r[5])
        if cls_id == 2:
            counts["car"] += 1
        elif cls_id == 5:
            counts["bus"] += 1
        elif cls_id == 7:
            counts["truck"] += 1
        elif cls_id == 3:
            counts["motorbike"] += 1

    return counts, annotated_frame

# ------------------------------
# Compute green times per direction
# ------------------------------
def compute_green_times(counts_per_direction):
    """
    counts_per_direction: dict = {"North":n, "East":n, "South":n, "West":n}
    Returns: green times in seconds per direction
    """
    total = sum(counts_per_direction.values())
    cycle_time = 60  # total cycle time (seconds)
    min_green = 5

    if total == 0:
        return {k: cycle_time // 4 for k in counts_per_direction}

    green_times = {}
    for d, c in counts_per_direction.items():
        share = (c / total) * cycle_time
        green_times[d] = max(min_green, int(share))

    # Fix rounding errors: make total = cycle_time
    diff = cycle_time - sum(green_times.values())
    if diff > 0:
        max_dir = max(green_times, key=green_times.get)
        green_times[max_dir] += diff

    return green_times
