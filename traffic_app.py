import streamlit as st
import cv2
import tempfile
import time
from ultralytics import YOLO
from collections import defaultdict, deque

# ----------------- Page Config -----------------
st.set_page_config(page_title="🚦 Smart Traffic Light — Pro", layout="wide")
st.title("🚦 Smart Traffic Light — Tunable & Smarter Detection")

# ----------------- Sidebar controls -----------------
st.sidebar.header("⚙️ Detection Settings")

model_choice = st.sidebar.selectbox("YOLO model", ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt"))
conf_thresh = st.sidebar.slider("Detection Sensitivity (confidence)", 0.01, 0.9, 0.10, 0.01)
iou_thresh = st.sidebar.slider("NMS IoU threshold", 0.1, 0.9, 0.4, 0.05)
resize_width = st.sidebar.slider("Resize longer side to (px)", 320, 1600, 960, 32)
min_box_area = st.sidebar.slider("Minimum box area (pixels)", 100, 50000, 400, 100)

# Vehicle selection
vehicle_options = {"Car": 2, "Motorcycle": 3, "Bus": 5, "Truck": 7, "Bicycle": 1}
selected_classes = st.sidebar.multiselect(
    "Count these vehicle types:", list(vehicle_options.keys()), ["Car", "Bus", "Truck"]
)
selected_class_ids = [vehicle_options[v] for v in selected_classes]

# Traffic light logic
STABLE_FRAMES = st.sidebar.slider("Stabilization frames before switching", 1, 12, 3)
MAX_GREEN_TIME = st.sidebar.slider("Max green time (sec)", 5, 30, 12)
FRAME_SKIP = st.sidebar.slider("Skip frames (higher = faster)", 1, 5, 1)

st.sidebar.markdown("---")
st.sidebar.info("Tip: Lower confidence → more detections.\nUse bigger models for more accuracy.")

# ----------------- File Upload -----------------
uploaded_file = st.file_uploader("📹 Upload a traffic video", type=["mp4", "mov", "avi", "mkv"])

# ----------------- Helper Functions -----------------
def get_sector(cx, cy, w, h):
    """Assign detected vehicles to sectors (N, S, E, W)."""
    center_tol = w // 4
    if cy < h // 2 and abs(cx - w // 2) <= center_tol:
        return "North"
    elif cy >= h // 2 and abs(cx - w // 2) <= center_tol:
        return "South"
    elif cx < w // 2:
        return "West"
    else:
        return "East"

# ----------------- Main -----------------
if uploaded_file is not None:
    # Save upload to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    if not cap.isOpened():
        st.error("❌ Could not open the uploaded video.")
        st.stop()

    with st.spinner(f"Loading YOLO model {model_choice} ..."):
        model = YOLO(model_choice)

    # Layout: Video left, Stats right
    col1, col2 = st.columns([3, 1])
    with col1:
        stframe = st.empty()
    with col2:
        st.subheader("📊 Live Stats")
        north_m = st.metric("North", 0)
        south_m = st.metric("South", 0)
        east_m = st.metric("East", 0)
        west_m = st.metric("West", 0)
        green_display = st.empty()
        total_display = st.empty()

    # State vars
    prev_green_side = None
    green_counter = 0
    green_timer_start = time.time()
    green_counts = defaultdict(int)  # cumulative green lights given
    total_detected_all = 0
    round_robin = deque(["North", "East", "South", "West"])

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        orig_h, orig_w = frame.shape[:2]

        # Resize if needed
        max_side = max(orig_w, orig_h)
        if max_side > resize_width:
            scale = resize_width / float(max_side)
            resized = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)))
        else:
            scale = 1.0
            resized = frame.copy()
            scale = 1.0
        inv_scale = 1.0 / scale

        # Run YOLO
        results = model(resized, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]

        counts = {'North': 0, 'South': 0, 'East': 0, 'West': 0}
        total_detected = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in selected_class_ids:
                continue

            x1r, y1r, x2r, y2r = map(float, box.xyxy[0])
            x1, y1, x2, y2 = map(int, [x1r * inv_scale, y1r * inv_scale, x2r * inv_scale, y2r * inv_scale])

            box_w, box_h = x2 - x1, y2 - y1
            if box_w * box_h < min_box_area:
                continue

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            sector = get_sector(cx, cy, orig_w, orig_h)
            counts[sector] += 1
            total_detected += 1
            total_detected_all += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(frame, sector, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Decide green side
        max_count = max(counts.values())
        if max_count > 0:
            top_sides = [k for k, v in counts.items() if v == max_count]

            if len(top_sides) == 1:
                green_side = top_sides[0]
            else:
                # Use previous if in tie, else round robin
                if prev_green_side in top_sides:
                    green_side = prev_green_side
                else:
                    while round_robin[0] not in top_sides:
                        round_robin.rotate(-1)
                    green_side = round_robin[0]
                    round_robin.rotate(-1)

            if green_side == prev_green_side:
                green_counter += 1
            else:
                green_counter = 1
                green_timer_start = time.time()
                prev_green_side = green_side

            if green_counter < STABLE_FRAMES:
                green_side = prev_green_side

            # Check max green time
            if time.time() - green_timer_start > MAX_GREEN_TIME:
                round_robin.rotate(-1)
                green_side = round_robin[0]
                prev_green_side = green_side
                green_timer_start = time.time()
                green_counter = 1

            green_counts[green_side] += 1
        else:
            green_side = None

        # Overlay
        cv2.putText(frame, f"Green: {green_side if green_side else 'None'}",
                    (10, orig_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0) if green_side else (0, 0, 255), 3)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        # Update metrics
        north_m.metric("North", counts["North"])
        south_m.metric("South", counts["South"])
        east_m.metric("East", counts["East"])
        west_m.metric("West", counts["West"])

        if green_side:
            green_display.success(f"🟢 GREEN → *{green_side}* (active {int(time.time() - green_timer_start)}s)")
        total_display.info(f"🔎 Total detected so far: {total_detected_all} | "
                           f"Greens: {dict(green_counts)}")

        time.sleep(0.02)

    cap.release()
    cv2.destroyAllWindows()        