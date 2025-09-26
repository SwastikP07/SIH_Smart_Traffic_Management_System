# # import streamlit as st
# # import cv2
# # import tempfile
# # import time
# # from ultralytics import YOLO  # for vehicle detection

# # # ----------------------------------------------------------------------
# # # Streamlit setup
# # st.set_page_config(page_title="Smart Traffic Light – 90° Compass Grid", layout="wide")
# # st.title("🚦 Smart Traffic Light with 90° Compass Grid")

# # uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4","mov","avi","mkv"])

# # # ----------------------------------------------------------------------
# # # Helper: assign detected vehicles to North, South, East, West (quadrant-based)
# # def get_sector(cx, cy, w, h):
# #     if cy < h // 2 and abs(cx - w // 2) <= w // 4:   # top-middle → North
# #         return "North"
# #     elif cy >= h // 2 and abs(cx - w // 2) <= w // 4:  # bottom-middle → South
# #         return "South"
# #     elif cx < w // 2:   # left side → West
# #         return "West"
# #     else:               # right side → East
# #         return "East"
# # # ----------------------------------------------------------------------

# # if uploaded_file is not None:
# #     # Save uploaded file to temp
# #     tfile = tempfile.NamedTemporaryFile(delete=False)
# #     tfile.write(uploaded_file.read())
# #     cap = cv2.VideoCapture(tfile.name)

# #     if not cap.isOpened():
# #         st.error("❌ Could not open the uploaded video.")
# #     else:
# #         stframe = st.empty()

# #         # Load YOLO model (better than nano)
# #         model = YOLO("yolov8s.pt")  # try yolov8m.pt for even better accuracy

# #         prev_green_side = None
# #         green_counter = 0
# #         STABLE_FRAMES = 3  # consecutive frames needed to switch

# #         while cap.isOpened():
# #             ret, frame = cap.read()
# #             if not ret:
# #                 break

# #             h, w = frame.shape[:2]

# #             # Run YOLO inference with tuned thresholds
# #             results = model(frame, conf=0.1, iou=0.4, verbose=False)[0]

# #             # Vehicle counts per sector
# #             counts = {'North': 0, 'South': 0, 'East': 0, 'West': 0}

# #             # Loop over detections
# #             for box in results.boxes:
# #                 cls_id = int(box.cls[0])
# #                 conf = float(box.conf[0])
# #                 x1, y1, x2, y2 = map(int, box.xyxy[0])

# #                 # Only consider vehicles (car=2, motorcycle=3, bus=5, truck=7)
# #                 if cls_id in [2, 3, 5, 7]:
# #                     cx = (x1 + x2) // 2
# #                     cy = (y1 + y2) // 2
# #                     sector = get_sector(cx, cy, w, h)
# #                     counts[sector] += 1

# #                     # Draw detection box
# #                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
# #                     cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
# #                     cv2.putText(frame, sector, (x1, y1 - 5),
# #                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# #             # Determine green side with stabilization
# #             max_count = max(counts.values())
# #             if max_count > 0:
# #                 top_sides = [k for k, v in counts.items() if v == max_count]

# #                 if len(top_sides) == 1:
# #                     green_side = top_sides[0]
# #                 else:
# #                     # If tie, keep previous green if it’s part of the tie
# #                     green_side = prev_green_side if prev_green_side in top_sides else top_sides[0]

# #                 # Stabilize: only change after STABLE_FRAMES
# #                 if green_side == prev_green_side:
# #                     green_counter += 1
# #                 else:
# #                     green_counter = 1
# #                     prev_green_side = green_side

# #                 if green_counter < STABLE_FRAMES:
# #                     green_side = prev_green_side
# #             else:
# #                 green_side = None

# #             # Draw compass cross
# #             cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 0, 0), 2)
# #             cv2.line(frame, (0, h // 2), (w, h // 2), (255, 0, 0), 2)
# #             cv2.putText(frame, "N", (w // 2 - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
# #             cv2.putText(frame, "S", (w // 2 - 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
# #             cv2.putText(frame, "W", (10, h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
# #             cv2.putText(frame, "E", (w - 30, h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# #             # Show green side
# #             if green_side:
# #                 cv2.putText(frame, f"Green Light: {green_side}", (20, h - 20),
# #                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

# #             # Streamlit display
# #             stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
# #                           channels="RGB",
# #                           use_container_width=True)

# #             # Slow down update to reduce flicker
# #             time.sleep(0.2)  # ~5 FPS

# #         cap.release()
        
        
        






# # # traffic_light_tunable.py
# # import streamlit as st
# # import cv2
# # import tempfile
# # import time
# # from ultralytics import YOLO

# # st.set_page_config(page_title="Smart Traffic Light – Tunable Detection", layout="wide")
# # st.title("🚦 Smart Traffic Light — Tunable Vehicle Detection")

# # # ----------------- Sidebar controls -----------------
# # st.sidebar.header("Detection Settings")

# # model_choice = st.sidebar.selectbox("YOLO model", ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt"))
# # conf_thresh = st.sidebar.slider("Detection Sensitivity (confidence) — lower = more detections",
# #                                 min_value=0.01, max_value=0.9, value=0.10, step=0.01)
# # iou_thresh = st.sidebar.slider("NMS IoU threshold", min_value=0.1, max_value=0.9, value=0.4, step=0.05)
# # resize_width = st.sidebar.slider("Resize longer side to (px) — lower = faster, less detail",
# #                                  min_value=320, max_value=1600, value=1280, step=32)
# # min_box_area = st.sidebar.slider("Minimum box area (pixels) — ignore tiny detections",
# #                                  min_value=100, max_value=50000, value=400, step=100)
# # include_bicycle = st.sidebar.checkbox("Include bicycle in vehicle count (COCO class 1)", value=False)
# # STABLE_FRAMES = st.sidebar.slider("Stabilization frames before switching green", 1, 12, 3)

# # st.sidebar.markdown("---")
# # st.sidebar.markdown("Tip: Lower confidence → more detections. Use bigger model for more accuracy.")
# # # ----------------------------------------------------

# # uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "mov", "avi", "mkv"])

# # # ----------------------------------------------------------------------
# # # Helper: assign detected vehicles to North, South, East, West (quadrant-based)
# # def get_sector(cx, cy, w, h):
# #     # top-center region counts as North, bottom-center as South,
# #     # left half as West, right half as East.
# #     # Adjust the center-width threshold if you want a narrower/fatter center corridor.
# #     center_tol = w // 4
# #     if cy < h // 2 and abs(cx - w // 2) <= center_tol:
# #         return "North"
# #     elif cy >= h // 2 and abs(cx - w // 2) <= center_tol:
# #         return "South"
# #     elif cx < w // 2:
# #         return "West"
# #     else:
# #         return "East"
# # # ----------------------------------------------------------------------

# # if uploaded_file is not None:
# #     # save upload to temp file
# #     tfile = tempfile.NamedTemporaryFile(delete=False)
# #     tfile.write(uploaded_file.read())
# #     cap = cv2.VideoCapture(tfile.name)

# #     if not cap.isOpened():
# #         st.error("❌ Could not open the uploaded video.")
# #         st.stop()

# #     # load chosen YOLO model (will download if needed)
# #     with st.spinner(f"Loading model {model_choice} ..."):
# #         model = YOLO(model_choice)

# #     stframe = st.empty()
# #     info_area = st.empty()

# #     prev_green_side = None
# #     green_counter = 0

# #     # Run through frames
# #     while cap.isOpened():
# #         ret, frame = cap.read()
# #         if not ret:
# #             break

# #         orig_h, orig_w = frame.shape[:2]

# #         # Resize frame if longer side > resize_width
# #         max_side = max(orig_w, orig_h)
# #         if max_side > resize_width:
# #             scale = resize_width / float(max_side)
# #             resized_w = int(orig_w * scale)
# #             resized_h = int(orig_h * scale)
# #             resized = cv2.resize(frame, (resized_w, resized_h))
# #         else:
# #             scale = 1.0
# #             resized = frame.copy()

# #         # Run YOLO on resized frame with tuned thresholds
# #         # conf -> confidence threshold; iou -> NMS IoU
# #         results = model(resized, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]

# #         counts = {'North': 0, 'South': 0, 'East': 0, 'West': 0}
# #         total_detected = 0

# #         inv_scale = 1.0 / scale if scale != 0 else 1.0

# #         # Loop over detections
# #         for box in results.boxes:
# #             try:
# #                 cls_id = int(box.cls[0])
# #                 conf = float(box.conf[0])
# #                 x1r, y1r, x2r, y2r = map(float, box.xyxy[0])  # coords on resized frame
# #             except Exception:
# #                 # fallback if attributes differ
# #                 coords = box.xyxy
# #                 if len(coords) == 0:
# #                     continue
# #                 arr = coords[0]
# #                 x1r, y1r, x2r, y2r = map(float, arr)

# #                 cls_id = int(box.cls) if hasattr(box, "cls") else None
# #                 conf = float(box.conf) if hasattr(box, "conf") else 0.0

# #             # Map to original frame coordinates
# #             x1 = int(x1r * inv_scale)
# #             y1 = int(y1r * inv_scale)
# #             x2 = int(x2r * inv_scale)
# #             y2 = int(y2r * inv_scale)

# #             # compute area on original frame
# #             box_w = max(0, x2 - x1)
# #             box_h = max(0, y2 - y1)
# #             area = box_w * box_h

# #             # optionally filter by area
# #             if area < min_box_area:
# #                 continue

# #             # Only count vehicles: car=2, motorcycle=3, bus=5, truck=7 (+ bicycle=1 optional)
# #             vehicle_classes = [2, 3, 5, 7]
# #             if include_bicycle:
# #                 vehicle_classes.append(1)

# #             if cls_id in vehicle_classes:
# #                 cx = int((x1 + x2) / 2)
# #                 cy = int((y1 + y2) / 2)
# #                 sector = get_sector(cx, cy, orig_w, orig_h)
# #                 counts[sector] += 1
# #                 total_detected += 1

# #                 # draw box & label on original frame
# #                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
# #                 cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
# #                 label = f"{sector} {conf:.2f}"
# #                 cv2.putText(frame, label, (x1, max(0, y1 - 6)),
# #                             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

# #         # Decide green side (the side with maximum vehicles)
# #         max_count = max(counts.values())
# #         if max_count > 0:
# #             top_sides = [k for k, v in counts.items() if v == max_count]
# #             if len(top_sides) == 1:
# #                 green_side = top_sides[0]
# #             else:
# #                 # if tied, prefer previous green if in tie, else pick first
# #                 green_side = prev_green_side if prev_green_side in top_sides else top_sides[0]

# #             # stabilize across frames
# #             if green_side == prev_green_side:
# #                 green_counter += 1
# #             else:
# #                 green_counter = 1
# #                 prev_green_side = green_side

# #             if green_counter < STABLE_FRAMES:
# #                 green_side = prev_green_side
# #         else:
# #             green_side = None

# #         # Overlay counts & green side on frame
# #         overlay_lines = [
# #             f"Model: {model_choice}  Conf: {conf_thresh:.2f}  IoU: {iou_thresh:.2f}",
# #             f"Resize max-side: {resize_width}px  Min box area: {min_box_area}px",
# #             f"Detected vehicles (frame): {total_detected}",
# #             f"Counts → N:{counts['North']}  S:{counts['South']}  E:{counts['East']}  W:{counts['West']}",
# #             f"Green: {green_side if green_side else 'None'}"
# #         ]

# #         # draw overlay text
# #         y0 = 20
# #         for i, line in enumerate(overlay_lines):
# #             cv2.putText(frame, line, (10, y0 + i * 20),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

# #         # draw compass lines
# #         cv2.line(frame, (orig_w // 2, 0), (orig_w // 2, orig_h), (255, 0, 0), 1)
# #         cv2.line(frame, (0, orig_h // 2), (orig_w, orig_h // 2), (255, 0, 0), 1)
# #         cv2.putText(frame, "N", (orig_w // 2 - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# #         cv2.putText(frame, "S", (orig_w // 2 - 10, orig_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# #         cv2.putText(frame, "W", (10, orig_h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# #         cv2.putText(frame, "E", (orig_w - 30, orig_h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# #         # show green side in green color
# #         if green_side:
# #             cv2.putText(frame, f"GREEN → {green_side}", (10, orig_h - 40),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3, cv2.LINE_AA)

# #         # Streamlit update
# #         stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

# #         # Info panel (below or to side)
# #         info_area.markdown(
# #             f"**Frame counts:** North={counts['North']}  •  South={counts['South']}  •  East={counts['East']}  •  West={counts['West']}\n\n"
# #             f"**Detected this frame:** {total_detected}  •  **Green:** {green_side if green_side else 'None'}"
# #         )

# #         # small delay to reduce CPU usage & keep UI responsive
# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             break
# #         time.sleep(0.02)

# #     cap.release()
# #     cv2.destroyAllWindows()








# # traffic_light_simple.py
# import streamlit as st
# import cv2
# import tempfile
# import time
# from ultralytics import YOLO

# st.set_page_config(page_title="Smart Traffic Light – Simple", layout="wide")
# st.title("🚦 Smart Traffic Light — Vehicle Detection (Simple Mode)")

# # Sidebar: only 1 slider for sensitivity
# st.sidebar.header("Detection Control")
# conf_thresh = st.sidebar.slider("Detection Sensitivity — lower = more vehicles",
#                                 min_value=0.01, max_value=0.9, value=0.15, step=0.01)
# STABLE_FRAMES = 3  # keep green stable for a few frames

# uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "mov", "avi", "mkv"])

# # Assign detected vehicles to North, South, East, West
# def get_sector(cx, cy, w, h):
#     center_tol = w // 4
#     if cy < h // 2 and abs(cx - w // 2) <= center_tol:
#         return "North"
#     elif cy >= h // 2 and abs(cx - w // 2) <= center_tol:
#         return "South"
#     elif cx < w // 2:
#         return "West"
#     else:
#         return "East"

# if uploaded_file is not None:
#     # Save upload
#     tfile = tempfile.NamedTemporaryFile(delete=False)
#     tfile.write(uploaded_file.read())
#     cap = cv2.VideoCapture(tfile.name)

#     if not cap.isOpened():
#         st.error("❌ Could not open video.")
#         st.stop()

#     # Load a stronger model for better results
#     with st.spinner("Loading YOLOv8s model..."):
#         model = YOLO("yolov8s.pt")

#     stframe = st.empty()
#     info_area = st.empty()

#     prev_green_side = None
#     green_counter = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         orig_h, orig_w = frame.shape[:2]

#         # Resize automatically for speed/accuracy balance
#         max_side = max(orig_w, orig_h)
#         target_size = 1280
#         scale = target_size / float(max_side) if max_side > target_size else 1.0
#         if scale < 1.0:
#             new_w = int(orig_w * scale)
#             new_h = int(orig_h * scale)
#             resized = cv2.resize(frame, (new_w, new_h))
#         else:
#             resized = frame.copy()

#         results = model(resized, conf=conf_thresh, iou=0.45, verbose=False)[0]

#         counts = {'North': 0, 'South': 0, 'East': 0, 'West': 0}
#         total_detected = 0
#         inv_scale = 1.0 / scale

#         for box in results.boxes:
#             cls_id = int(box.cls[0])
#             x1r, y1r, x2r, y2r = map(float, box.xyxy[0])
#             x1 = int(x1r * inv_scale)
#             y1 = int(y1r * inv_scale)
#             x2 = int(x2r * inv_scale)
#             y2 = int(y2r * inv_scale)

#             # Only vehicles: car=2, motorcycle=3, bus=5, truck=7
#             if cls_id in [2, 3, 5, 7]:
#                 cx = int((x1 + x2) / 2)
#                 cy = int((y1 + y2) / 2)
#                 sector = get_sector(cx, cy, orig_w, orig_h)
#                 counts[sector] += 1
#                 total_detected += 1

#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
#                 cv2.putText(frame, sector, (x1, max(0, y1 - 6)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

#         # Decide green side
#         max_count = max(counts.values())
#         if max_count > 0:
#             top_sides = [k for k, v in counts.items() if v == max_count]
#             if len(top_sides) == 1:
#                 green_side = top_sides[0]
#             else:
#                 green_side = prev_green_side if prev_green_side in top_sides else top_sides[0]

#             if green_side == prev_green_side:
#                 green_counter += 1
#             else:
#                 green_counter = 1
#                 prev_green_side = green_side

#             if green_counter < STABLE_FRAMES:
#                 green_side = prev_green_side
#         else:
#             green_side = None

#         # Overlay info
#         cv2.putText(frame, f"Detected: {total_detected}", (10, 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#         cv2.putText(frame, f"Counts N:{counts['North']} S:{counts['South']} "
#                            f"E:{counts['East']} W:{counts['West']}", (10, 45),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#         if green_side:
#             cv2.putText(frame, f"GREEN → {green_side}", (10, orig_h - 20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

#         stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
#         info_area.markdown(f"**Detected vehicles:** {total_detected}  •  **Green:** {green_side or 'None'}")

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#         time.sleep(0.02)

#     cap.release()
#     cv2.destroyAllWindows()




# traffic_light_tunable_plus.py
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