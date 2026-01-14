import cv2
import numpy as np
from ultralytics import YOLO
import zmq
import time

# =====================================================
# SETTINGS (RECOMMENDED)
# =====================================================
LEFT_CAM_ID  = 0
RIGHT_CAM_ID = 1

CALIB_FILE = "stereo_clean.npz"
CONF_THRESH = 0.20            # Slightly higher = more stable
MAX_DEPTH = 30.0              # meters

YOLO_MODEL_PATH = r"C:\Users\ASUS\Downloads\Semester 5_Automotive\Computer Vision\FINAL\VeryVeryNewVehicleDetection\car and truck detection.v3i.yolov8\Fix\VehicleDetection_Hybrid\train_manual_conv3\weights\best.pt"

# =====================================================
# ZMQ (Stereo → SFV)
# =====================================================
context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.connect("tcp://127.0.0.1:7000")
print("[ZMQ] Connected")

# =====================================================
# LOAD STEREO CALIBRATION
# =====================================================
data = np.load(CALIB_FILE)

K1 = data["K1"]
D1 = data["D1"]
K2 = data["K2"]
D2 = data["D2"]
R  = data["R"]
T  = data["T"]
IMAGE_SIZE = tuple(data["image_size"])

baseline = float(np.linalg.norm(T))
print(f"[INFO] Baseline = {baseline:.3f} m")

# =====================================================
# RECTIFICATION
# =====================================================
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K1, D1, K2, D2,
    IMAGE_SIZE, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0
)

mapLx, mapLy = cv2.initUndistortRectifyMap(
    K1, D1, R1, P1, IMAGE_SIZE, cv2.CV_32FC1
)
mapRx, mapRy = cv2.initUndistortRectifyMap(
    K2, D2, R2, P2, IMAGE_SIZE, cv2.CV_32FC1
)

fx = P1[0, 0]
print(f"[INFO] fx = {fx:.2f}")

# =====================================================
# YOLO
# =====================================================
model = YOLO(YOLO_MODEL_PATH)
print("[YOLO] Classes:", model.names)

# =====================================================
# STEREO MATCHER (VEHICLE-TUNED)
# =====================================================
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=192,     # 192 = stable for ~30m
    blockSize=7,
    P1=8 * 3 * 7**2,
    P2=32 * 3 * 7**2,
    disp12MaxDiff=2,
    uniquenessRatio=12,
    speckleWindowSize=100,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# =====================================================
# CAMERAS (HARD SYNC)
# =====================================================
capL = cv2.VideoCapture(LEFT_CAM_ID, cv2.CAP_MSMF)
capR = cv2.VideoCapture(RIGHT_CAM_ID, cv2.CAP_MSMF)
time.sleep(1)

if not capL.isOpened() or not capR.isOpened():
    raise RuntimeError("❌ Failed to open cameras")

capL.set(cv2.CAP_PROP_FRAME_WIDTH,  IMAGE_SIZE[0])
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])
capR.set(cv2.CAP_PROP_FRAME_WIDTH,  IMAGE_SIZE[0])
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])

print("[INFO] Cameras opened")

# =====================================================
# MAIN LOOP
# =====================================================
while True:

    # ---------- HARD SYNC ----------
    capL.grab()
    capR.grab()
    retL, frameL = capL.retrieve()
    retR, frameR = capR.retrieve()

    if not retL or not retR:
        print("[ERROR] Frame grab failed")
        break

    # ---------- RECTIFY ----------
    rectL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    # ---------- DISPARITY ----------
    disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    disp[disp < 1.0] = np.nan

    # ---------- DEPTH ----------
    depth = np.full_like(disp, np.nan)
    valid = np.isfinite(disp)
    depth[valid] = (fx * baseline) / disp[valid]
    depth[depth > MAX_DEPTH] = np.nan

    # ---------- DEPTH VIS ----------
    depth_vis = np.nan_to_num(depth, nan=MAX_DEPTH)
    depth_vis = np.clip(depth_vis, 0.5, MAX_DEPTH)
    depth_vis = (depth_vis / MAX_DEPTH * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # ---------- YOLO ----------
    results = model(rectL, conf=CONF_THRESH, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id].lower()

        if cls_name not in ["car", "truck"]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # ---------- STABLE VEHICLE ROI ----------
        cx1 = int(x1 + 0.35 * (x2 - x1))
        cx2 = int(x2 - 0.35 * (x2 - x1))
        cy1 = int(y1 + 0.55 * (y2 - y1))
        cy2 = int(y2 - 0.15 * (y2 - y1))

        roi = depth[cy1:cy2, cx1:cx2]
        roi = roi[np.isfinite(roi)]

        if roi.size < 5:
            continue

        distance = float(np.median(roi))
        variance = float(np.var(roi))

        # ---------- SEND ----------
        socket.send_json({
            "timestamp": time.time(),
            "stereo": {
                "distance": distance,
                "variance": variance
            }
        })

        # ---------- DRAW ----------
        cv2.rectangle(rectL, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(
            rectL,
            f"{cls_name}: {distance:.2f} m",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

    # ---------- DISPLAY ----------
    cv2.imshow("Stereo YOLO + Distance", rectL)
    cv2.imshow("Depth Map", depth_color)

    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

# =====================================================
# CLEANUP
# =====================================================
capL.release()
capR.release()
cv2.destroyAllWindows()
