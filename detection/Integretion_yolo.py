from ultralytics import YOLO
import numpy as np
import cv2

# --- Load model ---
model = YOLO(r"path_to_your_best.pt")

# --- Load stereo pair ---
left = cv2.imread("left_frame.png")
right = cv2.imread("right_frame.png")
grayL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

# Compute disparity map
stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=5)
disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

# --- YOLO detection ---
results = model(left)

# Camera parameters
f = 700   # focal length in pixels
B = 0.15  # baseline (m)

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        # Extract disparity region
        region = disparity[y1:y2, x1:x2]
        valid = region[region > 0]
        if len(valid) > 0:
            mean_disp = np.mean(valid)
            Z = (f * B) / mean_disp
            print(f"{label} at approx {Z:.2f} m")

        # Draw result
        cv2.rectangle(left, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(left, f"{label} {Z:.2f}m", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

cv2.imshow("Depth + Detection", left)
cv2.waitKey(0)
