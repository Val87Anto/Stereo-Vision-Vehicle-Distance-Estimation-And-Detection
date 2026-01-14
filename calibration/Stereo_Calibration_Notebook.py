import cv2
import numpy as np
import glob
import os

# ================= CONFIG =================
CHECKERBOARD = (8, 5)
SQUARE_SIZE = 0.020  # meters

LEFT_DIR = r"C:\Users\ASUS\Downloads\Semester 5_Automotive\Computer Vision\FINAL\VeryVeryNewVehicleDetection\car and truck detection.v3i.yolov8\Fix\stereo_pairs\L"
RIGHT_DIR = r"C:\Users\ASUS\Downloads\Semester 5_Automotive\Computer Vision\FINAL\VeryVeryNewVehicleDetection\car and truck detection.v3i.yolov8\Fix\stereo_pairs\R"

# =========================================
left_imgs = sorted(glob.glob(os.path.join(LEFT_DIR, "*.png")))
right_imgs = sorted(glob.glob(os.path.join(RIGHT_DIR, "*.png")))

assert len(left_imgs) == len(right_imgs) > 10, "Bad stereo image count"

print(f"📸 Using {len(left_imgs)} synchronized stereo pairs")

# Object points
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpointsL = []
imgpointsR = []

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

# =========================================
# Detect corners
# =========================================
for i, (l, r) in enumerate(zip(left_imgs, right_imgs)):
    imgL = cv2.imread(l)
    imgR = cv2.imread(r)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCornersSB(grayL, CHECKERBOARD)
    retR, cornersR = cv2.findChessboardCornersSB(grayR, CHECKERBOARD)

    print(f"[{i:03d}] L={retL} R={retR}")

    if retL and retR:
        objpoints.append(objp)
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)

assert len(objpoints) >= 10, "Too few valid pairs"

image_size = grayL.shape[::-1]

# =========================================
# Mono calibration (ONCE)
# =========================================
print("\nCalibrating mono cameras...")
rmsL, K1, D1, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, image_size, None, None)
rmsR, K2, D2, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, image_size, None, None)

print(f"Left RMS : {rmsL:.4f}")
print(f"Right RMS: {rmsR:.4f}")

# =========================================
# Stereo calibration (FIX_INTRINSIC)
# =========================================
print("\nStereo calibration...")
flags = cv2.CALIB_FIX_INTRINSIC
rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpointsL,
    imgpointsR,
    K1, D1,
    K2, D2,
    image_size,
    criteria=criteria,
    flags=flags
)

baseline = np.linalg.norm(T)

print("\n===== STEREO RESULT =====")
print(f"Stereo RMS  : {rms:.4f} px  (GOOD < 1.0)")
print(f"Baseline    : {baseline:.4f} m (measure this physically!)")

# =========================================
# Rectification
# =========================================
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K1, D1, K2, D2, image_size, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
)

np.savez(
    "stereo_clean.npz",
    K1=K1, D1=D1,
    K2=K2, D2=D2,
    R=R, T=T,
    R1=R1, R2=R2,
    P1=P1, P2=P2,
    Q=Q,
    image_size=image_size,
    rms=rms
)

print("\n✅ Clean stereo calibration saved: stereo_clean.npz")
