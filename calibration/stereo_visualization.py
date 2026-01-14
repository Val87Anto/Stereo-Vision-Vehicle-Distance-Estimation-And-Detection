import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ================= CONFIG =================
CHECKERBOARD = (8, 5)          # INNER corners
SQUARE_SIZE = 0.020            # meters

LEFT_DIR = r"C:\Users\ASUS\Downloads\Semester 5_Automotive\Computer Vision\FINAL\VeryVeryNewVehicleDetection\car and truck detection.v3i.yolov8\Fix\stereo_pairs\L"
RIGHT_DIR = r"C:\Users\ASUS\Downloads\Semester 5_Automotive\Computer Vision\FINAL\VeryVeryNewVehicleDetection\car and truck detection.v3i.yolov8\Fix\stereo_pairs\R"

SAMPLE_LEFT  = r"C:\Users\ASUS\Downloads\Semester 5_Automotive\Computer Vision\FINAL\VeryVeryNewVehicleDetection\car and truck detection.v3i.yolov8\Fix\stereo_pairs\L\left_043.png"
SAMPLE_RIGHT = r"C:\Users\ASUS\Downloads\Semester 5_Automotive\Computer Vision\FINAL\VeryVeryNewVehicleDetection\car and truck detection.v3i.yolov8\Fix\stereo_pairs\R\right_043.png"

# =========================================
left_imgs  = sorted(glob.glob(os.path.join(LEFT_DIR, "*.png")))
right_imgs = sorted(glob.glob(os.path.join(RIGHT_DIR, "*.png")))

assert len(left_imgs) == len(right_imgs) > 10, "Bad stereo image count"
print(f"📸 Using {len(left_imgs)} synchronized stereo pairs")

# =========================================
# Object points
# =========================================
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints, imgpointsL, imgpointsR = [], [], []
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

# =========================================
# Detect chessboard corners
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
# Mono calibration
# =========================================
print("\nCalibrating mono cameras...")
rmsL, K1, D1, rvecsL, tvecsL = cv2.calibrateCamera(
    objpoints, imgpointsL, image_size, None, None
)
rmsR, K2, D2, rvecsR, tvecsR = cv2.calibrateCamera(
    objpoints, imgpointsR, image_size, None, None
)

print(f"Left RMS : {rmsL:.4f}")
print(f"Right RMS: {rmsR:.4f}")

# =========================================
# Stereo calibration
# =========================================
print("\nStereo calibration...")
flags = cv2.CALIB_FIX_INTRINSIC
rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    K1, D1, K2, D2,
    image_size, criteria=criteria, flags=flags
)

baseline = np.linalg.norm(T)

print("\n===== STEREO RESULT =====")
print(f"Stereo RMS : {rms:.4f} px")
print(f"Baseline   : {baseline:.4f} m")

# =========================================
# Rotation & Translation
# =========================================
print("\n===== STEREO EXTRINSICS =====")
print("\nRotation Matrix R:")
print(R)

print("\nTranslation Vector T [m]:")
print(T.reshape(-1))

def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return np.degrees([x, y, z])

euler = rotationMatrixToEulerAngles(R)
print("\nEuler Angles (deg)")
print(f"Roll  : {euler[0]:.3f}")
print(f"Pitch : {euler[1]:.3f}")
print(f"Yaw   : {euler[2]:.3f}")

# =========================================
# Rectification
# =========================================
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K1, D1, K2, D2, image_size, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
)

np.savez("stereo_clean.npz",
    K1=K1, D1=D1, K2=K2, D2=D2,
    R=R, T=T, R1=R1, R2=R2,
    P1=P1, P2=P2, Q=Q,
    image_size=image_size, rms=rms
)

# =========================================
# Load sample images
# =========================================
imgL = cv2.imread(SAMPLE_LEFT)
imgR = cv2.imread(SAMPLE_RIGHT)

# -------- Undistortion --------
undistL = cv2.undistort(imgL, K1, D1)
undistR = cv2.undistort(imgR, K2, D2)

plt.figure(figsize=(10,6))
plt.subplot(2,2,1); plt.imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)); plt.title("Left Distorted")
plt.subplot(2,2,2); plt.imshow(cv2.cvtColor(undistL, cv2.COLOR_BGR2RGB)); plt.title("Left Undistorted")
plt.subplot(2,2,3); plt.imshow(cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)); plt.title("Right Distorted")
plt.subplot(2,2,4); plt.imshow(cv2.cvtColor(undistR, cv2.COLOR_BGR2RGB)); plt.title("Right Undistorted")
plt.tight_layout(); plt.show()

# -------- Rectification --------
mapLx, mapLy = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

rectL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
rectR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)

combined = np.hstack((rectL, rectR))
for y in range(0, combined.shape[0], 40):
    cv2.line(combined, (0,y), (combined.shape[1],y), (0,255,0), 1)

plt.figure(figsize=(12,6))
plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
plt.title("Rectified Images with Epipolar Lines")
plt.axis("off")
plt.show()

# -------- Disparity --------
stereo = cv2.StereoBM_create(numDisparities=128, blockSize=15)
disp = stereo.compute(
    cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY),
    cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
)

plt.figure(figsize=(8,5))
plt.imshow(disp, cmap="plasma")
plt.colorbar(label="Disparity")
plt.title("Disparity Map")
plt.show()

# -------- Reprojection Error (CORRECT) --------
errors = []
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecsL[i], tvecsL[i], K1, D1
    )
    err = cv2.norm(imgpointsL[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    errors.append(err)

plt.figure(figsize=(7,4))
plt.plot(errors, marker="o")
plt.title("Reprojection Error per Image (Left Camera)")
plt.xlabel("Image Index")
plt.ylabel("Pixel Error")
plt.grid()
plt.show()

print(f"\nMean reprojection error: {np.mean(errors):.4f} px")

# -------- 3D Point Cloud --------
points_3D = cv2.reprojectImageTo3D(disp.astype(np.float32)/16.0, Q)
mask = disp > disp.min()

pts = points_3D[mask]

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=1)
ax.set_title("3D Point Cloud")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# -------- Camera Pose Visualization --------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")

camL = np.zeros(3)
camR = T.reshape(3)

ax.scatter(*camL, c='blue', s=80, label="Left Camera")
ax.scatter(*camR, c='red', s=80, label="Right Camera")
ax.plot([0, camR[0]], [0, camR[1]], [0, camR[2]], 'k--', label="Baseline")

ax.set_title("Stereo Camera Geometry")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.legend()
ax.view_init(20, -60)
plt.show()

print("\n✅ Stereo vision pipeline complete")
 