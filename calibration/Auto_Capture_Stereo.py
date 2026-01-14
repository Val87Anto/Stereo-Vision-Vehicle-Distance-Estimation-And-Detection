import cv2
import os
import time

# =====================
# SETTINGS
# =====================
LEFT_ID = 0
RIGHT_ID = 1

PATTERN = (8, 5)         # chessboard inside corners
MAX_PAIRS = 70            # how many to capture
SAVE_DIR = "stereo_pairs"
CAPTURE_DELAY = 1.0       # seconds between captures

os.makedirs(SAVE_DIR + "/L", exist_ok=True)
os.makedirs(SAVE_DIR + "/R", exist_ok=True)

# =====================
# OPEN CAMERAS
# =====================
capL = cv2.VideoCapture(LEFT_ID, cv2.CAP_DSHOW)
capR = cv2.VideoCapture(RIGHT_ID, cv2.CAP_DSHOW)

capL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

capR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =====================
# MAIN LOOP
# =====================
pair_id = 0
last_time = 0

while pair_id < MAX_PAIRS:

    okL, frameL = capL.read()
    okR, frameR = capR.read()

    if not okL or not okR:
        print("Camera read error!")
        break

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    foundL, cornersL = cv2.findChessboardCornersSB(grayL, PATTERN)
    foundR, cornersR = cv2.findChessboardCornersSB(grayR, PATTERN)

    showL = frameL.copy()
    showR = frameR.copy()

    if foundL: cv2.drawChessboardCorners(showL, PATTERN, cornersL, foundL)
    if foundR: cv2.drawChessboardCorners(showR, PATTERN, cornersR, foundR)

    preview = cv2.hconcat([showL, showR])
    cv2.putText(preview, f"{pair_id}/{MAX_PAIRS}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)

    cv2.imshow("Stereo Capture", preview)

    # Auto-capture
    now = time.time()
    if foundL and foundR and now - last_time > CAPTURE_DELAY:

        cv2.imwrite(f"{SAVE_DIR}/L/left_{pair_id:03d}.png", frameL)
        cv2.imwrite(f"{SAVE_DIR}/R/right_{pair_id:03d}.png", frameR)

        print(f"[OK] Saved pair {pair_id}")
        pair_id += 1
        last_time = now

    if cv2.waitKey(1) & 0xFF == 27:
        break

capL.release()
capR.release()
cv2.destroyAllWindows()

print("DONE!")
