import cv2

print("Scanning webcams...")

for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"Camera ID {i} not available")
        continue

    ret, frame = cap.read()
    if not ret:
        print(f"Camera ID {i} opened but cannot read frame")
        cap.release()
        continue

    print(f"Camera ID {i} is AVAILABLE — press SPACE to close")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Show camera ID on screen
        cv2.putText(frame, f"Camera ID: {i}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        cv2.imshow("Webcam Viewer", frame)

        key = cv2.waitKey(1) & 0xFF

        # Press SPACE to close this camera and go to next
        if key == 32:  # SPACE KEY
            print(f"Camera ID {i} closed.")
            break

        # ESC to stop entire scan
        if key == 27:
            print("Scan cancelled by user.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cap.release()
    cv2.destroyAllWindows()

print("Finished scanning all webcams.")
 