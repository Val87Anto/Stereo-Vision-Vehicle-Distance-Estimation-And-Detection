from ultralytics import YOLO
import cv2

# --- Paths ---
model_path = r"C:\Users\ASUS\Downloads\Semester 5_Automotive\Computer Vision\FINAL\VeryVeryNewVehicleDetection\car and truck detection.v3i.yolov8\VehicleDetection_Hybrid\train_manual_conv3\weights\best.pt"
video_path = r"C:\Users\ASUS\Downloads\Semester 5_Automotive\Computer Vision\FINAL\VeryVeryNewVehicleDetection\car and truck detection.v3i.yolov8\TRUCKS ON M6 MOTORWAY IN HEAVY TRAFFIC AUGUST 2011 - DaveSpencer32 (720p, h264).mp4"

# --- Load model ---
model = YOLO(model_path)

# --- Run detection on video ---
results = model.predict(
    source=video_path,
    conf=0.4,        # Confidence threshold (adjust higher = fewer detections)
    save=True,       # Save output video
    show=False,      # Set to True if you want to open a preview window
    project="runs/detect",
    name="video_test_car_truck",
    device=0
)

print("✅ Detection complete!")
print(f"📁 Saved result video to: {results[0].save_dir}")
