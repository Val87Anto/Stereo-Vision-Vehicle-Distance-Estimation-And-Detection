from ultralytics import YOLO
import cv2
import os
from preprocess import enhance_image

def main():
    # Define paths
    DATA_YAML = r"C:\Users\ASUS\Downloads\Semester 5_Automotive\Computer Vision\FINAL\VeryVeryNewVehicleDetection\car and truck detection.v3i.yolov8\data.yaml"
    PREPROC_DIR = r"C:\Users\ASUS\Downloads\Semester 5_Automotive\Computer Vision\FINAL\VeryVeryNewVehicleDetection\Processed"

    # 1️⃣ Create enhanced dataset
    os.makedirs(PREPROC_DIR, exist_ok=True)
    for subfolder in ["train", "valid", "test"]:
        src = os.path.join(os.path.dirname(DATA_YAML), subfolder, "images")
        dst = os.path.join(PREPROC_DIR, subfolder, "images")
        os.makedirs(dst, exist_ok=True)

        for img_name in os.listdir(src):
            if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                img = cv2.imread(os.path.join(src, img_name))
                if img is None:
                    continue
                enh = enhance_image(img)
                cv2.imwrite(os.path.join(dst, img_name), enh)

    print("✅ Preprocessing complete — starting YOLOv8 training...")

    # 2️⃣ Train YOLOv8 on enhanced data
    model = YOLO("yolov8s.pt")
    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=8,
        project="VehicleDetection_Hybrid",
        name="train_manual_conv",
        workers=0,  # 👈 safer for Windows
        device=0
    )

if __name__ == "__main__":
    main()
