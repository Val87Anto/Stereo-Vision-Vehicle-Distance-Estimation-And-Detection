# Stereo-Vision-Vehicle-Distance-Estimation-And-Detection

Computer Vision Final Project

👥 Team Members

1. Valerie Liang Alianto (Leader)

2. Jonathan Matthew Halim

3. Sachika Valenlie

4. Bryan Stevensen Kho


📖 Overview

This project implements a computer vision system for vehicle detection and distance estimation using a stereo camera setup and YOLOv8.
The system detects cars and trucks in real time and estimates their distance in meters by computing depth from stereo image disparity.

By combining deep learning-based object detection with classical stereo vision, the system provides both semantic understanding (what object) and spatial awareness (how far it is).

🎯 Project Objectives

- Detect vehicles (cars and trucks) using YOLOv8

- Estimate object distance using stereo vision

- Perform stereo camera calibration and rectification

- Compute disparity and depth maps

- Demonstrate vision-based spatial perception for autonomous systems

🧠 System Concept

- Two identical webcams simulate human binocular vision

- Stereo calibration aligns camera geometry

- Image preprocessing enhances visual features

- YOLOv8 performs vehicle detection

- Stereo matching generates disparity maps

- Depth is calculated in real-world units (meters)

🔄 System Pipeline

- Capture stereo images

- Image preprocessing

- Feature enhancement

- Vehicle detection (YOLOv8)

- Stereo matching & disparity computation

- Depth estimation (meters)

- Output for sensor fusion / perception

🧪 Image Enhancement

- Grayscale Conversion
Reduces computational cost and focuses on intensity patterns

- Histogram Equalization
Enhances contrast under uneven lighting

- Gaussian Blur
Reduces noise before edge detection

🧩 Feature Extraction

- Sobel Edge Enhancement
Highlights vehicle contours and structural edges

- Edge Normalization
Improves consistency across frames

- Histogram of Oriented Gradients (HOG)
Captures vehicle shape and texture robustly

🚗 Vehicle Detection

- Model: YOLOv8n

- Classes:

  - 0 → Car

  - 1 → Truck

- Dataset:

  - 2531 training images

  - 361 testing images

- YOLOv8n is selected for its lightweight architecture and real-time performance.

👁 Stereo Vision & Depth Estimation

- Stereo calibration and rectification ensure accurate epipolar alignment

- Disparity maps are computed from rectified image pairs

- Depth is calculated using:
  Depth = (Focal Length × Baseline) / Disparity

  
📊 Calibration & Results Summary

- Stereo RMS (baseline): 0.1035 m

- Stereo RMS (pixel): 0.9896 px

- Mean reprojection error: 0.0422 px

- Rectified images show parallel epipolar lines

- Depth estimation results are visually and numerically consistent

📸 Result Visualizations

- Screenshots of the following are provided in the results/ folder:

- YOLO vehicle detection outputs

- Rectified stereo image pairs

- Disparity maps

- Depth visualization

- 3D point cloud reconstruction

📁 Project Structure

Stereo-Vision-Vehicle-Distance-Estimation-And-Detection/

│

├── detection/        # YOLOv8 training and inference

├── calibration/      # Camera calibration & rectification

├── stereo_vision/    # Disparity & depth estimation

├── results/          # Output images and visual results

├── AV/               # Additional vision experiments

├── SFV/              # Sensor fusion vision modules

├── requirements.txt

└── README.md

⚙️ Installation

git clone https://github.com/Val87Anto/Stereo-Vision-Vehicle-Distance-Estimation-And-Detection.git

cd Stereo-Vision-Vehicle-Distance-Estimation-And-Detection

pip install -r requirements.txt



▶️ Usage

1. Perform stereo calibration using checkerboard images

2. Rectify left and right images

3. Run YOLOv8 vehicle detection

4. Compute disparity maps

5. Estimate depth and distance


📌 Conclusion

This project successfully integrates YOLOv8-based vehicle detection with stereo vision depth estimation, enabling spatially-aware perception.
The system demonstrates how vision-based methods can provide both object recognition and real-world distance measurement, which is essential for autonomous driving and sensor fusion systems.
