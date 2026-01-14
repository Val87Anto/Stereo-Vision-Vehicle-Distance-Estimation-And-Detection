# feature_extraction.py
import cv2
import numpy as np
from skimage.feature import hog

def extract_features(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to standard input size for consistent features
    img = cv2.resize(img, (128, 128))

    # Compute HOG features
    features, _ = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        block_norm='L2-Hys'
    )
    return features
