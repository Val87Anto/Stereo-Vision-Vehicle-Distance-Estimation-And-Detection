# preprocess.py
import cv2
import numpy as np

def enhance_image(img):
    if img is None:
        raise ValueError("Input image is None.")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Histogram equalization for contrast
    eq = cv2.equalizeHist(gray)

    # Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(eq, (3, 3), 0)

    # Edge enhancement using Sobel
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobelx, sobely)

    # Normalize
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
    return edges.astype(np.uint8)
