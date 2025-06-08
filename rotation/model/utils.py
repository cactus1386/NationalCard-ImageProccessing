# utils.py
import cv2
import numpy as np

def apply_homography(img, src_pts, predicted_offset):
    dst_pts = src_pts + predicted_offset.reshape(4, 2)
    H, _ = cv2.findHomography(dst_pts, src_pts)  # برگردوندن به حالت اصلی
    warped = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
    return warped
