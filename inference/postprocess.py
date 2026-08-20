import cv2
import numpy as np

def mask_postprocess(mask_prob, threshold=0.5, expand_pixel=10):
    mask_bin = (mask_prob > threshold).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_bin)
    max_area = 0
    max_idx = 1
    for i in range(1, num_labels):
        area = np.sum(labels == i)
        if area>max_area:
            max_area = area
            max_idx = i
    largest = (labels == max_idx).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_pixel*2+1, expand_pixel*2+1))
    dilated = cv2.dilate(largest, kernel)
    return dilated

def apply_mask_to_frame(frame, mask):
    return frame * mask