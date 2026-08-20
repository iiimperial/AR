import pydicom
import cv2
import numpy as np
import os
import argparse
from pathlib import Path

def load_dicom_video(dicom_path):
    dcm = pydicom.dcmread(dicom_path)
    frames = dcm.pixel_array
    return frames

def mask_scan_sector(frame: np.ndarray):
    mask = np.zeros_like(frame, dtype=np.uint8)
    mask[frame > 0] = 1
    return frame * mask

def normalize_intensity(img: np.ndarray):
    maxv = np.max(img)
    if maxv < 1e-6:
        return img
    return img / maxv

def sample_uniform_frames(total_num: int, n_sample=16):
    indices = np.linspace(0, total_num - 1, n_sample, dtype=int)
    return indices

def postprocess_mask(mask: np.ndarray, expand_pixel=10):
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    max_area = 0
    max_idx = 1
    for i in range(1, num_labels):
        area = np.sum(labels == i)
        if area > max_area:
            max_area = area
            max_idx = i
    largest_mask = (labels == max_idx).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_pixel*2+1, expand_pixel*2+1))
    expand_mask = cv2.dilate(largest_mask, kernel)
    return expand_mask

def resize_frame(img:np.ndarray, size=(256,256)):
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

def process_single_dicom(dicom_path, out_npz_path, n_sample_frames=16):
    frames = load_dicom_video(dicom_path)
    n_total = frames.shape[0]
    sel_idx = sample_uniform_frames(n_total, n_sample=n_sample_frames)
    out_list = []
    for idx in sel_idx:
        f = frames[idx]
        f = mask_scan_sector(f)
        f = normalize_intensity(f)
        f = resize_frame(f, (256,256))
        out_list.append(f)
    arr = np.stack(out_list, axis=0)
    np.savez_compressed(out_npz_path, video=arr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_sample_frames", default=16, type=int)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    p = Path(args.dicom_dir)
    for fpath in list(p.glob("*.dcm")):
        out_file = os.path.join(args.out_dir, fpath.stem + ".npz")
        process_single_dicom(str(fpath), out_file, n_sample_frames=args.n_sample_frames)

if __name__ == "__main__":
    main()