import pydicom
import numpy as np

def read_dicom_video(path):
    dcm = pydicom.dcmread(path)
    arr = dcm.pixel_array
    return arr

def save_npz(save_path,**kwargs):
    np.savez_compressed(save_path,**kwargs)

