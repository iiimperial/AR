import numpy as np

def uniform_sample(total_frames, n_sample=16):
    indices = np.linspace(0, total_frames-1, n_sample, dtype=int)
    return indices