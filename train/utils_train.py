import numpy as np
import torch

def find_optimal_youden_threshold(y_true, y_score):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_thr = thresholds[best_idx]
    return best_thr

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score <= self.best_score:
            self.counter +=1
            if self.counter >= self.patience:
                self.early_stop=True
        else:
            self.best_score = val_score
            self.counter=0
        return self.early_stop