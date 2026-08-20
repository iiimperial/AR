import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        pred = torch.sigmoid(logits)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class SegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, target):
        loss_bce = self.bce(logits, target)
        loss_dice = self.dice(logits, target)
        return loss_bce + loss_dice