import torch
import numpy as np

# mIoU
def compute_miou(pred, target, num_classes):
    """
    pred: [B, C, H, W]
    target: [B, H, W]
    """
    pred = torch.argmax(pred, dim=1)

    ious = []

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            continue

        ious.append(intersection / union)

    return np.mean(ious) if len(ious) > 0 else 0.0


# RMSE 
def compute_rmse(pred, target):
    """
    pred, target: [B,1,H,W]
    """
    return torch.sqrt(((pred - target) ** 2).mean()).item()