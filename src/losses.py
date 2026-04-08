import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_depth=1.0):
        super().__init__()
        self.segmentation_loss_fn = nn.CrossEntropyLoss()
        self.depth_loss_fn = nn.MSELoss()
        self.lambda_depth = lambda_depth

    def forward(self, pred_seg, pred_depth, gt_seg, gt_depth):

        segmentation_loss = self.segmentation_loss_fn(pred_seg, gt_seg)
        depth_loss = self.depth_loss_fn(pred_depth, gt_depth)

        total_loss = segmentation_loss + self.lambda_depth * depth_loss

        return total_loss, segmentation_loss, depth_loss