import os
import torch
import matplotlib.pyplot as plt

def save_predictions(images, gt_masks, pred_masks, gt_depth, pred_depth, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    images     = images.cpu()
    gt_masks   = gt_masks.cpu()
    pred_masks = pred_masks.cpu()
    gt_depth   = gt_depth.cpu()
    pred_depth = pred_depth.cpu()

    num_samples = min(10, images.size(0))
    indices = torch.randperm(images.size(0))[:num_samples]
 
    for idx, i in enumerate(indices):
        img  = images[i].permute(1, 2, 0).numpy()
        gt_m = gt_masks[i].numpy()
        pr_m = pred_masks[i].numpy()
        gt_d = gt_depth[i][0].numpy()
        pr_d = pred_depth[i][0].numpy()
 
        fig, axs = plt.subplots(1, 5, figsize=(15, 3))
        axs[0].imshow(img);                    axs[0].set_title("Image")
        axs[1].imshow(gt_m);                   axs[1].set_title("GT Mask")
        axs[2].imshow(pr_m);                   axs[2].set_title("Pred Mask")
        axs[3].imshow(gt_d, cmap="inferno");   axs[3].set_title("GT Depth")
        axs[4].imshow(pr_d, cmap="inferno");   axs[4].set_title("Pred Depth")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sample_{i}.png"))
        plt.close()