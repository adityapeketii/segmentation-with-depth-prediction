# Q2 Report: Multi-Task Learning (Segmentation + Depth)

## Required WandB Links
- Vanilla Multi-Task U-Net: https://api.wandb.ai/links/adityapeketii-iiit-hyderabad/qtxe4x2p
- U-Net without Skip Connections: https://api.wandb.ai/links/adityapeketii-iiit-hyderabad/au3a1h7m
- U-Net with Residual Blocks: https://api.wandb.ai/links/adityapeketii-iiit-hyderabad/67bhjdun

## Folder Map
- vanialla_unet/
- without_skip/
- residual/
- comparisons/

## What Each Image Represents

For each model folder (vanialla_unet, without_skip, residual):

1. losses.png
- Train/validation loss curves (combined and/or task-wise).

2. mIOU_plot.png
- Segmentation quality trend over epochs.

3. RMSE_plot.png
- Depth error trend over epochs (lower is better).

4. qualitative_results.png
- Side-by-side examples: image, GT mask, predicted mask, GT depth, predicted depth.
- 10 examples should be shown.

There can be more images if you want to show additional insights (e.g. boundary quality, failure cases) or multiple images for a visualization (e.g. individual losses along with total loss).

## 2.1 Vanilla Multi-Task U-Net
- Combined loss used (formula and coefficients): L_total = L_seg + 0.2 · L_depth, where `L_seg` is cross-entropy loss and `L_depth` is MSE loss.
- Why this combination: Cross-entropy is standard for per-pixel classification; MSE penalizes large depth errors directly. The 1/0.2 split was found to give stable convergence without either task dominating and 0.2 was give to rmse because of its nature to be high when doing per pixel.

- Final test metrics:
    - mIoU: 0.76552
    - RMSE: 0.0507
    - Final test loss: 0.068421

## 2.2 U-Net without Skip Connections

- Final test metrics:
    - mIoU: 0.58627
    - RMSE: 0.05118
    - Final test loss: 0.1821
- Comparison with vanilla (brief):
    - Segmentation boundaries: Noticeably blurrier. mIoU drops by ~0.18, indicating the decoder struggles to recover fine spatial detail from the bottleneck alone. Object boundaries and small regions are poorly delineated.
    - Depth quality: Marginally worse RMSE. Global depth structure is roughly preserved but edge sharpness in the depth map slightly degrades.

## 2.3 U-Net with Residual Connections
- Residual block summary: ach double-conv block is replaced with two 3×3 conv layers with BatchNorm and ReLU, plus a skip connection adding the block input to its output which helps in better gradient flows and better learning indicating the highest mIoU
- Final test metrics:
    - mIoU: 0.79335
    - RMSE: 0.0560
    - Final test loss: 0.0506
- Comparison with vanilla (brief):
    - Segmentation boundaries: Slightly sharper and more accurate. mIoU improves by little, suggesting residual blocks learn richer features that better distinguish class boundaries.
    - Depth quality: Segmentation benefits more from residual learning than depth does; the depth head may require further tuning of the loss weight to fully benefit.

## Comparison Table

| Model Variant      | mIoU    | RMSE   | Final Test Loss |
|--------------------|---------|--------|-----------------|
| Vanilla UNet       | 0.76552 | 0.0507 | 0.068421        |
| Without Skip UNet  | 0.58627 | 0.0512 | 0.1821          |
| With Residual UNet | 0.79335 | 0.0560 | 0.0506          |

## Comparison Images (in comparisons/)
- comparison_plots_if_any.png
    - Put cross-model metric comparison (bar/line plot).
- comparison_results_if_any.png
    - Put cross-model qualitative visual comparison.
