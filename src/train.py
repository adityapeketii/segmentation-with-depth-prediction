import argparse
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
import os

from model import Mycool_UNet
from dataset import get_loaders
from losses import MultiTaskLoss
from evaluate import compute_miou, compute_rmse
from viz import save_predictions

EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
LAMBDA_DEPTH = 0.2
VALIDATION_SPLIT = 0.25
NUM_CLASSES = 13

MODEL_CONFIGS = {
    "vanilla":  {"use_skip": True,  "use_residual": False},
    "no_skip":  {"use_skip": False, "use_residual": False},
    "residual": {"use_skip": True, "use_residual": True},
}

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="vanilla", choices=list(MODEL_CONFIGS.keys()))
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

run = wandb.init(
    project="CV-A3-multitask-unet",
    name=args.name,
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "lambda_depth": LAMBDA_DEPTH,
        "validation_split": VALIDATION_SPLIT,
        "num_classes": NUM_CLASSES,
        **MODEL_CONFIGS[args.name],
    }
)
config = wandb.config

train_loader, test_loader = get_loaders(root_dir="../Dataset", batch_size=BATCH_SIZE)

full_train = train_loader.dataset
train_size = int((1 - VALIDATION_SPLIT) * len(full_train))
val_size = len(full_train) - train_size
train_ds, val_ds = random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

model = Mycool_UNet(
    in_ch=3,
    num_classes=NUM_CLASSES,
    skip=config.use_skip,
    residual=config.use_residual,
).to(device)

criterion = MultiTaskLoss(lambda_depth=LAMBDA_DEPTH)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def evaluate(loader):
    model.eval()
    total_loss = total_miou = total_rmse = 0.0
    with torch.no_grad():
        for images, labels, depth in loader:
            images, labels, depth = images.to(device), labels.to(device), depth.to(device)
            seg_pred, depth_pred = model(images)
            loss, _, _ = criterion(seg_pred, depth_pred, labels, depth)
            total_loss += loss.item()
            total_miou += compute_miou(seg_pred, labels, NUM_CLASSES)
            total_rmse += compute_rmse(depth_pred, depth)
    n = len(loader)
    return total_loss / n, total_miou / n, total_rmse / n


best_miou = 0.0
save_path = f"{run.name}.pth"

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    model.train()
    train_loss = train_miou = train_rmse = 0.0

    loop = tqdm(train_loader, desc="Train")
    for images, labels, depth in loop:
        images, labels, depth = images.to(device), labels.to(device), depth.to(device)

        seg_pred, depth_pred = model(images)
        loss, _, _ = criterion(seg_pred, depth_pred, labels, depth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        miou = compute_miou(seg_pred.detach(), labels, NUM_CLASSES)
        rmse = compute_rmse(depth_pred.detach(), depth)

        train_loss += loss.item()
        train_miou += miou
        train_rmse += rmse
        loop.set_postfix(loss=loss.item(), miou=f"{miou:.3f}", rmse=f"{rmse:.3f}")

    n = len(train_loader)
    train_loss /= n
    train_miou /= n
    train_rmse /= n

    val_loss, val_miou, val_rmse = evaluate(val_loader)

    print(f"  Train -> loss: {train_loss:.4f}  mIoU: {train_miou:.4f}  RMSE: {train_rmse:.4f}")
    print(f"  Val   -> loss: {val_loss:.4f}  mIoU: {val_miou:.4f}  RMSE: {val_rmse:.4f}")

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss, "train_miou": train_miou, "train_rmse": train_rmse,
        "val_loss": val_loss,     "val_miou": val_miou,     "val_rmse": val_rmse,
    })

    if val_miou > best_miou:
        best_miou = val_miou
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": dict(config),
            "epoch": epoch + 1,
            "val_miou": val_miou,
        }, save_path)
        print(f"  Saved best model -> {save_path}  (mIoU={best_miou:.4f})")

print("\nEvaluating on TEST set...")
checkpoint = torch.load(save_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()
 
test_loss = test_miou = test_rmse = 0.0
all_images = all_gt_masks = all_pred_masks = all_gt_depth = all_pred_depth = None
 
with torch.no_grad():
    for images, labels, depth in test_loader:
        images, labels, depth = images.to(device), labels.to(device), depth.to(device)
        seg_pred, depth_pred = model(images)
        loss, _, _ = criterion(seg_pred, depth_pred, labels, depth)
 
        test_loss += loss.item()
        test_miou += compute_miou(seg_pred, labels, NUM_CLASSES)
        test_rmse += compute_rmse(depth_pred, depth)
 
        if all_images is None:
            all_images     = images
            all_gt_masks   = labels
            all_pred_masks = seg_pred.argmax(dim=1)
            all_gt_depth   = depth
            all_pred_depth = depth_pred
 
n = len(test_loader)
test_loss /= n
test_miou /= n
test_rmse /= n
 
print(f"Test mIoU: {test_miou:.4f}  RMSE: {test_rmse:.4f}")
 
wandb.log({"test_loss": test_loss, "test_miou": test_miou, "test_rmse": test_rmse})
 
save_predictions(
    all_images, all_gt_masks, all_pred_masks, all_gt_depth, all_pred_depth,
    save_dir=os.path.join("outputs", args.name)
)
print(f"Predictions saved to outputs/{args.name}/")
 
wandb.finish()
print("\nTraining complete!")