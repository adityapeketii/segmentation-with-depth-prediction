"""
check_classes.py  —  Scan all label masks and report unique class indices.
Run from inside src/:  python check_classes.py
"""

import os
import numpy as np
from PIL import Image

LABEL_DIRS = [
    "../Dataset/train/labels",
    "../Dataset/test/labels",
]

all_classes = set()

for label_dir in LABEL_DIRS:
    if not os.path.exists(label_dir):
        print(f"[SKIP] {label_dir} not found")
        continue

    files = sorted(os.listdir(label_dir))
    print(f"\nScanning {label_dir}  ({len(files)} files)...")

    for fname in files:
        path = os.path.join(label_dir, fname)
        img  = np.array(Image.open(path))

        # Class index is in R channel
        if img.ndim == 3:
            img = img[:, :, 0]

        all_classes.update(np.unique(img).tolist())

print("\n" + "="*40)
print(f"Unique class indices found : {sorted(all_classes)}")
print(f"Number of classes          : {len(all_classes)}")
print("="*40)