"""
train.py - Training Pipeline for Custom YOLO-seg Lane Segmentation
------------------------------------------------------------------
Features:
1. Fallback Dummy Dataset Generator (1280x720 RGB & YOLO-seg polygon txt).
2. Explicit 65% Train / 35% Test split with saved split metadata (test_split.json).
3. Polygon parsing (Ultralytics YOLO-seg) to 1280x720 binary mask -> resized to 64x36.
4. Torchvision / OpenCV Augmentations (White balance, Brightness, Blur).
5. Combined Loss = BCE + Dice Loss.
6. LR Scheduler (CosineAnnealingLR / StepLR).
7. Tensorboard Logging & Model Weight Saving (model.pt).
"""

import os
import glob
import json
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import CustomYOLOSeg


# =====================================================================
# 0. MOCK DATASET GENERATOR (Fallback if no real data directory)
# =====================================================================
def generate_dummy_dataset(dataset_dir="Dataset", num_samples=30):
    """
    Generates dummy 1280x720 images and YOLO-seg polygon annotations (.txt)
    for end-to-end testing when no dataset is present.
    """
    img_dir = os.path.join(dataset_dir, "images")
    lbl_dir = os.path.join(dataset_dir, "labels")

    if os.path.exists(img_dir) and len(os.listdir(img_dir)) > 0:
        print(f"[Dataset] Using existing dataset found in '{img_dir}'.")
        return

    print(f"[Dataset] Data directory empty or missing. Generating {num_samples} dummy samples...")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for i in range(num_samples):
        # Create synthetic 1280x720 image (asphalt road with noise)
        img = np.full((720, 1280, 3), fill_value=60, dtype=np.uint8)
        # Add random texture noise
        noise = np.random.randint(-15, 15, (720, 1280, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Draw a synthetic white/yellow lane line polygon
        # Points defining a trapezoid/quadrilateral lane marking
        p1 = (550 + random.randint(-20, 20), 450 + random.randint(-10, 10))
        p2 = (600 + random.randint(-20, 20), 450 + random.randint(-10, 10))
        p3 = (800 + random.randint(-30, 30), 720)
        p4 = (700 + random.randint(-30, 30), 720)
        pts = np.array([p1, p2, p3, p4], dtype=np.int32)

        cv2.fillPoly(img, [pts], color=(240, 240, 240))  # Synthetic lane line

        img_name = f"lane_{i:04d}.jpg"
        lbl_name = f"lane_{i:04d}.txt"

        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, lbl_name)

        cv2.imwrite(img_path, img)

        # Write normalized YOLO-seg polygon annotation (class 3: Lane)
        norm_pts = []
        for x, y in [p1, p2, p3, p4]:
            norm_pts.extend([x / 1280.0, y / 720.0])

        polygon_str = "3 " + " ".join([f"{coord:.6f}" for coord in norm_pts])
        with open(lbl_path, "w") as f:
            f.write(polygon_str + "\n")

    print(f"[Dataset] Successfully generated {num_samples} dummy samples in '{dataset_dir}'.")


# =====================================================================
# 1. HELPER: PARSE YOLO POLYGON TO 1280x720 BINARY MASK
# =====================================================================
def parse_yolo_polygon(label_path, img_w=1280, img_h=720, target_class=3):
    """
    Parses Ultralytics YOLO-seg annotation text file and converts
    normalized polygon coordinates for target_class (default 3: Lane Area) to a 1280x720 binary mask (0 or 1).
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not os.path.exists(label_path):
        return mask

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:  # Class + at least 3 pairs of (x, y)
            continue

        class_id = int(parts[0])
        if target_class is not None:
            if isinstance(target_class, (list, tuple, set)):
                if class_id not in target_class:
                    continue
            elif class_id != target_class:
                continue

        coords = [float(p) for p in parts[1:]]

        polygon_pts = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * img_w)
            y = int(coords[i + 1] * img_h)
            polygon_pts.append([x, y])

        if len(polygon_pts) >= 3:
            pts_arr = np.array([polygon_pts], dtype=np.int32)
            cv2.fillPoly(mask, pts_arr, color=1)

    return mask


# =====================================================================
# 2. AUGMENTATIONS & DATASET CLASS
# =====================================================================
class LaneAugmentations:
    """Torchvision/OpenCV Data Augmentation for environmental noise."""
    def __init__(self, white_balance_range=(0.85, 1.15), brightness_range=(0.8, 1.2), blur_prob=0.5):
        self.wb_range = white_balance_range
        self.bright_range = brightness_range
        self.blur_prob = blur_prob

    def __call__(self, image_np):
        """
        image_np: (H, W, C) RGB float32 in [0, 1]
        """
        augmented = image_np.copy()

        # 1. White Balance Shift (scale channels individually)
        wb_factors = np.random.uniform(self.wb_range[0], self.wb_range[1], size=(1, 1, 3))
        augmented = augmented * wb_factors

        # 2. Brightness Shift (sunlight simulation)
        bright_factor = np.random.uniform(self.bright_range[0], self.bright_range[1])
        augmented = augmented * bright_factor

        # Clip values to [0, 1]
        augmented = np.clip(augmented, 0.0, 1.0)

        # 3. Blur shift (environmental noise)
        if random.random() < self.blur_prob:
            ksize = random.choice([3, 5])
            augmented = cv2.GaussianBlur(augmented, (ksize, ksize), 0)

        return augmented


class LaneSegmentationDataset(Dataset):
    """Dataset for loading 1280x720 images & YOLO-seg annotations, resizing to 64x36."""
    def __init__(self, image_paths, label_paths, target_size=(64, 36), is_train=True):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.target_w, self.target_h = target_size
        self.is_train = is_train
        self.augmentor = LaneAugmentations() if is_train else None

        self.cached_images = []
        self.cached_masks = []

        print(f"[Dataset] Pre-loading and caching {len(image_paths)} samples to RAM...")
        for img_path, lbl_path in zip(image_paths, label_paths):
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise FileNotFoundError(f"Image not found at {img_path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            mask = parse_yolo_polygon(lbl_path, img_w=1280, img_h=720).astype(np.float32)

            img_resized = cv2.resize(img_rgb, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST)
            mask_resized = (mask_resized > 0.5).astype(np.float32)

            self.cached_images.append(img_resized)
            self.cached_masks.append(mask_resized)
        print("[Dataset] RAM caching complete.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_rgb = self.cached_images[idx].copy()
        mask_resized = self.cached_masks[idx]

        # Data Augmentations (on image only)
        if self.is_train and self.augmentor:
            img_rgb = self.augmentor(img_rgb)

        # Convert to Tensors: Image (3, 36, 64), Mask (1, 36, 64)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float()

        return img_tensor, mask_tensor, self.image_paths[idx]



# =====================================================================
# 3. COMBINED LOSS FUNCTION (BCE + Dice)
# =====================================================================
class CombinedBCEDiceLoss(nn.Module):
    """Combines Binary Cross Entropy with Dice Loss."""
    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = dice_loss.mean()

        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total_loss, bce_loss, dice_loss


# =====================================================================
# 4. MAIN TRAINING LOOP
# =====================================================================
def train_model(
    dataset_dir="Dataset",
    epochs=30,
    batch_size=16,
    learning_rate=1e-3,
    save_path="model.pt"
):
    print("=" * 65)
    print("      CUSTOM YOLO-SEG TRAINER - SINGLE CLASS LANE SEGMENTATION     ")
    print("=" * 65)

    # 1. Fallback dummy dataset generation
    generate_dummy_dataset(dataset_dir=dataset_dir, num_samples=30)

    # 2. Collect image and label pairs
    img_dir = os.path.join(dataset_dir, "images")
    lbl_dir = os.path.join(dataset_dir, "labels")

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.[jJ][pP][gG]")) +
                       glob.glob(os.path.join(img_dir, "*.[pP][nN][gG]")))

    pairs = []
    for img_path in img_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, f"{basename}.txt")
        pairs.append((img_path, lbl_path))

    if len(pairs) == 0:
        raise RuntimeError("No image-label pairs found in dataset!")

    # 3. Explicit Data Splitting: 65% Train, 35% Test
    random.seed(42)
    random.shuffle(pairs)

    num_train = int(len(pairs) * 0.65)
    train_pairs = pairs[:num_train]
    test_pairs = pairs[num_train:]

    print(f"[Split] Total Samples: {len(pairs)} | Train (65%): {len(train_pairs)} | Test (35%): {len(test_pairs)}")

    # Save test set split list so predict.py and evaluation.py can run strictly on test set
    split_meta = {
        "train_pairs": train_pairs,
        "test_pairs": test_pairs
    }
    with open("test_split.json", "w") as f:
        json.dump(split_meta, f, indent=2)
    print("[Split] Saved test split metadata to 'test_split.json'.")

    # 4. Create DataLoaders
    train_dataset = LaneSegmentationDataset(
        image_paths=[p[0] for p in train_pairs],
        label_paths=[p[1] for p in train_pairs],
        target_size=(64, 36),
        is_train=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    val_dataset = LaneSegmentationDataset(
        image_paths=[p[0] for p in test_pairs],
        label_paths=[p[1] for p in test_pairs],
        target_size=(64, 36),
        is_train=False
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # 5. Device Setup & Model Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using compute device: {device}")

    model = CustomYOLOSeg(in_channels=3, num_classes=1).to(device)

    # 6. Loss, Optimizer, Scheduler & Tensorboard Logger
    criterion = CombinedBCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    writer = SummaryWriter(log_dir="runs/lane_yolo_seg")

    # 7. Epoch Loop
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # --- Training Phase ---
        model.train()
        running_total_loss = 0.0
        running_bce_loss = 0.0
        running_dice_loss = 0.0

        for imgs, masks, _ in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)

            total_loss, bce_loss, dice_loss = criterion(logits, masks)
            total_loss.backward()
            optimizer.step()

            running_total_loss += total_loss.item() * imgs.size(0)
            running_bce_loss += bce_loss.item() * imgs.size(0)
            running_dice_loss += dice_loss.item() * imgs.size(0)

        scheduler.step()
        curr_lr = optimizer.param_groups[0]['lr']

        epoch_train_total = running_total_loss / len(train_dataset)
        epoch_train_bce = running_bce_loss / len(train_dataset)
        epoch_train_dice = running_dice_loss / len(train_dataset)

        # --- Validation Phase ---
        model.eval()
        val_running_total_loss = 0.0
        val_running_bce_loss = 0.0
        val_running_dice_loss = 0.0

        with torch.no_grad():
            for v_imgs, v_masks, _ in val_loader:
                v_imgs = v_imgs.to(device)
                v_masks = v_masks.to(device)

                v_logits = model(v_imgs)
                v_total, v_bce, v_dice = criterion(v_logits, v_masks)

                val_running_total_loss += v_total.item() * v_imgs.size(0)
                val_running_bce_loss += v_bce.item() * v_imgs.size(0)
                val_running_dice_loss += v_dice.item() * v_imgs.size(0)

        epoch_val_total = val_running_total_loss / len(val_dataset)
        epoch_val_bce = val_running_bce_loss / len(val_dataset)
        epoch_val_dice = val_running_dice_loss / len(val_dataset)

        # Tensorboard logging
        writer.add_scalar("Loss/Train_Total", epoch_train_total, epoch)
        writer.add_scalar("Loss/Train_BCE", epoch_train_bce, epoch)
        writer.add_scalar("Loss/Train_Dice", epoch_train_dice, epoch)
        writer.add_scalar("Loss/Val_Total", epoch_val_total, epoch)
        writer.add_scalar("Loss/Val_BCE", epoch_val_bce, epoch)
        writer.add_scalar("Loss/Val_Dice", epoch_val_dice, epoch)
        writer.add_scalar("LearningRate", curr_lr, epoch)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | "
              f"Train Loss: {epoch_train_total:.4f} (Dice: {epoch_train_dice:.4f}) | "
              f"Val Loss: {epoch_val_total:.4f} (Dice: {epoch_val_dice:.4f}) | "
              f"LR: {curr_lr:.6f}", flush=True)

        # Save best weights based on validation loss
        if epoch_val_total < best_loss:
            best_loss = epoch_val_total
            torch.save(model.state_dict(), save_path)  # model.pt

    writer.close()
    print("=" * 65)
    print(f"[Complete] Model training finished!")
    print(f" - Best Weights saved to : '{save_path}'")
    print("=" * 65)


if __name__ == "__main__":
    train_model(dataset_dir="Dataset", epochs=30, batch_size=16)
