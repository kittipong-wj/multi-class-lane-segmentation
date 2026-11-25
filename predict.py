"""
predict.py - Inference & Mask Resizing Pipeline
-----------------------------------------------
Requirements:
1. Load trained model weights from 'model.pt'.
2. Load strictly the 35% test dataset split from 'test_split.json'.
3. Run model inference on 64x36 inputs to obtain predicted binary logits/masks.
4. CRITICAL POST-PROCESSING: Resize output 64x36 predicted mask BACK to original 1280x720 resolution.
5. Save 1280x720 binary mask files (0 or 255 PNG) into '/result' subdirectory.
"""

import os
import json
import numpy as np
import cv2
import torch

from model import CustomYOLOSeg


def run_inference(
    weights_path="model.pt",
    split_file="test_split.json",
    output_dir="result",
    threshold=0.5
):
    print("=" * 65)
    print("       CUSTOM YOLO-SEG INFERENCE - 64x36 to 1280x720 RESIZING       ")
    print("=" * 65)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at '{weights_path}'. Run train.py first!")

    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Test split metadata not found at '{split_file}'. Run train.py first!")

    # 1. Load test set file pairs
    with open(split_file, "r") as f:
        split_meta = json.load(f)
    test_pairs = split_meta.get("test_pairs", [])

    if len(test_pairs) == 0:
        raise ValueError("No test pairs found in test_split.json!")

    print(f"[Inference] Loaded {len(test_pairs)} test samples from '{split_file}'.")

    # 2. Setup Device & Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomYOLOSeg(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    print(f"[Inference] Saving 1280x720 predicted masks to directory: '{output_dir}/'")

    # 3. Process each test image
    with torch.no_grad():
        for idx, (img_path, lbl_path) in enumerate(test_pairs):
            basename = os.path.splitext(os.path.basename(img_path))[0]

            # Read original image (1280x720 expected)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"[Warning] Could not read image '{img_path}'. Skipping.")
                continue

            orig_h, orig_w = img_bgr.shape[:2]  # Should be 720, 1280

            # Preprocess image to 64x36 for model input
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_resized = cv2.resize(img_rgb, (64, 36), interpolation=cv2.INTER_LINEAR)

            # Convert to Tensor (1, 3, 36, 64)
            input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)

            # Forward pass
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # (36, 64) float [0, 1]

            # =================================================================
            # POST-PROCESSING: Smooth Upscaling (Bilinear Probability Interpolation)
            # Upsample continuous float probabilities (36x64) -> (720x1280) first,
            # then threshold to produce crisp, smooth, anti-aliased lane edges.
            # =================================================================
            probs_1280x720 = cv2.resize(
                probs,
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR
            )
            pred_mask_1280x720 = (probs_1280x720 >= threshold).astype(np.uint8) * 255

            # Save 1280x720 result mask image to /result directory
            save_mask_path = os.path.join(output_dir, f"{basename}_mask.png")
            cv2.imwrite(save_mask_path, pred_mask_1280x720)

            if (idx + 1) % 10 == 0 or (idx + 1) == len(test_pairs):
                print(f"[{idx + 1:02d}/{len(test_pairs):02d}] Resized {probs.shape} -> "
                      f"{pred_mask_1280x720.shape} -> Saved '{save_mask_path}'")

    print("=" * 65)
    print(f"[Complete] Inference complete. All 1280x720 masks written to '{output_dir}/'.")
    print("=" * 65)


if __name__ == "__main__":
    run_inference(weights_path="model.pt", split_file="test_split.json", output_dir="result")
