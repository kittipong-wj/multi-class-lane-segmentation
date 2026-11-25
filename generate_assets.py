"""
generate_assets.py - Generate High-Quality Visualizations, Metrics, and Snapshots for Assignment Submission
"""

import os
import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

from train import parse_yolo_polygon

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

ASSETS_DIR = r"C:\Users\ASUS\Desktop\lane-segmentation-asg10\assets"
os.makedirs(ASSETS_DIR, exist_ok=True)


# =====================================================================
# 1. EXTRACT & PLOT TENSORBOARD LOSS CONVERGENCE
# =====================================================================
def plot_loss_convergence():
    tb_files = glob.glob(r"C:\Users\ASUS\Desktop\lane-segmentation-asg10\runs\lane_yolo_seg\events.out.tfevents*")
    if not tb_files:
        print("[Error] No TensorBoard event files found!")
        return

    ea = event_accumulator.EventAccumulator(tb_files[0])
    ea.Reload()

    def get_data(tag):
        events = ea.Scalars(tag)
        return [e.step for e in events], [e.value for e in events]

    epochs, train_total = get_data("Loss/Train_Total")
    _, val_total = get_data("Loss/Val_Total")
    _, train_bce = get_data("Loss/Train_BCE")
    _, val_bce = get_data("Loss/Val_BCE")
    _, train_dice = get_data("Loss/Train_Dice")
    _, val_dice = get_data("Loss/Val_Dice")
    _, lr = get_data("LearningRate")

    # 4-Panel Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # 1. Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_total, color="#e74c3c", lw=2.5, marker="o", markersize=4, label="Train Total Loss")
    ax.plot(epochs, val_total, color="#2980b9", lw=2.5, marker="s", markersize=4, linestyle="--", label="Val Total Loss")
    ax.set_title("Total Loss (BCE + Dice) vs Epochs", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss Value", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10, loc="upper right")
    ax.annotate(f"Final Val Loss: {val_total[-1]:.4f}", xy=(epochs[-1], val_total[-1]),
                xytext=(epochs[-1] - 8, val_total[-1] + 0.1),
                arrowprops=dict(facecolor="#2980b9", shrink=0.05, width=1, headwidth=6),
                fontsize=9, fontweight="bold", color="#2980b9")

    # 2. BCE Loss
    ax = axes[0, 1]
    ax.plot(epochs, train_bce, color="#8e44ad", lw=2.2, marker="o", markersize=4, label="Train BCE Loss")
    ax.plot(epochs, val_bce, color="#16a085", lw=2.2, marker="^", markersize=4, linestyle="--", label="Val BCE Loss")
    ax.set_title("Binary Cross-Entropy (BCE) Loss vs Epochs", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("BCE Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10, loc="upper right")

    # 3. Dice Loss
    ax = axes[1, 0]
    ax.plot(epochs, train_dice, color="#d35400", lw=2.2, marker="o", markersize=4, label="Train Dice Loss")
    ax.plot(epochs, val_dice, color="#27ae60", lw=2.2, marker="d", markersize=4, linestyle="--", label="Val Dice Loss")
    ax.set_title("Dice Loss (Overlap Loss) vs Epochs", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Dice Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10, loc="upper right")

    # 4. Learning Rate Schedule
    ax = axes[1, 1]
    ax.plot(epochs, lr, color="#2c3e50", lw=2.5, marker=".", markersize=6, label="Cosine Annealing LR")
    ax.set_title("Learning Rate Schedule (Cosine Annealing)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Learning Rate", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(ASSETS_DIR, "loss_convergence.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Done] Saved loss convergence plot to: {out_path}")


# =====================================================================
# 2. CREATE SNAPSHOT COMPARISONS & INFERENCE GRID
# =====================================================================
def create_snapshot_comparisons():
    with open("test_split.json", "r") as f:
        meta = json.load(f)
    test_pairs = meta.get("test_pairs", [])

    # Pick 3 representative samples
    selected_indices = [0, 1, 2]  # frame_00498, frame_00569, frame_00897
    grid_rows = []

    for idx in selected_indices:
        img_rel, lbl_rel = test_pairs[idx]
        basename = os.path.splitext(os.path.basename(img_rel))[0]

        img_path = os.path.join(r"C:\Users\ASUS\Desktop\lane-segmentation-asg10", img_rel)
        lbl_path = os.path.join(r"C:\Users\ASUS\Desktop\lane-segmentation-asg10", lbl_rel)
        pred_mask_path = os.path.join(r"C:\Users\ASUS\Desktop\lane-segmentation-asg10\result", f"{basename}_mask.png")

        # Load RGB image
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Load GT Mask
        gt_mask = parse_yolo_polygon(lbl_path, img_w=w, img_h=h)  # 0 or 1

        # Load Predicted Mask
        pred_img = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
        pred_bin = (pred_img > 127).astype(np.uint8)

        # Create Overlay: Blend green on lane area + cyan boundary contour
        overlay = img_rgb.copy()
        green_mask = np.zeros_like(img_rgb)
        green_mask[pred_bin == 1] = [0, 230, 100]  # Vibrant green
        # Alpha blend: 65% image + 35% color
        overlay = cv2.addWeighted(overlay, 0.7, green_mask, 0.3, 0)
        # Find contours and draw crisp boundary
        contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 3)  # Bright cyan line

        # Individual 4-panel figure
        fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=300)
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"1. Input Image ({basename})", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title("2. Ground Truth Mask (Polygon)", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(pred_bin, cmap="gray")
        axes[2].set_title("3. Predicted Mask (Resized 1280x720)", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        axes[3].imshow(overlay)
        axes[3].set_title("4. Lane Segmentation Overlay", fontsize=12, fontweight="bold")
        axes[3].axis("off")

        plt.tight_layout()
        single_path = os.path.join(ASSETS_DIR, f"inference_{basename}.png")
        plt.savefig(single_path, dpi=300)
        plt.close()
        print(f"[Done] Saved individual snapshot: {single_path}")

        grid_rows.append((basename, img_rgb, gt_mask, pred_bin, overlay))

    # Composite 3-Scene Grid
    fig, axes = plt.subplots(len(grid_rows), 4, figsize=(20, 13), dpi=300)
    col_titles = ["(A) Input Road Frame", "(B) Ground Truth Polygon Mask",
                  "(C) Custom YOLO-seg Prediction", "(D) Lane Segmentation Blend"]

    for row_idx, (bname, img, gt, pred, ov) in enumerate(grid_rows):
        axes[row_idx, 0].imshow(img)
        axes[row_idx, 0].set_ylabel(f"Sample: {bname}", fontsize=11, fontweight="bold")
        axes[row_idx, 0].set_xticks([]); axes[row_idx, 0].set_yticks([])

        axes[row_idx, 1].imshow(gt, cmap="gray")
        axes[row_idx, 1].set_xticks([]); axes[row_idx, 1].set_yticks([])

        axes[row_idx, 2].imshow(pred, cmap="gray")
        axes[row_idx, 2].set_xticks([]); axes[row_idx, 2].set_yticks([])

        axes[row_idx, 3].imshow(ov)
        axes[row_idx, 3].set_xticks([]); axes[row_idx, 3].set_yticks([])

        if row_idx == 0:
            for c_idx, title in enumerate(col_titles):
                axes[row_idx, c_idx].set_title(title, fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()
    grid_path = os.path.join(ASSETS_DIR, "inference_grid.png")
    plt.savefig(grid_path, dpi=300)
    plt.close()
    print(f"[Done] Saved 3-scene inference grid to: {grid_path}")


# =====================================================================
# 3. DETAILED METRICS COMPILATION
# =====================================================================
def compile_detailed_metrics():
    with open("evaluation_report.json", "r") as f:
        rep = json.load(f)

    print("\n" + "=" * 60)
    print("           DETAILED PERFORMANCE METRICS SUMMARY           ")
    print("=" * 60)
    print(f"Total Test Samples          : {rep['total_test_samples']}")
    print(f"Positive Detections         : {rep['positive_detections']} / {rep['total_test_samples']} ({rep['detection_rate_pct']}%)")
    print(f"Detection IoU Threshold     : {rep['detection_threshold']}")
    print(f"Average IoU (Detected Only) : {rep['average_iou_detected_only']:.4f} ({rep['average_iou_detected_only']*100:.2f}%)")
    print(f"Overall Mean IoU (mIoU)     : {rep['overall_mean_iou']:.4f} ({rep['overall_mean_iou']*100:.2f}%)")
    print(f"Mean Average Precision (mAP): {rep['mAP']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    plot_loss_convergence()
    create_snapshot_comparisons()
    compile_detailed_metrics()
