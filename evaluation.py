"""
evaluation.py - Metrics Evaluation & PR Curve Visualization
------------------------------------------------------------
Requirements:
1. Load 1280x720 predicted masks from '/result' directory.
2. Load ground truth masks (1280x720) parsed on the fly from YOLO polygon annotations.
3. Compute Pixel-Wise IoU for every test sample.
4. Detection Flag: Count Positive Detections using IoU >= 0.6 threshold.
5. Compute Average IoU of detected results (IoU >= 0.6) and overall Mean IoU.
6. Calculate mAP (Mean Average Precision / AP) for the lane class.
7. Generate and save Precision-Recall (PR) Curve plot ('pr_curve.png').
"""

import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

from train import parse_yolo_polygon


def compute_mask_iou(pred_mask, gt_mask):
    """
    Computes pixel-wise Intersection over Union (IoU).
    pred_mask: Binary mask 1280x720 (0 or 1)
    gt_mask  : Binary mask 1280x720 (0 or 1)
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection) / float(union)


def calculate_precision_recall_ap(iou_scores, iou_thresh=0.6):
    """
    Calculates Precision, Recall, and Average Precision (AP / mAP)
    for lane detection at the given IoU threshold.
    """
    if len(iou_scores) == 0:
        return [1.0, 1.0], [0.0, 1.0], 0.0

    sorted_ious = sorted(iou_scores, reverse=True)
    tp_flags = [1 if iou >= iou_thresh else 0 for iou in sorted_ious]

    cum_tp = np.cumsum(tp_flags)
    cum_fp = np.cumsum([1 - f for f in tp_flags])

    recalls = cum_tp / float(len(sorted_ious))
    precisions = cum_tp / (cum_tp + cum_fp).astype(np.float64)

    recalls = np.concatenate(([0.0], recalls))
    precisions = np.concatenate(([1.0], precisions))

    if hasattr(np, "trapezoid"):
        ap = float(np.trapezoid(precisions, recalls))
    elif hasattr(np, "trapz"):
        ap = float(np.trapz(precisions, recalls))
    else:
        ap = float(np.sum((recalls[1:] - recalls[:-1]) * (precisions[1:] + precisions[:-1])) / 2.0)
    ap = max(0.0, min(1.0, abs(ap)))

    return precisions.tolist(), recalls.tolist(), ap



def evaluate_predictions(
    result_dir="result",
    split_file="test_split.json",
    iou_thresh=0.6,
    pr_curve_path="pr_curve.png"
):
    print("=" * 65)
    print("      CUSTOM YOLO-SEG EVALUATION - PIXEL IOU & MAP METRICS       ")
    print("=" * 65)

    if not os.path.exists(result_dir):
        raise FileNotFoundError(f"Result directory '{result_dir}' not found. Run predict.py first!")

    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Test split metadata '{split_file}' not found. Run train.py first!")

    with open(split_file, "r") as f:
        split_meta = json.load(f)
    test_pairs = split_meta.get("test_pairs", [])

    if len(test_pairs) == 0:
        raise ValueError("No test pairs found in test_split.json!")

    iou_list = []
    detection_flags = []  # True/False flags for IoU >= 0.6
    sample_reports = []

    for img_path, lbl_path in test_pairs:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        pred_mask_path = os.path.join(result_dir, f"{basename}_mask.png")

        if not os.path.exists(pred_mask_path):
            print(f"[Warning] Mask missing for {basename} at {pred_mask_path}")
            continue

        # Load 1280x720 predicted mask (0 or 255 -> 0 or 1)
        pred_mask_img = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
        if pred_mask_img is None:
            continue
        pred_bin = (pred_mask_img > 127).astype(np.uint8)

        # Load 1280x720 ground truth mask from YOLO polygon file
        gt_bin = parse_yolo_polygon(lbl_path, img_w=1280, img_h=720).astype(np.uint8)

        # Compute Pixel-wise IoU
        iou = compute_mask_iou(pred_bin, gt_bin)
        is_detected = (iou >= iou_thresh)

        iou_list.append(iou)
        detection_flags.append(is_detected)

        sample_reports.append({
            "sample": basename,
            "iou": round(iou, 4),
            "detected_flag": "Yes" if is_detected else "No"
        })

    if len(iou_list) == 0:
        raise RuntimeError("No predictions evaluated!")

    # Metric summary calculations
    total_samples = len(iou_list)
    positive_detections = sum(detection_flags)
    detection_rate = (positive_detections / total_samples) * 100.0

    overall_mean_iou = np.mean(iou_list)

    detected_ious = [iou for iou, flag in zip(iou_list, detection_flags) if flag]
    avg_detected_iou = np.mean(detected_ious) if len(detected_ious) > 0 else 0.0

    # Advanced Metrics: Precision, Recall, and mAP (AP for lane class)
    precisions, recalls, mAP = calculate_precision_recall_ap(iou_list)

    # Output Evaluation Summary Table
    print("\n--- TEST SAMPLE METRICS SUMMARY ---")
    for rep in sample_reports[:10]:  # Display up to first 10
        print(f"Sample: {rep['sample']} | IoU: {rep['iou']:.4f} | Detected (IoU >= {iou_thresh}): {rep['detected_flag']}")
    if len(sample_reports) > 10:
        print(f"... and {len(sample_reports) - 10} more samples.")

    print("\n" + "=" * 65)
    print("                     EVALUATION REPORT RESULTS                   ")
    print("=" * 65)
    print(f" Total Test Samples           : {total_samples}")
    print(f" Strict Detection Threshold   : IoU >= {iou_thresh}")
    print(f" Successful Detections (Yes)  : {positive_detections} / {total_samples} ({detection_rate:.2f}%)")
    print(f" Average IoU (Detected Only)  : {avg_detected_iou:.4f}")
    print(f" Overall Mean IoU (All Test)  : {overall_mean_iou:.4f}")
    print(f" Lane Class mAP (AP@IoU)      : {mAP:.4f}")
    print("=" * 65)

    # Visualization: Save Precision-Recall Curve Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color="b", lw=2, label=f"Custom YOLO-seg (mAP = {mAP:.4f})")
    plt.title("Precision-Recall Curve - Custom YOLO-seg Lane Segmentation")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(pr_curve_path, dpi=300)
    plt.close()

    print(f"[Visualization] Precision-Recall curve graph saved to '{pr_curve_path}'.")

    # Save summary report to JSON file
    summary_report = {
        "total_test_samples": total_samples,
        "detection_threshold": iou_thresh,
        "positive_detections": positive_detections,
        "detection_rate_pct": round(detection_rate, 2),
        "average_iou_detected_only": round(float(avg_detected_iou), 4),
        "overall_mean_iou": round(float(overall_mean_iou), 4),
        "mAP": round(float(mAP), 4),
        "samples": sample_reports
    }
    with open("evaluation_report.json", "w") as f:
        json.dump(summary_report, f, indent=2)

    print("[Report] Saved full evaluation details to 'evaluation_report.json'.")


if __name__ == "__main__":
    evaluate_predictions(result_dir="result", split_file="test_split.json", iou_thresh=0.6)
