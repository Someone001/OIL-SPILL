"""
Ground Truth Benchmark & Model Validation Module
SpillTrace - SIH 2026 PS 26143 (NTRO)

Computes statistical performance metrics:
- Intersection-over-Union (IoU / Jaccard Index)
- Dice Similarity Coefficient (F1-score)
- Precision (Positive Predictive Value)
- Recall (Sensitivity)
- False Alarm Rate
"""

from typing import Dict, Any, Tuple
import numpy as np
import cv2


def evaluate_segmentation(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluates detected binary mask against ground truth binary mask.
    Both masks must be 2D arrays where > 0 represents positive oil slick class.
    """
    # Resize ground truth if sizes differ
    if predicted_mask.shape != ground_truth_mask.shape:
        ground_truth_mask = cv2.resize(
            ground_truth_mask,
            (predicted_mask.shape[1], predicted_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
    pred_bin = (predicted_mask > 0).astype(bool)
    gt_bin = (ground_truth_mask > 0).astype(bool)
    
    # Pixel counts
    tp = np.sum(pred_bin & gt_bin)  # True Positives
    fp = np.sum(pred_bin & (~gt_bin))  # False Positives
    fn = np.sum((~pred_bin) & gt_bin)  # False Negatives
    tn = np.sum((~pred_bin) & (~gt_bin))  # True Negatives
    
    intersection = float(tp)
    union = float(tp + fp + fn)
    
    # 1. IoU (Jaccard Index)
    iou = (intersection / union) if union > 0 else (1.0 if np.sum(gt_bin) == 0 else 0.0)
    
    # 2. Dice Coefficient (F1 Score)
    dice = (2.0 * intersection / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 1.0
    
    # 3. Precision & Recall
    precision = (intersection / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (intersection / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    # 4. Accuracy & Specificity
    total_pixels = float(pred_bin.size)
    accuracy = (tp + tn) / total_pixels if total_pixels > 0 else 1.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    
    return {
        "iou": round(float(iou), 4),
        "iou_percentage": f"{round(float(iou) * 100, 1)}%",
        "dice_f1": round(float(dice), 4),
        "dice_percentage": f"{round(float(dice) * 100, 1)}%",
        "precision": round(float(precision), 4),
        "precision_percentage": f"{round(float(precision) * 100, 1)}%",
        "recall": round(float(recall), 4),
        "recall_percentage": f"{round(float(recall) * 100, 1)}%",
        "accuracy": round(float(accuracy), 4),
        "specificity": round(float(specificity), 4),
        "pixel_confusion_matrix": {
            "true_positive_pixels": int(tp),
            "false_positive_pixels": int(fp),
            "false_negative_pixels": int(fn),
            "true_negative_pixels": int(tn)
        },
        "performance_rating": (
            "Excellent (Benchmark SOTA)" if iou >= 0.75 else
            "Strong Target Match" if iou >= 0.60 else
            "Moderate Alignment" if iou >= 0.40 else "Low Fit"
        )
    }
