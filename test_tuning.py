import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sar_loader import load_sar_image
from sar_detector import enhanced_lee_filter, morphological_cleanup, filter_lookalikes
from validator import evaluate_segmentation
from samples_generator import SAMPLES_DIR

for s_name in ["sample_1_tanker_slick", "sample_2_bilge_dump", "sample_3_offshore_platform", "sample_4_lookalike_calm_water"]:
    tif_path = os.path.join(SAMPLES_DIR, f"{s_name}.tif")
    gt_path = os.path.join(SAMPLES_DIR, f"{s_name}_gt.png")
    img, geo_meta = load_sar_image(file_path=tif_path)
    lee = enhanced_lee_filter(img, window_size=7, num_looks=4.0)
    
    # Otsu directly on dark patches
    # Invert so dark patches become bright targets for standard thresholding
    inv = 255 - lee
    otsu_thresh, _ = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # The actual threshold in original intensity is: 255 - otsu_thresh
    raw_thresh = 255 - otsu_thresh
    
    # Also refine with adaptive local contrast
    binary_seg = (lee <= raw_thresh).astype(np.uint8) * 255
    clean = morphological_cleanup(binary_seg, open_kernel_size=3, close_kernel_size=5)
    final_mask, slicks = filter_lookalikes(clean, img, geo_meta, min_area_km2=0.01)
    
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(gt_path) else None
    
    print(f"\n=== {s_name} ===")
    print(f"Otsu raw threshold: {raw_thresh:.1f}, Detected slicks: {len(slicks)}")
    if slicks:
        for s in slicks:
            print(f"  -> {s['id']}: Area={s['area_km2']} km², Elongation={s['elongation']}, Age={s['estimated_age']}, Conf={s['confidence_score']}%")
    if gt_mask is not None and np.sum(gt_mask) > 0:
        metrics = evaluate_segmentation(final_mask, gt_mask)
        print(f"  Validation: IoU={metrics['iou_percentage']}, Dice={metrics['dice_percentage']}, Precision={metrics['precision_percentage']}, Recall={metrics['recall_percentage']}")
