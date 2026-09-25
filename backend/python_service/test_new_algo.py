import os
import sys
import numpy as np
import cv2
from scipy.ndimage import uniform_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sar_loader import load_sar_image
from validator import evaluate_segmentation
from metrics import calculate_geometric_properties
from samples_generator import SAMPLES_DIR

def enhanced_lee_filter(image: np.ndarray, window_size: int = 7, num_looks: float = 4.0) -> np.ndarray:
    img_f = image.astype(np.float32)
    mean = uniform_filter(img_f, size=window_size)
    sq_mean = uniform_filter(img_f**2, size=window_size)
    variance = np.maximum(sq_mean - mean**2, 1e-6)
    
    cu = 1.0 / np.sqrt(num_looks)
    cmax = np.sqrt(1.0 + 2.0 / num_looks)
    ci = np.sqrt(variance) / np.maximum(mean, 1e-6)
    
    w = np.zeros_like(img_f)
    mask_homo = ci <= cu
    mask_edge = ci >= cmax
    mask_inter = (~mask_homo) & (~mask_edge)
    
    w[mask_homo] = 1.0
    w[mask_edge] = 0.0
    
    denom = np.maximum(cmax - ci[mask_inter], 1e-6)
    w[mask_inter] = np.exp(-1.0 * (ci[mask_inter] - cu) / denom)
    
    filtered = mean * w + img_f * (1.0 - w)
    return np.clip(filtered, 0, 255).astype(np.uint8)

def segment_sar_oil(img: np.ndarray, geo_meta: dict, gt_mask: np.ndarray = None, s_id: str = ""):
    lee = enhanced_lee_filter(img, window_size=7, num_looks=4.0)
    h, w = img.shape
    
    # 1. Robust Sea Clutter Statistics
    # Exclude extreme vessel returns (> 98th percentile) and extreme dark (< 2nd percentile)
    p2, p50, p95 = np.percentile(lee, [2, 50, 95])
    sea_pixels = lee[(lee >= p2) & (lee <= p95)]
    sea_mean = np.mean(sea_pixels)
    sea_std = np.std(sea_pixels)
    
    # 2. Multi-scale background estimation
    bg_blur = cv2.GaussianBlur(lee, (51, 51), 0).astype(np.float32)
    
    # 3. Adaptive Thresholding:
    # Oil dampens capillary waves resulting in backscatter 1.5 to 2.2 std below local/global sea mean
    # In addition, calculate Otsu on dark subset (below median)
    dark_subset = lee[lee < sea_mean]
    if len(dark_subset) > 50:
        inv_dark = (255 - dark_subset).astype(np.uint8)
        otsu_val, _ = cv2.threshold(inv_dark, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_dark_thresh = 255 - otsu_val
    else:
        otsu_dark_thresh = sea_mean - 1.8 * sea_std
        
    # Combined adaptive threshold
    # A pixel must be:
    # (a) at least 1.6 std below the local background AND
    # (b) below the dark Otsu threshold / global dark ceiling
    thresh_local = bg_blur - 1.6 * sea_std
    thresh_global = min(otsu_dark_thresh + 6, sea_mean - 1.4 * sea_std)
    
    # Threshold condition
    oil_binary = ((lee.astype(np.float32) < thresh_local) & (lee.astype(np.float32) <= thresh_global)).astype(np.uint8) * 255
    
    # 4. Morphological Cleanup
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(oil_binary, cv2.MORPH_OPEN, k_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)
    
    # 5. Connected Components & Look-alike Discrimination
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(closed)
    slicks = []
    
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    
    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px < 100:  # Ignore tiny noise artifacts
            continue
            
        geom = calculate_geometric_properties(cnt, geo_meta)
        
        # Ring Contrast Measurement
        mask_in = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_in, [cnt], -1, 255, -1)
        dilated = cv2.dilate(mask_in, ring_kernel)
        mask_ring = cv2.bitwise_and(dilated, cv2.bitwise_not(mask_in))
        
        mean_in = float(cv2.mean(img, mask=mask_in)[0])
        mean_out = float(cv2.mean(img, mask=mask_ring)[0])
        contrast_drop = (mean_out - mean_in) / max(mean_out, 1.0)
        
        # Look-alike discrimination rule:
        # Real slicks have:
        # - High contrast drop (>= 0.32) OR
        # - High elongation (>= 1.8) with moderate contrast drop (>= 0.25)
        # Calm water lookalikes have low contrast drop (<0.25) and low elongation (<1.6)
        is_lookalike = (contrast_drop < 0.25) or (geom["elongation"] < 1.6 and contrast_drop < 0.35)
        
        if not is_lookalike:
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)
            slicks.append({
                "area_km2": geom["area_km2"],
                "elongation": geom["elongation"],
                "contrast_drop": contrast_drop,
                "confidence": round(min(99.0, max(50.0, 0.6 * contrast_drop * 100 + 0.4 * geom["elongation"] * 25)), 1)
            })
            
    print(f"\n[{s_id}] Detected Slicks: {len(slicks)}, Total Pixels: {np.sum(final_mask > 0)}")
    for s in slicks:
        print(f"   -> Area: {s['area_km2']:.3f} km², Elong: {s['elongation']:.2f}, CDrop: {s['contrast_drop']:.2f}, Conf: {s['confidence']}%")
        
    if gt_mask is not None and np.sum(gt_mask > 0) > 0:
        val = evaluate_segmentation(final_mask, gt_mask)
        print(f"   VALIDATION => IoU: {val['iou_percentage']}, Dice: {val['dice_percentage']}, Precision: {val['precision_percentage']}, Recall: {val['recall_percentage']}")
    elif gt_mask is not None:
        fp_pixels = np.sum(final_mask > 0)
        print(f"   FALSE ALARM TEST => FP Pixels: {fp_pixels} ({'PASS (100% Rejection)' if fp_pixels == 0 else 'FAIL'})")

for s_id in ["sample_1_tanker_slick", "sample_2_bilge_dump", "sample_3_offshore_platform", "sample_4_lookalike_calm_water"]:
    tif_path = os.path.join(SAMPLES_DIR, f"{s_id}.tif")
    gt_path = os.path.join(SAMPLES_DIR, f"{s_id}_gt.png")
    img, geo_meta = load_sar_image(file_path=tif_path)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(gt_path) else None
    segment_sar_oil(img, geo_meta, gt_mask, s_id)


