import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sar_loader import load_sar_image
from sar_detector import enhanced_lee_filter, segment_dark_patches, morphological_cleanup, filter_lookalikes, detect_oil_spills
from samples_generator import SAMPLES_DIR

for s_id in ["sample_1_tanker_slick", "sample_2_bilge_dump", "sample_3_offshore_platform", "sample_4_lookalike_calm_water"]:
    print(f"\n==================== {s_id} ====================")
    tif_path = os.path.join(SAMPLES_DIR, f"{s_id}.tif")
    gt_path = os.path.join(SAMPLES_DIR, f"{s_id}_gt.png")
    
    img, geo_meta = load_sar_image(file_path=tif_path)
    print(f"Image shape: {img.shape}, min: {img.min()}, max: {img.max()}, mean: {img.mean():.2f}, std: {img.std():.2f}")
    
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(gt_path) else None
    if gt_mask is not None:
        print(f"GT positive pixels: {np.sum(gt_mask > 0)} ({np.sum(gt_mask > 0)/(512*512)*100:.2f}%)")
        if np.sum(gt_mask > 0) > 0:
            print(f"GT mean intensity in raw img: {cv2.mean(img, mask=(gt_mask>0).astype(np.uint8))[0]:.2f}")
            print(f"GT mean intensity outside in raw img: {cv2.mean(img, mask=(gt_mask==0).astype(np.uint8))[0]:.2f}")
    
    lee = enhanced_lee_filter(img)
    blur = cv2.GaussianBlur(lee, (31, 31), 0)
    diff = blur.astype(np.float32) - lee.astype(np.float32)
    inv_img = 255 - lee
    otsu_val, _ = cv2.threshold(inv_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    global_thresh = min(255 - otsu_val, 80)
    print(f"Diff min: {diff.min():.2f}, max: {diff.max():.2f}, mean: {diff.mean():.2f}")
    print(f"Otsu val (inv): {otsu_val:.1f} -> raw otsu: {255 - otsu_val:.1f} -> global_thresh: {global_thresh}")
    
    seg = segment_dark_patches(lee, k_factor=1.2, use_otsu=True)
    print(f"Segmented pixels: {np.sum(seg > 0)}")
    
    clean = morphological_cleanup(seg)
    print(f"Cleaned pixels: {np.sum(clean > 0)}")
    
    final, slicks = filter_lookalikes(clean, img, geo_meta, min_area_km2=0.02, min_elongation=1.35)
    print(f"Final detected slicks: {len(slicks)}")
    for i, s in enumerate(slicks):
        print(f"  Slick {i+1}: ID={s['id']}, Area={s['area_km2']:.3f} km², Elong={s['elongation']:.2f}, CDrop={s['contrast_drop_ratio']:.2f}, In={s['mean_inside_intensity']}, Out={s['mean_outside_intensity']}, Conf={s['confidence_score']}%")
    
    # Also test detect_oil_spills
    res = detect_oil_spills(img, geo_meta)
    print(f"detect_oil_spills result slicks count: {res['slicks_detected_count']}, total area: {res['total_spill_area_km2']}")
    print(f"vessels detected: {len(res['in_image_vessels'])}")

