import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sar_loader import load_sar_image
from sar_detector import enhanced_lee_filter, morphological_cleanup
from metrics import calculate_geometric_properties
from samples_generator import SAMPLES_DIR

def analyze_contour_contrast(cnt, img, h, w):
    # Mask inside
    mask_in = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask_in, [cnt], -1, 255, -1)
    
    # Mask outside (dilated ring around contour)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(mask_in, k)
    mask_ring = cv2.bitwise_and(dilated, cv2.bitwise_not(mask_in))
    
    mean_in = float(cv2.mean(img, mask=mask_in)[0])
    mean_out = float(cv2.mean(img, mask=mask_ring)[0])
    
    contrast_drop = (mean_out - mean_in) / max(mean_out, 1.0)
    return mean_in, mean_out, contrast_drop

for s_name in ["sample_1_tanker_slick", "sample_2_bilge_dump", "sample_3_offshore_platform", "sample_4_lookalike_calm_water"]:
    tif_path = os.path.join(SAMPLES_DIR, f"{s_name}.tif")
    img, geo_meta = load_sar_image(file_path=tif_path)
    lee = enhanced_lee_filter(img)
    inv = 255 - lee
    otsu_thresh, _ = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    raw_thresh = 255 - otsu_thresh
    binary_seg = (lee <= raw_thresh).astype(np.uint8) * 255
    clean = morphological_cleanup(binary_seg)
    
    h, w = img.shape
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"\n--- {s_name} ---")
    for cnt in contours:
        if cv2.contourArea(cnt) < 50:
            continue
        geom = calculate_geometric_properties(cnt, geo_meta)
        m_in, m_out, c_drop = analyze_contour_contrast(cnt, img, h, w)
        
        # Look-alike decision rule:
        # 1. Oil slicks have elongation >= 1.5 OR strong damping contrast >= 0.55
        # 2. Lookalikes (calm water) have low elongation (< 1.35) AND moderate damping (< 0.50)
        is_oil = (geom["elongation"] >= 1.5) or (c_drop >= 0.55)
        print(f"Area: {geom['area_km2']:.2f} km² | Elongation: {geom['elongation']:.2f} | Contrast Drop: {c_drop:.2f} | Inside: {m_in:.1f} | Outside: {m_out:.1f} => IS_OIL: {is_oil}")
