"""
In-Image SAR Vessel Candidate Detection Module
SpillTrace - SIH 2026 PS 26143 (NTRO)

Physical Principle:
Ships and metallic marine structures act as dihedral/trihedral corner reflectors,
scattering microwave radar pulses directly back to the Sentinel-1 SAR antenna.
They manifest as distinct, high-backscatter point targets standing out sharply
against ambient ocean clutter.

Algorithm:
1. High-Intensity Thresholding + Adaptive CFAR (Constant False Alarm Rate).
2. Connected Components extraction (filters speckle noise <4px and landmasses >150px).
3. Contrast check over local sea clutter.
4. Returns top 10 highest-confidence vessel candidates.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
from scipy.ndimage import uniform_filter

from sar_loader import pixel_to_latlon


def detect_vessel_candidates(
    image: np.ndarray,
    geo_meta: Dict[str, Any],
    min_vessel_area_px: int = 4,
    max_vessel_area_px: int = 150,
    max_candidates: int = 10
) -> List[Dict[str, Any]]:
    """
    Detects bright point reflectors (vessel candidates) in SAR backscatter imagery.
    Works on all images regardless of whether georeferencing metadata exists.
    """
    h, w = image.shape[:2]
    img_f = image.astype(np.float32)
    
    # 1. Local sea background statistics for adaptive CFAR threshold
    bg_window = max(21, (min(h, w) // 20) | 1)
    local_mean = uniform_filter(img_f, size=bg_window)
    local_sq_mean = uniform_filter(img_f**2, size=bg_window)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 1e-6))
    
    # Target pixel must exceed local clutter by >= 2.5 standard deviations AND have high absolute backscatter
    cfar_threshold = np.minimum(248.0, local_mean + 2.5 * local_std)
    bright_mask = (img_f >= cfar_threshold) & (image >= 215)
    bright_mask = bright_mask.astype(np.uint8) * 255
    
    # Mask out outer 6px image border to avoid boundary crop artifacts
    bright_mask[:6, :] = 0
    bright_mask[-6:, :] = 0
    bright_mask[:, :6] = 0
    bright_mask[:, -6:] = 0
    
    # Small morphological closing to group multi-pixel ship reflections
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    
    # 2. Extract Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
    
    candidates: List[Dict[str, Any]] = []
    gsd_m = geo_meta.get("gsd_meters", 10.0)
    
    for idx in range(1, num_labels):
        area_px = int(stats[idx, cv2.CC_STAT_AREA])
        if area_px < min_vessel_area_px or area_px > max_vessel_area_px:
            continue
            
        cx, cy = float(centroids[idx][0]), float(centroids[idx][1])
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        
        blob_mask = (labels == idx)
        peak_intensity = int(np.max(image[blob_mask])) if np.any(blob_mask) else int(image[int(cy), int(cx)])
        mean_intensity = float(np.mean(image[blob_mask])) if np.any(blob_mask) else float(peak_intensity)
        
        # Local background around this blob
        bg_val = float(local_mean[int(np.clip(cy, 0, h-1)), int(np.clip(cx, 0, w-1))])
        contrast_margin = peak_intensity - bg_val
        if contrast_margin < 40.0:
            continue
            
        # Estimate physical length in meters (major dimension * GSD)
        major_axis_px = max(bw, bh)
        estimated_length_m = round(major_axis_px * gsd_m, 1)
        
        # Coordinates (if georeferenced)
        has_geo = geo_meta.get("has_geotiff", False) or geo_meta.get("has_location", False)
        lat, lon = pixel_to_latlon(cx, cy, geo_meta) if has_geo else (None, None)
        
        # Radar RCS confidence score
        rcs_confidence = min(99.0, max(65.0, 60.0 + (peak_intensity - 200) * 0.6 + min(area_px * 2, 20.0)))
        
        vessel_candidate = {
            "id": f"RADAR-TARGET-{len(candidates)+1:02d}",
            "pixel_position": [round(cx, 1), round(cy, 1)],
            "bounding_box_px": {"x": x, "y": y, "width": bw, "height": bh},
            "area_px": area_px,
            "estimated_length_meters": estimated_length_m,
            "peak_radar_backscatter": peak_intensity,
            "mean_backscatter": round(mean_intensity, 1),
            "contrast_over_sea_clutter": round(contrast_margin, 1),
            "rcs_confidence_score": round(rcs_confidence, 1),
            "has_geocoordinates": lat is not None and lon is not None,
            "lat": lat,
            "lon": lon,
            "description": f"High-backscatter radar point target (~{estimated_length_m}m estimated length)"
        }
        candidates.append(vessel_candidate)
        
    candidates.sort(key=lambda v: v["peak_radar_backscatter"], reverse=True)
    return candidates[:max_candidates]
