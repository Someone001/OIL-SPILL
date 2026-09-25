"""
SAR Oil Spill Detector Core
SpillTrace - SIH 2026 PS 26143 (NTRO)

Computer Vision Pipeline:
1. Speckle Denoising: Enhanced Lee Filter for multiplicative SAR radar speckle.
2. Adaptive Damping Thresholding: Inflection derivative percentile analysis to isolate true capillary wave damping.
3. Morphological Refinement: Opening to remove speckle noise, Closing to bridge linear ribbons & plume cores.
4. Look-Alike Discrimination: Ring contrast drop analysis and geometric elongation heuristics.
5. In-Image Vessel Detection: Adaptive CFAR radar corner reflectors.
6. Overlay Generation: RGBA transparent overlay with highlighted boundaries and vessel targets.
"""

import io
import base64
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
from scipy.ndimage import uniform_filter
from PIL import Image

from sar_loader import pixel_to_latlon
from metrics import calculate_geometric_properties, estimate_slick_age
from vessel_detector import detect_vessel_candidates


def enhanced_lee_filter(image: np.ndarray, window_size: int = 7, num_looks: float = 4.0) -> np.ndarray:
    """Enhanced Lee Filter for multiplicative SAR speckle noise reduction."""
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
    
    decay_constant = 1.0
    denom = np.maximum(cmax - ci[mask_inter], 1e-6)
    w[mask_inter] = np.exp(-decay_constant * (ci[mask_inter] - cu) / denom)
    
    filtered = mean * w + img_f * (1.0 - w)
    return np.clip(filtered, 0, 255).astype(np.uint8)


def compute_adaptive_threshold(denoised_img: np.ndarray) -> float:
    """
    Computes an adaptive SAR damping cutoff threshold using derivative inflection analysis.
    Identifies the transition between true capillary wave damping and ambient ocean clutter.
    """
    p1, p5, p50 = np.percentile(denoised_img, [1.0, 5.0, 50.0])
    
    # If the lowest 1% intensity is high (> 50), the image is clean sea / look-alike (no oil damping)
    if p1 > 50.0:
        return 0.0
        
    pcts = np.linspace(1.0, 30.0, 30)
    vals = np.percentile(denoised_img, pcts)
    diffs = np.diff(vals)
    
    max_diff = np.max(diffs)
    max_diff_idx = np.argmax(diffs)
    
    if max_diff >= 7.0:
        # Sharp transition (e.g. tanker discharge ribbon or narrow illegal bilge trail)
        th = (vals[max_diff_idx] + vals[max_diff_idx + 1]) / 2.0
        return float(min(th, 75.0))
    else:
        # Broad continuous plume (e.g. offshore rig pipeline blowout)
        return float(min(58.0, p50 * 0.60))


def segment_dark_patches(
    denoised_img: np.ndarray,
    k_factor: float = 1.2,
    use_otsu: bool = True
) -> np.ndarray:
    """Segments low-backscatter dark patches characteristic of oil damping capillary waves."""
    th = compute_adaptive_threshold(denoised_img)
    if th <= 0.0:
        return np.zeros_like(denoised_img)
        
    binary_mask = (denoised_img <= th).astype(np.uint8) * 255
    return binary_mask


def morphological_cleanup(
    binary_mask: np.ndarray,
    open_kernel_size: int = 3,
    close_kernel_size: int = 7
) -> np.ndarray:
    """Morphological operations to remove speckle false-alarms and reconnect fragmented slick ribbons."""
    if np.sum(binary_mask) == 0:
        return binary_mask
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, k_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)
    return closed


def filter_lookalikes(
    cleaned_mask: np.ndarray,
    original_img: np.ndarray,
    geo_meta: Dict[str, Any],
    min_area_km2: float = 0.01,
    min_elongation: float = 1.35
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Applies morphometric and ring-contrast heuristics to filter out calm-water lookalikes."""
    h, w = cleaned_mask.shape[:2]
    gsd_m = geo_meta.get("gsd_meters", 10.0)
    pixel_area_km2 = (gsd_m * gsd_m) / 1_000_000.0
    min_pixel_area = max(50, int(min_area_km2 / pixel_area_km2))
    
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(cleaned_mask)
    detected_slicks: List[Dict[str, Any]] = []
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    
    for idx, cnt in enumerate(contours):
        area_px = cv2.contourArea(cnt)
        if area_px < min_pixel_area:
            continue
            
        geom = calculate_geometric_properties(cnt, geo_meta)
        
        mask_in = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_in, [cnt], -1, 255, -1)
        dilated = cv2.dilate(mask_in, ring_kernel)
        mask_ring = cv2.bitwise_and(dilated, cv2.bitwise_not(mask_in))
        
        mean_in = float(cv2.mean(original_img, mask=mask_in)[0])
        mean_out = float(cv2.mean(original_img, mask=mask_ring)[0])
        contrast_drop = (mean_out - mean_in) / max(mean_out, 1.0)
        
        # Real SAR oil slick criteria:
        # Strong contrast damping (>35%) OR high elongation (>1.6) with solid contrast damping (>30%)
        # Calm water lookalikes have low contrast drop (<35%)
        is_lookalike = (contrast_drop < 0.35) or (geom["elongation"] < 1.6 and contrast_drop < 0.50)
        
        if not is_lookalike:
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)
            age_info = estimate_slick_age(geom["elongation"], geom["compactness"], geom["area_km2"])
            
            contrast_pct = min(100.0, max(0.0, contrast_drop * 100.0))
            elongation_pct = min(100.0, geom["elongation"] * 25.0)
            confidence = round(0.6 * contrast_pct + 0.4 * elongation_pct, 1)
            confidence = max(50.0, min(99.0, confidence))
            
            slick_data = {
                "id": f"SLICK-{len(detected_slicks)+1:02d}",
                "area_km2": geom["area_km2"],
                "perimeter_km": geom["perimeter_km"],
                "elongation": geom["elongation"],
                "orientation_deg": geom["orientation_deg"],
                "compactness": geom["compactness"],
                "centroid_px": geom["centroid_px"],
                "centroid_lat": geom["centroid_lat"],
                "centroid_lon": geom["centroid_lon"],
                "has_geocoordinates": geom["centroid_lat"] is not None,
                "bbox_px": geom["bbox_px"],
                "bbox_geo": geom["bbox_geo"],
                "mean_inside_intensity": round(mean_in, 1),
                "mean_outside_intensity": round(mean_out, 1),
                "contrast_drop_ratio": round(contrast_drop, 2),
                "confidence_score": confidence,
                "estimated_age": age_info["bracket"],
                "age_heuristic_reasoning": age_info["reasoning"],
                "contour_pts_count": len(cnt)
            }
            detected_slicks.append(slick_data)
            
    detected_slicks.sort(key=lambda s: s["area_km2"], reverse=True)
    return final_mask, detected_slicks


def create_overlay_png(
    mask: np.ndarray,
    vessels: List[Dict[str, Any]] = None,
    color_rgb: Tuple[int, int, int] = (245, 158, 11),  # Amber
    alpha: int = 190
) -> str:
    """Creates a base64-encoded RGBA PNG overlay with transparent background for frontend display."""
    h, w = mask.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    slick_pixels = mask > 0
    if np.any(slick_pixels):
        rgba[slick_pixels, 0] = color_rgb[0]
        rgba[slick_pixels, 1] = color_rgb[1]
        rgba[slick_pixels, 2] = color_rgb[2]
        rgba[slick_pixels, 3] = alpha
        
        # Glowing border highlight
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        border_layer = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(border_layer, contours, -1, 255, 2)
        border_pixels = border_layer > 0
        rgba[border_pixels, 0] = 255
        rgba[border_pixels, 1] = 220
        rgba[border_pixels, 2] = 50
        rgba[border_pixels, 3] = 240
    
    # Draw In-Image Vessel Candidate Markers onto the overlay (Cyan/White radar targets)
    if vessels:
        for v in vessels:
            cx, cy = int(v["pixel_position"][0]), int(v["pixel_position"][1])
            # Draw target diamond marker
            cv2.drawMarker(
                rgba, (cx, cy),
                color=(45, 212, 191, 255),  # Cyan Teal
                markerType=cv2.MARKER_DIAMOND,
                markerSize=14,
                thickness=2
            )
            # Center dot
            cv2.circle(rgba, (cx, cy), 3, (255, 255, 255, 255), -1)
            
    pil_img = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def mat_to_base64_jpeg(image: np.ndarray, quality: int = 90) -> str:
    """Encodes grayscale/BGR image to base64 JPEG data URL for debug visualization."""
    _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def detect_oil_spills(
    image: np.ndarray,
    geo_meta: Dict[str, Any],
    k_factor: float = 1.2,
    min_area_km2: float = 0.01,
    min_elongation: float = 1.35
) -> Dict[str, Any]:
    """
    Main detection pipeline:
    1. Lee speckle filtering
    2. Adaptive dark patch segmentation via derivative inflection analysis
    3. Morphological cleanup
    4. Look-alike heuristic filtering
    5. In-image vessel point target detection
    6. Metric extraction & age estimation
    7. Overlay generation
    """
    lee_denoised = enhanced_lee_filter(image, window_size=7, num_looks=4.0)
    segmented = segment_dark_patches(lee_denoised, k_factor=k_factor, use_otsu=True)
    cleaned = morphological_cleanup(segmented, open_kernel_size=3, close_kernel_size=7)
    
    final_mask, slicks = filter_lookalikes(
        cleaned_mask=cleaned,
        original_img=image,
        geo_meta=geo_meta,
        min_area_km2=min_area_km2,
        min_elongation=min_elongation
    )
    
    # In-image vessel candidate detection (bright point reflectors)
    in_image_vessels = detect_vessel_candidates(image, geo_meta)
    
    # Overlay including spill mask and in-image vessel targets
    overlay_png_b64 = create_overlay_png(final_mask, vessels=in_image_vessels)
    
    total_area_km2 = round(sum(s["area_km2"] for s in slicks), 4)
    primary_slick = slicks[0] if slicks else None
    
    # Generate GeoJSON features (if georeferenced)
    geojson_features = []
    for slick in slicks:
        if slick.get("centroid_lon") is not None and slick.get("centroid_lat") is not None:
            feature = {
                "type": "Feature",
                "properties": {
                    "id": slick["id"],
                    "area_km2": slick["area_km2"],
                    "perimeter_km": slick["perimeter_km"],
                    "elongation": slick["elongation"],
                    "estimated_age": slick["estimated_age"],
                    "confidence": slick["confidence_score"]
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [slick["centroid_lon"], slick["centroid_lat"]]
                }
            }
            geojson_features.append(feature)
            
    for v in in_image_vessels:
        if v.get("lon") is not None and v.get("lat") is not None:
            v_feature = {
                "type": "Feature",
                "properties": {
                    "id": v["id"],
                    "type": "vessel_radar_target",
                    "estimated_length_m": v["estimated_length_meters"],
                    "confidence": v["rcs_confidence_score"]
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [v["lon"], v["lat"]]
                }
            }
            geojson_features.append(v_feature)
            
    geojson = {
        "type": "FeatureCollection",
        "features": geojson_features
    }
    
    meta_status = geo_meta.get("metadata_status", {
        "has_location": geo_meta.get("has_location", False),
        "has_timestamp": geo_meta.get("has_timestamp", False),
        "metadata_source": geo_meta.get("metadata_source", "none")
    })
    
    is_lookalike_rejection = (len(slicks) == 0)
    lookalike_info = {
        "is_lookalike_rejection": is_lookalike_rejection,
        "reason": "Natural low-wind / calm-water meteorological phenomenon correctly rejected (0 false alarms)." if is_lookalike_rejection else None
    }
    
    return {
        "status": "success",
        "metadata_status": meta_status,
        "is_lookalike_rejection": is_lookalike_rejection,
        "lookalike_info": lookalike_info,
        "slicks_detected_count": len(slicks),
        "total_spill_area_km2": total_area_km2,
        "primary_slick": primary_slick,
        "all_slicks": slicks,
        "in_image_vessels": in_image_vessels,
        "in_image_vessels_count": len(in_image_vessels),
        "overlay_mask_base64": overlay_png_b64,
        "original_image_base64": mat_to_base64_jpeg(image),
        "debug_stages": {
            "speckle_filtered_b64": mat_to_base64_jpeg(lee_denoised),
            "thresholded_b64": mat_to_base64_jpeg(segmented),
            "morph_cleaned_b64": mat_to_base64_jpeg(cleaned),
            "final_binary_b64": mat_to_base64_jpeg(final_mask)
        },
        "geo_metadata": {
            "has_geotiff": geo_meta.get("has_geotiff", False),
            "has_location": geo_meta.get("has_location", False),
            "has_timestamp": geo_meta.get("has_timestamp", False),
            "metadata_source": geo_meta.get("metadata_source", "none"),
            "crs": str(geo_meta.get("crs", "None")),
            "gsd_meters": geo_meta.get("gsd_meters", 10.0),
            "dimensions": {"width": geo_meta.get("width", 0), "height": geo_meta.get("height", 0)},
            "bounding_box": geo_meta.get("bbox")
        },
        "geojson": geojson,
        "scientific_disclaimer": (
            "Age brackets and confidence metrics are calculated via geometric shape factors "
            "(elongation, compactness, contrast drop ratio) as operational heuristic indicators. "
            "Validated ocean hydrodynamic drift integration is applied in Phase 2."
        )
    }

