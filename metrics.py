"""
Geometric Metrics & Age Estimation Module
SpillTrace - SIH 2026 PS 26143 (NTRO)

Computes:
1. Physical Area (km²) using Sentinel-1 GSD resolution
2. Perimeter (km) from contour arc length
3. Elongation / Aspect Ratio via inertia moment tensors
4. Orientation angle (degrees from North)
5. Compactness (Isoperimetric Quotient)
6. Centroid coordinates (Pixel and WGS84 Lat/Lon if available)
7. Age Estimation Heuristic (with scientific commentary)
"""

import math
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2

from sar_loader import pixel_to_latlon


def calculate_geometric_properties(contour: np.ndarray, geo_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts physical and spatial geometric properties of a detected slick contour.
    """
    gsd_m = geo_meta.get("gsd_meters", 10.0)  # meters per pixel
    pixel_area_km2 = (gsd_m * gsd_m) / 1_000_000.0  # km² per pixel
    
    # 1. Area in pixels & physical km²
    area_px = cv2.contourArea(contour)
    area_km2 = round(area_px * pixel_area_km2, 4)
    
    # 2. Perimeter in pixels & physical km
    perimeter_px = cv2.arcLength(contour, closed=True)
    perimeter_km = round((perimeter_px * gsd_m) / 1_000.0, 4)
    
    # 3. Compactness (Isoperimetric quotient: 4 * pi * Area / Perimeter^2)
    # Circle = 1.0, highly elongated / fractured slick << 0.2
    if perimeter_px > 0:
        compactness = round((4.0 * math.pi * area_px) / (perimeter_px ** 2), 4)
    else:
        compactness = 0.0
        
    # 4. Centroid & Moments
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx_px = float(M["m10"] / M["m00"])
        cy_px = float(M["m01"] / M["m00"])
    else:
        cx_px, cy_px = float(contour[0][0][0]), float(contour[0][0][1])
        
    # Convert pixel centroid to Latitude / Longitude if georeferenced
    centroid_lat, centroid_lon = pixel_to_latlon(cx_px, cy_px, geo_meta)
    
    # 5. Elongation / Aspect Ratio via Normalized Central Moments
    # Moment tensor eigenvalues lambda1 (major), lambda2 (minor)
    elongation = 1.0
    orientation_deg = 0.0
    
    if M["m00"] > 0:
        mu20 = M["mu20"] / M["m00"]
        mu02 = M["mu02"] / M["m00"]
        mu11 = M["mu11"] / M["m00"]
        
        delta = math.sqrt(max(0.0, (mu20 - mu02)**2 + 4.0 * (mu11**2)))
        lambda1 = (mu20 + mu02 + delta) / 2.0
        lambda2 = max(1e-6, (mu20 + mu02 - delta) / 2.0)
        
        elongation = round(math.sqrt(lambda1 / lambda2), 2)
        
        # Orientation angle in degrees
        angle_rad = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
        orientation_deg = round(math.degrees(angle_rad), 1)
        
    # 6. Bounding Box in Pixels and Lat/Lon
    x, y, w, h = cv2.boundingRect(contour)
    min_lat, min_lon = pixel_to_latlon(x, y + h, geo_meta)
    max_lat, max_lon = pixel_to_latlon(x + w, y, geo_meta)
    
    bbox_geo = None
    if min_lat is not None and max_lat is not None and min_lon is not None and max_lon is not None:
        bbox_geo = {
            "min_lat": min(min_lat, max_lat),
            "max_lat": max(min_lat, max_lat),
            "min_lon": min(min_lon, max_lon),
            "max_lon": max(min_lon, max_lon)
        }
    
    return {
        "area_px": round(area_px, 1),
        "area_km2": area_km2,
        "perimeter_px": round(perimeter_px, 1),
        "perimeter_km": perimeter_km,
        "compactness": compactness,
        "elongation": elongation,
        "orientation_deg": orientation_deg,
        "centroid_px": [round(cx_px, 1), round(cy_px, 1)],
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "bbox_px": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
        "bbox_geo": bbox_geo
    }


def estimate_slick_age(elongation: float, compactness: float, area_km2: float) -> Dict[str, str]:
    """
    Operational heuristic for slick age bracket estimation based on morphometrics.
    
    Scientific Basis & Hackathon Transparency:
    - Fresh spills (<6 hrs): Released in a continuous line from moving vessel or localized blowout;
      high core density, distinct boundary, relatively high compactness or sharp linear alignment.
    - Spreading (6-24 hrs): Subjected to ocean surface wind shear and Langmuir circulation,
      spreading into elongated ribbons/windrows with moderate elongation (2.5 - 6.0).
    - Weathered / Dispersed (>24 hrs): Fragmented into patches and sheen, lower edge contrast.
    """
    if elongation >= 4.0:
        return {
            "bracket": "Fresh (<6 hrs)",
            "reasoning": f"Pronounced linear wake trail (Aspect Ratio: {elongation:.1f}x) characteristic of an underway vessel discharge."
        }
    elif elongation >= 2.0 or compactness < 0.15:
        return {
            "bracket": "Spreading (6-24 hrs)",
            "reasoning": f"Moderate elongation ({elongation:.1f}x) indicating wind/current spreading."
        }
    else:
        return {
            "bracket": "Weathered (>24 hrs)",
            "reasoning": f"Diffused patch geometry ({elongation:.1f}x) with non-linear fragmentation."
        }
