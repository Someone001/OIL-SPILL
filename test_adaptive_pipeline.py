"""
Unit tests for Metadata Detection & In-Image Vessel Detection
SpillTrace - SIH 2026 PS 26143 (NTRO)
"""

import os
import cv2
import numpy as np
from sar_loader import load_sar_image, extract_sar_metadata
from sar_detector import detect_oil_spills
from vessel_detector import detect_vessel_candidates
from ais_correlator import attribute_spill_to_vessels


def test_metadata_extraction():
    print("=== Testing Phase 1: Metadata Detection ===")
    
    # 1. Test GeoTIFF file
    tif_path = "backend/python_service/samples/sample_1_tanker_slick.tif"
    if os.path.exists(tif_path):
        meta_tif = extract_sar_metadata(file_path=tif_path)
        print(f"GeoTIFF Metadata Status: {meta_tif}")
        assert meta_tif["has_location"] is True, "GeoTIFF should have location metadata"
        assert meta_tif["metadata_source"] == "embedded_geotiff"
        print(" GeoTIFF metadata extraction test passed.")
        
    # 2. Test plain PNG without GeoTIFF headers
    png_path = "backend/python_service/samples/sample_1_tanker_slick.png"
    with open(png_path, "rb") as f:
        png_bytes = f.read()
        
    meta_png = extract_sar_metadata(file_bytes=png_bytes)
    print(f"Plain PNG Bytes Metadata Status: {meta_png}")
    assert meta_png["has_location"] is False, "Plain PNG bytes should honestly report has_location: False"
    assert meta_png["metadata_source"] == "none"
    print(" Plain PNG (no geodata) metadata extraction test passed.")


def test_in_image_vessel_detector():
    print("\n=== Testing Phase 4: In-Image Vessel Detection ===")
    
    # Create test SAR image with synthetic ship bright spot
    img = np.full((300, 300), 40, dtype=np.uint8) # Dark sea
    # Add a bright corner reflector ship target at (150, 150)
    cv2.circle(img, (150, 150), 4, 240, -1)
    
    geo_meta = {"has_geotiff": False, "has_location": False, "gsd_meters": 10.0, "width": 300, "height": 300}
    vessels = detect_vessel_candidates(img, geo_meta)
    print(f"Detected {len(vessels)} vessel targets in test image:")
    for v in vessels:
        print(f" - {v['id']}: Pixel pos {v['pixel_position']}, Brightness {v['peak_radar_backscatter']}, Est Length {v['estimated_length_meters']}m")
        
    assert len(vessels) >= 1, "Should detect at least 1 vessel point target"
    assert abs(vessels[0]["pixel_position"][0] - 150) <= 2.0
    print(" In-Image Vessel Detection test passed.")


def test_satellite_cross_referencing():
    print("\n=== Testing Phase 5: Satellite Target & AIS Cross-Referencing ===")
    
    in_image_vessels = [{
        "id": "RADAR-VESSEL-01",
        "pixel_position": [250, 250],
        "lat": 25.815,
        "lon": 54.825,
        "rcs_confidence_score": 95.0
    }]
    
    suspects = attribute_spill_to_vessels(
        origin_lat=25.815,
        origin_lon=54.825,
        in_image_vessels=in_image_vessels
    )
    
    print(f"Top suspect: {suspects[0]['vessel_name']} ({suspects[0]['total_suspicion_score']}%)")
    print(f"Confirmed by satellite target: {suspects[0]['is_satellite_confirmed']}")
    print(f"Anomaly flags: {suspects[0]['anomaly_flags']}")
    
    assert suspects[0]["is_satellite_confirmed"] is True, "Top suspect should be confirmed by satellite radar target"
    print(" Satellite & AIS Cross-Referencing test passed.")


if __name__ == "__main__":
    test_metadata_extraction()
    test_in_image_vessel_detector()
    test_satellite_cross_referencing()
    print("\n ALL ADAPTIVE PIPELINE CORE TESTS PASSED!")
