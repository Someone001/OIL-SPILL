"""
Automated Test Suite for SAR Oil Spill Detection Pipeline
SpillTrace - SIH 2026 PS 26143 (NTRO)

Tests:
1. GeoTIFF loading and Affine Georeferencing
2. Enhanced Lee speckle denoising
3. Adaptive dark patch segmentation
4. Look-alike rejection on calm water test case
5. Geometric feature extraction (Area km², perimeter, elongation, centroid lat/lon)
6. Ground-truth benchmark evaluation (IoU, Dice, Precision, Recall)
"""

import os
import sys
import unittest
import numpy as np
import cv2

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sar_loader import load_sar_image, pixel_to_latlon
from sar_detector import detect_oil_spills, enhanced_lee_filter
from validator import evaluate_segmentation
from samples_generator import SAMPLES_DIR, generate_all_samples


class TestSARPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Ensure sample datasets exist
        generate_all_samples()
        
    def test_01_geotiff_loading(self):
        """Test GeoTIFF loading with WGS84 bounding box and GSD."""
        tif_path = os.path.join(SAMPLES_DIR, "sample_1_tanker_slick.tif")
        self.assertTrue(os.path.exists(tif_path), f"File {tif_path} not found")
        
        img, geo_meta = load_sar_image(file_path=tif_path)
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(img.shape, (512, 512))
        self.assertIn("bbox", geo_meta)
        self.assertGreater(geo_meta["bbox"]["max_lat"], geo_meta["bbox"]["min_lat"])
        print("\n[TEST 1 PASSED] GeoTIFF loaded with valid georeferencing.")
        
    def test_02_enhanced_lee_filter(self):
        """Test speckle reduction while preserving dynamic range."""
        tif_path = os.path.join(SAMPLES_DIR, "sample_1_tanker_slick.tif")
        img, _ = load_sar_image(file_path=tif_path)
        
        filtered = enhanced_lee_filter(img, window_size=7, num_looks=4.0)
        self.assertEqual(filtered.shape, img.shape)
        self.assertEqual(filtered.dtype, np.uint8)
        
        # Local variance in homogeneous area should be reduced
        std_raw = float(np.std(img[:100, :100]))
        std_filt = float(np.std(filtered[:100, :100]))
        self.assertLess(std_filt, std_raw)
        print(f"[TEST 2 PASSED] Enhanced Lee filter reduced speckle variance: {std_raw:.2f} -> {std_filt:.2f}")

    def test_03_tanker_slick_detection_and_validation(self):
        """Test detection and ground truth benchmark for Tanker Discharge (Sample 1)."""
        tif_path = os.path.join(SAMPLES_DIR, "sample_1_tanker_slick.tif")
        gt_path = os.path.join(SAMPLES_DIR, "sample_1_tanker_slick_gt.png")
        
        img, geo_meta = load_sar_image(file_path=tif_path)
        result = detect_oil_spills(img, geo_meta)
        
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["slicks_detected_count"], 0)
        self.assertGreater(result["total_spill_area_km2"], 0.0)
        
        primary = result["primary_slick"]
        self.assertIsNotNone(primary)
        self.assertGreater(primary["elongation"], 1.5)
        self.assertIn("Fresh", primary["estimated_age"])
        
        # Validate against ground truth mask
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        # Decode predicted mask from debug stage or recompute
        from sar_detector import segment_dark_patches, morphological_cleanup, filter_lookalikes
        lee = enhanced_lee_filter(img)
        seg = segment_dark_patches(lee)
        clean = morphological_cleanup(seg)
        pred_mask, _ = filter_lookalikes(clean, img, geo_meta)
        
        eval_metrics = evaluate_segmentation(pred_mask, gt_mask)
        print(f"\n[TEST 3 PASSED] Tanker Slick IoU: {eval_metrics['iou_percentage']}, Dice: {eval_metrics['dice_percentage']}, Precision: {eval_metrics['precision_percentage']}")
        self.assertGreater(eval_metrics["iou"], 0.70, "IoU should exceed benchmark threshold 0.70")

    def test_04_bilge_dump_detection(self):
        """Test detection and validation on narrow high-elongation bilge dumping (Sample 2)."""
        tif_path = os.path.join(SAMPLES_DIR, "sample_2_bilge_dump.tif")
        gt_path = os.path.join(SAMPLES_DIR, "sample_2_bilge_dump_gt.png")
        img, geo_meta = load_sar_image(file_path=tif_path)
        result = detect_oil_spills(img, geo_meta)
        
        self.assertGreater(result["slicks_detected_count"], 0)
        primary = result["primary_slick"]
        self.assertGreater(primary["elongation"], 2.0)
        
        # Ground truth validation
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        lee = enhanced_lee_filter(img)
        from sar_detector import segment_dark_patches, morphological_cleanup, filter_lookalikes
        seg = segment_dark_patches(lee)
        clean = morphological_cleanup(seg)
        pred_mask, _ = filter_lookalikes(clean, img, geo_meta)
        
        eval_metrics = evaluate_segmentation(pred_mask, gt_mask)
        print(f"\n[TEST 4 PASSED] Malacca Bilge Dump IoU: {eval_metrics['iou_percentage']}, Dice: {eval_metrics['dice_percentage']}, Precision: {eval_metrics['precision_percentage']}")
        self.assertGreater(eval_metrics["iou"], 0.70, "Bilge Dump IoU should exceed benchmark threshold 0.70")

    def test_05_offshore_platform_detection(self):
        """Test detection and validation on spreading offshore platform blowout plume (Sample 3)."""
        tif_path = os.path.join(SAMPLES_DIR, "sample_3_offshore_platform.tif")
        gt_path = os.path.join(SAMPLES_DIR, "sample_3_offshore_platform_gt.png")
        img, geo_meta = load_sar_image(file_path=tif_path)
        result = detect_oil_spills(img, geo_meta)
        
        self.assertGreater(result["slicks_detected_count"], 0)
        
        # Ground truth validation
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        lee = enhanced_lee_filter(img)
        from sar_detector import segment_dark_patches, morphological_cleanup, filter_lookalikes
        seg = segment_dark_patches(lee)
        clean = morphological_cleanup(seg)
        pred_mask, _ = filter_lookalikes(clean, img, geo_meta)
        
        eval_metrics = evaluate_segmentation(pred_mask, gt_mask)
        print(f"\n[TEST 5 PASSED] Mumbai High Offshore Platform IoU: {eval_metrics['iou_percentage']}, Dice: {eval_metrics['dice_percentage']}, Precision: {eval_metrics['precision_percentage']}")
        self.assertGreater(eval_metrics["iou"], 0.70, "Offshore Platform IoU should exceed benchmark threshold 0.70")

    def test_06_lookalike_rejection(self):
        """Test look-alike discrimination heuristic on calm water false alarm test (Sample 4)."""
        tif_path = os.path.join(SAMPLES_DIR, "sample_4_lookalike_calm_water.tif")
        img, geo_meta = load_sar_image(file_path=tif_path)
        result = detect_oil_spills(img, geo_meta)
        
        # Natural calm water should be filtered out
        self.assertEqual(result["slicks_detected_count"], 0, "Calm water look-alike should be rejected")
        self.assertTrue(result["is_lookalike_rejection"], "Should flag is_lookalike_rejection: True")
        print("\n[TEST 6 PASSED] Look-alike false alarm correctly rejected (0 false slicks detected).")


if __name__ == "__main__":
    unittest.main(verbosity=2)

