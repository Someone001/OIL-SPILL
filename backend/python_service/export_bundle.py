import os
import json
import base64
import numpy as np
import cv2
from sar_loader import load_sar_image, extract_sar_metadata
from sar_detector import detect_oil_spills, enhanced_lee_filter, segment_dark_patches, morphological_cleanup, filter_lookalikes
from validator import evaluate_segmentation
from drift_tracer import trace_backward_drift
from ais_correlator import attribute_spill_to_vessels
from samples_generator import SAMPLES_DIR

def export_all():
    samples_list = [
        {
            "id": "sample_1_tanker_slick",
            "title": "Persian Gulf Crude Tanker Discharge",
            "simple_title": "Case 1: Persian Gulf Ship Discharge",
            "category": "Operational Tanker Spill",
            "region": "Strait of Hormuz, Persian Gulf",
            "vessel_hint": "MT AL-BARAKA (VLCC Crude Carrier)",
            "lat": 25.82,
            "lon": 55.12,
            "gsd_meters": 10.0,
            "description": "Elongated continuous crude discharge from transit corridor with pronounced wake trail.",
            "simple_desc": "Long oil slick trailed behind a crude oil supertanker along a major shipping lane.",
            "has_ground_truth": True,
            "has_embedded_geotiff": True
        },
        {
            "id": "sample_2_bilge_dump",
            "title": "Strait of Malacca Bilge Dumping",
            "simple_title": "Case 2: Malacca Strait Illegal Wash",
            "category": "Illegal Oily Bilge Wash",
            "region": "Malacca Strait Shipping Fairway",
            "vessel_hint": "MV OCEAN VOYAGER (Container Carrier)",
            "lat": 3.15,
            "lon": 100.80,
            "gsd_meters": 10.0,
            "description": "High-speed narrow illegal bilge wash discharge along major Indo-Pacific transit corridor.",
            "simple_desc": "Narrow high-speed illegal oily bilge water dump at night along a busy cargo route.",
            "has_ground_truth": True,
            "has_embedded_geotiff": True
        },
        {
            "id": "sample_3_offshore_platform",
            "title": "Mumbai High Offshore Rig Leakage",
            "simple_title": "Case 3: Mumbai High Rig Pipeline Leak",
            "category": "Offshore Platform Leak",
            "region": "Mumbai High Basin, Arabian Sea",
            "vessel_hint": "OFFSHORE PLATFORM MH-NORTH",
            "lat": 19.42,
            "lon": 71.33,
            "gsd_meters": 10.0,
            "description": "Continuous offshore subsea riser pipeline leak with asymmetric wind shear spreading.",
            "simple_desc": "Spreading oil plume from an offshore production rig pipeline in the Arabian Sea.",
            "has_ground_truth": True,
            "has_embedded_geotiff": True
        },
        {
            "id": "sample_4_lookalike_calm_water",
            "title": "Goa Coastal Low-Wind Pocket (Look-Alike Rejection Test)",
            "simple_title": "Case 4: Calm Sea False-Alarm Test",
            "category": "Natural Meteorological Look-Alike",
            "region": "Konkan Coast, Arabian Sea",
            "vessel_hint": "N/A — Natural Phenomenon",
            "lat": 15.20,
            "lon": 73.50,
            "gsd_meters": 10.0,
            "description": "Atmospheric wind shelter creating low radar backscatter look-alike without sharp slick edges.",
            "simple_desc": "Natural calm-water patch created by low wind (correctly identified as NOT oil).",
            "has_ground_truth": False,
            "has_embedded_geotiff": True
        }
    ]

    bundle = {
        "samples": samples_list,
        "presets": {}
    }

    for s in samples_list:
        sid = s["id"]
        tif_path = os.path.join(SAMPLES_DIR, f"{sid}.tif")
        png_path = os.path.join(SAMPLES_DIR, f"{sid}.png")
        
        if os.path.exists(tif_path):
            img_array, geo_meta = load_sar_image(file_path=tif_path, is_sample=True)
        else:
            img_array, geo_meta = load_sar_image(file_path=png_path, is_sample=True)
            
        detect_res = detect_oil_spills(img_array, geo_meta=geo_meta, k_factor=1.2, min_area_km2=0.01, min_elongation=1.35)
        detect_res["sample_id"] = sid

        # Ground truth evaluation
        gt_path = os.path.join(SAMPLES_DIR, f"{sid}_gt.png")
        gt_eval = None
        if os.path.exists(gt_path):
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None and np.sum(gt_mask) > 0:
                lee = enhanced_lee_filter(img_array)
                seg = segment_dark_patches(lee, k_factor=1.2)
                clean = morphological_cleanup(seg)
                pred_mask, _ = filter_lookalikes(clean, img_array, geo_meta, 0.01, 1.35)
                gt_eval = evaluate_segmentation(pred_mask, gt_mask)
        detect_res["ground_truth_benchmark"] = gt_eval

        is_lookalike = detect_res.get("is_lookalike_rejection", False) or (detect_res.get("slicks_detected_count", 0) == 0)

        drift_res = None
        ais_res = None

        if not is_lookalike and detect_res.get("primary_slick"):
            lat = detect_res["primary_slick"]["centroid_lat"]
            lon = detect_res["primary_slick"]["centroid_lon"]
            drift_res = trace_backward_drift(lat, lon, hours_back=24)
            drift_res["is_manual_override"] = False

            orig_lat = drift_res["estimated_origin"]["lat"]
            orig_lon = drift_res["estimated_origin"]["lon"]
            suspects = attribute_spill_to_vessels(
                origin_lat=orig_lat,
                origin_lon=orig_lon,
                scenario_hint=sid,
                in_image_vessels=detect_res.get("in_image_vessels", [])
            )
            ais_res = {
                "status": "success",
                "is_manual_override": False,
                "candidate_count": len(suspects),
                "top_suspect": suspects[0] if suspects else None,
                "ranked_vessels": suspects
            }

        bundle["presets"][sid] = {
            "detect": detect_res,
            "drift": drift_res,
            "ais": ais_res
        }
        print(f"Exported preset for {sid}: detected {detect_res.get('slicks_detected_count')} slicks, GT IoU: {gt_eval['iou'] if gt_eval else 'N/A'}")

    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exported_samples_bundle.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(bundle, f)
    print(f"Done! Written to {out_file}, file size: {os.path.getsize(out_file)} bytes.")

if __name__ == "__main__":
    export_all()
