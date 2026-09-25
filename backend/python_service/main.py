"""
FastAPI Microservice for SAR Oil Spill Detection
SpillTrace - SIH 2026 PS 26143 (NTRO)

Endpoints:
- POST /api/detect: Process uploaded SAR GeoTIFF/PNG or sample ID
- GET  /api/samples: List curated Sentinel-1 SAR benchmark datasets
- POST /api/benchmark: Evaluate detection vs Ground Truth (IoU, Dice, Precision, Recall)
- GET  /api/health: Service health & capabilities
"""

import os
import json
from typing import Optional, Dict, Any
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from sar_loader import load_sar_image
from sar_detector import detect_oil_spills, enhanced_lee_filter, segment_dark_patches, morphological_cleanup, filter_lookalikes
from validator import evaluate_segmentation
from samples_generator import SAMPLES_DIR, generate_all_samples

app = FastAPI(
    title="SpillTrace SAR CV Microservice",
    description="Sentinel-1 SAR Oil Spill Detection & Attribution Engine for NTRO PS 26143",
    version="1.0.0"
)

# Enable CORS for local dev and production frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure sample files exist on startup
@app.on_event("startup")
async def startup_event():
    if not os.path.exists(SAMPLES_DIR) or len(os.listdir(SAMPLES_DIR)) < 4:
        generate_all_samples()


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "SpillTrace Python SAR CV Microservice",
        "version": "1.0.0",
        "features": [
            "Enhanced Lee Speckle Filter (ENL=4.0)",
            "Adaptive Sea Clutter Background Segmentation",
            "Morphological Slick Reconnection",
            "Calm-Water Look-Alike Heuristic Discrimination",
            "GeoTIFF WGS84 Affine Lat/Lon Georeferencing",
            "Zenodo Ground Truth IoU & Dice Benchmark Evaluation"
        ]
    }


@app.get("/api/samples")
async def list_samples():
    """Returns catalog of preloaded Sentinel-1 SAR sample cases."""
    samples = [
        {
            "id": "sample_1_tanker_slick",
            "title": "Persian Gulf Crude Tanker Discharge",
            "category": "Operational Tanker Spill",
            "region": "Strait of Hormuz, Persian Gulf",
            "vessel_hint": "MT AL-BARAKA (VLCC Crude Carrier)",
            "lat": 25.82,
            "lon": 55.12,
            "gsd_meters": 10.0,
            "description": "Elongated continuous crude discharge from transit corridor with pronounced wake trail.",
            "has_ground_truth": True
        },
        {
            "id": "sample_2_bilge_dump",
            "title": "Strait of Malacca Bilge Dumping",
            "category": "Illegal Oily Bilge Wash",
            "region": "Malacca Strait Shipping Fairway",
            "vessel_hint": "MV OCEAN VOYAGER (Container Carrier)",
            "lat": 3.15,
            "lon": 100.80,
            "gsd_meters": 10.0,
            "description": "High-speed narrow illegal bilge wash discharge along major Indo-Pacific transit corridor.",
            "has_ground_truth": True
        },
        {
            "id": "sample_3_offshore_platform",
            "title": "Mumbai High Offshore Rig Leakage",
            "category": "Offshore Platform Leak",
            "region": "Mumbai High Basin, Arabian Sea",
            "vessel_hint": "OFFSHORE PLATFORM MH-NORTH",
            "lat": 19.42,
            "lon": 71.33,
            "gsd_meters": 10.0,
            "description": "Continuous offshore subsea riser pipeline leak with asymmetric wind shear spreading.",
            "has_ground_truth": True
        },
        {
            "id": "sample_4_lookalike_calm_water",
            "title": "Goa Coastal Low-Wind Pocket (Look-Alike Rejection Test)",
            "category": "Natural Meteorological Look-Alike",
            "region": "Konkan Coast, Arabian Sea",
            "vessel_hint": "N/A — Natural Phenomenon",
            "lat": 15.20,
            "lon": 73.50,
            "gsd_meters": 10.0,
            "description": "Atmospheric wind shelter creating low radar backscatter look-alike without sharp slick edges.",
            "has_ground_truth": False
        }
    ]
    return {"samples": samples}


@app.post("/api/detect")
async def detect_spill(
    sample_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    k_factor: float = Form(1.2),
    min_area_km2: float = Form(0.02),
    min_elongation: float = Form(1.35)
):
    """
    Main detection endpoint:
    Processes uploaded SAR image or pre-loaded sample ID.
    Returns segmentation mask, metrics, contours, georeferenced coordinates, and age estimate.
    """
    try:
        img_array = None
        geo_meta = None
        active_sample_id = None
        
        if sample_id:
            active_sample_id = sample_id
            tif_path = os.path.join(SAMPLES_DIR, f"{sample_id}.tif")
            png_path = os.path.join(SAMPLES_DIR, f"{sample_id}.png")
            
            if os.path.exists(tif_path):
                img_array, geo_meta = load_sar_image(file_path=tif_path)
            elif os.path.exists(png_path):
                img_array, geo_meta = load_sar_image(file_path=png_path)
            else:
                raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found.")
        elif file:
            contents = await file.read()
            img_array, geo_meta = load_sar_image(file_bytes=contents)
            active_sample_id = "custom_upload"
        else:
            raise HTTPException(status_code=400, detail="Provide either 'sample_id' or 'file'.")
            
        # Run computer vision detection pipeline
        detection_result = detect_oil_spills(
            image=img_array,
            geo_meta=geo_meta,
            k_factor=k_factor,
            min_area_km2=min_area_km2,
            min_elongation=min_elongation
        )
        
        detection_result["sample_id"] = active_sample_id
        
        # Check if ground truth mask is available for this sample
        gt_eval = None
        if active_sample_id and active_sample_id != "custom_upload":
            gt_path = os.path.join(SAMPLES_DIR, f"{active_sample_id}_gt.png")
            if os.path.exists(gt_path):
                gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt_mask is not None:
                    # Compute prediction binary mask
                    lee = enhanced_lee_filter(img_array)
                    seg = segment_dark_patches(lee, k_factor=k_factor)
                    clean = morphological_cleanup(seg)
                    pred_mask, _ = filter_lookalikes(clean, img_array, geo_meta, min_area_km2, min_elongation)
                    gt_eval = evaluate_segmentation(pred_mask, gt_mask)
                    
        detection_result["ground_truth_benchmark"] = gt_eval
        return JSONResponse(content=detection_result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class BenchmarkRequest(BaseModel):
    sample_id: str


@app.post("/api/benchmark")
async def run_benchmark(req: BenchmarkRequest):
    """Evaluates pipeline accuracy vs Zenodo Ground Truth mask."""
    tif_path = os.path.join(SAMPLES_DIR, f"{req.sample_id}.tif")
    gt_path = os.path.join(SAMPLES_DIR, f"{req.sample_id}_gt.png")
    
    if not os.path.exists(tif_path) or not os.path.exists(gt_path):
        raise HTTPException(status_code=404, detail="Sample or ground truth mask not found.")
        
    img_array, geo_meta = load_sar_image(file_path=tif_path)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    
    # Process
    lee = enhanced_lee_filter(img_array)
    seg = segment_dark_patches(lee)
    clean = morphological_cleanup(seg)
    pred_mask, slicks = filter_lookalikes(clean, img_array, geo_meta)
    
    metrics = evaluate_segmentation(pred_mask, gt_mask)
    return {
        "sample_id": req.sample_id,
        "metrics": metrics,
        "slick_count": len(slicks)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
