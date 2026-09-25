"""
SpillTrace Adaptive Maritime Intelligence System
SIH 2026 PS 26143 (NTRO)

Adaptive Multi-Phase Real-World Pipeline:
- Phase 1: Metadata Pre-Flight Check (GeoTIFF tags, EXIF, honest reporting)
- Phase 2: Graceful Pipeline Branching (Auto-continue if geodata found; honest halt if missing)
- Phase 3: Manual Location/Time Override (Analyst input with presets & validation)
- Phase 4: In-Image Vessel Point-Target Detection (Bright radar corner reflectors on ALL images)
- Phase 5: Satellite Target & AIS Transponder Cross-Referencing
- Dual-Mode UI: Simple (Plain-English) vs Technical (CV / MetOcean telemetry)
"""

import os
import io
import json
import base64
import datetime
from typing import Optional, Dict, Any, List
import numpy as np
import cv2
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel

from sar_loader import load_sar_image, extract_sar_metadata
from sar_detector import (
    detect_oil_spills,
    enhanced_lee_filter,
    segment_dark_patches,
    morphological_cleanup,
    filter_lookalikes
)
from vessel_detector import detect_vessel_candidates
from validator import evaluate_segmentation
from samples_generator import SAMPLES_DIR, generate_all_samples
from drift_tracer import trace_backward_drift
from ais_correlator import attribute_spill_to_vessels

app = FastAPI(title="SpillTrace Adaptive Server", version="4.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    if not os.path.exists(SAMPLES_DIR) or len(os.listdir(SAMPLES_DIR)) < 4:
        generate_all_samples()

@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "service": "SpillTrace Adaptive Maritime Intelligence System",
        "version": "4.2.0",
        "modes": ["simple", "technical"],
        "adaptive_pipeline": {
            "metadata_detection": "active",
            "graceful_branching": "active",
            "manual_override": "active",
            "in_image_vessel_detection": "active",
            "satellite_ais_cross_referencing": "active"
        }
    }

@app.get("/api/samples")
async def list_samples():
    samples = [
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
    return {"samples": samples}

@app.post("/api/metadata")
async def check_image_metadata(
    sample_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """Phase 1 Pre-Flight Metadata Inspector."""
    try:
        if sample_id and sample_id != "custom_upload" and sample_id != "null":
            tif_path = os.path.join(SAMPLES_DIR, f"{sample_id}.tif")
            if os.path.exists(tif_path):
                meta = extract_sar_metadata(file_path=tif_path)
            else:
                meta = extract_sar_metadata(file_path=os.path.join(SAMPLES_DIR, f"{sample_id}.png"))
        elif file:
            contents = await file.read()
            meta = extract_sar_metadata(file_bytes=contents)
        else:
            raise HTTPException(status_code=400, detail="Provide sample_id or file.")
        return JSONResponse(content=meta)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect")
async def detect_spill(
    sample_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    k_factor: Optional[float] = Form(1.2),
    min_area_km2: Optional[float] = Form(0.01),
    min_elongation: Optional[float] = Form(1.35)
):
    """Phase 1 + 2 + 4: SAR Detection, Metadata Evaluation, In-Image Vessel Extraction."""
    try:
        img_array = None
        geo_meta = None
        active_sample_id = None
        is_sample = False
        
        if sample_id and sample_id != "custom_upload" and sample_id != "null":
            active_sample_id = sample_id
            is_sample = True
            tif_path = os.path.join(SAMPLES_DIR, f"{sample_id}.tif")
            png_path = os.path.join(SAMPLES_DIR, f"{sample_id}.png")
            
            if os.path.exists(tif_path):
                img_array, geo_meta = load_sar_image(file_path=tif_path, is_sample=True)
            elif os.path.exists(png_path):
                img_array, geo_meta = load_sar_image(file_path=png_path, is_sample=True)
            else:
                raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found.")
        elif file:
            contents = await file.read()
            img_array, geo_meta = load_sar_image(file_bytes=contents, is_sample=False)
            active_sample_id = "custom_upload"
        else:
            raise HTTPException(status_code=400, detail="Provide either 'sample_id' or 'file'.")
            
        detection_result = detect_oil_spills(
            image=img_array,
            geo_meta=geo_meta,
            k_factor=k_factor or 1.2,
            min_area_km2=min_area_km2 or 0.01,
            min_elongation=min_elongation or 1.35
        )
        detection_result["sample_id"] = active_sample_id
        
        gt_eval = None
        if active_sample_id and active_sample_id != "custom_upload":
            gt_path = os.path.join(SAMPLES_DIR, f"{active_sample_id}_gt.png")
            if os.path.exists(gt_path):
                gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt_mask is not None and np.sum(gt_mask) > 0:
                    lee = enhanced_lee_filter(img_array)
                    seg = segment_dark_patches(lee, k_factor=k_factor or 1.2)
                    clean = morphological_cleanup(seg)
                    pred_mask, _ = filter_lookalikes(clean, img_array, geo_meta, min_area_km2 or 0.01, min_elongation or 1.35)
                    gt_eval = evaluate_segmentation(pred_mask, gt_mask)
                    
        detection_result["ground_truth_benchmark"] = gt_eval
        return JSONResponse(content=detection_result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class DriftRequest(BaseModel):
    lat: float
    lon: float
    timestamp: Optional[str] = None
    hours_back: int = 24
    wind_factor: float = 0.03
    coriolis_angle_deg: float = 20.0
    is_manual_override: bool = False


@app.post("/api/drift")
async def run_drift_hindcast(req: DriftRequest):
    try:
        drift_res = trace_backward_drift(
            detection_lat=req.lat,
            detection_lon=req.lon,
            detection_timestamp=req.timestamp,
            hours_back=req.hours_back,
            wind_factor=req.wind_factor,
            coriolis_angle_deg=req.coriolis_angle_deg
        )
        drift_res["is_manual_override"] = req.is_manual_override
        return JSONResponse(content=drift_res)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class AISRequest(BaseModel):
    origin_lat: float
    origin_lon: float
    origin_time_iso: Optional[str] = None
    scenario_hint: Optional[str] = None
    in_image_vessels: Optional[List[Dict[str, Any]]] = None
    max_radius_km: float = 50.0
    is_manual_override: bool = False


@app.post("/api/ais/correlate")
async def correlate_ais_vessels(req: AISRequest):
    try:
        suspects = attribute_spill_to_vessels(
            origin_lat=req.origin_lat,
            origin_lon=req.origin_lon,
            origin_time_iso=req.origin_time_iso,
            scenario_hint=req.scenario_hint,
            in_image_vessels=req.in_image_vessels,
            max_radius_km=req.max_radius_km
        )
        return JSONResponse(content={
            "status": "success",
            "is_manual_override": req.is_manual_override,
            "candidate_count": len(suspects),
            "top_suspect": suspects[0] if suspects else None,
            "ranked_vessels": suspects
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/bg.png")
async def get_bg_image():
    bg_path = os.path.join(os.path.dirname(__file__), "bg.png")
    if os.path.exists(bg_path):
        return FileResponse(bg_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Background image not found")

@app.get("/logo.png")
async def get_logo_image():
    logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
    if not os.path.exists(logo_path):
        logo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logo.png"))
    if os.path.exists(logo_path):
        return FileResponse(logo_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Logo image not found")

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serves the SpillTrace Maritime Intelligence Dashboard."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(html_path):
        html_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "index.html"))
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Dashboard file index.html not found.</h1>", status_code=404)


if __name__ == "__main__":
    uvicorn.run("local_server:app", host="0.0.0.0", port=8000, reload=False)
