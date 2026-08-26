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
from fastapi.responses import HTMLResponse, JSONResponse
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
    min_area_km2: Optional[float] = Form(0.02),
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
            min_area_km2=min_area_km2 or 0.02,
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
                    pred_mask, _ = filter_lookalikes(clean, img_array, geo_meta, min_area_km2 or 0.02, min_elongation or 1.35)
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


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serves the complete state-of-the-art SpillTrace Adaptive Dashboard."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SpillTrace — Adaptive Oil Spill Intelligence & Vessel Attribution (NTRO PS 26143)</title>
  <!-- Google Fonts: Space Grotesk, Inter, IBM Plex Mono -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
  <!-- Tailwind CSS CDN -->
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- Leaflet CSS & JS for Maritime Map -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <!-- Lucide Icons -->
  <script src="https://unpkg.com/lucide@latest"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            navy: { 950: '#070C14', 900: '#0C1322', 850: '#111B30', 800: '#18243C', 700: '#263756' },
            slate: { 850: '#152033', 800: '#1E293B', 700: '#334155', 600: '#475569', 400: '#94A3B8' },
            teal: { 400: '#2DD4BF', 500: '#14B8A6', 600: '#0D9488', 950: '#042F2E' },
            amber: { 400: '#FBBF24', 500: '#F59E0B', 600: '#D97706', 950: '#451A03' },
            crimson: { 400: '#F87171', 500: '#EF4444', 600: '#DC2626', 950: '#450A0A' },
            radar: '#10B981'
          },
          fontFamily: {
            heading: ['"Space Grotesk"', 'sans-serif'],
            body: ['"Inter"', 'sans-serif'],
            mono: ['"IBM Plex Mono"', 'monospace']
          }
        }
      }
    }
  </script>
  <style>
    body { background-color: #070C14; color: #E2E8F0; font-family: 'Inter', sans-serif; }
    .ops-panel { background: #0C1322; border: 1px solid #1E293B; }
    .ops-card { background: #111B30; border: 1px solid #1E293B; }
    .scan-line {
      background: linear-gradient(180deg, rgba(20,184,166,0) 0%, rgba(20,184,166,0.3) 50%, rgba(20,184,166,0) 100%);
      animation: scan 4s ease-in-out infinite;
    }
    @keyframes scan { 0% { top: 0%; } 50% { top: 92%; } 100% { top: 0%; } }
    .radar-pulse { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
    .leaflet-container { background: #070C14 !important; }
    
    /* Interactive Tooltip Style */
    .has-tooltip { position: relative; display: inline-flex; align-items: center; cursor: help; }
    .tooltip-box {
      visibility: hidden; opacity: 0; position: absolute; bottom: 125%; left: 50%;
      transform: translateX(-50%); width: 220px; background: #070C14; border: 1px solid #14B8A6;
      color: #E2E8F0; font-family: 'Inter', sans-serif; font-size: 11px; padding: 6px 9px;
      border-radius: 6px; z-index: 100; transition: opacity 0.2s; pointer-events: none;
      box-shadow: 0 4px 14px rgba(0,0,0,0.6);
    }
    .has-tooltip:hover .tooltip-box { visibility: visible; opacity: 1; }
    
    /* Drag over active state */
    .drag-active {
      border-color: #2DD4BF !important;
      background-color: rgba(20, 184, 166, 0.15) !important;
    }
  </style>
</head>
<body class="min-h-screen flex flex-col antialiased selection:bg-teal-500 selection:text-black">

  <!-- Top Mission Bar & Stepper -->
  <header class="border-b border-slate-800 bg-navy-950 sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      
      <!-- Brand & Mode Indicator -->
      <div class="flex items-center space-x-3">
        <div class="w-8 h-8 rounded bg-teal-500 text-navy-950 flex items-center justify-center font-bold">
          <i data-lucide="crosshair" class="w-4 h-4"></i>
        </div>
        <div>
          <div class="flex items-center space-x-2">
            <span class="font-heading font-bold text-lg tracking-wider text-white">SPILL<span class="text-teal-400">TRACE</span></span>
            <span id="header-mode-badge" class="px-2 py-0.5 text-[10px] font-mono rounded bg-teal-950 text-teal-400 border border-teal-500/30">SIMPLE MODE</span>
          </div>
          <p id="header-subtitle" class="text-[10px] text-gray-400 font-mono">ADAPTIVE SATELLITE SPILL & SHIP ATTRIBUTION SYSTEM</p>
        </div>
      </div>

      <!-- Center: 4-Stage Minimal Progress Stepper -->
      <nav class="hidden md:flex items-center space-x-1 font-mono text-xs">
        <button id="step-btn-1" onclick="setPipelineStep(1)" class="px-3 py-1.5 rounded bg-teal-500 text-navy-950 font-bold flex items-center space-x-1.5 transition">
          <span class="w-4 h-4 rounded-full bg-navy-950 text-teal-400 text-[10px] flex items-center justify-center font-bold">1</span>
          <span id="step-txt-1">Spot Spill</span>
        </button>
        <span class="text-gray-600">→</span>
        <button id="step-btn-2" onclick="setPipelineStep(2)" class="px-3 py-1.5 rounded text-gray-400 hover:text-white flex items-center space-x-1.5 transition">
          <span class="w-4 h-4 rounded-full bg-slate-800 text-gray-400 text-[10px] flex items-center justify-center font-bold">2</span>
          <span id="step-txt-2">Trace Drift</span>
        </button>
        <span class="text-gray-600">→</span>
        <button id="step-btn-3" onclick="setPipelineStep(3)" class="px-3 py-1.5 rounded text-gray-400 hover:text-white flex items-center space-x-1.5 transition">
          <span class="w-4 h-4 rounded-full bg-slate-800 text-gray-400 text-[10px] flex items-center justify-center font-bold">3</span>
          <span id="step-txt-3">Find Ship</span>
        </button>
        <span class="text-gray-600">→</span>
        <button id="step-btn-4" onclick="setPipelineStep(4)" class="px-3 py-1.5 rounded text-gray-400 hover:text-white flex items-center space-x-1.5 transition">
          <span class="w-4 h-4 rounded-full bg-slate-800 text-gray-400 text-[10px] flex items-center justify-center font-bold">4</span>
          <span id="step-txt-4">Evidence Report</span>
        </button>
      </nav>

      <!-- Right: Simple / Technical Switch & Quick Guide -->
      <div class="flex items-center space-x-3">
        
        <!-- Global Dual-Mode Toggle Switch -->
        <div class="flex items-center bg-navy-900 border border-slate-700 rounded-lg p-0.5 text-xs font-mono">
          <button id="btn-mode-simple" onclick="setExplanationMode('simple')" class="px-2.5 py-1 rounded bg-teal-500 text-navy-950 font-bold transition flex items-center space-x-1">
            <span>Simple</span>
          </button>
          <button id="btn-mode-technical" onclick="setExplanationMode('technical')" class="px-2.5 py-1 rounded text-gray-400 hover:text-white transition flex items-center space-x-1">
            <span>Technical</span>
          </button>
        </div>

        <!-- Quick Help Explainer Button -->
        <button onclick="document.getElementById('explainer-modal').classList.remove('hidden')" class="px-2.5 py-1.5 rounded-lg bg-slate-850 hover:bg-slate-800 border border-slate-700 text-xs text-gray-300 hover:text-white transition flex items-center space-x-1">
          <i data-lucide="help-circle" class="w-3.5 h-3.5 text-teal-400"></i>
          <span class="hidden sm:inline">What is this?</span>
        </button>

      </div>
    </div>
  </header>

  <!-- Mission Overview Plain-Language Banner -->
  <div id="intro-banner" class="bg-navy-900 border-b border-slate-800 px-4 py-2.5 text-xs">
    <div class="max-w-7xl mx-auto flex items-center justify-between">
      <div class="flex items-center space-x-2 text-gray-300">
        <span class="text-teal-400 font-bold">ADAPTIVE PIPELINE:</span>
        <span id="banner-text">Works on all satellite images. Automatically extracts embedded coordinates, detects in-image ship radar targets, and allows manual location input when metadata is absent.</span>
      </div>
      <button onclick="document.getElementById('intro-banner').classList.add('hidden')" class="text-gray-500 hover:text-gray-300 ml-4 font-mono text-[11px]">[close]</button>
    </div>
  </div>

  <!-- Main Grid Workspace -->
  <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-5 grid grid-cols-1 lg:grid-cols-12 gap-5">

    <!-- Left Sidebar: Upload & Scenarios (4 Cols) -->
    <div class="lg:col-span-4 space-y-4">
      
      <!-- Upload / Drag-and-Drop Area -->
      <div class="ops-panel rounded-lg p-4">
        <div class="flex items-center justify-between mb-2.5">
          <div class="flex items-center space-x-2">
            <i data-lucide="upload-cloud" class="w-4 h-4 text-teal-400"></i>
            <h2 class="font-heading font-semibold text-xs text-white uppercase tracking-wider">Upload / Drop Any Image</h2>
          </div>
          <span class="text-[10px] font-mono text-teal-400 bg-teal-950 px-1.5 py-0.5 rounded border border-teal-500/30">GEOTIFF / PNG / JPG</span>
        </div>

        <!-- Interactive Drag & Drop Box -->
        <div id="drop-zone" onclick="document.getElementById('file-input').click()" class="border-2 border-dashed border-slate-700 hover:border-teal-400 bg-navy-950/70 hover:bg-slate-850 p-4 rounded-lg text-center cursor-pointer transition flex flex-col items-center justify-center space-y-1.5 group">
          <i data-lucide="cloud-upload" class="w-7 h-7 text-slate-500 group-hover:text-teal-400 transition"></i>
          <div class="text-xs font-semibold text-gray-300 group-hover:text-white">
            Drag & Drop Any SAR Image Here, or <span class="text-teal-400 underline">Browse</span>
          </div>
          <div class="text-[10px] text-gray-500 font-mono">Supports GeoTIFF (.tif), PNG, JPG</div>
        </div>
        <input type="file" id="file-input" class="hidden" accept=".tif,.tiff,.png,.jpg,.jpeg" onchange="handleFileInputChange(event)">

        <!-- Custom Upload Active Indicator -->
        <div id="custom-file-indicator" class="hidden mt-2 p-2 rounded bg-teal-950 border border-teal-500/30 text-xs font-mono flex items-center justify-between">
          <div class="flex items-center space-x-1.5 overflow-hidden">
            <i data-lucide="file" class="w-3.5 h-3.5 text-teal-400 flex-shrink-0"></i>
            <span id="custom-file-name" class="text-teal-300 truncate">uploaded_image.png</span>
          </div>
          <button onclick="clearCustomUpload()" class="text-gray-400 hover:text-white text-[11px] ml-2 font-bold">×</button>
        </div>
      </div>

      <!-- Test Case Dataset Selector -->
      <div class="ops-panel rounded-lg p-4">
        <div class="flex items-center justify-between mb-3">
          <div class="flex items-center space-x-2">
            <i data-lucide="database" class="w-4 h-4 text-teal-400"></i>
            <h2 id="cases-panel-title" class="font-heading font-semibold text-xs text-white uppercase tracking-wider">Benchmark Scenarios</h2>
          </div>
          <span id="cases-panel-tag" class="text-[10px] font-mono text-teal-400 bg-teal-950 px-1.5 py-0.5 rounded border border-teal-500/30">CALIBRATED SAR</span>
        </div>

        <div id="samples-list" class="space-y-2">
          <!-- Populated dynamically -->
        </div>
      </div>

      <!-- In-Image Detected Vessels Card (Phase 4) -->
      <div class="ops-panel rounded-lg p-4">
        <div class="flex items-center justify-between mb-2">
          <div class="flex items-center space-x-1.5">
            <i data-lucide="radar" class="w-4 h-4 text-teal-400"></i>
            <h2 id="vessel-targets-panel-title" class="font-heading font-semibold text-xs text-white uppercase tracking-wider">In-Image Ship Targets</h2>
          </div>
          <span id="vessel-targets-count-badge" class="text-[10px] font-mono text-teal-400 bg-teal-950 px-1.5 py-0.5 rounded border border-teal-500/30">0 DETECTED</span>
        </div>

        <div id="in-image-vessels-list" class="space-y-2 text-xs font-mono">
          <!-- Populated dynamically -->
        </div>
        <div class="text-[10px] text-gray-500 mt-2 text-center">
          Identifies bright radar point reflectors (ships) directly from image pixels.
        </div>
      </div>

      <!-- Prime Suspect Card -->
      <div id="suspect-panel-container" class="ops-panel rounded-lg p-4">
        <div class="flex items-center justify-between mb-2.5">
          <div class="flex items-center space-x-1.5">
            <i data-lucide="shield-alert" class="w-4 h-4 text-amber-400"></i>
            <h2 id="suspect-panel-title" class="font-heading font-semibold text-xs text-white uppercase tracking-wider">Prime Suspect Ship</h2>
          </div>
          <span id="suspect-mode-pill" class="text-[10px] font-mono text-amber-400 bg-amber-950 px-1.5 py-0.5 rounded border border-amber-500/30">ATTRIBUTION MATCH</span>
        </div>

        <div id="suspect-summary-box" class="ops-card p-3 rounded border border-amber-500/30 space-y-2">
          <div class="flex justify-between items-start">
            <div>
              <div id="suspect-vessel-name" class="font-bold text-white text-sm font-heading">MT AL-BARAKA</div>
              <div id="suspect-vessel-type" class="text-[11px] text-gray-400">Crude Oil Tanker (Panama Flag)</div>
            </div>
            <div id="suspect-score-badge" class="px-2 py-0.5 rounded bg-crimson-950 text-crimson-400 font-bold border border-crimson-500/40 text-xs font-mono">99.4% SUSPICIOUS</div>
          </div>
          
          <div id="suspect-explanation" class="text-[11px] text-amber-300 leading-snug border-t border-slate-800 pt-1.5">
            Passed directly through the spill origin 18 hours ago and turned off its position tracker for 42 minutes.
          </div>
        </div>
      </div>

    </div>

    <!-- Right Area: 4 Interactive Workflow Stages (8 Cols) -->
    <div class="lg:col-span-8 space-y-4">
      
      <!-- STAGE 1: Spot Spill (SAR Satellite Detection) -->
      <div id="view-step-1" class="ops-panel rounded-lg p-4 relative">
        
        <!-- Header & View Toggles -->
        <div class="flex flex-wrap items-center justify-between gap-2 mb-2.5 pb-2 border-b border-slate-800 font-mono text-xs">
          <div class="flex items-center space-x-2">
            <span class="w-2 h-2 rounded-full bg-teal-400"></span>
            <span id="sar-stage-title" class="font-heading font-semibold text-sm text-white">Satellite Radar Spill Detection</span>
          </div>

          <div class="flex items-center space-x-3">
            <div class="flex items-center space-x-1.5 bg-slate-850 px-2 py-1 rounded border border-slate-700 text-[11px]">
              <span class="text-gray-400">Overlay:</span>
              <input type="range" id="opacity-slider" min="0" max="100" value="85" class="w-16 accent-amber-500 cursor-pointer" oninput="updateOverlayOpacity(this.value)">
              <span id="opacity-val" class="text-amber-400 font-mono w-6">85%</span>
            </div>

            <div class="flex rounded border border-slate-700 bg-slate-850 p-0.5 text-[11px]">
              <button id="btn-view-composite" onclick="setViewMode('composite')" class="px-2 py-0.5 rounded bg-teal-500 text-navy-950 font-bold">Composite</button>
              <button id="btn-view-raw" onclick="setViewMode('raw')" class="px-2 py-0.5 rounded text-gray-400 hover:text-white">Raw Radar</button>
              <button id="btn-view-mask" onclick="setViewMode('mask')" class="px-2 py-0.5 rounded text-gray-400 hover:text-white">Spill Mask</button>
            </div>
          </div>
        </div>

        <!-- Phase 1: Metadata Pre-Flight Status Strip -->
        <div id="metadata-status-strip" class="mb-3 p-2 rounded bg-navy-950 border border-slate-800 flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
          <div class="flex items-center space-x-3">
            <div class="flex items-center space-x-1.5">
              <span class="text-gray-400 text-[11px]">Location Data:</span>
              <span id="meta-loc-badge" class="px-2 py-0.5 rounded text-[10px] font-bold bg-teal-950 text-teal-400 border border-teal-500/30">Found (GeoTIFF)</span>
            </div>
            <div class="flex items-center space-x-1.5">
              <span class="text-gray-400 text-[11px]">Timestamp:</span>
              <span id="meta-time-badge" class="px-2 py-0.5 rounded text-[10px] font-bold bg-teal-950 text-teal-400 border border-teal-500/30">Available</span>
            </div>
          </div>
          <div id="meta-details-text" class="text-[10px] text-gray-400 truncate max-w-xs">
            EPSG:4326 WGS84 Georeferenced
          </div>
        </div>

        <!-- Radar Frame -->
        <div id="radar-canvas-container" class="relative w-full aspect-square max-h-[400px] rounded bg-black border border-slate-800 flex items-center justify-center overflow-hidden transition">
          <div class="absolute inset-0 scan-line pointer-events-none z-20"></div>
          <img id="sar-base-img" src="" alt="Satellite SAR Base" class="w-full h-full object-contain select-none">
          <img id="sar-overlay-img" src="" alt="Spill Mask Overlay" class="absolute inset-0 w-full h-full object-contain pointer-events-none select-none z-10" style="opacity: 0.85;">
          
          <div id="loading-spinner" class="hidden absolute inset-0 bg-navy-950/90 flex flex-col items-center justify-center z-40">
            <div class="w-10 h-10 border-2 border-teal-500/30 border-t-teal-400 rounded-full animate-spin mb-2"></div>
            <span id="loading-text" class="text-xs font-mono text-teal-300">ANALYZING SATELLITE RADAR REFLECTIONS...</span>
          </div>
        </div>

        <!-- Stage 1 Stats Ribbon -->
        <div class="mt-3 grid grid-cols-2 md:grid-cols-4 gap-2 text-xs font-mono">
          <div class="ops-card p-2.5 rounded">
            <div id="lbl-stat-area" class="text-[10px] text-gray-400">SPILL SIZE</div>
            <div id="stat-area" class="text-sm font-bold text-teal-300 font-heading">12.3 km²</div>
            <div id="stat-area-sub" class="text-[10px] text-gray-500">~1,700 soccer fields</div>
          </div>
          <div class="ops-card p-2.5 rounded">
            <div id="lbl-stat-shape" class="text-[10px] text-gray-400">SHAPE PATTERN</div>
            <div id="stat-elong" class="text-sm font-bold text-amber-400 font-heading">Elongated Trail</div>
            <div id="stat-elong-sub" class="text-[10px] text-amber-500/80">Ship wake pattern</div>
          </div>
          <div class="ops-card p-2.5 rounded">
            <div id="lbl-stat-age" class="text-[10px] text-gray-400">SPILL AGE</div>
            <div id="stat-age" class="text-sm font-bold text-white font-heading">Fresh (&lt;6 hrs)</div>
            <div id="stat-age-sub" class="text-[10px] text-teal-400/80">Sharp boundary edges</div>
          </div>
          <div class="ops-card p-2.5 rounded">
            <div id="lbl-stat-conf" class="text-[10px] text-gray-400">CERTAINTY</div>
            <div id="stat-conf" class="text-sm font-bold text-radar font-heading">Very High (99%)</div>
            <div id="stat-conf-sub" class="text-[10px] text-emerald-400/70">Clear radar dampening</div>
          </div>
        </div>

        <!-- Phase 2: Graceful Stop Banner (Shown when location metadata is missing) -->
        <div id="missing-metadata-banner" class="hidden mt-3 p-3.5 rounded bg-amber-950/40 border border-amber-500/40 text-xs">
          <div class="flex items-start space-x-2.5">
            <i data-lucide="info" class="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5"></i>
            <div class="flex-1 space-y-1.5">
              <div class="font-bold text-amber-300">Detection complete. Drift tracing and vessel attribution require location and time data.</div>
              <p class="text-gray-300 text-[11px] leading-relaxed">
                This image does not contain embedded GPS coordinates or timestamps. The oil slick and in-image ship targets have been successfully detected above. To calculate ocean drift and correlate AIS ship traffic, enter the incident location below.
              </p>
              <div class="pt-1 flex flex-wrap gap-2">
                <button onclick="openManualOverrideModal()" class="px-3 py-1.5 rounded bg-amber-500 hover:bg-amber-400 text-navy-950 font-bold font-mono text-xs flex items-center space-x-1.5 transition">
                  <i data-lucide="map-pin" class="w-3.5 h-3.5"></i>
                  <span>Enter Location & Time to Continue</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Next / Proceed Button (When metadata is present) -->
        <div id="proceed-step-1-container" class="mt-3 flex justify-end">
          <button onclick="setPipelineStep(2)" class="px-4 py-1.5 rounded bg-teal-500 hover:bg-teal-400 text-navy-950 font-bold text-xs flex items-center space-x-1 transition">
            <span>Proceed to Drift Tracing</span>
            <i data-lucide="arrow-right" class="w-3.5 h-3.5"></i>
          </button>
        </div>
      </div>

      <!-- STAGE 2: Trace Drift (Ocean Current & Wind Hindcast) -->
      <div id="view-step-2" class="hidden ops-panel rounded-lg p-4 relative">
        <div class="flex flex-wrap items-center justify-between gap-2 mb-3 pb-2.5 border-b border-slate-800 font-mono text-xs">
          <div class="flex items-center space-x-2">
            <span class="w-2 h-2 rounded-full bg-amber-400 radar-pulse"></span>
            <span id="drift-stage-title" class="font-heading font-semibold text-sm text-white">Rewind Ocean Drift to Find Origin</span>
          </div>
          <div class="flex items-center space-x-2">
            <button onclick="openManualOverrideModal()" class="px-2.5 py-1 rounded bg-slate-850 hover:bg-slate-800 border border-slate-700 text-amber-300 text-xs flex items-center space-x-1">
              <i data-lucide="edit-3" class="w-3 h-3"></i>
              <span>Edit Coordinates</span>
            </button>
            <button onclick="playDriftAnimation()" class="px-3 py-1 rounded bg-amber-500 hover:bg-amber-400 text-navy-950 font-bold flex items-center space-x-1 text-xs">
              <i data-lucide="play" class="w-3 h-3"></i>
              <span>Animate Rewind Path</span>
            </button>
          </div>
        </div>

        <!-- Map Frame -->
        <div id="drift-map" class="w-full h-[400px] rounded border border-slate-800 z-10"></div>

        <!-- Stage 2 Stats Ribbon -->
        <div class="mt-3 grid grid-cols-1 md:grid-cols-3 gap-2 text-xs font-mono">
          <div class="ops-card p-2.5 rounded">
            <div id="lbl-drift-origin" class="text-[10px] text-gray-400">ESTIMATED START LOCATION</div>
            <div id="origin-gps-val" class="text-sm font-bold text-amber-400">25.8154°N, 54.8252°E</div>
            <div id="origin-gps-sub" class="text-[10px] text-gray-500">Strait of Hormuz shipping lane</div>
          </div>
          <div class="ops-card p-2.5 rounded">
            <div id="lbl-drift-time" class="text-[10px] text-gray-400">TIME OF DISCHARGE</div>
            <div id="origin-time-val" class="text-sm font-bold text-teal-300">~18 Hours Ago</div>
            <div id="origin-time-sub" class="text-[10px] text-gray-500">Drifted 48.4 km across sea</div>
          </div>
          <div class="ops-card p-2.5 rounded">
            <div id="lbl-drift-uncert" class="text-[10px] text-gray-400">SEARCH RADIUS ACCURACY</div>
            <div id="origin-uncert-val" class="text-sm font-bold text-white">Within ± 5.0 km</div>
            <div id="origin-uncert-sub" class="text-[10px] text-gray-500">95% confidence zone</div>
          </div>
        </div>

        <!-- Next Button -->
        <div class="mt-3 flex justify-between">
          <button onclick="setPipelineStep(1)" class="px-3 py-1.5 rounded bg-slate-850 hover:bg-slate-800 text-gray-300 font-mono text-xs">
            ← Back to Detection
          </button>
          <button onclick="setPipelineStep(3)" class="px-4 py-1.5 rounded bg-teal-500 hover:bg-teal-400 text-navy-950 font-bold text-xs flex items-center space-x-1 transition">
            <span>Proceed to Vessel Attribution</span>
            <i data-lucide="arrow-right" class="w-3.5 h-3.5"></i>
          </button>
        </div>
      </div>

      <!-- STAGE 3: Find Ship (AIS Candidate Attribution) -->
      <div id="view-step-3" class="hidden ops-panel rounded-lg p-4 relative">
        <div class="flex items-center justify-between mb-3 pb-2.5 border-b border-slate-800 font-mono text-xs">
          <div class="flex items-center space-x-2">
            <i data-lucide="crosshair" class="w-4 h-4 text-crimson-400"></i>
            <span id="ais-stage-title" class="font-heading font-semibold text-sm text-white">Vessels in Area During Spill Window</span>
          </div>
          <span id="ais-source-tag" class="text-gray-400">Marine AIS Transponder Records</span>
        </div>

        <div id="ais-vessels-list" class="space-y-2.5 font-mono">
          <!-- Dynamic vessel cards -->
        </div>

        <!-- Next Button -->
        <div class="mt-4 flex justify-between">
          <button onclick="setPipelineStep(2)" class="px-3 py-1.5 rounded bg-slate-850 hover:bg-slate-800 text-gray-300 font-mono text-xs">
            ← Back to Drift Map
          </button>
          <button onclick="setPipelineStep(4)" class="px-4 py-1.5 rounded bg-teal-500 hover:bg-teal-400 text-navy-950 font-bold text-xs flex items-center space-x-1 transition">
            <span>Generate Executive Evidence Report</span>
            <i data-lucide="arrow-right" class="w-3.5 h-3.5"></i>
          </button>
        </div>
      </div>

      <!-- STAGE 4: Evidence Report (Incident Dossier for Judges) -->
      <div id="view-step-4" class="hidden ops-panel rounded-lg p-5 border border-teal-500/40 relative">
        
        <!-- Report Header -->
        <div class="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800 mb-4 font-mono">
          <div>
            <div class="flex items-center space-x-2">
              <span class="px-2 py-0.5 rounded bg-crimson-950 text-crimson-400 font-bold text-[11px] border border-crimson-500/40">OFFICIAL EVIDENCE DOSSIER</span>
              <span id="rep-source-badge" class="px-2 py-0.5 rounded bg-teal-950 text-teal-300 text-[10px] border border-teal-500/30">Location: Auto-Extracted GeoTIFF</span>
            </div>
            <h2 id="rep-main-title" class="font-heading font-bold text-lg text-white mt-1">Maritime Oil Pollution Attribution Report</h2>
          </div>
          <button onclick="window.print()" class="px-3 py-1.5 rounded bg-teal-500 text-navy-950 font-bold text-xs flex items-center space-x-1.5 transition">
            <i data-lucide="printer" class="w-3.5 h-3.5"></i>
            <span>Export / Print Report</span>
          </button>
        </div>

        <!-- 3-Pillar Evidence Grid -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-3 font-mono text-xs mb-4">
          
          <!-- Evidence 1 -->
          <div class="ops-card p-3 rounded space-y-2">
            <div class="text-teal-400 font-bold flex items-center space-x-1.5 pb-1.5 border-b border-slate-800">
              <i data-lucide="satellite" class="w-3.5 h-3.5"></i>
              <span id="rep-col-1-title">1. SATELLITE DETECTION</span>
            </div>
            <div class="space-y-1 text-gray-300 text-[11px]">
              <div>Sensor: <strong class="text-white">Sentinel-1 C-Band SAR</strong></div>
              <div>Spill Size: <strong class="text-teal-300" id="rep-area">12.3 km²</strong></div>
              <div>Shape: <strong class="text-white" id="rep-elong">Elongated Ribbon</strong></div>
              <div>Estimated Age: <strong class="text-amber-400" id="rep-age">Fresh (&lt;6h)</strong></div>
              <div>Spill Centroid: <strong class="text-white" id="rep-det-gps">25.820°N, 55.120°E</strong></div>
            </div>
          </div>

          <!-- Evidence 2 -->
          <div class="ops-card p-3 rounded space-y-2">
            <div class="text-amber-400 font-bold flex items-center space-x-1.5 pb-1.5 border-b border-slate-800">
              <i data-lucide="wind" class="w-3.5 h-3.5"></i>
              <span id="rep-col-2-title">2. OCEAN DRIFT ORIGIN</span>
            </div>
            <div class="space-y-1 text-gray-300 text-[11px]">
              <div>Model: <strong class="text-white">Ocean Current + Wind Leeway</strong></div>
              <div>Origin Spot: <strong class="text-amber-300" id="rep-orig-gps">25.815°N, 54.825°E</strong></div>
              <div>Drift Distance: <strong class="text-white" id="rep-drift-dist">48.4 km</strong></div>
              <div>Discharge Time: <strong class="text-white" id="rep-window">~18 Hours Ago</strong></div>
              <div>Search Radius: <strong class="text-white" id="rep-uncert">± 5.0 km</strong></div>
            </div>
          </div>

          <!-- Evidence 3 -->
          <div class="ops-card p-3 rounded space-y-2 border border-crimson-500/40 bg-crimson-950/20">
            <div class="text-crimson-400 font-bold flex items-center space-x-1.5 pb-1.5 border-b border-slate-800">
              <i data-lucide="shield-alert" class="w-3.5 h-3.5"></i>
              <span id="rep-col-3-title">3. PRIME SUSPECT VESSEL</span>
            </div>
            <div class="space-y-1 text-gray-300 text-[11px]">
              <div>Vessel: <strong class="text-white text-xs" id="rep-vessel-name">MT AL-BARAKA</strong></div>
              <div>Type: <strong class="text-white" id="rep-vessel-flag">Crude Tanker (Panama)</strong></div>
              <div>MMSI: <strong class="text-white" id="rep-mmsi">477123456</strong></div>
              <div>Distance to Origin: <strong class="text-crimson-400" id="rep-cpa">1.2 km (Direct Intercept)</strong></div>
              <div>Suspicion Score: <strong class="text-crimson-400 text-xs font-bold" id="rep-score">99.4% Critical</strong></div>
            </div>
          </div>
        </div>

        <!-- Plain English Executive Summary Banner -->
        <div class="ops-card p-3.5 rounded border border-slate-700 text-xs">
          <div class="text-teal-400 font-bold mb-1.5 flex items-center space-x-1">
            <i data-lucide="file-text" class="w-3.5 h-3.5"></i>
            <span>EXECUTIVE SUMMARY FOR DECISION MAKERS:</span>
          </div>
          <p id="rep-plain-summary" class="text-gray-300 leading-relaxed font-sans text-xs">
            Satellite radar confirmed a 12.3 km² crude oil spill in the Persian Gulf. Hydrodynamic drift modeling traced the spill 48.4 km backwards in time to an origin point 18 hours prior. Vessel tracker analysis identified VLCC tanker MT AL-BARAKA as the primary suspect (99.4% suspicion) after it passed within 1.2 km of the origin while switching off its transponder for 42 minutes and abruptly reducing speed.
          </p>
        </div>

        <!-- Red Flag Breakdown -->
        <div class="mt-3 ops-card p-3 rounded font-mono text-xs">
          <div class="text-amber-400 font-bold mb-1.5 flex items-center space-x-1.5">
            <i data-lucide="alert-triangle" class="w-3.5 h-3.5"></i>
            <span>Recorded Red Flags & Anomalies:</span>
          </div>
          <ul id="rep-anomalies-list" class="space-y-1 text-gray-300 list-disc pl-4 text-[11px]">
            <!-- Dynamic items -->
          </ul>
        </div>

        <!-- Navigation -->
        <div class="mt-4 flex justify-between">
          <button onclick="setPipelineStep(3)" class="px-3 py-1.5 rounded bg-slate-850 hover:bg-slate-800 text-gray-300 font-mono text-xs">
            ← Back to Vessel List
          </button>
          <button onclick="setPipelineStep(1)" class="px-3 py-1.5 rounded bg-slate-850 hover:bg-slate-800 text-teal-300 font-mono text-xs">
            ↺ Restart / Inspect Another Case
          </button>
        </div>

      </div>

    </div>
  </main>

  <!-- Phase 3: Manual Location/Time Override Modal -->
  <div id="manual-override-modal" class="hidden fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
    <div class="ops-panel max-w-lg w-full rounded-lg p-5 border border-amber-500/50 shadow-2xl space-y-4">
      <div class="flex items-center justify-between pb-3 border-b border-slate-800 font-mono">
        <h3 class="font-heading font-bold text-sm text-white flex items-center space-x-2">
          <i data-lucide="map-pin" class="w-4 h-4 text-amber-400"></i>
          <span>Provide Incident Location & Time</span>
        </h3>
        <button onclick="document.getElementById('manual-override-modal').classList.add('hidden')" class="text-gray-400 hover:text-white"><i data-lucide="x" class="w-4 h-4"></i></button>
      </div>

      <p class="text-xs text-gray-300">
        Enter coordinates and estimated capture time to enable ocean drift modeling and AIS ship correlation for this image.
      </p>

      <!-- Quick Location Presets -->
      <div class="space-y-1 text-[11px] font-mono">
        <div class="text-gray-400">Quick Regional Presets:</div>
        <div class="flex flex-wrap gap-1.5">
          <button onclick="setManualCoords(25.82, 55.12, 'Persian Gulf')" class="px-2 py-1 rounded bg-slate-850 hover:bg-slate-800 text-teal-300 border border-slate-700 text-[10px]">Persian Gulf (25.82°N, 55.12°E)</button>
          <button onclick="setManualCoords(3.15, 100.80, 'Malacca Strait')" class="px-2 py-1 rounded bg-slate-850 hover:bg-slate-800 text-teal-300 border border-slate-700 text-[10px]">Malacca Strait (3.15°N, 100.80°E)</button>
          <button onclick="setManualCoords(19.42, 71.33, 'Mumbai High')" class="px-2 py-1 rounded bg-slate-850 hover:bg-slate-800 text-teal-300 border border-slate-700 text-[10px]">Mumbai High (19.42°N, 71.33°E)</button>
        </div>
      </div>

      <div class="grid grid-cols-2 gap-3 font-mono text-xs">
        <div>
          <label class="block text-gray-400 text-[11px] mb-1">Latitude (°N):</label>
          <input type="number" id="input-manual-lat" step="0.0001" min="-90" max="90" value="25.8200" class="w-full bg-navy-950 border border-slate-700 rounded p-2 text-white text-xs focus:border-amber-400 outline-none">
        </div>
        <div>
          <label class="block text-gray-400 text-[11px] mb-1">Longitude (°E):</label>
          <input type="number" id="input-manual-lon" step="0.0001" min="-180" max="180" value="55.1200" class="w-full bg-navy-950 border border-slate-700 rounded p-2 text-white text-xs focus:border-amber-400 outline-none">
        </div>
      </div>

      <div class="font-mono text-xs">
        <label class="block text-gray-400 text-[11px] mb-1">Capture Date & Time (UTC):</label>
        <input type="datetime-local" id="input-manual-time" class="w-full bg-navy-950 border border-slate-700 rounded p-2 text-white text-xs focus:border-amber-400 outline-none">
      </div>

      <div class="pt-2 flex justify-end space-x-2 font-mono text-xs">
        <button onclick="document.getElementById('manual-override-modal').classList.add('hidden')" class="px-3 py-1.5 rounded bg-slate-850 hover:bg-slate-800 text-gray-400">Cancel</button>
        <button onclick="submitManualOverride()" class="px-4 py-1.5 rounded bg-amber-500 hover:bg-amber-400 text-navy-950 font-bold flex items-center space-x-1">
          <i data-lucide="check" class="w-3.5 h-3.5"></i>
          <span>Apply & Run Drift Analysis</span>
        </button>
      </div>
    </div>
  </div>

  <!-- Explainer Modal ("What is this?") -->
  <div id="explainer-modal" class="hidden fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
    <div class="ops-panel max-w-xl w-full rounded-lg p-5 border border-slate-700 shadow-2xl">
      <div class="flex items-center justify-between pb-3 border-b border-slate-800 mb-3 font-mono">
        <h3 class="font-heading font-bold text-sm text-white flex items-center space-x-2">
          <i data-lucide="info" class="w-4 h-4 text-teal-400"></i>
          <span>What Does SpillTrace Do?</span>
        </h3>
        <button onclick="document.getElementById('explainer-modal').classList.add('hidden')" class="text-gray-400 hover:text-white"><i data-lucide="x" class="w-4 h-4"></i></button>
      </div>
      
      <div class="space-y-3 text-xs text-gray-300 leading-relaxed font-sans">
        <p class="font-semibold text-white">
          SpillTrace is an autonomous maritime intelligence tool designed for Smart India Hackathon (NTRO PS 26143) to detect illegal oil dumping at sea and hold polluters accountable.
        </p>
        
        <div class="grid grid-cols-1 gap-2 pt-1 font-mono text-[11px]">
          <div class="p-2 bg-navy-950 rounded border border-slate-800">
            <strong class="text-teal-400">1. Spots Oil Slicks & Ships:</strong> Scans satellite radar imagery to segment dampening dark patches and bright ship reflectors.
          </div>
          <div class="p-2 bg-navy-950 rounded border border-slate-800">
            <strong class="text-amber-400">2. Traces Drift Backward:</strong> Uses real wind and ocean current vectors to rewind the spill's trajectory to where and when it started.
          </div>
          <div class="p-2 bg-navy-950 rounded border border-slate-800">
            <strong class="text-crimson-400">3. Identifies the Vessel:</strong> Correlates ship transponder (AIS) records to find who was at the origin point and flags suspicious behavior.
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Client Script for Adaptive Pipeline -->
  <script>
    let explanationMode = 'simple'; // 'simple' or 'technical'
    let currentSampleId = 'sample_1_tanker_slick';
    let currentDetectionData = null;
    let currentDriftData = null;
    let currentAISData = null;
    let isManualLocation = false;
    let currentStep = 1;
    let leafletMap = null;
    let mapLayers = [];

    async function init() {
      lucide.createIcons();
      initMap();
      setupDragAndDrop();
      setDefaultDateTime();
      await loadSamplesList();
      await executeFullPipeline(currentSampleId);
    }

    function setDefaultDateTime() {
      const now = new Date();
      now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
      document.getElementById('input-manual-time').value = now.toISOString().slice(0, 16);
    }

    function setupDragAndDrop() {
      const dropZone = document.getElementById('drop-zone');
      const radarContainer = document.getElementById('radar-canvas-container');

      [dropZone, radarContainer].forEach(el => {
        if (!el) return;
        ['dragenter', 'dragover'].forEach(eventName => {
          el.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            el.classList.add('drag-active');
          }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
          el.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            el.classList.remove('drag-active');
          }, false);
        });

        el.addEventListener('drop', (e) => {
          const dt = e.dataTransfer;
          const files = dt.files;
          if (files && files.length > 0) {
            handleFile(files[0]);
          }
        }, false);
      });
    }

    function handleFileInputChange(event) {
      const file = event.target.files[0];
      if (file) {
        handleFile(file);
      }
    }

    function handleFile(file) {
      document.getElementById('custom-file-indicator').classList.remove('hidden');
      document.getElementById('custom-file-name').innerText = file.name + ' (' + Math.round(file.size / 1024) + ' KB)';
      
      const titleEl = document.getElementById('sar-stage-title');
      if (titleEl) {
        titleEl.innerText = 'Custom Upload: ' + file.name;
      }
      
      currentSampleId = null;
      isManualLocation = false;
      loadSamplesList();
      executeFullPipeline(null, file);
    }

    function clearCustomUpload() {
      document.getElementById('custom-file-indicator').classList.add('hidden');
      document.getElementById('file-input').value = '';
      isManualLocation = false;
      selectSample('sample_1_tanker_slick');
    }

    function openManualOverrideModal() {
      if (currentDetectionData && currentDetectionData.primary_slick && currentDetectionData.primary_slick.centroid_lat) {
        document.getElementById('input-manual-lat').value = currentDetectionData.primary_slick.centroid_lat.toFixed(4);
        document.getElementById('input-manual-lon').value = currentDetectionData.primary_slick.centroid_lon.toFixed(4);
      }
      document.getElementById('manual-override-modal').classList.remove('hidden');
    }

    function setManualCoords(lat, lon, label) {
      document.getElementById('input-manual-lat').value = lat.toFixed(4);
      document.getElementById('input-manual-lon').value = lon.toFixed(4);
    }

    async function submitManualOverride() {
      const lat = parseFloat(document.getElementById('input-manual-lat').value);
      const lon = parseFloat(document.getElementById('input-manual-lon').value);
      const timeVal = document.getElementById('input-manual-time').value;

      if (isNaN(lat) || lat < -90 || lat > 90) {
        alert('Please enter a valid latitude between -90 and 90.');
        return;
      }
      if (isNaN(lon) || lon < -180 || lon > 180) {
        alert('Please enter a valid longitude between -180 and 180.');
        return;
      }

      document.getElementById('manual-override-modal').classList.add('hidden');
      isManualLocation = true;

      // Update metadata status badges to show manual entry
      const locBadge = document.getElementById('meta-loc-badge');
      locBadge.innerText = 'Provided (Manual Input)';
      locBadge.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-amber-950 text-amber-300 border border-amber-500/30';
      document.getElementById('meta-details-text').innerText = `Manual GPS Fix: ${lat.toFixed(4)}°N, ${lon.toFixed(4)}°E`;

      document.getElementById('missing-metadata-banner').classList.add('hidden');
      document.getElementById('proceed-step-1-container').classList.remove('hidden');

      // Run Phase 2 & 3
      const spinner = document.getElementById('loading-spinner');
      spinner.classList.remove('hidden');

      try {
        const driftRes = await fetch('/api/drift', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ lat, lon, timestamp: timeVal, hours_back: 24, is_manual_override: true })
        });
        const driftData = await driftRes.json();
        currentDriftData = driftData;
        renderDriftMap(driftData);

        const origLat = driftData.estimated_origin.lat;
        const origLon = driftData.estimated_origin.lon;
        const aisRes = await fetch('/api/ais/correlate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            origin_lat: origLat,
            origin_lon: origLon,
            origin_time_iso: timeVal,
            in_image_vessels: currentDetectionData ? currentDetectionData.in_image_vessels : [],
            is_manual_override: true
          })
        });
        const aisData = await aisRes.json();
        currentAISData = aisData;
        renderAISResults(aisData);

        populateExecutiveReport(currentDetectionData, driftData, aisData);
        setPipelineStep(2);
      } catch (err) {
        console.error('Manual override pipeline error:', err);
      } finally {
        spinner.classList.add('hidden');
      }
    }

    function initMap() {
      leafletMap = L.map('drift-map', {
        zoomControl: true,
        attributionControl: false
      }).setView([25.82, 55.12], 9);

      L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        maxZoom: 19
      }).addTo(leafletMap);
    }

    function setExplanationMode(mode) {
      explanationMode = mode;
      const simpleBtn = document.getElementById('btn-mode-simple');
      const techBtn = document.getElementById('btn-mode-technical');
      const badge = document.getElementById('header-mode-badge');

      if (mode === 'simple') {
        simpleBtn.className = 'px-2.5 py-1 rounded bg-teal-500 text-navy-950 font-bold transition flex items-center space-x-1';
        techBtn.className = 'px-2.5 py-1 rounded text-gray-400 hover:text-white transition flex items-center space-x-1';
        badge.innerText = 'SIMPLE MODE';
        badge.className = 'px-2 py-0.5 text-[10px] font-mono rounded bg-teal-950 text-teal-400 border border-teal-500/30';
      } else {
        simpleBtn.className = 'px-2.5 py-1 rounded text-gray-400 hover:text-white transition flex items-center space-x-1';
        techBtn.className = 'px-2.5 py-1 rounded bg-amber-500 text-navy-950 font-bold transition flex items-center space-x-1';
        badge.innerText = 'TECHNICAL MODE';
        badge.className = 'px-2 py-0.5 text-[10px] font-mono rounded bg-amber-950 text-amber-400 border border-amber-500/30';
      }

      updateTextForCurrentMode();
    }

    function updateTextForCurrentMode() {
      const isSimple = explanationMode === 'simple';

      document.getElementById('step-txt-1').innerText = isSimple ? 'Spot Spill' : 'SAR Detect';
      document.getElementById('step-txt-2').innerText = isSimple ? 'Trace Drift' : 'Drift Hindcast';
      document.getElementById('step-txt-3').innerText = isSimple ? 'Find Ship' : 'AIS Attribution';
      document.getElementById('step-txt-4').innerText = isSimple ? 'Evidence Report' : 'Incident Dossier';

      document.getElementById('lbl-stat-area').innerText = isSimple ? 'SPILL SIZE' : 'CALIBRATED AREA';
      document.getElementById('lbl-stat-shape').innerText = isSimple ? 'SHAPE PATTERN' : 'ELONGATION RATIO';
      document.getElementById('lbl-stat-age').innerText = isSimple ? 'SPILL AGE' : 'AGE BRACKET';
      document.getElementById('lbl-stat-conf').innerText = isSimple ? 'CERTAINTY' : 'CONFIDENCE SCORE';

      document.getElementById('lbl-drift-origin').innerText = isSimple ? 'ESTIMATED START LOCATION' : 'LAGRANGIAN ORIGIN FIX';
      document.getElementById('lbl-drift-time').innerText = isSimple ? 'TIME OF DISCHARGE' : 'HINDCAST TEMPORAL WINDOW';
      document.getElementById('lbl-drift-uncert').innerText = isSimple ? 'SEARCH RADIUS ACCURACY' : 'DIFFUSIVE UNCERTAINTY (95%)';

      if (currentDetectionData) renderDetectionResults(currentDetectionData);
      if (currentDriftData) renderDriftMap(currentDriftData);
      if (currentAISData) renderAISResults(currentAISData);
      if (currentDetectionData && currentDriftData && currentAISData) {
        populateExecutiveReport(currentDetectionData, currentDriftData, currentAISData);
      }
    }

    function setPipelineStep(step) {
      currentStep = step;
      for (let s = 1; s <= 4; s++) {
        const btn = document.getElementById(`step-btn-${s}`);
        const view = document.getElementById(`view-step-${s}`);
        if (s === step) {
          btn.className = 'px-3 py-1.5 rounded bg-teal-500 text-navy-950 font-bold flex items-center space-x-1.5 transition';
          view.classList.remove('hidden');
        } else {
          btn.className = 'px-3 py-1.5 rounded text-gray-400 hover:text-white flex items-center space-x-1.5 transition';
          view.classList.add('hidden');
        }
      }

      if (step === 2) {
        setTimeout(() => {
          leafletMap.invalidateSize();
          if (currentDriftData) renderDriftMap(currentDriftData);
        }, 100);
      }
    }

    async function loadSamplesList() {
      try {
        const res = await fetch('/api/samples');
        const data = await res.json();
        const container = document.getElementById('samples-list');
        container.innerHTML = '';

        data.samples.forEach(s => {
          const isSelected = s.id === currentSampleId;
          const item = document.createElement('div');
          item.className = `p-2.5 rounded border transition cursor-pointer ${
            isSelected 
              ? 'border-teal-400 bg-slate-850' 
              : 'border-slate-800 bg-navy-950 hover:border-slate-700'
          }`;
          item.onclick = () => selectSample(s.id);

          const title = explanationMode === 'simple' ? s.simple_title : s.title;
          const desc = explanationMode === 'simple' ? s.simple_desc : s.description;

          item.innerHTML = `
            <div class="flex justify-between items-start">
              <span class="text-xs font-semibold ${isSelected ? 'text-teal-300' : 'text-gray-200'}">${title}</span>
              ${s.has_ground_truth 
                ? '<span class="text-[9px] font-mono px-1.5 py-0.5 rounded bg-amber-950 text-amber-400 border border-amber-500/40">GT MASK</span>'
                : '<span class="text-[9px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-gray-400">LOOK-ALIKE</span>'
              }
            </div>
            <div class="text-[10px] text-gray-400 mt-0.5 leading-snug">${desc}</div>
          `;
          container.appendChild(item);
        });
      } catch (err) {
        console.error('Failed to load samples:', err);
      }
    }

    async function selectSample(sampleId) {
      currentSampleId = sampleId;
      isManualLocation = false;
      document.getElementById('custom-file-indicator').classList.add('hidden');
      await loadSamplesList();
      await executeFullPipeline(sampleId);
    }

    async function executeFullPipeline(sampleId = null, file = null) {
      const spinner = document.getElementById('loading-spinner');
      spinner.classList.remove('hidden');

      try {
        const formData = new FormData();
        if (sampleId) formData.append('sample_id', sampleId);
        if (file) formData.append('file', file);

        const detectRes = await fetch('/api/detect', { method: 'POST', body: formData });
        const detectData = await detectRes.json();
        currentDetectionData = detectData;
        renderDetectionResults(detectData);
        renderInImageVessels(detectData.in_image_vessels || []);

        const meta = detectData.metadata_status || {};
        const hasLocation = meta.has_location || detectData.geo_metadata.has_location;

        // Update Metadata Pre-Flight Strip
        const locBadge = document.getElementById('meta-loc-badge');
        const timeBadge = document.getElementById('meta-time-badge');
        const detailsText = document.getElementById('meta-details-text');

        if (hasLocation) {
          locBadge.innerText = 'Found (Embedded)';
          locBadge.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-teal-950 text-teal-400 border border-teal-500/30';
          detailsText.innerText = detectData.geo_metadata.crs || 'WGS84 Coordinates Extracted';
        } else {
          locBadge.innerText = 'Not found in file';
          locBadge.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-amber-950 text-amber-400 border border-amber-500/30';
          detailsText.innerText = 'No GPS tags present. Location can be manually provided.';
        }

        if (meta.has_timestamp) {
          timeBadge.innerText = 'Found';
          timeBadge.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-teal-950 text-teal-400 border border-teal-500/30';
        } else {
          timeBadge.innerText = 'Not found';
          timeBadge.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-gray-400';
        }

        // Phase 2: Graceful Branching
        if (!hasLocation) {
          // Honest halt
          document.getElementById('missing-metadata-banner').classList.remove('hidden');
          document.getElementById('proceed-step-1-container').classList.add('hidden');
          return;
        }

        document.getElementById('missing-metadata-banner').classList.add('hidden');
        document.getElementById('proceed-step-1-container').classList.remove('hidden');

        if (detectData.primary_slick && detectData.primary_slick.centroid_lat) {
          const lat = detectData.primary_slick.centroid_lat;
          const lon = detectData.primary_slick.centroid_lon;

          const driftRes = await fetch('/api/drift', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lat, lon, hours_back: 24 })
          });
          const driftData = await driftRes.json();
          currentDriftData = driftData;
          renderDriftMap(driftData);

          const origLat = driftData.estimated_origin.lat;
          const origLon = driftData.estimated_origin.lon;
          const aisRes = await fetch('/api/ais/correlate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              origin_lat: origLat,
              origin_lon: origLon,
              scenario_hint: sampleId,
              in_image_vessels: detectData.in_image_vessels
            })
          });
          const aisData = await aisRes.json();
          currentAISData = aisData;
          renderAISResults(aisData);

          populateExecutiveReport(detectData, driftData, aisData);
        }
      } catch (err) {
        console.error('Pipeline error:', err);
      } finally {
        spinner.classList.add('hidden');
      }
    }

    function renderDetectionResults(data) {
      const isSimple = explanationMode === 'simple';
      document.getElementById('sar-base-img').src = data.original_image_base64;
      document.getElementById('sar-overlay-img').src = data.overlay_mask_base64;

      const primary = data.primary_slick;
      if (primary) {
        document.getElementById('stat-area').innerText = `${data.total_spill_area_km2} km²`;
        document.getElementById('stat-area-sub').innerText = isSimple 
          ? `~${Math.round(data.total_spill_area_km2 * 140)} soccer fields` 
          : `${Math.round(primary.area_km2 * 10000)} pixels (GSD 10m)`;

        document.getElementById('stat-elong').innerText = isSimple 
          ? (primary.elongation > 3.0 ? 'Linear Ribbon' : 'Spreading Plume') 
          : `${primary.elongation}x ratio`;
        document.getElementById('stat-elong-sub').innerText = isSimple ? 'Ship discharge wake' : 'Moment tensor ratio';

        document.getElementById('stat-age').innerText = primary.estimated_age;
        document.getElementById('stat-age-sub').innerText = isSimple ? 'High edge contrast' : 'Boundary dispersion model';

        document.getElementById('stat-conf').innerText = isSimple ? `Very High (${primary.confidence_score}%)` : `${primary.confidence_score}%`;
        document.getElementById('stat-conf-sub').innerText = isSimple ? 'Matches oil dampening' : 'S/N & contrast margin';
      } else {
        document.getElementById('stat-area').innerText = '0.0 km²';
        document.getElementById('stat-area-sub').innerText = isSimple ? 'No oil found' : 'Look-alike rejected';
        document.getElementById('stat-elong').innerText = '1.0x (Circular)';
        document.getElementById('stat-age').innerText = 'N/A';
        document.getElementById('stat-conf').innerText = isSimple ? 'Clean Water' : 'Below threshold';
      }
    }

    function renderInImageVessels(vessels) {
      const listContainer = document.getElementById('in-image-vessels-list');
      const badge = document.getElementById('vessel-targets-count-badge');
      listContainer.innerHTML = '';
      badge.innerText = `${vessels.length} DETECTED`;

      if (!vessels || vessels.length === 0) {
        listContainer.innerHTML = '<div class="text-gray-400 p-2 text-center text-[11px]">No bright radar point reflectors detected in immediate vicinity.</div>';
        return;
      }

      vessels.slice(0, 3).forEach(v => {
        const item = document.createElement('div');
        item.className = 'ops-card p-2 rounded border border-teal-500/20 text-[11px]';
        item.innerHTML = `
          <div class="flex justify-between items-center text-teal-300">
            <span class="font-bold flex items-center space-x-1">
              <span>💎</span>
              <span>${v.id}</span>
            </span>
            <span class="text-amber-400 font-bold">${v.rcs_confidence_score}% RCS Match</span>
          </div>
          <div class="text-gray-400 text-[10px] mt-0.5">
            Est Length: ~${v.estimated_length_meters}m • Pixel: [${v.pixel_position[0]}, ${v.pixel_position[1]}]
          </div>
        `;
        listContainer.appendChild(item);
      });
    }

    function renderDriftMap(drift) {
      const isSimple = explanationMode === 'simple';
      mapLayers.forEach(l => leafletMap.removeLayer(l));
      mapLayers = [];

      const pts = drift.hourly_trajectory;
      const latlngs = pts.map(p => [p.lat, p.lon]);

      const poly = L.polyline(latlngs, {
        color: '#F59E0B',
        weight: 3,
        dashArray: '5, 5',
        opacity: 0.9
      }).addTo(leafletMap);
      mapLayers.push(poly);

      pts.forEach((p, idx) => {
        const isOrigin = idx === pts.length - 1;
        const isDetect = idx === 0;
        const circle = L.circleMarker([p.lat, p.lon], {
          radius: isOrigin || isDetect ? 7 : 3.5,
          fillColor: isDetect ? '#2DD4BF' : isOrigin ? '#EF4444' : '#FBBF24',
          color: '#FFFFFF',
          weight: 1.5,
          fillOpacity: 0.95
        }).addTo(leafletMap);

        circle.bindPopup(`
          <div class="font-mono text-xs text-navy-950">
            <strong>${isDetect ? '🛰️ Spot Location (T=0)' : isOrigin ? '🎯 Origin Point (T-' + p.step_hours_back + 'h)' : 'T - ' + p.step_hours_back + ' Hours'}</strong><br>
            GPS: ${p.lat.toFixed(4)}°N, ${p.lon.toFixed(4)}°E<br>
            Drift Speed: ${p.drift_speed_knots} knots<br>
            Uncertainty: ± ${p.uncertainty_radius_km} km
          </div>
        `);
        mapLayers.push(circle);
      });

      const origin = drift.estimated_origin;
      const uncertCircle = L.circle([origin.lat, origin.lon], {
        radius: origin.uncertainty_radius_km * 1000,
        color: '#EF4444',
        fillColor: '#EF4444',
        fillOpacity: 0.15,
        weight: 1.5,
        dashArray: '3, 3'
      }).addTo(leafletMap);
      mapLayers.push(uncertCircle);

      leafletMap.fitBounds(poly.getBounds(), { padding: [30, 30] });

      document.getElementById('origin-gps-val').innerText = `${origin.lat.toFixed(4)}°N, ${origin.lon.toFixed(4)}°E`;
      document.getElementById('origin-time-val').innerText = isSimple ? `~${drift.hours_traced - 6} Hours Ago` : `T - ${drift.hours_traced}h Window`;
      document.getElementById('origin-time-sub').innerText = `Drifted ${origin.total_drift_distance_km} km`;
      document.getElementById('origin-uncert-val').innerText = `± ${origin.uncertainty_radius_km} km`;
    }

    function playDriftAnimation() {
      if (!currentDriftData) return;
      const pts = [...currentDriftData.hourly_trajectory];
      let i = 0;
      const animMarker = L.circleMarker([pts[0].lat, pts[0].lon], {
        radius: 8,
        fillColor: '#10B981',
        color: '#FFFFFF',
        weight: 2,
        fillOpacity: 1.0
      }).addTo(leafletMap);

      const interval = setInterval(() => {
        if (i >= pts.length) {
          clearInterval(interval);
          setTimeout(() => leafletMap.removeLayer(animMarker), 2000);
          return;
        }
        animMarker.setLatLng([pts[i].lat, pts[i].lon]);
        i++;
      }, 120);
    }

    function renderAISResults(ais) {
      const isSimple = explanationMode === 'simple';
      const container = document.getElementById('ais-vessels-list');
      container.innerHTML = '';

      if (!ais.ranked_vessels || ais.ranked_vessels.length === 0) {
        container.innerHTML = '<div class="text-gray-400 p-4 text-center">No commercial AIS tracks found in this area.</div>';
        return;
      }

      ais.ranked_vessels.forEach((v, idx) => {
        const isPrime = idx === 0;
        const card = document.createElement('div');
        card.className = `p-3.5 rounded border transition ${
          isPrime 
            ? 'bg-navy-950 border-crimson-500/60' 
            : 'bg-navy-900 border-slate-800'
        }`;

        card.innerHTML = `
          <div class="flex flex-wrap justify-between items-start gap-2 mb-2">
            <div>
              <div class="flex items-center space-x-2">
                <span class="text-sm font-bold ${isPrime ? 'text-white' : 'text-gray-300'}">#${idx+1} ${v.vessel_name}</span>
                <span class="text-[10px] px-1.5 py-0.5 rounded ${
                  isPrime ? 'bg-crimson-950 text-crimson-400 border border-crimson-500/40 font-bold' : 'bg-slate-800 text-gray-400'
                }">${v.attribution_tier}</span>
                ${v.is_satellite_confirmed 
                  ? '<span class="text-[9px] px-1.5 py-0.5 rounded bg-teal-950 text-teal-400 border border-teal-500/40 font-bold">🛰️ SATELLITE RADAR CONFIRMED</span>' 
                  : ''}
              </div>
              <div class="text-[11px] text-gray-400 mt-0.5">${v.vessel_type} • Flag: ${v.flag} • MMSI: ${v.mmsi}</div>
            </div>
            <div class="text-right">
              <div class="text-base font-bold ${isPrime ? 'text-crimson-400' : 'text-gray-300'}">${v.total_suspicion_score}%</div>
              <div class="text-[9px] text-gray-500">${isSimple ? 'SUSPICION RATING' : 'TOTAL SUSPICION SCORE'}</div>
            </div>
          </div>

          <div class="grid grid-cols-3 gap-2 text-[10px] bg-slate-850 p-2 rounded mb-2">
            <div>${isSimple ? 'Proximity:' : 'Proximity (40%):'} <strong class="text-white">${v.score_breakdown.proximity_score}%</strong> (${v.min_distance_to_origin_km}km)</div>
            <div>${isSimple ? 'Course Match:' : 'Trajectory (30%):'} <strong class="text-white">${v.score_breakdown.trajectory_score}%</strong></div>
            <div>${isSimple ? 'Red Flags:' : 'Anomaly (30%):'} <strong class="text-amber-400">${v.score_breakdown.anomaly_score}%</strong></div>
          </div>

          ${v.anomaly_flags.length > 0 ? `
            <div class="text-[10px] text-amber-300 space-y-0.5">
              ${v.anomaly_flags.map(f => `<div>⚠️ ${f}</div>`).join('')}
            </div>
          ` : '<div class="text-[10px] text-gray-500">Normal commercial transit profile.</div>'}
        `;
        container.appendChild(card);
      });

      const top = ais.top_suspect;
      if (top) {
        document.getElementById('suspect-vessel-name').innerText = top.vessel_name;
        document.getElementById('suspect-score-badge').innerText = `${top.total_suspicion_score}% SUSPICIOUS`;
        document.getElementById('suspect-vessel-type').innerText = `${top.vessel_type} (${top.flag} Flag)`;
        document.getElementById('suspect-explanation').innerText = isSimple
          ? `Passed within ${top.min_distance_to_origin_km} km of origin. ${top.is_satellite_confirmed ? 'Cross-verified by in-image satellite radar target spotting.' : ''}`
          : `CPA: ${top.min_distance_to_origin_km} km | Flags: ${top.anomaly_flags.join('; ')}`;
      }
    }

    function populateExecutiveReport(detect, drift, ais) {
      const isSimple = explanationMode === 'simple';
      const primary = detect.primary_slick;
      const origin = drift.estimated_origin;
      const top = ais.top_suspect;

      document.getElementById('rep-source-badge').innerText = isManualLocation
        ? 'Location: Manually Provided by Analyst'
        : 'Location: Auto-Extracted GeoTIFF';
      document.getElementById('rep-source-badge').className = isManualLocation
        ? 'px-2 py-0.5 rounded bg-amber-950 text-amber-300 text-[10px] border border-amber-500/30'
        : 'px-2 py-0.5 rounded bg-teal-950 text-teal-300 text-[10px] border border-teal-500/30';

      if (primary) {
        document.getElementById('rep-area').innerText = `${detect.total_spill_area_km2} km²`;
        document.getElementById('rep-elong').innerText = isSimple ? 'Elongated Trail' : `${primary.elongation}x Ribbon`;
        document.getElementById('rep-age').innerText = primary.estimated_age;
        if (primary.centroid_lat) {
          document.getElementById('rep-det-gps').innerText = `${primary.centroid_lat.toFixed(4)}°N, ${primary.centroid_lon.toFixed(4)}°E`;
        }
      }

      if (origin) {
        document.getElementById('rep-orig-gps').innerText = `${origin.lat.toFixed(4)}°N, ${origin.lon.toFixed(4)}°E`;
        document.getElementById('rep-drift-dist').innerText = `${origin.total_drift_distance_km} km`;
        document.getElementById('rep-window').innerText = isSimple ? '~18 Hours Ago' : `T - ${drift.hours_traced}h (± 2h)`;
        document.getElementById('rep-uncert').innerText = `± ${origin.uncertainty_radius_km} km`;
      }

      if (top) {
        document.getElementById('rep-vessel-name').innerText = top.vessel_name;
        document.getElementById('rep-vessel-flag').innerText = `${top.vessel_type} (${top.flag})`;
        document.getElementById('rep-mmsi').innerText = top.mmsi;
        document.getElementById('rep-cpa').innerText = `${top.min_distance_to_origin_km} km to Origin`;
        document.getElementById('rep-score').innerText = `${top.total_suspicion_score}% (${top.attribution_tier})`;

        const anomContainer = document.getElementById('rep-anomalies-list');
        anomContainer.innerHTML = top.anomaly_flags.map(f => `<li><strong>Red Flag:</strong> ${f}</li>`).join('');

        document.getElementById('rep-plain-summary').innerText = isSimple
          ? `Satellite radar confirmed a ${detect.total_spill_area_km2} km² oil slick. Ocean drift modeling traced the spill back ${origin.total_drift_distance_km} km to its release location ~18 hours ago (${isManualLocation ? 'origin location manually provided' : 'origin location extracted from GeoTIFF'}). Ship transponder records identify ${top.vessel_name} as the prime suspect (${top.total_suspicion_score}% suspicion)${top.is_satellite_confirmed ? ' with matching in-image satellite radar target correlation.' : '.'}`
          : `Sentinel-1 SAR C-band detection confirmed ${detect.total_spill_area_km2} km² backscatter dampening. Lagrangian hydrodynamic hindcast localized release coordinates to ${origin.lat.toFixed(4)}°N, ${origin.lon.toFixed(4)}°E. AIS cross-correlation attributed highest suspicion (${top.total_suspicion_score}%) to ${top.vessel_name} (MMSI: ${top.mmsi}) with direct intercept CPA of ${top.min_distance_to_origin_km} km.`;
      }
    }

    function setViewMode(mode) {
      const overlay = document.getElementById('sar-overlay-img');
      ['btn-view-composite', 'btn-view-raw', 'btn-view-mask'].forEach(id => {
        document.getElementById(id).className = 'px-2 py-0.5 rounded text-gray-400 hover:text-white';
      });

      if (mode === 'composite') {
        document.getElementById('btn-view-composite').className = 'px-2 py-0.5 rounded bg-teal-500 text-navy-950 font-bold';
        overlay.style.display = 'block';
        overlay.style.opacity = (document.getElementById('opacity-slider').value / 100).toString();
      } else if (mode === 'raw') {
        document.getElementById('btn-view-raw').className = 'px-2 py-0.5 rounded bg-teal-500 text-navy-950 font-bold';
        overlay.style.display = 'none';
      } else if (mode === 'mask') {
        document.getElementById('btn-view-mask').className = 'px-2 py-0.5 rounded bg-teal-500 text-navy-950 font-bold';
        overlay.style.display = 'block';
        overlay.style.opacity = '1.0';
      }
    }

    function updateOverlayOpacity(val) {
      document.getElementById('opacity-val').innerText = val + '%';
      document.getElementById('sar-overlay-img').style.opacity = (val / 100).toString();
    }

    window.onload = init;
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run("local_server:app", host="0.0.0.0", port=8000, reload=False)
