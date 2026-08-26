"""
SAR Image Loader & Georeferencing Module
SpillTrace - SIH 2026 PS 26143 (NTRO)

Supports:
- GeoTIFF files (.tif, .tiff) via Rasterio (File path or MemoryFile bytes)
- Standard SAR imagery (PNG, JPG, BMP) with honest metadata extraction (EXIF/TIFF tags)
- Explicit metadata extraction status object:
  {
    "has_location": bool,
    "has_timestamp": bool,
    "coordinates": {"min_lat", "max_lat", "min_lon", "max_lon"} or None,
    "timestamp": str or None,
    "crs": str or None,
    "gsd_meters": float or None,
    "metadata_source": "embedded_geotiff" | "exif" | "none"
  }
"""

import io
import os
import math
import datetime
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image, ExifTags

try:
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import xy
    from rasterio.warp import transform as warp_transform
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


def extract_sar_metadata(
    file_bytes: Optional[bytes] = None,
    file_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Inspects image file for embedded georeferencing and timestamp tags without modifying the file.
    Returns honest metadata availability status.
    """
    meta_status = {
        "has_location": False,
        "has_timestamp": False,
        "coordinates": None,
        "timestamp": None,
        "crs": None,
        "gsd_meters": None,
        "metadata_source": "none",
        "details": "No embedded georeferencing or capture timestamp found in file headers."
    }
    
    # 1. Inspect GeoTIFF tags via Rasterio
    if RASTERIO_AVAILABLE:
        try:
            if file_bytes is not None:
                with MemoryFile(file_bytes) as memfile:
                    with memfile.open() as src:
                        if src.crs is not None:
                            meta_status["has_location"] = True
                            meta_status["crs"] = str(src.crs)
                            meta_status["metadata_source"] = "embedded_geotiff"
                            res_x = abs(src.res[0])
                            if "deg" in str(src.crs).lower() or (hasattr(src.crs, "is_geographic") and src.crs.is_geographic):
                                meta_status["gsd_meters"] = round(float(res_x * 111320.0), 2)
                            else:
                                meta_status["gsd_meters"] = round(float(res_x), 2)
                                
                            bounds = src.bounds
                            meta_status["coordinates"] = {
                                "min_lat": float(bounds.bottom),
                                "max_lat": float(bounds.top),
                                "min_lon": float(bounds.left),
                                "max_lon": float(bounds.right)
                            }
                            # Check tags for timestamp
                            tags = src.tags()
                            for t_key in ["TIFFTAG_DATETIME", "ACQUISITION_DATETIME", "DATETIME", "TIME"]:
                                if t_key in tags:
                                    meta_status["has_timestamp"] = True
                                    meta_status["timestamp"] = tags[t_key]
                                    break
            elif file_path and os.path.exists(file_path) and (file_path.endswith('.tif') or file_path.endswith('.tiff')):
                with rasterio.open(file_path) as src:
                    if src.crs is not None:
                        meta_status["has_location"] = True
                        meta_status["crs"] = str(src.crs)
                        meta_status["metadata_source"] = "embedded_geotiff"
                        res_x = abs(src.res[0])
                        if "deg" in str(src.crs).lower() or (hasattr(src.crs, "is_geographic") and src.crs.is_geographic):
                            meta_status["gsd_meters"] = round(float(res_x * 111320.0), 2)
                        else:
                            meta_status["gsd_meters"] = round(float(res_x), 2)
                        bounds = src.bounds
                        meta_status["coordinates"] = {
                            "min_lat": float(bounds.bottom),
                            "max_lat": float(bounds.top),
                            "min_lon": float(bounds.left),
                            "max_lon": float(bounds.right)
                        }
                        tags = src.tags()
                        for t_key in ["TIFFTAG_DATETIME", "ACQUISITION_DATETIME", "DATETIME", "TIME"]:
                            if t_key in tags:
                                meta_status["has_timestamp"] = True
                                meta_status["timestamp"] = tags[t_key]
                                break
        except Exception as e:
            print(f"[sar_loader] extract_sar_metadata rasterio error: {e}")

    # 2. Inspect EXIF metadata via PIL (for JPG/PNG/TIFF)
    if not meta_status["has_timestamp"] or not meta_status["has_location"]:
        try:
            if file_bytes:
                pil_img = Image.open(io.BytesIO(file_bytes))
            elif file_path and os.path.exists(file_path):
                pil_img = Image.open(file_path)
            else:
                pil_img = None
                
            if pil_img and hasattr(pil_img, '_getexif') and pil_img._getexif():
                exif = pil_img._getexif()
                for tag_id, value in exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag_name in ["DateTimeOriginal", "DateTime", "DateTimeDigitized"]:
                        meta_status["has_timestamp"] = True
                        meta_status["timestamp"] = str(value)
                        if meta_status["metadata_source"] == "none":
                            meta_status["metadata_source"] = "exif"
        except Exception:
            pass

    if meta_status["has_location"]:
        meta_status["details"] = f"Embedded {meta_status['metadata_source'].upper()} georeferencing detected ({meta_status['crs']})."
    else:
        meta_status["details"] = "No embedded geographic location found in file. Location can be manually provided."
        
    return meta_status


def pixel_to_latlon(px: float, py: float, geo_meta: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Converts pixel (x, y) coordinates to WGS84 Latitude and Longitude.
    Uses GeoTIFF affine transform if available, otherwise uses interpolated bounding box if present.
    """
    if geo_meta.get("has_geotiff") and RASTERIO_AVAILABLE and "transform" in geo_meta:
        trans = geo_meta["transform"]
        x_coord, y_coord = trans * (px, py)
        crs = geo_meta.get("crs", "EPSG:4326")
        
        if str(crs).upper() not in ["EPSG:4326", "WGS84", "+PROJ=LONGPLAT"]:
            try:
                lon_arr, lat_arr = warp_transform(crs, "EPSG:4326", [x_coord], [y_coord])
                return float(lat_arr[0]), float(lon_arr[0])
            except Exception:
                pass
        return float(y_coord), float(x_coord)
    
    # If bounding box is available
    if geo_meta.get("bbox") and geo_meta.get("has_location", False):
        bbox = geo_meta["bbox"]
        width = max(1, geo_meta.get("width", 1024))
        height = max(1, geo_meta.get("height", 1024))
        u = np.clip(px / width, 0.0, 1.0)
        v = np.clip(py / height, 0.0, 1.0)
        lat = bbox["max_lat"] - v * (bbox["max_lat"] - bbox["min_lat"])
        lon = bbox["min_lon"] + u * (bbox["max_lon"] - bbox["min_lon"])
        return round(float(lat), 6), round(float(lon), 6)
        
    return None, None


def load_sar_image(
    file_bytes: Optional[bytes] = None,
    file_path: Optional[str] = None,
    default_bbox: Optional[Dict[str, float]] = None,
    is_sample: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Loads SAR image from raw bytes or file path.
    Returns:
        gray_image: 2D uint8 numpy array (normalized SAR backscatter intensity)
        geo_meta: Dictionary containing georeferencing, metadata status, and resolution (GSD)
    """
    # First extract honest metadata
    meta_status = extract_sar_metadata(file_bytes=file_bytes, file_path=file_path)
    
    geo_meta: Dict[str, Any] = {
        "has_geotiff": meta_status["has_location"] and meta_status["metadata_source"] == "embedded_geotiff",
        "has_location": meta_status["has_location"] or (is_sample and default_bbox is not None),
        "has_timestamp": meta_status["has_timestamp"],
        "metadata_source": meta_status["metadata_source"] if meta_status["has_location"] else ("curated_sample" if is_sample else "none"),
        "timestamp": meta_status["timestamp"],
        "crs": meta_status["crs"] or "EPSG:4326",
        "gsd_meters": meta_status["gsd_meters"] or 10.0,
        "width": 0,
        "height": 0,
        "bbox": meta_status["coordinates"] or (default_bbox if is_sample else None),
        "metadata_status": meta_status
    }
    
    img_array = None
    
    # 1. Try Rasterio MemoryFile if bytes provided
    if RASTERIO_AVAILABLE and file_bytes is not None:
        try:
            with MemoryFile(file_bytes) as memfile:
                with memfile.open() as src:
                    band = src.read(1)
                    geo_meta["width"] = src.width
                    geo_meta["height"] = src.height
                    if src.crs is not None:
                        geo_meta["has_geotiff"] = True
                        geo_meta["has_location"] = True
                        geo_meta["transform"] = src.transform
                        geo_meta["crs"] = str(src.crs)
                        min_lat, min_lon = pixel_to_latlon(0, src.height, geo_meta)
                        max_lat, max_lon = pixel_to_latlon(src.width, 0, geo_meta)
                        if min_lat is not None and max_lat is not None:
                            geo_meta["bbox"] = {
                                "min_lat": min(min_lat, max_lat),
                                "max_lat": max(min_lat, max_lat),
                                "min_lon": min(min_lon, max_lon),
                                "max_lon": max(min_lon, max_lon),
                            }
                    img_array = band
        except Exception:
            pass

    # 2. Try rasterio with file_path
    if img_array is None and RASTERIO_AVAILABLE and file_path and (file_path.endswith('.tif') or file_path.endswith('.tiff')):
        try:
            with rasterio.open(file_path) as src:
                band = src.read(1)
                geo_meta["width"] = src.width
                geo_meta["height"] = src.height
                if src.crs is not None:
                    geo_meta["has_geotiff"] = True
                    geo_meta["has_location"] = True
                    geo_meta["transform"] = src.transform
                    geo_meta["crs"] = str(src.crs)
                    min_lat, min_lon = pixel_to_latlon(0, src.height, geo_meta)
                    max_lat, max_lon = pixel_to_latlon(src.width, 0, geo_meta)
                    if min_lat is not None and max_lat is not None:
                        geo_meta["bbox"] = {
                            "min_lat": min(min_lat, max_lat),
                            "max_lat": max(min_lat, max_lat),
                            "min_lon": min(min_lon, max_lon),
                            "max_lon": max(min_lon, max_lon),
                        }
                img_array = band
        except Exception as e:
            print(f"[sar_loader] Rasterio file error: {e}")

    # 3. Fallback to OpenCV / PIL for standard images (PNG, JPG, etc.)
    if img_array is None:
        if file_bytes is not None:
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        elif file_path is not None and os.path.exists(file_path):
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        else:
            raise ValueError("No valid file_bytes or file_path provided to load_sar_image.")
            
        if img is None:
            if file_bytes:
                pil_img = Image.open(io.BytesIO(file_bytes))
            else:
                pil_img = Image.open(file_path)
            img = np.array(pil_img)
            
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_array = img

    # Normalize image to uint8 [0, 255] with percentile contrast stretching
    img_float = img_array.astype(np.float32)
    img_float = np.nan_to_num(img_float, nan=0.0, posinf=255.0, neginf=0.0)
    
    p2, p98 = np.percentile(img_float, (2, 98))
    if p98 > p2:
        stretched = np.clip((img_float - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
    else:
        stretched = cv2.normalize(img_float, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
    h, w = stretched.shape[:2]
    geo_meta["width"] = w
    geo_meta["height"] = h
    
    return stretched, geo_meta
