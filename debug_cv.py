import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sar_loader import load_sar_image
from sar_detector import enhanced_lee_filter, segment_dark_patches, morphological_cleanup, filter_lookalikes
from samples_generator import SAMPLES_DIR

tif_path = os.path.join(SAMPLES_DIR, "sample_1_tanker_slick.tif")
img, geo_meta = load_sar_image(file_path=tif_path)
print("Image shape:", img.shape, "min:", img.min(), "max:", img.max(), "mean:", img.mean())

lee = enhanced_lee_filter(img)
print("Lee min:", lee.min(), "max:", lee.max(), "mean:", lee.mean())

seg = segment_dark_patches(lee, k_factor=1.2, use_otsu=True)
print("Seg sum pixels:", np.sum(seg > 0))

clean = morphological_cleanup(seg)
print("Clean sum pixels:", np.sum(clean > 0))

final, slicks = filter_lookalikes(clean, img, geo_meta, min_area_km2=0.01)
print("Slicks count:", len(slicks))
if slicks:
    print("Slick 1:", slicks[0])
