"""
OrthoTwin AI — X-ray Image Processing Module
Generates visual review copies (CLAHE) while preserving original raw files untouched.
Prepares evaluation-compatible inputs for AI detection without destructive alterations.
"""

import io
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
import numpy as np
import cv2


def apply_clahe_enhancement(
    image_np: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to improve local bone edge visibility for clinician visual review.
    Does NOT overwrite the original raw radiograph.
    """
    if len(image_np.shape) == 3:
        # Convert RGB to LAB, apply CLAHE to L-channel, convert back
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(image_np)
    return enhanced


def generate_processed_review_copy(raw_image_path: Path) -> bytes:
    """
    Reads the original raw scan, generates an enhanced CLAHE copy for visual review,
    and returns JPEG bytes for separate storage.
    """
    with Image.open(raw_image_path) as img:
        rgb_img = img.convert("RGB")
        np_arr = np.array(rgb_img)

    enhanced = apply_clahe_enhancement(np_arr)
    enhanced_pil = Image.fromarray(enhanced)

    buffer = io.BytesIO()
    enhanced_pil.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def prepare_inference_input(
    raw_image_path: Path,
    target_size: int = 640
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Loads raw image and letterboxes it for YOLOv8 inference,
    maintaining aspect ratio and matching standard YOLO training input pipeline.
    Returns (letterboxed_rgb_image, scale_factor, (pad_w, pad_h)).
    """
    with Image.open(raw_image_path) as img:
        rgb_img = img.convert("RGB")
        np_arr = np.array(rgb_img)

    h, w = np_arr.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(np_arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target_size x target_size (letterboxing)
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2

    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas, scale, (pad_left, pad_top)
