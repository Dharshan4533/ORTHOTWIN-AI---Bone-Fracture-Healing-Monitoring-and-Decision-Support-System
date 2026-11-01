"""
OrthoTwin AI — X-ray Validation Module
Performs multi-stage content inspection, corrupted image rejection,
minimum dimension enforcement, duplicate detection (SHA-256 + dHash),
and radiographic technical quality checks.
"""

import io
import hashlib
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image, ImageOps
import numpy as np
import cv2
from config import (
    MIN_IMAGE_DIMENSION, MAX_FILE_SIZE_BYTES, ALLOWED_IMAGE_EXTENSIONS,
    BLUR_LAPLACIAN_THRESHOLD, LOW_CONTRAST_THRESHOLD,
    MIN_BRIGHTNESS, MAX_BRIGHTNESS
)


class XrayValidationError(Exception):
    """Raised when an uploaded file fails fundamental integrity or security checks."""
    pass


def compute_sha256(data: bytes) -> str:
    """Computes SHA-256 hash for byte-level duplicate detection."""
    return hashlib.sha256(data).hexdigest()


def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """
    Computes difference hash (dHash) for perceptual visual similarity tracking.
    Resizes to (hash_size + 1, hash_size), converts to grayscale, and compares adjacent pixels.
    """
    try:
        resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(resized)
        diff = pixels[:, 1:] > pixels[:, :-1]
        decimal_val = 0
        hex_str = []
        for index, val in enumerate(diff.flatten()):
            if val:
                decimal_val += 2 ** (index % 4)
            if (index % 4) == 3:
                hex_str.append(hex(decimal_val)[2:])
                decimal_val = 0
        return "".join(hex_str)
    except Exception:
        return ""


def compute_dhash_distance(hash1: str, hash2: str) -> int:
    """
    Computes Hamming distance between two hex dHash strings.
    A distance of 0 indicates visually identical images.
    Distances <= 10 indicate high visual similarity / near duplicates.
    """
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return 64
    try:
        val1 = int(hash1, 16)
        val2 = int(hash2, 16)
        return bin(val1 ^ val2).count("1")
    except ValueError:
        return 64


def verify_magic_bytes(data: bytes) -> str:
    """
    Validates file magic bytes (signatures) to prevent extension spoofing.
    Supports JPEG, PNG, and flags DICOM preamble.
    """
    if len(data) < 132:
        raise XrayValidationError("File too small to be a valid image file.")

    # DICOM check (prefix at byte 128 is 'DICM')
    if data[128:132] == b"DICM":
        return "DICOM"

    # JPEG signatures
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"

    # PNG signature
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"

    raise XrayValidationError("Invalid image signature. Only genuine JPEG and PNG files are accepted.")


def assess_technical_quality(image_np: np.ndarray) -> Tuple[float, float, float, float, List[str]]:
    """
    Analyzes radiographic image technical attributes:
    - Blur via Laplacian variance
    - Dynamic range / contrast via standard deviation
    - Mean brightness
    - Brightness clipping (overexposure/underexposure)
    Returns (blur_score, contrast_score, min_val, max_val, warnings).
    """
    warnings: List[str] = []

    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np

    # Blur score
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if laplacian_var < BLUR_LAPLACIAN_THRESHOLD:
        warnings.append(
            f"Image blur detected (Laplacian score {laplacian_var:.1f} < threshold {BLUR_LAPLACIAN_THRESHOLD:.1f}). Fine cortical margins may be obscured."
        )

    # Contrast score
    contrast_std = float(np.std(gray))
    if contrast_std < LOW_CONTRAST_THRESHOLD:
        warnings.append(
            f"Low radiographic contrast (std dev {contrast_std:.1f} < threshold {LOW_CONTRAST_THRESHOLD:.1f}). Dynamic range is restricted."
        )

    # Brightness extremes
    mean_val = float(np.mean(gray))
    min_val = float(np.min(gray))
    max_val = float(np.max(gray))

    if mean_val < MIN_BRIGHTNESS:
        warnings.append(f"Image is severely underexposed (mean intensity {mean_val:.1f} < {MIN_BRIGHTNESS:.1f}).")
    elif mean_val > MAX_BRIGHTNESS:
        warnings.append(f"Image is severely overexposed/washed out (mean intensity {mean_val:.1f} > {MAX_BRIGHTNESS:.1f}).")

    return laplacian_var, contrast_std, min_val, max_val, warnings


def validate_xray_upload(
    file_bytes: bytes,
    original_filename: str
) -> Dict[str, Any]:
    """
    Comprehensive validation pipeline for incoming X-ray uploads:
    1. Size limit check
    2. Magic byte / header check
    3. PIL decoding and corruption test
    4. Minimum dimension check
    5. Hash computation
    6. Technical quality metrics
    7. Clear unverified medical suitability notice
    """
    file_size = len(file_bytes)
    if file_size > MAX_FILE_SIZE_BYTES:
        raise XrayValidationError(
            f"File size exceeds maximum limit of {MAX_FILE_SIZE_BYTES / (1024*1024):.0f} MB."
        )

    # Magic byte verification
    detected_format = verify_magic_bytes(file_bytes)
    if detected_format == "DICOM":
        raise XrayValidationError(
            "DICOM file detected. DICOM format requires the dedicated hospital PACS ingestion pathway. Please provide exported JPEG/PNG for this prototype."
        )

    # Attempt PIL load and decode
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img.verify()  # Fast corruption check
        # Reopen because verify() closes the stream
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img.load()  # Full decompression check
    except Exception as e:
        raise XrayValidationError(f"Corrupted or unreadable image stream: {str(e)}")

    width, height = pil_img.size
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise XrayValidationError(
            f"Image resolution {width}x{height} is below minimum requirement of {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}."
        )

    # Convert to RGB numpy array for OpenCV
    rgb_img = pil_img.convert("RGB")
    np_img = np.array(rgb_img)

    # Hashes
    sha256_hash = compute_sha256(file_bytes)
    dhash_val = compute_dhash(rgb_img)

    # Technical Quality
    blur_score, contrast_score, min_val, max_val, quality_warnings = assess_technical_quality(np_img)

    # Medical Suitability Notice (Mandatory Safeguard)
    medical_suitability_notice = (
        "UNVERIFIED RADIOGRAPH: Technical file validation verifies structural image integrity only. "
        "It does NOT guarantee or certify that this image is a genuine human bone radiograph. "
        "Visual validation by a qualified medical professional is mandatory."
    )

    return {
        "is_valid": True,
        "format": detected_format,
        "width": width,
        "height": height,
        "file_size": file_size,
        "sha256": sha256_hash,
        "dhash": dhash_val,
        "blur_score": blur_score,
        "contrast_score": contrast_score,
        "min_brightness": min_val,
        "max_brightness": max_val,
        "warnings": quality_warnings,
        "medical_suitability_notice": medical_suitability_notice
    }
