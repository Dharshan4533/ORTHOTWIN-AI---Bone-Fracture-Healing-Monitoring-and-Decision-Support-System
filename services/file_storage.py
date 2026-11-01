"""
OrthoTwin AI — Secure File Storage Service
Prevents path-traversal attacks, isolates raw medical scans from processed copies,
and manages filesystem boundaries.
"""

import os
import uuid
from pathlib import Path
from typing import Tuple, Optional
from config import (
    STORAGE_DIR, STORAGE_RAW_DIR, STORAGE_PROCESSED_DIR,
    STORAGE_REPORTS_DIR, ALLOWED_IMAGE_EXTENSIONS
)


def sanitize_filename(filename: str) -> str:
    """Removes path separators and non-standard characters from a filename."""
    clean = os.path.basename(filename)
    clean = clean.replace("..", "").replace("/", "").replace("\\", "")
    return clean or "upload.dat"


def is_safe_path(target_path: Path, base_directory: Path = STORAGE_DIR) -> bool:
    """Verifies that a target path resolves strictly within the designated storage directory."""
    try:
        resolved_target = target_path.resolve()
        resolved_base = base_directory.resolve()
        return resolved_base in resolved_target.parents or resolved_target == resolved_base
    except Exception:
        return False


def get_safe_file_path(relative_or_absolute: str) -> Path:
    """Resolves and validates that a stored path stays within the application storage area."""
    p = Path(relative_or_absolute)
    if not p.is_absolute():
        p = (STORAGE_DIR.parent / p).resolve()
    else:
        p = p.resolve()
    if not is_safe_path(p, STORAGE_DIR.parent):
        raise ValueError("Security violation: path traversal detected.")
    return p


def save_raw_image(file_bytes: bytes, original_filename: str) -> Tuple[Path, str]:
    """
    Saves the original uploaded image file directly to raw storage.
    Generates a secure UUID prefix while preserving the original sanitized extension.
    Returns (absolute_path, relative_storage_path).
    """
    clean_name = sanitize_filename(original_filename)
    ext = Path(clean_name).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported file extension '{ext}'. Allowed: {ALLOWED_IMAGE_EXTENSIONS}")

    unique_filename = f"{uuid.uuid4().hex[:12]}_{clean_name}"
    target_path = STORAGE_RAW_DIR / unique_filename

    if not is_safe_path(target_path, STORAGE_RAW_DIR):
        raise ValueError("Invalid target storage path.")

    with open(target_path, "wb") as f:
        f.write(file_bytes)

    relative_path = str(target_path.relative_to(STORAGE_DIR.parent)).replace("\\", "/")
    return target_path, relative_path


def save_processed_image(processed_bytes: bytes, raw_relative_path: str) -> Tuple[Path, str]:
    """
    Saves an enhanced/contrast-adjusted copy to the processed storage folder.
    Keeps processed copies strictly separated from raw original scans.
    """
    raw_name = Path(raw_relative_path).name
    processed_filename = f"proc_{raw_name}"
    target_path = STORAGE_PROCESSED_DIR / processed_filename

    if not is_safe_path(target_path, STORAGE_PROCESSED_DIR):
        raise ValueError("Invalid processed storage path.")

    with open(target_path, "wb") as f:
        f.write(processed_bytes)

    relative_path = str(target_path.relative_to(STORAGE_DIR.parent)).replace("\\", "/")
    return target_path, relative_path
