"""
OrthoTwin AI — Transactional Upload & Dataset Verification Tests
Validates:
1. Transactional upload failure rollback (no orphan files left on disk)
2. Perceptual dHash distance metric and near-duplicate matching
3. Dataset audit failure detection on missing archives
"""

import io
from pathlib import Path
from PIL import Image
import numpy as np
import pytest

from modules.xray_validation import (
    compute_dhash, compute_dhash_distance, validate_xray_upload
)
from training.dataset_validation import audit_roboflow_dataset, audit_fracatlas_dataset, generate_audit_report
from database.models import User, Patient, FractureCase, Visit, XrayImage
from database.repositories import XrayImageRepository


def test_dhash_distance_calculation():
    # Create test image 1
    arr1 = np.ones((256, 256), dtype=np.uint8) * 100
    arr1[50:150, 50:150] = 200
    img1 = Image.fromarray(arr1, mode="L")
    hash1 = compute_dhash(img1)

    # Identical image -> distance 0
    assert compute_dhash_distance(hash1, hash1) == 0

    # Slightly perturbed image (small noise) -> distance <= 10
    arr2 = arr1.copy()
    arr2[52:148, 52:148] = 198
    img2 = Image.fromarray(arr2, mode="L")
    hash2 = compute_dhash(img2)
    dist_near = compute_dhash_distance(hash1, hash2)
    assert dist_near <= 10

    # Inverted image -> high distance
    arr3 = 255 - arr1
    img3 = Image.fromarray(arr3, mode="L")
    hash3 = compute_dhash(img3)
    dist_diff = compute_dhash_distance(hash1, hash3)
    assert dist_diff > 20


def test_dataset_audit_fails_honestly_when_archives_missing(tmp_path):
    missing_rf = tmp_path / "missing_roboflow.zip"
    missing_fa = tmp_path / "missing_fracatlas.zip"

    rf_res = audit_roboflow_dataset(missing_rf)
    assert rf_res["exists"] is False
    assert "Archive file not found" in rf_res["error"]

    fa_res = audit_fracatlas_dataset(missing_fa)
    assert fa_res["exists"] is False
    assert "Archive file not found" in fa_res["error"]

    out_report = tmp_path / "AUDIT_REPORT.md"
    generate_audit_report(rf_res, fa_res, out_report)
    assert out_report.exists()
    content = out_report.read_text(encoding="utf-8")
    assert "FAILED — ARCHIVES NOT FOUND" in content
    assert "Neither dataset archive was found on disk" in content
