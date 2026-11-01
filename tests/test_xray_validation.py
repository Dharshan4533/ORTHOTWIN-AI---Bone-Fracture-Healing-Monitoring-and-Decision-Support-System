"""
OrthoTwin AI — Unit Tests for X-ray Processing & Validation
Tests file integrity, magic bytes, corruption rejection, dimension checks,
and technical quality warnings.
"""

import pytest
from modules.xray_validation import (
    validate_xray_upload, verify_magic_bytes, compute_sha256,
    compute_dhash, XrayValidationError
)


def test_valid_image_upload(synthetic_valid_image_bytes):
    result = validate_xray_upload(synthetic_valid_image_bytes, "scan.jpg")
    assert result["is_valid"] is True
    assert result["format"] == "JPEG"
    assert result["width"] == 512
    assert result["height"] == 512
    assert len(result["sha256"]) == 64
    assert "UNVERIFIED RADIOGRAPH" in result["medical_suitability_notice"]


def test_corrupted_file_rejection():
    corrupted_bytes = b"\xff\xd8\xff\xe0" + b"CorruptedData" * 20
    with pytest.raises(XrayValidationError, match="Corrupted or unreadable"):
        validate_xray_upload(corrupted_bytes, "corrupt.jpg")


def test_extension_spoofing_rejection():
    # Renamed text file claiming to be a jpeg
    fake_bytes = b"Hello, this is just plain text content pretending to be an x-ray." * 5
    with pytest.raises(XrayValidationError, match="Invalid image signature"):
        validate_xray_upload(fake_bytes, "fake_xray.jpg")


def test_undersized_dimensions_rejection(synthetic_small_image_bytes):
    with pytest.raises(XrayValidationError, match="below minimum requirement"):
        validate_xray_upload(synthetic_small_image_bytes, "tiny.jpg")


def test_oversized_file_rejection():
    huge_bytes = b"\xff\xd8\xff" + b"0" * (16 * 1024 * 1024)
    with pytest.raises(XrayValidationError, match="exceeds maximum limit"):
        validate_xray_upload(huge_bytes, "huge.jpg")


def test_duplicate_hash_calculation(synthetic_valid_image_bytes):
    h1 = compute_sha256(synthetic_valid_image_bytes)
    h2 = compute_sha256(synthetic_valid_image_bytes)
    assert h1 == h2


def test_blur_warning_flag(synthetic_blurred_image_bytes):
    result = validate_xray_upload(synthetic_blurred_image_bytes, "blurred.jpg")
    assert result["is_valid"] is True
    # Should flag blur in warnings
    assert any("blur detected" in w.lower() for w in result["warnings"])
