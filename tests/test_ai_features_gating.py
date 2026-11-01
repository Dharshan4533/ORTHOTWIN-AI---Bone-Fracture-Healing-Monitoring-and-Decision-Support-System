"""
OrthoTwin AI — AI Features & Gating Integrity Tests
Validates:
1. U-Net segmentation returns SEGMENTATION_UNAVAILABLE when valid weights are not loaded.
   Never claims READY with a fake zero-mask.
2. COCO polygon rasterization converts polygon vertices accurately.
3. Prediction engine enforces PREDICTION_UNAVAILABLE invariant.
4. Digital twin preserves missing measurements without fabricating 0.0 or 1.0.
"""

import numpy as np
import pytest
from pathlib import Path

from modules.fracture_segmentation import (
    SegmentationEngine, rasterize_coco_polygon_to_mask
)
from modules.prediction_engine import get_healing_prediction
from modules.digital_twin import extract_experimental_features, build_longitudinal_trajectory_plot


def test_segmentation_unavailable_without_real_weights(tmp_path):
    # Test 1: Non-existent weights
    engine_missing = SegmentationEngine(weights_path=tmp_path / "non_existent.pth")
    assert engine_missing.is_loaded is False
    res_missing = engine_missing.predict_mask(np.zeros((256, 256), dtype=np.uint8))
    assert res_missing["available"] is False
    assert res_missing["status"] == "SEGMENTATION_UNAVAILABLE"
    assert res_missing["mask"] is None

    # Test 2: Corrupted or empty weights file
    corrupt_pth = tmp_path / "corrupt_unet.pth"
    corrupt_pth.write_bytes(b"corrupt dummy bytes not a real pytorch model")
    engine_corrupt = SegmentationEngine(weights_path=corrupt_pth)
    assert engine_corrupt.is_loaded is False
    res_corrupt = engine_corrupt.predict_mask(np.zeros((256, 256), dtype=np.uint8))
    assert res_corrupt["available"] is False
    assert res_corrupt["status"] == "SEGMENTATION_UNAVAILABLE"
    assert res_corrupt["mask"] is None


def test_coco_polygon_rasterization():
    # Polygon defining a 20x20 square from (10,10) to (30,30)
    polygon = [[10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]]
    mask = rasterize_coco_polygon_to_mask(polygon, height=50, width=50)
    assert mask.shape == (50, 50)
    assert mask.dtype == np.uint8
    # Inside the square should be 255
    assert mask[20, 20] == 255
    # Outside should be 0
    assert mask[5, 5] == 0
    assert mask[40, 40] == 0


def test_healing_prediction_gating():
    pred = get_healing_prediction("CASE-TEST-001", visit_count=3)
    assert pred["is_available"] is False
    assert pred["status"] == "PREDICTION_UNAVAILABLE"
    assert pred["prediction"] is None
    assert pred["predicted_union_date"] is None
    assert len(pred["requirements_for_enablement"]) >= 3


def test_digital_twin_preserves_missing_measurements():
    visits_data = [
        {"visit_date": "2026-01-01", "days_post_injury": 0, "fracture_gap_px": None, "callus_density_ratio": None, "pain_score": 7},
        {"visit_date": "2026-01-14", "days_post_injury": 14, "fracture_gap_px": 8.5, "callus_density_ratio": 1.15, "pain_score": 4},
    ]
    fig = build_longitudinal_trajectory_plot(visits_data)
    # Check trace 0 (gap proxy)
    gap_trace = fig.data[0]
    assert gap_trace.y[0] is None
    assert gap_trace.y[1] == 8.5
    assert gap_trace.connectgaps is False

    # Check trace 1 (callus ratio)
    callus_trace = fig.data[1]
    assert callus_trace.y[0] is None
    assert callus_trace.y[1] == 1.15
    assert callus_trace.connectgaps is False
