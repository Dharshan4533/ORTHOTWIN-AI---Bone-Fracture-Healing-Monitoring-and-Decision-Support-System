"""
OrthoTwin AI — Unit Tests for Missing Weights & Defensive Degradation
Verifies that missing model weights or corrupted weights do NOT crash the system,
and that an explicit unavailable status is returned.
"""

from pathlib import Path
from modules.fracture_detection import FractureDetector
from modules.fracture_segmentation import SegmentationEngine


def test_missing_detection_weights_graceful_handling(tmp_path):
    non_existent = tmp_path / "does_not_exist.pt"
    detector = FractureDetector(weights_path=str(non_existent))
    assert detector.is_loaded is False
    assert "not found" in detector.load_error.lower()

    # Predict call should fail gracefully without crashing or throwing
    res = detector.predict(tmp_path / "dummy.jpg")
    assert res["success"] is False
    assert "unavailable" in res["error"].lower() or "not found" in res["error"].lower()
    assert res["detections"] == []


def test_missing_segmentation_weights_graceful_handling(tmp_path):
    non_existent = tmp_path / "no_unet.pth"
    engine = SegmentationEngine(weights_path=non_existent)
    assert engine.is_loaded is False

    res = engine.predict_mask(None)
    assert res["available"] is False
    assert res["status"] == "SEGMENTATION_UNAVAILABLE"
    assert "UNAVAILABLE" in res["message"]
