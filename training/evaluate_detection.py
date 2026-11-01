"""
OrthoTwin AI — YOLOv8 Evaluation Pipeline
Evaluates trained checkpoint on held-out test split.
Does NOT tune hyperparameters on test data.
Exports empirical evaluation metrics to models/evaluation_report.json.
"""

import sys
import json
from pathlib import Path

# Ensure project root is on sys.path
sys_path = Path(__file__).resolve().parent.parent
if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

from config import BASE_DIR, DATA_DIR, WEIGHTS_DIR, MODELS_DIR


def evaluate_checkpoint(
    weights_path: Path,
    data_yaml: Path
) -> dict:
    """Evaluates checkpoint on test partition and records metrics."""
    from ultralytics import YOLO

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_yaml}")

    print(f"Evaluating {weights_path.name} against held-out test split...")
    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        device="cpu",
        verbose=True
    )

    results = {
        "model_weights": weights_path.name,
        "evaluation_split": "test",
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "class_names": metrics.names,
        "class_mAP50": [float(c) for c in metrics.box.maps] if hasattr(metrics.box, "maps") else []
    }

    report_path = MODELS_DIR / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation report written to {report_path}")
    return results


if __name__ == "__main__":
    w_path = WEIGHTS_DIR / "fracture_yolov8.pt"
    d_yaml = DATA_DIR / "raw" / "roboflow_100" / "data.yaml"
    if w_path.exists() and d_yaml.exists():
        evaluate_checkpoint(w_path, d_yaml)
    else:
        print("Model weights or dataset YAML not found. Run training/train_detection.py first.")
