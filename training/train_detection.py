"""
OrthoTwin AI — Offline YOLOv8 Fracture Detection Training Pipeline
Unpacks primary Roboflow 100 dataset, configures training parameters,
executes fine-tuning with reproducibility seeds, and registers model version in DB.
"""

import sys
import os
import zipfile
import shutil
from pathlib import Path
import yaml
import torch

# Ensure project root is on sys.path
sys_path = Path(__file__).resolve().parent.parent
if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

import argparse
from typing import Optional
from config import BASE_DIR, DATA_DIR, WEIGHTS_DIR, ROBOFLOW_ZIP_PATH
from database.db import get_db, init_db
from database.repositories import ModelVersionRepository
from training.dataset_validation import audit_roboflow_dataset


def prepare_roboflow_dataset(dest_dir: Path, zip_path: Optional[Path] = None) -> Path:
    """
    Unpacks Roboflow 100 zip if not already extracted, or uses pre-extracted dataset directory.
    Enforces audit verification before training.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    yaml_target = dest_dir / "data.yaml"

    if not yaml_target.exists():
        chosen_zip = Path(zip_path or ROBOFLOW_ZIP_PATH)
        if not chosen_zip.exists():
            raise FileNotFoundError(
                f"Primary dataset not found. Neither extracted dir at {dest_dir} nor archive at {chosen_zip} exists."
            )

        print(f"Running pre-training audit on {chosen_zip.name}...")
        audit_res = audit_roboflow_dataset(chosen_zip)
        shared_val = audit_res.get("train_val_shared_stems", 0)
        shared_test = audit_res.get("train_test_shared_stems", 0)
        if shared_val > 0 or shared_test > 0:
            print(f"[AUDIT WARNING] Data leakage detected: {shared_val} shared base stems between train and val, {shared_test} between train and test.")
            print("[DISCLOSURE] Reported evaluation metrics will be bounded by split augmentation overlap. Patient-level independence cannot be claimed.")

        print(f"Extracting {chosen_zip.name} to {dest_dir}...")
        with zipfile.ZipFile(chosen_zip, "r") as z:
            z.extractall(dest_dir)
        print("Extraction complete.")

    # Rewrite data.yaml with absolute paths for Ultralytics
    with open(yaml_target, "r") as f:
        data_cfg = yaml.safe_load(f)

    data_cfg["train"] = str((dest_dir / "train" / "images").resolve()).replace("\\", "/")
    data_cfg["val"] = str((dest_dir / "valid" / "images").resolve()).replace("\\", "/")
    data_cfg["test"] = str((dest_dir / "test" / "images").resolve()).replace("\\", "/")

    with open(yaml_target, "w") as f:
        yaml.dump(data_cfg, f, default_flow_style=False)

    return yaml_target


def train_yolo(
    epochs: int = 10,
    batch_size: int = 8,
    imgsz: int = 640,
    lr0: float = 0.005,
    seed: int = 42,
    device: str = "cpu"
):
    """Executes YOLOv8 fine-tuning on the primary bone fracture dataset."""
    from ultralytics import YOLO

    dataset_dir = DATA_DIR / "raw" / "roboflow_100"
    data_yaml = prepare_roboflow_dataset(dataset_dir)

    print(f"Loading base YOLOv8n architecture...")
    model = YOLO("yolov8n.pt")

    print(f"Starting training for {epochs} epochs on {device} (seed={seed})...")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        lr0=lr0,
        seed=seed,
        device=device,
        project=str(BASE_DIR / "runs" / "detect"),
        name="fracture_yolov8_run",
        exist_ok=True,
        plots=True,
        verbose=True
    )

    # Copy best weights to models/weights/fracture_yolov8.pt
    best_weights = Path(model.trainer.best)
    target_weights = WEIGHTS_DIR / "fracture_yolov8.pt"

    if best_weights.exists():
        shutil.copy(best_weights, target_weights)
        print(f"Best model checkpoint saved to: {target_weights}")

    # Register in database
    init_db()
    with get_db() as db:
        ModelVersionRepository.register_version(
            db=db,
            model_name="YOLOv8n-BoneFracture",
            version_tag="v1.0.0-roboflow100",
            task_type="DETECTION",
            weights_path=str(target_weights.relative_to(BASE_DIR)).replace("\\", "/"),
            training_dataset="Roboflow 100 — Bone Fracture",
            map_score=getattr(results, "results_dict", {}).get("metrics/mAP50(B)", None),
            parameters={
                "epochs": epochs,
                "batch_size": batch_size,
                "imgsz": imgsz,
                "seed": seed,
                "lr0": lr0
            }
        )
    print("Model version registered in OrthoTwin database successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OrthoTwin AI — YOLOv8 Fracture Detection Training")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=416, help="Image resolution")
    parser.add_argument("--lr0", type=float, default=0.005, help="Initial learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or 0)")
    parser.add_argument("--dataset-zip", type=str, default=None, help="Optional dataset zip path")
    args = parser.parse_args()

    train_yolo(
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        lr0=args.lr0,
        seed=args.seed,
        device=args.device
    )
