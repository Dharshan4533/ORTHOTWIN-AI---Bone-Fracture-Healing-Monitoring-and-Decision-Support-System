# OrthoTwin AI — Model Training Documentation

This directory contains standalone offline training and validation pipelines for OrthoTwin AI. Training is decoupled from runtime application inference.

---

## 1. Datasets Supported

- **Roboflow 100 — Bone Fracture** (`bone fracture.v1-release-640.yolov8.zip`):
  - Primary dataset for YOLOv8 fine-tuning.
  - Standardized 640×640 letterbox images.
  - Classes: `angle`, `fracture`, `line`, `messed_up_angle`.
- **FracAtlas** (`FracAtlas.zip`):
  - Secondary dataset for cross-dataset evaluation and segmentation.
  - 719 fractured scans with YOLO annotations and COCO polygon masks.
  - 3,366 non-fractured scans for negative control testing.

---

## 2. Dataset Auditing Protocol

Always audit datasets prior to training:
```bash
python training/dataset_validation.py
```
This generates `data/DATASET_AUDIT_REPORT.md` reporting exact counts, split overlaps, and potential augmentations leakage.

---

## 3. YOLOv8 Fracture Detection Fine-Tuning

To train the YOLOv8 detector offline on local CPU or GPU:
```bash
python training/train_detection.py
```
Key parameters:
- `epochs`: 10 to 50
- `batch_size`: 8 or 16
- `imgsz`: 640
- `seed`: 42 (ensures reproducibility)

The best checkpoint is automatically saved to `models/weights/fracture_yolov8.pt` and registered in the OrthoTwin SQLite database.

---

## 4. Evaluation on Held-Out Test Split

To evaluate the trained checkpoint without data contamination:
```bash
python training/evaluate_detection.py
```
Reports Precision, Recall, mAP@50, and mAP@50-95 in `models/evaluation_report.json`.

---

## 5. U-Net Segmentation Pipeline

To inspect and validate polygon mask rasterization:
```bash
python training/train_segmentation.py
```

---

## 6. Longitudinal Prediction Safety Gate

Prediction models are strictly gated:
```bash
python training/train_prediction.py
```
Documented why longitudinal outcome prediction cannot run without clinical cohorts and ground-truth union labels.
