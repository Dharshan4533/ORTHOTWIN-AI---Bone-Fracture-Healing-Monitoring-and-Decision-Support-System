"""
OrthoTwin AI — Fracture Detection Module
Wraps YOLOv8 inference with defensive handling for missing checkpoints,
inference exceptions, and unverified candidate disclaimers.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from database.models import XrayImage, AIAnalysisResult, ModelVersion
from database.repositories import (
    XrayImageRepository, AIAnalysisRepository, ModelVersionRepository
)
from services.reliability_service import (
    transition_status, STATUS_ANALYSIS_RUNNING,
    STATUS_RESULT_READY, STATUS_ANALYSIS_FAILED
)
from services.audit_service import log_audit_event
from config import (
    MODEL_WEIGHTS_PATH, DETECTION_CONF_THRESHOLD,
    DETECTION_IOU_THRESHOLD, ROBOFLOW_CLASSES
)


class FractureDetector:
    """
    Singleton / Cached YOLOv8 Detector wrapper.
    Gracefully handles missing weights without crashing the application.
    """
    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = Path(weights_path or MODEL_WEIGHTS_PATH)
        self.model = None
        self.is_loaded = False
        self._load_attempted = False
        self.load_error: Optional[str] = None
        self._try_load()

    def _try_load(self):
        self._load_attempted = True
        if not self.weights_path.exists():
            self.load_error = f"Model checkpoint not found at: {self.weights_path}"
            self.is_loaded = False
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.weights_path))
            self.is_loaded = True
            self.load_error = None
        except Exception as e:
            self.load_error = f"Failed to initialize YOLO model: {str(e)}"
            self.is_loaded = False

    def predict(
        self,
        image_path: Path,
        conf_threshold: float = DETECTION_CONF_THRESHOLD,
        iou_threshold: float = DETECTION_IOU_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Runs inference on an eligible X-ray image file.
        Returns structured detection output.
        """
        if not self.is_loaded:
            # Re-attempt load in case weights were placed recently
            self._try_load()

        if not self.is_loaded:
            return {
                "success": False,
                "error": self.load_error or "Model weights unavailable.",
                "detections": [],
                "confidence_summary": None,
                "raw_scores": {}
            }

        try:
            results = self.model.predict(
                source=str(image_path),
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            detections: List[Dict[str, Any]] = []
            scores: List[float] = []

            for r in results:
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        xyxy = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_idx = int(box.cls[0])
                        cls_name = r.names.get(cls_idx, f"class_{cls_idx}") if hasattr(r, "names") else "fracture"

                        detections.append({
                            "bbox": [round(c, 2) for c in xyxy],
                            "confidence": round(conf, 4),
                            "class_id": cls_idx,
                            "class_name": cls_name
                        })
                        scores.append(conf)

            conf_summary = round(float(sum(scores) / len(scores)), 4) if scores else None
            return {
                "success": True,
                "error": None,
                "detections": detections,
                "confidence_summary": conf_summary,
                "raw_scores": {"box_confidences": scores, "count": len(detections)}
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Inference execution failed: {str(e)}",
                "detections": [],
                "confidence_summary": None,
                "raw_scores": {}
            }


# Cached global detector instance
_global_detector: Optional[FractureDetector] = None


def get_detector() -> FractureDetector:
    global _global_detector
    if _global_detector is None:
        _global_detector = FractureDetector()
    return _global_detector


def run_xray_analysis(
    db: Session,
    image_id: int,
    conf_threshold: float = DETECTION_CONF_THRESHOLD,
    current_user: Optional[Any] = None
) -> Tuple[bool, Optional[AIAnalysisResult], str]:
    """
    Executes end-to-end fracture analysis for an image:
    1. Enforces transition to ANALYSIS_RUNNING.
    2. Runs inference.
    3. On failure: transitions to ANALYSIS_FAILED while preserving previous valid results.
    4. On success: persists AIAnalysisResult and transitions to RESULT_READY.
    """
    image = XrayImageRepository.get_by_id(db, image_id)
    if not image:
        return False, None, f"Image ID {image_id} not found."

    # Transition state to running
    try:
        transition_status(db, image_id, STATUS_ANALYSIS_RUNNING)
    except Exception as e:
        return False, None, f"State transition rejected: {str(e)}"

    detector = get_detector()
    img_path = Path(image.file_path)
    if not img_path.is_absolute():
        from config import BASE_DIR
        img_path = BASE_DIR / img_path

    inference_result = detector.predict(img_path, conf_threshold=conf_threshold)

    if not inference_result["success"]:
        # Transition to ANALYSIS_FAILED, preserving previous analysis if present
        transition_status(db, image_id, STATUS_ANALYSIS_FAILED, preserve_existing_on_fail=True)
        log_audit_event(
            db=db,
            action="AI_ANALYSIS_FAILED",
            user=current_user,
            entity_type="XrayImage",
            entity_id=str(image_id),
            details={"error": inference_result["error"]}
        )
        return False, None, inference_result["error"]

    import hashlib
    from config import BASE_DIR

    # Resolve exact model version from loaded checkpoint hash
    weights_file = detector.weights_path
    weights_hash = "NO_CHECKPOINT"
    if weights_file.exists() and weights_file.is_file():
        try:
            with open(weights_file, "rb") as f:
                weights_hash = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            weights_hash = "UNREADABLE_CHECKPOINT"

    # Match or register model version record with exact hash
    rel_weights = str(weights_file.relative_to(BASE_DIR)).replace("\\", "/") if weights_file.is_relative_to(BASE_DIR) else str(weights_file).replace("\\", "/")
    model_version = db.query(ModelVersion).filter(
        ModelVersion.weights_path == rel_weights
    ).first()

    if not model_version:
        model_version = ModelVersionRepository.register_version(
            db=db,
            model_name="YOLOv8-FractureDetector",
            version_tag=f"v8n-{weights_hash[:8]}",
            task_type="DETECTION",
            weights_path=rel_weights,
            training_dataset="Roboflow 100 — Bone Fracture",
            parameters={
                "default_conf": conf_threshold,
                "weights_sha256": weights_hash,
                "classes": ROBOFLOW_CLASSES
            }
        )

    model_ver_id = model_version.id if model_version else None

    # Attach immutable inference provenance to raw_scores
    raw_scores_payload = inference_result.get("raw_scores", {})
    raw_scores_payload["model_provenance"] = {
        "model_version_id": model_ver_id,
        "checkpoint_path": rel_weights,
        "checkpoint_sha256": weights_hash,
        "conf_threshold": conf_threshold,
        "classes": list(ROBOFLOW_CLASSES.values())
    }

    # Save results
    analysis = AIAnalysisRepository.save_analysis(
        db=db,
        image_id=image_id,
        model_version_id=model_ver_id,
        detections=inference_result["detections"],
        confidence_summary=inference_result["confidence_summary"],
        raw_scores=raw_scores_payload,
        status=STATUS_RESULT_READY
    )

    # Transition to RESULT_READY
    transition_status(db, image_id, STATUS_RESULT_READY)

    log_audit_event(
        db=db,
        action="AI_ANALYSIS_SUCCESS",
        user=current_user,
        entity_type="AIAnalysisResult",
        entity_id=str(analysis.id),
        details={
            "detection_count": len(inference_result["detections"]),
            "confidence_summary": inference_result["confidence_summary"],
            "model_version": model_version.version_tag if model_version else "unregistered"
        }
    )

    return True, analysis, "Analysis completed successfully."
