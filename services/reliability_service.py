"""
OrthoTwin AI — Reliability & Workflow State Machine Service
Implements backend-enforced status transitions, failure handling with
result preservation, and computes the structured Reliability Summary.
"""

from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from database.models import XrayImage, AIAnalysisResult, Visit, ImageValidationResult
from database.repositories import XrayImageRepository
from config import (
    STATUS_UPLOADED, STATUS_INVALID_FILE, STATUS_NEEDS_REVIEW,
    STATUS_READY_FOR_ANALYSIS, STATUS_ANALYSIS_RUNNING, STATUS_ANALYSIS_FAILED,
    STATUS_RESULT_READY, STATUS_CLINICIAN_REVIEWED,
    BLUR_LAPLACIAN_THRESHOLD, LOW_CONTRAST_THRESHOLD,
    ACADEMIC_PROTOTYPE_DISCLAIMER
)

# Valid state machine transitions
ALLOWED_TRANSITIONS = {
    STATUS_UPLOADED: [STATUS_INVALID_FILE, STATUS_NEEDS_REVIEW, STATUS_READY_FOR_ANALYSIS],
    STATUS_INVALID_FILE: [],  # Terminal state
    STATUS_NEEDS_REVIEW: [STATUS_READY_FOR_ANALYSIS, STATUS_ANALYSIS_RUNNING, STATUS_INVALID_FILE],
    STATUS_READY_FOR_ANALYSIS: [STATUS_ANALYSIS_RUNNING],
    STATUS_ANALYSIS_RUNNING: [STATUS_RESULT_READY, STATUS_ANALYSIS_FAILED],
    STATUS_ANALYSIS_FAILED: [STATUS_READY_FOR_ANALYSIS, STATUS_ANALYSIS_RUNNING],  # Retry permitted
    STATUS_RESULT_READY: [STATUS_CLINICIAN_REVIEWED, STATUS_ANALYSIS_RUNNING],  # Review or re-run
    STATUS_CLINICIAN_REVIEWED: [STATUS_CLINICIAN_REVIEWED, STATUS_ANALYSIS_RUNNING],
}


class InvalidStateTransitionError(Exception):
    """Raised when an illegal workflow state transition is attempted."""
    pass


def transition_status(
    db: Session,
    image_id: int,
    target_status: str,
    preserve_existing_on_fail: bool = True
) -> XrayImage:
    """
    Executes an atomic status transition on an X-ray image record.
    Enforces valid workflow paths and preserves previous valid analyses upon failure.
    """
    image = XrayImageRepository.get_by_id(db, image_id)
    if not image:
        raise ValueError(f"Image ID {image_id} not found.")

    current = image.status
    allowed = ALLOWED_TRANSITIONS.get(current, [])

    if target_status not in allowed:
        raise InvalidStateTransitionError(
            f"Illegal transition: '{current}' -> '{target_status}'. "
            f"Permitted next states: {allowed}"
        )

    # When failing an analysis, ensure we do NOT destroy any previously successful analysis
    if target_status == STATUS_ANALYSIS_FAILED and preserve_existing_on_fail:
        # Keep existing AIAnalysisResult untouched if it was already RESULT_READY or CLINICIAN_REVIEWED
        pass

    image.status = target_status
    db.commit()
    db.refresh(image)
    return image


def compute_reliability_summary(
    validation: Optional[ImageValidationResult],
    analysis: Optional[AIAnalysisResult],
    visit: Optional[Visit],
    model_tag: Optional[str] = None
) -> Dict[str, Any]:
    """
    Computes a multi-component Reliability Summary.
    Separates:
      1. Image Technical Quality
      2. Detector Score & Detection Count
      3. Missing Clinical Context
      4. Comparability & Technical Warnings
      5. Review Status & Prototype Limitations

    NEVER aggregates raw detector scores into a single pseudo-diagnostic probability.
    """
    # 1. Technical Image Quality
    quality_warnings: List[str] = []
    quality_status = "UNKNOWN"
    blur_val = 0.0
    contrast_val = 0.0

    if validation:
        blur_val = validation.blur_score
        contrast_val = validation.contrast_score
        if validation.blur_score < BLUR_LAPLACIAN_THRESHOLD:
            quality_warnings.append(
                f"Image sharpness below threshold ({validation.blur_score:.1f} < {BLUR_LAPLACIAN_THRESHOLD:.1f}). Possible motion blur or low resolution."
            )
        if validation.contrast_score < LOW_CONTRAST_THRESHOLD:
            quality_warnings.append(
                f"Low dynamic contrast ({validation.contrast_score:.1f} < {LOW_CONTRAST_THRESHOLD:.1f}). Lesions may have poor visibility."
            )
        quality_status = "ACCEPTABLE" if not quality_warnings else "DEGRADED"
        if not validation.is_valid:
            quality_status = "UNACCEPTABLE"

    # 2. Detector Model Scores
    detection_status = "UNANALYZED"
    conf_score = None
    num_detections = 0
    if analysis and analysis.status in (STATUS_RESULT_READY, STATUS_CLINICIAN_REVIEWED):
        conf_score = analysis.confidence_summary
        import json
        try:
            boxes = json.loads(analysis.detections_json)
            num_detections = len(boxes)
        except Exception:
            boxes = []

        if num_detections == 0:
            detection_status = "NO_FRACTURE_CANDIDATE_DETECTED"
        elif conf_score is not None:
            if conf_score >= 0.70:
                detection_status = "HIGH_CONFIDENCE_DETECTION"
            elif conf_score >= 0.40:
                detection_status = "MODERATE_CONFIDENCE_DETECTION"
            else:
                detection_status = "UNCERTAIN_LOW_CONFIDENCE"
                quality_warnings.append("Model confidence is borderline (< 0.40). High risk of false positive.")

    # 3. Missing Clinical Information
    missing_info: List[str] = []
    if visit:
        if not visit.clinical_notes or len(visit.clinical_notes.strip()) < 5:
            missing_info.append("Clinical physical examination notes are missing or incomplete.")
        if visit.pain_score is None:
            missing_info.append("Patient pain scale score (0-10) was not recorded for this visit.")
        if not visit.weight_bearing_status:
            missing_info.append("Weight-bearing status is unspecified.")

    # 4. Clinician Review Status
    review_status = "PENDING_REVIEW"
    reviewer_notes = None
    if analysis and analysis.clinician_review:
        review_status = analysis.clinician_review.review_status
        reviewer_notes = analysis.clinician_review.clinical_notes

    return {
        "technical_quality": {
            "status": quality_status,
            "blur_laplacian": round(blur_val, 2),
            "contrast_std": round(contrast_val, 2),
            "warnings": quality_warnings
        },
        "detection_evaluation": {
            "status": detection_status,
            "candidate_count": num_detections,
            "raw_confidence_score": round(conf_score, 3) if conf_score is not None else None,
            "model_version": model_tag or "unspecified",
            "score_interpretation_notice": "Raw detector confidence is a localized bounding-box metric, NOT a verified clinical diagnostic probability."
        },
        "missing_clinical_context": missing_info,
        "clinician_review": {
            "status": review_status,
            "notes": reviewer_notes or "Awaiting qualified orthopedic specialist review."
        },
        "disclaimer": ACADEMIC_PROTOTYPE_DISCLAIMER
    }
