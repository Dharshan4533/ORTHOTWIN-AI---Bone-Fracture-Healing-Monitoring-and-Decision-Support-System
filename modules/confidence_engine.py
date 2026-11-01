"""
OrthoTwin AI — Confidence & Reliability Engine
Analyzes technical input quality, model uncertainty, temporal contradictions,
and missing clinical data.
Explains all warning flags and enforces prototype reliability boundaries.
"""

from typing import Dict, Any, List, Optional
from database.models import ImageValidationResult, AIAnalysisResult, Visit
from config import (
    BLUR_LAPLACIAN_THRESHOLD, LOW_CONTRAST_THRESHOLD,
    DETECTION_CONF_THRESHOLD, ACADEMIC_PROTOTYPE_DISCLAIMER
)


def evaluate_case_confidence(
    validation: Optional[ImageValidationResult],
    analysis: Optional[AIAnalysisResult],
    current_visit: Optional[Visit],
    previous_visits: Optional[List[Visit]] = None
) -> Dict[str, Any]:
    """
    Evaluates multi-factor confidence and flags clinical/technical risks:
    1. Input Image Quality degradation flags
    2. Model confidence & uncertainty zones (0.25 - 0.50)
    3. Longitudinal inconsistency flags
    4. Missing clinical examination data
    """
    warning_details: List[Dict[str, str]] = []
    uncertainty_level = "LOW"

    # 1. Technical Image Input Quality
    if validation:
        if validation.blur_score < BLUR_LAPLACIAN_THRESHOLD:
            warning_details.append({
                "category": "IMAGE_QUALITY",
                "severity": "HIGH",
                "code": "HIGH_BLUR",
                "message": f"Blur index ({validation.blur_score:.1f}) is below standard threshold ({BLUR_LAPLACIAN_THRESHOLD:.1f}). Fine microfracture lines may be masked."
            })
            uncertainty_level = "ELEVATED"

        if validation.contrast_score < LOW_CONTRAST_THRESHOLD:
            warning_details.append({
                "category": "IMAGE_QUALITY",
                "severity": "MEDIUM",
                "code": "LOW_CONTRAST",
                "message": f"Low contrast ({validation.contrast_score:.1f}). Trabecular bone texture differentiation is reduced."
            })
            uncertainty_level = "ELEVATED"

    # 2. Model Detection Confidence & Uncertainty
    if analysis:
        conf = analysis.confidence_summary
        if conf is not None:
            if conf < 0.40:
                warning_details.append({
                    "category": "MODEL_UNCERTAINTY",
                    "severity": "HIGH",
                    "code": "BORDERLINE_CONFIDENCE",
                    "message": f"Model confidence ({conf:.2f}) falls in the uncertain zone (0.25 - 0.40). Risk of false positive or artifact detection."
                })
                uncertainty_level = "HIGH"
            elif conf < 0.60:
                warning_details.append({
                    "category": "MODEL_UNCERTAINTY",
                    "severity": "MEDIUM",
                    "code": "MODERATE_CONFIDENCE",
                    "message": f"Model confidence ({conf:.2f}) is moderate. Requires radiologist confirmation."
                })
                if uncertainty_level == "LOW":
                    uncertainty_level = "MODERATE"

    # 3. Longitudinal Consistency & Disagreements
    if previous_visits and len(previous_visits) > 0:
        prev_analyses = []
        for pv in previous_visits:
            for img in pv.xray_images:
                if img.ai_analysis:
                    prev_analyses.append(img.ai_analysis)

        if prev_analyses and analysis:
            import json
            curr_boxes = json.loads(analysis.detections_json) if analysis.detections_json else []
            prev_boxes = json.loads(prev_analyses[-1].detections_json) if prev_analyses[-1].detections_json else []

            # If previous scan had no fracture, but current visit has high confidence fracture (late onset or new trauma)
            if len(prev_boxes) == 0 and len(curr_boxes) > 0:
                warning_details.append({
                    "category": "TEMPORAL_DISAGREEMENT",
                    "severity": "MEDIUM",
                    "code": "NEW_FINDING_POST_BASELINE",
                    "message": "Fracture candidate detected in current visit that was not flagged in the immediate prior scan. Verify scan alignment or secondary displacement."
                })

    # 4. Missing Clinical Data
    if current_visit:
        if not current_visit.clinical_notes:
            warning_details.append({
                "category": "MISSING_DATA",
                "severity": "LOW",
                "code": "NO_CLINICAL_NOTES",
                "message": "No clinical physical exam notes logged for this visit."
            })
        if current_visit.pain_score is None:
            warning_details.append({
                "category": "MISSING_DATA",
                "severity": "LOW",
                "code": "NO_PAIN_SCORE",
                "message": "Visual analog scale pain score was omitted."
            })

    return {
        "uncertainty_level": uncertainty_level,
        "warning_count": len(warning_details),
        "warnings": warning_details,
        "calibration_notice": (
            "NOTICE: Raw detector bounding box scores reflect localized feature activation. "
            "They are NOT calibrated clinical diagnostic probabilities and should never be interpreted as such."
        ),
        "disclaimer": ACADEMIC_PROTOTYPE_DISCLAIMER
    }
