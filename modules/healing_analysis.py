"""
OrthoTwin AI — Longitudinal Healing Analysis Module
Compares historical visits, computes elapsed timelines, and tracks observed changes
in experimental image features without predicting future clinical outcomes.
"""

from typing import Dict, Any, List, Optional
from database.models import FractureCase, Visit
from config import ACADEMIC_PROTOTYPE_DISCLAIMER


def compute_historical_healing_progression(
    fracture_case: FractureCase
) -> Dict[str, Any]:
    """
    Analyzes historical follow-up visits in chronological order.
    Calculates change in experimental gap proxy and reported symptoms over time.
    """
    visits = sorted(fracture_case.visits, key=lambda v: v.visit_date)
    if not visits:
        return {
            "has_data": False,
            "message": "No visits recorded for this fracture case.",
            "timeline": []
        }

    timeline: List[Dict[str, Any]] = []
    baseline_gap: Optional[float] = None
    baseline_density: Optional[float] = None

    for idx, v in enumerate(visits):
        gap_val = None
        density_val = None

        # Inspect extracted features from images in this visit
        for img in v.xray_images:
            if img.extracted_features:
                gap_val = img.extracted_features.fracture_gap_px
                density_val = img.extracted_features.callus_density_ratio
                break

        if baseline_gap is None and gap_val is not None:
            baseline_gap = gap_val
        if baseline_density is None and density_val is not None:
            baseline_density = density_val

        # Compute relative change if baseline available
        gap_change_pct = None
        if baseline_gap is not None and gap_val is not None and baseline_gap > 0:
            gap_change_pct = round(((baseline_gap - gap_val) / baseline_gap) * 100.0, 1)

        density_change_pct = None
        if baseline_density is not None and density_val is not None and baseline_density > 0:
            density_change_pct = round(((density_val - baseline_density) / baseline_density) * 100.0, 1)

        img_path = v.xray_images[0].file_path if v.xray_images else None
        img_id = v.xray_images[0].id if v.xray_images else None

        timeline.append({
            "visit_code": v.visit_code,
            "visit_date": v.visit_date,
            "days_post_injury": v.days_post_injury,
            "pain_score": v.pain_score,
            "weight_bearing_status": v.weight_bearing_status or "Not specified",
            "clinical_notes": v.clinical_notes or "No notes",
            "fracture_gap_px": gap_val,
            "gap_reduction_pct": gap_change_pct,
            "callus_density_ratio": density_val,
            "density_change_pct": density_change_pct,
            "scheduled_next_date": v.scheduled_next_date,
            "image_path": img_path,
            "image_id": img_id
        })

    baseline_visit = timeline[0] if timeline else None
    latest_visit = timeline[-1] if timeline else None

    pain_delta = None
    if baseline_visit and latest_visit:
        b_pain = baseline_visit.get("pain_score")
        l_pain = latest_visit.get("pain_score")
        if b_pain is not None and l_pain is not None:
            pain_delta = l_pain - b_pain

    latest_days = latest_visit["days_post_injury"] if latest_visit else 0
    if latest_days <= 5:
        bio_phase = "Phase 1: Inflammatory & Hematoma (Days 0–5)"
        bio_desc = "Initial cellular response, hematoma formation, and osteoprogenitor cell recruitment."
        bio_progress = 0.15
    elif latest_days <= 21:
        bio_phase = "Phase 2: Reparative / Soft Callus (Days 5–21)"
        bio_desc = "Cartilage bridge formation, fibrocartilage deposition, and initial fragment stabilization."
        bio_progress = 0.40
    elif latest_days <= 60:
        bio_phase = "Phase 3: Hard Callus / Bony Consolidation (Weeks 3–8)"
        bio_desc = "Endochondral ossification, woven bone conversion, and progressive cortical continuity."
        bio_progress = 0.75
    else:
        bio_phase = "Phase 4: Cortical Remodeling (Months 2+)"
        bio_desc = "Lamellar bone replacement, medullary canal restoration, and structural contouring."
        bio_progress = 0.95

    return {
        "has_data": True,
        "case_code": fracture_case.case_code,
        "anatomical_site": fracture_case.anatomical_site,
        "injury_date": fracture_case.injury_date,
        "visit_count": len(visits),
        "timeline": timeline,
        "summary": {
            "baseline_visit": baseline_visit,
            "latest_visit": latest_visit,
            "pain_delta": pain_delta,
            "latest_days": latest_days,
            "biological_phase": bio_phase,
            "biological_phase_description": bio_desc,
            "biological_progress": bio_progress
        },
        "safety_notice": (
            "HISTORICAL OBSERVATIONS ONLY: Trend metrics show observed changes between historical visits. "
            "They do not project future healing time, nor do they guarantee anatomical union."
        ),
        "disclaimer": ACADEMIC_PROTOTYPE_DISCLAIMER
    }
