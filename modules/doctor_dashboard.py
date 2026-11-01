"""
OrthoTwin AI — Doctor Dashboard Logic Module
Provides clinician triage queues, scan inspection, review sign-off routines,
and case management summaries.
"""

from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc
from database.models import (
    User, Patient, FractureCase, Visit, XrayImage,
    AIAnalysisResult, ClinicianReview
)
from database.repositories import (
    PatientRepository, FractureCaseRepository, VisitRepository,
    AIAnalysisRepository, XrayImageRepository
)
from services.reliability_service import compute_reliability_summary, transition_status, STATUS_CLINICIAN_REVIEWED
from services.audit_service import log_audit_event


def get_dashboard_metrics(db: Session) -> Dict[str, Any]:
    """Retrieves high-level clinical overview metrics for the doctor dashboard."""
    total_patients = db.query(Patient).count()
    active_cases = db.query(FractureCase).filter(FractureCase.status == "ACTIVE").count()
    
    # Scans awaiting clinician review
    pending_reviews = db.query(AIAnalysisResult).filter(
        AIAnalysisResult.clinician_review == None
    ).count()

    # Completed clinician reviews
    signed_reviews = db.query(ClinicianReview).count()

    return {
        "total_patients": total_patients,
        "active_cases": active_cases,
        "pending_reviews": pending_reviews,
        "signed_reviews": signed_reviews
    }


def get_pending_review_queue(db: Session) -> List[Dict[str, Any]]:
    """Fetches list of AI analysis results awaiting clinical attestation."""
    unreviewed = db.query(AIAnalysisResult).filter(
        AIAnalysisResult.clinician_review == None
    ).order_by(desc(AIAnalysisResult.created_at)).all()

    queue = []
    for a in unreviewed:
        img = a.xray_image
        visit = img.visit if img else None
        case = visit.fracture_case if visit else None
        patient = case.patient if case else None

        queue.append({
            "analysis_id": a.id,
            "image_id": img.id if img else None,
            "patient_code": patient.patient_code if patient else "N/A",
            "patient_name": f"{patient.first_name} {patient.last_name}" if patient else "N/A",
            "case_code": case.case_code if case else "N/A",
            "anatomical_site": case.anatomical_site if case else "N/A",
            "visit_code": visit.visit_code if visit else "N/A",
            "visit_date": visit.visit_date if visit else "N/A",
            "confidence_summary": a.confidence_summary,
            "created_at": a.created_at.strftime("%Y-%m-%d %H:%M") if a.created_at else "N/A"
        })
    return queue


from services.authorization import is_clinician_or_admin, AccessDeniedError


def submit_clinician_attestation(
    db: Session,
    analysis_id: int,
    reviewer_user: User,
    review_status: str,  # APPROVED, MODIFIED, REJECTED
    clinical_notes: str
) -> Tuple[bool, Optional[ClinicianReview], str]:
    """
    Submits a binding clinician sign-off on an AI candidate analysis.
    Transitions scan status to CLINICIAN_REVIEWED.
    Enforces that reviewer_user is a clinician or administrator and ensures atomic execution.
    """
    if not reviewer_user or not is_clinician_or_admin(reviewer_user):
        return False, None, "Access denied: Only licensed clinicians or administrators may submit medical attestations."

    if review_status not in ("APPROVED", "MODIFIED", "REJECTED"):
        return False, None, "Invalid review status. Must be APPROVED, MODIFIED, or REJECTED."

    if not clinical_notes or len(clinical_notes.strip()) < 5:
        return False, None, "Clinical rationale notes must be at least 5 characters long."

    analysis = db.query(AIAnalysisResult).filter(AIAnalysisResult.id == analysis_id).first()
    if not analysis:
        return False, None, f"Analysis record {analysis_id} not found."

    try:
        review = AIAnalysisRepository.save_clinician_review(
            db=db,
            analysis_id=analysis_id,
            reviewer_id=reviewer_user.id,
            review_status=review_status,
            clinical_notes=clinical_notes.strip()
        )

        # Transition image status to CLINICIAN_REVIEWED atomically
        if analysis.xray_image:
            transition_status(db, analysis.xray_image.id, STATUS_CLINICIAN_REVIEWED)

        db.commit()
    except Exception as e:
        db.rollback()
        return False, None, f"Attestation rejected: state transition error ({str(e)})."

    log_audit_event(
        db=db,
        action="SUBMIT_CLINICIAN_REVIEW",
        user=reviewer_user,
        entity_type="ClinicianReview",
        entity_id=str(review.id),
        details={"status": review_status, "analysis_id": analysis_id}
    )

    return True, review, "Clinician review recorded successfully."
