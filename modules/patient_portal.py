"""
OrthoTwin AI — Patient Portal Logic Module
Provides patient-specific view-only access to approved clinical summaries,
recovery timelines, and simulated appointment reminders.
Strictly isolates records to prevent cross-patient data leakage.
"""

from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from database.models import User, Patient, FractureCase, Visit, Report
from services.authorization import can_access_patient, AccessDeniedError
from config import ROLE_PATIENT


def get_patient_portal_data(
    db: Session,
    patient_user: User
) -> Dict[str, Any]:
    """
    Fetches strictly scoped patient portal records for the authenticated patient user.
    Prevents patient from viewing unapproved raw AI detections or other patients' files.
    """
    if not patient_user or patient_user.role.name != ROLE_PATIENT:
        raise AccessDeniedError("Access restricted to authenticated patient accounts.")

    if not patient_user.patient_id:
        return {
            "has_profile": False,
            "message": "No clinical patient record is associated with this user account.",
            "patient": None,
            "cases": [],
            "simulated_notifications": []
        }

    patient = db.query(Patient).filter(Patient.id == patient_user.patient_id).first()
    if not patient:
        return {
            "has_profile": False,
            "message": "Patient profile record could not be found.",
            "patient": None,
            "cases": [],
            "simulated_notifications": []
        }

    # Extract fracture cases and visits
    cases_summary: List[Dict[str, Any]] = []
    simulated_notifications: List[Dict[str, Any]] = []

    for c in patient.fracture_cases:
        visits_summary: List[Dict[str, Any]] = []
        for v in sorted(c.visits, key=lambda x: x.visit_date):
            # Check for approved reports: Must be marked is_final AND have verified clinician approval
            approved_reports = []
            
            # Check if visit has an authorized clinician review that is APPROVED or MODIFIED
            visit_has_clinician_approval = False
            for img in v.xray_images:
                if img.ai_analysis and img.ai_analysis.clinician_review:
                    if img.ai_analysis.clinician_review.review_status in ("APPROVED", "MODIFIED"):
                        visit_has_clinician_approval = True
                        break

            for r in v.reports:
                if r.is_final and visit_has_clinician_approval:
                    approved_reports.append({
                        "report_code": r.report_code,
                        "created_at": r.created_at.strftime("%Y-%m-%d"),
                        "pdf_path": r.pdf_path
                    })

            visits_summary.append({
                "visit_code": v.visit_code,
                "visit_date": v.visit_date,
                "days_post_injury": v.days_post_injury,
                "weight_bearing_status": v.weight_bearing_status or "Standard precaution",
                "has_approved_report": len(approved_reports) > 0,
                "reports": approved_reports
            })

            # Check if there is a scheduled future appointment
            if v.scheduled_next_date:
                simulated_notifications.append({
                    "type": "UPCOMING_APPOINTMENT",
                    "label": "SIMULATED DEMO NOTIFICATION",
                    "case_code": c.case_code,
                    "scheduled_date": v.scheduled_next_date,
                    "message": f"Simulated Reminder: You have an upcoming orthopedic follow-up scheduled on {v.scheduled_next_date} for your {c.anatomical_site} fracture."
                })

        cases_summary.append({
            "case_code": c.case_code,
            "anatomical_site": c.anatomical_site,
            "injury_date": c.injury_date,
            "status": c.status,
            "visits": visits_summary
        })

    return {
        "has_profile": True,
        "patient": {
            "patient_code": patient.patient_code,
            "full_name": f"{patient.first_name} {patient.last_name}",
            "dob": patient.dob,
            "gender": patient.gender
        },
        "cases": cases_summary,
        "simulated_notifications": simulated_notifications
    }
