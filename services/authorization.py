"""
OrthoTwin AI — Role-Based Authorization & Patient Data Isolation Service
Strictly prevents patients from accessing other patients' clinical records.
"""

from typing import List, Optional
from sqlalchemy.orm import Session
from database.models import User, Patient, FractureCase, Visit
from config import ROLE_ADMIN, ROLE_DOCTOR, ROLE_PATIENT


class AccessDeniedError(Exception):
    """Raised when an operation violates role-based permissions or patient data isolation."""
    pass


def has_role(user: Optional[User], required_role: str) -> bool:
    """Checks if the user has the specified role."""
    if not user or not user.role:
        return False
    return user.role.name == required_role


def is_clinician_or_admin(user: Optional[User]) -> bool:
    """Checks if the user is authorized to perform clinical reviews and manage cases."""
    if not user or not user.role:
        return False
    return user.role.name in (ROLE_ADMIN, ROLE_DOCTOR)


def can_access_patient(user: Optional[User], patient_id: int) -> bool:
    """
    Enforces strict patient record isolation:
    - Admin and Doctor can view all patient records.
    - Patient can ONLY view their own patient profile and approved records.
    """
    if not user or not user.role:
        return False
    if is_clinician_or_admin(user):
        return True
    if user.role.name == ROLE_PATIENT:
        return user.patient_id is not None and user.patient_id == patient_id
    return False


def can_access_case(db: Session, user: Optional[User], case_id: int) -> bool:
    """Verifies user authorization to access a specific fracture case."""
    if not user:
        return False
    if is_clinician_or_admin(user):
        return True
    case = db.query(FractureCase).filter(FractureCase.id == case_id).first()
    if not case:
        return False
    return can_access_patient(user, case.patient_id)


def can_access_visit(db: Session, user: Optional[User], visit_id: int) -> bool:
    """Verifies user authorization to access a specific visit."""
    if not user:
        return False
    if is_clinician_or_admin(user):
        return True
    visit = db.query(Visit).filter(Visit.id == visit_id).first()
    if not visit or not visit.fracture_case:
        return False
    return can_access_patient(user, visit.fracture_case.patient_id)


def can_access_image(db: Session, user: Optional[User], image_id: int) -> bool:
    """Verifies user authorization to access an X-ray image."""
    if not user:
        return False
    if is_clinician_or_admin(user):
        return True
    from database.models import XrayImage
    img = db.query(XrayImage).filter(XrayImage.id == image_id).first()
    if not img or not img.visit or not img.visit.fracture_case:
        return False
    return can_access_patient(user, img.visit.fracture_case.patient_id)


def can_access_report(db: Session, user: Optional[User], report) -> bool:
    """
    Verifies user authorization to access a clinical report.
    For patients: report must belong to them, AND must be marked is_final,
    AND associated with an approved/modified clinician review (never unreviewed or rejected).
    """
    if not user or not report:
        return False
    if is_clinician_or_admin(user):
        return True
    if user.role.name == ROLE_PATIENT:
        if not report.visit or not report.visit.fracture_case:
            return False
        if not can_access_patient(user, report.visit.fracture_case.patient_id):
            return False
        # Clinical safety gate: patient cannot see non-final or rejected reports
        if not report.is_final:
            return False
        # Check if visit has any approved or modified review
        has_approved = False
        for img in report.visit.xray_images:
            if img.ai_analysis and img.ai_analysis.clinician_review:
                if img.ai_analysis.clinician_review.review_status in ("APPROVED", "MODIFIED"):
                    has_approved = True
                    break
        return has_approved
    return False


def filter_accessible_patients(db: Session, user: Optional[User]) -> List[Patient]:
    """
    Returns only the patient records the current user is legally permitted to access.
    Guarantees that a patient user never sees other patients in lists or search results.
    """
    if not user or not user.role:
        return []
    if is_clinician_or_admin(user):
        return db.query(Patient).all()
    if user.role.name == ROLE_PATIENT and user.patient_id:
        p = db.query(Patient).filter(Patient.id == user.patient_id).first()
        return [p] if p else []
    return []


def require_role(user: Optional[User], *required_roles: str):
    """Enforces role check or raises AccessDeniedError."""
    if not user or not user.role or user.role.name not in required_roles:
        if len(required_roles) == 1:
            raise AccessDeniedError(f"Operation requires '{required_roles[0]}' privilege.")
        roles_str = ", ".join(required_roles)
        raise AccessDeniedError(f"Operation requires privileged role [{roles_str}].")

