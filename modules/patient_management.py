"""
OrthoTwin AI — Patient & Fracture Case Management Module
Provides patient registration, duplicate warnings, case history tracking,
and chronological follow-up visit logging with separate future appointments.
"""

from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from datetime import datetime
from database.models import Patient, FractureCase, Visit, User
from database.repositories import (
    PatientRepository, FractureCaseRepository, VisitRepository
)
from services.audit_service import log_audit_event


def register_new_patient(
    db: Session,
    first_name: str,
    last_name: str,
    dob: str,
    gender: str,
    contact_phone: Optional[str] = None,
    medical_history: Optional[str] = None,
    current_user: Optional[User] = None
) -> Tuple[Patient, Optional[str]]:
    """
    Registers a new patient with persistent unique patient code.
    Checks for existing duplicates and returns (patient, duplicate_warning_message).
    """
    # Validation
    if not first_name.strip() or not last_name.strip():
        raise ValueError("First and last names are mandatory.")
    try:
        dob_dt = datetime.strptime(dob.strip(), "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("Date of birth must be formatted as YYYY-MM-DD.")

    today = datetime.now().date()
    if dob_dt > today:
        raise ValueError("Date of birth cannot be in the future.")
    if dob_dt.year < 1900:
        raise ValueError("Date of birth is outside valid range (must be after 1900).")

    duplicate_warning = None
    existing = PatientRepository.find_duplicate(
        db, first_name=first_name, last_name=last_name, dob=dob, phone=contact_phone
    )
    if existing:
        duplicate_warning = (
            f"Notice: A patient record with identical name and date of birth already exists "
            f"({existing.patient_code}: {existing.first_name} {existing.last_name}). "
            f"Please verify this is not an unintended duplicate."
        )

    patient = PatientRepository.create_patient(
        db=db,
        first_name=first_name,
        last_name=last_name,
        dob=dob,
        gender=gender,
        contact_phone=contact_phone,
        medical_history=medical_history
    )

    log_audit_event(
        db=db,
        action="REGISTER_PATIENT",
        user=current_user,
        entity_type="Patient",
        entity_id=patient.patient_code,
        details={"status": "REGISTERED", "gender": gender}
    )

    return patient, duplicate_warning


def create_patient_fracture_case(
    db: Session,
    patient_id: int,
    anatomical_site: str,
    injury_date: str,
    injury_mechanism: Optional[str] = None,
    initial_treatment: Optional[str] = None,
    notes: Optional[str] = None,
    current_user: Optional[User] = None
) -> FractureCase:
    """Creates a new fracture case linked to an existing patient."""
    patient = PatientRepository.get_by_id(db, patient_id)
    if not patient:
        raise ValueError(f"Patient with ID {patient_id} does not exist.")

    try:
        inj_dt = datetime.strptime(injury_date.strip(), "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("Injury date must be formatted as YYYY-MM-DD.")

    today = datetime.now().date()
    if inj_dt > today:
        raise ValueError("Injury date cannot be in the future.")

    case = FractureCaseRepository.create_case(
        db=db,
        patient_id=patient_id,
        anatomical_site=anatomical_site,
        injury_date=injury_date,
        injury_mechanism=injury_mechanism,
        initial_treatment=initial_treatment,
        notes=notes
    )

    log_audit_event(
        db=db,
        action="CREATE_FRACTURE_CASE",
        user=current_user,
        entity_type="FractureCase",
        entity_id=case.case_code,
        details={"anatomical_site": anatomical_site}
    )

    return case


def record_patient_visit(
    db: Session,
    case_id: int,
    visit_date: str,
    clinical_notes: Optional[str] = None,
    weight_bearing_status: Optional[str] = None,
    pain_score: Optional[int] = None,
    scheduled_next_date: Optional[str] = None,
    current_user: Optional[User] = None
) -> Visit:
    """
    Records a clinical follow-up visit.
    Enforces separation between historical visit date and future appointment date.
    Calculates exact elapsed days post injury.
    """
    case = FractureCaseRepository.get_by_id(db, case_id)
    if not case:
        raise ValueError(f"Fracture Case ID {case_id} does not exist.")

    # Validate dates
    try:
        v_dt = datetime.strptime(visit_date.strip(), "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("Visit date must be formatted as YYYY-MM-DD.")

    today = datetime.now().date()
    if v_dt > today:
        raise ValueError("Visit date cannot be in the future.")

    try:
        inj_dt = datetime.strptime(case.injury_date, "%Y-%m-%d").date()
        if v_dt < inj_dt:
            raise ValueError(f"Visit date ({visit_date}) cannot be prior to injury date ({case.injury_date}).")
    except Exception as e:
        if "cannot be prior" in str(e):
            raise

    if pain_score is not None:
        try:
            pain_score_int = int(pain_score)
            if pain_score_int < 0 or pain_score_int > 10:
                raise ValueError("Pain score must be an integer between 0 and 10.")
            pain_score = pain_score_int
        except (TypeError, ValueError):
            raise ValueError("Pain score must be an integer between 0 and 10.")

    if scheduled_next_date:
        try:
            next_dt = datetime.strptime(scheduled_next_date.strip(), "%Y-%m-%d").date()
            if next_dt <= v_dt:
                raise ValueError("Scheduled next appointment date must be in the future relative to this visit.")
        except ValueError as e:
            if "must be in the future" in str(e):
                raise
            raise ValueError("Scheduled next appointment date must be formatted as YYYY-MM-DD.")

    visit = VisitRepository.create_visit(
        db=db,
        case_id=case_id,
        visit_date=visit_date,
        clinical_notes=clinical_notes,
        weight_bearing_status=weight_bearing_status,
        pain_score=pain_score,
        scheduled_next_date=scheduled_next_date
    )

    log_audit_event(
        db=db,
        action="RECORD_VISIT",
        user=current_user,
        entity_type="Visit",
        entity_id=visit.visit_code,
        details={
            "case_code": case.case_code,
            "days_post_injury": visit.days_post_injury
        }
    )

    return visit
