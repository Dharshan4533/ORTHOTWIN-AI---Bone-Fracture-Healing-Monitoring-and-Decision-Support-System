"""
OrthoTwin AI — Unit Tests for Patient & Case Management
Tests patient registration, duplicate warnings, case tracking, and date validation.
"""

import pytest
from database.models import Patient, FractureCase, Visit
from modules.patient_management import (
    register_new_patient, create_patient_fracture_case, record_patient_visit
)


def test_patient_registration_success(test_db):
    patient, warning = register_new_patient(
        db=test_db,
        first_name="John",
        last_name="Doe",
        dob="1990-05-15",
        gender="Male",
        contact_phone="+1-555-1234"
    )
    assert patient.id is not None
    assert patient.patient_code.startswith("PAT-")
    assert warning is None


def test_patient_duplicate_warning(test_db):
    # Register first patient
    p1, _ = register_new_patient(
        db=test_db,
        first_name="Jane",
        last_name="Smith",
        dob="1985-11-20",
        gender="Female",
        contact_phone="+1-555-5678"
    )
    # Register duplicate with identical name and DOB
    p2, warning = register_new_patient(
        db=test_db,
        first_name="Jane",
        last_name="Smith",
        dob="1985-11-20",
        gender="Female",
        contact_phone="+1-555-5678"
    )
    assert p2.id != p1.id
    assert warning is not None
    assert "identical name and date of birth already exists" in warning


def test_patient_invalid_dob(test_db):
    with pytest.raises(ValueError, match="Date of birth must be formatted"):
        register_new_patient(
            db=test_db,
            first_name="Bad",
            last_name="Date",
            dob="15/05/1990",  # Invalid format
            gender="Other"
        )


def test_fracture_case_and_visit_dates(test_db):
    p, _ = register_new_patient(
        db=test_db, first_name="Tom", last_name="Hardy", dob="1980-01-01", gender="Male"
    )
    case = create_patient_fracture_case(
        db=test_db,
        patient_id=p.id,
        anatomical_site="Right Tibia",
        injury_date="2026-07-01",
        initial_treatment="Cast"
    )
    assert case.case_code.startswith("CASE-")

    # Valid visit 14 days later
    visit = record_patient_visit(
        db=test_db,
        case_id=case.id,
        visit_date="2026-07-15",
        clinical_notes="Two week follow up",
        scheduled_next_date="2026-08-01"
    )
    assert visit.days_post_injury == 14
    assert visit.visit_code.startswith("VISIT-")

    # Invalid visit: visit prior to injury date
    with pytest.raises(ValueError, match="cannot be prior to injury date"):
        record_patient_visit(
            db=test_db,
            case_id=case.id,
            visit_date="2026-06-20"  # Before July 1
        )

    # Invalid appointment: next appointment in the past
    with pytest.raises(ValueError, match="must be in the future"):
        record_patient_visit(
            db=test_db,
            case_id=case.id,
            visit_date="2026-07-20",
            scheduled_next_date="2026-07-10"
        )
