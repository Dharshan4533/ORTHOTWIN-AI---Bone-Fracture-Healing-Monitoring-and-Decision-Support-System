"""
OrthoTwin AI — Unit Tests for Authorization & Patient Record Isolation
Tests role-based access control and guarantees Patient A cannot access Patient B's data.
"""

import pytest
from database.models import User, Patient, FractureCase
from database.repositories import PatientRepository, FractureCaseRepository
from services.authorization import (
    can_access_patient, filter_accessible_patients, is_clinician_or_admin,
    AccessDeniedError, require_role
)
from services.authentication import verify_password, hash_password
from config import ROLE_ADMIN, ROLE_DOCTOR, ROLE_PATIENT


def test_password_hashing_and_verification():
    pwd = "StrongSecurePassword2026!"
    h, s = hash_password(pwd)
    assert verify_password(pwd, h, s) is True
    assert verify_password("WrongPassword", h, s) is False


def test_patient_record_isolation(test_db):
    # Create two patients
    p1 = PatientRepository.create_patient(test_db, "Patient", "One", "1990-01-01", "Male")
    p2 = PatientRepository.create_patient(test_db, "Patient", "Two", "1992-02-02", "Female")

    # Patient 1 user account
    u_pat1 = test_db.query(User).filter(User.username == "patient_test").first()
    u_pat1.patient_id = p1.id
    test_db.commit()

    # Clinician user account
    u_doc = test_db.query(User).filter(User.username == "doctor_test").first()
    u_admin = test_db.query(User).filter(User.username == "admin_test").first()

    # Clinician & Admin can access both patients
    assert can_access_patient(u_doc, p1.id) is True
    assert can_access_patient(u_doc, p2.id) is True
    assert can_access_patient(u_admin, p1.id) is True
    assert can_access_patient(u_admin, p2.id) is True

    # Patient 1 can access Patient 1
    assert can_access_patient(u_pat1, p1.id) is True

    # Patient 1 CANNOT access Patient 2 (CRITICAL PRIVACY SAFEGUARD)
    assert can_access_patient(u_pat1, p2.id) is False

    # Scoped query check: Patient 1 only sees Patient 1
    accessible = filter_accessible_patients(test_db, u_pat1)
    assert len(accessible) == 1
    assert accessible[0].id == p1.id

    # Clinician sees all patients
    accessible_doc = filter_accessible_patients(test_db, u_doc)
    assert len(accessible_doc) >= 2


def test_require_role_enforcement(test_db):
    u_pat = test_db.query(User).filter(User.username == "patient_test").first()
    with pytest.raises(AccessDeniedError, match="requires 'doctor' privilege"):
        require_role(u_pat, ROLE_DOCTOR)
