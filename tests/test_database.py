"""
OrthoTwin AI — Unit Tests for Database Persistence & ORM Relationships
Tests schema creation, foreign key constraints, cascading deletions, and query isolation.
"""

import pytest
from database.models import Patient, FractureCase, Visit, XrayImage, AIAnalysisResult, ClinicianReview
from database.repositories import (
    PatientRepository, FractureCaseRepository, VisitRepository,
    XrayImageRepository, AIAnalysisRepository
)


def test_patient_cascade_deletion(test_db):
    p = PatientRepository.create_patient(test_db, "Bruce", "Wayne", "1972-02-19", "Male")
    c = FractureCaseRepository.create_case(test_db, p.id, "Clavicle", "2026-05-01")
    v = VisitRepository.create_visit(test_db, c.id, "2026-05-02")
    img = XrayImageRepository.create_image(
        test_db, v.id, "storage/uploads/raw/test.jpg", "test.jpg", 1024, "dummyhash123"
    )

    assert test_db.query(Patient).filter(Patient.id == p.id).count() == 1
    assert test_db.query(FractureCase).filter(FractureCase.patient_id == p.id).count() == 1
    assert test_db.query(Visit).filter(Visit.case_id == c.id).count() == 1
    assert test_db.query(XrayImage).filter(XrayImage.visit_id == v.id).count() == 1

    # Delete patient -> should cascade to cases, visits, images
    test_db.delete(p)
    test_db.commit()

    assert test_db.query(FractureCase).filter(FractureCase.patient_id == p.id).count() == 0
    assert test_db.query(Visit).filter(Visit.case_id == c.id).count() == 0
    assert test_db.query(XrayImage).filter(XrayImage.visit_id == v.id).count() == 0


def test_unique_codes_generation(test_db):
    p1 = PatientRepository.create_patient(test_db, "Diana", "Prince", "1980-01-01", "Female")
    p2 = PatientRepository.create_patient(test_db, "Clark", "Kent", "1982-02-28", "Male")
    assert p1.patient_code != p2.patient_code
    assert p1.patient_code.startswith("PAT-")
    assert p2.patient_code.startswith("PAT-")
