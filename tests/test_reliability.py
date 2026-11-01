"""
OrthoTwin AI — Unit Tests for Reliability Engine & Workflow State Machine
Tests valid transitions, rejection of illegal states, preservation of results on failure,
and structured Reliability Summary computation.
"""

import pytest
from database.models import Patient, FractureCase, Visit, XrayImage, AIAnalysisResult
from database.repositories import (
    PatientRepository, FractureCaseRepository, VisitRepository,
    XrayImageRepository, AIAnalysisRepository
)
from services.reliability_service import (
    transition_status, compute_reliability_summary,
    InvalidStateTransitionError,
    STATUS_UPLOADED, STATUS_READY_FOR_ANALYSIS, STATUS_ANALYSIS_RUNNING,
    STATUS_RESULT_READY, STATUS_ANALYSIS_FAILED, STATUS_CLINICIAN_REVIEWED
)


def test_valid_state_transitions(test_db):
    p = PatientRepository.create_patient(test_db, "Reliability", "Tester", "1995-03-12", "Other")
    c = FractureCaseRepository.create_case(test_db, p.id, "Femur", "2026-06-01")
    v = VisitRepository.create_visit(test_db, c.id, "2026-06-02")
    img = XrayImageRepository.create_image(
        test_db, v.id, "storage/uploads/raw/test.jpg", "test.jpg", 1024, "hash999"
    )
    assert img.status == STATUS_UPLOADED

    # UPLOADED -> READY_FOR_ANALYSIS
    transition_status(test_db, img.id, STATUS_READY_FOR_ANALYSIS)
    assert img.status == STATUS_READY_FOR_ANALYSIS

    # READY_FOR_ANALYSIS -> ANALYSIS_RUNNING
    transition_status(test_db, img.id, STATUS_ANALYSIS_RUNNING)
    assert img.status == STATUS_ANALYSIS_RUNNING

    # ANALYSIS_RUNNING -> RESULT_READY
    transition_status(test_db, img.id, STATUS_RESULT_READY)
    assert img.status == STATUS_RESULT_READY

    # RESULT_READY -> CLINICIAN_REVIEWED
    transition_status(test_db, img.id, STATUS_CLINICIAN_REVIEWED)
    assert img.status == STATUS_CLINICIAN_REVIEWED


def test_illegal_state_transition_rejection(test_db):
    p = PatientRepository.create_patient(test_db, "Illegal", "Transition", "1990-01-01", "Male")
    c = FractureCaseRepository.create_case(test_db, p.id, "Fibula", "2026-06-01")
    v = VisitRepository.create_visit(test_db, c.id, "2026-06-02")
    img = XrayImageRepository.create_image(
        test_db, v.id, "storage/uploads/raw/test.jpg", "test.jpg", 1024, "hash888"
    )

    # Cannot jump directly from UPLOADED to CLINICIAN_REVIEWED
    with pytest.raises(InvalidStateTransitionError, match="Illegal transition"):
        transition_status(test_db, img.id, STATUS_CLINICIAN_REVIEWED)


def test_preservation_of_previous_valid_results_on_failure(test_db):
    p = PatientRepository.create_patient(test_db, "Result", "Preserver", "1987-07-07", "Female")
    c = FractureCaseRepository.create_case(test_db, p.id, "Humerus", "2026-06-01")
    v = VisitRepository.create_visit(test_db, c.id, "2026-06-02")
    img = XrayImageRepository.create_image(
        test_db, v.id, "storage/uploads/raw/test.jpg", "test.jpg", 1024, "hash777",
        status=STATUS_RESULT_READY
    )

    # Previous valid analysis result
    analysis = AIAnalysisRepository.save_analysis(
        test_db, img.id, model_version_id=None,
        detections=[{"bbox": [10, 20, 100, 200], "confidence": 0.88, "class_name": "fracture"}],
        confidence_summary=0.88, raw_scores={"box_confidences": [0.88]}, status=STATUS_RESULT_READY
    )

    # Re-run starts
    transition_status(test_db, img.id, STATUS_ANALYSIS_RUNNING)
    assert img.status == STATUS_ANALYSIS_RUNNING

    # Re-run encounters an exception and fails
    transition_status(test_db, img.id, STATUS_ANALYSIS_FAILED, preserve_existing_on_fail=True)
    assert img.status == STATUS_ANALYSIS_FAILED

    # Check that previous analysis is preserved and not destroyed
    persisted_analysis = AIAnalysisRepository.get_by_image_id(test_db, img.id)
    assert persisted_analysis is not None
    assert persisted_analysis.confidence_summary == 0.88
    assert "fracture" in persisted_analysis.detections_json


def test_reliability_summary_structure():
    summary = compute_reliability_summary(
        validation=None,
        analysis=None,
        visit=None,
        model_tag="v1.0-test"
    )
    assert "technical_quality" in summary
    assert "detection_evaluation" in summary
    assert "missing_clinical_context" in summary
    assert "clinician_review" in summary
    assert "ACADEMIC PROTOTYPE" in summary["disclaimer"]
    assert "score_interpretation_notice" in summary["detection_evaluation"]
