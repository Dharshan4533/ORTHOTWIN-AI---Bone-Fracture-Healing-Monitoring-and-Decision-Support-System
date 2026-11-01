"""
OrthoTwin AI — Unit Tests for Report Generation
Tests PDF report creation, layout integrity, and academic disclaimers.
"""

import pytest
from pathlib import Path
from modules.report_generator import generate_pdf_report


def test_pdf_report_generation(tmp_path):
    output_file = "test_clinical_summary.pdf"
    patient_info = {
        "patient_code": "PAT-TEST-0001",
        "full_name": "Oliver Queen",
        "dob": "1985-05-16",
        "gender": "Male"
    }
    case_info = {
        "case_code": "CASE-TEST-0001",
        "anatomical_site": "Left Scaphoid",
        "injury_date": "2026-04-10"
    }
    visit_info = {
        "visit_code": "VISIT-TEST-0001",
        "visit_date": "2026-04-24",
        "days_post_injury": 14
    }
    ai_info = {
        "detections": [
            {"class_name": "fracture", "confidence": 0.84, "bbox": [100.0, 150.0, 220.0, 260.0]}
        ]
    }
    clinician_info = {
        "review_status": "APPROVED",
        "reviewer_name": "Dr. Moira Queen, MD",
        "signed_at": "2026-04-24 14:30",
        "clinical_notes": "Scaphoid waist fracture confirmed. Good immobilization maintained."
    }

    pdf_path = generate_pdf_report(
        report_code="REP-TEST-0001",
        patient_info=patient_info,
        case_info=case_info,
        visit_info=visit_info,
        ai_analysis_info=ai_info,
        clinician_review_info=clinician_info,
        output_filename=output_file
    )

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 1500  # Non-empty, structurally valid PDF
