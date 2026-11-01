"""
OrthoTwin AI — Complete Database & Storage Reset Utility
Clears all patient records, fracture cases, visits, radiographs, AI analyses,
and generated PDF reports for a 100% clean-slate evaluation.
Preserves trained model weights (fracture_yolov8.pt) and re-seeds clean login accounts.
"""

import os
import sys
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from database.db import init_db, get_db
from database.models import (
    Role, User, Patient, FractureCase, Visit, XrayImage,
    ImageValidationResult, ExtractedFeature, AIAnalysisResult,
    ClinicianReview, Report, AuditLog, ModelVersion
)
from database.repositories import UserRepository, ModelVersionRepository
from services.authentication import create_development_user
from config import (
    ROLE_DOCTOR, ROLE_ADMIN, ROLE_PATIENT,
    STORAGE_RAW_DIR, STORAGE_PROCESSED_DIR, STORAGE_REPORTS_DIR,
    DEFAULT_WEIGHTS_FILE
)


def clean_directory(dir_path: Path):
    """Removes all files in a directory while keeping directory structure and .gitkeep."""
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        return
    for item in dir_path.iterdir():
        if item.name == ".gitkeep":
            continue
        try:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except Exception as e:
            print(f"  [Warning] Could not remove {item.name}: {e}")


def reset_to_clean_slate():
    print("\n=======================================================")
    print(" OrthoTwin AI — Clean Slate System Reset               ")
    print("=======================================================\n")

    init_db()

    with get_db() as db:
        print("[1/4] Clearing patient data, visits, and clinical records...")
        # Clear child tables first for relational integrity
        db.query(Report).delete()
        db.query(ClinicianReview).delete()
        db.query(AIAnalysisResult).delete()
        db.query(ExtractedFeature).delete()
        db.query(ImageValidationResult).delete()
        db.query(XrayImage).delete()
        db.query(Visit).delete()
        db.query(FractureCase).delete()
        db.query(AuditLog).delete()

        # Unlink patient_id from users before deleting patients
        for u in db.query(User).all():
            u.patient_id = None
        db.commit()

        db.query(Patient).delete()
        db.query(User).delete()
        db.commit()
        print("      [OK] Cleared all clinical records and patient data.")

        print("\n[2/4] Re-creating standard development accounts...")
        UserRepository.get_or_create_roles(db)
        
        defaults = [
            ("dr_ortho", "DoctorSecure2026!", ROLE_DOCTOR, "Dr. Marcus Vance, MD", "dr.vance@orthotwin.local"),
            ("admin_user", "AdminSecure2026!", ROLE_ADMIN, "System Administrator", "admin@orthotwin.local"),
            ("patient_demo", "PatientSecure2026!", ROLE_PATIENT, "Demo Patient User", "patient@orthotwin.local")
        ]
        for uname, pwd, role, fname, email in defaults:
            create_development_user(
                db=db,
                username=uname,
                password=pwd,
                role_name=role,
                full_name=fname,
                email=email,
                patient_id=None
            )
            print(f"      [OK] Created account: {uname:<14} (Role: {role})")

        print("\n[3/4] Registering trained YOLOv8 model version in database...")
        if DEFAULT_WEIGHTS_FILE.exists():
            ModelVersionRepository.register_version(
                db=db,
                model_name="YOLOv8n-BoneFracture",
                version_tag="v1.0.0-roboflow100",
                task_type="DETECTION",
                weights_path=str(DEFAULT_WEIGHTS_FILE.relative_to(BASE_DIR)).replace("\\", "/"),
                training_dataset="Roboflow 100 — Bone Fracture",
                map_score=0.0244,
                parameters={"epochs": 3, "batch_size": 8, "imgsz": 416}
            )
            print("      [OK] ModelVersion registered for fracture_yolov8.pt")
        else:
            print("      [Notice] models/weights/fracture_yolov8.pt not found.")

    print("\n[4/4] Purging uploaded radiographs and generated PDF reports...")
    clean_directory(STORAGE_RAW_DIR)
    clean_directory(STORAGE_PROCESSED_DIR)
    clean_directory(STORAGE_REPORTS_DIR)
    print("      [OK] Storage uploads and reports directories cleared.")

    print("\n=======================================================")
    print(" System successfully reset to clean slate!             ")
    print("                                                       ")
    print(" Current Database Status:                              ")
    print("  • Registered Patients:   0                           ")
    print("  • Active Cases:          0                           ")
    print("  • Scans Awaiting Review: 0 (Queue is EMPTY)          ")
    print("  • Clinician Sign-Offs:   0                           ")
    print("                                                       ")
    print(" Logins ready:                                         ")
    print("  • Doctor:  dr_ortho      / DoctorSecure2026!         ")
    print("  • Patient: patient_demo  / PatientSecure2026!        ")
    print("  • Admin:   admin_user    / AdminSecure2026!          ")
    print("=======================================================\n")


if __name__ == "__main__":
    reset_to_clean_slate()
