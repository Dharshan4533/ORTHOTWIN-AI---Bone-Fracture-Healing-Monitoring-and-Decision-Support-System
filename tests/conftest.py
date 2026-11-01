"""
OrthoTwin AI — Pytest Configuration & Test Fixtures
Provides isolated in-memory database sessions and synthetic test images.
"""

import pytest
import io
import sys
from pathlib import Path

# Add project root to sys.path for test discovery
sys_path = Path(__file__).resolve().parent.parent
if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

from PIL import Image, ImageDraw
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base, Role, User, Patient, FractureCase, Visit
from database.repositories import UserRepository, PatientRepository, FractureCaseRepository, VisitRepository
from services.authentication import hash_password
from config import ROLE_ADMIN, ROLE_DOCTOR, ROLE_PATIENT


@pytest.fixture(scope="function")
def test_db():
    """Provides an isolated, in-memory SQLite database for each test function."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()

    # Seed roles
    UserRepository.get_or_create_roles(db)

    # Seed sample users
    admin_hash, admin_salt = hash_password("AdminPass123!")
    doctor_hash, doctor_salt = hash_password("DoctorPass123!")
    patient_hash, patient_salt = hash_password("PatientPass123!")

    admin_role = db.query(Role).filter(Role.name == ROLE_ADMIN).first()
    doctor_role = db.query(Role).filter(Role.name == ROLE_DOCTOR).first()
    patient_role = db.query(Role).filter(Role.name == ROLE_PATIENT).first()

    u_admin = User(username="admin_test", password_hash=admin_hash, salt=admin_salt, role_id=admin_role.id, full_name="Admin Test", is_active=True)
    u_doctor = User(username="doctor_test", password_hash=doctor_hash, salt=doctor_salt, role_id=doctor_role.id, full_name="Dr. Test", is_active=True)
    u_patient = User(username="patient_test", password_hash=patient_hash, salt=patient_salt, role_id=patient_role.id, full_name="Patient Test", is_active=True)

    db.add_all([u_admin, u_doctor, u_patient])
    db.commit()

    yield db
    db.close()


@pytest.fixture
def synthetic_valid_image_bytes():
    """Generates valid 512x512 JPEG image bytes."""
    img = Image.new("RGB", (512, 512), color=(50, 50, 50))
    draw = ImageDraw.Draw(img)
    draw.rectangle([180, 50, 320, 450], fill=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def synthetic_small_image_bytes():
    """Generates undersized (100x100) image bytes."""
    img = Image.new("RGB", (100, 100), color=(50, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def synthetic_blurred_image_bytes():
    """Generates a severely blurred image (low Laplacian variance)."""
    import cv2
    import numpy as np
    img = np.full((512, 512), 128, dtype=np.uint8)
    blurred = cv2.GaussianBlur(img, (25, 25), 10)
    pil_img = Image.fromarray(blurred)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return buf.getvalue()
