"""
OrthoTwin AI — Central Application Configuration
Defines directory paths, security parameters, workflow statuses,
image processing thresholds, and medical prototype safety disclaimers.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"

STORAGE_DIR = BASE_DIR / "storage"
STORAGE_RAW_DIR = STORAGE_DIR / "uploads" / "raw"
STORAGE_PROCESSED_DIR = STORAGE_DIR / "uploads" / "processed"
STORAGE_REPORTS_DIR = STORAGE_DIR / "reports"

# Ensure runtime directories exist
for directory in [STORAGE_RAW_DIR, STORAGE_PROCESSED_DIR, STORAGE_REPORTS_DIR, WEIGHTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'orthotwin.db'}")
DATABASE_ECHO = os.getenv("DATABASE_ECHO", "False").lower() in ("true", "1", "yes")

# Environment & Development Mode
DEVELOPMENT_MODE = os.getenv("DEVELOPMENT_MODE", "True").lower() in ("true", "1", "yes")
ENV = os.getenv("ENV", "development")

# Security & Authentication Configuration
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    if DEVELOPMENT_MODE:
        SECRET_KEY = "orthotwin_secure_dev_secret_key_2026_academic"
    else:
        raise RuntimeError("CRITICAL SECURITY ERROR: SECRET_KEY environment variable must be set in production mode.")

HASH_ALGORITHM = "sha256"
HASH_ITERATIONS = 100_000

# User Roles
ROLE_ADMIN = "admin"
ROLE_DOCTOR = "doctor"
ROLE_PATIENT = "patient"
ALL_ROLES = [ROLE_ADMIN, ROLE_DOCTOR, ROLE_PATIENT]

# Dataset Archive Paths (Configurable via environment or CLI)
ROBOFLOW_ZIP_PATH = os.getenv("ROBOFLOW_ZIP_PATH", str(BASE_DIR / "bone fracture.v1-release-640.yolov8.zip"))
FRACATLAS_ZIP_PATH = os.getenv("FRACATLAS_ZIP_PATH", str(BASE_DIR / "FracAtlas.zip"))
SEGMENTATION_WEIGHTS_PATH = os.getenv("SEGMENTATION_WEIGHTS_PATH", str(WEIGHTS_DIR / "fracture_unet.pth"))

# Backend-Enforced Workflow State Machine
STATUS_UPLOADED = "UPLOADED"
STATUS_INVALID_FILE = "INVALID_FILE"
STATUS_NEEDS_REVIEW = "NEEDS_REVIEW"
STATUS_READY_FOR_ANALYSIS = "READY_FOR_ANALYSIS"
STATUS_ANALYSIS_RUNNING = "ANALYSIS_RUNNING"
STATUS_ANALYSIS_FAILED = "ANALYSIS_FAILED"
STATUS_RESULT_READY = "RESULT_READY"
STATUS_CLINICIAN_REVIEWED = "CLINICIAN_REVIEWED"

ALL_WORKFLOW_STATUSES = [
    STATUS_UPLOADED,
    STATUS_INVALID_FILE,
    STATUS_NEEDS_REVIEW,
    STATUS_READY_FOR_ANALYSIS,
    STATUS_ANALYSIS_RUNNING,
    STATUS_ANALYSIS_FAILED,
    STATUS_RESULT_READY,
    STATUS_CLINICIAN_REVIEWED,
]

# Image Validation Thresholds
MIN_IMAGE_DIMENSION = int(os.getenv("MIN_IMAGE_DIMENSION", "256"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "15"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Technical Quality Metric Thresholds
BLUR_LAPLACIAN_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "100.0"))
LOW_CONTRAST_THRESHOLD = float(os.getenv("LOW_CONTRAST_THRESHOLD", "25.0"))
MIN_BRIGHTNESS = 20.0
MAX_BRIGHTNESS = 235.0

# Detection Model Settings
DEFAULT_WEIGHTS_FILE = WEIGHTS_DIR / "fracture_yolov8.pt"
MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", str(DEFAULT_WEIGHTS_FILE))
DETECTION_CONF_THRESHOLD = float(os.getenv("DETECTION_CONF_THRESHOLD", "0.20"))
DETECTION_IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))

# Primary Dataset Metadata
ROBOFLOW_CLASSES = {
    0: "angle",
    1: "fracture",
    2: "line",
    3: "messed_up_angle"
}
FRACATLAS_CLASSES = {
    0: "fracture"
}

# Institution & Academic Prototype Disclaimers
INSTITUTION_NAME = os.getenv("INSTITUTION_NAME", "OrthoTwin AI Research Laboratory")
ACADEMIC_PROTOTYPE_DISCLAIMER = (
    "ACADEMIC PROTOTYPE ONLY: OrthoTwin AI is designed strictly for research and educational purposes. "
    "It is NOT a cleared or certified medical device (FDA/CE-MDR). It does not provide definitive medical "
    "diagnoses or autonomous treatment recommendations. All findings require independent clinician verification."
)

DIGITAL_TWIN_DISCLAIMER = (
    "2D PROJECTION NOTICE: The 2D Digital Twin utilizes planar radiographic projections. "
    "Extracted features (gap width proxy, callus density proxy, cortical continuity) are experimental "
    "computational metrics and must NOT be interpreted as validated clinical healing biomarkers. "
    "Planar projections do not provide 3D volumetric structural verification."
)

PREDICTION_UNAVAILABLE_DISCLAIMER = (
    "PREDICTIVE TRAJECTORY STATUS: Longitudinal outcome prediction is currently UNAVAILABLE. "
    "Reliable predictive healing modeling requires a clinically validated cohort with matched patient-linked "
    "longitudinal scans and standardized clinical union outcome labels. Historical progress comparisons only."
)
