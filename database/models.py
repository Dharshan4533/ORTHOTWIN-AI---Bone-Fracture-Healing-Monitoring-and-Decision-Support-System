"""
OrthoTwin AI — SQLAlchemy ORM Data Models
Implements complete database schema with strict relational integrity,
foreign keys, timestamps, and indexes.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    ForeignKey, Text, Index
)
from sqlalchemy.orm import relationship
from database.db import Base


def utc_now():
    return datetime.now(timezone.utc)


class Role(Base):
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False, index=True)  # admin, doctor, patient
    description = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    users = relationship("User", back_populates="role")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    salt = Column(String(64), nullable=False)
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=False)
    full_name = Column(String(120), nullable=False)
    email = Column(String(120), unique=True, nullable=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)  # Populated if user is a patient
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    role = relationship("Role", back_populates="users")
    patient = relationship("Patient", back_populates="user_account")
    audit_logs = relationship("AuditLog", back_populates="user")
    reviews = relationship("ClinicianReview", back_populates="reviewer")
    reports_generated = relationship("Report", back_populates="generated_by")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_code = Column(String(50), unique=True, nullable=False, index=True)  # PAT-YYYYMMDD-XXXX
    first_name = Column(String(80), nullable=False)
    last_name = Column(String(80), nullable=False)
    dob = Column(String(10), nullable=False)  # YYYY-MM-DD
    gender = Column(String(20), nullable=False)
    contact_phone = Column(String(30), nullable=True)
    medical_history = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    user_account = relationship("User", back_populates="patient", uselist=False)
    fracture_cases = relationship("FractureCase", back_populates="patient", cascade="all, delete-orphan")


class FractureCase(Base):
    __tablename__ = "fracture_cases"

    id = Column(Integer, primary_key=True, index=True)
    case_code = Column(String(50), unique=True, nullable=False, index=True)  # CASE-YYYYMMDD-XXXX
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    anatomical_site = Column(String(80), nullable=False)  # Radius, Tibia, Femur, etc.
    injury_date = Column(String(10), nullable=False)  # YYYY-MM-DD
    injury_mechanism = Column(String(255), nullable=True)
    initial_treatment = Column(String(100), nullable=True)  # Cast, Splint, ORIF
    status = Column(String(30), default="ACTIVE", nullable=False)  # ACTIVE, CLOSED, ARCHIVED
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    patient = relationship("Patient", back_populates="fracture_cases")
    visits = relationship("Visit", back_populates="fracture_case", cascade="all, delete-orphan", order_by="Visit.visit_date")
    predictions = relationship("PredictionResult", back_populates="fracture_case", cascade="all, delete-orphan")


class Visit(Base):
    __tablename__ = "visits"

    id = Column(Integer, primary_key=True, index=True)
    visit_code = Column(String(50), unique=True, nullable=False, index=True)  # VISIT-YYYYMMDD-XXXX
    case_id = Column(Integer, ForeignKey("fracture_cases.id"), nullable=False, index=True)
    visit_date = Column(String(10), nullable=False)  # YYYY-MM-DD
    days_post_injury = Column(Integer, default=0, nullable=False)
    clinical_notes = Column(Text, nullable=True)
    weight_bearing_status = Column(String(50), nullable=True)  # Non-weight-bearing, Partial, Full
    pain_score = Column(Integer, nullable=True)  # 0 to 10
    scheduled_next_date = Column(String(10), nullable=True)  # Future appointment date
    created_at = Column(DateTime, default=utc_now, nullable=False)

    fracture_case = relationship("FractureCase", back_populates="visits")
    xray_images = relationship("XrayImage", back_populates="visit", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="visit", cascade="all, delete-orphan")


class XrayImage(Base):
    __tablename__ = "xray_images"

    id = Column(Integer, primary_key=True, index=True)
    visit_id = Column(Integer, ForeignKey("visits.id"), nullable=False, index=True)
    file_path = Column(String(255), nullable=False)
    processed_path = Column(String(255), nullable=True)
    original_filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    image_hash = Column(String(64), nullable=False, index=True)  # SHA-256 for exact duplicate check
    dhash = Column(String(32), nullable=True)  # Perceptual hash
    status = Column(String(50), default="UPLOADED", nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    visit = relationship("Visit", back_populates="xray_images")
    validation_result = relationship(
        "ImageValidationResult",
        back_populates="xray_image",
        uselist=False,
        cascade="all, delete-orphan",
        foreign_keys="ImageValidationResult.image_id"
    )
    ai_analysis = relationship("AIAnalysisResult", back_populates="xray_image", uselist=False, cascade="all, delete-orphan")
    extracted_features = relationship("ExtractedFeature", back_populates="xray_image", uselist=False, cascade="all, delete-orphan")


class ImageValidationResult(Base):
    __tablename__ = "image_validation_results"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("xray_images.id"), unique=True, nullable=False)
    is_valid = Column(Boolean, nullable=False)
    format = Column(String(20), nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    blur_score = Column(Float, nullable=False)
    contrast_score = Column(Float, nullable=False)
    min_brightness = Column(Float, nullable=False)
    max_brightness = Column(Float, nullable=False)
    duplicate_of_id = Column(Integer, ForeignKey("xray_images.id"), nullable=True)
    warnings_json = Column(Text, default="[]", nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    xray_image = relationship("XrayImage", back_populates="validation_result", foreign_keys=[image_id])
    duplicate_of = relationship("XrayImage", foreign_keys=[duplicate_of_id])


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    version_tag = Column(String(50), unique=True, nullable=False)
    task_type = Column(String(50), nullable=False)  # DETECTION, SEGMENTATION
    weights_path = Column(String(255), nullable=True)
    training_dataset = Column(String(100), nullable=True)
    map_score = Column(Float, nullable=True)
    parameters_json = Column(Text, default="{}", nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    analyses = relationship("AIAnalysisResult", back_populates="model_version")


class AIAnalysisResult(Base):
    __tablename__ = "ai_analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("xray_images.id"), unique=True, nullable=False)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=True)
    detections_json = Column(Text, default="[]", nullable=False)  # [{bbox: [x1, y1, x2, y2], conf: 0.85, class: 'fracture'}]
    confidence_summary = Column(Float, nullable=True)
    raw_scores_json = Column(Text, default="{}", nullable=False)
    status = Column(String(50), default="RESULT_READY", nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    xray_image = relationship("XrayImage", back_populates="ai_analysis")
    model_version = relationship("ModelVersion", back_populates="analyses")
    clinician_review = relationship("ClinicianReview", back_populates="ai_analysis", uselist=False, cascade="all, delete-orphan")


class ExtractedFeature(Base):
    __tablename__ = "extracted_features"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("xray_images.id"), unique=True, nullable=False)
    fracture_gap_px = Column(Float, nullable=True)  # Estimated gap width in pixels
    callus_density_ratio = Column(Float, nullable=True)  # Relative density index
    cortical_continuity_index = Column(Float, nullable=True)  # Contour continuity index [0, 1]
    features_json = Column(Text, default="{}", nullable=False)
    is_experimental = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    xray_image = relationship("XrayImage", back_populates="extracted_features")


class PredictionResult(Base):
    __tablename__ = "prediction_results"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("fracture_cases.id"), nullable=False, index=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=True)
    prediction_type = Column(String(50), nullable=False)  # UNION_PROBABILITY, HEALING_TRAJECTORY
    progress_status = Column(String(50), nullable=False)  # UNAVAILABLE, EXPERIMENTAL
    confidence_interval = Column(String(100), nullable=True)
    disclaimer = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    fracture_case = relationship("FractureCase", back_populates="predictions")


class ClinicianReview(Base):
    __tablename__ = "clinician_reviews"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("ai_analysis_results.id"), unique=True, nullable=False)
    reviewer_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    review_status = Column(String(50), nullable=False)  # APPROVED, REJECTED, MODIFIED
    clinical_notes = Column(Text, nullable=False)
    signed_at = Column(DateTime, default=utc_now, nullable=False)

    ai_analysis = relationship("AIAnalysisResult", back_populates="clinician_review")
    reviewer = relationship("User", back_populates="reviews")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    visit_id = Column(Integer, ForeignKey("visits.id"), nullable=False, index=True)
    report_code = Column(String(60), unique=True, nullable=False, index=True)
    pdf_path = Column(String(255), nullable=False)
    generated_by_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    is_final = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    visit = relationship("Visit", back_populates="reports")
    generated_by = relationship("User", back_populates="reports_generated")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    action = Column(String(80), nullable=False, index=True)  # LOGIN, VIEW_PATIENT, UPLOAD_XRAY, SIGN_OFF
    entity_type = Column(String(50), nullable=True)  # Patient, Visit, XrayImage, AIAnalysis
    entity_id = Column(String(50), nullable=True)
    details_json = Column(Text, default="{}", nullable=False)
    ip_address = Column(String(50), nullable=True)
    timestamp = Column(DateTime, default=utc_now, nullable=False, index=True)

    user = relationship("User", back_populates="audit_logs")
