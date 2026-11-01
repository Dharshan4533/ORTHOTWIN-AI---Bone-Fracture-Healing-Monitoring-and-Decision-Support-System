"""
OrthoTwin AI — Database Repositories
Clean Data Access Layer providing atomic CRUD operations, search filters,
code generators, and relational integrity.
"""

import json
from datetime import datetime, date
from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, desc
from database.models import (
    User, Role, Patient, FractureCase, Visit, XrayImage,
    ImageValidationResult, AIAnalysisResult, ExtractedFeature,
    PredictionResult, ClinicianReview, Report, ModelVersion, AuditLog
)
from config import (
    ROLE_ADMIN, ROLE_DOCTOR, ROLE_PATIENT, ALL_ROLES,
    STATUS_UPLOADED, STATUS_RESULT_READY, STATUS_CLINICIAN_REVIEWED
)


# ==============================================================================
# Helper ID Generators
# ==============================================================================

def generate_code(prefix: str, db: Session, model_cls, code_attr: str) -> str:
    """Generates sequential date-stamped codes e.g. PAT-20261014-0001"""
    date_str = datetime.now().strftime("%Y%m%d")
    count = db.query(model_cls).filter(
        getattr(model_cls, code_attr).like(f"{prefix}-{date_str}-%")
    ).count() + 1
    return f"{prefix}-{date_str}-{count:04d}"


# ==============================================================================
# Role & User Repository
# ==============================================================================

class UserRepository:
    @staticmethod
    def get_or_create_roles(db: Session):
        for role_name in ALL_ROLES:
            existing = db.query(Role).filter(Role.name == role_name).first()
            if not existing:
                db.add(Role(name=role_name, description=f"{role_name.capitalize()} user role"))
        db.commit()

    @staticmethod
    def get_by_username(db: Session, username: str) -> Optional[User]:
        return db.query(User).filter(User.username == username).first()

    @staticmethod
    def get_by_id(db: Session, user_id: int) -> Optional[User]:
        return db.query(User).filter(User.id == user_id).first()

    @staticmethod
    def create_user(
        db: Session,
        username: str,
        password_hash: str,
        salt: str,
        role_name: str,
        full_name: str,
        email: Optional[str] = None,
        patient_id: Optional[int] = None
    ) -> User:
        role = db.query(Role).filter(Role.name == role_name).first()
        if not role:
            role = Role(name=role_name, description=f"{role_name.capitalize()} role")
            db.add(role)
            db.flush()

        user = User(
            username=username,
            password_hash=password_hash,
            salt=salt,
            role_id=role.id,
            full_name=full_name,
            email=email,
            patient_id=patient_id,
            is_active=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user


# ==============================================================================
# Patient Repository
# ==============================================================================

class PatientRepository:
    @staticmethod
    def create_patient(
        db: Session,
        first_name: str,
        last_name: str,
        dob: str,
        gender: str,
        contact_phone: Optional[str] = None,
        medical_history: Optional[str] = None
    ) -> Patient:
        patient_code = generate_code("PAT", db, Patient, "patient_code")
        patient = Patient(
            patient_code=patient_code,
            first_name=first_name.strip(),
            last_name=last_name.strip(),
            dob=dob.strip(),
            gender=gender.strip(),
            contact_phone=contact_phone.strip() if contact_phone else None,
            medical_history=medical_history.strip() if medical_history else None
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)
        return patient

    @staticmethod
    def find_duplicate(db: Session, first_name: str, last_name: str, dob: str, phone: Optional[str] = None) -> Optional[Patient]:
        query = db.query(Patient).filter(
            Patient.first_name.ilike(first_name.strip()),
            Patient.last_name.ilike(last_name.strip()),
            Patient.dob == dob.strip()
        )
        if phone:
            query = query.filter(Patient.contact_phone == phone.strip())
        return query.first()

    @staticmethod
    def get_by_id(db: Session, patient_id: int) -> Optional[Patient]:
        return db.query(Patient).filter(Patient.id == patient_id).first()

    @staticmethod
    def get_by_code(db: Session, patient_code: str) -> Optional[Patient]:
        return db.query(Patient).filter(Patient.patient_code == patient_code).first()

    @staticmethod
    def search_patients(db: Session, query_str: str) -> List[Patient]:
        q = f"%{query_str.strip()}%"
        return db.query(Patient).filter(
            or_(
                Patient.patient_code.ilike(q),
                Patient.first_name.ilike(q),
                Patient.last_name.ilike(q),
                Patient.contact_phone.ilike(q)
            )
        ).order_by(desc(Patient.created_at)).all()

    @staticmethod
    def get_all(db: Session) -> List[Patient]:
        return db.query(Patient).order_by(desc(Patient.created_at)).all()


# ==============================================================================
# Fracture Case Repository
# ==============================================================================

class FractureCaseRepository:
    @staticmethod
    def create_case(
        db: Session,
        patient_id: int,
        anatomical_site: str,
        injury_date: str,
        injury_mechanism: Optional[str] = None,
        initial_treatment: Optional[str] = None,
        notes: Optional[str] = None
    ) -> FractureCase:
        case_code = generate_code("CASE", db, FractureCase, "case_code")
        case = FractureCase(
            case_code=case_code,
            patient_id=patient_id,
            anatomical_site=anatomical_site,
            injury_date=injury_date,
            injury_mechanism=injury_mechanism,
            initial_treatment=initial_treatment,
            status="ACTIVE",
            notes=notes
        )
        db.add(case)
        db.commit()
        db.refresh(case)
        return case

    @staticmethod
    def get_by_id(db: Session, case_id: int) -> Optional[FractureCase]:
        return db.query(FractureCase).filter(FractureCase.id == case_id).first()

    @staticmethod
    def get_by_code(db: Session, case_code: str) -> Optional[FractureCase]:
        return db.query(FractureCase).filter(FractureCase.case_code == case_code).first()

    @staticmethod
    def list_by_patient(db: Session, patient_id: int) -> List[FractureCase]:
        return db.query(FractureCase).filter(
            FractureCase.patient_id == patient_id
        ).order_by(desc(FractureCase.created_at)).all()


# ==============================================================================
# Visit Repository
# ==============================================================================

class VisitRepository:
    @staticmethod
    def calculate_days_post_injury(injury_date_str: str, visit_date_str: str) -> int:
        try:
            injury_dt = datetime.strptime(injury_date_str, "%Y-%m-%d").date()
            visit_dt = datetime.strptime(visit_date_str, "%Y-%m-%d").date()
            delta = (visit_dt - injury_dt).days
            return max(0, delta)
        except Exception:
            return 0

    @staticmethod
    def create_visit(
        db: Session,
        case_id: int,
        visit_date: str,
        clinical_notes: Optional[str] = None,
        weight_bearing_status: Optional[str] = None,
        pain_score: Optional[int] = None,
        scheduled_next_date: Optional[str] = None
    ) -> Visit:
        case = db.query(FractureCase).filter(FractureCase.id == case_id).first()
        days_post = VisitRepository.calculate_days_post_injury(case.injury_date if case else visit_date, visit_date)
        visit_code = generate_code("VISIT", db, Visit, "visit_code")
        visit = Visit(
            visit_code=visit_code,
            case_id=case_id,
            visit_date=visit_date,
            days_post_injury=days_post,
            clinical_notes=clinical_notes,
            weight_bearing_status=weight_bearing_status,
            pain_score=pain_score,
            scheduled_next_date=scheduled_next_date
        )
        db.add(visit)
        db.commit()
        db.refresh(visit)
        return visit

    @staticmethod
    def get_by_id(db: Session, visit_id: int) -> Optional[Visit]:
        return db.query(Visit).filter(Visit.id == visit_id).first()

    @staticmethod
    def list_by_case(db: Session, case_id: int) -> List[Visit]:
        return db.query(Visit).filter(
            Visit.case_id == case_id
        ).order_by(Visit.visit_date).all()


# ==============================================================================
# X-ray Image & Validation Repository
# ==============================================================================

class XrayImageRepository:
    @staticmethod
    def create_image(
        db: Session,
        visit_id: int,
        file_path: str,
        original_filename: str,
        file_size: int,
        image_hash: str,
        dhash: Optional[str] = None,
        processed_path: Optional[str] = None,
        status: str = STATUS_UPLOADED
    ) -> XrayImage:
        image = XrayImage(
            visit_id=visit_id,
            file_path=file_path,
            original_filename=original_filename,
            file_size=file_size,
            image_hash=image_hash,
            dhash=dhash,
            processed_path=processed_path,
            status=status
        )
        db.add(image)
        db.commit()
        db.refresh(image)
        return image

    @staticmethod
    def find_by_hash(db: Session, image_hash: str) -> Optional[XrayImage]:
        return db.query(XrayImage).filter(XrayImage.image_hash == image_hash).first()

    @staticmethod
    def find_near_duplicate(db: Session, dhash: str, threshold: int = 10) -> Optional[Tuple[XrayImage, int]]:
        """Finds the closest near-duplicate image by perceptual dHash Hamming distance."""
        from modules.xray_validation import compute_dhash_distance
        if not dhash:
            return None
        candidates = db.query(XrayImage).filter(XrayImage.dhash != None).all()
        best_match = None
        min_dist = threshold + 1
        for img in candidates:
            if img.dhash:
                dist = compute_dhash_distance(dhash, img.dhash)
                if dist <= threshold and dist < min_dist:
                    min_dist = dist
                    best_match = img
        if best_match:
            return best_match, min_dist
        return None

    @staticmethod
    def get_by_id(db: Session, image_id: int) -> Optional[XrayImage]:
        return db.query(XrayImage).filter(XrayImage.id == image_id).first()

    @staticmethod
    def update_status(db: Session, image_id: int, new_status: str) -> Optional[XrayImage]:
        img = db.query(XrayImage).filter(XrayImage.id == image_id).first()
        if img:
            img.status = new_status
            db.commit()
            db.refresh(img)
        return img

    @staticmethod
    def save_validation_result(
        db: Session,
        image_id: int,
        is_valid: bool,
        format: str,
        width: int,
        height: int,
        blur_score: float,
        contrast_score: float,
        min_brightness: float,
        max_brightness: float,
        duplicate_of_id: Optional[int] = None,
        warnings: Optional[List[str]] = None
    ) -> ImageValidationResult:
        res = ImageValidationResult(
            image_id=image_id,
            is_valid=is_valid,
            format=format,
            width=width,
            height=height,
            blur_score=blur_score,
            contrast_score=contrast_score,
            min_brightness=min_brightness,
            max_brightness=max_brightness,
            duplicate_of_id=duplicate_of_id,
            warnings_json=json.dumps(warnings or [])
        )
        db.add(res)
        db.commit()
        db.refresh(res)
        return res


# ==============================================================================
# AI Analysis & Review Repository
# ==============================================================================

class AIAnalysisRepository:
    @staticmethod
    def save_analysis(
        db: Session,
        image_id: int,
        model_version_id: Optional[int],
        detections: List[Dict[str, Any]],
        confidence_summary: Optional[float],
        raw_scores: Dict[str, Any],
        status: str = STATUS_RESULT_READY
    ) -> AIAnalysisResult:
        existing = db.query(AIAnalysisResult).filter(AIAnalysisResult.image_id == image_id).first()
        if existing:
            existing.model_version_id = model_version_id
            existing.detections_json = json.dumps(detections)
            existing.confidence_summary = confidence_summary
            existing.raw_scores_json = json.dumps(raw_scores)
            existing.status = status
            db.commit()
            db.refresh(existing)
            return existing

        res = AIAnalysisResult(
            image_id=image_id,
            model_version_id=model_version_id,
            detections_json=json.dumps(detections),
            confidence_summary=confidence_summary,
            raw_scores_json=json.dumps(raw_scores),
            status=status
        )
        db.add(res)
        db.commit()
        db.refresh(res)
        return res

    @staticmethod
    def get_by_image_id(db: Session, image_id: int) -> Optional[AIAnalysisResult]:
        return db.query(AIAnalysisResult).filter(AIAnalysisResult.image_id == image_id).first()

    @staticmethod
    def save_features(
        db: Session,
        image_id: int,
        gap_px: Optional[float],
        callus_ratio: Optional[float],
        continuity_idx: Optional[float],
        details: Dict[str, Any]
    ) -> ExtractedFeature:
        existing = db.query(ExtractedFeature).filter(ExtractedFeature.image_id == image_id).first()
        if existing:
            existing.fracture_gap_px = gap_px
            existing.callus_density_ratio = callus_ratio
            existing.cortical_continuity_index = continuity_idx
            existing.features_json = json.dumps(details)
            db.commit()
            db.refresh(existing)
            return existing

        feat = ExtractedFeature(
            image_id=image_id,
            fracture_gap_px=gap_px,
            callus_density_ratio=callus_ratio,
            cortical_continuity_index=continuity_idx,
            features_json=json.dumps(details),
            is_experimental=True
        )
        db.add(feat)
        db.commit()
        db.refresh(feat)
        return feat

    @staticmethod
    def save_clinician_review(
        db: Session,
        analysis_id: int,
        reviewer_id: int,
        review_status: str,
        clinical_notes: str
    ) -> ClinicianReview:
        existing = db.query(ClinicianReview).filter(ClinicianReview.analysis_id == analysis_id).first()
        if existing:
            existing.reviewer_id = reviewer_id
            existing.review_status = review_status
            existing.clinical_notes = clinical_notes
            existing.signed_at = datetime.now()
            db.commit()
            db.refresh(existing)
            return existing

        review = ClinicianReview(
            analysis_id=analysis_id,
            reviewer_id=reviewer_id,
            review_status=review_status,
            clinical_notes=clinical_notes
        )
        db.add(review)
        db.commit()
        db.refresh(review)
        return review


# ==============================================================================
# Model Version & Audit Repository
# ==============================================================================

class ModelVersionRepository:
    @staticmethod
    def get_latest(db: Session, task_type: str = "DETECTION") -> Optional[ModelVersion]:
        return db.query(ModelVersion).filter(
            ModelVersion.task_type == task_type
        ).order_by(desc(ModelVersion.created_at)).first()

    @staticmethod
    def register_version(
        db: Session,
        model_name: str,
        version_tag: str,
        task_type: str,
        weights_path: str,
        training_dataset: str,
        map_score: Optional[float] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> ModelVersion:
        existing = db.query(ModelVersion).filter(ModelVersion.version_tag == version_tag).first()
        if existing:
            return existing
        mv = ModelVersion(
            model_name=model_name,
            version_tag=version_tag,
            task_type=task_type,
            weights_path=weights_path,
            training_dataset=training_dataset,
            map_score=map_score,
            parameters_json=json.dumps(parameters or {})
        )
        db.add(mv)
        db.commit()
        db.refresh(mv)
        return mv


class AuditRepository:
    @staticmethod
    def log(
        db: Session,
        action: str,
        user_id: Optional[int] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ) -> AuditLog:
        # Sanitize details to avoid logging sensitive plain passwords
        clean_details = dict(details or {})
        for secret_key in ["password", "secret", "token"]:
            if secret_key in clean_details:
                clean_details[secret_key] = "[REDACTED]"

        entry = AuditLog(
            user_id=user_id,
            action=action,
            entity_type=entity_type,
            entity_id=str(entity_id) if entity_id is not None else None,
            details_json=json.dumps(clean_details),
            ip_address=ip_address
        )
        db.add(entry)
        db.commit()
        return entry

    @staticmethod
    def list_recent(db: Session, limit: int = 50) -> List[AuditLog]:
        return db.query(AuditLog).order_by(desc(AuditLog.timestamp)).limit(limit).all()


class ReportRepository:
    @staticmethod
    def save_report(
        db: Session,
        visit_id: int,
        pdf_path: str,
        generated_by_id: int,
        is_final: bool = False
    ) -> Report:
        report_code = generate_code("REP", db, Report, "report_code")
        rep = Report(
            visit_id=visit_id,
            report_code=report_code,
            pdf_path=pdf_path,
            generated_by_id=generated_by_id,
            is_final=is_final
        )
        db.add(rep)
        db.commit()
        db.refresh(rep)
        return rep

    @staticmethod
    def get_by_visit(db: Session, visit_id: int) -> List[Report]:
        return db.query(Report).filter(Report.visit_id == visit_id).order_by(desc(Report.created_at)).all()
