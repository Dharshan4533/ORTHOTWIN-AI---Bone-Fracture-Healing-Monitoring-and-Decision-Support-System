"""
OrthoTwin AI — Native HTML & CSS Web Application (Flask Entrypoint)
Provides a clean, responsive, multi-page HTML5/CSS3 medical dashboard interface
connecting to all 8 core modules, SQLAlchemy database, and ReportLab PDF generator.
Strictly hardened: Enforces session authentication, RBAC, CSRF protection,
transactional uploads, and authorized access to clinical records and reports.
"""

import sys
import os
import json
import uuid
import secrets
from pathlib import Path
from datetime import datetime
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_file, session, abort
)
from werkzeug.exceptions import HTTPException
from PIL import Image
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    SECRET_KEY, INSTITUTION_NAME, MODEL_WEIGHTS_PATH,
    STATUS_READY_FOR_ANALYSIS, STATUS_NEEDS_REVIEW, STATUS_RESULT_READY,
    ROLE_DOCTOR, ROLE_PATIENT, ROLE_ADMIN, BASE_DIR, STORAGE_DIR,
    DEVELOPMENT_MODE
)
from database.db import init_db, get_db
from database.models import (
    User, Patient, FractureCase, Visit, XrayImage,
    AIAnalysisResult, ClinicianReview, Report, ModelVersion, AuditLog
)
from database.repositories import (
    UserRepository, PatientRepository, FractureCaseRepository,
    VisitRepository, XrayImageRepository, AIAnalysisRepository,
    ReportRepository, ModelVersionRepository, AuditRepository
)
from services.authentication import authenticate_user
from services.authorization import (
    is_clinician_or_admin, can_access_patient, can_access_image,
    can_access_report, require_role, AccessDeniedError
)
from services.file_storage import (
    save_raw_image, save_processed_image, get_safe_file_path
)
from services.reliability_service import compute_reliability_summary
from services.audit_service import log_audit_event
from modules.patient_management import (
    register_new_patient, create_patient_fracture_case, record_patient_visit
)
from modules.xray_validation import validate_xray_upload, XrayValidationError
from modules.xray_processing import generate_processed_review_copy
from modules.fracture_detection import run_xray_analysis, get_detector
from modules.digital_twin import (
    extract_experimental_features, build_longitudinal_trajectory_plot
)
from modules.doctor_dashboard import (
    get_dashboard_metrics, get_pending_review_queue, submit_clinician_attestation
)
from modules.patient_portal import get_patient_portal_data
from modules.report_generator import generate_pdf_report


app = Flask(__name__)
app.secret_key = SECRET_KEY

init_db()


# ==============================================================================
# Security & Session Helpers
# ==============================================================================

def get_current_user(db) -> User | None:
    """Retrieves current user strictly from authenticated session. No silent fallbacks."""
    username = session.get("username")
    if not username:
        return None
    return UserRepository.get_by_username(db, username)


def get_csrf_token() -> str:
    """Generates or retrieves cryptographic session CSRF token."""
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(16)
    return session["csrf_token"]


@app.context_processor
def inject_template_globals():
    return {
        "csrf_token": get_csrf_token,
        "is_dev_mode": DEVELOPMENT_MODE,
        "current_time": datetime.now()
    }


@app.before_request
def csrf_protect():
    """Validates CSRF token on all state-mutating POST requests."""
    if request.method == "POST":
        if app.config.get("TESTING") or app.config.get("WTF_CSRF_ENABLED") is False:
            return  # Allow automated test harness bypass when explicitly configured
        token = request.form.get("csrf_token")
        expected = session.get("csrf_token")
        if not token or not expected or token != expected:
            abort(400, description="Security Notice: CSRF token missing or invalid.")


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        with get_db() as db:
            user = get_current_user(db)
            if not user:
                flash("Please sign in to access the requested medical portal.", "warning")
                return redirect(url_for("login_view", next=request.url))
        return f(*args, **kwargs)
    return decorated_function


def roles_required(*allowed_roles: str):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            with get_db() as db:
                user = get_current_user(db)
                if not user:
                    flash("Please sign in to continue.", "warning")
                    return redirect(url_for("login_view", next=request.url))
                if user.role.name not in allowed_roles:
                    abort(403)
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# ==============================================================================
# Authentication Routes
# ==============================================================================

@app.route("/login", methods=["GET"])
def login_view():
    with get_db() as db:
        if get_current_user(db):
            return redirect(url_for("index"))
    return render_template("login.html", current_user=None)


@app.route("/login", methods=["POST"])
def login_action():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    with get_db() as db:
        user = authenticate_user(db, username, password)
        if user:
            session["username"] = user.username
            get_csrf_token()  # Ensure token initialized
            log_audit_event(db, "LOGIN_SUCCESS", user=user, details={"role": user.role.name})
            flash(f"Welcome back, {user.full_name}!", "success")

            if user.role.name == ROLE_PATIENT:
                return redirect(url_for("patient_portal_view"))
            next_url = request.args.get("next") or url_for("index")
            return redirect(next_url)
        else:
            flash("Invalid username or password.", "danger")
            return redirect(url_for("login_view"))


@app.route("/logout")
def logout_action():
    with get_db() as db:
        user = get_current_user(db)
        if user:
            log_audit_event(db, "LOGOUT", user=user, details={})
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("login_view"))


# ==============================================================================
# Clinician Dashboard & Patient Management
# ==============================================================================

@app.route("/")
@roles_required(ROLE_DOCTOR, ROLE_ADMIN)
def index():
    with get_db() as db:
        user = get_current_user(db)
        metrics = get_dashboard_metrics(db)
        queue = get_pending_review_queue(db)
        patients = PatientRepository.get_all(db)
        return render_template(
            "index.html",
            current_user=user,
            metrics=metrics,
            pending_queue=queue,
            patients=patients
        )


@app.route("/patients")
@roles_required(ROLE_DOCTOR, ROLE_ADMIN)
def patients_view():
    with get_db() as db:
        user = get_current_user(db)
        patients = PatientRepository.get_all(db)
        return render_template(
            "patients.html",
            current_user=user,
            patients=patients
        )


@app.route("/patients/create", methods=["POST"])
@roles_required(ROLE_DOCTOR, ROLE_ADMIN)
def create_patient_action():
    with get_db() as db:
        user = get_current_user(db)
        first_name = request.form.get("first_name", "").strip()
        last_name = request.form.get("last_name", "").strip()
        dob = request.form.get("dob", "").strip()
        gender = request.form.get("gender", "Male")
        phone = request.form.get("contact_phone", "").strip() or None
        history = request.form.get("medical_history", "").strip() or None

        try:
            p, warn = register_new_patient(
                db, first_name=first_name, last_name=last_name,
                dob=dob, gender=gender, contact_phone=phone,
                medical_history=history, current_user=user
            )
            flash(f"Patient {p.patient_code} ({p.first_name} {p.last_name}) registered successfully!", "success")
            if warn:
                flash(warn, "warning")
        except ValueError as e:
            flash(f"Validation error: {str(e)}", "danger")
        except Exception:
            flash("An unexpected error occurred while registering the patient.", "danger")

    return redirect(url_for("patients_view"))


@app.route("/cases/create", methods=["POST"])
@roles_required(ROLE_DOCTOR, ROLE_ADMIN)
def create_case_action():
    with get_db() as db:
        user = get_current_user(db)
        patient_id_raw = request.form.get("patient_id")
        try:
            patient_id = int(patient_id_raw)
        except (TypeError, ValueError):
            flash("Invalid patient identifier provided.", "danger")
            return redirect(url_for("patients_view"))

        anatomical_site = request.form.get("anatomical_site", "").strip()
        injury_date = request.form.get("injury_date", "").strip()
        mechanism = request.form.get("injury_mechanism", "").strip() or None
        treatment = request.form.get("initial_treatment", "").strip() or None

        try:
            case = create_patient_fracture_case(
                db, patient_id=patient_id, anatomical_site=anatomical_site,
                injury_date=injury_date, injury_mechanism=mechanism,
                initial_treatment=treatment, current_user=user
            )
            flash(f"Fracture Case {case.case_code} ({case.anatomical_site}) created!", "success")
        except ValueError as e:
            flash(f"Validation error: {str(e)}", "danger")
        except Exception:
            flash("An unexpected error occurred while creating the fracture case.", "danger")

    return redirect(url_for("patients_view"))


@app.route("/visits/create", methods=["POST"])
@roles_required(ROLE_DOCTOR, ROLE_ADMIN)
def create_visit_action():
    with get_db() as db:
        user = get_current_user(db)
        case_id_raw = request.form.get("case_id")
        try:
            case_id = int(case_id_raw)
        except (TypeError, ValueError):
            flash("Invalid case identifier provided.", "danger")
            return redirect(url_for("patients_view"))

        visit_date = request.form.get("visit_date", "").strip()
        pain_score_raw = request.form.get("pain_score", "4")
        try:
            pain_score = int(pain_score_raw)
        except (TypeError, ValueError):
            pain_score = None

        weight_bearing = request.form.get("weight_bearing_status")
        next_date = request.form.get("scheduled_next_date", "").strip() or None
        notes = request.form.get("clinical_notes", "").strip() or None

        try:
            visit = record_patient_visit(
                db, case_id=case_id, visit_date=visit_date,
                clinical_notes=notes, weight_bearing_status=weight_bearing,
                pain_score=pain_score, scheduled_next_date=next_date,
                current_user=user
            )
            flash(f"Visit {visit.visit_code} recorded ({visit.days_post_injury} days post injury)!", "success")
        except ValueError as e:
            flash(f"Validation error: {str(e)}", "danger")
        except Exception:
            flash("An unexpected error occurred while logging the visit.", "danger")

    return redirect(url_for("patients_view"))


# ==============================================================================
# Transactional X-Ray Ingestion
# ==============================================================================

@app.route("/upload", methods=["GET"])
@roles_required(ROLE_DOCTOR, ROLE_ADMIN)
def upload_view():
    with get_db() as db:
        user = get_current_user(db)
        patients = PatientRepository.get_all(db)
        recent_images = db.query(XrayImage).order_by(XrayImage.id.desc()).limit(10).all()
        return render_template(
            "upload.html",
            current_user=user,
            patients=patients,
            recent_images=recent_images
        )


@app.route("/upload", methods=["POST"])
@roles_required(ROLE_DOCTOR, ROLE_ADMIN)
def upload_action():
    with get_db() as db:
        user = get_current_user(db)
        visit_id_raw = request.form.get("visit_id")
        try:
            visit_id = int(visit_id_raw)
            visit = VisitRepository.get_by_id(db, visit_id)
            if not visit:
                flash("Selected follow-up visit does not exist.", "danger")
                return redirect(url_for("upload_view"))
        except (TypeError, ValueError):
            flash("Invalid visit identifier provided.", "danger")
            return redirect(url_for("upload_view"))

        uploaded_file = request.files.get("xray_file")
        if not uploaded_file or not uploaded_file.filename:
            flash("Please select a file to upload.", "danger")
            return redirect(url_for("upload_view"))

        file_bytes = uploaded_file.read()
        filename = uploaded_file.filename

        created_files = []
        try:
            # 1. Structural & Quality Validation
            val = validate_xray_upload(file_bytes, filename)
            existing_hash = XrayImageRepository.find_by_hash(db, val["sha256"])
            near_dup_info = XrayImageRepository.find_near_duplicate(db, val["dhash"])

            # 2. File Persistence (tracked for rollback)
            raw_path, rel_path = save_raw_image(file_bytes, filename)
            created_files.append(raw_path)

            proc_bytes = generate_processed_review_copy(raw_path)
            proc_path, proc_rel = save_processed_image(proc_bytes, rel_path)
            created_files.append(proc_path)

            # 3. Database Records Creation (Atomic transaction)
            status = STATUS_READY_FOR_ANALYSIS if not val["warnings"] else STATUS_NEEDS_REVIEW
            img_rec = XrayImageRepository.create_image(
                db=db,
                visit_id=visit_id,
                file_path=rel_path,
                original_filename=filename,
                file_size=val["file_size"],
                image_hash=val["sha256"],
                dhash=val["dhash"],
                processed_path=proc_rel,
                status=status
            )

            XrayImageRepository.save_validation_result(
                db=db,
                image_id=img_rec.id,
                is_valid=True,
                format=val["format"],
                width=val["width"],
                height=val["height"],
                blur_score=val["blur_score"],
                contrast_score=val["contrast_score"],
                min_brightness=val["min_brightness"],
                max_brightness=val["max_brightness"],
                duplicate_of_id=existing_hash.id if existing_hash else None,
                warnings=val["warnings"]
            )

            db.commit()

            log_audit_event(
                db=db,
                action="UPLOAD_XRAY",
                user=user,
                entity_type="XrayImage",
                entity_id=str(img_rec.id),
                details={"format": val["format"], "size_bytes": val["file_size"]}
            )

            flash(f"Radiograph #{img_rec.id} ingested successfully! Sharpness: {val['blur_score']:.1f}, Contrast: {val['contrast_score']:.1f}", "success")
            if existing_hash:
                flash(f"Notice: Exact duplicate content detected (matches Scan #{existing_hash.id}).", "warning")
            elif near_dup_info:
                match_img, dist = near_dup_info
                flash(f"Notice: Visual near-duplicate detected (Hamming distance {dist} to Scan #{match_img.id}). Legitimate follow-up scan preserved.", "info")

            if val["warnings"]:
                for w in val["warnings"]:
                    flash(f"Warning: {w}", "warning")

        except XrayValidationError as e:
            db.rollback()
            for p in created_files:
                if p.exists():
                    p.unlink()
            flash(f"Upload rejected: {str(e)}", "danger")
        except Exception as e:
            db.rollback()
            for p in created_files:
                if p.exists():
                    p.unlink()
            flash("An unexpected error occurred during radiograph ingestion. Filesystem state was safely rolled back.", "danger")

    return redirect(url_for("upload_view"))


# ==============================================================================
# AI Detection & Digital Twin
# ==============================================================================

@app.route("/analysis")
@roles_required(ROLE_DOCTOR, ROLE_ADMIN)
def analysis_view():
    with get_db() as db:
        user = get_current_user(db)
        all_images = db.query(XrayImage).order_by(XrayImage.id.desc()).all()
        selected_id = request.args.get("image_id", type=int)
        selected_image = None
        detections = []

        if all_images:
            if selected_id:
                selected_image = XrayImageRepository.get_by_id(db, selected_id)
            if not selected_image:
                selected_image = all_images[0]

        if selected_image and selected_image.ai_analysis:
            try:
                detections = json.loads(selected_image.ai_analysis.detections_json)
            except Exception:
                pass

        detector = get_detector()
        return render_template(
            "analysis.html",
            current_user=user,
            all_images=all_images,
            selected_image=selected_image,
            detections=detections,
            detector_loaded=detector.is_loaded
        )


@app.route("/analysis/run/<int:image_id>", methods=["POST"])
@roles_required(ROLE_DOCTOR, ROLE_ADMIN)
def run_detection_action(image_id):
    try:
        conf_thresh = min(1.0, max(0.01, float(request.form.get("conf_threshold", 0.20))))
    except (TypeError, ValueError):
        conf_thresh = 0.20

    with get_db() as db:
        user = get_current_user(db)
        ok, res, msg = run_xray_analysis(db, image_id, conf_threshold=conf_thresh, current_user=user)
        if ok:
            flash(f"AI Detection completed! {msg}", "success")
        else:
            flash(f"Detection note: {msg}", "warning")
    return redirect(url_for("analysis_view", image_id=image_id))


@app.route("/digital-twin")
@roles_required(ROLE_DOCTOR, ROLE_ADMIN)
def digital_twin_view():
    with get_db() as db:
        user = get_current_user(db)
        cases = db.query(FractureCase).all()
        case_id = request.args.get("case_id", type=int)
        selected_case = None

        if cases:
            if case_id:
                selected_case = FractureCaseRepository.get_by_id(db, case_id)
            if not selected_case:
                selected_case = cases[0]

        plot_json = None
        if selected_case and selected_case.visits:
            plot_data = []
            for v in sorted(selected_case.visits, key=lambda x: x.visit_date):
                gap = None
                dens = None
                if v.xray_images and v.xray_images[0].extracted_features:
                    gap = v.xray_images[0].extracted_features.fracture_gap_px
                    dens = v.xray_images[0].extracted_features.callus_density_ratio

                # Preserve None for missing measurements without fabricating 0.0 or 1.0
                plot_data.append({
                    "visit_code": v.visit_code,
                    "visit_date": v.visit_date,
                    "days_post_injury": v.days_post_injury,
                    "fracture_gap_px": gap,
                    "callus_density_ratio": dens,
                    "pain_score": v.pain_score
                })
            fig = build_longitudinal_trajectory_plot(plot_data)
            plot_json = fig.to_json()

        return render_template(
            "digital_twin.html",
            current_user=user,
            all_cases=cases,
            selected_case=selected_case,
            plot_json=plot_json
        )


# ==============================================================================
# Clinician Review & PDF Reports
# ==============================================================================

@app.route("/doctor-review")
@roles_required(ROLE_DOCTOR, ROLE_ADMIN)
def doctor_review_view():
    with get_db() as db:
        user = get_current_user(db)
        all_images = db.query(XrayImage).order_by(XrayImage.id.desc()).all()
        selected_id = request.args.get("image_id", type=int)
        selected_image = None
        current_review = None
        reports = []

        if all_images:
            if selected_id:
                selected_image = XrayImageRepository.get_by_id(db, selected_id)
            if not selected_image:
                selected_image = all_images[0]

        if selected_image:
            if selected_image.ai_analysis and selected_image.ai_analysis.clinician_review:
                current_review = selected_image.ai_analysis.clinician_review
            reports = ReportRepository.get_by_visit(db, selected_image.visit_id)

        return render_template(
            "doctor_review.html",
            current_user=user,
            all_images=all_images,
            selected_image=selected_image,
            current_review=current_review,
            reports=reports
        )


@app.route("/doctor-review/submit/<int:image_id>", methods=["POST"])
@roles_required(ROLE_DOCTOR, ROLE_ADMIN)
def submit_review_action(image_id):
    with get_db() as db:
        user = get_current_user(db)
        image = XrayImageRepository.get_by_id(db, image_id)
        if not image or not image.ai_analysis:
            flash("AI Analysis must be completed before submitting clinical attestation.", "danger")
            return redirect(url_for("doctor_review_view", image_id=image_id))

        status = request.form.get("review_status", "APPROVED")
        notes = request.form.get("clinical_notes", "").strip()

        ok, rev, msg = submit_clinician_attestation(
            db=db,
            analysis_id=image.ai_analysis.id,
            reviewer_user=user,
            review_status=status,
            clinical_notes=notes
        )
        if ok:
            flash(f"Clinician review saved as {status}!", "success")
        else:
            flash(f"Error recording review: {msg}", "danger")

    return redirect(url_for("doctor_review_view", image_id=image_id))


@app.route("/doctor-review/generate-pdf/<int:image_id>", methods=["POST"])
@roles_required(ROLE_DOCTOR, ROLE_ADMIN)
def generate_pdf_action(image_id):
    with get_db() as db:
        user = get_current_user(db)
        image = XrayImageRepository.get_by_id(db, image_id)
        if not image:
            flash("Image record not found.", "danger")
            return redirect(url_for("doctor_review_view"))

        v = image.visit
        c = v.fracture_case
        p = c.patient

        ai_data = None
        if image.ai_analysis:
            try:
                ai_data = {"detections": json.loads(image.ai_analysis.detections_json)}
            except Exception:
                pass

        rev_data = None
        if image.ai_analysis and image.ai_analysis.clinician_review:
            r = image.ai_analysis.clinician_review
            rev_data = {
                "review_status": r.review_status,
                "reviewer_name": r.reviewer.full_name,
                "signed_at": r.signed_at.strftime("%Y-%m-%d %H:%M"),
                "clinical_notes": r.clinical_notes
            }

        rel_sum = compute_reliability_summary(image.validation_result, image.ai_analysis, v)

        # Unique report code preventing accidental collisions on repeated generation
        unique_suffix = uuid.uuid4().hex[:6].upper()
        rep_code = f"REP-{datetime.now().strftime('%Y%m%d')}-{image.id:04d}-{unique_suffix}"

        # Resolve snapshot image path
        scan_path = None
        if image.processed_path:
            scan_path = (STORAGE_DIR.parent / image.processed_path).resolve()
        elif image.file_path:
            scan_path = (STORAGE_DIR.parent / image.file_path).resolve()

        pdf_path = generate_pdf_report(
            report_code=rep_code,
            patient_info={"patient_code": p.patient_code, "full_name": f"{p.first_name} {p.last_name}", "dob": p.dob, "gender": p.gender},
            case_info={"case_code": c.case_code, "anatomical_site": c.anatomical_site, "injury_date": c.injury_date},
            visit_info={"visit_code": v.visit_code, "visit_date": v.visit_date, "days_post_injury": v.days_post_injury},
            ai_analysis_info=ai_data,
            clinician_review_info=rev_data,
            reliability_summary=rel_sum,
            scan_image_path=scan_path
        )

        ReportRepository.save_report(db, v.id, str(pdf_path), user.id, is_final=True)
        flash(f"Report {rep_code}.pdf generated successfully!", "success")

    return redirect(url_for("doctor_review_view", image_id=image_id))


# ==============================================================================
# Patient Portal & Diagnostics
# ==============================================================================

@app.route("/patient-portal")
@login_required
def patient_portal_view():
    with get_db() as db:
        user = get_current_user(db)
        if user.role.name == ROLE_PATIENT:
            target_user = user
        else:
            # Clinicians can inspect patient portal data by patient_id
            pat_id = request.args.get("patient_id", type=int)
            if pat_id:
                target_user = db.query(User).filter(User.patient_id == pat_id).first() or user
            else:
                target_user = db.query(User).filter(User.role.has(name=ROLE_PATIENT)).first() or user

        portal_data = get_patient_portal_data(db, target_user)
        return render_template(
            "patient_portal.html",
            current_user=user,
            portal_data=portal_data
        )


@app.route("/diagnostics")
@roles_required(ROLE_DOCTOR, ROLE_ADMIN)
def diagnostics_view():
    with get_db() as db:
        user = get_current_user(db)
        sys_info = {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "weights_path": MODEL_WEIGHTS_PATH,
            "weights_exist": Path(MODEL_WEIGHTS_PATH).exists()
        }
        model_versions = db.query(ModelVersion).all()
        recent_logs = AuditRepository.list_recent(db, limit=20)

        audit_content = ""
        audit_file = BASE_DIR / "data" / "DATASET_AUDIT_REPORT.md"
        if audit_file.exists():
            with open(audit_file, "r", encoding="utf-8") as f:
                audit_content = f.read()

        return render_template(
            "diagnostics.html",
            current_user=user,
            sys_info=sys_info,
            model_versions=model_versions,
            recent_logs=recent_logs,
            audit_content=audit_content
        )


# ==============================================================================
# Secure File Downloads (Protected by Authentication & Permissions)
# ==============================================================================

@app.route("/storage/<path:filename>")
@login_required
def serve_storage_file(filename):
    """Safely serves storage images and reports with path traversal protection and authorization."""
    try:
        safe_path = get_safe_file_path(filename)
        if not safe_path.exists():
            abort(404)

        with get_db() as db:
            user = get_current_user(db)
            if not user:
                abort(401)

            if is_clinician_or_admin(user):
                return send_file(safe_path)

            # Patient authorization check
            img = db.query(XrayImage).filter(
                (XrayImage.file_path.like(f"%{safe_path.name}")) |
                (XrayImage.processed_path.like(f"%{safe_path.name}"))
            ).first()
            if img:
                if can_access_image(db, user, img.id):
                    return send_file(safe_path)
                abort(403)

            rep = db.query(Report).filter(Report.pdf_path.like(f"%{safe_path.name}")).first()
            if rep:
                if can_access_report(db, user, rep):
                    return send_file(safe_path)
                abort(403)

            abort(403)
    except HTTPException:
        raise
    except Exception:
        abort(403)


@app.route("/reports/<int:report_id>")
@login_required
def download_report(report_id):
    with get_db() as db:
        user = get_current_user(db)
        rep = db.query(Report).filter(Report.id == report_id).first()
        if not rep or not Path(rep.pdf_path).exists():
            abort(404)
        if not can_access_report(db, user, rep):
            abort(403)
        return send_file(rep.pdf_path, as_attachment=True, download_name=f"{rep.report_code}.pdf")


@app.route("/reports/code/<report_code>")
@login_required
def download_report_by_code(report_code):
    with get_db() as db:
        user = get_current_user(db)
        rep = db.query(Report).filter(Report.report_code == report_code).first()
        if not rep or not Path(rep.pdf_path).exists():
            abort(404)
        if not can_access_report(db, user, rep):
            abort(403)
        return send_file(rep.pdf_path, as_attachment=True, download_name=f"{rep.report_code}.pdf")


if __name__ == "__main__":
    print(f"\n=======================================================")
    print(f" OrthoTwin AI — HTML5/CSS3 Clinical Decision Support   ")
    print(f" Web Server Running at: http://127.0.0.1:5000         ")
    print(f"=======================================================\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
