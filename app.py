"""
OrthoTwin AI — Bone Fracture Healing Monitoring & Decision Support System
Main Streamlit Application Entrypoint.
Comprehensive, modular, runnable academic research prototype.
"""

import sys
import io
import json
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, List, Dict, Any

# Configure system path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    INSTITUTION_NAME, ACADEMIC_PROTOTYPE_DISCLAIMER, DIGITAL_TWIN_DISCLAIMER,
    PREDICTION_UNAVAILABLE_DISCLAIMER, ROLE_ADMIN, ROLE_DOCTOR, ROLE_PATIENT,
    STATUS_UPLOADED, STATUS_READY_FOR_ANALYSIS, STATUS_RESULT_READY,
    STATUS_CLINICIAN_REVIEWED, STATUS_ANALYSIS_FAILED, STATUS_NEEDS_REVIEW,
    BLUR_LAPLACIAN_THRESHOLD, LOW_CONTRAST_THRESHOLD, MODEL_WEIGHTS_PATH,
    DEVELOPMENT_MODE
)
from database.db import init_db, get_db
from database.models import (
    User, Patient, FractureCase, Visit, XrayImage,
    AIAnalysisResult, ExtractedFeature, ClinicianReview, Report, AuditLog
)
from database.repositories import (
    UserRepository, PatientRepository, FractureCaseRepository,
    VisitRepository, XrayImageRepository, AIAnalysisRepository,
    ReportRepository, ModelVersionRepository, AuditRepository
)
from services.authentication import authenticate_user
from services.authorization import (
    can_access_patient, filter_accessible_patients, is_clinician_or_admin
)
from services.file_storage import save_raw_image, save_processed_image
from services.reliability_service import (
    transition_status, compute_reliability_summary
)
from services.audit_service import log_audit_event
from modules.patient_management import (
    register_new_patient, create_patient_fracture_case, record_patient_visit
)
from modules.xray_validation import validate_xray_upload, XrayValidationError
from modules.xray_processing import generate_processed_review_copy
from modules.fracture_detection import run_xray_analysis, get_detector
from modules.digital_twin import (
    extract_experimental_features, compare_scans_visual,
    build_longitudinal_trajectory_plot
)
from modules.confidence_engine import evaluate_case_confidence
from modules.healing_analysis import compute_historical_healing_progression
from modules.prediction_engine import get_healing_prediction
from modules.doctor_dashboard import (
    get_dashboard_metrics, get_pending_review_queue, submit_clinician_attestation
)
from modules.patient_portal import get_patient_portal_data
from modules.report_generator import generate_pdf_report


# ==============================================================================
# Streamlit App Configuration & Styling
# ==============================================================================

st.set_page_config(
    page_title="OrthoTwin AI — Clinical Decision Support",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Medical Aesthetic Styling
st.markdown("""
<style>
    .main-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #4a5568;
        margin-bottom: 1rem;
    }
    .disclaimer-banner {
        background-color: #fff5f5;
        border-left: 4px solid #e53e3e;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        color: #9b2c2c;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-ready { background-color: #c6f6d5; color: #22543d; }
    .badge-reviewed { background-color: #bee3f8; color: #2a4365; }
    .badge-pending { background-color: #feebc8; color: #744210; }
    .badge-failed { background-color: #fed7d7; color: #742a2a; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Session State Initialization
# ==============================================================================

init_db()

if "current_user" not in st.session_state:
    st.session_state["current_user"] = None
if "selected_patient_id" not in st.session_state:
    st.session_state["selected_patient_id"] = None
if "selected_case_id" not in st.session_state:
    st.session_state["selected_case_id"] = None
if "selected_visit_id" not in st.session_state:
    st.session_state["selected_visit_id"] = None


def get_current_user_obj(db) -> Optional[User]:
    if st.session_state["current_user"]:
        return UserRepository.get_by_id(db, st.session_state["current_user"]["id"])
    return None


# ==============================================================================
# Sidebar: Authentication & Navigation
# ==============================================================================

with st.sidebar:
    st.markdown("### 🦴 OrthoTwin AI")
    st.markdown(f"**{INSTITUTION_NAME}**")
    st.caption("Academic Decision Support Prototype")

    st.markdown("---")

    # Authentication section
    with get_db() as db:
        user_obj = get_current_user_obj(db)

        if not user_obj:
            st.markdown("#### 🔐 User Sign In")
            with st.form("login_form"):
                login_user = st.text_input("Username").strip()
                login_pass = st.text_input("Password", type="password")
                submit_login = st.form_submit_button("Sign In")

                if submit_login:
                    auth_user = authenticate_user(db, login_user, login_pass)
                    if auth_user:
                        st.session_state["current_user"] = {
                            "id": auth_user.id,
                            "username": auth_user.username,
                            "full_name": auth_user.full_name,
                            "role": auth_user.role.name,
                            "patient_id": auth_user.patient_id
                        }
                        log_audit_event(db, "LOGIN_SUCCESS", user=auth_user, details={"username": login_user})
                        st.success(f"Welcome, {auth_user.full_name}")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

            if DEVELOPMENT_MODE:
                st.caption("⚠️ Demo Access (Development Mode Only):")
                col_l1, col_l2 = st.columns(2)
                with col_l1:
                    if st.button("Dr. Vance (MD)", use_container_width=True):
                        auth_user = authenticate_user(db, "dr_ortho", "DoctorSecure2026!")
                        if auth_user:
                            st.session_state["current_user"] = {
                                "id": auth_user.id, "username": auth_user.username,
                                "full_name": auth_user.full_name, "role": auth_user.role.name,
                                "patient_id": auth_user.patient_id
                            }
                            st.rerun()
                with col_l2:
                    if st.button("Alex Mercer (Pt)", use_container_width=True):
                        auth_user = authenticate_user(db, "patient_demo", "PatientSecure2026!")
                        if auth_user:
                            st.session_state["current_user"] = {
                                "id": auth_user.id, "username": auth_user.username,
                                "full_name": auth_user.full_name, "role": auth_user.role.name,
                                "patient_id": auth_user.patient_id
                            }
                            st.rerun()
            else:
                st.caption("Production authentication active. Enter verified account credentials above.")

        else:
            role_label = user_obj.role.name.upper()
            st.markdown(f"**Logged In:** {user_obj.full_name}")
            st.markdown(f"**Role:** `{role_label}`")
            if st.button("Sign Out", use_container_width=True):
                st.session_state["current_user"] = None
                st.session_state["selected_patient_id"] = None
                st.session_state["selected_case_id"] = None
                st.rerun()

    st.markdown("---")

    # Role-gated Navigation Menu
    current_role = st.session_state["current_user"]["role"] if st.session_state["current_user"] else None

    if current_role in (ROLE_ADMIN, ROLE_DOCTOR):
        nav_options = [
            "📊 Clinical Dashboard",
            "🧑‍⚕️ Patient Management",
            "📤 X-ray Upload & Validation",
            "🧠 AI Fracture Detection",
            "🦴 2D Digital Twin",
            "🛡️ Reliability & Confidence",
            "📈 Healing Progress",
            "📋 Doctor Review & Reports",
            "⚙️ System Diagnostics & Audit"
        ]
    elif current_role == ROLE_PATIENT:
        nav_options = [
            "👤 Patient Portal",
            "🦴 My Recovery Timeline",
            "ℹ️ Educational Disclaimer"
        ]
    else:
        nav_options = [
            "ℹ️ System Overview & Disclaimer"
        ]

    selected_nav = st.radio("Navigation", nav_options, index=0)

    st.markdown("---")
    st.markdown(
        f"<div class='disclaimer-banner'><b>⚠️ Academic Prototype</b><br>{ACADEMIC_PROTOTYPE_DISCLAIMER}</div>",
        unsafe_allow_html=True
    )


# ==============================================================================
# Main Views
# ==============================================================================

# Header
st.markdown("<div class='main-header'>OrthoTwin AI — Decision Support System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Bone Fracture Longitudinal Healing Monitoring & Multi-Stage Validation</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 1. Unauthenticated Overview
# ------------------------------------------------------------------------------
if not st.session_state["current_user"]:
    st.info("👋 Welcome to OrthoTwin AI. Please sign in via the left sidebar to access clinical or patient workflows.")
    
    st.markdown("""
    ### About OrthoTwin AI
    OrthoTwin AI is a modular academic decision support platform developed to investigate:
    1. **Multi-Stage X-Ray Validation:** Content-based format checking, integrity testing, resolution enforcement, and technical quality assessment (blur, contrast).
    2. **Deep Learning Fracture Localization:** YOLOv8 bounding-box candidate detection with defensive degradation for missing model weights.
    3. **2D Digital Twin & Image-Derived Proxies:** Chronological visit alignment, fracture gap width proxy measurements, callus density ratios, and cortical continuity metrics.
    4. **Reliability & Uncertainty Summary:** Clear separation of image sharpness, detector confidence, and clinical completeness.
    5. **Role-Based Isolation:** Absolute separation between attending clinicians and patient records.
    6. **Clinician Attestation & Structured Reporting:** Direct radiologist review workflows and ReportLab PDF clinical export.
    """)
    st.stop()


# ------------------------------------------------------------------------------
# 2. Clinical Dashboard (Doctor & Admin)
# ------------------------------------------------------------------------------
if selected_nav == "📊 Clinical Dashboard":
    st.markdown("### 📊 Clinical Dashboard Overview")
    with get_db() as db:
        metrics = get_dashboard_metrics(db)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registered Patients", metrics["total_patients"])
        c2.metric("Active Fracture Cases", metrics["active_cases"])
        c3.metric("Scans Awaiting Review", metrics["pending_reviews"])
        c4.metric("Clinician Sign-Offs", metrics["signed_reviews"])

        st.markdown("#### ⏳ Attestation Queue (Pending Clinician Sign-Off)")
        queue = get_pending_review_queue(db)
        if queue:
            df_q = pd.DataFrame(queue)
            st.dataframe(
                df_q[[
                    "patient_code", "patient_name", "case_code",
                    "anatomical_site", "visit_date", "confidence_summary", "created_at"
                ]],
                use_container_width=True
            )
        else:
            st.success("✅ No scans currently pending review in the queue.")

        st.markdown("#### 🔍 Quick Patient Directory")
        all_patients = PatientRepository.get_all(db)
        if all_patients:
            pat_data = []
            for p in all_patients:
                pat_data.append({
                    "Patient ID": p.patient_code,
                    "Name": f"{p.first_name} {p.last_name}",
                    "DOB": p.dob,
                    "Gender": p.gender,
                    "Active Cases": len(p.fracture_cases)
                })
            st.dataframe(pd.DataFrame(pat_data), use_container_width=True)


# ------------------------------------------------------------------------------
# 3. Patient Management (Doctor & Admin)
# ------------------------------------------------------------------------------
elif selected_nav == "🧑‍⚕️ Patient Management":
    st.markdown("### 🧑‍⚕️ Patient & Case Management")
    t1, t2, t3 = st.tabs(["Search & Directory", "Register New Patient", "Create Case / Record Visit"])

    with get_db() as db:
        user_obj = get_current_user_obj(db)

        # Tab 1: Directory
        with t1:
            search_query = st.text_input("Search by Name, Patient ID, or Phone", "").strip()
            if search_query:
                results = PatientRepository.search_patients(db, search_query)
            else:
                results = PatientRepository.get_all(db)

            if results:
                for p in results:
                    with st.expander(f"👤 {p.patient_code}: {p.first_name} {p.last_name} ({p.dob})"):
                        st.write(f"**Gender:** {p.gender} | **Phone:** {p.contact_phone or 'N/A'}")
                        st.write(f"**Medical History:** {p.medical_history or 'None documented'}")
                        st.markdown("**Fracture Cases:**")
                        if p.fracture_cases:
                            for fc in p.fracture_cases:
                                st.markdown(
                                    f"- **{fc.case_code}** ({fc.anatomical_site}) | Injured: `{fc.injury_date}` | Status: `{fc.status}`"
                                )
                                for v in fc.visits:
                                    st.markdown(
                                        f"  - Visit `{v.visit_code}` on `{v.visit_date}` ({v.days_post_injury} days post injury). Pain: `{v.pain_score}/10`"
                                    )
                        else:
                            st.caption("No cases registered for this patient yet.")
            else:
                st.info("No matching patient records found.")

        # Tab 2: Registration
        with t2:
            st.markdown("#### Register New Patient Record")
            with st.form("reg_patient_form"):
                r_first = st.text_input("First Name *")
                r_last = st.text_input("Last Name *")
                r_dob = st.date_input("Date of Birth *", min_value=datetime(1900, 1, 1), max_value=datetime.today())
                r_gender = st.selectbox("Gender *", ["Male", "Female", "Other", "Prefer not to say"])
                r_phone = st.text_input("Contact Phone")
                r_history = st.text_area("Medical History / Comorbidities")
                submit_reg = st.form_submit_button("Save Patient Record")

                if submit_reg:
                    if not r_first.strip() or not r_last.strip():
                        st.error("First and last names are mandatory.")
                    else:
                        dob_str = r_dob.strftime("%Y-%m-%d")
                        try:
                            new_pat, warn = register_new_patient(
                                db=db,
                                first_name=r_first,
                                last_name=r_last,
                                dob=dob_str,
                                gender=r_gender,
                                contact_phone=r_phone,
                                medical_history=r_history,
                                current_user=user_obj
                            )
                            st.success(f"Successfully registered patient {new_pat.patient_code} ({new_pat.first_name} {new_pat.last_name})")
                            if warn:
                                st.warning(warn)
                        except Exception as e:
                            st.error(f"Registration failed: {str(e)}")

        # Tab 3: Create Case & Record Visit
        with t3:
            st.markdown("#### Create Fracture Case or Log Visit")
            all_pats = PatientRepository.get_all(db)
            if not all_pats:
                st.info("Please register a patient first.")
            else:
                pat_options = {f"{p.patient_code}: {p.first_name} {p.last_name}": p.id for p in all_pats}
                sel_pat_label = st.selectbox("Select Patient", list(pat_options.keys()))
                sel_pat_id = pat_options[sel_pat_label]
                sel_patient = PatientRepository.get_by_id(db, sel_pat_id)

                sub_t1, sub_t2 = st.tabs(["New Fracture Case", "Record Follow-up Visit"])

                with sub_t1:
                    with st.form("new_case_form"):
                        c_site = st.selectbox("Anatomical Fracture Site", [
                            "Distal Radius", "Radial Shaft", "Ulna", "Tibial Shaft",
                            "Tibia Plateau", "Fibula", "Femoral Shaft", "Femoral Neck",
                            "Clavicle", "Humerus", "Metacarpal", "Ankle", "Other"
                        ])
                        c_injury_date = st.date_input("Date of Injury *", max_value=datetime.today())
                        c_mechanism = st.text_input("Injury Mechanism (e.g. Fall, Sports, Motor Vehicle)")
                        c_treatment = st.selectbox("Initial Stabilization Method", [
                            "Closed Reduction & Cast", "Splint", "ORIF Plate & Screws",
                            "Intramedullary Nail", "External Fixator", "Conservative / Sling"
                        ])
                        c_notes = st.text_area("Clinical Case Notes")
                        submit_case = st.form_submit_button("Create Case")

                        if submit_case:
                            try:
                                case = create_patient_fracture_case(
                                    db=db,
                                    patient_id=sel_pat_id,
                                    anatomical_site=c_site,
                                    injury_date=c_injury_date.strftime("%Y-%m-%d"),
                                    injury_mechanism=c_mechanism,
                                    initial_treatment=c_treatment,
                                    notes=c_notes,
                                    current_user=user_obj
                                )
                                st.success(f"Fracture Case `{case.case_code}` created successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to create case: {str(e)}")

                with sub_t2:
                    if not sel_patient.fracture_cases:
                        st.info("No fracture cases registered for this patient. Create a case first.")
                    else:
                        case_options = {f"{c.case_code} ({c.anatomical_site}, Injured {c.injury_date})": c.id for c in sel_patient.fracture_cases}
                        sel_case_label = st.selectbox("Select Fracture Case", list(case_options.keys()))
                        sel_case_id = case_options[sel_case_label]

                        from datetime import timedelta
                        with st.form("new_visit_form"):
                            v_date = st.date_input("Visit Date *", max_value=datetime.today())
                            v_weight = st.selectbox("Weight Bearing Status", [
                                "Non-weight-bearing", "Touch-down weight-bearing",
                                "Partial weight-bearing", "Full weight-bearing as tolerated"
                            ])
                            v_pain = st.slider("Reported Pain Score (VAS 0-10)", 0, 10, 4)
                            v_notes = st.text_area("Visit Examination & Physical Findings")
                            schedule_next = st.checkbox("Schedule Follow-Up Appointment", value=True)
                            default_next = v_date + timedelta(days=14)
                            v_next_appt = st.date_input("Scheduled Next Appointment (Future)", value=default_next, min_value=v_date + timedelta(days=1))
                            submit_visit = st.form_submit_button("Record Clinical Visit")

                            if submit_visit:
                                try:
                                    next_date_str = v_next_appt.strftime("%Y-%m-%d") if schedule_next else None
                                    visit = record_patient_visit(
                                        db=db,
                                        case_id=sel_case_id,
                                        visit_date=v_date.strftime("%Y-%m-%d"),
                                        clinical_notes=v_notes,
                                        weight_bearing_status=v_weight,
                                        pain_score=v_pain,
                                        scheduled_next_date=next_date_str,
                                        current_user=user_obj
                                    )
                                    st.success(f"Visit `{visit.visit_code}` recorded ({visit.days_post_injury} days post injury).")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to record visit: {str(e)}")


# ------------------------------------------------------------------------------
# 4. X-ray Upload & Validation (Doctor & Admin)
# ------------------------------------------------------------------------------
elif selected_nav == "📤 X-ray Upload & Validation":
    st.markdown("### 📤 X-ray Ingestion & Technical Validation")

    with get_db() as db:
        user_obj = get_current_user_obj(db)

        # Select visit target
        all_cases = db.query(FractureCase).all()
        if not all_cases:
            st.info("No fracture cases in database. Please register a patient and case first.")
            st.stop()

        case_map = {f"{c.case_code} — {c.patient.first_name} {c.patient.last_name} ({c.anatomical_site})": c.id for c in all_cases}
        sel_c_label = st.selectbox("Select Fracture Case", list(case_map.keys()))
        case_id = case_map[sel_c_label]
        case_obj = FractureCaseRepository.get_by_id(db, case_id)

        if not case_obj.visits:
            st.warning("This fracture case has no clinical visits logged. Please record a visit first.")
            st.stop()

        visit_map = {f"{v.visit_code} (Date: {v.visit_date}, Day {v.days_post_injury})": v.id for v in case_obj.visits}
        sel_v_label = st.selectbox("Associate Scan with Visit", list(visit_map.keys()))
        visit_id = visit_map[sel_v_label]

        uploaded_file = st.file_uploader(
            "Upload Bone Radiograph (JPEG or PNG, Max 15MB, Min 256x256)",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            filename = uploaded_file.name

            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.markdown("#### Technical Validation Analysis")
                try:
                    val_result = validate_xray_upload(file_bytes, filename)
                    st.success("✅ File Structure & Content Validation Passed")

                    st.write(f"- **Format:** `{val_result['format']}`")
                    st.write(f"- **Dimensions:** `{val_result['width']} × {val_result['height']} px`")
                    st.write(f"- **File Size:** `{val_result['file_size'] / 1024:.1f} KB`")
                    st.write(f"- **Laplacian Blur Score:** `{val_result['blur_score']:.1f}` (Threshold: `{BLUR_LAPLACIAN_THRESHOLD}`)")
                    st.write(f"- **Dynamic Contrast Std:** `{val_result['contrast_score']:.1f}` (Threshold: `{LOW_CONTRAST_THRESHOLD}`)")
                    st.code(f"SHA-256: {val_result['sha256']}", language="text")

                    # Check for duplicates in DB
                    existing_hash = XrayImageRepository.find_by_hash(db, val_result["sha256"])
                    if existing_hash:
                        st.warning(f"⚠️ Duplicate Detected: Exact same scan already uploaded under image #{existing_hash.id} (Visit `{existing_hash.visit.visit_code}`).")

                    # Quality Warnings
                    if val_result["warnings"]:
                        for w in val_result["warnings"]:
                            st.warning(f"⚠️ {w}")
                    else:
                        st.info("No optical quality degradation alerts triggered.")

                    st.markdown(f"**Medical Notice:** `{val_result['medical_suitability_notice']}`")

                    # Save button
                    if st.button("Confirm Ingestion & Save Scan", use_container_width=True):
                        raw_path, rel_path = save_raw_image(file_bytes, filename)
                        proc_bytes = generate_processed_review_copy(raw_path)
                        proc_path, proc_rel = save_processed_image(proc_bytes, rel_path)

                        img_rec = XrayImageRepository.create_image(
                            db=db,
                            visit_id=visit_id,
                            file_path=rel_path,
                            original_filename=filename,
                            file_size=val_result["file_size"],
                            image_hash=val_result["sha256"],
                            dhash=val_result["dhash"],
                            processed_path=proc_rel,
                            status=STATUS_READY_FOR_ANALYSIS if not val_result["warnings"] else STATUS_NEEDS_REVIEW
                        )

                        XrayImageRepository.save_validation_result(
                            db=db,
                            image_id=img_rec.id,
                            is_valid=True,
                            format=val_result["format"],
                            width=val_result["width"],
                            height=val_result["height"],
                            blur_score=val_result["blur_score"],
                            contrast_score=val_result["contrast_score"],
                            min_brightness=val_result["min_brightness"],
                            max_brightness=val_result["max_brightness"],
                            duplicate_of_id=existing_hash.id if existing_hash else None,
                            warnings=val_result["warnings"]
                        )

                        # Extract experimental features immediately
                        with Image.open(raw_path) as raw_im:
                            feats = extract_experimental_features(np.array(raw_im.convert("RGB")))
                        AIAnalysisRepository.save_features(
                            db=db,
                            image_id=img_rec.id,
                            gap_px=feats["fracture_gap_px"],
                            callus_ratio=feats["callus_density_ratio"],
                            continuity_idx=feats["cortical_continuity_index"],
                            details=feats
                        )

                        log_audit_event(
                            db=db,
                            action="UPLOAD_XRAY",
                            user=user_obj,
                            entity_type="XrayImage",
                            entity_id=str(img_rec.id),
                            details={"filename": filename, "visit_id": visit_id}
                        )

                        st.success(f"Scan saved as Image #{img_rec.id} (Status: `{img_rec.status}`).")
                        st.rerun()

                except XrayValidationError as e:
                    st.error(f"❌ Validation Rejected: {str(e)}")

            with col_right:
                st.markdown("#### Upload Preview")
                try:
                    pil_preview = Image.open(io.BytesIO(file_bytes))
                    st.image(pil_preview, caption="Uploaded Original Radiograph", use_container_width=True)
                except Exception:
                    st.caption("Cannot render image preview.")


# ------------------------------------------------------------------------------
# 5. AI Fracture Detection (Doctor & Admin)
# ------------------------------------------------------------------------------
elif selected_nav == "🧠 AI Fracture Detection":
    st.markdown("### 🧠 AI Fracture Detection & Analysis")
    st.caption("Runs offline YOLOv8 fracture detection pipeline on eligible ingested radiographs.")

    with get_db() as db:
        user_obj = get_current_user_obj(db)

        images = db.query(XrayImage).order_by(XrayImage.id.desc()).all()
        if not images:
            st.info("No radiographs uploaded yet. Please ingest scans first.")
            st.stop()

        img_choices = {
            f"Image #{im.id} — Visit `{im.visit.visit_code}` ({im.visit.fracture_case.patient.first_name} {im.visit.fracture_case.patient.last_name}, {im.visit.fracture_case.anatomical_site}) [Status: {im.status}]": im.id
            for im in images
        }
        sel_im_label = st.selectbox("Select Radiograph for AI Analysis", list(img_choices.keys()))
        selected_img_id = img_choices[sel_im_label]
        img_record = XrayImageRepository.get_by_id(db, selected_img_id)

        col_img, col_det = st.columns([1, 1])

        # Load image
        raw_full_path = BASE_DIR / img_record.file_path

        with col_img:
            st.markdown("#### Radiograph View")
            view_mode = st.radio("Display Mode", ["Original Raw Scan", "CLAHE Contrast-Enhanced Review Copy"], horizontal=True)

            if view_mode == "Original Raw Scan" and raw_full_path.exists():
                st.image(str(raw_full_path), caption=f"Raw Original #{img_record.id}", use_container_width=True)
            elif view_mode == "CLAHE Contrast-Enhanced Review Copy" and img_record.processed_path:
                proc_full = BASE_DIR / img_record.processed_path
                if proc_full.exists():
                    st.image(str(proc_full), caption="CLAHE Enhanced Review Copy (Non-destructive)", use_container_width=True)
                else:
                    st.image(str(raw_full_path), caption="Raw Fallback", use_container_width=True)

        with col_det:
            st.markdown("#### AI Detection Engine")

            conf_slider = st.slider(
                "Confidence Cutoff Threshold",
                0.05, 0.90, 0.20, 0.05,
                help="Minimum detector confidence required to display candidate bounding boxes. Set lower (e.g. 0.15) for subtle or hairline fractures."
            )

            # Check detector status
            detector = get_detector()
            if not detector.is_loaded:
                st.warning(f"⚠️ Model Weights Notice: `{detector.load_error}`")
                st.caption("The system is functioning in unassisted clinician review mode. To enable automated bounding boxes, train weights via `python training/train_detection.py`.")

            if st.button("Execute Fracture Detection Pipeline", use_container_width=True):
                with st.spinner("Executing YOLOv8 inference pipeline..."):
                    success, analysis_res, msg = run_xray_analysis(
                        db=db, image_id=selected_img_id, conf_threshold=conf_slider, current_user=user_obj
                    )
                    if success:
                        st.success("✅ Analysis completed successfully!")
                        st.rerun()
                    else:
                        st.error(f"Analysis failed: {msg}")

            # Display existing analysis if present
            if img_record.ai_analysis:
                st.markdown("---")
                st.markdown("#### Candidate Detection Findings")
                ai_res = img_record.ai_analysis
                st.write(f"**Analysis Status:** `{ai_res.status}`")
                st.write(f"**Mean Confidence Score:** `{ai_res.confidence_summary if ai_res.confidence_summary is not None else 'N/A'}`")

                try:
                    boxes = json.loads(ai_res.detections_json)
                    if boxes:
                        df_boxes = pd.DataFrame(boxes)
                        st.dataframe(df_boxes, use_container_width=True)

                        # Render bounding box overlay
                        with Image.open(raw_full_path) as im_pil:
                            from PIL import ImageDraw
                            overlay_img = im_pil.convert("RGB")
                            draw = ImageDraw.Draw(overlay_img)
                            for b in boxes:
                                coords = b["bbox"]
                                conf = b["confidence"]
                                cls_name = b["class_name"]
                                draw.rectangle(coords, outline="red", width=3)
                                draw.text((coords[0], max(0, coords[1] - 12)), f"{cls_name} ({conf:.2f})", fill="red")
                            st.image(overlay_img, caption="AI Fracture Candidate Overlay", use_container_width=True)
                    else:
                        st.info("ℹ️ No fracture candidates met the current confidence threshold.")
                        st.caption("💡 **How to Rectify:** Subtle fractures or hairline fissures often have lower raw detector scores (e.g., 0.15–0.25). Move the **Confidence Cutoff Threshold** slider above down to **0.15** or **0.10**, then click **Execute Fracture Detection Pipeline** again.")
                except Exception as e:
                    st.caption(f"Error parsing detections: {str(e)}")


# ------------------------------------------------------------------------------
# 6. 2D Digital Twin (Doctor & Admin)
# ------------------------------------------------------------------------------
elif selected_nav == "🦴 2D Digital Twin":
    st.markdown("### 🦴 2D Digital Twin & Longitudinal Tracking")
    st.caption("Chronological radiographic alignment, side-by-side visit comparison, and experimental feature extraction.")

    with get_db() as db:
        cases = db.query(FractureCase).all()
        if not cases:
            st.info("No fracture cases found.")
            st.stop()

        case_dict = {f"{c.case_code} — {c.patient.first_name} {c.patient.last_name} ({c.anatomical_site})": c.id for c in cases}
        sel_c_label = st.selectbox("Select Fracture Case for Digital Twin", list(case_dict.keys()))
        c_obj = FractureCaseRepository.get_by_id(db, case_dict[sel_c_label])

        visits = sorted(c_obj.visits, key=lambda v: v.visit_date)
        if len(visits) < 2:
            st.info("At least two follow-up visits with radiographs are required for longitudinal digital twin comparisons.")
            st.stop()

        # Side by side visual selector
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            v1_opts = {f"Visit `{v.visit_code}` ({v.visit_date}, Day {v.days_post_injury})": v.id for v in visits}
            sel_v1_id = st.selectbox("Baseline / Earlier Visit", list(v1_opts.keys()), index=0)
            v1_id = v1_opts[sel_v1_id]
        with c_v2:
            v2_opts = {f"Visit `{v.visit_code}` ({v.visit_date}, Day {v.days_post_injury})": v.id for v in visits}
            sel_v2_id = st.selectbox("Follow-up / Later Visit", list(v2_opts.keys()), index=len(visits) - 1)
            v2_id = v2_opts[sel_v2_id]

        v1_obj = VisitRepository.get_by_id(db, v1_id)
        v2_obj = VisitRepository.get_by_id(db, v2_id)

        if v1_obj.xray_images and v2_obj.xray_images:
            img_a_path = BASE_DIR / v1_obj.xray_images[0].file_path
            img_b_path = BASE_DIR / v2_obj.xray_images[0].file_path

            if img_a_path.exists() and img_b_path.exists():
                np_a, np_b, diff, comp_warnings = compare_scans_visual(img_a_path, img_b_path)

                col_a, col_b, col_diff = st.columns(3)
                with col_a:
                    st.image(np_a, caption=f"Earlier: {v1_obj.visit_code} (Day {v1_obj.days_post_injury})", use_container_width=True)
                with col_b:
                    st.image(np_b, caption=f"Later: {v2_obj.visit_code} (Day {v2_obj.days_post_injury})", use_container_width=True)
                with col_diff:
                    st.image(diff, caption="Absolute Difference Map (Uncalibrated)", use_container_width=True)

                if comp_warnings:
                    for cw in comp_warnings:
                        st.warning(f"⚠️ {cw}")

        st.markdown("---")
        st.markdown("#### Longitudinal Progress Trajectory (Observed History)")

        # Prepare visits data for Plotly
        visits_plot_data = []
        for v in visits:
            gap = None
            dens = None
            if v.xray_images and v.xray_images[0].extracted_features:
                gap = v.xray_images[0].extracted_features.fracture_gap_px
                dens = v.xray_images[0].extracted_features.callus_density_ratio

            visits_plot_data.append({
                "visit_code": v.visit_code,
                "visit_date": v.visit_date,
                "days_post_injury": v.days_post_injury,
                "fracture_gap_px": gap or 0.0,
                "callus_density_ratio": dens or 1.0,
                "pain_score": v.pain_score
            })

        plotly_fig = build_longitudinal_trajectory_plot(visits_plot_data)
        st.plotly_chart(plotly_fig, use_container_width=True)

        st.info(f"ℹ️ {DIGITAL_TWIN_DISCLAIMER}")


# ------------------------------------------------------------------------------
# 7. Reliability & Confidence Engine (Doctor & Admin)
# ------------------------------------------------------------------------------
elif selected_nav == "🛡️ Reliability & Confidence":
    st.markdown("### 🛡️ Reliability Summary & Uncertainty Engine")
    st.caption("Multi-factor technical inspection, model uncertainty zones, and missing data alerts.")

    with get_db() as db:
        images = db.query(XrayImage).order_by(XrayImage.id.desc()).all()
        if not images:
            st.info("No radiographs in database.")
            st.stop()

        im_opts = {f"Image #{im.id} (Visit `{im.visit.visit_code}`, {im.visit.fracture_case.patient.first_name} {im.visit.fracture_case.patient.last_name})": im.id for im in images}
        sel_im_id = im_opts[st.selectbox("Select Radiograph to Inspect", list(im_opts.keys()))]
        im_rec = XrayImageRepository.get_by_id(db, sel_im_id)

        summary = compute_reliability_summary(
            validation=im_rec.validation_result,
            analysis=im_rec.ai_analysis,
            visit=im_rec.visit,
            model_tag="YOLOv8-Academic"
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 1. Technical Image Quality")
            t_qual = summary["technical_quality"]
            st.write(f"- **Sharpness (Laplacian):** `{t_qual['blur_laplacian']}` (Threshold: `{BLUR_LAPLACIAN_THRESHOLD}`)")
            st.write(f"- **Contrast (Std Dev):** `{t_qual['contrast_std']}` (Threshold: `{LOW_CONTRAST_THRESHOLD}`)")
            st.write(f"- **Quality Status:** `{t_qual['status']}`")
            if t_qual["warnings"]:
                for w in t_qual["warnings"]:
                    st.warning(f"⚠️ {w}")
            else:
                st.success("Quality within acceptable range.")

        with col2:
            st.markdown("#### 2. Model Detection Evaluation")
            d_eval = summary["detection_evaluation"]
            st.write(f"- **Detection Status:** `{d_eval['status']}`")
            st.write(f"- **Candidate Count:** `{d_eval['candidate_count']}`")
            st.write(f"- **Raw Confidence Score:** `{d_eval['raw_confidence_score']}`")
            st.caption(f"Notice: {d_eval['score_interpretation_notice']}")

        st.markdown("---")
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### 3. Clinical Data Completeness")
            missing = summary["missing_clinical_context"]
            if missing:
                for m in missing:
                    st.warning(f"⚠️ {m}")
            else:
                st.success("All standard clinical parameters recorded.")

        with c4:
            st.markdown("#### 4. Clinician Review Status")
            rev = summary["clinician_review"]
            st.write(f"- **Status:** `{rev['status']}`")
            st.write(f"- **Clinician Notes:** `{rev['notes']}`")


# ------------------------------------------------------------------------------
# 8. Healing Progress & Prediction Gating (Doctor & Admin)
# ------------------------------------------------------------------------------
elif selected_nav == "📈 Healing Progress":
    st.markdown("### 📈 Longitudinal Healing Progress & Recovery Tracking")
    st.caption("Chronological tracking of clinical recovery milestones, symptom improvement, and experimental radiographic biomarkers.")

    with get_db() as db:
        cases = db.query(FractureCase).all()
        if not cases:
            st.info("No fracture cases found.")
            st.stop()

        case_opts = {f"{c.case_code} — {c.patient.first_name} {c.patient.last_name} ({c.anatomical_site}, Injured: {c.injury_date})": c.id for c in cases}
        sel_c_id = case_opts[st.selectbox("Select Patient Fracture Case", list(case_opts.keys()))]
        c_obj = FractureCaseRepository.get_by_id(db, sel_c_id)

        # Compute structured longitudinal progress
        prog = compute_historical_healing_progression(c_obj)

        if not prog["has_data"] or not prog["timeline"]:
            st.warning("⚠️ No follow-up visits recorded for this fracture case yet.")
            st.stop()

        summary = prog.get("summary", {})
        timeline = prog["timeline"]
        latest = summary.get("latest_visit") or timeline[-1]
        baseline = summary.get("baseline_visit") or timeline[0]

        # ----------------------------------------------------------------------
        # 1. Clinical Summary Metric Cards (Top Banner)
        # ----------------------------------------------------------------------
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            lat_days = summary.get("latest_days", latest.get("days_post_injury", 0))
            base_days = baseline.get("days_post_injury", 0)
            st.metric(
                label="⏱️ Elapsed Post-Injury",
                value=f"{lat_days} Days",
                delta=f"+{lat_days - base_days}d since baseline" if lat_days > 0 else "Baseline Day 0",
                delta_color="off"
            )
        with m2:
            current_pain = latest.get("pain_score")
            pain_delta_str = None
            if summary.get("pain_delta") is not None:
                pain_delta_str = f"{summary['pain_delta']} pts vs Day 0"
            st.metric(
                label="🩹 Reported Pain Level",
                value=f"{current_pain} / 10" if current_pain is not None else "N/A",
                delta=pain_delta_str,
                delta_color="inverse"
            )
        with m3:
            st.metric(
                label="🚶 Weight-Bearing Status",
                value=latest.get("weight_bearing_status", "Unspecified"),
                delta="Latest Clinical Order",
                delta_color="off"
            )
        with m4:
            lat_dens = latest.get("callus_density_ratio")
            dens_delta_str = None
            if latest.get("density_change_pct") is not None:
                dens_delta_str = f"{latest['density_change_pct']:+.1f}% vs baseline"
            st.metric(
                label="🔬 Callus Mineralization",
                value=f"{lat_dens:.2f}" if lat_dens is not None else "Unmeasured",
                delta=dens_delta_str,
                delta_color="normal"
            )

        st.markdown("---")

        # ----------------------------------------------------------------------
        # 2. Biological Healing Phase Timeline
        # ----------------------------------------------------------------------
        st.markdown("#### 🧬 Biological Bone Healing Phase")
        bio_prog = summary.get("biological_progress", 0.5)
        st.progress(bio_prog)
        st.markdown(f"**Current Estimated Phase:** `{summary.get('biological_phase', 'In Progress')}`")
        st.caption(f"**Biological Reference:** {summary.get('biological_phase_description', 'Natural reparative tissue osteogenesis.')}")

        with st.expander("ℹ️ Clinical Reference: Standard 4 Phases of Fracture Repair"):
            st.markdown("""
            1. **Phase 1: Inflammatory & Hematoma (Days 0–5):** Fractured blood vessels form a hematoma; inflammatory cytokines and growth factors recruit osteoprogenitor cells.
            2. **Phase 2: Reparative / Soft Callus (Days 5–21):** Fibrocartilage forms an internal/external bridge across bone ends to mechanically stabilize the fracture site.
            3. **Phase 3: Hard Callus / Bony Consolidation (Weeks 3–8):** Endochondral ossification transforms soft callus into rigid woven bone (trabecular bridging across cortices).
            4. **Phase 4: Cortical Bone Remodeling (Months 2+):** Osteoclastic and osteoblastic activity replaces woven bone with lamellar bone and restores the medullary canal.

            *Reference timelines reflect clinical guidelines for typical long bone fractures. Healing varies with vascularity, mechanical fixation stability, patient age, and nutritional status.*
            """)

        st.markdown("---")

        # ----------------------------------------------------------------------
        # 3. Main Tabs: Visual Progress vs Film Strip vs Safety Gating
        # ----------------------------------------------------------------------
        t_prog, t_film, t_pred = st.tabs([
            "📊 Longitudinal Charts & Clinical Table",
            "🖼️ Chronological Radiograph Gallery",
            "🚫 Predictive Trajectory Gating (Medical Safety)"
        ])

        with t_prog:
            st.markdown("#### Longitudinal Metric Trajectories")
            # Interactive multi-trace plot
            fig = build_longitudinal_trajectory_plot(timeline)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("🔍 How to interpret these longitudinal trajectories"):
                st.markdown("""
                - **Reported Pain Score (Yellow Diamonds):** Subjective clinical symptom reporting on a scale from 0 (no pain) to 10 (worst pain). A downward trend indicates symptom resolution.
                - **Fracture Gap Discontinuity (Red Solid Line):** Estimated width (in pixels) of the fracture line discontinuity. As hard callus bridges the cortices, this measurement typically trends downward.
                - **Callus Density Ratio (Blue Dotted Line):** Optical density ratio of periosteal callus relative to cortical bone. A value $> 1.0$ indicates progressive mineral deposition.
                - *⚠️ Note: Gaps in the trendline indicate visits where radiographs did not have a calibrated region of interest. The system strictly avoids fabricating zero values.*
                """)

            st.markdown("#### Detailed Visit-by-Visit Clinical Record")
            # Formatted clean table
            table_rows = []
            for item in timeline:
                p_val = item.get("pain_score")
                p_badge = "N/A"
                if p_val is not None:
                    if p_val <= 3:
                        p_badge = f"🟢 Mild ({p_val}/10)"
                    elif p_val <= 6:
                        p_badge = f"🟡 Moderate ({p_val}/10)"
                    else:
                        p_badge = f"🔴 Severe ({p_val}/10)"

                gap_str = f"{item['fracture_gap_px']:.1f} px" if item.get("fracture_gap_px") is not None else "Unmeasured"
                if item.get("gap_reduction_pct") is not None:
                    gap_str += f" ({item['gap_reduction_pct']:+.1f}%)"

                dens_str = f"{item['callus_density_ratio']:.2f}" if item.get("callus_density_ratio") is not None else "Unmeasured"
                if item.get("density_change_pct") is not None:
                    dens_str += f" ({item['density_change_pct']:+.1f}%)"

                table_rows.append({
                    "Visit Code": item["visit_code"],
                    "Date": item["visit_date"],
                    "Elapsed Time": f"Day {item['days_post_injury']}",
                    "Pain Level": p_badge,
                    "Mobility Status": item["weight_bearing_status"],
                    "Fracture Gap Proxy": gap_str,
                    "Callus Density": dens_str,
                    "Clinical Notes": item["clinical_notes"]
                })

            st.table(pd.DataFrame(table_rows))
            st.info(prog["safety_notice"])

        with t_film:
            st.markdown("#### Chronological Radiographic Follow-up Gallery")
            st.caption("Side-by-side progression of serial radiographs to visually inspect cortical bridging and hardware alignment.")

            cols = st.columns(min(len(timeline), 3))
            for idx, item in enumerate(timeline):
                col_idx = idx % min(len(timeline), 3)
                with cols[col_idx]:
                    st.markdown(f"**{item['visit_code']} (Day {item['days_post_injury']})**")
                    st.caption(f"📅 Date: `{item['visit_date']}` | Pain: `{item['pain_score']}/10`")
                    if item.get("image_path"):
                        full_img_p = BASE_DIR / item["image_path"]
                        if full_img_p.exists():
                            st.image(str(full_img_p), use_container_width=True)
                        else:
                            st.caption("Scan file not found on disk.")
                    else:
                        st.caption("No radiograph attached to this visit.")
                    st.info(f"📝 *Notes:* {item['clinical_notes']}")

        with t_pred:
            st.markdown("#### Future Predictive Trajectory Gating")
            pred_status = get_healing_prediction(c_obj.case_code, len(c_obj.visits))
            st.error(f"🚫 Status: `{pred_status['status']}`")
            st.markdown("**Why is future healing prediction gated?**")
            st.markdown(pred_status["explanation"])

            st.markdown("---")
            st.markdown("#### 📋 Empirical Criteria Required Before Activation:")
            for req in pred_status["requirements_for_enablement"]:
                st.markdown(f"- {req}")

            st.markdown("---")
            st.warning("""
            **Clinical Guidance for Healthcare Providers:**
            Never rely on an algorithm to predict future union dates or determine when to remove a cast.
            Assess healing according to standard orthopedic criteria:
            - **Radiographic:** Bridging of at least 3 out of 4 cortices on AP and lateral radiographs.
            - **Clinical:** Absence of tenderness on direct palpation across the fracture line.
            - **Functional:** Pain-free weight-bearing or full active range-of-motion.
            """)
            st.caption(pred_status["disclaimer"])


# ------------------------------------------------------------------------------
# 9. Doctor Review & PDF Reports (Doctor & Admin)
# ------------------------------------------------------------------------------
elif selected_nav == "📋 Doctor Review & Reports":
    st.markdown("### 📋 Clinician Review Workspace & PDF Summary Export")

    with get_db() as db:
        user_obj = get_current_user_obj(db)

        images = db.query(XrayImage).order_by(XrayImage.id.desc()).all()
        if not images:
            st.info("No radiographs uploaded.")
            st.stop()

        im_opts = {
            f"Image #{im.id}: Visit `{im.visit.visit_code}` ({im.visit.fracture_case.patient.first_name} {im.visit.fracture_case.patient.last_name}, {im.visit.fracture_case.anatomical_site}) [Status: {im.status}]": im.id
            for im in images
        }
        sel_im_id = im_opts[st.selectbox("Select Scan to Review", list(im_opts.keys()))]
        im_rec = XrayImageRepository.get_by_id(db, sel_im_id)

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### Scan Inspection")
            img_path = BASE_DIR / im_rec.file_path
            if img_path.exists():
                st.image(str(img_path), caption=f"Scan #{im_rec.id}", use_container_width=True)

            if im_rec.ai_analysis:
                ai = im_rec.ai_analysis
                st.write(f"**Candidate Detections:** {ai.detections_json}")

        with col_right:
            st.markdown("#### Clinician Attestation & Sign-off")
            current_status = "PENDING"
            existing_notes = ""
            if im_rec.ai_analysis and im_rec.ai_analysis.clinician_review:
                current_status = im_rec.ai_analysis.clinician_review.review_status
                existing_notes = im_rec.ai_analysis.clinician_review.clinical_notes

            with st.form("attestation_form"):
                review_choice = st.selectbox(
                    "Clinical Review Decision",
                    ["APPROVED", "MODIFIED", "REJECTED"],
                    index=["APPROVED", "MODIFIED", "REJECTED"].index(current_status) if current_status in ["APPROVED", "MODIFIED", "REJECTED"] else 0
                )
                clinician_notes = st.text_area(
                    "Clinician Review Rationale & Observational Notes *",
                    value=existing_notes,
                    height=120
                )
                submit_review = st.form_submit_button("Submit Binding Review Sign-Off")

                if submit_review:
                    if not im_rec.ai_analysis:
                        st.error("Cannot sign off on scan before AI analysis record is initialized.")
                    else:
                        ok, rev_obj, msg = submit_clinician_attestation(
                            db=db,
                            analysis_id=im_rec.ai_analysis.id,
                            reviewer_user=user_obj,
                            review_status=review_choice,
                            clinical_notes=clinician_notes
                        )
                        if ok:
                            st.success(f"Review recorded successfully ({review_choice})!")
                            st.rerun()
                        else:
                            st.error(msg)

            st.markdown("---")
            st.markdown("#### Export Formal PDF Decision Support Report")

            if st.button("Generate & Download ReportLab PDF", use_container_width=True):
                with st.spinner("Generating official PDF document..."):
                    v = im_rec.visit
                    c = v.fracture_case
                    p = c.patient

                    ai_data = None
                    if im_rec.ai_analysis:
                        try:
                            ai_data = {"detections": json.loads(im_rec.ai_analysis.detections_json)}
                        except Exception:
                            pass

                    rev_data = None
                    if im_rec.ai_analysis and im_rec.ai_analysis.clinician_review:
                        r = im_rec.ai_analysis.clinician_review
                        rev_data = {
                            "review_status": r.review_status,
                            "reviewer_name": r.reviewer.full_name,
                            "signed_at": r.signed_at.strftime("%Y-%m-%d %H:%M"),
                            "clinical_notes": r.clinical_notes
                        }

                    import uuid
                    from config import STORAGE_DIR

                    rel_sum = compute_reliability_summary(im_rec.validation_result, im_rec.ai_analysis, v)

                    unique_suffix = uuid.uuid4().hex[:6].upper()
                    rep_code = f"REP-{datetime.now().strftime('%Y%m%d')}-{im_rec.id:04d}-{unique_suffix}"

                    scan_path = None
                    if im_rec.processed_path:
                        scan_path = (STORAGE_DIR.parent / im_rec.processed_path).resolve()
                    elif im_rec.file_path:
                        scan_path = (STORAGE_DIR.parent / im_rec.file_path).resolve()

                    pdf_p = generate_pdf_report(
                        report_code=rep_code,
                        patient_info={"patient_code": p.patient_code, "full_name": f"{p.first_name} {p.last_name}", "dob": p.dob, "gender": p.gender},
                        case_info={"case_code": c.case_code, "anatomical_site": c.anatomical_site, "injury_date": c.injury_date},
                        visit_info={"visit_code": v.visit_code, "visit_date": v.visit_date, "days_post_injury": v.days_post_injury},
                        ai_analysis_info=ai_data,
                        clinician_review_info=rev_data,
                        reliability_summary=rel_sum,
                        scan_image_path=scan_path
                    )

                    ReportRepository.save_report(db, v.id, str(pdf_p), user_obj.id, is_final=True)

                    with open(pdf_p, "rb") as f:
                        pdf_bytes = f.read()

                    st.download_button(
                        label="⬇️ Download Generated PDF Report",
                        data=pdf_bytes,
                        file_name=f"{rep_code}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )


# ------------------------------------------------------------------------------
# 10. Patient Portal (Patient Only)
# ------------------------------------------------------------------------------
elif selected_nav == "👤 Patient Portal":
    st.markdown("### 👤 Patient Portal — My Healing Journey")

    with get_db() as db:
        user_obj = get_current_user_obj(db)
        portal_data = get_patient_portal_data(db, user_obj)

        if not portal_data["has_profile"]:
            st.info(portal_data["message"])
        else:
            p_info = portal_data["patient"]
            st.markdown(f"**Patient:** {p_info['full_name']} | **ID:** `{p_info['patient_code']}` | **DOB:** `{p_info['dob']}`")

            # Simulated Notifications Banner
            if portal_data["simulated_notifications"]:
                for notif in portal_data["simulated_notifications"]:
                    st.warning(f"🔔 **[{notif['label']}]** {notif['message']}")

            st.markdown("---")
            st.markdown("#### My Active Fracture Cases")

            for c in portal_data["cases"]:
                with st.expander(f"🦴 {c['anatomical_site']} Fracture (Injured: {c['injury_date']})", expanded=True):
                    st.write(f"**Case Reference:** `{c['case_code']}` | **Status:** `{c['status']}`")

                    st.markdown("##### Follow-Up Visits & Clinician-Approved Summaries")
                    if c["visits"]:
                        for v in c["visits"]:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"**Visit {v['visit_code']}** ({v['visit_date']} — Day {v['days_post_injury']})")
                                st.write(f"- Activity: `{v['weight_bearing_status']}`")
                            with col2:
                                if v["has_approved_report"]:
                                    st.success("Summary Approved")
                                    for rep in v["reports"]:
                                        pdf_file = Path(rep["pdf_path"])
                                        if pdf_file.exists():
                                            with open(pdf_file, "rb") as f:
                                                st.download_button(
                                                    f"Download Report ({rep['report_code']})",
                                                    data=f.read(),
                                                    file_name=f"{rep['report_code']}.pdf",
                                                    mime="application/pdf",
                                                    key=f"dl_{rep['report_code']}"
                                                )
                                else:
                                    st.caption("Awaiting Doctor Sign-off")
                    else:
                        st.caption("No visits logged yet.")


# ------------------------------------------------------------------------------
# 11. Patient Educational Disclaimer
# ------------------------------------------------------------------------------
elif selected_nav in ("ℹ️ Educational Disclaimer", "ℹ️ System Overview & Disclaimer"):
    st.markdown("### ℹ️ Medical Information & Disclaimer")
    st.markdown(f"""
    > **NOTICE:** {ACADEMIC_PROTOTYPE_DISCLAIMER}

    #### Patient Recovery Guidance
    - Bone healing is a biological process involving inflammatory, repair (soft and hard callus formation), and remodeling phases.
    - X-ray radiographs represent 2-dimensional projections of complex 3-dimensional skeletal structures.
    - Never alter your weight-bearing status, remove a splint or cast, or resume sports activities without direct instruction from your treating orthopedic specialist.
    - If you experience sudden swelling, acute pain, numbness, or tingling in your extremities, contact emergency medical services immediately.
    """)


# ------------------------------------------------------------------------------
# 12. Diagnostics & Audit Log (Admin & Doctor)
# ------------------------------------------------------------------------------
elif selected_nav == "⚙️ System Diagnostics & Audit":
    st.markdown("### ⚙️ System Diagnostics & Immutable Audit Trail")

    with get_db() as db:
        user_obj = get_current_user_obj(db)

        t_diag, t_audit, t_audit_doc = st.tabs(["Diagnostics & Storage", "Audit Logs", "Dataset Audit Report"])

        with t_diag:
            st.markdown("#### Hardware & Environment Configuration")
            import torch
            st.write(f"- **Python Version:** `{sys.version.split()[0]}`")
            st.write(f"- **PyTorch Version:** `{torch.__version__}`")
            st.write(f"- **CUDA Available:** `{torch.cuda.is_available()}`")
            st.write(f"- **Model Weights Path:** `{MODEL_WEIGHTS_PATH}`")
            st.write(f"- **Weights File Exists:** `{Path(MODEL_WEIGHTS_PATH).exists()}`")

            st.markdown("#### Model Versions Registered in Database")
            m_vers = db.query(ModelVersionRepository.get_latest(db, task_type="DETECTION")).all() if False else db.query(ModelVersion).all()
            if m_vers:
                mv_data = [{"Model": m.model_name, "Tag": m.version_tag, "Type": m.task_type, "Dataset": m.training_dataset, "Date": m.created_at} for m in m_vers]
                st.dataframe(pd.DataFrame(mv_data), use_container_width=True)
            else:
                st.caption("No model versions formally registered in DB yet.")

        with t_audit:
            st.markdown("#### System Security & Action Audit Trail")
            recent_logs = AuditRepository.list_recent(db, limit=50)
            if recent_logs:
                log_data = []
                for l in recent_logs:
                    log_data.append({
                        "Timestamp": l.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "Action": l.action,
                        "User ID": l.user_id,
                        "Entity": f"{l.entity_type} #{l.entity_id}" if l.entity_type else "System",
                        "Details": l.details_json
                    })
                st.dataframe(pd.DataFrame(log_data), use_container_width=True)
            else:
                st.info("No audit logs recorded.")

        with t_audit_doc:
            st.markdown("#### Empirical Dataset Audit Findings (`DATASET_AUDIT_REPORT.md`)")
            audit_file = BASE_DIR / "data" / "DATASET_AUDIT_REPORT.md"
            if audit_file.exists():
                with open(audit_file, "r", encoding="utf-8") as f:
                    st.markdown(f.read())
            else:
                st.info("Run `python training/dataset_validation.py` to generate the audit report.")
