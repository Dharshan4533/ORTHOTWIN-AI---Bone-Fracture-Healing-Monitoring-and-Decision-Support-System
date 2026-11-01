"""
OrthoTwin AI — Clinical Demonstration Seeder
Seeds realistic, de-identified clinical longitudinal cases with synthetic radiographs
for end-to-end evaluation, testing, and UI demonstration.
"""

import sys
import io
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw
import numpy as np

sys_path = Path(__file__).resolve().parent
if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

from database.db import init_db, get_db
from database.models import User, Patient, FractureCase, Visit
from database.repositories import (
    PatientRepository, FractureCaseRepository, VisitRepository,
    XrayImageRepository, AIAnalysisRepository
)
from modules.xray_validation import validate_xray_upload
from modules.xray_processing import generate_processed_review_copy
from services.file_storage import save_raw_image, save_processed_image
from services.reliability_service import transition_status, STATUS_RESULT_READY, STATUS_CLINICIAN_REVIEWED
from modules.digital_twin import extract_experimental_features
from modules.report_generator import generate_pdf_report


def create_synthetic_radiograph(
    filename: str,
    gap_width: int = 15,
    blur: bool = False,
    density_factor: float = 1.0
) -> bytes:
    """
    Draws a synthetic radiographic projection of a long bone (e.g. radius)
    with a transverse fracture gap and surrounding soft-tissue density.
    """
    img = Image.new("L", (512, 512), color=30)
    draw = ImageDraw.Draw(img)

    # Draw soft tissue ellipse
    draw.ellipse([80, 20, 430, 490], fill=60)

    # Draw cortical bone shaft
    bone_color = int(min(240, 180 * density_factor))
    draw.rectangle([180, 40, 330, 470], fill=bone_color)
    # Medullary canal (slightly darker marrow center)
    marrow_color = int(min(200, 130 * density_factor))
    draw.rectangle([220, 50, 290, 460], fill=marrow_color)

    # Fracture line (transverse gap)
    if gap_width > 0:
        y_pos = 250
        draw.rectangle([170, y_pos - (gap_width // 2), 340, y_pos + (gap_width // 2)], fill=45)
        # Jagged cortex edge
        draw.polygon([(170, y_pos - gap_width), (190, y_pos), (170, y_pos + gap_width)], fill=bone_color)
        draw.polygon([(340, y_pos - gap_width), (320, y_pos), (340, y_pos + gap_width)], fill=bone_color)

    if blur:
        import cv2
        np_arr = np.array(img)
        blurred = cv2.GaussianBlur(np_arr, (15, 15), 5)
        img = Image.fromarray(blurred)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def seed_demo_cases():
    init_db()

    with get_db() as db:
        # Check if demo patient already exists
        p1 = db.query(Patient).filter(Patient.first_name == "Alex", Patient.last_name == "Mercer").first()
        if not p1:
            print("Creating Patient 1: Alex Mercer...")
            p1 = PatientRepository.create_patient(
                db=db,
                first_name="Alex",
                last_name="Mercer",
                dob="1988-04-12",
                gender="Male",
                contact_phone="+1-555-019-4821",
                medical_history="Non-smoker, no prior bone pathologies. Occasional NSAID use."
            )
            # Link to patient_demo user
            user = db.query(User).filter(User.username == "patient_demo").first()
            if user:
                user.patient_id = p1.id
                db.commit()

            # Create Fracture Case
            c1 = FractureCaseRepository.create_case(
                db=db,
                patient_id=p1.id,
                anatomical_site="Distal Radius",
                injury_date="2026-08-01",
                injury_mechanism="Fall on outstretched hand (FOOSH)",
                initial_treatment="Closed reduction and cast immobilization",
                notes="Colles-type distal radial fracture with mild dorsal displacement."
            )

            # Visit 1: Day 0 (Baseline)
            v1 = VisitRepository.create_visit(
                db=db,
                case_id=c1.id,
                visit_date="2026-08-01",
                clinical_notes="Acute post-reduction radiograph. Cast applied. Good alignment.",
                weight_bearing_status="Non-weight-bearing",
                pain_score=8,
                scheduled_next_date="2026-08-15"
            )
            img1_bytes = create_synthetic_radiograph("alex_d0.jpg", gap_width=18, density_factor=0.9)
            val1 = validate_xray_upload(img1_bytes, "alex_distal_radius_day0.jpg")
            raw_path1, rel_path1 = save_raw_image(img1_bytes, "alex_distal_radius_day0.jpg")
            proc_bytes1 = generate_processed_review_copy(raw_path1)
            proc_path1, proc_rel1 = save_processed_image(proc_bytes1, rel_path1)
            img1 = XrayImageRepository.create_image(
                db=db, visit_id=v1.id, file_path=rel_path1, original_filename="alex_distal_radius_day0.jpg",
                file_size=len(img1_bytes), image_hash=val1["sha256"], dhash=val1["dhash"],
                processed_path=proc_rel1, status=STATUS_RESULT_READY
            )
            XrayImageRepository.save_validation_result(
                db=db, image_id=img1.id, is_valid=True, format=val1["format"],
                width=val1["width"], height=val1["height"], blur_score=val1["blur_score"],
                contrast_score=val1["contrast_score"], min_brightness=val1["min_brightness"],
                max_brightness=val1["max_brightness"], warnings=val1["warnings"]
            )
            feat1 = extract_experimental_features(np.array(Image.open(raw_path1)), bbox=[170, 230, 340, 270])
            AIAnalysisRepository.save_features(
                db=db, image_id=img1.id, gap_px=feat1["fracture_gap_px"],
                callus_ratio=feat1["callus_density_ratio"], continuity_idx=feat1["cortical_continuity_index"],
                details=feat1
            )
            AIAnalysisRepository.save_analysis(
                db=db, image_id=img1.id, model_version_id=None,
                detections=[{"bbox": [170, 230, 340, 270], "confidence": 0.89, "class_name": "fracture", "class_id": 1}],
                confidence_summary=0.89, raw_scores={"box_confidences": [0.89], "count": 1},
                status=STATUS_CLINICIAN_REVIEWED
            )
            # Add doctor sign-off
            dr = db.query(User).filter(User.username == "dr_ortho").first()
            if dr:
                AIAnalysisRepository.save_clinician_review(
                    db=db, analysis_id=img1.ai_analysis.id, reviewer_id=dr.id,
                    review_status="APPROVED",
                    clinical_notes="Confirmed transverse distal radius fracture. Alignment acceptable post-reduction."
                )

            # Visit 2: Day 14
            v2 = VisitRepository.create_visit(
                db=db,
                case_id=c1.id,
                visit_date="2026-08-15",
                clinical_notes="Two-week follow-up. Swelling decreasing. Pain controlled in splint.",
                weight_bearing_status="Non-weight-bearing",
                pain_score=4,
                scheduled_next_date="2026-09-12"
            )
            img2_bytes = create_synthetic_radiograph("alex_d14.jpg", gap_width=12, density_factor=1.05)
            val2 = validate_xray_upload(img2_bytes, "alex_distal_radius_day14.jpg")
            raw_path2, rel_path2 = save_raw_image(img2_bytes, "alex_distal_radius_day14.jpg")
            proc_bytes2 = generate_processed_review_copy(raw_path2)
            proc_path2, proc_rel2 = save_processed_image(proc_bytes2, rel_path2)
            img2 = XrayImageRepository.create_image(
                db=db, visit_id=v2.id, file_path=rel_path2, original_filename="alex_distal_radius_day14.jpg",
                file_size=len(img2_bytes), image_hash=val2["sha256"], dhash=val2["dhash"],
                processed_path=proc_rel2, status=STATUS_RESULT_READY
            )
            XrayImageRepository.save_validation_result(
                db=db, image_id=img2.id, is_valid=True, format=val2["format"],
                width=val2["width"], height=val2["height"], blur_score=val2["blur_score"],
                contrast_score=val2["contrast_score"], min_brightness=val2["min_brightness"],
                max_brightness=val2["max_brightness"], warnings=val2["warnings"]
            )
            feat2 = extract_experimental_features(np.array(Image.open(raw_path2)), bbox=[170, 235, 340, 265])
            AIAnalysisRepository.save_features(
                db=db, image_id=img2.id, gap_px=feat2["fracture_gap_px"],
                callus_ratio=feat2["callus_density_ratio"], continuity_idx=feat2["cortical_continuity_index"],
                details=feat2
            )
            AIAnalysisRepository.save_analysis(
                db=db, image_id=img2.id, model_version_id=None,
                detections=[{"bbox": [170, 235, 340, 265], "confidence": 0.81, "class_name": "fracture", "class_id": 1}],
                confidence_summary=0.81, raw_scores={"box_confidences": [0.81], "count": 1},
                status=STATUS_CLINICIAN_REVIEWED
            )
            if dr:
                AIAnalysisRepository.save_clinician_review(
                    db=db, analysis_id=img2.ai_analysis.id, reviewer_id=dr.id,
                    review_status="APPROVED",
                    clinical_notes="Early periosteal callus formation noted at dorsal cortex. Maintaining reduction."
                )

            # Visit 3: Day 42 (Week 6)
            v3 = VisitRepository.create_visit(
                db=db,
                case_id=c1.id,
                visit_date="2026-09-12",
                clinical_notes="Six-week follow-up. Cast removed. Minimal tenderness. Starting gentle wrist active ROM.",
                weight_bearing_status="Partial weight-bearing",
                pain_score=1,
                scheduled_next_date="2026-10-24"
            )
            img3_bytes = create_synthetic_radiograph("alex_d42.jpg", gap_width=4, density_factor=1.25)
            val3 = validate_xray_upload(img3_bytes, "alex_distal_radius_day42.jpg")
            raw_path3, rel_path3 = save_raw_image(img3_bytes, "alex_distal_radius_day42.jpg")
            proc_bytes3 = generate_processed_review_copy(raw_path3)
            proc_path3, proc_rel3 = save_processed_image(proc_bytes3, rel_path3)
            img3 = XrayImageRepository.create_image(
                db=db, visit_id=v3.id, file_path=rel_path3, original_filename="alex_distal_radius_day42.jpg",
                file_size=len(img3_bytes), image_hash=val3["sha256"], dhash=val3["dhash"],
                processed_path=proc_rel3, status=STATUS_RESULT_READY
            )
            XrayImageRepository.save_validation_result(
                db=db, image_id=img3.id, is_valid=True, format=val3["format"],
                width=val3["width"], height=val3["height"], blur_score=val3["blur_score"],
                contrast_score=val3["contrast_score"], min_brightness=val3["min_brightness"],
                max_brightness=val3["max_brightness"], warnings=val3["warnings"]
            )
            feat3 = extract_experimental_features(np.array(Image.open(raw_path3)), bbox=[170, 240, 340, 260])
            AIAnalysisRepository.save_features(
                db=db, image_id=img3.id, gap_px=feat3["fracture_gap_px"],
                callus_ratio=feat3["callus_density_ratio"], continuity_idx=feat3["cortical_continuity_index"],
                details=feat3
            )
            AIAnalysisRepository.save_analysis(
                db=db, image_id=img3.id, model_version_id=None,
                detections=[{"bbox": [170, 240, 340, 260], "confidence": 0.62, "class_name": "fracture", "class_id": 1}],
                confidence_summary=0.62, raw_scores={"box_confidences": [0.62], "count": 1},
                status=STATUS_CLINICIAN_REVIEWED
            )
            if dr:
                AIAnalysisRepository.save_clinician_review(
                    db=db, analysis_id=img3.ai_analysis.id, reviewer_id=dr.id,
                    review_status="APPROVED",
                    clinical_notes="Bridging trabecular callus bridging the cortices. Advanced osseous healing."
                )
                # Generate sample report for visit 3
                from database.repositories import ReportRepository
                rep_path = generate_pdf_report(
                    report_code="REP-DEMO-2026-0001",
                    patient_info={"patient_code": p1.patient_code, "full_name": f"{p1.first_name} {p1.last_name}", "dob": p1.dob, "gender": p1.gender},
                    case_info={"case_code": c1.case_code, "anatomical_site": c1.anatomical_site, "injury_date": c1.injury_date},
                    visit_info={"visit_code": v3.visit_code, "visit_date": v3.visit_date, "days_post_injury": v3.days_post_injury},
                    ai_analysis_info={"detections": [{"class_name": "fracture", "confidence": 0.62, "bbox": [170, 240, 340, 260]}]},
                    clinician_review_info={"review_status": "APPROVED", "reviewer_name": dr.full_name, "signed_at": "2026-09-12 11:30", "clinical_notes": "Bridging callus across fracture site. Clinical union evident."},
                    reliability_summary={"technical_quality": {"status": "ACCEPTABLE", "blur_laplacian": val3["blur_score"], "contrast_std": val3["contrast_score"], "warnings": []}},
                    output_filename="REP_DEMO_0001.pdf"
                )
                ReportRepository.save_report(
                    db=db, visit_id=v3.id, pdf_path=str(rep_path), generated_by_id=dr.id, is_final=True
                )

            print("Demo Case 1 (Alex Mercer, 3 visits) created successfully!")

        # Create Patient 2: Sarah Connor (Tibia fracture)
        p2 = db.query(Patient).filter(Patient.first_name == "Sarah", Patient.last_name == "Connor").first()
        if not p2:
            print("Creating Patient 2: Sarah Connor...")
            p2 = PatientRepository.create_patient(
                db=db,
                first_name="Sarah",
                last_name="Connor",
                dob="1975-09-23",
                gender="Female",
                contact_phone="+1-555-014-9988",
                medical_history="High athletic activity, no osteopenia."
            )
            c2 = FractureCaseRepository.create_case(
                db=db,
                patient_id=p2.id,
                anatomical_site="Tibial Shaft",
                injury_date="2026-09-01",
                injury_mechanism="Bicycle collision",
                initial_treatment="Intramedullary Nailing (IMN)",
                notes="Midshaft tibial fracture stabilized with locked IM nail."
            )
            v2_1 = VisitRepository.create_visit(
                db=db,
                case_id=c2.id,
                visit_date="2026-09-02",
                clinical_notes="Immediate post-operative AP/Lateral radiograph. Hardware stable.",
                weight_bearing_status="Touch-down weight-bearing",
                pain_score=7,
                scheduled_next_date="2026-10-15"
            )
            img2_1_bytes = create_synthetic_radiograph("sarah_d1.jpg", gap_width=10, density_factor=1.0)
            val2_1 = validate_xray_upload(img2_1_bytes, "sarah_tibia_postop.jpg")
            raw2_path, raw2_rel = save_raw_image(img2_1_bytes, "sarah_tibia_postop.jpg")
            proc2_bytes = generate_processed_review_copy(raw2_path)
            proc2_path, proc2_rel = save_processed_image(proc2_bytes, raw2_rel)
            img2_1 = XrayImageRepository.create_image(
                db=db, visit_id=v2_1.id, file_path=raw2_rel, original_filename="sarah_tibia_postop.jpg",
                file_size=len(img2_1_bytes), image_hash=val2_1["sha256"], dhash=val2_1["dhash"],
                processed_path=proc2_rel, status=STATUS_RESULT_READY
            )
            XrayImageRepository.save_validation_result(
                db=db, image_id=img2_1.id, is_valid=True, format=val2_1["format"],
                width=val2_1["width"], height=val2_1["height"], blur_score=val2_1["blur_score"],
                contrast_score=val2_1["contrast_score"], min_brightness=val2_1["min_brightness"],
                max_brightness=val2_1["max_brightness"], warnings=val2_1["warnings"]
            )
            AIAnalysisRepository.save_analysis(
                db=db, image_id=img2_1.id, model_version_id=None,
                detections=[{"bbox": [180, 245, 330, 260], "confidence": 0.76, "class_name": "fracture", "class_id": 1}],
                confidence_summary=0.76, raw_scores={"box_confidences": [0.76], "count": 1},
                status=STATUS_RESULT_READY
            )
            print("Demo Case 2 (Sarah Connor) created successfully!")

        print("\nAll clinical demonstration data seeded successfully!")


if __name__ == "__main__":
    seed_demo_cases()
