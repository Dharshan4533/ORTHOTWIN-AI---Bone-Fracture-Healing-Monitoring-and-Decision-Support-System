# OrthoTwin AI — Bone Fracture Healing Monitoring & Decision Support System

> [!CAUTION]
> **ACADEMIC & EDUCATIONAL PROTOTYPE NOTICE**  
> OrthoTwin AI is strictly an **academic research and educational prototype**. It is **not** a cleared or certified medical device under FDA (21 CFR Part 820) or EU CE-MDR regulations. It must **never** be used for autonomous diagnosis, surgical decision-making, patient discharge, or independent clinical triage. All AI candidate detections, 2D digital twin proxies, and reliability summaries are experimental decision-support aids requiring independent evaluation and sign-off by a licensed orthopedic surgeon or radiologist.

---

## 1. System Overview

**OrthoTwin AI** is a modular decision-support system designed to assist orthopedic clinicians in managing bone fracture cases, validating radiographs, performing AI-assisted fracture localization, tracking longitudinal follow-up visits, extracting experimental healing proxies, and compiling structured clinical reports.

### The 8 System Modules

```
OrthoTwin AI Architecture
├── Module 1: Patient Management       (Registration, case tracking, duplicate detection, audit logging)
├── Module 2: X-ray Validation         (Magic bytes, blur/contrast, corruption rejection, CLAHE copies)
├── Module 3: AI Fracture Detection    (YOLOv8 bounding boxes, missing weight fallback, U-Net gating)
├── Module 4: 2D Digital Twin          (Chronological scan alignment, gap & density proxies, difference maps)
├── Module 5: Reliability Engine       (Reliability Summary, uncertainty flags, comparability warnings)
├── Module 6: Healing Progress         (Historical progress comparisons vs. gated future prediction)
├── Module 7: Doctor Dashboard         (Attestation queues, sign-off workflow, ReportLab PDF export)
└── Module 8: Patient Portal           (Role-isolated records, view-only access, simulated reminders)
```

---

## 2. Windows 11 Beginner-Friendly Setup Guide

### Prerequisites
- **Operating System:** Windows 10 or Windows 11 (64-bit).
- **Python:** Python 3.12 or 3.13 installed (ensure *"Add Python to PATH"* was selected during Python installation).

---

### Step 1: Open Terminal in Project Directory
Open **PowerShell** or **Command Prompt** and navigate to your project directory:
```powershell
cd "c:\Users\dhars\Desktop\web project"
```

---

### Step 2: (Optional) Create & Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

---

### Step 3: Install Required Dependencies
Run:
```powershell
pip install -r requirements.txt
```
*Dependencies installed:* `streamlit`, `sqlalchemy`, `pillow`, `opencv-python`, `numpy`, `pandas`, `plotly`, `reportlab`, `pyyaml`, `pytest`, `torch`, `torchvision`, `ultralytics`.

---

### Step 4: Provision Development User Accounts
OrthoTwin AI does not use hardcoded passwords. Run the explicit account provisioning command to seed development test accounts:
```powershell
python create_dev_user.py --seed-defaults
```
This provisions three role-based test accounts:
| Role | Username | Password | Purpose |
|---|---|---|---|
| **Clinician (Doctor)** | `dr_ortho` | `DoctorSecure2026!` | Review scans, sign-off on AI candidates, export PDFs |
| **System Admin** | `admin_user` | `AdminSecure2026!` | Manage users, view system diagnostics & audit logs |
| **Patient** | `patient_demo` | `PatientSecure2026!` | View-only recovery timeline & approved summaries |

*(To create a custom account interactively, run `python create_dev_user.py`.)*

---

### Step 5: (Recommended) Seed Longitudinal Demonstration Cases
To populate the database with realistic de-identified clinical cases (Alex Mercer distal radius fracture with 3 follow-up visits and synthetic radiographs; Sarah Connor tibial fracture):
```powershell
python seed_demo_data.py
```

---

### Step 6: Launch the Streamlit Application
```powershell
streamlit run app.py
```
Streamlit will automatically open your default web browser at `http://localhost:8501`.

---

## 3. Running Automated Tests

OrthoTwin AI includes an automated test suite verifying file validation, security, patient isolation, missing weight fallback, state transitions, and PDF report generation.

Execute all tests from PowerShell:
```powershell
pytest -v
```
*Current test status:* **23 passed in 4.06s (100% passing)**.

---

## 4. Machine Learning & Dataset Auditing

### Empirical Dataset Audit
Run the dataset validation tool to inspect both workspace archives (`bone fracture.v1-release-640.yolov8.zip` and `FracAtlas.zip`):
```powershell
python training/dataset_validation.py
```
This generates `data/DATASET_AUDIT_REPORT.md`.

### Dataset Audit Key Findings
1. **Roboflow 100 — Bone Fracture:**
   - Standardized 640×640 resolution (326 train, 88 val, 44 test images).
   - 4 classes: `angle`, `fracture`, `line`, `messed_up_angle`.
   - **Audit finding:** Data leakage exists across the default splits: augmented variations of the same original source image stems are distributed across train, validation, and test splits (52 shared base stems between train and val; 26 shared base stems between train and test).
2. **FracAtlas Dataset:**
   - 4,085 total images: 719 Fractured, 3,366 Non-fractured.
   - **Discrepancy resolved:** The author-provided `Fracture Split` CSVs only partition the **719 Fractured scans** (574 train + 82 val + 63 test = 719). The 3,366 non-fractured images were excluded from the split files and serve as unpartitioned negative controls.
   - **Segmentation masks:** `COCO_fracture_masks.json` contains 924 valid polygon annotations matching all 719 fractured images.
3. **Absence of Patient Identifiers & Longitudinal Follow-ups:**
   - Neither dataset contains patient IDs, clinical dates, or sequential follow-up scans.
   - Therefore, patient-level splitting cannot be performed, and healing outcome prediction cannot be trained on these datasets without hallucinating clinical progress.

### Training the YOLOv8 Fracture Detector
To fine-tune YOLOv8 offline on the primary Roboflow dataset:
```powershell
python training/train_detection.py
```
*Weights are saved to `models/weights/fracture_yolov8.pt` and registered in the database.*

### Evaluating on Held-Out Test Set
```powershell
python training/evaluate_detection.py
```
*Exports mAP@50 and precision/recall to `models/evaluation_report.json`.*

### Gated Future Prediction Feasibility Check
```powershell
python training/train_prediction.py
```
*Documents exact prerequisite criteria needed before predictive modeling may be enabled.*

---

## 5. Honest Implementation & Capability Status Matrix

As mandated by academic integrity standards, the status of each system component is reported transparently:

| Feature / Capability | Status | Implementation Details |
|---|---|---|
| **Patient Registration & Search** | ✅ **Implemented & Tested** | Full demographic logging, date formatting validation, duplicate warnings, and audit logging. |
| **Fracture Case & Visit Tracking** | ✅ **Implemented & Tested** | Multi-visit chronological history, days post-injury calculations, separation of past visits from future appointments. |
| **X-Ray Technical Validation** | ✅ **Implemented & Tested** | Magic byte inspection, corruption rejection, minimum dimensions (256x256), size limits (15MB), SHA-256 duplicate detection. |
| **Optical Quality Assessment** | ✅ **Implemented & Tested** | Laplacian blur variance and contrast standard deviation analysis with configurable thresholds. |
| **Image Storage & Preservation** | ✅ **Implemented & Tested** | Original raw scans preserved untouched; enhanced CLAHE review copies stored in separate directory; path-traversal proof. |
| **YOLOv8 Detection Inference** | ✅ **Implemented & Tested** | Letterboxed inference, bounding-box parsing, and **defensive fallback** when weights are unconfigured. |
| **2D Digital Twin** | 🔬 **Experimental & Tested** | Chronological alignment, side-by-side comparison, difference maps, uncalibrated gap width and callus density proxies. |
| **Reliability Summary** | ✅ **Implemented & Tested** | Multi-component summary (technical quality, raw detector confidence, missing clinical context, clinician status). |
| **Longitudinal Healing Progress** | ✅ **Implemented & Tested** | Historical visit comparisons and relative observed changes in gap/callus proxies over time. |
| **Predictive Trajectory Modeling** | 🚫 **Blocked by Clinical Data** | Strictly locked in `PREDICTION_UNAVAILABLE` status. Current datasets lack longitudinal patient links and ground-truth union labels. |
| **U-Net Pixel Segmentation** | 🔬 **Gated / Functional** | COCO polygon mask rasterizer verified; architecture implemented; inference gated until verified weights loaded. |
| **Clinician Review Workspace** | ✅ **Implemented & Tested** | Attestation queue, APPROVE/MODIFY/REJECT workflow, mandatory clinical notes, image status transition. |
| **ReportLab PDF Generation** | ✅ **Implemented & Tested** | Institutional header, patient/case metadata table, unverified AI findings, clinician notes, and academic disclaimers. |
| **Role-Based Access Control** | ✅ **Implemented & Tested** | Backend-enforced role separation (Admin, Doctor, Patient) with PBKDF2-HMAC-SHA256 password hashing. |
| **Cross-Patient Data Isolation** | ✅ **Implemented & Tested** | Query-level filtering prevents patients from viewing other patients' records or unreviewed raw AI outputs. |
| **Simulated Patient Reminders** | ✅ **Implemented & Tested** | Scheduled appointment notifications explicitly labeled `[SIMULATED DEMO NOTIFICATION]`. |
| **State Machine Transitions** | ✅ **Implemented & Tested** | Backend transition validation (`UPLOADED` $\to$ `READY_FOR_ANALYSIS` $\to$ `RESULT_READY` $\to$ `CLINICIAN_REVIEWED`). |
| **Result Preservation on Failure** | ✅ **Implemented & Tested** | Retaining previous valid analysis results when a re-run fails. |

---

## 6. Security, Privacy & Regulatory Boundaries

1. **Authentication Security:** Passwords are never stored in plaintext. They are salted with cryptographically secure random salts and hashed using PBKDF2-HMAC-SHA256 with 100,000 iterations.
2. **Access Isolation:** Patient users are strictly prohibited from issuing queries outside their personal patient ID scope.
3. **Path Traversal Protection:** All file paths are sanitized, UUID-stamped, and checked against designated storage directory boundaries.
4. **Audit Logging:** System actions are written to an immutable SQLite audit log; passwords and sensitive keys are automatically redacted.
5. **Regulatory Status:** Any potential clinical translation would require:
   - Institutional Review Board (IRB) ethical protocol approval.
   - Multi-center clinical prospective validation on paired orthogonal radiographs (AP and Lateral).
   - Compliance with FDA Software as a Medical Device (SaMD) and CE-MDR Class IIb/III standards.
   - Calibration with physical radiopaque markers to convert pixel measurements into millimeter metrics.
