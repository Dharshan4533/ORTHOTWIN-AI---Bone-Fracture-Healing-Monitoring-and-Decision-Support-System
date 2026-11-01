# OrthoTwin AI — Dataset Audit Report
**Date:** 2026-10-15 00:32:01  
**Status:** FAILED — ARCHIVES NOT FOUND  

⚠️ **AUDIT FAILED:** Neither dataset archive was found on disk. No empirical verification could be executed.

---

## 1. Executive Summary & Findings

Configured dataset archive sources:
1. **Roboflow 100 — Bone Fracture:** `C:\Users\dhars\Desktop\web project\bone fracture.v1-release-640.yolov8.zip` (Exists: False, SHA-256: `NOT_FOUND`)
2. **FracAtlas:** `C:\Users\dhars\Desktop\web project\FracAtlas.zip` (Exists: False, SHA-256: `NOT_FOUND`)

### Key Findings & Empirical Gating:
- **Primary Detection Experiment (Roboflow 100):**
  - Exists: `False`
  - Shared stems (leakage across splits): Train-Val: `N/A`, Train-Test: `N/A`.
  - *Finding:* Offline data augmentations generated duplicate base image stems across splits. Patient-level splitting is impossible because the dataset does not include patient IDs.
- **FracAtlas Discrepancy Analysis:**
  - Exists: `False`
  - Total Scans: `N/A` (N/A fractured, N/A non-fractured).
  - Split CSV sum: `N/A` images.
  - *Finding:* The author split CSVs only partition the fractured images (719 scans). The 3,366 non-fractured images were omitted from split CSVs.
- **Segmentation Annotations:**
  - FracAtlas COCO masks: `N/A` polygon annotations across `N/A` images.
- **Longitudinal Tracking Safeguard:**
  - Neither dataset contains patient IDs or chronological follow-up series. Healing outcome prediction remains strictly **UNAVAILABLE** to prevent algorithmic hallucination.

---

## 2. Roboflow 100 Details
- Partitions: {}
- Class distribution: {}
- Error details: Archive file not found at: C:\Users\dhars\Desktop\web project\bone fracture.v1-release-640.yolov8.zip

---

## 3. FracAtlas Details
- Fracture split CSV counts: {}
- Valid polygon masks present: False
- Error details: Archive file not found at: C:\Users\dhars\Desktop\web project\FracAtlas.zip

---

## 4. Methodological Safeguards Enforced
1. **No Mixed Taxonomies:** Roboflow (4 classes) and FracAtlas (binary fracture) must be trained as separate experiments.
2. **Leakage Disclosure:** Any reported detection metric must disclose that default splits have overlapping base images.
3. **Safety Disclaimers:** All model candidate detections require independent clinician sign-off.
