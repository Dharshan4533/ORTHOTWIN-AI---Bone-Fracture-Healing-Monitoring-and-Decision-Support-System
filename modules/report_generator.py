"""
OrthoTwin AI — Clinical Report Generator (ReportLab PDF)
Generates structured PDF clinical decision support summaries with
configurable institution headers, explicit AI vs. clinician distinction,
and comprehensive academic prototype disclaimers.
"""

import os
import html
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from config import (
    STORAGE_REPORTS_DIR, INSTITUTION_NAME, ACADEMIC_PROTOTYPE_DISCLAIMER
)


def _esc(val: Any) -> str:
    """Safely escapes user or system text strings for ReportLab XML/HTML Paragraph markup."""
    if val is None:
        return "N/A"
    return html.escape(str(val))


def generate_pdf_report(
    report_code: str,
    patient_info: Dict[str, Any],
    case_info: Dict[str, Any],
    visit_info: Dict[str, Any],
    ai_analysis_info: Optional[Dict[str, Any]] = None,
    clinician_review_info: Optional[Dict[str, Any]] = None,
    reliability_summary: Optional[Dict[str, Any]] = None,
    scan_image_path: Optional[Path] = None,
    output_filename: Optional[str] = None
) -> Path:
    """
    Builds a professional, structured PDF report using ReportLab.
    Saves strictly to STORAGE_REPORTS_DIR and returns the absolute Path.
    Safely escapes all user content and optionally embeds an annotated X-ray snapshot.
    """
    if not output_filename:
        output_filename = f"{report_code}.pdf"

    # Constrain filename to basename to prevent path traversal
    clean_name = os.path.basename(output_filename).replace("..", "")
    pdf_path = (STORAGE_REPORTS_DIR / clean_name).resolve()
    
    # Path containment check
    resolved_reports_dir = STORAGE_REPORTS_DIR.resolve()
    if resolved_reports_dir not in pdf_path.parents and pdf_path != resolved_reports_dir:
        raise ValueError("Security violation: report destination path traversal detected.")

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    header_style = ParagraphStyle(
        "InstHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#1a365d"),
        alignment=1  # Centered
    )
    banner_style = ParagraphStyle(
        "DisclaimerBanner",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#c53030"),
        alignment=1
    )
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=14,
        textColor=colors.HexColor("#2d3748")
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#2d3748")
    )
    bold_style = ParagraphStyle(
        "ReportBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#2d3748")
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph(f"{INSTITUTION_NAME}", header_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("ORTHOTWIN AI — LONGITUDINAL DECISION SUPPORT SUMMARY", ParagraphStyle(
        "SubHeader", parent=header_style, fontSize=10, leading=12, textColor=colors.HexColor("#4a5568")
    )))
    story.append(Spacer(1, 6))

    # Academic Disclaimer Box
    disclaimer_table = Table(
        [[Paragraph(f"<b>NOTICE:</b> {ACADEMIC_PROTOTYPE_DISCLAIMER}", banner_style)]],
        colWidths=[540]
    )
    disclaimer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff5f5")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#feb2b2")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(disclaimer_table)
    story.append(Spacer(1, 10))

    # 2. Patient & Case Metadata Table
    story.append(Paragraph("1. Administrative & Clinical Identifiers", title_style))
    story.append(Spacer(1, 4))

    meta_data = [
        [
            Paragraph("<b>Patient ID:</b>", bold_style), Paragraph(_esc(patient_info.get("patient_code")), body_style),
            Paragraph("<b>Patient Name:</b>", bold_style), Paragraph(_esc(patient_info.get("full_name")), body_style)
        ],
        [
            Paragraph("<b>DOB:</b>", bold_style), Paragraph(_esc(patient_info.get("dob")), body_style),
            Paragraph("<b>Gender:</b>", bold_style), Paragraph(_esc(patient_info.get("gender")), body_style)
        ],
        [
            Paragraph("<b>Case Code:</b>", bold_style), Paragraph(_esc(case_info.get("case_code")), body_style),
            Paragraph("<b>Fracture Site:</b>", bold_style), Paragraph(_esc(case_info.get("anatomical_site")), body_style)
        ],
        [
            Paragraph("<b>Injury Date:</b>", bold_style), Paragraph(_esc(case_info.get("injury_date")), body_style),
            Paragraph("<b>Visit Code:</b>", bold_style), Paragraph(_esc(visit_info.get("visit_code")), body_style)
        ],
        [
            Paragraph("<b>Visit Date:</b>", bold_style), Paragraph(_esc(visit_info.get("visit_date")), body_style),
            Paragraph("<b>Days Post Injury:</b>", bold_style), Paragraph(f"{_esc(visit_info.get('days_post_injury', 0))} days", body_style)
        ]
    ]

    meta_table = Table(meta_data, colWidths=[90, 180, 90, 180])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7fafc")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Optional Embedded Scan Snapshot
    if scan_image_path and Path(scan_image_path).exists():
        try:
            story.append(Paragraph("2. Radiographic Scan Snapshot", title_style))
            story.append(Spacer(1, 4))
            img_element = RLImage(str(scan_image_path), width=220, height=220)
            img_table = Table([[img_element]], colWidths=[540])
            img_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("PADDING", (0, 0), (-1, -1), 2),
            ]))
            story.append(img_table)
            story.append(Spacer(1, 8))
        except Exception:
            pass

    # 3. AI Analysis Candidate Findings (Explicitly Labeled Unverified)
    story.append(Paragraph("3. AI Fracture Analysis Output (Unverified Candidate Findings)", title_style))
    story.append(Spacer(1, 4))

    if ai_analysis_info and ai_analysis_info.get("detections"):
        detections = ai_analysis_info["detections"]
        det_data = [[
            Paragraph("<b>#</b>", bold_style),
            Paragraph("<b>Candidate Class</b>", bold_style),
            Paragraph("<b>Raw Confidence</b>", bold_style),
            Paragraph("<b>Bounding Box (x1, y1, x2, y2)</b>", bold_style)
        ]]
        for idx, d in enumerate(detections, 1):
            bbox_str = ", ".join(f"{c:.0f}" for c in d.get("bbox", []))
            det_data.append([
                Paragraph(str(idx), body_style),
                Paragraph(_esc(d.get("class_name", "fracture")), body_style),
                Paragraph(f"{d.get('confidence', 0.0):.3f}", body_style),
                Paragraph(f"[{_esc(bbox_str)}]", body_style)
            ])
        det_table = Table(det_data, colWidths=[30, 150, 120, 240])
        det_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(det_table)
    else:
        story.append(Paragraph("<i>No AI fracture candidates detected or analysis not yet executed.</i>", body_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<i>* Note: Model scores reflect raw localized detector confidence. They do not constitute a definitive medical diagnosis.</i>",
        ParagraphStyle("Notice", parent=body_style, fontSize=7, textColor=colors.HexColor("#718096"))
    ))
    story.append(Spacer(1, 10))

    # 4. Reliability Summary & Quality Warnings
    if reliability_summary:
        story.append(Paragraph("4. Technical Reliability & Data Integrity Summary", title_style))
        story.append(Spacer(1, 4))
        tech = reliability_summary.get("technical_quality", {})
        warnings = tech.get("warnings", [])
        rel_text = (
            f"<b>Image Sharpness (Laplacian):</b> {_esc(tech.get('blur_laplacian', 'N/A'))} | "
            f"<b>Contrast Deviation:</b> {_esc(tech.get('contrast_std', 'N/A'))} | "
            f"<b>Quality Status:</b> {_esc(tech.get('status', 'N/A'))}"
        )
        story.append(Paragraph(rel_text, body_style))
        if warnings:
            for w in warnings:
                story.append(Paragraph(f"• <b>Warning:</b> {_esc(w)}", ParagraphStyle("Warn", parent=body_style, textColor=colors.HexColor("#c53030"))))
        else:
            story.append(Paragraph("• No technical image quality alerts triggered.", body_style))
        story.append(Spacer(1, 10))

    # 5. Clinician Review & Attestation
    story.append(Paragraph("5. Clinician Independent Verification & Review Notes", title_style))
    story.append(Spacer(1, 4))

    rev_status = clinician_review_info.get("review_status", "PENDING_REVIEW") if clinician_review_info else "PENDING_REVIEW"
    rev_notes = clinician_review_info.get("clinical_notes", "Awaiting physician evaluation.") if clinician_review_info else "Awaiting physician evaluation."
    rev_by = clinician_review_info.get("reviewer_name", "Unassigned") if clinician_review_info else "Unassigned"
    rev_time = clinician_review_info.get("signed_at", "N/A") if clinician_review_info else "N/A"

    rev_box_color = colors.HexColor("#ebf8ff") if rev_status == "APPROVED" else colors.HexColor("#fffaf0")
    rev_data = [
        [Paragraph(f"<b>Review Status:</b> {_esc(rev_status)}", bold_style), Paragraph(f"<b>Reviewer:</b> {_esc(rev_by)}", bold_style)],
        [Paragraph(f"<b>Attestation Date:</b> {_esc(rev_time)}", body_style), Paragraph(f"<b>Report Code:</b> {_esc(report_code)}", body_style)],
        [Paragraph("<b>Clinician Notes:</b>", bold_style), Paragraph(_esc(rev_notes), body_style)]
    ]
    rev_table = Table(rev_data, colWidths=[200, 340])
    rev_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), rev_box_color),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bee3f8")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(rev_table)
    story.append(Spacer(1, 12))

    # Signature line
    sig_data = [
        ["", "________________________________________________"],
        ["", f"Attending Clinician Signature ({_esc(rev_by)})"]
    ]
    sig_table = Table(sig_data, colWidths=[280, 260])
    sig_table.setStyle(TableStyle([
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#4a5568"))
    ]))
    story.append(sig_table)

    # Build document
    doc.build(story)
    return pdf_path
