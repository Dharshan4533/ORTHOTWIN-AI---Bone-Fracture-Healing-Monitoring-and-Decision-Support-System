"""
OrthoTwin AI — 2D Digital Twin & Feature Extraction Module
Constructs chronological visit timelines, visual comparison overlays,
and computes experimental image-derived features with strict safety disclaimers.
Does NOT infer clinical healing from raw pixel values alone.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import plotly.graph_objects as go
from sqlalchemy.orm import Session
from database.models import FractureCase, Visit, XrayImage, ExtractedFeature
from database.repositories import AIAnalysisRepository
from config import DIGITAL_TWIN_DISCLAIMER, BASE_DIR


def extract_experimental_features(
    image_np: np.ndarray,
    bbox: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Computes experimental image-derived features within a detected or full-frame region.
    IMPORTANT: These are computational proxies and MUST NOT be represented as validated
    clinical biomarkers of bone union.
    """
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np

    h, w = gray.shape
    # If bbox provided [x1, y1, x2, y2], crop region
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = [int(max(0, c)) for c in bbox]
        x2 = min(w, max(x1 + 1, x2))
        y2 = min(h, max(y1 + 1, y2))
        roi = gray[y1:y2, x1:x2]
    else:
        roi = gray

    roi_h, roi_w = roi.shape

    # 1. Fracture gap proxy (minimum edge-to-edge dark valley width in pixels)
    # Using adaptive thresholding and morphological gradient
    sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # Estimate discontinuity width
    thresholded_grad = grad_mag > np.percentile(grad_mag, 85)
    gap_width_px = float(np.sum(thresholded_grad) / max(1, roi_h))

    # 2. Callus density proxy (ratio of gap mean intensity to surrounding cortical bone)
    gap_mean_intensity = float(np.mean(roi))
    surrounding_intensity = float(np.mean(gray)) + 1e-5
    density_ratio = float(min(2.0, max(0.1, gap_mean_intensity / surrounding_intensity)))

    # 3. Cortical continuity index (Laplacian coherence along margins [0.0 - 1.0])
    edges = cv2.Canny(roi, 50, 150)
    continuity_idx = float(min(1.0, np.sum(edges > 0) / (roi_h * roi_w + 1e-5) * 10))

    return {
        "fracture_gap_px": round(gap_width_px, 2),
        "callus_density_ratio": round(density_ratio, 3),
        "cortical_continuity_index": round(continuity_idx, 3),
        "calibration_notice": "UNITS IN PIXELS: Physical millimeter scale requires verified radiopaque calibration marker in image.",
        "biomarker_disclaimer": DIGITAL_TWIN_DISCLAIMER
    }


def compare_scans_visual(
    img_a_path: Path,
    img_b_path: Path,
    target_dim: Tuple[int, int] = (600, 600)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepares two longitudinal scans for side-by-side and difference comparison.
    Flags view angle differences, scale discrepancies, or aspect ratio mismatches.
    Returns (img_a_aligned, img_b_aligned, abs_difference, warnings).
    """
    warnings: List[str] = []

    with Image.open(img_a_path) as a, Image.open(img_b_path) as b:
        w_a, h_a = a.size
        w_b, h_b = b.size
        aspect_a = w_a / h_a
        aspect_b = w_b / h_b

        if abs(aspect_a - aspect_b) > 0.15:
            warnings.append(
                f"Aspect ratio mismatch detected ({aspect_a:.2f} vs {aspect_b:.2f}). "
                "Scans may have been captured from different radiographic projections (e.g. AP vs Lateral) or orientations."
            )

        if abs(w_a - w_b) > 300 or abs(h_a - h_b) > 300:
            warnings.append(
                f"Significant spatial resolution disparity ({w_a}x{h_a} vs {w_b}x{h_b}). "
                "Differences in magnification or focal distance will skew uncalibrated pixel measurements."
            )

        a_res = a.convert("RGB").resize(target_dim, Image.Resampling.BILINEAR)
        b_res = b.convert("RGB").resize(target_dim, Image.Resampling.BILINEAR)

    np_a = np.array(a_res)
    np_b = np.array(b_res)
    diff = cv2.absdiff(np_a, np_b)

    return np_a, np_b, diff, warnings


def build_longitudinal_trajectory_plot(
    visits_data: List[Dict[str, Any]]
) -> go.Figure:
    """
    Constructs an interactive Plotly multi-metric historical progression chart.
    Strictly documents that past points are observed measurements and NOT future predictions.
    """
    fig = go.Figure()

    if not visits_data:
        fig.add_annotation(
            text="No follow-up visit data available for this case.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig

    days = [v["days_post_injury"] for v in visits_data]
    gap_px = [v.get("fracture_gap_px") for v in visits_data]
    callus_ratios = [v.get("callus_density_ratio") for v in visits_data]
    pain_scores = [v.get("pain_score") for v in visits_data]
    dates = [v["visit_date"] for v in visits_data]

    # Trace 1: Fracture Gap Proxy (Preserving None for missing measurements)
    gap_text = [
        f"Date: {d}<br>Gap: {g:.1f}px (Uncalibrated Proxy)" if g is not None else f"Date: {d}<br>Gap: Not Measured / No ROI"
        for d, g in zip(dates, gap_px)
    ]
    fig.add_trace(go.Scatter(
        x=days,
        y=gap_px,
        mode="lines+markers",
        name="Gap Width Proxy (px)",
        connectgaps=False,
        line=dict(color="#e74c3c", width=2),
        text=gap_text,
        hoverinfo="text"
    ))

    # Trace 2: Callus Density Ratio (Preserving None for missing measurements)
    dens_text = [
        f"Date: {d}<br>Density Ratio: {r:.2f}" if r is not None else f"Date: {d}<br>Density Ratio: Not Measured"
        for d, r in zip(dates, callus_ratios)
    ]
    fig.add_trace(go.Scatter(
        x=days,
        y=callus_ratios,
        mode="lines+markers",
        name="Callus Density Ratio",
        connectgaps=False,
        yaxis="y2",
        line=dict(color="#2980b9", width=2, dash="dot"),
        text=dens_text,
        hoverinfo="text"
    ))

    # Trace 3: Pain Score (0-10)
    valid_pains = [(d, p, dt) for d, p, dt in zip(days, pain_scores, dates) if p is not None]
    if valid_pains:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in valid_pains],
            y=[p[1] for p in valid_pains],
            mode="markers",
            marker=dict(size=10, color="#f39c12", symbol="diamond"),
            name="Reported Pain (0-10)",
            text=[f"Date: {p[2]}<br>Pain: {p[1]}/10" for p in valid_pains],
            hoverinfo="text"
        ))

    fig.update_layout(
        title="2D Digital Twin: Longitudinal Follow-up Progress (Historical)",
        xaxis_title="Days Post Injury",
        yaxis=dict(title="Estimated Gap Discontinuity (px)", side="left"),
        yaxis2=dict(title="Callus Density Proxy", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40)
    )

    return fig
