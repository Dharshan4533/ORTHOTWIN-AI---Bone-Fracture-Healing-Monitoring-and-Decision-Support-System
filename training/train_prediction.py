"""
OrthoTwin AI — Longitudinal Prediction Pipeline (Safety-Gated)
Enforces prototype safety constraints:
Does NOT fabricate outcome labels or train on cross-sectional data.
Explains the rigorous data requirements needed prior to enabling predictive modeling.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys_path = Path(__file__).resolve().parent.parent
if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

from config import PREDICTION_UNAVAILABLE_DISCLAIMER


def inspect_longitudinal_feasibility() -> dict:
    """Checks whether prerequisites for predictive modeling are satisfied."""
    requirements = [
        {"criterion": "Patient-Identified Longitudinal Scans", "met": False, "note": "Public datasets lack patient IDs and longitudinal links."},
        {"criterion": "Clinically Validated Union Labels (RUST)", "met": False, "note": "Ground truth union times are not available in Roboflow/FracAtlas."},
        {"criterion": "Calibration & Uncertainty Metrics", "met": False, "note": "No held-out longitudinal test cohort exists."}
    ]

    return {
        "prediction_training_permitted": False,
        "reason": "Prerequisite clinical data criteria are unsatisfied.",
        "requirements": requirements,
        "disclaimer": PREDICTION_UNAVAILABLE_DISCLAIMER
    }


if __name__ == "__main__":
    status = inspect_longitudinal_feasibility()
    print("=== ORTHOTWIN PREDICTION ENGINE STATUS ===")
    print(f"Permitted: {status['prediction_training_permitted']}")
    print(f"Reason: {status['reason']}")
    for r in status["requirements"]:
        print(f"  • {r['criterion']}: {'PASS' if r['met'] else 'FAIL'} ({r['note']})")
    print("\n" + status["disclaimer"])
