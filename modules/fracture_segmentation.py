"""
OrthoTwin AI — Fracture Segmentation Module (Optional U-Net)
Provides mask rasterization from COCO polygon annotations and inference gating.
Strictly kept UNAVAILABLE unless validated checkpoint weights exist.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
from config import WEIGHTS_DIR


class SegmentationEngine:
    """
    U-Net Segmentation Engine.
    Only enables segmentation when verified trained checkpoint weights exist and load cleanly.
    Never returns fake-ready status or all-zero dummy masks.
    """
    def __init__(self, weights_path: Optional[Path] = None):
        self.weights_path = Path(weights_path or (WEIGHTS_DIR / "fracture_unet.pth"))
        self.is_loaded = False
        self.model = None
        self.load_error: Optional[str] = None
        self._check_and_load_model()

    def _check_and_load_model(self):
        if not self.weights_path.exists() or self.weights_path.stat().st_size == 0:
            self.is_loaded = False
            self.load_error = f"Checkpoint not found or empty at {self.weights_path}"
            return

        try:
            import torch
            from training.train_segmentation import UNet
            model = UNet(in_channels=1, out_channels=1)
            checkpoint = torch.load(self.weights_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            elif isinstance(checkpoint, torch.nn.Module):
                model = checkpoint
            else:
                self.is_loaded = False
                self.load_error = "Checkpoint does not contain a valid PyTorch model or state_dict."
                return

            model.eval()
            self.model = model
            self.is_loaded = True
            self.load_error = None
        except Exception as e:
            self.is_loaded = False
            self.model = None
            self.load_error = f"Failed to load U-Net weights: {str(e)}"

    def predict_mask(self, image_np: np.ndarray) -> Dict[str, Any]:
        """Runs segmentation inference if genuine weights are loaded; otherwise returns SEGMENTATION_UNAVAILABLE."""
        if not self.is_loaded or self.model is None:
            return {
                "available": False,
                "status": "SEGMENTATION_UNAVAILABLE",
                "message": (
                    "U-Net segmentation model is currently UNAVAILABLE. "
                    "Pixel-level segmentation requires training on verified ground-truth masks "
                    "(e.g., FracAtlas COCO polygon annotations) and configuring a valid 'fracture_unet.pth'."
                ),
                "load_error": self.load_error,
                "mask": None
            }

        try:
            import torch
            # Real preprocessing: convert to 1-channel grayscale, resize to 256x256, normalize
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np

            orig_h, orig_w = gray.shape[:2]
            resized = cv2.resize(gray, (256, 256)).astype(np.float32) / 255.0
            tensor_in = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0)  # (1, 1, 256, 256)

            with torch.no_grad():
                pred = self.model(tensor_in)  # Sigmoid output in [0, 1]
                pred_np = pred.squeeze().cpu().numpy()
                binary_mask_256 = (pred_np > 0.5).astype(np.uint8) * 255
                final_mask = cv2.resize(binary_mask_256, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            return {
                "available": True,
                "status": "READY",
                "mask": final_mask,
                "message": "Segmentation inference executed successfully."
            }
        except Exception as e:
            return {
                "available": False,
                "status": "SEGMENTATION_UNAVAILABLE",
                "message": f"Inference execution failed: {str(e)}",
                "mask": None
            }


def rasterize_coco_polygon_to_mask(
    polygons: List[List[float]],
    height: int,
    width: int
) -> np.ndarray:
    """
    Converts COCO format polygon coordinates [[x1, y1, x2, y2, ...]]
    into a binary segmentation mask (0 or 255).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        # Reshape to (N, 1, 2)
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
    return mask


_global_segmenter: Optional[SegmentationEngine] = None


def get_segmentation_engine() -> SegmentationEngine:
    global _global_segmenter
    if _global_segmenter is None:
        _global_segmenter = SegmentationEngine()
    return _global_segmenter
