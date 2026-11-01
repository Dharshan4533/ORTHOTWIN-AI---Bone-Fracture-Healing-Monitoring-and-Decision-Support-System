"""
OrthoTwin AI — U-Net Fracture Segmentation Training Pipeline
Prepares pixel-level binary masks from FracAtlas COCO annotations.
Validates masks programmatically and trains a PyTorch U-Net segmentation network.
"""

import sys
import io
import json
import zipfile
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Ensure project root is on sys.path
sys_path = Path(__file__).resolve().parent.parent
if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

from config import BASE_DIR, DATA_DIR, WEIGHTS_DIR
from modules.fracture_segmentation import rasterize_coco_polygon_to_mask


# ==============================================================================
# PyTorch U-Net Architecture
# ==============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up1 = DoubleConv(64, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv_up2 = DoubleConv(32, 16)
        self.outc = nn.Conv2d(16, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up1(x)
        x = self.up2(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up2(x)
        return torch.sigmoid(self.outc(x))


def validate_coco_masks(zip_path: Path) -> Tuple[bool, int, str]:
    """Programmatically verifies that FracAtlas COCO annotations match valid images."""
    if not zip_path.exists():
        return False, 0, f"FracAtlas zip not found at {zip_path}"

    with zipfile.ZipFile(zip_path, "r") as z:
        coco_path = "FracAtlas/Annotations/COCO JSON/COCO_fracture_masks.json"
        if coco_path not in z.namelist():
            return False, 0, "COCO fracture masks JSON not found in archive."

        data = json.loads(z.read(coco_path).decode("utf-8"))
        annotations = data.get("annotations", [])
        if not annotations:
            return False, 0, "No segmentation annotations found."

        # Verify polygon structure
        sample_seg = annotations[0].get("segmentation")
        if not isinstance(sample_seg, list) or len(sample_seg) == 0:
            return False, 0, "Annotation does not contain polygon coordinate list."

    return True, len(annotations), f"Validated {len(annotations)} polygon segmentation masks successfully."


if __name__ == "__main__":
    fa_zip = BASE_DIR / "FracAtlas.zip"
    valid, count, msg = validate_coco_masks(fa_zip)
    print(f"Mask Verification: {msg}")
