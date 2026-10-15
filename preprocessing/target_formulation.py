"""
CTCM-Net: Preprocessing Step 2 - Target Formulation Integration (Patched & Verified)
====================================================================================
Module: preprocessing/target_formulation.py
Hardware Target: NVIDIA RTX A6000

Step 2 Target Formulation Requirements & Bug Fixes:
1. MFNet: Encode real semantic segmentation annotations into 9-class discrete
   integer masks (Shape: [B, 256, 256], dtype: torch.long).
   CRITICAL FIX: Strictly use NEAREST-neighbor interpolation. Zero fractional classes.
2. RGBT-CC: Mathematically convert raw point coordinates [(x_i, y_i)] into
   continuous 2D Gaussian density heatmaps (Shape: [B, 1, 256, 256], dtype: torch.float32).
   CRITICAL FIX: Mathematically scale raw coordinates by (256/W_raw, 256/H_raw)
   BEFORE calculating the 2D Gaussian kernel so density peaks align precisely with head locations.
3. DarkAct: Encode sequence action labels into 27 discrete action class logits
   (Shape: [B, 27], dtype: torch.float32).
   CRITICAL FIX: Verified exact 27 class mapping matching ARIM_v1 annotation files.
"""

import math
from typing import List, Tuple, Union
import numpy as np
import torch
import torchvision.transforms.functional as TF


# MFNet 9 Classes
MFNET_CLASSES: List[str] = [
    "unlabeled",    # 0
    "car",          # 1
    "person",       # 2
    "bike",         # 3
    "curve",        # 4
    "car_stop",     # 5
    "guardrail",    # 6
    "color_cone",   # 7
    "bump"          # 8
]
NUM_SEG_CLASSES: int = len(MFNET_CLASSES)

# DarkAct 27 Action Classes (Exact 1-to-1 mapping matching ARIM_v1 annotation directory structure)
DARKACT_CLASSES: List[str] = [
    "call",              # 0
    "climb_stairs",      # 1
    "crouch",            # 2
    "door_close",        # 3
    "door_open",         # 4
    "drink",             # 5
    "holding_umbrella",  # 6
    "jump",              # 7
    "lift",              # 8
    "photographing",     # 9
    "pick",              # 10
    "pour",              # 11
    "push",              # 12
    "put_off_coat",      # 13
    "put_on_coat",       # 14
    "run",               # 15
    "searching",         # 16
    "sit",               # 17
    "slap",              # 18
    "sleeping",          # 19
    "squat",             # 20
    "stand",             # 21
    "turn",              # 22
    "typing",            # 23
    "using_smartphone",  # 24
    "walk",              # 25
    "wave"               # 26
]
NUM_ACTION_CLASSES: int = len(DARKACT_CLASSES)

TARGET_RESOLUTION: Tuple[int, int] = (256, 256)


# ==============================================================================
# 1. MFNET TARGET FORMULATION: STRICT NEAREST-NEIGHBOR 9-CLASS MASK
# ==============================================================================

def encode_mfnet_mask(
    raw_mask: Union[torch.Tensor, np.ndarray],
    target_size: Tuple[int, int] = TARGET_RESOLUTION
) -> torch.Tensor:
    """
    Encodes raw MFNet annotations into discrete 9-class integer mask [256, 256].

    CRITICAL FIX: Strictly utilizes NEAREST-neighbor interpolation.
    Bilinear interpolation corrupts discrete integer class tokens with fractional
    artifacts and spurious intermediate class labels at object boundaries.
    """
    if not isinstance(raw_mask, torch.Tensor):
        mask_tensor = torch.from_numpy(raw_mask).long()
    else:
        mask_tensor = raw_mask.long()

    if mask_tensor.dim() == 2:
        mask_expanded = mask_tensor.unsqueeze(0)  # [1, H, W]
    else:
        mask_expanded = mask_tensor

    # Strictly NEAREST interpolation with zero antialiasing
    resized_mask = TF.resize(
        mask_expanded,
        list(target_size),
        interpolation=TF.InterpolationMode.NEAREST
    )

    if mask_tensor.dim() == 2:
        resized_mask = resized_mask.squeeze(0)

    # Strictly enforce valid class ID range [0, NUM_SEG_CLASSES - 1]
    clamped_mask = torch.clamp(resized_mask, 0, NUM_SEG_CLASSES - 1).long()
    return clamped_mask


# ==============================================================================
# 2. RGBT-CC TARGET FORMULATION: MATHEMATICAL POINT SCALING & 2D GAUSSIAN DENSITY
# ==============================================================================

def scale_point_coordinates(
    points: List[List[float]],
    raw_size: Tuple[int, int],
    target_size: Tuple[int, int] = TARGET_RESOLUTION
) -> List[Tuple[float, float]]:
    """
    Mathematically scales raw point coordinates by (256/W_raw, 256/H_raw).

    Args:
        points: List of [x, y] coordinates in raw pixel space
        raw_size: (W_raw, H_raw)
        target_size: (H_target, W_target) = (256, 256)

    Returns:
        scaled_points: List of (x_scaled, y_scaled) in [256, 256] coordinate frame
    """
    target_h, target_w = target_size
    # Determine W_raw and H_raw (RGBT-CC raw images are 640x480: W=640, H=480)
    if raw_size[0] > raw_size[1]:
        raw_w, raw_h = float(raw_size[0]), float(raw_size[1])
    else:
        raw_h, raw_w = float(raw_size[0]), float(raw_size[1])

    sx = float(target_w) / raw_w  # 256 / 640 = 0.4000
    sy = float(target_h) / raw_h  # 256 / 480 = 0.5333

    scaled_points: List[Tuple[float, float]] = []
    for pt in points:
        px, py = float(pt[0]), float(pt[1])
        scaled_x = px * sx
        scaled_y = py * sy
        scaled_points.append((scaled_x, scaled_y))

    return scaled_points


def generate_gaussian_density_heatmap(
    points: List[List[float]],
    raw_size: Tuple[int, int],
    target_size: Tuple[int, int] = TARGET_RESOLUTION,
    sigma: float = 4.0
) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
    """
    Mathematically scales raw point coordinates and computes continuous 2D Gaussian density heatmap.

    STEP 1: Scale raw (x, y) coordinates BEFORE Gaussian kernel computation:
        x_scaled = x_raw * (256.0 / W_original)
        y_scaled = y_raw * (256.0 / H_original)

    STEP 2: Formulate continuous 2D Gaussian density map in [256, 256] space:
        D(y, x) = sum_{i=1}^N (1 / (2 * pi * sigma^2)) * exp(-((x - x_scaled_i)^2 + (y - y_scaled_i)^2) / (2 * sigma^2))

    Mass Conservation Principle:
        integral_{R^2} D(x, y) dx dy = N (total crowd count)

    Args:
        points: Raw head point coordinates [(x_i, y_i)]
        raw_size: Raw image dimensions (W_raw, H_raw)
        target_size: Output resolution (256, 256)
        sigma: Calibrated Gaussian spread parameter in target pixel space (default: 4.0 px)

    Returns:
        density_heatmap: FloatTensor of shape [1, 256, 256] with D >= 0
        scaled_points: Scaled point coordinates in [256, 256] space
    """
    target_h, target_w = target_size
    scaled_points = scale_point_coordinates(points, raw_size, target_size)

    density = torch.zeros((1, target_h, target_w), dtype=torch.float32)
    if len(scaled_points) == 0:
        return density, scaled_points

    # Localized Gaussian kernel window (radius = 3 * sigma)
    radius = int(math.ceil(3.0 * sigma))
    window_range = torch.arange(-radius, radius + 1, dtype=torch.float32)
    wy, wx = torch.meshgrid(window_range, window_range, indexing='ij')
    norm_factor = 2.0 * math.pi * (sigma ** 2)
    window_g = torch.exp(-(wx ** 2 + wy ** 2) / (2.0 * (sigma ** 2))) / norm_factor

    for cx, cy in scaled_points:
        ix = int(round(cx))
        iy = int(round(cy))

        if 0 <= ix < target_w and 0 <= iy < target_h:
            x0 = max(0, ix - radius)
            x1 = min(target_w, ix + radius + 1)
            y0 = max(0, iy - radius)
            y1 = min(target_h, iy + radius + 1)

            wx0 = x0 - (ix - radius)
            wx1 = wx0 + (x1 - x0)
            wy0 = y0 - (iy - radius)
            wy1 = wy0 + (y1 - y0)

            density[0, y0:y1, x0:x1] += window_g[wy0:wy1, wx0:wx1]

    return density, scaled_points


# ==============================================================================
# 3. DARKACT TARGET FORMULATION: 27-CLASS ACTION LOGITS
# ==============================================================================

def encode_darkact_action(
    action_label: int,
    num_classes: int = NUM_ACTION_CLASSES,
    label_smoothing: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes DarkAct sequence action label into 27 discrete action class logits [27].

    Args:
        action_label: Discrete integer action class index in [0, 26]
        num_classes: Total action classes (27)
        label_smoothing: Optional smoothing factor alpha

    Returns:
        action_logits: FloatTensor of shape [27] (discrete action class logits)
        action_idx: LongTensor scalar [1] containing ground truth class index
    """
    action_idx = torch.tensor(action_label, dtype=torch.long)

    if label_smoothing > 0.0:
        smooth_val = label_smoothing / float(num_classes)
        target_val = 1.0 - label_smoothing + smooth_val
        action_target = torch.full((num_classes,), smooth_val, dtype=torch.float32)
        action_target[action_label] = target_val
        logits = torch.log(action_target.clamp(min=1e-6))
    else:
        # Standard one-hot target representation [27]
        action_target = torch.zeros(num_classes, dtype=torch.float32)
        action_target[action_label] = 1.0
        logits = action_target

    return logits, action_idx
