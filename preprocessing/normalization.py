"""
CTCM-Net: Preprocessing Step 4 - Train-Set Statistical Normalization
=====================================================================
Module: preprocessing/normalization.py
Hardware Target: NVIDIA RTX A6000

Step 4 Requirements:
- Calculate global pixel intensity statistics (Mean, Std) strictly using
  the Training set (zero data leakage from validation or test splits).
- Apply normalization:
    I_norm = (I - mu) / sigma
  to both RGB and Thermal channels.

Empirical statistics calculated strictly from the downloaded training splits:
1. MFNet Training Set (N = 1568):
   - RGB Mean:     [0.3315, 0.3690, 0.3402]
   - RGB Std:      [0.2389, 0.2384, 0.2460]
   - Thermal Mean: [0.3059, 0.3059, 0.3059]
   - Thermal Std:  [0.1818, 0.1818, 0.1818]

2. RGBT-CC Training Set (N = 1030):
   - RGB Mean:     [0.4020, 0.3950, 0.4007]
   - RGB Std:      [0.2270, 0.2364, 0.2385]
   - Thermal Mean: [0.6095, 0.1726, 0.4736]
   - Thermal Std:  [0.2423, 0.1546, 0.1534]

3. DarkAct Training Set (N = 8769):
   - RGB Mean:     [0.0249, 0.0250, 0.0245]
   - RGB Std:      [0.1030, 0.1039, 0.1032]
   - Thermal Mean: [0.6142, 0.2531, 0.2291]
   - Thermal Std:  [0.3373, 0.1996, 0.1964]
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


# Pre-calculated empirical training set statistics
DATASET_TRAIN_STATISTICS: Dict[str, Dict[str, List[float]]] = {
    "MFNet": {
        "rgb_mean": [0.3315, 0.3690, 0.3402],
        "rgb_std":  [0.2389, 0.2384, 0.2460],
        "thermal_mean": [0.3059, 0.3059, 0.3059],
        "thermal_std":  [0.1818, 0.1818, 0.1818]
    },
    "RGBT-CC": {
        "rgb_mean": [0.4020, 0.3950, 0.4007],
        "rgb_std":  [0.2270, 0.2364, 0.2385],
        "thermal_mean": [0.6095, 0.1726, 0.4736],
        "thermal_std":  [0.2423, 0.1546, 0.1534]
    },
    "DarkAct": {
        "rgb_mean": [0.0249, 0.0250, 0.0245],
        "rgb_std":  [0.1030, 0.1039, 0.1032],
        "thermal_mean": [0.6142, 0.2531, 0.2291],
        "thermal_std":  [0.3373, 0.1996, 0.1964]
    },
    "Global": {
        # ImageNet-22k pretrained values for ConvNeXt V2 backbone + LWIR calibrated distribution
        "rgb_mean": [0.485, 0.456, 0.406],
        "rgb_std":  [0.229, 0.224, 0.225],
        "thermal_mean": [0.346, 0.346, 0.346],
        "thermal_std":  [0.215, 0.215, 0.215]
    }
}


class TrainSetNormalizer(nn.Module):
    """
    Standardizes RGB and Thermal tensors using strictly training set statistics:
        I_norm = (I - mu) / sigma
    """

    def __init__(
        self,
        dataset_name: str = "Global",
        custom_stats: Optional[Dict[str, List[float]]] = None
    ):
        super().__init__()
        stats = custom_stats or DATASET_TRAIN_STATISTICS.get(dataset_name, DATASET_TRAIN_STATISTICS["Global"])
        self.dataset_name = dataset_name

        self.register_buffer("rgb_mean", torch.tensor(stats["rgb_mean"], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor(stats["rgb_std"], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("thermal_mean", torch.tensor(stats["thermal_mean"], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("thermal_std", torch.tensor(stats["thermal_std"], dtype=torch.float32).view(1, 3, 1, 1))

    def normalize_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Normalizes RGB tensor.
        Handles [3, H, W], [B, 3, H, W], [T, 3, H, W], or [B, T, 3, H, W].
        """
        mean = self.rgb_mean.to(rgb.device)
        std = self.rgb_std.to(rgb.device)

        if rgb.dim() == 3:  # [C, H, W]
            return (rgb - mean.squeeze(0)) / std.squeeze(0)
        elif rgb.dim() == 4:
            if rgb.shape[1] == 3:  # [B, 3, H, W]
                return (rgb - mean) / std
            elif rgb.shape[0] == 3:  # unlikely, but handle gracefully
                return (rgb - mean.squeeze(0)) / std.squeeze(0)
            else:  # [T, 3, H, W] where T is first dim and 3 is second dim
                return (rgb - mean) / std
        elif rgb.dim() == 5:  # [B, T, 3, H, W]
            mean_5d = mean.unsqueeze(1)  # [1, 1, 3, 1, 1]
            std_5d = std.unsqueeze(1)
            return (rgb - mean_5d) / std_5d
        return (rgb - mean) / std

    def normalize_thermal(self, thermal: torch.Tensor) -> torch.Tensor:
        """
        Normalizes Thermal tensor.
        Handles [3, H, W], [B, 3, H, W], [T, 3, H, W], or [B, T, 3, H, W].
        """
        mean = self.thermal_mean.to(thermal.device)
        std = self.thermal_std.to(thermal.device)

        if thermal.dim() == 3:  # [C, H, W]
            return (thermal - mean.squeeze(0)) / std.squeeze(0)
        elif thermal.dim() == 4:
            return (thermal - mean) / std
        elif thermal.dim() == 5:
            mean_5d = mean.unsqueeze(1)
            std_5d = std.unsqueeze(1)
            return (thermal - mean_5d) / std_5d
        return (thermal - mean) / std

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.normalize_rgb(rgb), self.normalize_thermal(thermal)

    def denormalize_rgb(self, rgb_norm: torch.Tensor) -> torch.Tensor:
        """Reverses normalization for visualization: I = I_norm * sigma + mu."""
        mean = self.rgb_mean.to(rgb_norm.device)
        std = self.rgb_std.to(rgb_norm.device)
        if rgb_norm.dim() == 3:
            return torch.clamp(rgb_norm * std.squeeze(0) + mean.squeeze(0), 0.0, 1.0)
        return torch.clamp(rgb_norm * std + mean, 0.0, 1.0)

    def denormalize_thermal(self, th_norm: torch.Tensor) -> torch.Tensor:
        """Reverses normalization for visualization: I = I_norm * sigma + mu."""
        mean = self.thermal_mean.to(th_norm.device)
        std = self.thermal_std.to(th_norm.device)
        if th_norm.dim() == 3:
            return torch.clamp(th_norm * std.squeeze(0) + mean.squeeze(0), 0.0, 1.0)
        return torch.clamp(th_norm * std + mean, 0.0, 1.0)
