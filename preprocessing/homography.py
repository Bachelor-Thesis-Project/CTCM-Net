"""
CTCM-Net: Preprocessing Step 1 - Spatial Planar Homography Alignment
===================================================================
Module: preprocessing/homography.py
Hardware Target: NVIDIA RTX A6000

In dual-sensor optical setups (such as FLIR Tau2 LWIR microbolometer paired
with a Sony RGB visible camera in MFNet), an optical baseline separation b
causes spatial parallax between modalities:
    x_RGB ~ H * x_T

This module applies calibrated 3x3 planar homography transformation via
differentiable bilinear grid sampling, mapping the thermal coordinate frame
into the optical RGB reference frame.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPlanarHomographyAlignment(nn.Module):
    """
    Applies calibrated planar homography registration between RGB and Thermal modalities.

    Given optical baseline separation b and focal length disparity, a planar
    homography H in SE(3) maps coordinates from the thermal sensor plane (x_T)
    to the RGB reference plane (x_RGB):
        x_RGB ~ H * x_T
    """

    def __init__(self, tolerance_px: float = 1.5):
        super().__init__()
        self.tolerance_px = tolerance_px
        # Calibrated baseline homography matrix between optical and LWIR sensors
        self.register_buffer(
            "default_homography",
            torch.tensor([
                [1.002, -0.001,  0.45],
                [0.001,  0.999, -0.30],
                [0.000,  0.000,  1.00]
            ], dtype=torch.float32)
        )

    @staticmethod
    def get_calibrated_homography() -> torch.Tensor:
        """Returns the default calibrated homography matrix [3, 3]."""
        return torch.tensor([
            [1.002, -0.001,  0.45],
            [0.001,  0.999, -0.30],
            [0.000,  0.000,  1.00]
        ], dtype=torch.float32)

    def warp_thermal_to_rgb(
        self,
        thermal: torch.Tensor,
        homography: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Warps the thermal tensor into the RGB reference frame using homography H.

        Args:
            thermal: Thermal tensor [C, H, W] or [B, C, H, W]
            homography: Optional 3x3 homography matrix. If None, uses calibrated H.

        Returns:
            aligned_thermal: Warped thermal tensor matching RGB coordinate frame
            H: Applied homography matrix [3, 3]
        """
        if homography is None:
            homography = self.default_homography.to(thermal.device)
        else:
            homography = homography.to(thermal.device)

        is_batched = (thermal.dim() == 4)
        if not is_batched:
            thermal_batch = thermal.unsqueeze(0)
        else:
            thermal_batch = thermal

        B, C, H, W = thermal_batch.shape

        # Construct normalized destination grid in [-1, 1]
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0, H - 1, H, device=thermal.device, dtype=torch.float32),
            torch.linspace(0, W - 1, W, device=thermal.device, dtype=torch.float32),
            indexing='ij'
        )
        ones = torch.ones_like(x_grid)
        # Shape: [3, H * W]
        dst_coords = torch.stack([x_grid, y_grid, ones], dim=0).view(3, -1)

        # Inverse mapping: p_src ~ H^-1 @ p_dst
        H_inv = torch.inverse(homography)
        src_coords = H_inv @ dst_coords  # [3, H * W]

        # Perspective division
        z_src = src_coords[2, :] + 1e-8
        src_x = src_coords[0, :] / z_src
        src_y = src_coords[1, :] / z_src

        # Normalize to [-1, 1] for grid_sample
        norm_x = (src_x / max(1.0, float(W - 1))) * 2.0 - 1.0
        norm_y = (src_y / max(1.0, float(H - 1))) * 2.0 - 1.0

        grid = torch.stack([norm_x, norm_y], dim=-1).view(1, H, W, 2).repeat(B, 1, 1, 1)

        aligned_thermal = F.grid_sample(
            thermal_batch,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        if not is_batched:
            aligned_thermal = aligned_thermal.squeeze(0)

        return aligned_thermal, homography

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        homography: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes Step 1: Spatial Planar Homography Alignment.

        Args:
            rgb: RGB image tensor [C, H, W] or [B, C, H, W] (Reference frame)
            thermal: Thermal image tensor [C, H, W] or [B, C, H, W] (Moving frame)
            homography: Optional 3x3 homography matrix

        Returns:
            aligned_rgb: Spatial reference RGB tensor
            aligned_thermal: Registered thermal tensor warped into RGB frame
            H: Applied homography matrix [3, 3]
        """
        aligned_thermal, H = self.warp_thermal_to_rgb(thermal, homography)
        return rgb, aligned_thermal, H
