"""
Cross-Task Consistency Engine (L_mask-density).
================================================
Core Architectural Novelty: Bidirectional spatial regularization binding
Semantic Pedestrian Segmentation Masks with Continuous Crowd Density Maps.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import PEDESTRIAN_CLASS_INDEX


class CrossTaskConsistencyEngine(nn.Module):
    """
    Core Architectural Novelty: Cross-Task Consistency Engine.

    Enforces bidirectional spatial regularization between Semantic Segmentation
    (Pedestrian Mask, Class ID 2) and Continuous Crowd Density Maps:
    1. Out-of-Bounds Suppression Penalty:
       Penalizes density mass predicted in background or non-pedestrian pixels:
           L_out = sum( D * (1 - P_person) ) / (sum(D) + eps)
    2. Existence / Spatial Alignment Encouragement:
       Encourages positive density response wherever a pedestrian mask is detected
       via a soft spatial Dice/Jaccard constraint:
           L_align = 1 - 2 * sum( P_person * D_norm ) / (sum(P_person^2) + sum(D_norm^2) + eps)
    3. Composite Bidirectional Regularization:
           L_mask_density = L_out + beta * L_align
    """

    def __init__(
        self,
        pedestrian_class_idx: int = PEDESTRIAN_CLASS_INDEX,
        beta: float = 1.0
    ):
        super().__init__()
        self.pedestrian_class_idx = pedestrian_class_idx
        self.beta = beta

    def forward(
        self,
        seg_logits: torch.Tensor,
        density_map: torch.Tensor,
        eps: float = 1e-6
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            seg_logits: Segmentation logits [B, 9, 256, 256]
            density_map: Continuous density map [B, 1, 256, 256]

        Returns:
            Dictionary containing:
                loss_mask_density: Total bidirectional consistency loss
                loss_out_of_bounds: Out-of-bounds density suppression penalty
                loss_align: Pedestrian existence spatial alignment penalty
                p_person: Pedestrian probability map [B, 1, 256, 256]
                d_norm: Dynamically normalized density map [B, 1, 256, 256]
        """
        # 1. Extract pedestrian probability map from segmentation logits: [B, 1, H, W]
        prob_all = F.softmax(seg_logits, dim=1)
        p_person = prob_all[:, self.pedestrian_class_idx:self.pedestrian_class_idx+1, :, :]

        # 2. Normalized density map: [B, 1, H, W] in [0, 1]
        max_d = density_map.amax(dim=(-2, -1), keepdim=True)
        d_norm = density_map / (max_d + eps)

        # 3. Out-of-bounds Penalty: Density in non-pedestrian regions (1 - P_person)
        out_of_bounds_mass = (density_map * (1.0 - p_person)).sum(dim=(-2, -1))
        total_density_mass = density_map.sum(dim=(-2, -1)) + eps
        l_out = (out_of_bounds_mass / total_density_mass).mean()

        # 4. Spatial Alignment / Existence Penalty (Soft Dice formulation)
        intersection = (p_person * d_norm).sum(dim=(-2, -1))
        cardinality = (p_person ** 2).sum(dim=(-2, -1)) + (d_norm ** 2).sum(dim=(-2, -1)) + eps
        dice_coeff = (2.0 * intersection) / cardinality
        l_align = (1.0 - dice_coeff).mean()

        # 5. Composite Bidirectional Loss
        l_mask_density = l_out + (self.beta * l_align)

        return {
            "loss_mask_density": l_mask_density,
            "loss_out_of_bounds": l_out,
            "loss_align": l_align,
            "p_person": p_person,
            "d_norm": d_norm
        }
