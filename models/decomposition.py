"""
Illumination-Aware Modality Decomposition & Regularization Engine.
==================================================================
Provides modules for:
1. ConvNeXtBlock1x1: Depthwise-separable projection head.
2. IlluminationAwareSpatialGate: Dynamic spatial-channel gating conditioned on ambient illumination.
3. IlluminationAwareDecomposition: Subspace separation into Modality-Invariant & Modality-Specific features.
4. DecompositionLossEngine: Similarity (L_sim) and Orthogonality (L_diff) regularization constraints.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock1x1(nn.Module):
    """Depthwise-Separable Feature Projection Head with GroupNorm and GELU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.GroupNorm(num_groups=min(8, in_channels), num_channels=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class IlluminationAwareSpatialGate(nn.Module):
    """
    Dynamic Illumination-Aware Spatial Gate.

    Ingests the ambient scene illumination level L in [0, 1] (where L -> 0 denotes
    low-light / nighttime conditions and L -> 1 denotes bright daylight) alongside
    local spatial cues from Modality-Specific features.

    Computes spatial-channel gating weights alpha in [0, 1]^(B x C x H x W):
        F_spec_gated = alpha * F_spec_RGB + (1 - alpha) * F_spec_Thermal

    Mathematical Property:
        When L -> 0 (Darkness): alpha -> 0, heavily favoring Thermal LWIR emission.
        When L -> 1 (Daylight): alpha -> 1, heavily favoring RGB chromatic texture.
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        self.in_channels = in_channels
        mid_channels = max(16, in_channels // reduction)

        # 1. Global Illumination Conditioning Branch
        self.illum_scale = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 3.0)
        self.illum_mlp = nn.Sequential(
            nn.Linear(1, mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, in_channels)
        )

        # 2. Local Spatial Saliency Branch
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, mid_channels), num_channels=mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=True)
        )

    def forward(
        self,
        feat_rgb_spec: torch.Tensor,
        feat_thermal_spec: torch.Tensor,
        illumination_level: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_rgb_spec: Modality-Specific RGB features [B, C, H, W]
            feat_thermal_spec: Modality-Specific Thermal features [B, C, H, W]
            illumination_level: Illumination scalar in [0, 1] with shape [B, 1]

        Returns:
            gated_spec: Dynamically fused modality-specific features [B, C, H, W]
            alpha: Gating weight map [B, C, H, W] (alpha closer to 1 favors RGB)
        """
        B, C, H, W = feat_rgb_spec.shape

        # Global illumination prior: (L - 0.5) in [-0.5, +0.5] mapped to logits
        illum_centered = (illumination_level - 0.5) * 2.0  # Range [-1.0, +1.0]
        illum_prior = (illum_centered.view(B, 1, 1, 1) * self.illum_scale) + self.illum_mlp(illum_centered).reshape(B, C, 1, 1)

        # Local spatial saliency response
        spec_cat = torch.cat([feat_rgb_spec, feat_thermal_spec], dim=1)
        spatial_logits = self.spatial_conv(spec_cat)

        # Combined dynamic gating logits with illumination prior
        gate_logits = spatial_logits + illum_prior
        alpha = torch.sigmoid(gate_logits)  # [B, C, H, W] in [0, 1]

        # Dynamically re-weight specific features
        gated_spec = alpha * feat_rgb_spec + (1.0 - alpha) * feat_thermal_spec

        return gated_spec, alpha


class IlluminationAwareDecomposition(nn.Module):
    """
    Decomposition Module at a single hierarchical scale.

    Separates incoming RGB (F_rgb) and Thermal (F_th) features into:
    1. Modality-Invariant Features (F_inv_rgb, F_inv_th):
       Shared geometric edges, boundaries, and object contours constrained by L_sim.
    2. Modality-Specific Features (F_spec_rgb, F_spec_th):
       Sensor-exclusive chromatic patterns and infrared heat signatures constrained by L_diff.
    3. Illumination-Aware Dynamic Gating:
       Combines specific features based on illumination_level before concatenation with invariant features.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        # Modality-Invariant Projection Heads (Shared geometric subspace)
        self.proj_inv_rgb = ConvNeXtBlock1x1(in_channels, in_channels)
        self.proj_inv_thermal = ConvNeXtBlock1x1(in_channels, in_channels)

        # Modality-Specific Projection Heads (Sensor-exclusive subspace)
        self.proj_spec_rgb = ConvNeXtBlock1x1(in_channels, in_channels)
        self.proj_spec_thermal = ConvNeXtBlock1x1(in_channels, in_channels)

        # Illumination-Aware Spatial Gate
        self.spatial_gate = IlluminationAwareSpatialGate(in_channels)

        # Final Feature Fusion Head: Concatenates Invariant and Gated-Specific features
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, in_channels), num_channels=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        )

    def forward(
        self,
        feat_rgb: torch.Tensor,
        feat_thermal: torch.Tensor,
        illumination_level: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            feat_rgb: RGB feature tensor at scale i [B, C, H, W]
            feat_thermal: Thermal feature tensor at scale i [B, C, H, W]
            illumination_level: Scene illumination factor [B, 1]

        Returns:
            Dictionary containing fused_feature, feat_inv_rgb, feat_inv_th,
            feat_inv_shared, feat_spec_rgb, feat_spec_th, feat_spec_gated, gate_alpha.
        """
        # 1. Project into Modality-Invariant Subspace
        feat_inv_rgb = self.proj_inv_rgb(feat_rgb)
        feat_inv_th  = self.proj_inv_thermal(feat_thermal)
        feat_inv_shared = 0.5 * (feat_inv_rgb + feat_inv_th)

        # 2. Project into Modality-Specific Subspace
        feat_spec_rgb = self.proj_spec_rgb(feat_rgb)
        feat_spec_th  = self.proj_spec_thermal(feat_thermal)

        # 3. Dynamic Illumination-Aware Spatial Gating
        feat_spec_gated, gate_alpha = self.spatial_gate(
            feat_spec_rgb, feat_spec_th, illumination_level
        )

        # 4. Feature Synthesis: Concatenation of Invariant & Gated Specific Features
        fused_concat = torch.cat([feat_inv_shared, feat_spec_gated], dim=1)
        fused_feature = self.fusion_conv(fused_concat)

        return {
            "fused_feature": fused_feature,
            "feat_inv_rgb": feat_inv_rgb,
            "feat_inv_th": feat_inv_th,
            "feat_inv_shared": feat_inv_shared,
            "feat_spec_rgb": feat_spec_rgb,
            "feat_spec_th": feat_spec_th,
            "feat_spec_gated": feat_spec_gated,
            "gate_alpha": gate_alpha
        }


class DecompositionLossEngine(nn.Module):
    """
    Computes mathematical regularization losses to enforce feature disentanglement:

    1. Similarity Loss (L_sim):
       Penalizes differences between RGB and Thermal Modality-Invariant features:
           L_sim = MSE(F_inv_RGB, F_inv_Thermal)

    2. Difference / Orthogonality Loss (L_diff):
       Enforces mutual orthogonality across feature subspaces using normalized
       Frobenius inner products to drive mutual information to zero:
           L_diff = Ortho(F_inv, F_spec_RGB) + Ortho(F_inv, F_spec_Thermal) + Ortho(F_spec_RGB, F_spec_Thermal)
    """

    def __init__(self, lambda_sim: float = 1.0, lambda_diff: float = 0.5):
        super().__init__()
        self.lambda_sim = lambda_sim
        self.lambda_diff = lambda_diff

    @staticmethod
    def _orthogonality_constraint(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Computes normalized cosine squared orthogonality loss between flattened spatial feature vectors:
            loss = ( <x, y> / (||x||_2 * ||y||_2 + eps) )^2
        """
        x_flat = x.flatten(start_dim=1)
        y_flat = y.flatten(start_dim=1)

        dot_product = (x_flat * y_flat).sum(dim=1)
        x_norm = torch.norm(x_flat, p=2, dim=1)
        y_norm = torch.norm(y_flat, p=2, dim=1)

        cos_sim = dot_product / (x_norm * y_norm + eps)
        ortho_loss = (cos_sim ** 2).mean()
        return ortho_loss

    def forward(
        self,
        decomp_outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Computes L_sim and L_diff across all 4 hierarchical pyramid scales.
        """
        total_l_sim = torch.tensor(0.0, device=decomp_outputs[0]["feat_inv_rgb"].device)
        total_l_diff = torch.tensor(0.0, device=decomp_outputs[0]["feat_inv_rgb"].device)
        stage_records = []

        for stage_idx, stage_dict in enumerate(decomp_outputs):
            f_inv_rgb = stage_dict["feat_inv_rgb"]
            f_inv_th  = stage_dict["feat_inv_th"]
            f_inv_shared = stage_dict["feat_inv_shared"]
            f_spec_rgb = stage_dict["feat_spec_rgb"]
            f_spec_th  = stage_dict["feat_spec_th"]

            # 1. Similarity Loss (MSE on Invariant Geometries)
            stage_l_sim = F.mse_loss(f_inv_rgb, f_inv_th)

            # 2. Difference / Orthogonality Loss (3 Subspace Interactions)
            diff_inv_spec_rgb = self._orthogonality_constraint(f_inv_shared, f_spec_rgb)
            diff_inv_spec_th  = self._orthogonality_constraint(f_inv_shared, f_spec_th)
            diff_spec_inter   = self._orthogonality_constraint(f_spec_rgb, f_spec_th)

            stage_l_diff = diff_inv_spec_rgb + diff_inv_spec_th + diff_spec_inter

            total_l_sim = total_l_sim + stage_l_sim
            total_l_diff = total_l_diff + stage_l_diff

            stage_records.append({
                "stage": stage_idx + 1,
                "l_sim": stage_l_sim.item(),
                "l_diff": stage_l_diff.item(),
                "diff_inv_rgb": diff_inv_spec_rgb.item(),
                "diff_inv_th": diff_inv_spec_th.item(),
                "diff_spec_inter": diff_spec_inter.item()
            })

        num_stages = len(decomp_outputs)
        avg_l_sim = total_l_sim / num_stages
        avg_l_diff = total_l_diff / num_stages

        loss_decomp_total = (self.lambda_sim * avg_l_sim) + (self.lambda_diff * avg_l_diff)

        return {
            "loss_sim": avg_l_sim,
            "loss_diff": avg_l_diff,
            "loss_decomp_total": loss_decomp_total,
            "stage_records": stage_records
        }
