"""
CTCM-Net: Cross-Task Consistency Multi-Modal Network
=====================================================
Phase 2: Dual-Stream ConvNeXt V2 Backbone & Illumination-Aware Decomposition

This module provides the core multi-modal feature extraction and representation
disentanglement architecture for CTCM-Net:
1. DualStreamConvNeXtV2: Dual-stream hierarchical encoder initialized with
   ImageNet-22k pretrained FCMAE ConvNeXt V2 (Tiny) weights, extracting 4-scale
   feature pyramids (strides 4, 8, 16, 32).
2. IlluminationAwareSpatialGate: Spatial-channel dynamic gating mechanism that
   modulates thermal vs. RGB modality-specific feature contributions conditioned
   on physical scene illumination.
3. IlluminationAwareDecomposition: Multi-scale mathematical projection isolating
   Modality-Invariant (shared geometry) and Modality-Specific (sensor-exclusive)
   features at all 4 pyramid stages.
4. DecompositionLossEngine: PyTorch implementation of Similarity Loss (L_sim)
   and Difference/Orthogonality Loss (L_diff) enforcing subspace separation.
5. CTCMNetPhase2: Unified Phase 2 architectural pipeline.

Authors: CTCM-Net Research Team
Target Hardware: NVIDIA RTX A6000 (48GB VRAM)
"""

from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ==============================================================================
# ARCHITECTURAL CONSTANTS
# ==============================================================================

# ConvNeXt V2 (Tiny) Feature Pyramid Channel Dimensions
# Stage 1: 1/4  resolution (64x64)  -> 96 channels
# Stage 2: 1/8  resolution (32x32)  -> 192 channels
# Stage 3: 1/16 resolution (16x16)  -> 384 channels
# Stage 4: 1/32 resolution (8x8)    -> 768 channels
PYRAMID_CHANNELS: List[int] = [96, 192, 384, 768]
PYRAMID_STRIDES: List[int] = [4, 8, 16, 32]

# Multi-Task Dense Prediction Specifications
NUM_SEGMENTATION_CLASSES: int = 9   # MFNet Semantic Segmentation (9 classes)
PEDESTRIAN_CLASS_INDEX: int = 2     # "person" class in MFNet (Class ID 2)
NUM_ACTION_CLASSES: int = 27        # DarkAct Action Recognition (27 classes)


# ==============================================================================
# MODULE 1: DUAL-STREAM CONVNEXT V2 ENCODER
# ==============================================================================

class DualStreamConvNeXtV2(nn.Module):
    """
    Dual-stream hierarchical backbone instantiated from ConvNeXt V2 (Tiny)
    with Fully Convolutional Masked Autoencoder (FCMAE) ImageNet-22k weights.

    Extracts synchronized multi-scale feature pyramids from visible (RGB) and
    long-wave infrared thermal (LWIR) modalities:
        RGB     -> [F_rgb_1, F_rgb_2, F_rgb_3, F_rgb_4]
        Thermal -> [F_th_1,  F_th_2,  F_th_3,  F_th_4]
    """

    def __init__(
        self,
        model_name: str = "convnextv2_tiny.fcmae",
        pretrained: bool = True,
        in_chans: int = 3
    ):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained

        # 1. RGB Modality Feature Extractor (Visible spectrum stream)
        self.rgb_encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_chans
        )

        # 2. Thermal (LWIR) Modality Feature Extractor (Infrared radiation stream)
        self.thermal_encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_chans
        )

        self.channels = PYRAMID_CHANNELS
        self.strides = PYRAMID_STRIDES

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through both backbone streams.

        Args:
            rgb: Normalized RGB tensor [B, 3, 256, 256]
            thermal: Normalized Thermal tensor [B, 3, 256, 256]

        Returns:
            rgb_features: 4-stage feature pyramid [S1, S2, S3, S4]
            thermal_features: 4-stage feature pyramid [S1, S2, S3, S4]
        """
        rgb_features = self.rgb_encoder(rgb)
        thermal_features = self.thermal_encoder(thermal)
        return rgb_features, thermal_features


# ==============================================================================
# MODULE 2: ILLUMINATION-AWARE SPATIAL GATE
# ==============================================================================

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
        # Learnable channel-wise scaling vector calibrated to illumination prior
        self.illum_scale = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 3.0)
        self.illum_mlp = nn.Sequential(
            nn.Linear(1, mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, in_channels)
        )

        # 2. Local Spatial Saliency Branch
        # Ingests concatenated specific features [F_spec_rgb, F_spec_th]
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
        # When L -> 1 (day), illum_centered > 0 -> alpha -> 1 (favors RGB)
        # When L -> 0 (night), illum_centered < 0 -> alpha -> 0 (favors Thermal)
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


# ==============================================================================
# MODULE 3: ILLUMINATION-AWARE MODALITY DECOMPOSITION MODULE
# ==============================================================================

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
        # [F_inv, F_spec_gated] -> 2*C -> C
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
            Dictionary containing:
                fused_feature: Fused multi-modal feature map [B, C, H, W]
                feat_inv_rgb: Invariant features from RGB [B, C, H, W]
                feat_inv_th: Invariant features from Thermal [B, C, H, W]
                feat_inv_shared: Averaged shared invariant representation [B, C, H, W]
                feat_spec_rgb: Specific features from RGB [B, C, H, W]
                feat_spec_th: Specific features from Thermal [B, C, H, W]
                feat_spec_gated: Illumination-reweighted specific features [B, C, H, W]
                gate_alpha: Gating spatial weight map [B, C, H, W]
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


# ==============================================================================
# MODULE 4: DECOMPOSITION LOSS ENGINE (L_sim & L_diff)
# ==============================================================================

class DecompositionLossEngine(nn.Module):
    """
    Computes mathematical regularization losses to enforce feature disentanglement:

    1. Similarity Loss (L_sim):
       Penalizes differences between RGB and Thermal Modality-Invariant features,
       forcing them to converge to an identical geometric manifold:
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
        B = x.shape[0]
        # Flatten spatial and channel dimensions: [B, C*H*W]
        x_flat = x.flatten(start_dim=1)
        y_flat = y.flatten(start_dim=1)

        # Dot product per batch item: [B]
        dot_product = (x_flat * y_flat).sum(dim=1)

        # L2 norms
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

        Args:
            decomp_outputs: List of 4 decomposition dictionaries from Stages 1..4

        Returns:
            Dictionary containing:
                loss_sim: Total Similarity Loss
                loss_diff: Total Orthogonality / Difference Loss
                loss_decomp_total: Weighted sum (lambda_sim * L_sim + lambda_diff * L_diff)
                stage_losses: Per-stage breakdown of L_sim and L_diff
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
            # a. Invariant vs. RGB-Specific
            diff_inv_spec_rgb = self._orthogonality_constraint(f_inv_shared, f_spec_rgb)
            # b. Invariant vs. Thermal-Specific
            diff_inv_spec_th  = self._orthogonality_constraint(f_inv_shared, f_spec_th)
            # c. RGB-Specific vs. Thermal-Specific
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

        # Average across the 4 stages
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


# ==============================================================================
# MODULE 5: UNIFIED CTCM-NET PHASE 2 MODEL
# ==============================================================================

class CTCMNetPhase2(nn.Module):
    """
    CTCM-Net Phase 2 End-to-End Architectural Pipeline:
    Dual-Stream ConvNeXt V2 Backbone + 4-Stage Illumination-Aware Modality Decomposition.
    """

    def __init__(
        self,
        backbone_name: str = "convnextv2_tiny.fcmae",
        pretrained: bool = True
    ):
        super().__init__()
        # 1. Dual-Stream Hierarchical Backbone
        self.backbone = DualStreamConvNeXtV2(
            model_name=backbone_name,
            pretrained=pretrained,
            in_chans=3
        )

        # 2. 4-Stage Illumination-Aware Modality Decomposition Modules
        self.decomp_stages = nn.ModuleList([
            IlluminationAwareDecomposition(in_channels=ch)
            for ch in PYRAMID_CHANNELS
        ])

        # 3. Regularization Loss Engine
        self.loss_engine = DecompositionLossEngine(lambda_sim=1.0, lambda_diff=0.5)

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        illumination_level: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Args:
            rgb: Normalized RGB batch [B, 3, 256, 256]
            thermal: Normalized Thermal batch [B, 3, 256, 256]
            illumination_level: Physical scene illumination scalar [B, 1]

        Returns:
            Dictionary containing:
                fused_pyramid: List of 4 fused tensors [B, Ci, Hi, Wi] at strides [4, 8, 16, 32]
                decomp_stages_output: List of stage-wise decomposition outputs
                decomp_losses: Calculated L_sim, L_diff, and L_decomp_total
        """
        # 1. Dual-stream hierarchical feature extraction
        rgb_feats, thermal_feats = self.backbone(rgb, thermal)

        # 2. Hierarchical decomposition and gating at all 4 scales
        fused_pyramid: List[torch.Tensor] = []
        decomp_stages_output: List[Dict[str, torch.Tensor]] = []

        for stage_idx in range(4):
            stage_out = self.decomp_stages[stage_idx](
                rgb_feats[stage_idx],
                thermal_feats[stage_idx],
                illumination_level
            )
            fused_pyramid.append(stage_out["fused_feature"])
            decomp_stages_output.append(stage_out)

        # 3. Compute Decomposition Regularization Losses (L_sim, L_diff)
        decomp_losses = self.loss_engine(decomp_stages_output)

        return {
            "fused_pyramid": fused_pyramid,
            "decomp_stages_output": decomp_stages_output,
            "decomp_losses": decomp_losses,
            "rgb_pyramid": rgb_feats,
            "thermal_pyramid": thermal_feats
        }


# ==============================================================================
# MODULE 6: SEMANTIC SEGMENTATION TOP-DOWN DECODER (MFNet: 9 Classes)
# ==============================================================================

class SemanticSegmentationDecoder(nn.Module):
    """
    Top-Down Feature Pyramid Decoder for High-Resolution Semantic Segmentation.
    Ingests the 4 multi-scale fused features [F1, F2, F3, F4] and progressively
    upsamples and fuses lateral skip features to generate high-resolution logits:
        Output: [B, num_classes=9, 256, 256]
    """

    def __init__(
        self,
        in_channels: List[int] = PYRAMID_CHANNELS,
        num_classes: int = NUM_SEGMENTATION_CLASSES
    ):
        super().__init__()
        c1, c2, c3, c4 = in_channels  # [96, 192, 384, 768]

        # Lateral 1x1 Convolutions projecting pyramid stages to decoder channels
        self.lateral4 = nn.Conv2d(c4, 256, kernel_size=1)
        self.lateral3 = nn.Conv2d(c3, 256, kernel_size=1)
        self.lateral2 = nn.Conv2d(c2, 128, kernel_size=1)
        self.lateral1 = nn.Conv2d(c1, 64, kernel_size=1)

        # Inter-stage projection layers
        self.p4_to_p3 = nn.Conv2d(256, 256, kernel_size=1)
        self.p3_to_p2 = nn.Conv2d(256, 128, kernel_size=1)
        self.p2_to_p1 = nn.Conv2d(128, 64, kernel_size=1)

        # Smooth Convolutions
        self.smooth3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=256),
            nn.GELU()
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=128),
            nn.GELU()
        )
        self.smooth1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.GELU()
        )

        # High-Resolution Progressive Upsampling Head (64x64 -> 128x128 -> 256x256)
        self.head = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=48),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 64 -> 128
            nn.Conv2d(48, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 128 -> 256
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.GELU(),
            nn.Conv2d(32, num_classes, kernel_size=1)  # 9 classes
        )

    def forward(self, pyramid: List[torch.Tensor]) -> torch.Tensor:
        f1, f2, f3, f4 = pyramid  # [64x64], [32x32], [16x16], [8x8]

        # Top-down pathway
        p4 = self.lateral4(f4)
        p3_in = self.lateral3(f3) + F.interpolate(self.p4_to_p3(p4), scale_factor=2, mode="bilinear", align_corners=True)
        p3 = self.smooth3(p3_in)

        p2_in = self.lateral2(f2) + F.interpolate(self.p3_to_p2(p3), scale_factor=2, mode="bilinear", align_corners=True)
        p2 = self.smooth2(p2_in)

        p1_in = self.lateral1(f1) + F.interpolate(self.p2_to_p1(p2), scale_factor=2, mode="bilinear", align_corners=True)
        p1 = self.smooth1(p1_in)

        # Progressive upsampling to 256x256
        seg_logits = self.head(p1)  # [B, 9, 256, 256]
        return seg_logits


# ==============================================================================
# MODULE 7: CROWD DENSITY MULTI-SCALE CONTEXT AGGREGATOR (RGBT-CC: Continuous 2D)
# ==============================================================================

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.GELU()
        )


class CrowdDensityDecoder(nn.Module):
    """
    Multi-Scale Context Aggregator for Continuous Crowd Density Estimation.
    Combines high-receptive-field ASPP multi-scale dilation with high-resolution
    skip features to estimate continuous density response maps:
        Output: [B, 1, 256, 256] (strictly non-negative continuous density)
    """

    def __init__(self, in_channels: List[int] = PYRAMID_CHANNELS):
        super().__init__()
        c1, c2, c3, c4 = in_channels  # [96, 192, 384, 768]
        mid_ch = 128

        # 1. Atrous Spatial Pyramid Pooling (ASPP) on deep context Stage 4 (c4 = 768)
        self.aspp_1x1 = nn.Sequential(
            nn.Conv2d(c4, mid_ch, 1, bias=False),
            nn.GroupNorm(8, mid_ch),
            nn.GELU()
        )
        self.aspp_d2 = ASPPConv(c4, mid_ch, dilation=2)
        self.aspp_d4 = ASPPConv(c4, mid_ch, dilation=4)
        self.aspp_d6 = ASPPConv(c4, mid_ch, dilation=6)
        self.aspp_proj = nn.Sequential(
            nn.Conv2d(mid_ch * 4, mid_ch, 1, bias=False),
            nn.GroupNorm(8, mid_ch),
            nn.GELU()
        )

        # 2. Multi-Scale Context Aggregation & Progressive Fusion
        self.skip3_proj = nn.Conv2d(c3, mid_ch, 1)  # 384 -> 128
        self.skip2_proj = nn.Conv2d(c2, 64, 1)      # 192 -> 64
        self.skip1_proj = nn.Conv2d(c1, 32, 1)      # 96 -> 32

        self.fuse_stage3 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, mid_ch),
            nn.GELU()
        )
        self.fuse_stage2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU()
        )
        self.fuse_stage1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU()
        )

        # 3. High-Resolution Density Head (64x64 -> 128x128 -> 256x256)
        self.density_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 64 -> 128
            nn.Conv2d(32, 24, 3, padding=1, bias=False),
            nn.GroupNorm(4, 24),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 128 -> 256
            nn.Conv2d(24, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1)  # 1-channel continuous response
        )

    def forward(self, pyramid: List[torch.Tensor]) -> torch.Tensor:
        f1, f2, f3, f4 = pyramid  # [64x64], [32x32], [16x16], [8x8]

        # ASPP context on Stage 4
        aspp_out = torch.cat([
            self.aspp_1x1(f4),
            self.aspp_d2(f4),
            self.aspp_d4(f4),
            self.aspp_d6(f4)
        ], dim=1)
        aspp_feats = self.aspp_proj(aspp_out)  # [B, 128, 8, 8]

        # Progressive Context Aggregation
        up3 = F.interpolate(aspp_feats, scale_factor=2, mode="bilinear", align_corners=True) + self.skip3_proj(f3)
        feat3 = self.fuse_stage3(up3)  # [B, 128, 16, 16]

        up2 = F.interpolate(feat3[:, :64], scale_factor=2, mode="bilinear", align_corners=True) + self.skip2_proj(f2)
        feat2 = self.fuse_stage2(up2)  # [B, 64, 32, 32]

        up1 = F.interpolate(feat2[:, :32], scale_factor=2, mode="bilinear", align_corners=True) + self.skip1_proj(f1)
        feat1 = self.fuse_stage1(up1)  # [B, 32, 64, 64]

        # Output continuous density map (ReLU enforces physical non-negativity D >= 0)
        density_logits = self.density_head(feat1)
        density_map = F.relu(density_logits)  # [B, 1, 256, 256]
        return density_map


# ==============================================================================
# MODULE 8: ACTION RECOGNITION HEAD (DarkAct: 27 Classes)
# ==============================================================================

class ActionRecognitionHead(nn.Module):
    """
    Global Context Pooling and Classification Head for Human Action Recognition.
    Ingests highest-level semantic representations from Stage 4 [B, 768, 8, 8],
    applies dual global average and max pooling, followed by an MLP classifier:
        Output: [B, num_classes=27] logits
    """

    def __init__(
        self,
        in_channels: int = PYRAMID_CHANNELS[-1],  # 768
        num_classes: int = NUM_ACTION_CLASSES,    # 27
        dropout: float = 0.25
    ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        pooled_dim = in_channels * 2  # 768 * 2 = 1536

        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.75),
            nn.Linear(256, num_classes)
        )

    def forward(self, pyramid: List[torch.Tensor]) -> torch.Tensor:
        f4 = pyramid[-1]  # [B, 768, 8, 8]
        avg_feat = self.avg_pool(f4).flatten(start_dim=1)  # [B, 768]
        max_feat = self.max_pool(f4).flatten(start_dim=1)  # [B, 768]
        pooled = torch.cat([avg_feat, max_feat], dim=1)    # [B, 1536]
        action_logits = self.classifier(pooled)             # [B, 27]
        return action_logits


# ==============================================================================
# MODULE 9: CROSS-TASK CONSISTENCY ENGINE (L_mask-density)
# ==============================================================================

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


# ==============================================================================
# MODULE 10: UNIFIED CTCM_NET MULTI-TASK ARCHITECTURE
# ==============================================================================

class CTCM_Net(nn.Module):
    """
    CTCM-Net: Unified Cross-Task Consistency Multi-Modal Network.

    Integrates:
    - Phase 2: Dual-Stream ConvNeXt V2 (Tiny) Encoder + Illumination-Aware Modality Decomposition
    - Phase 3: Semantic Segmentation Head (MFNet, 9-cls), Crowd Density Head (RGBT-CC),
               Action Recognition Head (DarkAct, 27-cls), and Cross-Task Consistency Engine.
    """

    def __init__(
        self,
        backbone_name: str = "convnextv2_tiny.fcmae",
        pretrained: bool = True,
        num_seg_classes: int = NUM_SEGMENTATION_CLASSES,
        num_action_classes: int = NUM_ACTION_CLASSES,
        beta_consistency: float = 1.0
    ):
        super().__init__()
        # 1. Phase 2 Feature Extractor & Decomposition Engine
        self.phase2_engine = CTCMNetPhase2(
            backbone_name=backbone_name,
            pretrained=pretrained
        )

        # 2. Phase 3 Dense Task Prediction Heads
        self.seg_head = SemanticSegmentationDecoder(
            in_channels=PYRAMID_CHANNELS,
            num_classes=num_seg_classes
        )
        self.density_head = CrowdDensityDecoder(
            in_channels=PYRAMID_CHANNELS
        )
        self.action_head = ActionRecognitionHead(
            in_channels=PYRAMID_CHANNELS[-1],
            num_classes=num_action_classes
        )

        # 3. Phase 3 Cross-Task Consistency Engine (L_mask-density)
        self.consistency_engine = CrossTaskConsistencyEngine(
            pedestrian_class_idx=PEDESTRIAN_CLASS_INDEX,
            beta=beta_consistency
        )

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        illumination_level: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Forward pass through complete multi-task architecture.

        Returns:
            Dictionary containing:
                seg_logits: [B, 9, 256, 256]
                density_map: [B, 1, 256, 256]
                action_logits: [B, 27]
                loss_decomp: Dictionary with Phase 2 losses (L_sim, L_diff, L_decomp_total)
                consistency_outputs: Dictionary with Phase 3 L_mask-density, L_out, L_align
                fused_pyramid: Multi-scale fused features [S1..S4]
        """
        # 1. Phase 2: Feature extraction and modality decomposition
        phase2_out = self.phase2_engine(rgb, thermal, illumination_level)
        fused_pyramid = phase2_out["fused_pyramid"]

        # 2. Phase 3: Task-specific prediction heads
        seg_logits = self.seg_head(fused_pyramid)
        density_map = self.density_head(fused_pyramid)
        action_logits = self.action_head(fused_pyramid)

        # 3. Phase 3: Cross-Task Consistency Regularization
        consistency_outputs = self.consistency_engine(seg_logits, density_map)

        return {
            "seg_logits": seg_logits,
            "density_map": density_map,
            "action_logits": action_logits,
            "loss_decomp": phase2_out["decomp_losses"],
            "consistency_outputs": consistency_outputs,
            "fused_pyramid": fused_pyramid,
            "decomp_stages_output": phase2_out["decomp_stages_output"]
        }
