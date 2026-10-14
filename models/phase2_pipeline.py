"""
CTCM-Net Phase 2 End-to-End Pipeline.
======================================
Integrates Dual-Stream ConvNeXt V2 Backbone with 4-Stage Illumination-Aware Modality Decomposition.
"""

from typing import Any, Dict, List
import torch
import torch.nn as nn

from .constants import PYRAMID_CHANNELS
from .backbone import DualStreamConvNeXtV2
from .decomposition import IlluminationAwareDecomposition, DecompositionLossEngine


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
                rgb_pyramid: Raw RGB features from backbone
                thermal_pyramid: Raw Thermal features from backbone
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
