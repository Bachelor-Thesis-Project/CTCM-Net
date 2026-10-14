"""
CTCM-Net: Unified Cross-Task Consistency Multi-Modal Network.
=============================================================
Integrates:
- Phase 2: Dual-Stream ConvNeXt V2 (Tiny) Encoder + Illumination-Aware Modality Decomposition
- Phase 3: Semantic Segmentation Head (MFNet, 9-cls), Crowd Density Head (RGBT-CC),
           Action Recognition Head (DarkAct, 27-cls), and Cross-Task Consistency Engine.
"""

from typing import Any, Dict
import torch
import torch.nn as nn

from .constants import (
    PYRAMID_CHANNELS,
    NUM_SEGMENTATION_CLASSES,
    NUM_ACTION_CLASSES,
    PEDESTRIAN_CLASS_INDEX
)
from .phase2_pipeline import CTCMNetPhase2
from .heads import (
    SemanticSegmentationDecoder,
    CrowdDensityDecoder,
    ActionRecognitionHead
)
from .consistency import CrossTaskConsistencyEngine


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
                decomp_stages_output: List of stage-wise decomposition outputs
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
