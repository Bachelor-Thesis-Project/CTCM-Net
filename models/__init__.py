"""
CTCM-Net: Cross-Task Consistency Multi-Modal Network Package.
=============================================================
Modular PyTorch architecture for multi-modal RGB-Thermal perception:
- Backbone: Dual-Stream ConvNeXt V2 (Tiny) with FCMAE weights
- Fusion: Illumination-Aware Modality Decomposition (L_sim, L_diff)
- Task Decoders: Semantic Segmentation, Crowd Density, Action Recognition
- Cross-Task Consistency Engine: Bidirectional spatial regularization (L_mask-density)
- Full Architecture: CTCM_Net
"""

from .constants import (
    PYRAMID_CHANNELS,
    PYRAMID_STRIDES,
    NUM_SEGMENTATION_CLASSES,
    PEDESTRIAN_CLASS_INDEX,
    NUM_ACTION_CLASSES
)
from .backbone import DualStreamConvNeXtV2
from .decomposition import (
    ConvNeXtBlock1x1,
    IlluminationAwareSpatialGate,
    IlluminationAwareDecomposition,
    DecompositionLossEngine
)
from .phase2_pipeline import CTCMNetPhase2
from .heads import (
    SemanticSegmentationDecoder,
    ASPPConv,
    CrowdDensityDecoder,
    ActionRecognitionHead
)
from .consistency import CrossTaskConsistencyEngine
from .ctcm_net import CTCM_Net

__all__ = [
    # Constants
    "PYRAMID_CHANNELS",
    "PYRAMID_STRIDES",
    "NUM_SEGMENTATION_CLASSES",
    "PEDESTRIAN_CLASS_INDEX",
    "NUM_ACTION_CLASSES",
    # Backbone
    "DualStreamConvNeXtV2",
    # Decomposition & Fusion
    "ConvNeXtBlock1x1",
    "IlluminationAwareSpatialGate",
    "IlluminationAwareDecomposition",
    "DecompositionLossEngine",
    # Phase 2 Pipeline
    "CTCMNetPhase2",
    # Dense Prediction Heads
    "SemanticSegmentationDecoder",
    "ASPPConv",
    "CrowdDensityDecoder",
    "ActionRecognitionHead",
    # Consistency Regularization
    "CrossTaskConsistencyEngine",
    # Unified Network
    "CTCM_Net"
]
