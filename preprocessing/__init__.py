"""
CTCM-Net: Preprocessing & Data Pipeline Package
===============================================
Exposes the modular 5-step preprocessing pipeline and dataset classes for:
- MFNet (RGB-Thermal Urban Semantic Segmentation)
- RGBT-CC (RGB-Thermal Crowd Density Estimation)
- DarkAct (Synchronized Night Action Recognition)
"""

from preprocessing.homography import SpatialPlanarHomographyAlignment
from preprocessing.temporal_sync import TemporalSynchronizer, HardwareTemporalGenlockSampler, MAX_SYNC_JITTER_MS
from preprocessing.target_formulation import (
    encode_mfnet_mask,
    generate_gaussian_density_heatmap,
    encode_darkact_action,
    MFNET_CLASSES,
    NUM_SEG_CLASSES,
    DARKACT_CLASSES,
    NUM_ACTION_CLASSES,
    TARGET_RESOLUTION
)
from preprocessing.transforms import AlignedMultimodalTransform
from preprocessing.normalization import (
    TrainSetNormalizer,
    DATASET_TRAIN_STATISTICS
)
from preprocessing.mfnet import MFNetDataset, build_mfnet_dataloader
from preprocessing.rgbtcc import RGBTCCDataset, build_rgbtcc_dataloader, rgbtcc_collate_fn
from preprocessing.darkact import DarkActDataset, build_darkact_dataloader

__all__ = [
    "SpatialPlanarHomographyAlignment",
    "TemporalSynchronizer",
    "HardwareTemporalGenlockSampler",
    "MAX_SYNC_JITTER_MS",
    "encode_mfnet_mask",
    "generate_gaussian_density_heatmap",
    "encode_darkact_action",
    "MFNET_CLASSES",
    "NUM_SEG_CLASSES",
    "DARKACT_CLASSES",
    "NUM_ACTION_CLASSES",
    "TARGET_RESOLUTION",
    "AlignedMultimodalTransform",
    "TrainSetNormalizer",
    "DATASET_TRAIN_STATISTICS",
    "MFNetDataset",
    "build_mfnet_dataloader",
    "RGBTCCDataset",
    "build_rgbtcc_dataloader",
    "rgbtcc_collate_fn",
    "DarkActDataset",
    "build_darkact_dataloader",
]
