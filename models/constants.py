"""
CTCM-Net Architectural Constants & Hyperparameters.
===================================================
Defines multi-scale feature pyramid channel depths, spatial strides,
and dataset class configurations.
"""

from typing import List

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
