"""
Dual-Stream ConvNeXt V2 Backbone Module.
=========================================
Dual-stream hierarchical encoder initialized with ImageNet-22k pretrained
FCMAE ConvNeXt V2 (Tiny) weights, extracting 4-scale feature pyramids (strides 4, 8, 16, 32).
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import timm

from .constants import PYRAMID_CHANNELS, PYRAMID_STRIDES


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
