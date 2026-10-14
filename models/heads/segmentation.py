"""
Semantic Segmentation Top-Down Decoder Head (MFNet: 9 Classes).
================================================================
Top-down Feature Pyramid Decoder progressively upsampling fused multi-scale
representations [F1, F2, F3, F4] with lateral skip connections to [B, 9, 256, 256].
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import PYRAMID_CHANNELS, NUM_SEGMENTATION_CLASSES


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
