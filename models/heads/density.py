"""
Crowd Density Estimation Multi-Scale Context Aggregator Head (RGBT-CC: Continuous 2D).
======================================================================================
Combines ASPP multi-scale dilation with high-resolution skip features to predict
continuous density maps [B, 1, 256, 256].
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import PYRAMID_CHANNELS


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
