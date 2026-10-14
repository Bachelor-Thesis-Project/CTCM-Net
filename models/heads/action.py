"""
Human Action Recognition Classification Head (DarkAct: 27 Classes).
=====================================================================
Global Context Pooling and Classification Head for Action Recognition
ingesting Stage 4 semantic representations to output [B, 27] logits.
"""

from typing import List
import torch
import torch.nn as nn

from ..constants import PYRAMID_CHANNELS, NUM_ACTION_CLASSES


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
