"""3D CNN encoder with late fusion for multi-phase liver lesion classification.

Architecture:
  PhaseEncoder: 4 × ConvBlock3D (Conv3d → BN → ReLU → MaxPool) followed by
    AdaptiveAvgPool3d → 256-dim feature vector.
    Input shape: (B, 1, 32, 64, 64)
    Channel progression: 1→32→64→128→256
    After 4 MaxPool(2,2): spatial 32→2, 64→4, 64→4 → (B,256,2,4,4)
    After AdaptiveAvgPool3d(1,1,1): (B,256)

  MultiPhaseClassifier: one PhaseEncoder with SHARED WEIGHTS applied to each
    phase independently; phase features concatenated; 2-layer MLP head.

Parameter count (8 phases):
  Encoder: ~1.16M params
  Head (2048→256→2): ~0.53M
  Total: ~1.69M  (within 1-3M target)
"""
from typing import List

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Conv3d(3×3×3) → BN → ReLU → MaxPool(2×2×2)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PhaseEncoder(nn.Module):
    """4-block 3D CNN; (B,1,32,64,64) → (B,256)."""

    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            ConvBlock3D(1, 32),    # → (B, 32, 16, 32, 32)
            ConvBlock3D(32, 64),   # → (B, 64,  8, 16, 16)
            ConvBlock3D(64, 128),  # → (B,128,  4,  8,  8)
            ConvBlock3D(128, 256), # → (B,256,  2,  4,  4)
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.enc(x)).flatten(1)  # (B, 256)


class MultiPhaseClassifier(nn.Module):
    """Shared-weight encoder per phase → concatenate → classify.

    Args:
        n_phases: number of input phases.
        dropout: dropout probability in the head.
    """

    def __init__(self, n_phases: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.n_phases = n_phases
        self.encoder = PhaseEncoder()
        feat_dim = 256 * n_phases
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, phases: List[torch.Tensor]) -> torch.Tensor:
        """Args: phases, a list of n_phases tensors each shaped (B,1,32,64,64).
        Returns: (B, 2) logits.
        """
        feats = [self.encoder(p) for p in phases]
        return self.head(torch.cat(feats, dim=1))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
