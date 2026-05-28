from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TaskHeadMLP(nn.Module):
    """Map pooled VLM prefix tokens to a normalized task embedding."""

    def __init__(self, in_dim: int, hidden: int = 1024, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)
