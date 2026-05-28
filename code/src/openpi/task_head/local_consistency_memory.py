from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as _OfficialMamba
except Exception:
    _OfficialMamba = None


def _sinusoidal_pos_embed(length: int, dim: int, device, dtype) -> torch.Tensor:
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    pos = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / max(1, dim)))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class _DepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(channels, channels, kernel_size=self.kernel_size, padding=0, groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        return self.conv(x).transpose(1, 2)


class _SelectiveSSMBlock(nn.Module):
    """Small Mamba-style selective SSM block used when mamba_ssm is unavailable."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        del d_state  # Kept for API compatibility with official Mamba.
        self.d_model = int(d_model)
        self.d_inner = int(d_model) * int(expand)
        self.norm = nn.LayerNorm(self.d_model)
        self.in_proj = nn.Linear(self.d_model, self.d_inner)
        self.dwconv = _DepthwiseConv1d(self.d_inner, kernel_size=d_conv)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        self.gate_proj = nn.Linear(self.d_model, self.d_inner)
        self.A_log = nn.Parameter(torch.randn(self.d_inner) * 0.02)
        self.B_log = nn.Parameter(torch.randn(self.d_inner) * 0.02)
        self.C = nn.Parameter(torch.randn(self.d_inner) * 0.02)
        self.out_proj = nn.Linear(self.d_inner, self.d_model)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.res_scale = nn.Parameter(torch.tensor(1.0))

        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.dt_proj.bias)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        h = self.in_proj(x_norm)
        u = self.dwconv(h)

        a = -F.softplus(self.A_log)
        b = F.softplus(self.B_log)
        c = self.C
        dt = F.softplus(self.dt_proj(h)) + 1e-3

        state = torch.zeros(x.shape[0], self.d_inner, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(x.shape[1]):
            alpha = torch.exp(a.unsqueeze(0) * dt[:, t])
            beta = b.unsqueeze(0) * dt[:, t] * u[:, t]
            state = alpha * state + beta
            ys.append((c.unsqueeze(0) * state).unsqueeze(1))
        y = torch.cat(ys, dim=1)
        y = y * torch.sigmoid(self.gate_proj(x_norm))
        y = self.dropout(self.out_proj(y))
        return residual + self.res_scale * y


class _OfficialMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        if _OfficialMamba is None:
            raise RuntimeError("mamba_ssm is not installed, cannot build official Mamba block.")
        self.norm = nn.LayerNorm(int(d_model))
        self.mamba = _OfficialMamba(d_model=int(d_model), d_state=int(d_state), d_conv=int(d_conv), expand=int(expand))
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mamba(self.norm(x)))


def _resolve_mamba_impl(mamba_impl: str) -> str:
    impl = str(mamba_impl).lower()
    if impl == "auto":
        return "mamba_ssm" if _OfficialMamba is not None else "selective_ssm"
    if impl in {"mamba_ssm", "selective_ssm"}:
        return impl
    raise ValueError(f"Unknown mamba_impl: {mamba_impl}")


class LocalConsistencyMemory(nn.Module):
    """LCM: cross-attention consistency layer followed by Mamba temporal modeling."""

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        hidden: int = 256,
        n_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.0,
        use_tanh: bool = False,
        mamba_impl: str = "auto",
        mamba_state: int = 16,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.hidden = int(hidden)
        self.use_tanh = bool(use_tanh)
        self.mamba_impl = _resolve_mamba_impl(mamba_impl)

        self.prev_proj = nn.Linear(self.action_dim, self.hidden)
        self.prior_proj = nn.Linear(self.action_dim, self.hidden)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden,
            num_heads=int(n_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.cross_drop = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.cross_norm = nn.LayerNorm(self.hidden)
        self.cross_ffn = nn.Sequential(
            nn.Linear(self.hidden, self.hidden * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(self.hidden * 4, self.hidden),
        )
        self.cross_ffn_norm = nn.LayerNorm(self.hidden)

        block_cls = _OfficialMambaBlock if self.mamba_impl == "mamba_ssm" else _SelectiveSSMBlock
        self.temporal = nn.ModuleList(
            [
                block_cls(
                    d_model=self.hidden,
                    d_state=int(mamba_state),
                    d_conv=int(mamba_conv),
                    expand=int(mamba_expand),
                    dropout=float(dropout),
                )
                for _ in range(int(n_layers))
            ]
        )

        self.out_norm = nn.LayerNorm(self.hidden)
        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.action_dim),
        )

    def forward(
        self,
        prev_chunk: torch.Tensor,
        prior_chunk: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
    ):
        del h  # Mamba models temporal structure inside the chunk; no GRU hidden state is used.
        if prev_chunk.ndim != 3:
            raise ValueError(f"prev_chunk must be [B,H,A], got {tuple(prev_chunk.shape)}")
        B, H, A = prev_chunk.shape
        if H != self.horizon or A != self.action_dim:
            raise ValueError(f"prev_chunk shape mismatch: expected H={self.horizon}, A={self.action_dim}, got H={H}, A={A}")

        if prior_chunk is None:
            prior_chunk = torch.zeros_like(prev_chunk)
        if prior_chunk.ndim == 2:
            prior_chunk = prior_chunk.unsqueeze(0)
        if prior_chunk.shape[0] == 1 and B > 1:
            prior_chunk = prior_chunk.expand(B, -1, -1).contiguous()
        if prior_chunk.shape != prev_chunk.shape:
            raise ValueError(f"prior_chunk shape must match prev_chunk, got {tuple(prior_chunk.shape)} vs {tuple(prev_chunk.shape)}")

        pos = _sinusoidal_pos_embed(H, self.hidden, prev_chunk.device, prev_chunk.dtype)
        q = self.prev_proj(prev_chunk) + pos
        kv = self.prior_proj(prior_chunk.to(device=prev_chunk.device, dtype=prev_chunk.dtype)) + pos

        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = self.cross_norm(q + self.cross_drop(attn_out))
        x = self.cross_ffn_norm(x + self.cross_drop(self.cross_ffn(x)))

        for block in self.temporal:
            x = block(x)

        bias = self.out_proj(self.out_norm(x))
        if self.use_tanh:
            bias = torch.tanh(bias)
        return bias, None
