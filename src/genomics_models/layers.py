"""Common layers for AlphaGenome (PyTorch)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU approximation used in the JAX model."""
    return torch.sigmoid(torch.tensor(1.702).to(x.dtype) * x) * x


def pool(x: torch.Tensor, by: int = 2, reduce: str = "max") -> torch.Tensor:
    """Pool along the sequence dimension of a [B, S, D] tensor."""
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor [B, S, D], got {x.shape}.")
    if reduce == "max":
        pooled = F.max_pool1d(x.transpose(1, 2), kernel_size=by, stride=by)
    elif reduce in {"avg", "mean"}:
        pooled = F.avg_pool1d(x.transpose(1, 2), kernel_size=by, stride=by)
    else:
        raise NotImplementedError(f"Reduce method={reduce} unknown.")
    return pooled.transpose(1, 2)


def pool_channels_first(
    x: torch.Tensor,
    by: int = 2,
    reduce: str = "max",
) -> torch.Tensor:
    """Pool along the sequence dimension of a [B, C, L] tensor."""
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor [B, C, L], got {x.shape}.")
    if reduce == "max":
        return F.max_pool1d(x, kernel_size=by, stride=by)
    if reduce in {"avg", "mean"}:
        return F.avg_pool1d(x, kernel_size=by, stride=by)
    raise NotImplementedError(f"Reduce method={reduce} unknown.")


class RMSBatchNorm(nn.Module):
    """RMSBatchNorm (inference-only)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("var_ema", torch.ones(1, 1, dim))
        self.scale = nn.Parameter(torch.ones(1, 1, dim))
        self.offset = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv = self.scale * torch.rsqrt(self.var_ema + self.eps).to(x.dtype)
        return x * inv + self.offset


class RMSBatchNormChannelsFirst(nn.Module):
    """RMSBatchNorm variant for [B, C, L] tensors."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Keep parameter shapes aligned with the original [B, S, C] module so
        # existing checkpoints load without conversion.
        self.register_buffer("var_ema", torch.ones(1, 1, dim))
        self.scale = nn.Parameter(torch.ones(1, 1, dim))
        self.offset = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv = (self.scale * torch.rsqrt(self.var_ema + self.eps).to(x.dtype)).transpose(1, 2)
        offset = self.offset.transpose(1, 2)
        return x * inv + offset


class LayerNorm(nn.Module):
    """Layer norm with optional RMS normalization."""

    def __init__(self, dim: int, rms_norm: bool = False, eps: float = 1e-5):
        super().__init__()
        self.rms_norm = rms_norm
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0:
            raise ValueError("LayerNorm expects at least 1 dimension.")
        input_dtype = x.dtype
        if not self.rms_norm:
            mean = x.float().mean(dim=-1, keepdim=True)
            x = x - mean.to(input_dtype)
            var = (x.float() ** 2).mean(dim=-1, keepdim=True)
        else:
            var = (x.float() ** 2).mean(dim=-1, keepdim=True)
        inv = torch.rsqrt(var + self.eps).to(input_dtype)
        weight = self.weight.reshape((1,) * (x.ndim - 1) + (-1,))
        bias = self.bias.reshape((1,) * (x.ndim - 1) + (-1,))
        return x * inv * weight + bias


__all__ = [
    "gelu",
    "pool",
    "pool_channels_first",
    "RMSBatchNorm",
    "RMSBatchNormChannelsFirst",
    "LayerNorm",
]
