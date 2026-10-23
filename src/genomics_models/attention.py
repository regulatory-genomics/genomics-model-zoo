from __future__ import annotations

import math

import torch
from torch import nn

import torch.nn.functional as F


_MAX_RELATIVE_DISTANCE = 8192  # 1Mb / 128bp.

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



def _geomspace(start: torch.Tensor, end: torch.Tensor, steps: int, endpoint: bool = False) -> torch.Tensor:
    start = start.to(torch.float64)
    end = end.to(torch.float64)
    if steps <= 1:
        return start.to(start.dtype)
    log_start = torch.log(start)
    log_end = torch.log(end)
    if endpoint:
        t = torch.linspace(0, 1, steps, device=start.device, dtype=log_start.dtype)
    else:
        t = torch.arange(steps, device=start.device, dtype=log_start.dtype) / steps
    return torch.exp(log_start + t * (log_end - log_start)).to(start.dtype)


def _shift(x: torch.Tensor, query_length: int, key_length: int) -> torch.Tensor:
    """Shifts the diagonal of a 2D array."""
    *batch, n_rows, n_diags = x.shape
    x = x.reshape(*batch, n_diags, n_rows)
    x = x[..., 1:, :]
    x = x.reshape(*batch, n_rows, n_diags - 1)
    return x[..., :key_length]


def _central_mask_features(
    distances: torch.Tensor, feature_size: int, seq_length: int
) -> torch.Tensor:
    """Positional features using exponentially-spaced central mask."""
    if feature_size > seq_length:
        raise ValueError("feature_size must be <= seq_length.")
    device = distances.device
    dtype = torch.float32
    start = 1.0
    end = float(seq_length - feature_size + 1)
    log_start = math.log(start)
    log_end = math.log(end)
    log_step = (log_end - log_start) / feature_size
    exponents = torch.arange(feature_size, device=device, dtype=dtype) * log_step
    geomspace_values = torch.exp(
        torch.tensor(log_start, device=device, dtype=dtype) + exponents
    )
    center_widths = torch.arange(feature_size, device=device, dtype=dtype) + geomspace_values
    while center_widths.ndim < distances.ndim:
        center_widths = center_widths.unsqueeze(0)
    return (center_widths > distances.unsqueeze(-1)).to(dtype)


def _apply_rope_inplace(
    x: torch.Tensor, cos_theta: torch.Tensor, sin_theta: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE in-place to match the reference implementation."""
    x_even = x[..., ::2].clone()
    x[..., ::2] = x_even * cos_theta[..., ::2] - x[..., 1::2] * sin_theta[..., ::2]
    x[..., 1::2] = x_even * sin_theta[..., 1::2] + x[..., 1::2] * cos_theta[..., 1::2]
    return x


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor | None,
    max_position: int,
    inplace: bool = False,
) -> torch.Tensor:
    """Applies rotary position embeddings to the input tensor."""
    if positions is None:
        positions = torch.arange(x.shape[1], device=x.device, dtype=x.dtype).reshape(1, -1)
    num_freq = x.shape[-1] // 2
    base_freqs = torch.logspace(
        math.log10(1.0),
        math.log10(max_position - num_freq + 1),
        steps=num_freq,
        base=10,
        device=x.device,
        dtype=x.dtype,
    )
    inv_freq = 1.0 / (torch.arange(num_freq, device=x.device, dtype=x.dtype) + base_freqs)
    theta = torch.einsum("bs,f->bsf", positions, inv_freq)
    theta = theta.repeat_interleave(2, dim=-1).unsqueeze(2)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    if inplace:
        return _apply_rope_inplace(x, cos_theta, sin_theta)
    x_rot = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)
    return x * cos_theta + x_rot * sin_theta


class MLPBlock(nn.Module):
    """MLP block for sequence representations."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSBatchNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.out_norm = RMSBatchNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = torch.relu(self.fc1(h))
        h = self.fc2(h)
        return self.out_norm(h)


class PairMLPBlock(nn.Module):
    """MLP block for pairwise representations."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = LayerNorm(dim, rms_norm=True)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, pair_input: torch.Tensor) -> torch.Tensor:
        x = self.norm(pair_input)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class MHABlock(nn.Module):
    """Multi-head attention block with residual connection."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSBatchNorm(dim)
        self.q = nn.Linear(dim, 8 * 128, bias=False)
        self.k = nn.Linear(dim, 128, bias=False)
        self.v = nn.Linear(dim, 192, bias=False)
        self.norm_q = LayerNorm(128)
        self.norm_k = LayerNorm(128)
        self.norm_v = LayerNorm(192)
        self.proj = nn.Linear(8 * 192, dim)
        self.out_norm = RMSBatchNorm(dim)

    def forward(self, x: torch.Tensor, attention_bias: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = self.norm(x)
        q = self.norm_q(self.q(h).reshape(batch_size, seq_len, 8, 128))
        k = self.norm_k(self.k(h).reshape(batch_size, seq_len, 1, 128))
        v = self.norm_v(self.v(h).reshape(batch_size, seq_len, 1, 192))
        q = apply_rope(q, None, max_position=_MAX_RELATIVE_DISTANCE, inplace=True)
        k = apply_rope(k, None, max_position=_MAX_RELATIVE_DISTANCE, inplace=True)
        q_t = q.permute(0, 2, 1, 3)
        k_t = k.permute(0, 2, 1, 3)
        attention_logits = torch.matmul(q_t, k_t.transpose(-2, -1)).float()
        attention_logits = attention_logits / math.sqrt(128.0)
        attention_logits = attention_logits + attention_bias.float()
        logits_soft_cap = 5.0
        attention_logits = torch.tanh(attention_logits / logits_soft_cap) * logits_soft_cap
        attention_weights = torch.softmax(attention_logits, dim=-1)
        v_t = v.permute(0, 2, 1, 3)
        y = torch.matmul(attention_weights.to(x.dtype), v_t).float()
        y = y.to(x.dtype).permute(0, 2, 1, 3)
        y = self.proj(y.reshape(batch_size, seq_len, -1))
        return self.out_norm(y)



class AttentionBiasBlock(nn.Module):
    """Generates attention bias for multi-head attention."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSBatchNorm(dim)
        self.fc = nn.Linear(dim, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.norm(x))
        x = self.fc(x)
        x = x.repeat_interleave(16, dim=1).repeat_interleave(16, dim=2)
        return x.permute(0, 3, 1, 2)


class RowAttentionBlock(nn.Module):
    """Self-attention applied along rows of pairwise representations."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = LayerNorm(dim, rms_norm=True)
        self.k = nn.Linear(dim, 128, bias=False)
        self.q = nn.Linear(dim, 128, bias=False)
        self.v = nn.Linear(dim, 128)
    def forward(
        self,
        pair_input: torch.Tensor,
        compute_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if compute_dtype is None:
            compute_dtype = pair_input.dtype
        x = pair_input.to(compute_dtype)
        x = self.norm(x)
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)
        scale = 1.0 / math.sqrt(128.0)
        attn = torch.einsum("bpqf,bpkf->bpqk", q, k).float() * scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bpqk,bpkf->bpqf", attn.to(compute_dtype), v).float()
        return out.to(compute_dtype)



class SequenceToPairBlock(nn.Module):
    """Converts sequence representations to pairwise representations."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = LayerNorm(dim, rms_norm=True)
        self.q = nn.Linear(dim, 32 * 128, bias=False)
        self.k = nn.Linear(dim, 32 * 128, bias=False)
        self.pos = nn.Linear(32 * 2, 32 * 128)
        self.q_bias = nn.Parameter(torch.zeros(1, 1, 32, 128))
        self.k_bias = nn.Parameter(torch.zeros(1, 1, 32, 128))
        self.y_q = nn.Linear(dim, 128, bias=False)
        self.y_k = nn.Linear(dim, 128, bias=False)
        self.pair_proj = nn.Linear(32, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(pool(x, by=16, reduce="mean"))
        batch_size, seq_len, _ = x.shape
        q = self.q(x).reshape(batch_size, seq_len, 32, 128)
        k = self.k(x).reshape(batch_size, seq_len, 32, 128)
        relative_positions = torch.arange(
            -seq_len, seq_len, device=x.device, dtype=torch.float32
        )
        pos_features = _central_mask_features(
            distances=relative_positions.abs(),
            feature_size=32,
            seq_length=_MAX_RELATIVE_DISTANCE // 16,
        )
        pos_features = torch.cat(
            [
                pos_features,
                torch.sign(relative_positions).unsqueeze(-1) * pos_features,
            ],
            dim=-1,
        ).to(x.dtype)
        pos_encoding = self.pos(pos_features).reshape(2 * seq_len, 32, 128)
        rel_q_a = _shift(
            torch.einsum("bqhc,phc->bhqp", q + self.q_bias, pos_encoding),
            seq_len,
            seq_len,
        )
        rel_k_a = _shift(
            torch.einsum("bkhc,phc->bhkp", k + self.k_bias, pos_encoding),
            seq_len,
            seq_len,
        )
        a = torch.einsum("bqhc,bkhc->bqkh", q, k)
        rel_q = rel_q_a.permute(0, 2, 3, 1)
        rel_k = rel_k_a.permute(0, 3, 2, 1)
        a = a + 0.5 * (rel_q + rel_k)
        x_gelu = F.gelu(x)
        y_q = self.y_q(x_gelu)
        y_k = self.y_k(x_gelu)
        pair_act = self.pair_proj(a) + y_q[:, :, None, :] + y_k[:, None, :, :]
        return pair_act


class PairUpdateBlock(nn.Module):
    """Updates pairwise representations."""

    def __init__(self, dim: int, pair_dim: int):
        super().__init__()
        self.seq_to_pair = SequenceToPairBlock(dim)
        self.row_attn = RowAttentionBlock(pair_dim)
        self.mlp = PairMLPBlock(pair_dim)

    def forward(
        self,
        sequence_input: torch.Tensor,
        pair_input: torch.Tensor | None,
        compute_dtype: torch.dtype | None = None,
    ):
        y = self.seq_to_pair(sequence_input)
        x = y if pair_input is None else (pair_input + y)
        if compute_dtype is None:
            compute_dtype = x.dtype
        x = x + self.row_attn(x, compute_dtype=compute_dtype)
        x = x + self.mlp(x)
        return x





__all__ = [
    "AttentionBiasBlock",
    "MHABlock",
    "MLPBlock",
    "PairUpdateBlock",
    "SequenceToPairBlock",
    "apply_rope",
]
