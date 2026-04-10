"""
RoPE and p-RoPE Transformer models for the MQAR synthetic task.

Based on:
  Arora et al., "Zoology: Measuring and Improving Recall in Efficient Language Models" (2023)

MQAR task (Definition 3.1):
  Given key-value pairs placed at the start of a sequence, followed by queries,
  the model must recall the value associated with each queried key.

Models:
  - RoPETransformer: Standard multi-head attention with Rotary Position Embeddings.
  - pRoPETransformer: Hybrid where a fraction p of heads use no positional encoding
    (NoPE-like, pure semantic matching) and the remaining (1-p) heads use standard RoPE.
  - GOATTransformer: Dot-product attention plus a learned additive Fourier bias over
    relative positions. Each head learns alpha[r], beta[r] coefficients for R
    frequencies geometrically spaced to match RoPE's frequency set.
  - ALiBiTransformer: Attention with Linear Biases (Press et al., 2022). Each head
    adds a fixed linear penalty m_h * (i - j) to attention logits, with head-specific
    slopes set to the geometric sequence prescribed in the original paper.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def precompute_freqs_cis(dim: int, max_seq_len: int, base: float = 10_000.0) -> torch.Tensor:
    """Precompute the complex exponential frequencies for RoPE.

    Returns a (max_seq_len, dim//2) complex tensor.
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (seq_len, dim//2)
    return torch.polar(torch.ones_like(freqs), freqs)  # e^{i * theta}


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to x.

    Args:
        x: (batch, n_heads, seq_len, head_dim)
        freqs_cis: (seq_len, head_dim//2) complex
    Returns:
        Tensor with same shape as x.
    """
    # Reshape x into pairs: (..., head_dim//2, 2) -> complex
    xf = x.float().reshape(*x.shape[:-1], -1, 2)
    xc = torch.view_as_complex(xf)  # (..., head_dim//2)

    # Broadcast freqs_cis to match: (1, 1, seq_len, head_dim//2)
    freqs = freqs_cis[None, None, :xc.shape[2], :]

    # Apply rotation and convert back
    xr = torch.view_as_real(xc * freqs).flatten(-2)
    return xr.type_as(x)


# ---------------------------------------------------------------------------
# Causal self-attention with optional per-head RoPE control
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Supports three modes controlled by ``rope_head_mask``:
      - All heads use RoPE (standard RoPE model)
      - Some heads use RoPE, others do not (p-RoPE model)
      - No heads use RoPE (NoPE baseline, if desired)

    Args:
        d_model: Model / embedding dimension.
        n_heads: Number of attention heads.
        max_seq_len: Maximum sequence length (for RoPE precomputation).
        rope_base: Base frequency for RoPE (default 10,000).
        rope_head_mask: Boolean tensor of shape (n_heads,).
            True  = apply RoPE to this head.
            False = no positional encoding (NoPE) for this head.
            If None, all heads use RoPE.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        rope_base: float = 10_000.0,
        rope_head_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Precompute RoPE frequencies
        freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len, base=rope_base)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Head mask: which heads get RoPE
        if rope_head_mask is None:
            rope_head_mask = torch.ones(n_heads, dtype=torch.bool)
        self.register_buffer("rope_head_mask", rope_head_mask, persistent=False)

        self.rope_heads = rope_head_mask.sum().item()
        self.nope_heads = n_heads - self.rope_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        B, T, D = x.shape
        qkv = self.Wqkv(x)  # (B, T, 3*D)
        q, k, v = qkv.split(D, dim=-1)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE only to designated heads
        if self.rope_heads > 0 and self.nope_heads > 0:
            # Mixed mode: split, apply RoPE to subset, recombine
            rope_idx = self.rope_head_mask.nonzero(as_tuple=True)[0]
            nope_idx = (~self.rope_head_mask).nonzero(as_tuple=True)[0]

            q_rope = apply_rotary_emb(q[:, rope_idx], self.freqs_cis[:T])
            k_rope = apply_rotary_emb(k[:, rope_idx], self.freqs_cis[:T])

            q_out = torch.empty_like(q)
            k_out = torch.empty_like(k)
            q_out[:, rope_idx] = q_rope
            k_out[:, rope_idx] = k_rope
            q_out[:, nope_idx] = q[:, nope_idx]
            k_out[:, nope_idx] = k[:, nope_idx]
            q, k = q_out, k_out
        elif self.rope_heads > 0:
            # All heads use RoPE
            q = apply_rotary_emb(q, self.freqs_cis[:T])
            k = apply_rotary_emb(k, self.freqs_cis[:T])
        # else: all NoPE, q and k are unchanged

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        attn = attn + causal_mask[None, None, :, :]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# GOAT self-attention: dot-product + learned Fourier additive bias
# ---------------------------------------------------------------------------

class GOATSelfAttention(nn.Module):
    """Multi-head causal self-attention with an additive Fourier relative-position bias.

    Logits are computed as:
        z[i,j] = (Q @ K.T) / sqrt(d_h) + bias[i,j]

    where the bias is a truncated Fourier series over relative displacement:
        bias[i,j] = sum_r  alpha[r] * cos(omega[r] * (i-j))
                          + beta[r]  * sin(omega[r] * (i-j))

    Frequencies omega[r] use the same geometric spacing as RoPE:
        omega[r] = base^{-2r / d_h},  r = 0, ..., R-1

    By default R = head_dim // 2 so the frequency set exactly matches RoPE.

    Args:
        d_model: Model / embedding dimension.
        n_heads: Number of attention heads.
        max_seq_len: Maximum sequence length (for precomputing the displacement matrix).
        rope_base: Base for the geometric frequency spacing (default 10,000).
        R: Number of Fourier frequencies per head. Defaults to head_dim // 2.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        rope_base: float = 10_000.0,
        R: Optional[int] = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        if R is None:
            R = self.head_dim // 2
        self.R = R

        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Fixed geometric frequencies matching RoPE: omega_r = base^{-2r / head_dim}
        omega = rope_base ** (-2.0 * torch.arange(R, dtype=torch.float32) / self.head_dim)
        self.register_buffer("omega", omega, persistent=False)  # (R,)

        # Learnable Fourier coefficients per head, initialized to small values
        self.alpha = nn.Parameter(torch.randn(n_heads, R) * 0.01)  # (n_heads, R)
        self.beta = nn.Parameter(torch.randn(n_heads, R) * 0.01)   # (n_heads, R)

        # Precompute relative displacement matrix for max_seq_len
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        delta = pos.unsqueeze(0) - pos.unsqueeze(1)  # (L, L) where delta[i,j] = i - j
        self.register_buffer("delta", delta, persistent=False)

    def _compute_bias(self, T: int) -> torch.Tensor:
        """Compute the additive positional bias for sequence length T.

        Returns: (n_heads, T, T)
        """
        delta = self.delta[:T, :T]  # (T, T)
        # omega * delta: broadcast (R,) with (T, T) -> (R, T, T)
        angles = self.omega[:, None, None] * delta[None, :, :]

        # Sum over R frequencies: alpha[h, r] * cos(...) + beta[h, r] * sin(...)
        # alpha: (n_heads, R) -> (n_heads, R, 1, 1)
        cos_term = self.alpha[:, :, None, None] * torch.cos(angles)[None, :, :, :]
        sin_term = self.beta[:, :, None, None] * torch.sin(angles)[None, :, :, :]
        # Sum over R dimension -> (n_heads, T, T)
        bias = (cos_term + sin_term).sum(dim=1)
        return bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        B, T, D = x.shape
        qkv = self.Wqkv(x)
        q, k, v = qkv.split(D, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product logits
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Additive Fourier bias (NOT scaled by 1/sqrt(d))
        bias = self._compute_bias(T)  # (n_heads, T, T)
        attn = attn + bias[None, :, :, :]

        # Causal mask
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        attn = attn + causal_mask[None, None, :, :]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# ALiBi self-attention (Press et al., 2022)
# ---------------------------------------------------------------------------

def _get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Compute the per-head slopes for ALiBi following the original paper.

    For n_heads that is a power of 2:
        slopes = 2^{-8/n_heads}, 2^{-16/n_heads}, ..., 2^{-8}

    For non-power-of-2, use the nearest larger power of 2 and interleave.
    """
    def _get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(8.0 / n))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return torch.tensor(_get_slopes_power_of_2(n_heads), dtype=torch.float32)
    else:
        closest_power = 2 ** math.floor(math.log2(n_heads))
        base_slopes = _get_slopes_power_of_2(closest_power)
        extra_slopes = _get_slopes_power_of_2(2 * closest_power)
        # Take alternating slopes from the finer grid for the remaining heads
        extra = [extra_slopes[i] for i in range(1, 2 * closest_power, 2)]
        slopes = base_slopes + extra[: n_heads - closest_power]
        return torch.tensor(slopes, dtype=torch.float32)


class ALiBiSelfAttention(nn.Module):
    """Multi-head causal self-attention with ALiBi positional bias.

    Logits are computed as:
        z[i,j] = (Q @ K.T) / sqrt(d_h) + m_h * (i - j)

    where m_h is a fixed, non-learned slope per head following the geometric
    schedule from Press et al. (2022).  Since the causal mask ensures j <= i,
    the bias (i - j) is non-positive (a recency penalty).

    Args:
        d_model: Model / embedding dimension.
        n_heads: Number of attention heads.
        max_seq_len: Maximum sequence length (for precomputing the bias matrix).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Fixed ALiBi slopes (not learned)
        slopes = _get_alibi_slopes(n_heads)  # (n_heads,)
        self.register_buffer("slopes", slopes, persistent=False)

        # Precompute relative position bias: slopes[h] * (i - j)
        # For causal attention (i - j) <= 0, so this is a penalty on distant tokens.
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        # (L, L) where entry [i,j] = i - j  (negative for j > i, masked anyway)
        rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)
        # (n_heads, L, L)
        alibi_bias = slopes[:, None, None] * rel_pos[None, :, :]
        self.register_buffer("alibi_bias", alibi_bias, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        B, T, D = x.shape
        qkv = self.Wqkv(x)
        q, k, v = qkv.split(D, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product logits + ALiBi bias
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn + self.alibi_bias[None, :, :T, :T]

        # Causal mask
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        attn = attn + causal_mask[None, None, :, :]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# SwiGLU MLP (following LLaMA / Zoology paper)
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU feed-forward block as used in LLaMA."""

    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()
        if d_ff is None:
            # Standard 8/3 expansion, rounded to nearest multiple of 8
            d_ff = int(2 * (4 * d_model) / 3)
            d_ff = ((d_ff + 7) // 8) * 8
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        rope_base: float = 10_000.0,
        rope_head_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            rope_head_mask=rope_head_mask,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = SwiGLU(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Full Transformer (RoPE or p-RoPE)
# ---------------------------------------------------------------------------

class RoPETransformer(nn.Module):
    """Transformer with RoPE positional encoding for the MQAR task.

    Architecture follows the Zoology paper's synthetic setup:
      - Token embedding (no separate positional embedding; RoPE is in attention)
      - L Transformer blocks (sequence mixer + SwiGLU MLP with LayerNorm)
      - Linear head projecting back to vocabulary

    Args:
        vocab_size: Size of the token vocabulary.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer blocks.
        max_seq_len: Maximum sequence length.
        rope_base: RoPE base frequency (default 10,000).
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 64,
        n_heads: int = 1,
        n_layers: int = 2,
        max_seq_len: int = 512,
        rope_base: float = 10_000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
                rope_head_mask=None,  # all heads use RoPE
            )
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token ids
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        h = self.tok_emb(x)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)


class pRoPETransformer(nn.Module):
    """Transformer with partial RoPE (p-RoPE) for the MQAR task.

    A fraction ``p`` of attention heads in every layer use NO positional
    encoding (NoPE-like, pure semantic / content-based matching).  The
    remaining ``(1-p)`` heads use standard RoPE.

    This lets us study how much positional information is actually needed
    for associative recall vs. pure content matching.

    Args:
        vocab_size: Size of the token vocabulary.
        d_model: Model dimension.
        n_heads: Number of attention heads (must be >= 2 for meaningful split).
        n_layers: Number of Transformer blocks.
        max_seq_len: Maximum sequence length.
        rope_base: RoPE base frequency (default 10,000).
        p: Fraction of heads reserved for NoPE (no rotation). Default 0.5.
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        max_seq_len: int = 512,
        rope_base: float = 10_000.0,
        p: float = 0.5,
    ):
        super().__init__()
        assert n_heads >= 2, "p-RoPE requires at least 2 heads to split between RoPE and NoPE"
        assert 0.0 < p < 1.0, "p must be in (0, 1)"

        self.d_model = d_model
        self.p = p

        # Compute how many heads are NoPE vs RoPE
        n_nope = max(1, round(p * n_heads))
        n_rope = n_heads - n_nope
        assert n_rope >= 1, "Must have at least 1 RoPE head; reduce p or increase n_heads"

        # Build a boolean mask: first n_nope heads are NoPE (False), rest are RoPE (True)
        rope_head_mask = torch.cat([
            torch.zeros(n_nope, dtype=torch.bool),
            torch.ones(n_rope, dtype=torch.bool),
        ])

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
                rope_head_mask=rope_head_mask,
            )
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

        # Store config for easy inspection
        self.n_nope = n_nope
        self.n_rope = n_rope

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token ids
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        h = self.tok_emb(x)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)


class GOATTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        rope_base: float = 10_000.0,
        R: Optional[int] = None,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GOATSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            R=R,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = SwiGLU(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GOATTransformer(nn.Module):
    """Transformer with GOAT positional encoding for the MQAR task.

    GOAT uses standard dot-product attention plus a learned additive bias
    computed as a truncated Fourier series over relative positions.  Each
    head independently learns alpha[r] and beta[r] coefficients for R
    frequencies that are geometrically spaced to match RoPE's frequency set.

    Args:
        vocab_size: Size of the token vocabulary.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer blocks.
        max_seq_len: Maximum sequence length.
        rope_base: Base for geometric frequency spacing (default 10,000).
        R: Number of Fourier frequencies per head. Defaults to head_dim // 2
           (matching the number of RoPE frequency components).
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 64,
        n_heads: int = 1,
        n_layers: int = 2,
        max_seq_len: int = 512,
        rope_base: float = 10_000.0,
        R: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            GOATTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
                R=R,
            )
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

        # Store R for inspection
        self.R = self.blocks[0].attn.R

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # Note: alpha/beta in GOATSelfAttention are initialized in __init__,
        # not here, to preserve the small-value initialization.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token ids
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        h = self.tok_emb(x)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)


class ALiBiTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = ALiBiSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = SwiGLU(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ALiBiTransformer(nn.Module):
    """Transformer with ALiBi positional encoding for the MQAR task.

    ALiBi (Press et al., 2022) adds a fixed, non-learned linear bias
    m_h * (i - j) to attention logits.  Each head has a different slope m_h
    following a geometric schedule, giving heads different "attention spans".

    No positional embedding is used; position information comes entirely
    from the additive ALiBi bias in each attention layer.

    Args:
        vocab_size: Size of the token vocabulary.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer blocks.
        max_seq_len: Maximum sequence length.
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 64,
        n_heads: int = 1,
        n_layers: int = 2,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            ALiBiTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
            )
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token ids
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        h = self.tok_emb(x)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# MQAR synthetic data generation (Algorithm 1 from the paper)
# ---------------------------------------------------------------------------

def generate_mqar_batch(
    vocab_size: int = 8192,
    seq_len: int = 128,
    num_kv_pairs: int = 8,
    alpha: float = 0.1,
    batch_size: int = 64,
    random_seed: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of MQAR sequences following Algorithm 1 of the Zoology paper.

    The vocabulary is split in half: the first half are key tokens, the second
    half are value tokens.  ``num_kv_pairs`` random (key, value) bigrams are
    placed at the start of the sequence (positions 0..2D-1).  Each key is then
    repeated (queried) at a later position drawn from a power-law distribution;
    the label at that position is the associated value.  All other positions
    are filled with a random noise token and their label is set to -100
    (ignored by cross-entropy loss).

    Args:
        vocab_size: Total vocabulary size (split equally into keys and values).
        seq_len: Length of each generated sequence (N in the paper).
        num_kv_pairs: Number of key-value pairs per example (D in the paper).
        alpha: Power-law exponent controlling query-position distribution.
        batch_size: Number of sequences to generate.
        random_seed: Optional seed for reproducibility.

    Returns:
        input_ids:  (batch_size, seq_len) int64 tensor of token ids.
        labels:     (batch_size, seq_len) int64 tensor. At query positions the
                    label is the correct value token; everywhere else it is -100.
    """
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random.RandomState()

    half = vocab_size // 2
    key_vocab = np.arange(0, half)           # first half: keys
    value_vocab = np.arange(half, vocab_size)  # second half: values

    assert 2 * num_kv_pairs < seq_len, (
        f"Need seq_len > 2*num_kv_pairs, got {seq_len} and {num_kv_pairs}"
    )

    input_ids = np.zeros((batch_size, seq_len), dtype=np.int64)
    labels = np.full((batch_size, seq_len), -100, dtype=np.int64)

    # Use a fixed noise token (last value token) for non-KV/query positions
    noise_token = int(value_vocab[-1])

    for b in range(batch_size):
        # 1-2. Randomly pair keys with values
        keys = rng.choice(key_vocab, size=num_kv_pairs, replace=False)
        values = rng.choice(value_vocab[:-1], size=num_kv_pairs, replace=False)

        # 3-4. Place KV pairs at the start: positions 0,1, 2,3, ..., 2D-2, 2D-1
        for d in range(num_kv_pairs):
            input_ids[b, 2 * d] = keys[d]
            input_ids[b, 2 * d + 1] = values[d]

        # Fill remaining positions with noise
        kv_end = 2 * num_kv_pairs
        input_ids[b, kv_end:] = noise_token

        # 5. Place query (second occurrence of each key) at power-law distributed positions
        available_positions = np.arange(kv_end, seq_len)
        # Power-law weights: P(pos) ~ (pos - kv_end + 1)^{-alpha}
        weights = (available_positions - kv_end + 1).astype(np.float64) ** (-alpha)
        weights /= weights.sum()

        # Sample D distinct query positions
        query_positions = rng.choice(
            available_positions, size=num_kv_pairs, replace=False, p=weights
        )
        query_positions.sort()

        for d in range(num_kv_pairs):
            pos = query_positions[d]
            input_ids[b, pos] = keys[d]
            labels[b, pos] = values[d]

    return torch.from_numpy(input_ids), torch.from_numpy(labels)


# ---------------------------------------------------------------------------
# Convenience: build everything ready to train
# ---------------------------------------------------------------------------

def build_rope_model(
    vocab_size: int = 8192,
    d_model: int = 64,
    n_heads: int = 1,
    n_layers: int = 2,
    max_seq_len: int = 512,
    rope_base: float = 10_000.0,
) -> RoPETransformer:
    """Instantiate a standard RoPE Transformer configured for MQAR."""
    return RoPETransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        rope_base=rope_base,
    )


def build_prope_model(
    vocab_size: int = 8192,
    d_model: int = 64,
    n_heads: int = 2,
    n_layers: int = 2,
    max_seq_len: int = 512,
    rope_base: float = 10_000.0,
    p: float = 0.5,
) -> pRoPETransformer:
    """Instantiate a p-RoPE Transformer configured for MQAR.

    With the default p=0.5 and n_heads=2, one head uses RoPE and the
    other performs pure content-based (NoPE) attention.
    """
    return pRoPETransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        rope_base=rope_base,
        p=p,
    )


def build_goat_model(
    vocab_size: int = 8192,
    d_model: int = 64,
    n_heads: int = 1,
    n_layers: int = 2,
    max_seq_len: int = 512,
    rope_base: float = 10_000.0,
    R: Optional[int] = None,
) -> GOATTransformer:
    """Instantiate a GOAT Transformer configured for MQAR.

    By default R = head_dim // 2, so the Fourier frequency set matches RoPE's.
    """
    return GOATTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        rope_base=rope_base,
        R=R,
    )


def build_alibi_model(
    vocab_size: int = 8192,
    d_model: int = 64,
    n_heads: int = 1,
    n_layers: int = 2,
    max_seq_len: int = 512,
) -> ALiBiTransformer:
    """Instantiate an ALiBi Transformer configured for MQAR."""
    return ALiBiTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
    )


# ---------------------------------------------------------------------------
# Quick sanity check (prints model summaries and runs a forward pass)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    VOCAB = 8192
    D_MODEL = 64
    SEQ_LEN = 128
    NUM_KV = 8
    BATCH = 4

    # --- Generate MQAR data ---
    inputs, labels = generate_mqar_batch(
        vocab_size=VOCAB,
        seq_len=SEQ_LEN,
        num_kv_pairs=NUM_KV,
        alpha=0.1,
        batch_size=BATCH,
        random_seed=42,
    )
    print(f"MQAR data: inputs {inputs.shape}, labels {labels.shape}")
    print(f"  Query positions per example: {(labels != -100).sum(dim=1).tolist()}")

    # --- RoPE model ---
    rope_model = build_rope_model(
        vocab_size=VOCAB, d_model=D_MODEL, max_seq_len=SEQ_LEN,
    )
    logits = rope_model(inputs)
    print(f"\nRoPE model:")
    print(f"  Parameters: {sum(p.numel() for p in rope_model.parameters()):,}")
    print(f"  Output shape: {logits.shape}")

    loss = F.cross_entropy(
        logits.view(-1, VOCAB), labels.view(-1), ignore_index=-100,
    )
    print(f"  Initial loss: {loss.item():.4f}")

    # --- p-RoPE model ---
    prope_model = build_prope_model(
        vocab_size=VOCAB, d_model=D_MODEL, n_heads=2, max_seq_len=SEQ_LEN, p=0.5,
    )
    logits2 = prope_model(inputs)
    print(f"\np-RoPE model (p=0.5):")
    print(f"  NoPE heads: {prope_model.n_nope}, RoPE heads: {prope_model.n_rope}")
    print(f"  Parameters: {sum(p.numel() for p in prope_model.parameters()):,}")
    print(f"  Output shape: {logits2.shape}")

    loss2 = F.cross_entropy(
        logits2.view(-1, VOCAB), labels.view(-1), ignore_index=-100,
    )
    print(f"  Initial loss: {loss2.item():.4f}")

    # --- GOAT model ---
    goat_model = build_goat_model(
        vocab_size=VOCAB, d_model=D_MODEL, max_seq_len=SEQ_LEN,
    )
    logits3 = goat_model(inputs)
    print(f"\nGOAT model (R={goat_model.R}):")
    print(f"  Parameters: {sum(p.numel() for p in goat_model.parameters()):,}")
    print(f"  Output shape: {logits3.shape}")

    loss3 = F.cross_entropy(
        logits3.view(-1, VOCAB), labels.view(-1), ignore_index=-100,
    )
    print(f"  Initial loss: {loss3.item():.4f}")

    # --- ALiBi model ---
    alibi_model = build_alibi_model(
        vocab_size=VOCAB, d_model=D_MODEL, max_seq_len=SEQ_LEN,
    )
    logits4 = alibi_model(inputs)
    print(f"\nALiBi model:")
    print(f"  Parameters: {sum(p.numel() for p in alibi_model.parameters()):,}")
    print(f"  Output shape: {logits4.shape}")

    loss4 = F.cross_entropy(
        logits4.view(-1, VOCAB), labels.view(-1), ignore_index=-100,
    )
    print(f"  Initial loss: {loss4.item():.4f}")
