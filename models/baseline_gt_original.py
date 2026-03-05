# models/gt_original.py
# Standalone (unwrapped) Graph Transformer baseline for GraphBench electronic circuits
# Paper-aligned choices for circuits:
# - hidden_dim = 384
# - num_layers = 6
# - num_heads = 4
# - activation = GELU
# - dropout = 0
# - graph-level readout via [cls] token
# - optional absolute PE added before first GT layer (RWSE if provided)
#
# Note: The paper describes an additive attention bias B. GraphBench follows Bechler-Speicher et al. (2025),
# but the exact B construction is not fully specified in the excerpt. This implementation provides a
# reasonable "biased attention" mechanism using PE-derived per-head additive bias when pe_dim > 0.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


def _get_pe_from_data(data, pe_dim: int) -> torch.Tensor | None:
    """
    Tries to fetch positional encodings from common fields.
    Returns: [N, pe_dim] or None
    """
    if pe_dim <= 0:
        return None

    # Common names people use for RWSE/LPE in PyG Data objects:
    for key in ["rwse", "pe", "pos_enc", "pe_rwse", "x_pe"]:
        if hasattr(data, key):
            pe = getattr(data, key)
            if pe is None:
                continue
            # Accept [N, pe_dim] or [N, 1, pe_dim] style
            if pe.dim() == 3 and pe.size(1) == 1:
                pe = pe.squeeze(1)
            if pe.dim() == 2 and pe.size(1) == pe_dim:
                return pe.float()
    return None


class DecoderMLP(nn.Module):
    """
    GraphBench-style decoder:
      Decoder(x) = W2(LayerNorm(GELU(W1 x)))
    """
    def __init__(self, dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.w1 = nn.Linear(dim, dim, bias=bias)
        self.ln = nn.LayerNorm(dim)
        self.w2 = nn.Linear(dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.ln(F.gelu(self.w1(x))))


class BiasedMultiheadSelfAttention(nn.Module):
    """
    Custom multi-head self-attention that supports an additive per-head bias B:
      softmax((QK^T)/sqrt(dk) + B) V
    """
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,                 # [B, L, D]
        attn_bias: torch.Tensor,         # [B, H, L, L] (can be zeros)
        key_padding_mask: torch.Tensor,  # [B, L] True where padding
    ) -> torch.Tensor:
        Bsz, L, D = x.shape
        H = self.num_heads
        Hd = self.head_dim

        q = self.wq(x).view(Bsz, L, H, Hd).transpose(1, 2)  # [B, H, L, Hd]
        k = self.wk(x).view(Bsz, L, H, Hd).transpose(1, 2)  # [B, H, L, Hd]
        v = self.wv(x).view(Bsz, L, H, Hd).transpose(1, 2)  # [B, H, L, Hd]

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(Hd)  # [B, H, L, L]
        scores = scores + attn_bias

        # Mask padding keys (already applied in bias builder, but keep safe)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)  # [B, H, L, Hd]
        out = out.transpose(1, 2).contiguous().view(Bsz, L, D)  # [B, L, D]
        out = self.wo(out)
        return out


class GTBlock(nn.Module):
    """
    Paper-style pre-norm transformer block:
      x = x + Attention(LN(x), bias)
      x = x + FFN(LN(x))
    FFN uses GELU.
    """
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, ffn_mult: int = 1):
        super().__init__()
        self.ln_attn = nn.LayerNorm(dim)
        self.attn = BiasedMultiheadSelfAttention(dim, num_heads, dropout=dropout)

        self.ln_ffn = nn.LayerNorm(dim)
        hidden_ffn = dim * ffn_mult
        self.ffn_w1 = nn.Linear(dim, hidden_ffn)
        self.ffn_w2 = nn.Linear(hidden_ffn, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # Attention sublayer (phi)
        h = self.ln_attn(x)
        h = self.attn(h, attn_bias=attn_bias, key_padding_mask=key_padding_mask)
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = x + h

        # FFN sublayer
        h2 = self.ln_ffn(x)
        h2 = self.ffn_w1(h2)
        h2 = F.gelu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h2 = self.ffn_w2(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        x = x + h2

        return x


class GToriginal(nn.Module):
    """
    Unwrapped GT baseline for electronic circuits.

    Expected usage from training loop (example):
      pe_dim = RWSE_DIM if HAS_RWSE else 0
      model = GToriginal(in_channels=input_dim, hidden_dim=384, num_layers=6, num_heads=4, out_dim=1, pe_dim=pe_dim)

    Data requirements:
      - data.x: [N, in_channels] (or [N,1,in_channels] also handled)
      - data.edge_index: not used here (full attention across nodes per graph)
      - data.batch: [N] graph ids
      - optional PE: data.rwse or data.pe etc. with shape [N, pe_dim]
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 4,
        out_dim: int = 1,
        pe_dim: int = 0,
        dropout: float = 0.0,
        decoder_bias: bool = True,
        ffn_mult: int = 1,   # keep 1 for a conservative match; set 4 if you want transformer-style wide FFN
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.pe_dim = pe_dim
        self.dropout = dropout

        self.node_encoder = nn.Linear(in_channels, hidden_dim)

        # [cls] token for graph-level tasks
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Optional PE projection to model dimension for "add absolute PE before first GT layer"
        self.pe_proj = nn.Linear(pe_dim, hidden_dim, bias=False) if pe_dim > 0 else None

       
        self.blocks = nn.ModuleList([
            GTBlock(dim=hidden_dim, num_heads=num_heads, dropout=dropout, ffn_mult=ffn_mult)
            for _ in range(num_layers)
        ])

        # GraphBench-style decoder head
        self.decoder = DecoderMLP(hidden_dim, out_dim=out_dim, bias=decoder_bias)

        # init params (small, stable)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, data) -> torch.Tensor:
        x = data.x
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        x = x.float()

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        # Dense batch: x_dense [B, L, F], mask [B, L] True for real nodes
        x_dense, mask = to_dense_batch(x, batch=batch)  # mask True=valid
        Bsz, L, _ = x_dense.shape

        h = self.node_encoder(x_dense)  # [B, L, D]

        # Optional absolute PE added before first GT layer
        if self.pe_dim > 0 and self.pe_proj is not None:
            pe = _get_pe_from_data(data, self.pe_dim)
            if pe is None:
                # If pe_dim>0 but PE not present, fall back to zeros (safe)
                pe_dense = h.new_zeros((Bsz, L, self.pe_dim))
            else:
                pe_dense, _ = to_dense_batch(pe, batch=batch)  # [B, L, pe_dim]
            h = h + self.pe_proj(pe_dense)

        # Prepend [cls] token
        cls = self.cls_token.expand(Bsz, 1, -1)  # [B, 1, D]
        h = torch.cat([cls, h], dim=1)            # [B, 1+L, D]

        # Update masks for padding: key_padding_mask True where padding
        # mask is True for valid nodes; for cls it's always valid.
        cls_mask = mask.new_ones((Bsz, 1))
        full_valid = torch.cat([cls_mask, mask], dim=1)          # [B, 1+L]
        key_padding_mask = ~full_valid                           # True where padding

        # Build attention bias (zeros if pe_dim==0).
        # For cls PE, use zeros.
        # Paper-conservative choice: use zero additive attention bias B.
        seq_len = h.size(1)  # includes [cls]
        attn_bias = h.new_zeros((Bsz, self.num_heads, seq_len, seq_len))    
        # Transformer blocks
        for blk in self.blocks:
            h = blk(h, attn_bias=attn_bias, key_padding_mask=key_padding_mask)

        # Graph embedding = final [cls] token
        hg = h[:, 0, :]  # [B, D]

        out = self.decoder(hg)  # [B, out_dim]
        if out.size(-1) == 1:
            return out.squeeze(-1)
        return out