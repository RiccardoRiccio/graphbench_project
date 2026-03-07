
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
        x: torch.Tensor,               
        attn_bias: torch.Tensor,         
        key_padding_mask: torch.Tensor,  
    ) -> torch.Tensor:
        Bsz, L, D = x.shape
        H = self.num_heads
        Hd = self.head_dim

        q = self.wq(x).view(Bsz, L, H, Hd).transpose(1, 2)  
        k = self.wk(x).view(Bsz, L, H, Hd).transpose(1, 2)  
        v = self.wv(x).view(Bsz, L, H, Hd).transpose(1, 2)  

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(Hd)  
        scores = scores + attn_bias

       
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v) 
        out = out.transpose(1, 2).contiguous().view(Bsz, L, D)  
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

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.ln_ffn = nn.LayerNorm(dim)
        hidden_ffn = dim * ffn_mult
        self.ffn_w1 = nn.Linear(dim, hidden_ffn)
        self.ffn_w2 = nn.Linear(hidden_ffn, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # Attention sublayer (phi)
        h = self.ln_attn(x)
        h = self.attn(h, attn_bias=attn_bias, key_padding_mask=key_padding_mask)
        h = self.attn_mlp(h)
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


class GTGraphBench(nn.Module):
    
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
        ffn_mult: int = 1,   
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

        
        x_dense, mask = to_dense_batch(x, batch=batch) 
        Bsz, L, _ = x_dense.shape

        h = self.node_encoder(x_dense)  # [B, L, D]

     
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

        
        cls_mask = mask.new_ones((Bsz, 1))
        full_valid = torch.cat([cls_mask, mask], dim=1)          # [B, 1+L]
        key_padding_mask = ~full_valid                           

        
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