import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GPSConv, GINConv, global_mean_pool


class GPSProcessorLayer(nn.Module):
    def __init__(self, d: int, attn_type: str, attn_kwargs: Dict[str, Any], heads: int = 4):
        super().__init__()
        self.norm_phi = nn.LayerNorm(d)
        local_mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.phi = GPSConv(
            channels=d,
            conv=GINConv(local_mlp),
            heads=heads,
            attn_type=attn_type,
            attn_kwargs=attn_kwargs,
        )
        self.norm_ffn_outer = nn.LayerNorm(d)
        self.norm_ffn_inner = nn.LayerNorm(d)
        self.w1 = nn.Linear(d, d)
        self.w2 = nn.Linear(d, d)

    def forward(self, x, edge_index, batch):
        x = self.phi(self.norm_phi(x), edge_index, batch=batch)
        outer = self.norm_ffn_outer(x)
        ffn   = self.w2(F.gelu(self.w1(self.norm_ffn_inner(outer))))
        x     = x + ffn
        return x


class GBDecoder(nn.Module):
    def __init__(self, d: int, out_dim: int = 1):
        super().__init__()
        self.w1   = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.w2   = nn.Linear(d, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.norm(F.gelu(self.w1(x))))


class GPS(nn.Module):
    def __init__(
        self,
        in_dim: int,
        channels: int,
        pe_dim: int,
        num_layers: int,
        attn_type: str = "multihead",
        attn_kwargs: Optional[Dict[str, Any]] = None,
        heads: int = 4,
        out_dim: int = 1,
    ):
        super().__init__()
        attn_kwargs = attn_kwargs or {}
        self.node_lin = Linear(in_dim, channels - pe_dim)
        self.pe_norm  = BatchNorm1d(pe_dim)
        self.pe_lin   = Linear(pe_dim, pe_dim)
        self.layers = nn.ModuleList([
            GPSProcessorLayer(
                d=channels,
                attn_type=attn_type,
                attn_kwargs=attn_kwargs,
                heads=heads,
            )
            for _ in range(num_layers)
        ])
        self.decoder = GBDecoder(channels, out_dim=out_dim)

    def forward(self, data) -> torch.Tensor:
        x = data.x
        if x.dim() == 3:
            x = x.squeeze(1)
        pe = self.pe_lin(self.pe_norm(data.pe))
        x  = torch.cat([self.node_lin(x), pe], dim=-1)
        for layer in self.layers:
            x = layer(x, data.edge_index, data.batch)
        hg = global_mean_pool(x, data.batch)
        return self.decoder(hg).squeeze(-1)