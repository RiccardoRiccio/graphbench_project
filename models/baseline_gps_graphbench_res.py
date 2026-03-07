import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional

from torch.nn import Linear, ModuleList, Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import GPSConv, GINConv, global_add_pool


# -------------------------
# Processor FFN (Eq. 3, second line)
# FNN(x) = W2( GELU( W1( LN(x) ) ) )
# -------------------------
class ProcessorFNN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.w1(h)
        h = F.gelu(h)
        h = self.w2(h)
        return h


# -------------------------
# GraphBench task decoder
# Decoder(x) = W2( LayerNorm( GELU( W1 x ) ) )
# -------------------------
class GraphBenchDecoder(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int = 1, bias: bool = True):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.norm = nn.LayerNorm(hidden_dim)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.norm(F.gelu(self.w1(x))))


# -------------------------
# One processor layer (Eq. 3 block) adapted for GPSConv
# 1) h = φ(LN(x), G) ; x = x + h
# 2) x = x + FNN(x)
# -------------------------
class GPSProcessorLayer(nn.Module):
    def __init__(self, hidden_dim: int, phi: GPSConv):
        super().__init__()
        self.ln_phi = nn.LayerNorm(hidden_dim)
        self.phi = phi
        self.fnn = ProcessorFNN(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.phi(self.ln_phi(x), edge_index, batch=batch)
        x = x + h
        x = x + self.fnn(x)
        return x


# -------------------------
# Full GraphBench-wrapped GPS model
# Encoder -> Processor(L layers) -> global_add_pool -> Decoder
# -------------------------
class GPSGraphBench(nn.Module):
    def __init__(
        self,
        in_dim: int,
        channels: int,
        pe_dim: int,
        num_layers: int,
        out_dim: int = 1,
        attn_type: str = "multihead",
        attn_kwargs: Optional[Dict[str, Any]] = None,
        decoder_bias: bool = True,
    ):
        super().__init__()
        attn_kwargs = attn_kwargs or {}

        # ---- Encoder (keep identical to baseline) ----
        self.node_lin = Linear(in_dim, channels - pe_dim)
        self.pe_norm = BatchNorm1d(pe_dim)
        self.pe_lin = Linear(pe_dim, pe_dim)

        # ---- Processor: stack Eq.3 blocks where φ = GPSConv ----
        self.processor_layers = nn.ModuleList()
        for _ in range(num_layers):
            mlp = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            local_conv = GINConv(mlp)
            phi = GPSConv(
                channels,
                local_conv,
                heads=4,
                attn_type=attn_type,
                attn_kwargs=attn_kwargs,
            )
            self.processor_layers.append(GPSProcessorLayer(channels, phi))

        # ---- Decoder (GraphBench) ----
        self.decoder = GraphBenchDecoder(channels, out_dim=out_dim, bias=decoder_bias)

    def forward(self, data) -> torch.Tensor:
        x = data.x.squeeze(1) if data.x.dim() == 3 else data.x
        x = x.float()

        # PE required for GPS
        pe = data.pe
        pe = self.pe_lin(self.pe_norm(pe))

        # ---- Encode ----
        x = torch.cat([self.node_lin(x), pe], dim=-1)  # [N, channels]

        # ---- Processor ----
        for layer in self.processor_layers:
            x = layer(x, data.edge_index, data.batch)

        # ---- Readout (keep baseline: add pool) ----
        hg = global_add_pool(x, data.batch)

        # ---- Decode ----
        out = self.decoder(hg)  # [B, out_dim]
        return out.squeeze(-1)