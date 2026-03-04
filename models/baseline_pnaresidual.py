import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn import PNAConv, global_mean_pool, global_add_pool


class PNAProcessorLayer(nn.Module):
    def __init__(self, d: int, deg: torch.Tensor):
        super().__init__()

        self.norm_phi = nn.LayerNorm(d)
        self.phi = PNAConv(
            in_channels=d,
            out_channels=d,
            aggregators=['mean', 'min', 'max', 'std'],
            scalers=['identity', 'amplification', 'attenuation'],
            deg=deg,
            towers=4,
            pre_layers=1,
            post_layers=1,
            edge_dim=None,
            divide_input=False,
        )

        self.norm_ffn_outer = nn.LayerNorm(d)
        self.norm_ffn_inner = nn.LayerNorm(d)
        self.w1 = nn.Linear(d, d)
        self.w2 = nn.Linear(d, d)

    def forward(self, x, edge_index, batch):
        # Line 1: X = phi(LN(X), G) — no residual
        h = self.phi(self.norm_phi(x), edge_index)
        x = x + h

        # Line 2: X = X + FNN(LN(X))
        outer = self.norm_ffn_outer(x)
        ffn   = self.w2(F.gelu(self.w1(self.norm_ffn_inner(outer))))
        x     = x + ffn
        return x


class GBDecoder(nn.Module):
    """GraphBench decoder: W2(LayerNorm(GELU(W1 x)))"""
    def __init__(self, d: int, out_dim: int = 1):
        super().__init__()
        self.w1   = Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.w2   = Linear(d, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.norm(F.gelu(self.w1(x))))


class PNAresidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        deg: torch.Tensor,
        out_dim: int = 1,
    ):
        super().__init__()

        # Encoder
        self.encoder = Linear(in_channels, hidden_dim)

        # Processor
        self.layers = ModuleList([
            PNAProcessorLayer(d=hidden_dim, deg=deg)
            for _ in range(num_layers)
        ])

        # Decoder
        self.decoder = GBDecoder(hidden_dim, out_dim=out_dim)

    def forward(self, data) -> torch.Tensor:
        x          = data.x
        edge_index = data.edge_index
        batch      = data.batch

        if x.dim() == 3:
            x = x.squeeze(1)              # [N,1,9] -> [N,9]

        # Encoder
        x = self.encoder(x)               # [N, hidden_dim]

        # Processor
        for layer in self.layers:
            x = layer(x, edge_index, batch)

        # Readout
        hg = global_add_pool(x, batch)   # [B, hidden_dim]

        # Decoder
        return self.decoder(hg).squeeze(-1)  # [B]s