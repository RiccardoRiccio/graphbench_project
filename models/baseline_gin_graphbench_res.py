
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool


# -------------------------
# Processor FFN (Eq. 3, second line)
# FNN(x) := σ(LayerNorm(x) W1) W2, with σ = GeLU
# Implemented as: W2(GeLU(W1(LN(x))))
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
# Decoder(x) = W2(LayerNorm(GELU(W1 x)))
# -------------------------
class Decoder(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int = 1, bias: bool = True):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.norm = nn.LayerNorm(hidden_dim)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.norm(F.gelu(self.w1(x))))


# -------------------------
# φ for the GraphBench GIN baseline: GINE message passing + post 2-layer MLP
# - pre-LN is applied outside (in ProcessorLayer), per the paper text
# - no edge features for circuits: edge_dim=None and edge_attr=None
# -------------------------
class GINEPhi(nn.Module):
    def __init__(self, hidden_dim: int, train_eps: bool = False):
        super().__init__()

        # GINE internal NN (message passing update)
        gine_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gine = GINEConv(nn=gine_nn, train_eps=train_eps, edge_dim=hidden_dim)

        self.post_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr=None) -> torch.Tensor:
     
        h = self.gine(x, edge_index, edge_attr)
        h = self.post_mlp(h)
        return h


# -------------------------
# One processor layer (Eq. 3 block)
# 1) h = φ(LayerNorm(x), G) ; x = x + h
# 2) x = x + FNN(x)  (FNN internally applies LayerNorm)
# -------------------------
class GraphBenchProcessorLayer(nn.Module):
    def __init__(self, hidden_dim: int, phi: nn.Module):
        super().__init__()
        self.ln_phi = nn.LayerNorm(hidden_dim)
        self.phi = phi
        self.fnn = ProcessorFNN(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr=None) -> torch.Tensor:
        h = self.phi(self.ln_phi(x), edge_index, edge_attr)
        x = x + h
        x = x + self.fnn(x)
        return x


# -------------------------
# Full GraphBench GIN baseline for graph-level regression (electronic circuits)
# Encoder -> Processor(L layers) -> mean pool -> Decoder
# -------------------------
class GINGraphBench(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 384,
        num_layers: int = 4,  
        out_dim: int = 1,
        train_eps: bool = False,
        decoder_bias: bool = True,
    ):
        super().__init__()
        self.encoder = nn.Linear(in_channels, hidden_dim)
        self.missing_edge_attr = nn.Parameter(torch.zeros(hidden_dim))
      

        self.processor = nn.ModuleList([
            GraphBenchProcessorLayer(hidden_dim, phi=GINEPhi(hidden_dim, train_eps=train_eps))
            for _ in range(num_layers)
        ])

        self.decoder = Decoder(hidden_dim, out_dim=out_dim, bias=decoder_bias)

    def forward(self, data) -> torch.Tensor:
        x = data.x
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)  

        x = x.float()

        edge_index = data.edge_index

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        # Encode
        h = self.encoder(x)

        # No edge features for circuits
        num_edges = data.edge_index.size(1)
        edge_attr = self.missing_edge_attr.unsqueeze(0).expand(num_edges, -1)

        # Processor stack
        for layer in self.processor:
            h = layer(h, edge_index, edge_attr=edge_attr)

        # Mean pooling for graph-level tasks
        hg = global_mean_pool(h, batch)

        # Decode
        out = self.decoder(hg).squeeze(-1)  # [B]
        return out