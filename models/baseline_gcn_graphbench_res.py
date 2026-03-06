import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


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
# One processor layer (Eq. 3 block)
# 1) h = φ(LN(x), G) ; x = x + h
#    (apply ReLU after φ for MPNN baselines)
# 2) x = x + FNN(x)
# -------------------------
class GraphBenchProcessorLayer(nn.Module):
    def __init__(self, hidden_dim: int, phi: nn.Module):
        super().__init__()
        self.ln_phi = nn.LayerNorm(hidden_dim)
        self.phi = phi
        self.fnn = ProcessorFNN(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.phi(self.ln_phi(x), edge_index)
        h = F.relu(h)  # GraphBench uses ReLU for MPNN baselines (GCN/GAT/GIN)
        x = x + h

        x = x + self.fnn(x)
        return x


# -------------------------
# GraphBench task decoder (Appendix)
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
# Full GraphBench-wrapped GCN model
# Encoder -> Processor -> Decoder
# -------------------------
class GCNGraphBench(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 384,
        num_layers: int = 4,
        out_dim: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = dropout

        # ---- Encoder ----
        self.encoder = nn.Linear(in_channels, hidden_dim)

        # ---- Processor (stack Eq.3 blocks) ----
        self.processor_layers = nn.ModuleList()
        for _ in range(num_layers):
            phi = GCNConv(hidden_dim, hidden_dim)
            self.processor_layers.append(GraphBenchProcessorLayer(hidden_dim, phi))

        # ---- Decoder ----
        self.decoder = GraphBenchDecoder(hidden_dim, out_dim=out_dim, bias=True)

    def forward(self, data) -> torch.Tensor:
        x = data.x.squeeze(1) if data.x.dim() == 3 else data.x
        x = x.float()
        edge_index = data.edge_index
        batch = data.batch

        # ---- Encoder ----
        x = self.encoder(x)

        # ---- Processor ----
        for layer in self.processor_layers:
            x = F.dropout(x, p=self.dropout, training=self.training)  # no-op if dropout=0
            x = layer(x, edge_index)

        # ---- Readout ----
        hg = global_mean_pool(x, batch)

        # ---- Decoder ----
        out = self.decoder(hg)
        return out.squeeze(-1) if out.size(-1) == 1 else out