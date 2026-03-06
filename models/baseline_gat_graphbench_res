import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


# -------------------------
# Processor FFN (Eq. 3, second line)
# FNN(x) = GELU( LN(x) W1 ) W2
# Implemented as: W2( GELU( W1( LN(x) ) ) )
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
# 2) x = x + FNN(x)
# -------------------------
class GraphBenchProcessorLayer(nn.Module):
    def __init__(self, hidden_dim: int, phi: nn.Module):
        super().__init__()
        self.ln_phi = nn.LayerNorm(hidden_dim)
        self.phi = phi
        self.fnn = ProcessorFNN(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Eq.3 line 1 (with residual add made explicit)
        h = self.phi(self.ln_phi(x), edge_index)
        h = F.relu(h)  # GraphBench uses ReLU after MPNN operator φ (e.g., GAT/GCN/GIN)
        x = x + h

        # Eq.3 line 2
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
# Full GraphBench-wrapped GAT model
# Encoder -> Processor -> Decoder
# -------------------------
class GATGraphBench(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 384,
        num_layers: int = 4,
        heads: int = 4,
        out_dim: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert hidden_dim % heads == 0
        head_dim = hidden_dim // heads
        self.dropout = dropout

        # ---- Encoder (task-specific) ----
        # Map raw node features -> common dim d=hidden_dim
        self.encoder = nn.Linear(in_channels, hidden_dim)

        # ---- Processor (stack Eq.3 blocks) ----
        self.processor_layers = nn.ModuleList()
        for _ in range(num_layers):
            phi = GATConv(hidden_dim, head_dim, heads=heads, concat=True, dropout=dropout)
            self.processor_layers.append(GraphBenchProcessorLayer(hidden_dim, phi))

        # ---- Decoder ----
        self.decoder = GraphBenchDecoder(hidden_dim, out_dim=out_dim, bias=True)

    def forward(self, data):
        # node features
        x = data.x.squeeze(1) if data.x.dim() == 3 else data.x
        x = x.float()
        edge_index = data.edge_index
        batch = data.batch

        # ---- Encoder ----
        x = self.encoder(x)

        # ---- Processor ----
        for layer in self.processor_layers:
            # Dropout is optional; if dropout=0 it does nothing.
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index)

        # ---- Graph readout (mean pooling for graph-level tasks) ----
        hg = global_mean_pool(x, batch)

        # ---- Decoder ----
        out = self.decoder(hg)  # [B, out_dim]
        return out.squeeze(-1) if out.size(-1) == 1 else out