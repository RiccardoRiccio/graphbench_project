import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


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
# φ for the GraphBench MLP baseline:
# Node-wise MLP (ignores edges / connectivity)
# -------------------------
class NodeWiseMLP(nn.Module):
    """
    A node-wise MLP applied independently per node.
    This is the "connectivity-agnostic" φ baseline.
    """
    def __init__(self, dim: int, hidden_dim: int, num_layers: int = 4):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------
# One processor layer (Eq. 3 block)
# 1) h = φ(LayerNorm(x), G) ; x = x + h
# 2) x = x + FNN(x)
# For MLP baseline, φ ignores graph structure, so edge_index is unused.
# -------------------------
class GraphBenchProcessorLayer(nn.Module):
    def __init__(self, hidden_dim: int, phi: nn.Module):
        super().__init__()
        self.ln_phi = nn.LayerNorm(hidden_dim)
        self.phi = phi
        self.fnn = ProcessorFNN(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.phi(self.ln_phi(x))
        x = x + h
        x = x + self.fnn(x)
        return x


# -------------------------
# Full GraphBench-wrapped MLP model
# Encoder -> Processor -> mean pool -> Decoder
# -------------------------
class MLPGraphBench(nn.Module):
    """
    GraphBench connectivity-agnostic baseline in encoder-processor-decoder format.

    Encoder: Linear(input_dim -> d)
    Processor: L layers of Eq.3 blocks with φ = node-wise MLP (ignores edges)
    Readout: global_mean_pool
    Decoder: GraphBench decoder
    """
    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 384,
        num_layers: int = 4,   # match baseline depth
        out_dim: int = 1,
        decoder_bias: bool = True,
    ):
        super().__init__()

        # ---- Encoder ----
        self.encoder = nn.Linear(input_dim, hidden_dim)

        # ---- Processor ----
        # We keep φ as a node-wise MLP, and we stack num_layers processor blocks.
        self.processor = nn.ModuleList([
            GraphBenchProcessorLayer(hidden_dim, phi=NodeWiseMLP(hidden_dim, hidden_dim, num_layers=1))
            for _ in range(num_layers)
        ])
        # Note: using num_layers blocks, each with a 1-layer φ MLP,
        # yields a total of 4 linear+ReLU applications across the processor,
        # analogous to your original 4-layer node encoder.

        # ---- Decoder ----
        self.decoder = GraphBenchDecoder(hidden_dim, out_dim=out_dim, bias=decoder_bias)

    def forward(self, data):
        x = data.x.squeeze(1) if data.x.dim() == 3 else data.x
        x = x.float()
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        # ---- Encode ----
        h = self.encoder(x)

        # ---- Processor (ignores edges) ----
        for layer in self.processor:
            h = layer(h)

        # ---- Readout ----
        hg = global_mean_pool(h, batch)

        # ---- Decode ----
        out = self.decoder(hg)
        return out.squeeze(-1) if out.size(-1) == 1 else out