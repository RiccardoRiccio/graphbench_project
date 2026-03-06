import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool


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
# φ for PNA: (PNAConv -> BatchNorm -> ReLU)
# We keep this exactly like your baseline, just packaged as a module.
# -------------------------
class PNAPhi(nn.Module):
    def __init__(self, conv: PNAConv, bn: BatchNorm):
        super().__init__()
        self.conv = conv
        self.bn = bn

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index)      # no edge_attr
        h = self.bn(h)
        h = F.relu(h)
        return h


# -------------------------
# One processor layer (Eq. 3 block)
# 1) h = φ(LayerNorm(x), G) ; x = x + h
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
        x = x + h
        x = x + self.fnn(x)
        return x


# -------------------------
# Full GraphBench-wrapped PNA model
# Encoder -> Processor(L layers) -> global_add_pool -> Decoder
# -------------------------
class PNAGraphBench(nn.Module):
    def __init__(self, deg: torch.Tensor):
        super().__init__()

        # ---- Encoder (keep original) ----
        self.encoder = nn.Linear(9, 75)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        # ---- Build processor layers ----
        self.processor_layers = nn.ModuleList()

        for _ in range(4):
            conv = PNAConv(
                in_channels=75,
                out_channels=75,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                edge_dim=None,
                towers=5,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
            bn = BatchNorm(75)

            phi = PNAPhi(conv, bn)
            self.processor_layers.append(GraphBenchProcessorLayer(hidden_dim=75, phi=phi))

        # ---- Decoder ----
        self.decoder = GraphBenchDecoder(hidden_dim=75, out_dim=1, bias=True)

    def forward(self, data):
        x = data.x.squeeze(1) if data.x.dim() == 3 else data.x
        x = x.float()
        edge_index = data.edge_index
        batch = data.batch

        # ---- Encode ----
        x = self.encoder(x)

        # ---- Processor ----
        for layer in self.processor_layers:
            x = layer(x, edge_index)

        # ---- Readout (keep baseline: add pool) ----
        hg = global_add_pool(x, batch)

        # ---- Decode ----
        out = self.decoder(hg)
        return out.squeeze(-1) if out.size(-1) == 1 else out