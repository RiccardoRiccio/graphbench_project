import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNLayer(nn.Module):
    """
    Literal-paper processor block for the GraphBench GCN baseline.

    Printed processor equations:
        X <- phi(LayerNorm(X), G)
        X <- X + MLP(LayerNorm(X))

    Printed FFN definition:
        FNN(x) := GELU(LayerNorm(x) W1) W2

    If taken literally and identifying MLP = FNN, then:
        X <- X + FNN(LayerNorm(X))
           = X + W2(GELU(W1(LN(LN(X)))))

    So this implementation intentionally uses:
      - NO residual around the GCNConv block
      - residual ONLY around the FFN block
      - TWO LayerNorms on the FFN branch
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        # First line: X <- phi(LN(X), G)
        self.norm_conv = nn.LayerNorm(hidden_dim)
        self.conv = GCNConv(hidden_dim, hidden_dim)

        # Second line: X <- X + FNN(LN(X))
        self.norm_ffn_outer = nn.LayerNorm(hidden_dim)
        self.norm_ffn_inner = nn.LayerNorm(hidden_dim)
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # First processor line: no residual
        x = self.conv(self.norm_conv(x), edge_index)

        # If you want to reflect Table 27 activation=RELU, uncomment this:
        x = F.relu(x)

        # Second processor line: residual around FFN only
        outer = self.norm_ffn_outer(x)
        ffn = self.w2(F.gelu(self.w1(self.norm_ffn_inner(outer))))
        x = x + ffn

        return x


class Decoder(nn.Module):
    """
    Literal paper decoder:
        W2(LayerNorm(GELU(W1 x)))

    Order:
        Linear -> GELU -> LayerNorm -> Linear
    """

    def __init__(self, hidden_dim: int, out_dim: int = 1):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.w2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.norm(F.gelu(self.w1(x))))


class GCN(nn.Module):
    """
    GCN baseline for GraphBench Electronic Circuits.

    Assumptions from your dataset sanity check:
      - x shape is [num_nodes, 1, 9], so we squeeze to [num_nodes, 9]
      - node input dim = 9
      - graph-level scalar regression target

    Architecture:
      - Encoder: Linear(9 -> hidden_dim)
      - Processor: num_layers x GCNLayer
      - Readout: global_mean_pool
      - Decoder: 2-layer MLP head -> scalar output
    """

    def __init__(
        self,
        in_channels: int = 9,
        hidden_dim: int = 384,
        num_layers: int = 4,
        out_dim: int = 1,
    ):
        super().__init__()

        self.encoder = nn.Linear(in_channels, hidden_dim)
        self.layers = nn.ModuleList([GCNLayer(hidden_dim) for _ in range(num_layers)])
        self.decoder = Decoder(hidden_dim, out_dim=out_dim)

    def forward(self, data) -> torch.Tensor:
        x = data.x
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)  # [N, 1, 9] -> [N, 9]
        x = x.float()

        edge_index = data.edge_index
        batch = data.batch

        h = self.encoder(x)

        for layer in self.layers:
            h = layer(h, edge_index)

        hg = global_mean_pool(h, batch)
        out = self.decoder(hg).squeeze(-1)

        return out