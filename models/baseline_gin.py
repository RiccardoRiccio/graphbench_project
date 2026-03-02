import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

class GINLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.norm_conv = nn.LayerNorm(hidden_dim)

        gine_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv = GINEConv(
            nn=gine_mlp,
            train_eps=True,
            edge_dim=hidden_dim,
        )

        self.post_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        x_msg = self.conv(self.norm_conv(x), edge_index, edge_attr)
        x_out = self.post_mlp(x_msg)
        x = x + x_out
        return x


class Decoder(nn.Module):
    """
    Shared decoder from the paper:
        W2(LayerNorm(GELU(W1 x)))
    """

    def __init__(self, hidden_dim: int, out_dim: int = 1):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.w2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.norm(F.gelu(self.w1(x))))


class GIN(nn.Module):
    """
    GraphBench electronic-circuits GIN baseline (implemented as GINE-based processor).

    Assumptions from your dataset:
      - x comes as [N, 1, 9] or [N, 9]
      - graph-level scalar target
      - no explicit edge features in the dataset

    Paper-aligned choices:
      - hidden_dim = 384
      - num_layers = 4
      - use learnable placeholder edge features
      - graph-level mean pooling
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

        # Missing edge features -> learned vector of dimension d
        self.edge_token = nn.Parameter(torch.zeros(hidden_dim))

        self.layers = nn.ModuleList(
            [GINLayer(hidden_dim) for _ in range(num_layers)]
        )
        self.decoder = Decoder(hidden_dim, out_dim=out_dim)

    def forward(self, data) -> torch.Tensor:
        x = data.x
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)  # [N, 1, 9] -> [N, 9]
        x = x.float()

        edge_index = data.edge_index
        batch = data.batch

        # Encode node features to hidden dimension
        h = self.encoder(x)

        # Missing edge features -> learned placeholder per edge
        num_edges = edge_index.size(1)
        edge_attr = self.edge_token.unsqueeze(0).expand(num_edges, -1)

        # Processor stack
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)

        # Graph-level readout + decoder
        hg = global_mean_pool(h, batch)
        out = self.decoder(hg)squeeze(-1)
        return out