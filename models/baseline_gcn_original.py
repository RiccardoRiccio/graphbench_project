import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNoriginal(nn.Module):
    """
    Original-style GCN adapted for GraphBench Electronic Circuits:
    - graph-level regression
    - global pooling
    """

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

        self.convs = nn.ModuleList()

        # first layer
        self.convs.append(GCNConv(in_channels, hidden_dim))

        # hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # regression head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data) -> torch.Tensor:
        x = data.x
        if x.dim() == 3:
            x = x.squeeze(1)  # [N,1,9] -> [N,9]

        edge_index = data.edge_index
        batch = getattr(data, "batch", None)

        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.relu(x)

       
        graph_emb = global_mean_pool(x, batch)

        return self.mlp(graph_emb).squeeze(-1)