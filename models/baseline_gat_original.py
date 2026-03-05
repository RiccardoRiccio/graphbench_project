import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GAToriginal(nn.Module):
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

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, head_dim, heads=heads, concat=True, dropout=dropout)
        )
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(hidden_dim, head_dim, heads=heads, concat=True, dropout=dropout)
            )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data):
        x = data.x.squeeze(1) if data.x.dim() == 3 else data.x
        edge_index = data.edge_index
        batch = getattr(data, "batch", None)

        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(conv(x, edge_index))

       
        graph_emb = global_mean_pool(x, batch)

        return self.mlp(graph_emb).squeeze(-1)