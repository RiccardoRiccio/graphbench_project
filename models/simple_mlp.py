import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, data):
        # x: [num_nodes, 1, 9] -> [num_nodes, 9]
        x = data.x.squeeze(1)

        # handle batching (single-graph case won't have data.batch)
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        # pooled graph embedding: [num_graphs, 9]
        x_graph = global_mean_pool(x, batch)

        # output: [num_graphs, 1]
        return self.net(x_graph)