import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

class BaselineMLP(nn.Module):
   
    def __init__(self, input_dim=9, hidden_dim=384, out_channels=1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, out_channels)

    def forward(self, data):
        x = data.x.squeeze(1) if data.x.dim() == 3 else data.x  # [N, 9]
        x = self.encoder(x)                 # [N, 384] — encode each node
        x = global_mean_pool(x, data.batch) # [B, 384] — pool to graph level
        return self.head(x).squeeze(-1)     # [B]