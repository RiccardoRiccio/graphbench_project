import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

# class BaselineMLP(nn.Module):
#     def __init__(self, input_dim=9, hidden_dim=384, out_channels=1):
#         super(BaselineMLP, self).__init__()
        
#         # 4-layer MLP as per paper standards for Electronic Circuits
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, out_channels)
#         )

#     def forward(self, data):
#         # 1. Extract node features: [TotalNodesInBatch, 9]
#         # Squeeze handles the [N, 1, 9] shape found in your sanity check
#         x = data.x.squeeze(1) if data.x.dim() == 3 else data.x
        
#         # 2. Global Mean Pooling: [BatchSize, 9]
#         # This averages the 11-17 nodes per circuit into a single vector
#         # per circuit, independent of batch size.
#         x_graph = global_mean_pool(x, data.batch)
        
#         # 3. Final Prediction
#         return self.net(x_graph)
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

class BaselineMLP(nn.Module):
    """
    Connectivity-agnostic baseline. 
    4 layers, hidden=384, ReLU — following GIN/GCN/GAT column of Table 27.
    Encodes each node independently, then mean-pools to graph level.
    """
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