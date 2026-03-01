import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

class GINEModel(nn.Module):
    def __init__(self, in_channels=9, hidden_channels=128, out_channels=1, num_layers=4):
        super(GINEModel, self).__init__()
        
        # Node Encoder: Projects 9 features to hidden dimension
        self.node_encoder = nn.Linear(in_channels, hidden_channels)
        
        # GINE Layers (Processor)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINEConv(mlp))

        # Decoder: Regression head for Efficiency/Voltage
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Standardize input shape [Nodes, 9]
        if x.dim() == 3: x = x.squeeze(1)
        x = self.node_encoder(x)
        
        # Message Passing
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        
        # Global Pooling to get a single vector for the whole circuit
        x = global_mean_pool(x, batch)
        return self.decoder(x)