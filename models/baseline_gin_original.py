
# import torch
# import torch.nn as nn
# from torch_geometric.nn import GINEConv, global_mean_pool


# class GINoriginal(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         hidden_dim: int = 384,
#         num_layers: int = 4,
#         out_dim: int = 1,   # set to 2 if predicting both circuit targets jointly
#         train_eps: bool = False,
#     ):
#         super().__init__()

#         self.encoder = nn.Linear(in_channels, hidden_dim)

#         self.layers = nn.ModuleList()
#         for _ in range(num_layers):
#             gine_mlp = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, hidden_dim),
#             )

#             self.layers.append(nn.ModuleDict({
#                 "ln": nn.LayerNorm(hidden_dim),
#                 "conv": GINEConv(
#                     nn=gine_mlp,
#                     train_eps=train_eps,
#                     edge_dim=None,  # circuits: no edge features
#                 ),
#                 "post_mlp": nn.Sequential(
#                     nn.Linear(hidden_dim, hidden_dim),
#                     nn.ReLU(),
#                     nn.Linear(hidden_dim, hidden_dim),
#                 ),
#             }))

#         self.head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, out_dim),
#         )

#     def forward(self, data):
#         x = data.x
#         if x.dim() == 3 and x.size(1) == 1:
#             x = x.squeeze(1)

#         x = x.float()
#         edge_index = data.edge_index
#         edge_attr = None

#         batch = getattr(data, "batch", None)
#         if batch is None:
#             batch = x.new_zeros(x.size(0), dtype=torch.long)

#         h = self.encoder(x)

#         for layer in self.layers:
#             z = layer["ln"](h)
#             z = layer["conv"](z, edge_index, edge_attr)
#             z = layer["post_mlp"](z)
#             h = h + z   # residual

#         hg = global_mean_pool(h, batch)
#         out = self.head(hg)

#         if out.size(-1) == 1:
#             return out.squeeze(-1)
#         return out
import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool


class GINoriginal(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 384,
        num_layers: int = 4,
        out_dim: int = 1,
        train_eps: bool = False,
    ):
        super().__init__()

        self.encoder = nn.Linear(in_channels, hidden_dim)

        # learnable edge feature used when dataset has no edge_attr
        self.missing_edge_attr = nn.Parameter(torch.zeros(hidden_dim))

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            gine_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.layers.append(nn.ModuleDict({
                "ln": nn.LayerNorm(hidden_dim),
                "conv": GINEConv(
                    nn=gine_mlp,
                    train_eps=train_eps,
                    edge_dim=hidden_dim,  # edge_attr must be [E, hidden_dim]
                ),
                "post_mlp": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
            }))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data):
        x = data.x
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        x = x.float()

        edge_index = data.edge_index
        batch = data.batch

        h = self.encoder(x)

        # build placeholder edge_attr
        num_edges = edge_index.size(1)
        edge_attr = self.missing_edge_attr.unsqueeze(0).expand(num_edges, -1)

        for layer in self.layers:
            z = layer["ln"](h)
            z = layer["conv"](z, edge_index, edge_attr)
            z = layer["post_mlp"](z)
            h = h + z

        hg = global_mean_pool(h, batch)
        out = self.head(hg)

        return out.squeeze(-1) if out.size(-1) == 1 else out