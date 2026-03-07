import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from torch.nn import Linear, ModuleList, Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import GPSConv, GINConv, global_mean_pool, global_add_pool


class GPSoriginal(nn.Module):
    def __init__(
        self,
        in_dim: int,
        channels: int,
        pe_dim: int,
        num_layers: int,
        attn_type: str = "multihead",
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        attn_kwargs = attn_kwargs or {}

        # node features are float one-hot vectors (size in_dim=9)
        self.node_lin = Linear(in_dim, channels - pe_dim)

        # RWSE PE is size pe_dim (e.g., 16)
        self.pe_norm = BatchNorm1d(pe_dim)
        self.pe_lin = Linear(pe_dim, pe_dim)

        self.convs = ModuleList()
        for _ in range(num_layers):
            mlp = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            local_conv = GINConv(mlp)
            self.convs.append(
                GPSConv(
                    channels,
                    local_conv,
                    heads=4,
                    attn_type=attn_type,
                    attn_kwargs=attn_kwargs,
                )
            )

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x = data.x
        if x.dim() == 3:
            x = x.squeeze(1)  # [N,1,9] -> [N,9]

        pe = data.pe
        pe = self.pe_lin(self.pe_norm(pe))

        x = torch.cat([self.node_lin(x), pe], dim=-1)  # [N, channels]

        for conv in self.convs:
            x = conv(x, data.edge_index, batch=data.batch)

        hg = global_add_pool(x, data.batch)
        return self.mlp(hg).squeeze(-1)