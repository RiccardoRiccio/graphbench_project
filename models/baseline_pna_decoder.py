import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool
from torch_geometric.utils import degree


class PNAdecoder(torch.nn.Module):
    def __init__(self, deg: torch.Tensor):
        super().__init__()

        self.node_emb = Linear(9, 75) 
       

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = PNAConv(in_channels=75, out_channels=75,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=None, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(75))

        self.decoder_w1   = Linear(75, 75)
        self.decoder_norm = nn.LayerNorm(75)
        self.decoder_w2   = Linear(75, 1)

    def forward(self, data):
        x = data.x
        if x.dim() == 3:
            x = x.squeeze(1)
        edge_index = data.edge_index
        batch = data.batch
        x = self.node_emb(x)
        

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))

        x = global_add_pool(x, batch)
        return self.decoder_w2(self.decoder_norm(F.gelu(self.decoder_w1(x)))).squeeze(-1)
