import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# 1) Shared Processor FFN (Eq. 3 second line)
#    FNN(x) := σ(LayerNorm(x) W1) W2   with σ = GeLU
# -------------------------
class ProcessorFNN(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.w1(h)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.w2(h)
        return h
    
class ProcessorLayer(nn.Module):
    """
    Implements one layer of Eq. (3):

      h = φ(LN(x), G)
      x = x + h
      x = x + FNN(x)   # where FNN internally does LN again per paper

    φ is injected as a callable module: phi(x, edge_index, edge_attr?) -> node features
    """
    def __init__(self, dim: int, phi: nn.Module, dropout: float = 0.0):
        super().__init__()
        self.ln_phi = nn.LayerNorm(dim)
        self.phi = phi
        self.dropout = dropout
        self.fnn = FNN(dim, dropout=dropout)

    def forward(self, x, edge_index, edge_attr=None):
        # 1) graph op with pre-LN + residual
        h = self.ln_phi(x)
        # call φ with or without edge_attr depending on what it supports
        if edge_attr is None:
            h = self.phi(h, edge_index)
        else:
            h = self.phi(h, edge_index, edge_attr)

        h = F.dropout(h, p=self.dropout, training=self.training)
        x = x + h

        # 2) residual feed-forward network
        x = x + self.fnn(x)

        return x

class Decoder(nn.Module):
    """
    GraphBench decoder:
      W2(LayerNorm(GELU(W1 x)))
    """
    def __init__(self, hidden_dim: int, out_dim: int = 1, bias: bool = True):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.norm = nn.LayerNorm(hidden_dim)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.w2(self.norm(F.gelu(self.w1(x))))