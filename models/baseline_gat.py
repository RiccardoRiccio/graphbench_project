
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GATConv, global_mean_pool


# class GATLayer(nn.Module):
#     def __init__(self, hidden_dim: int):
#         super().__init__()

#         # Line 1: X = phi(LayerNorm(X), G)
#         self.norm1 = nn.LayerNorm(hidden_dim)
#         self.conv = GATConv(
#             in_channels=hidden_dim,
#             out_channels=hidden_dim // 4,
#             heads=4,
#             concat=True,
#             dropout=0.0,
#         )

#         # Line 2: X = X + MLP(LayerNorm(X))
#         # FNN(x) := sigma(LayerNorm(x) W1) W2
#         self.norm2 = nn.LayerNorm(hidden_dim)  # outer LN
#         self.norm3 = nn.LayerNorm(hidden_dim)  # inner LN inside FNN
#         self.w1 = nn.Linear(hidden_dim, hidden_dim)
#         self.w2 = nn.Linear(hidden_dim, hidden_dim)

#     def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
#         # Eq. 3, first line
#         x = self.conv(self.norm1(x), edge_index)

#         # Eq. 3, second line with literal expansion
#         ffn_in = self.norm2(x)                           # outer LN
#         ffn_out = self.w2(F.gelu(self.w1(self.norm3(ffn_in))))  # inner LN BEFORE W1
#         x = x + ffn_out

#         return x


# class Decoder(nn.Module):
#     """
#     Appendix B Decoder:  W2( LayerNorm( GELU( W1·x ) ) )
#     Order: Linear → GELU → LayerNorm → Linear
#     """
#     def __init__(self, hidden_dim: int):
#         super().__init__()
#         self.w1   = nn.Linear(hidden_dim, hidden_dim)
#         self.norm = nn.LayerNorm(hidden_dim)
#         self.w2   = nn.Linear(hidden_dim, 1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.w2(self.norm(F.gelu(self.w1(x))))


# class GATElectronicCircuits(nn.Module):
#     """
#     GAT baseline for GraphBench Electronic Circuits (Table 27).

#     in_channels = 9   : one-hot node type, x shape [num_nodes, 1, 9] → squeezed to [num_nodes, 9]
#     hidden_dim  = 384
#     num_layers  = 4
#     out         = scalar (one target per run: eff or vout)
#     """
#     def __init__(
#         self,
#         in_channels: int = 9,
#         hidden_dim:  int = 384,
#         num_layers:  int = 4,
#     ):
#         super().__init__()
#         self.encoder = nn.Linear(in_channels, hidden_dim)
#         self.layers  = nn.ModuleList([GATLayer(hidden_dim) for _ in range(num_layers)])
#         self.decoder = Decoder(hidden_dim)

#     def forward(self, data) -> torch.Tensor:
#         x          = data.x.squeeze(1)   # [N, 1, 9] → [N, 9]
#         edge_index = data.edge_index
#         batch      = data.batch

#         h = self.encoder(x)
#         for layer in self.layers:
#             h = layer(h, edge_index)
#         return self.decoder(global_mean_pool(h, batch))  # [num_graphs, 1]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATLayer(nn.Module):
    """
    Literal implementation of the printed processor equations:

        X <- phi(LayerNorm(X), G)
        X <- X + MLP(LayerNorm(X))

    with
        FNN(x) := GELU(LayerNorm(x) W1) W2

    If we take MLP = FNN literally, then:
        X <- X + FNN(LayerNorm(X))
           = X + W2(GELU(W1(LN(LN(X)))))

    So this block intentionally applies TWO LayerNorms on the FFN branch.
    """
    def __init__(self, hidden_dim: int, heads: int = 4):
        super().__init__()

        # First line: X <- phi(LN(X), G)
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.conv = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            concat=True,
            dropout=0.0,
        )

        # Second line: X <- X + FNN(LN(X))
        # Outer LN from "MLP(LayerNorm(X))"
        self.norm_ffn_outer = nn.LayerNorm(hidden_dim)

        # Inner LN from FNN(x) := GELU(LayerNorm(x) W1) W2
        self.norm_ffn_inner = nn.LayerNorm(hidden_dim)

        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # X <- phi(LN(X), G)
        x = self.conv(self.norm_attn(x), edge_index)

        # Optional:
        # Table 27 lists RELU for the GAT architecture in Electronic Circuits.
        # The paper does not clearly say where this ReLU is inserted.
        # If you want the STRICT printed equations only, leave it out.
        # If you want to reflect the table too, uncomment:
        x = F.relu(x)

        # X <- X + FNN(LN(X))
        # literal expansion:
        # outer = LN(x)
        # FNN(outer) = W2(GELU(W1(LN(outer))))
        outer = self.norm_ffn_outer(x)
        ffn = self.w2(F.gelu(self.w1(self.norm_ffn_inner(outer))))
        x = x + ffn

        return x


class Decoder(nn.Module):
    """
    Literal decoder from the paper:

        W2(LayerNorm(GELU(W1 x)))
    """
    def __init__(self, hidden_dim: int, out_dim: int = 1):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.w2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.norm(F.gelu(self.w1(x))))


class GAT(nn.Module):
    """
    Literal paper-style GAT baseline for GraphBench Electronic Circuits.

    - Encoder: linear projection to hidden dim
    - Processor: 4 GAT layers, hidden dim 384, 4 heads
    - Readout: mean pooling (reasonable choice for graph-level regression;
      the snippet does not explicitly confirm mean pooling for GAT)
    - Decoder: W2(LN(GELU(W1 x)))
    """
    def __init__(
        self,
        in_channels: int = 9,
        hidden_dim: int = 384,
        num_layers: int = 4,
        heads: int = 4,
        out_dim: int = 1,
    ):
        super().__init__()
        self.encoder = nn.Linear(in_channels, hidden_dim)
        self.layers = nn.ModuleList(
            [GATLayer(hidden_dim, heads=heads) for _ in range(num_layers)]
        )
        self.decoder = Decoder(hidden_dim, out_dim=out_dim)

    def forward(self, data) -> torch.Tensor:
        x = data.x.squeeze(1)       # [N, 1, 9] -> [N, 9], if needed
        edge_index = data.edge_index
        batch = data.batch

        h = self.encoder(x)
        for layer in self.layers:
            h = layer(h, edge_index)

        hg = global_mean_pool(h, batch)
        return self.decoder(hg)