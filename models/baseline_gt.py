import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, to_dense_adj


class StructuralBiasedAttention(nn.Module):
    """
    Multi-head self-attention with additive graph structural bias:

        softmax(QK^T / sqrt(d_h) + B) V

    Bias design used here:
      - existing node-node edges: bias from a learned edge token projected to heads
      - non-edges: zero bias
      - CLS -> node and node -> CLS: separate learned per-head biases

    This is a practical approximation for graphs without explicit edge attributes.
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Learned placeholder "edge feature" -> per-head bias
        self.edge_token = nn.Parameter(torch.zeros(hidden_dim))
        self.edge_proj = nn.Linear(hidden_dim, num_heads, bias=False)

        # CLS-specific per-head biases
        self.cls_to_node_bias = nn.Parameter(torch.zeros(num_heads))
        self.node_to_cls_bias = nn.Parameter(torch.zeros(num_heads))

        nn.init.trunc_normal_(self.edge_token, std=0.02)

    def forward(
        self,
        x: torch.Tensor,                 # [B, L, D]
        adj: torch.Tensor,               # [B, L, L], includes CLS row/col at index 0
        key_padding_mask: torch.Tensor,  # [B, L], True means PAD
    ) -> torch.Tensor:
        B, L, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, L, H, Dh).transpose(1, 2)  # [B, H, L, Dh]
        k = self.k_proj(x).view(B, L, H, Dh).transpose(1, 2)  # [B, H, L, Dh]
        v = self.v_proj(x).view(B, L, H, Dh).transpose(1, 2)  # [B, H, L, Dh]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, L]

        # Start with zero bias everywhere (non-edges stay zero)
        bias = torch.zeros(B, H, L, L, device=x.device, dtype=x.dtype)

        # Existing node-node edges get learned projected bias
        # adj[:, 1:, 1:] corresponds to real node tokens (excluding CLS at index 0)
        node_adj = adj[:, 1:, 1:]  # [B, L-1, L-1]
        edge_bias = self.edge_proj(self.edge_token)  # [H]
        bias[:, :, 1:, 1:] = node_adj.unsqueeze(1) * edge_bias.view(1, H, 1, 1)

        # CLS-specific biases
        bias[:, :, 0, 1:] = self.cls_to_node_bias.view(1, H, 1)   # CLS query -> node keys
        bias[:, :, 1:, 0] = self.node_to_cls_bias.view(1, H, 1)   # node queries -> CLS key

        scores = scores + bias

        # Mask padded keys
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # [B,1,1,L]
                float("-inf"),
            )

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

        out = torch.matmul(attn, v)  # [B, H, L, Dh]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class GTPhi(nn.Module):
    """
    GT sublayer:
        phi(X, G) = MLP(Attention(XWQ, XWK, XWV, B))
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.attn = StructuralBiasedAttention(hidden_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, adj, key_padding_mask):
        x = self.attn(x, adj, key_padding_mask)
        x = self.mlp(x)
        return x


class GTLayer(nn.Module):
    """
    GraphBench-style processor layer requested by user:

        X = phi(LayerNorm(X), G)
        X = X + FNN(LayerNorm(X))
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.phi = GTPhi(hidden_dim, num_heads)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, adj, key_padding_mask):
        x = self.phi(self.norm1(x), adj, key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class Decoder(nn.Module):
    """
    Shared decoder:
        W2(LayerNorm(GELU(W1 x)))
    """

    def __init__(self, hidden_dim: int, out_dim: int = 1):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.w2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.w2(self.norm(F.gelu(self.w1(x))))


class GT(nn.Module):
    """
    GraphBench-style GT baseline for Electronic Circuits.

    Assumptions:
      - node-level tokenization
      - CLS token for graph-level prediction
      - no explicit edge attributes
      - structural attention bias from adjacency + learned edge token
      - output read only from CLS token

    Expected PyG data fields:
      - data.x         : [N, F] or [N, 1, F]
      - data.edge_index
      - data.batch
      - optional data.pe : [N, pe_dim]
    """

    def __init__(
        self,
        in_channels: int = 9,
        hidden_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 4,
        out_dim: int = 1,
        pe_dim: int = 0,
    ):
        super().__init__()

        self.node_encoder = nn.Linear(in_channels, hidden_dim)
        self.pe_proj = nn.Linear(pe_dim, hidden_dim) if pe_dim > 0 else None

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.layers = nn.ModuleList(
            [GTLayer(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.decoder = Decoder(hidden_dim, out_dim=out_dim)

    def forward(self, data):
        x = data.x
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)  # [N,1,F] -> [N,F]
        x = x.float()

        batch = data.batch
        edge_index = data.edge_index

        # Encode node features
        h = self.node_encoder(x)

        # Optional positional/structural encodings before first GT layer
        if self.pe_proj is not None and hasattr(data, "pe") and data.pe is not None:
            h = h + self.pe_proj(data.pe.float())

        # Dense node sequences
        h_dense, node_mask = to_dense_batch(h, batch)  # [B, L, D], [B, L]
        B, L, _ = h_dense.shape

        # Dense adjacency over real node tokens
        adj_nodes = to_dense_adj(edge_index, batch, max_num_nodes=L)  # [B, L, L]
        adj_nodes = (adj_nodes > 0).float()

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)         # [B, 1, D]
        h_seq = torch.cat([cls, h_dense], dim=1)       # [B, L+1, D]

        # Expand adjacency to include CLS row/column
        adj = torch.zeros(B, L + 1, L + 1, device=h.device, dtype=adj_nodes.dtype)
        adj[:, 1:, 1:] = adj_nodes

        # Padding mask: True means PAD
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=h.device)
        key_padding_mask = torch.cat([cls_pad, ~node_mask], dim=1)  # [B, L+1]

        pad_mask = key_padding_mask.clone()
        pad_mask[:, 0] = False  
        # GT processor
        for layer in self.layers:
            h_seq = layer(h_seq, adj, key_padding_mask)          # keep CLS
            h_seq = h_seq.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        # Graph-level prediction from CLS only
        cls_out = h_seq[:, 0, :]
        return self.decoder(cls_out).squeeze(-1)