import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


class DenseBiasedAttention(nn.Module):
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

        self.max_nodes = 128  # safe upper bound for your circuits; set higher if needed
        self.attn_bias = nn.Parameter(torch.zeros(num_heads, self.max_nodes, self.max_nodes))
        nn.init.trunc_normal_(self.attn_bias, std=0.02)



    def forward(self, x, key_padding_mask):
        B, L, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, L, H, Dh).transpose(1, 2)  # [B, H, L, Dh]
        k = self.k_proj(x).view(B, L, H, Dh).transpose(1, 2)  # [B, H, L, Dh]
        v = self.v_proj(x).view(B, L, H, Dh).transpose(1, 2)  # [B, H, L, Dh]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, L]

        bias = self.attn_bias[:, :L, :L]          # [H, L, L]
        scores = scores + bias.unsqueeze(0)       # [B, H, L, L]

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
        self.attn = DenseBiasedAttention(hidden_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, key_padding_mask):
        x = self.attn(x, key_padding_mask)
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

    def forward(self, x, key_padding_mask):
        x = x + self.phi(self.norm1(x), key_padding_mask)
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


class GTPooling(nn.Module):
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
    

        # Encode node features
        h = self.node_encoder(x)

        # Optional positional/structural encodings before first GT layer
        if self.pe_proj is not None and hasattr(data, "pe") and data.pe is not None:
            h = h + self.pe_proj(data.pe.float())

        # Dense node sequences
        h_dense, node_mask = to_dense_batch(h, batch)  # [B, L, D], [B, L]
        B, L, _ = h_dense.shape

        # True means PAD
        key_padding_mask = ~node_mask                   # [B, L]

        # GT processor on node tokens only
        for layer in self.layers:
            h_dense = layer(h_dense, key_padding_mask)
            h_dense = h_dense.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        # Mean pooling over valid nodes
        mask = node_mask.unsqueeze(-1).float()          # [B, L, 1]
        hg = (h_dense * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        return self.decoder(hg).squeeze(-1)