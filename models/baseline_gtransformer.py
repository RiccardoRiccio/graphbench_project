import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, to_dense_adj


class StructuralBiasedAttention(nn.Module):
    """
    Multi-head self-attention with additive graph structural bias:

        softmax(QK^T / sqrt(d_h) + B) V

    Inputs:
      x:              [B, L, D]
      attn_bias:      [B, H, L, L]
      key_padding_mask: [B, L], True means PAD
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

    def forward(
        self,
        x: torch.Tensor,                 # [B, L, D]
        attn_bias: torch.Tensor,         # [B, H, L, L]
        key_padding_mask: torch.Tensor,  # [B, L]
    ) -> torch.Tensor:
        B, L, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, L, H, Dh).transpose(1, 2)  # [B, H, L, Dh]
        k = self.k_proj(x).view(B, L, H, Dh).transpose(1, 2)  # [B, H, L, Dh]
        v = self.v_proj(x).view(B, L, H, Dh).transpose(1, 2)  # [B, H, L, Dh]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, L]
        scores = scores + attn_bias

        if key_padding_mask is not None:
            key_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)    # [B,1,1,L]
            query_mask = key_padding_mask.unsqueeze(1).unsqueeze(3)  # [B,1,L,1]
            scores = scores.masked_fill(key_mask, float("-inf"))
            scores = scores.masked_fill(query_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

        out = torch.matmul(attn, v)  # [B, H, L, Dh]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class GTPhi(nn.Module):
    """
    phi(X, G) = MultiHeadAttention(X, G)
    The FFN lives outside phi, in GTLayer.
    """
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.attn = StructuralBiasedAttention(hidden_dim, num_heads)

    def forward(self, x, attn_bias, key_padding_mask):
        return self.attn(x, attn_bias, key_padding_mask)


class GTLayer(nn.Module):
    """
    Pre-norm transformer-style graph layer:

        X'  = X + phi(LayerNorm(X), G)
        X'' = X' + FFN(LayerNorm(X'))
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

    def forward(self, x, attn_bias, key_padding_mask):
        x = x + self.phi(self.norm1(x), attn_bias, key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class Decoder(nn.Module):
    """
    W2(LayerNorm(GELU(W1 x)))
    """
    def __init__(self, hidden_dim: int, out_dim: int = 1):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.w2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.w2(self.norm(F.gelu(self.w1(x))))


class GTransformer(nn.Module):
    """
    GraphBench-style GT approximation for graph-level electronic-circuit prediction.

    Assumptions:
      - node-level tokenization
      - CLS token for graph-level prediction
      - if edge_attr is missing, use one learnable default edge feature
      - structural bias added before softmax
      - output read from CLS token
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

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.node_encoder = nn.Linear(in_channels, hidden_dim)
        self.pe_proj = nn.Linear(pe_dim, hidden_dim) if pe_dim > 0 else None

        self.default_edge_feature = nn.Parameter(torch.zeros(1, hidden_dim))
        self.edge_proj = nn.Linear(hidden_dim, num_heads, bias=False)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.cls_loop_bias = nn.Parameter(torch.zeros(num_heads))
        self.cls_in_bias = nn.Parameter(torch.zeros(num_heads))   # CLS -> node
        self.cls_out_bias = nn.Parameter(torch.zeros(num_heads))  # node -> CLS

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.default_edge_feature, std=0.02)

        self.layers = nn.ModuleList(
            [GTLayer(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.decoder = Decoder(hidden_dim, out_dim=out_dim)

    def forward(self, data):
        x = data.x
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        x = x.float()

        batch = data.batch
        edge_index = data.edge_index

        h = self.node_encoder(x)

        if self.pe_proj is not None and hasattr(data, "pe") and data.pe is not None:
            h = h + self.pe_proj(data.pe.float())

        # Dense node tokens
        h_dense, node_mask = to_dense_batch(h, batch)  # [B, N, D], [B, N]
        B, N, D = h_dense.shape
        H = self.num_heads

        # Dense edge bias for node-node pairs
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            edge_feat = data.edge_attr.float()
            if edge_feat.dim() == 1:
                edge_feat = edge_feat.unsqueeze(-1)
            if edge_feat.size(-1) != self.hidden_dim:
                raise ValueError(
                    f"edge_attr last dim must be {self.hidden_dim}, got {edge_feat.size(-1)}"
                )
        else:
            edge_feat = self.default_edge_feature.expand(edge_index.size(1), -1)

        edge_bias_heads = self.edge_proj(edge_feat)  # [E, H]
        node_node_bias = to_dense_adj(
            edge_index, batch, edge_attr=edge_bias_heads, max_num_nodes=N
        )  # [B, N, N, H]
        node_node_bias = node_node_bias.permute(0, 3, 1, 2).contiguous()  # [B, H, N, N]

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)      # [B, 1, D]
        h_seq = torch.cat([cls, h_dense], dim=1)    # [B, N+1, D]

        # Full attention bias including CLS
        attn_bias = torch.zeros(B, H, N + 1, N + 1, device=h.device, dtype=h.dtype)
        attn_bias[:, :, 1:, 1:] = node_node_bias
        attn_bias[:, :, 0, 0] = self.cls_loop_bias.view(1, H)
        attn_bias[:, :, 0, 1:] = self.cls_in_bias.view(1, H, 1)
        attn_bias[:, :, 1:, 0] = self.cls_out_bias.view(1, H, 1)

        # Padding mask
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=h.device)
        key_padding_mask = torch.cat([cls_pad, ~node_mask], dim=1)  # [B, N+1]

        pad_mask = key_padding_mask.clone()
        pad_mask[:, 0] = False

        for layer in self.layers:
            h_seq = layer(h_seq, attn_bias, key_padding_mask)
            h_seq = h_seq.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        cls_out = h_seq[:, 0, :]
        return self.decoder(cls_out).squeeze(-1)