# gt_circuits.py
"""
Graph Transformer (GT) Baseline for GraphBench Electronic Circuits.

What the paper specifies (Appendix B.2 "Graph transformer architecture" + Table 27):

  Citation: Bechler-Speicher et al. (2025) — non-public ArXiv preprint.
  This is the ONLY external reference given for the GT architecture.
  No standard library implementation is cited (unlike GIN → Hu et al. 2020b).

  Attention formula (stated explicitly in the paper):
      Attention(Q, K, V, B) = softmax( d^{-1/2} · QK^T + B ) V
      where B ∈ R^{L×L} is a learnable attention bias added to logits.

  [CLS] token: prepended per graph for graph-level readout.
      Paper states: "we add a [CLS] token for graph-level representations"
      Readout = CLS token at position 0 of the output sequence.
      This means GT does NOT use global_mean_pool — unlike GCN/GAT/GIN.

  Node-level tokenization: each graph node = one token.
  No positional encodings: no PE row in Table 27 for Electronic Circuits.

  Hyperparameters (Table 27):
      num_layers = 6  (only model with 6; all others have 4)
      hidden_dim = 384
      num_heads  = 4
      activation = GELU
      lr=1e-3, batch=512, weight_decay=0, dropout=0, epochs=700, Adam

  Processor residual pattern (Appendix B Eq.3 — same for all baselines):
      X = X + Attention( LayerNorm_1(X) )          # Pre-LN + attn + residual
      X = X + W2( GELU( W1 · LayerNorm_2(X) ) )   # FFN residual block

  Decoder (Appendix B):
      out = W2( LayerNorm( GELU( W1·x ) ) )
      Order: Linear → GELU → LayerNorm → Linear → scalar

  Ambiguity — exact nature of B:
      Paper references Bechler-Speicher et al. (2025) which has no public code.
      We implement B as a learnable scalar per head:
          - one value for edge-connected pairs
          - one value for non-connected pairs
      This is the simplest structural bias consistent with the formula.

Data notes (from sanity check):
  - x shape: [num_nodes, 1, 9] → squeeze(1) → [num_nodes, 9]
  - in_channels = 9  (one-hot over 9 node types)
  - edge_attr = None
  - y = scalar  (one target per run: eff or vout)
  - split keys: "train", "valid", "test"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, to_dense_adj


# ── Biased Multi-Head Attention ─────────────────────────────────────────────

class BiasedMHA(nn.Module):
    """
    Attention(Q, K, V, B) = softmax( d^{-1/2} · QK^T + B ) V

    B is built per-forward from the graph structure:
        B[i,j] = edge_bias[h]     if (i,j) is an edge
        B[i,j] = no_edge_bias[h]  otherwise
    One learnable scalar per head for each case.
    Padding positions are masked to -inf before softmax.
    """
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_O = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Learnable structural bias scalars per head
        self.edge_bias    = nn.Parameter(torch.zeros(num_heads))  # connected
        self.no_edge_bias = nn.Parameter(torch.zeros(num_heads))  # not connected

    def forward(
        self,
        x:                torch.Tensor,   # (B, L, d)  — dense, includes CLS at pos 0
        adj:              torch.Tensor,   # (B, L, L)  — adjacency (0/1), node rows only
        key_padding_mask: torch.Tensor,   # (B, L) bool, True = pad position
    ) -> torch.Tensor:                    # (B, L, d)

        B, L, d = x.shape
        H, Dh   = self.num_heads, self.head_dim

        # Project and reshape to (B, H, L, Dh)
        Q = self.W_Q(x).view(B, L, H, Dh).transpose(1, 2)
        K = self.W_K(x).view(B, L, H, Dh).transpose(1, 2)
        V = self.W_V(x).view(B, L, H, Dh).transpose(1, 2)

        # Attention logits (B, H, L, L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Build structural bias B of shape (B, H, L, L)
        # adj is over node tokens; CLS row/col gets no_edge_bias everywhere
        # adj shape: (B, L, L) where L = 1 + num_nodes (CLS + nodes)
        # We treat adj[b, 0, :] = adj[b, :, 0] = 0 (CLS has no graph edges)
        bias = (
            self.no_edge_bias.view(1, H, 1, 1) +
            adj.unsqueeze(1) * (
                self.edge_bias.view(1, H, 1, 1) -
                self.no_edge_bias.view(1, H, 1, 1)
            )
        )  # (B, H, L, L)
        scores = scores + bias

        # Mask padding positions
        if key_padding_mask is not None:
            # key_padding_mask: (B, L) True = pad → mask keys
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # (B, 1, 1, L)
                float("-inf")
            )

        attn    = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)                                # (B, H, L, Dh)
        context = context.transpose(1, 2).contiguous().view(B, L, d)  # (B, L, d)
        return self.W_O(context)


# ── GT processor layer ───────────────────────────────────────────────────────

class GTLayer(nn.Module):
    """
    Appendix B Eq.3:
        X = X + Attention( LayerNorm_1(X) )          # Pre-LN + attn + residual
        X = X + W2( GELU( W1 · LayerNorm_2(X) ) )   # FFN residual block
    """
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn  = BiasedMHA(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.w1    = nn.Linear(hidden_dim, hidden_dim)
        self.w2    = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x:                torch.Tensor,  # (B, L, d)
        adj:              torch.Tensor,  # (B, L, L)
        key_padding_mask: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), adj, key_padding_mask)
        x = x + self.w2(F.gelu(self.w1(self.norm2(x))))
        return x


# ── Decoder ─────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Appendix B Decoder:  W2( LayerNorm( GELU( W1·x ) ) )
    Order: Linear → GELU → LayerNorm → Linear
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w1   = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.w2   = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.norm(F.gelu(self.w1(x))))


# ── Full GT model ────────────────────────────────────────────────────────────

class GTElectronicCircuits(nn.Module):
    """
    GT baseline for GraphBench Electronic Circuits (Table 27).

    Encoder  : Linear(in_channels=9 → hidden_dim=384) + prepend learnable [CLS] token
    Processor: 6 × GTLayer  (Pre-LN + biased MHA + FFN residual)
    Readout  : CLS token at position 0  (NOT global_mean_pool)
    Decoder  : W2( LayerNorm( GELU( W1·x ) ) ) → scalar

    in_channels = 9   : x shape [num_nodes, 1, 9] → squeezed to [num_nodes, 9]
    num_layers  = 6   (GT only; all other baselines use 4)
    num_heads   = 4
    hidden_dim  = 384
    out         = scalar (one target per run: eff or vout)
    """
    def __init__(
        self,
        in_channels: int = 9,
        hidden_dim:  int = 384,
        num_layers:  int = 6,
        num_heads:   int = 4,
    ):
        super().__init__()

        # Encoder
        self.node_encoder = nn.Linear(in_channels, hidden_dim)
        self.cls_token    = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Processor
        self.layers = nn.ModuleList(
            [GTLayer(hidden_dim, num_heads) for _ in range(num_layers)]
        )

        # Decoder
        self.decoder = Decoder(hidden_dim)

    def forward(self, data) -> torch.Tensor:
        x          = data.x.squeeze(1) if data.x.dim() == 3 else data.x  # [N, 9]
        edge_index = data.edge_index
        batch      = data.batch

        # Encoder
        h = self.node_encoder(x)  # [N_total, d]

        # Pack into dense batch: (B, L_max, d), mask True = real node
        h_dense, node_mask = to_dense_batch(h, batch)   # (B, L, d), (B, L)
        B, L, d = h_dense.shape

        # Build adjacency for structural bias, shape (B, L, L)
        # to_dense_adj pads to L_max; we then pad for CLS at position 0
        adj_nodes = to_dense_adj(edge_index, batch, max_num_nodes=L)  # (B, L, L)

        # Prepend CLS token — row/col 0 in the sequence
        cls    = self.cls_token.expand(B, -1, -1)          # (B, 1, d)
        h_seq  = torch.cat([cls, h_dense], dim=1)          # (B, 1+L, d)

        # Adjacency padded for CLS: CLS has no structural edges (row/col of zeros)
        adj_cls   = torch.zeros(B, 1+L, 1+L, device=adj_nodes.device)
        adj_cls[:, 1:, 1:] = adj_nodes                     # (B, 1+L, 1+L)

        # key_padding_mask: True = position should be IGNORED
        # CLS is never masked; padded node positions are masked
        cls_valid        = torch.zeros(B, 1, dtype=torch.bool, device=h.device)
        key_padding_mask = torch.cat([cls_valid, ~node_mask], dim=1)  # (B, 1+L)

        # Processor
        for layer in self.layers:
            h_seq = layer(h_seq, adj_cls, key_padding_mask)

        # Readout: CLS token at position 0
        cls_out = h_seq[:, 0, :]   # (B, d)

        # Decoder
        return self.decoder(cls_out)  # (B, 1)