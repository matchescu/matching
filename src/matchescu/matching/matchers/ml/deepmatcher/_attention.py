import torch
import torch.nn as nn
import torch.nn.functional as F
from ._highway import HighwayNet


class HybridAttention(nn.Module):
    """DeepMatcher hybrid attention (primary sequence u1, context sequence u2).

    Steps (per paper Section 4.4):
      1. Soft alignment via HighwayNet-projected dot products.
      2. Comparison: concat(u'1[k], b1[k], |u'1[k]-b1[k]|) -> HighwayNet.
      3. Aggregation: weighted average of x1 using RNN2 last hidden state.
    """

    def __init__(
        self, rnn1: nn.GRU, rnn2: nn.GRU, hidden_size: int, dropout: float = 0.2
    ):
        """
        Args:
            rnn1: Bi-GRU used for encoding both u1 and u2 in alignment/comparison.
            rnn2: Bi-GRU used for encoding u2 in aggregation.
            hidden_size: Hidden size per direction of the RNNs.
            dropout: Dropout probability.
        """
        super().__init__()
        self.rnn1 = rnn1
        self.rnn2 = rnn2

        self.hidden_size = hidden_size
        rnn_out = 2 * hidden_size  # bidirectional output size

        self._align_highway = HighwayNet(rnn_out, num_layers=2, dropout=dropout)
        self._compare_highway = HighwayNet(3 * rnn_out, num_layers=2, dropout=dropout)

        # Input: concat(x1[k], g2) = 3*rnn_out + 1*rnn_out = 4*rnn_out
        agg_layer_size = 4 * rnn_out
        self.agg_highway = HighwayNet(agg_layer_size, num_layers=2, dropout=dropout)
        self.agg_linear = nn.Linear(agg_layer_size, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        u1_emb: torch.Tensor,  # (batch, L, embed_dim) — primary
        u2_emb: torch.Tensor,  # (batch, R, embed_dim) — context
    ) -> torch.Tensor:
        """
        Returns:
            Summary vector for u1 given u2 as context. Shape: (batch, 3 * 2 * hidden_size)
        """
        # --- RNN1 encodes both sequences ---
        u1_enc, _ = self.rnn1(u1_emb)  # (batch, L, 2h)
        u2_enc, _ = self.rnn1(u2_emb)  # (batch, R, 2h)

        # --- Step 1: Soft Alignment ---
        q1 = self._align_highway(u1_enc)  # (batch, L, 2h)
        q2 = self._align_highway(u2_enc)  # (batch, R, 2h)

        # Alignment matrix W: (batch, L, R)
        W = torch.bmm(q1, q2.transpose(1, 2))
        attn = F.softmax(W, dim=2)  # row-wise softmax

        # Soft-aligned encoding b1[k]: weighted average over RNN1-encoded u2
        b1 = torch.bmm(attn, u2_enc)  # (batch, L, 2h)

        # --- Step 2: Comparison ---
        # Input: [u'1[k]; b1[k]; |u'1[k] - b1[k]|]
        diff = torch.abs(u1_enc - b1)
        compare_input = torch.cat([u1_enc, b1, diff], dim=2)  # (batch, L, 6h)
        x1 = self._compare_highway(compare_input)  # (batch, L, 6h)

        # --- Step 3: Aggregation ---
        # RNN2 encodes u2; take last hidden state g2 (fix for Problems 3 & 4)
        _, h2 = self.rnn2(u2_emb)
        # h2 shape: (2, batch, hidden_size) for bidirectional; concatenate both directions
        g2 = torch.cat([h2[0], h2[1]], dim=1)  # (batch, 2h)
        g2_expanded = g2.unsqueeze(1).expand(-1, x1.size(1), -1)  # (batch, L, 2h)

        agg_input = torch.cat([x1, g2_expanded], dim=2)  # (batch, L, 8h)
        agg_scores = self.agg_linear(self.agg_highway(agg_input))  # (batch, L, 1)
        agg_weights = F.softmax(agg_scores, dim=1)  # (batch, L, 1)

        # Weighted average of x1
        summary = (agg_weights * x1).sum(dim=1)  # (batch, 6h)
        return summary
