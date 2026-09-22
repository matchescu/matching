import torch
from torch import nn


class PooledCrossAttention(nn.Module):
    """Cross-attention over pooled segment representations."""

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce ``(enc_a, enc_b)`` via cross-attention.

        Args:
            hidden: (batch, seq_len, hidden) — BERT token-level hidden states.
            mask_a: (batch, seq_len) — float mask for segment A tokens.
            mask_b: (batch, seq_len) — float mask for segment B tokens.

        Returns:
            (enc_a, enc_b) each of shape (batch, hidden).
        """
        pooled_a = (hidden * mask_a.unsqueeze(-1)).sum(1) / mask_a.sum(
            1, keepdim=True
        ).clamp(min=1e-9)
        pooled_b = (hidden * mask_b.unsqueeze(-1)).sum(1) / mask_b.sum(
            1, keepdim=True
        ).clamp(min=1e-9)

        seg_a = hidden * mask_a.unsqueeze(-1)
        seg_b = hidden * mask_b.unsqueeze(-1)

        key_padding_mask_a = mask_a == 0
        key_padding_mask_b = mask_b == 0

        enc_a, _ = self.attn(
            query=pooled_a.unsqueeze(1),
            key=seg_b,
            value=seg_b,
            key_padding_mask=key_padding_mask_b,
        )
        enc_b, _ = self.attn(
            query=pooled_b.unsqueeze(1),
            key=seg_a,
            value=seg_a,
            key_padding_mask=key_padding_mask_a,
        )
        return enc_a.squeeze(1), enc_b.squeeze(1)


class PerAttributeCrossAttention(nn.Module):
    """Directional per-attribute cross-attention.

    Splits BERT token-level hidden states into per-attribute spans at COL
    token boundaries (supplied externally via ``col_positions``) and applies
    directional ``MultiheadAttention`` per attribute in both directions.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.residual = residual
        self.attn_a = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_b = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

    @staticmethod
    def _extract_attr_spans(
        hidden_item: torch.Tensor,
        col_indices: list[int],
        segment_mask_item: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Extract contiguous per-attribute spans for a single sequence.

        Args:
            hidden_item: (seq_len, hidden) token embeddings.
            col_indices: Sorted list of COL token positions belonging to this segment.
            segment_mask_item: (seq_len,) mask indicating tokens in this segment.

        Returns:
            List of (span_len, hidden) tensors, one per attribute.
        """
        if not col_indices:
            return []

        active_indices = torch.nonzero(segment_mask_item > 0, as_tuple=False).squeeze(
            -1
        )
        if active_indices.numel() == 0:
            return []
        seg_end = active_indices[-1].item() + 1

        spans: list[torch.Tensor] = []
        for i, start in enumerate(col_indices):
            end = col_indices[i + 1] if i + 1 < len(col_indices) else seg_end
            if end > start:
                spans.append(hidden_item[start:end])
        return spans

    def forward(
        self,
        hidden: torch.Tensor,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
        col_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce ``(enc_a, enc_b)`` via directional per-attribute attention.

        :param hidden: (batch, seq_len, hidden) — BERT token-level hidden states.
        :param mask_a: (batch, seq_len) — mask for segment A tokens.
        :param mask_b: (batch, seq_len) — mask for segment B tokens.
        :param col_positions: (batch, max_cols) — positions of COL tokens, padded with -1.

        :returns:
            (enc_a, enc_b) each of shape (batch, hidden).
        """
        batch_size, seq_len, _ = hidden.shape
        enc_a_list: list[torch.Tensor] = []
        enc_b_list: list[torch.Tensor] = []

        for i in range(batch_size):
            # Identify valid COL positions belonging to segment A and segment B
            valid_cols = col_positions[i][col_positions[i] >= 0]
            cols_a = [p.item() for p in valid_cols if p < seq_len and mask_a[i, p] > 0]
            cols_b = [p.item() for p in valid_cols if p < seq_len and mask_b[i, p] > 0]
            cols_a.sort()
            cols_b.sort()

            spans_a = self._extract_attr_spans(hidden[i], cols_a, mask_a[i])
            spans_b = self._extract_attr_spans(hidden[i], cols_b, mask_b[i])

            attr_outs_a: list[torch.Tensor] = []
            attr_outs_b: list[torch.Tensor] = []

            # Pair attributes in order of appearance
            for span_a, span_b in zip(spans_a, spans_b):
                if span_a.size(0) == 0 or span_b.size(0) == 0:
                    continue

                sa = span_a.unsqueeze(0)  # (1, len_a, hidden)
                sb = span_b.unsqueeze(0)  # (1, len_b, hidden)

                # Cross-attend: A queries B, B queries A
                out_a, _ = self.attn_a(query=sa, key=sb, value=sb)  # (1, len_a, hidden)
                out_b, _ = self.attn_b(query=sb, key=sa, value=sa)  # (1, len_b, hidden)

                if self.residual:
                    rep_a = (sa + out_a).mean(dim=1).squeeze(0)
                    rep_b = (sb + out_b).mean(dim=1).squeeze(0)
                else:
                    rep_a = out_a.mean(dim=1).squeeze(0)
                    rep_b = out_b.mean(dim=1).squeeze(0)

                attr_outs_a.append(rep_a)
                attr_outs_b.append(rep_b)

            if attr_outs_a:
                enc_a_list.append(torch.stack(attr_outs_a).mean(dim=0))
            else:
                enc_a_list.append(
                    torch.zeros(
                        self.hidden_size, dtype=hidden.dtype, device=hidden.device
                    )
                )

            if attr_outs_b:
                enc_b_list.append(torch.stack(attr_outs_b).mean(dim=0))
            else:
                enc_b_list.append(
                    torch.zeros(
                        self.hidden_size, dtype=hidden.dtype, device=hidden.device
                    )
                )

        enc_a = torch.stack(enc_a_list)
        enc_b = torch.stack(enc_b_list)
        return enc_a, enc_b
