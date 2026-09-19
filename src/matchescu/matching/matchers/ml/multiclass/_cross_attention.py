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

    Splits BERT token-level hidden states into per-attribute chunks at COL
    token boundaries (supplied externally, keeping this module
    tokenizer-agnostic) and applies standard ``MultiheadAttention`` per
    attribute in both directions. The two directions use separate attention
    instances.
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
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
        hidden: torch.Tensor,
        col_positions: torch.Tensor,
        segment_mask: torch.Tensor,
    ) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
        """Split hidden states into per-attribute spans for each batch item.

        Only COL positions that belong to the same segment (``segment_mask`` is
        1.0 at the COL token) open a span. Each segment's span list holds
        that segment's attributes in order and index ``i`` of the two
        segments refers to the same attribute.

        Args:
            hidden: (batch, seq_len, hidden).
            col_positions: (batch, max_cols) — positions of COL tokens for
                each item, padded with -1.
            segment_mask: (batch, seq_len) — float mask for the segment
                (segment_a or segment_b).

        Returns:
            Nested list ``[batch][attr]`` of ``(span_len, hidden)`` and
            ``(span_len,)`` boolean mask tensors.
        """
        batch_size = hidden.size(0)
        spans: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        for b in range(batch_size):
            cols = col_positions[b]
            valid_cols = cols[cols >= 0].tolist()
            owns = segment_mask[b, valid_cols] == 1.0
            item_cols = [col_pos for col_pos, owns in zip(valid_cols, owns) if owns]
            item_spans: list[tuple[torch.Tensor, torch.Tensor]] = []
            for i, col_pos in enumerate(item_cols):
                if i + 1 < len(item_cols):
                    end_pos = item_cols[i + 1]
                else:
                    end_pos = next(
                        (p for p in valid_cols if p > col_pos), hidden.size(1)
                    )
                span_mask = segment_mask[b, col_pos:end_pos]
                span = hidden[b, col_pos:end_pos] * span_mask.unsqueeze(-1)
                item_spans.append((span, span_mask.eq(0)))
            spans.append(item_spans)
        return spans

    def forward(
        self,
        hidden: torch.Tensor,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
        col_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce ``(enc_a, enc_b)`` via directional per-attribute attention.

        Args:
            hidden: (batch, seq_len, hidden) — BERT token-level hidden states.
            mask_a: (batch, seq_len) — float mask for segment A tokens.
            mask_b: (batch, seq_len) — float mask for segment B tokens.
            col_positions: (batch, max_cols) — positions of COL tokens for
                each item, padded with -1.

        Returns:
            (enc_a, enc_b) each of shape (batch, hidden).
        """
        spans_a = self._extract_attr_spans(hidden, col_positions, mask_a)
        spans_b = self._extract_attr_spans(hidden, col_positions, mask_b)

        batch_size = hidden.size(0)
        enc_a_list: list[torch.Tensor] = []
        enc_b_list: list[torch.Tensor] = []

        for b in range(batch_size):
            attr_outs_a: list[torch.Tensor] = []
            attr_outs_b: list[torch.Tensor] = []
            for (span_a, pad_a), (span_b, pad_b) in zip(spans_a[b], spans_b[b]):
                if span_a.size(0) == 0 or span_b.size(0) == 0:
                    continue
                sa = span_a.unsqueeze(0)
                sb = span_b.unsqueeze(0)
                out_a, _ = self.attn_a(
                    query=sa, key=sb, value=sb, key_padding_mask=pad_b.unsqueeze(0)
                )
                out_b, _ = self.attn_b(
                    query=sb, key=sa, value=sa, key_padding_mask=pad_a.unsqueeze(0)
                )
                valid_a = out_a.squeeze(0)[~pad_b]
                valid_b = out_b.squeeze(0)[~pad_a]
                if valid_a.size(0) == 0 or valid_b.size(0) == 0:
                    continue
                attr_outs_a.append(valid_a.mean(dim=0))
                attr_outs_b.append(valid_b.mean(dim=0))
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
