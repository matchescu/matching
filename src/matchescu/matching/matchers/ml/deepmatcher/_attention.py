"""Bidirectional attention mechanism for symmetric entity comparison."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricAttention(nn.Module):
    """Bidirectional attention module ensuring input symmetry"""

    def __init__(self, hidden_size: int):
        """
        Args:
            hidden_size: Size of hidden dimension (per direction)
        """
        super().__init__()
        self.hidden_size = hidden_size

    def forward(
        self, left_encoded: torch.Tensor, right_encoded: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute symmetric comparison features using bidirectional attention

        Args:
            left_encoded: Left sequence encoding (batch, L, 2*hidden)
            right_encoded: Right sequence encoding (batch, R, 2*hidden)

        Returns:
            Symmetric comparison features (batch, L, 16*hidden)
        """
        batch_size, left_len, _ = left_encoded.shape
        _, right_len, _ = right_encoded.shape

        # Left → Right attention
        similarity_lr = torch.bmm(left_encoded, right_encoded.transpose(1, 2))
        attn_lr = F.softmax(similarity_lr, dim=2)
        left_context = torch.bmm(attn_lr, right_encoded)

        # Right → Left attention
        similarity_rl = torch.bmm(right_encoded, left_encoded.transpose(1, 2))
        attn_rl = F.softmax(similarity_rl, dim=2)
        right_context = torch.bmm(attn_rl, left_encoded)

        # Create symmetric comparison features
        left_diff = torch.abs(left_encoded - left_context)
        left_prod = left_encoded * left_context
        right_diff = torch.abs(right_encoded - right_context)
        right_prod = right_encoded * right_context

        return torch.cat(
            [
                left_encoded,
                left_context,
                left_diff,
                left_prod,
                right_encoded,
                right_context,
                right_diff,
                right_prod,
            ],
            dim=2,
        )
