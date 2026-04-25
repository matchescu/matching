import torch
import torch.nn as nn

from ._attention import HybridAttention
from ._params import DeepMatcherModelTrainingParams


class AttributeEncoder(nn.Module):
    """
    Encodes a single attribute pair using Hybrid attention.
    Processes the input in both directions.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int = 100,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.rnn1 = nn.GRU(
            embedding_dim, hidden_size, bidirectional=True, batch_first=True
        )
        self.rnn2 = nn.GRU(
            embedding_dim, hidden_size, bidirectional=True, batch_first=True
        )
        # Attention modules for each direction
        self.attn_lr = HybridAttention(self.rnn1, self.rnn2, hidden_size, dropout)
        self.attn_rl = HybridAttention(self.rnn1, self.rnn2, hidden_size, dropout)
        self._device = None

    def forward(
        self,
        left_tokens: torch.Tensor,
        right_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_emb = self.dropout(self.embedding(left_tokens))  # (batch, L, embed)
        right_emb = self.dropout(self.embedding(right_tokens))  # (batch, R, embed)

        # Both directions
        s1 = self.attn_lr(left_emb, right_emb)  # (batch, 6h) — left as primary
        s2 = self.attn_rl(right_emb, left_emb)  # (batch, 6h) — right as primary

        return s1, s2  # return separately for attribute comparison

    def train(self, mode: bool = True) -> "AttributeEncoder":
        self.embedding.train(mode)
        self.dropout.train(mode)
        self.rnn1.train(mode)
        self.rnn2.train(mode)
        self.attn_lr.train(mode)
        self.attn_rl.train(mode)
        return self

    def eval(self) -> "AttributeEncoder":
        self.to(torch.device("cpu"))
        self.embedding.eval()
        self.dropout.eval()
        self.rnn1.eval()
        self.rnn2.eval()
        self.attn_lr.eval()
        self.attn_rl.eval()
        return self

    def to(self, device: str | torch.device) -> "AttributeEncoder":
        super().to(device)
        self._device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.embedding.to(self._device)
        self.dropout.to(self._device)
        self.rnn1.to(self._device)
        self.rnn2.to(self._device)
        self.attn_lr.to(self._device)
        self.attn_rl.to(self._device)
        return self


class DeepMatcherModule(nn.Module):
    """Main entity matching model supporting multiple attributes."""

    def __init__(self, params: DeepMatcherModelTrainingParams):
        super().__init__()
        self.num_attributes = params.num_attributes
        rnn_out = 2 * params.hidden_size
        summary_dim = 3 * rnn_out  # 6h per direction

        self.attribute_encoder = AttributeEncoder(
            params.vocab_size, params.embedding_dim, params.hidden_size, params.dropout
        )
        # Each attribute contributes 3 * summary_dim = 3 * 6h = 18h
        attr_compare_dim = 3 * summary_dim  # per attribute
        total_dim = params.num_attributes * attr_compare_dim

        self.aggregator = nn.Sequential(
            nn.Linear(total_dim, params.hidden_size),
            nn.ReLU(),
            nn.Dropout(params.dropout),
            nn.Linear(params.hidden_size, 2),
        )
        self._device = None

    def forward(
        self, left_attrs: torch.Tensor, right_attrs: torch.Tensor
    ) -> torch.Tensor:
        attr_features = []

        for i in range(self.num_attributes):
            left = left_attrs[:, i, :]
            right = right_attrs[:, i, :]
            s1, s2 = self.attribute_encoder(left, right)
            diff = torch.abs(s1 - s2)
            attr_features.append(torch.cat([s1, s2, diff], dim=1))

        combined = torch.cat(attr_features, dim=1)
        return self.aggregator(combined)

    def train(self, mode: bool = True) -> "DeepMatcherModule":
        self.attribute_encoder.train(mode)
        self.aggregator.train(mode)
        return self

    def eval(self):
        self.to(torch.device("cpu"))
        self.attribute_encoder.eval()
        self.aggregator.eval()

    def to(self, device: str | torch.device) -> None:
        super().to(device)
        self._device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.attribute_encoder.to(device)
        self.aggregator.to(device)
