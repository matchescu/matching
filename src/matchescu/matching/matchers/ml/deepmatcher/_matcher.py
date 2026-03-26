"""Hybrid attention entity matching model implementation"""

import torch
import torch.nn as nn

from ._params import DeepMatcherModelTrainingParams
from ._highway import HighwayNet
from ._attention import SymmetricAttention


class AttributeEncoder(nn.Module):
    """Encodes a single attribute pair using bidirectional RNN and attention"""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int = 100,
        dropout: float = 0.2,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embedding vectors
            hidden_size: Size of RNN hidden state (per direction)
            dropout: Dropout probability
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bi_gru = nn.GRU(
            embedding_dim, hidden_size, bidirectional=True, batch_first=True
        )
        self.attention = SymmetricAttention(hidden_size)
        self.highway = HighwayNet(16 * hidden_size, num_layers=2, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def forward(
        self,
        left_tokens: torch.Tensor,
        right_tokens: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            left_tokens: Token indices for left entity (batch, seq_len)
            right_tokens: Token indices for right entity (batch, seq_len)
            return_features: If True, return intermediate features instead of score

        Returns:
            Match scores or intermediate features
        """
        # Embedding and encoding
        left_emb = self.dropout(self.embedding(left_tokens))
        right_emb = self.dropout(self.embedding(right_tokens))

        left_encoded, _ = self.bi_gru(left_emb)  # (batch, L, 2*hidden)
        right_encoded, _ = self.bi_gru(right_emb)  # (batch, R, 2*hidden)

        # Create symmetric comparison features
        comparison = self.attention(left_encoded, right_encoded)

        # Transform with Highway Network
        transformed = self.highway(comparison)

        # Aggregate through max pooling
        aggregated, _ = torch.max(transformed, dim=1)  # (batch, 16*hidden)

        if return_features:
            return aggregated

        # Final classification (will be handled by MultiAttributeMatcher)
        return aggregated

    def train(self, mode: bool = True) -> "AttributeEncoder":
        self.embedding.train(mode)
        self.bi_gru.train(mode)
        self.attention.train(mode)
        self.highway.train(mode)
        self.dropout.train(mode)
        return self

    def eval(self):
        self.to(torch.device("cpu"))
        self.embedding.eval()
        self.bi_gru.eval()
        self.attention.eval()
        self.highway.eval()
        self.dropout.eval()

    def to(self, device: str | torch.device) -> None:
        super().to(device)
        self.embedding.to(device)
        self.bi_gru.to(device)
        self.attention.to(device)
        self.highway.to(device)
        self.dropout.to(device)


class DeepMatcherModule(nn.Module):
    """Main entity matching model supporting multiple attributes"""

    def __init__(self, params: DeepMatcherModelTrainingParams):
        super().__init__()
        self.num_attributes = params.num_attributes

        # Shared encoder for all attributes (parameter sharing)
        self.attribute_encoder = AttributeEncoder(
            params.vocab_size, params.embedding_dim, params.hidden_size, params.dropout
        )

        # Final attribute aggregator
        self.aggregator = nn.Sequential(
            nn.Linear(
                params.num_attributes * 16 * params.hidden_size, params.hidden_size
            ),
            nn.ReLU(),
            nn.Dropout(params.dropout),
            nn.Linear(params.hidden_size, 2),
        )
        self._device = None

    def forward(
        self, left_attrs: torch.Tensor, right_attrs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            left_attrs: Left entity attributes (batch, num_attrs, seq_len)
            right_attrs: Right entity attributes (batch, num_attrs, seq_len)

        Returns:
            Match scores (batch_size, 2)
        """
        attr_features = []

        # Process each attribute independently
        for i in range(self.num_attributes):
            left = left_attrs[:, i, :]
            right = right_attrs[:, i, :]
            features = self.attribute_encoder(left, right, return_features=True)
            attr_features.append(features)

        # Concatenate attribute representations
        combined = torch.cat(attr_features, dim=1)

        # Final classification
        return self.aggregator(combined).squeeze(-1)

    def train(self, mode: bool = True) -> "DeepMatcherModule":
        self.attribute_encoder.train(mode)
        self.aggregator.train(mode)
        if not mode and self._device is not None:
            if self._device.type == "mps":
                torch.mps.empty_cache()
            elif self._device.type == "cuda":
                torch.cuda.empty_cache()
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
