"""Hybrid attention entity matching model implementation"""

import torch
import torch.nn as nn
from transformers import BertModel

from ._params import DeepERParams


class DeepERModule(nn.Module):
    """DeepER matcher - simple, independent attribute comparisons."""

    __DEFAULT_BERT_MODEL = "google-bert/bert-base-uncased"

    def __init__(self, params: DeepERParams):
        super().__init__()
        self.num_attributes = params.num_attributes
        self.bert: BertModel = BertModel.from_pretrained(
            params.model_name or self.__DEFAULT_BERT_MODEL
        )
        self.__freeze_bert(params.frozen_layer_count)
        embedding_dim = self.bert.config.hidden_size
        self._lstm = nn.LSTM(embedding_dim, params.lstm_hidden_size, batch_first=True)
        self._similarity = nn.Sequential(
            nn.Linear(params.lstm_hidden_size, params.similarity_hidden_size),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(
            params.num_attributes * params.similarity_hidden_size, params.output_size
        )
        self._device = torch.device("cpu")

    def __freeze_bert(self, frozen_layer_count: int):
        if frozen_layer_count < 1:
            return self

        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:frozen_layer_count]:
            for param in layer.parameters():
                param.requires_grad = False
        return self

    def __compose(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # (B, T, 768)
        emb = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        lengths = attention_mask.sum(1).cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        # h_n: (1, B, H)
        _, (h_n, _) = self._lstm(packed)

        return h_n.squeeze(0)  # (B, H)

    def forward(self, left_attrs: list[dict], right_attrs: list[dict]) -> torch.Tensor:
        """Serialize LHS and RHS of a comparison inside 2 lists.

        Length of lists = number of attributes. Each dict in each list contains
        the actual batched tokens and attention masks for each item. This model
        computes the similarity on each attribute independently before classifying
        their differences.

        Args:
            left_attrs: list of dicts for the LHS of a comparison. Each dict
                must contain the 'input_ids' and 'attention_mask' keys
            right_attrs: list of dicts for the RHS of a comparison. Each dict
                must contain the 'input_ids' and 'attention_mask' keys

        Returns:
            similarity reduced to logits (configurable, usually 2). length of
            return tensor along the first dimension = batch size.
        """
        if len(left_attrs) != len(right_attrs):
            raise ValueError("lhs and rhs must have the same number of attrs")
        similarities = []
        for lhs, rhs in zip(left_attrs, right_attrs):
            h_l = self.__compose(**lhs)
            h_r = self.__compose(**rhs)
            diff = torch.abs(h_l - h_r)  # (B, H)
            similarities.append(self._similarity(diff))  # (B, sim_dim)

        combined = torch.cat(similarities, dim=-1)  # (B, sim_dim * K)
        return self.classifier(combined)  # (B, 2)

    def train(self, mode: bool = True) -> "DeepERModule":
        self.bert.train(mode)
        self._lstm.train(mode)
        self._similarity.train(mode)
        self.classifier.train(mode)
        if not mode and self._device is not None:
            if self._device.type == "mps":
                torch.mps.empty_cache()
            elif self._device.type == "cuda":
                torch.cuda.empty_cache()
        return self

    def eval(self):
        self.to(torch.device("cpu"))
        self.bert.eval()
        self._lstm.eval()
        self._similarity.eval()
        self.classifier.eval()

    def to(self, device: str | torch.device) -> None:
        super().to(device)
        self._device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.bert.to(self._device)
        self._lstm.to(self._device)
        self._similarity.to(self._device)
        self.classifier.to(self._device)
