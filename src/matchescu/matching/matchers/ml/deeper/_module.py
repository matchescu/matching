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
        self.__freeze_bert()
        embedding_dim = self.bert.config.hidden_size
        self._lstm = nn.LSTM(embedding_dim, params.lstm_hidden_size, batch_first=True)
        self._similarity = nn.Sequential(
            nn.Linear(params.lstm_hidden_size, params.similarity_hidden_size),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(params.similarity_hidden_size, params.output_size)
        self._device = torch.device("cpu")

    def __freeze_bert(self):
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer:
            for param in layer.parameters():
                param.requires_grad = False
        for param in self.bert.pooler.parameters():
            param.requires_grad = False
        return self

    def __encode_all_attrs(self, attrs: list[dict]) -> list[torch.Tensor]:
        # B = self.__batch_size, K = len(attrs), T_k = len(attrs[k])
        if not attrs:
            return []
        batch_size = attrs[0]["input_ids"].size(0)
        stacked_ids = torch.cat([a["input_ids"] for a in attrs], dim=0)  # (B*K, T)
        stacked_mask = torch.cat(
            [a["attention_mask"] for a in attrs], dim=0
        )  # (B*K, T)
        emb = self.bert(
            input_ids=stacked_ids, attention_mask=stacked_mask
        ).last_hidden_state

        return emb.split(batch_size, dim=0)

    def __compose_attr(
        self,
        emb: torch.Tensor,  # (B, T, H)
        attention_mask: torch.Tensor,  # (B, T)
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        lengths = attention_mask.sum(1).cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, c_n) = self._lstm(packed, state)
        return h_n.squeeze(0), (h_n, c_n)  # (B, H), state

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

        left_embs = self.__encode_all_attrs(left_attrs)
        right_embs = self.__encode_all_attrs(right_attrs)

        h_l, h_r, left_lstm, right_lstm = None, None, None, None
        for lhs_emb, rhs_emb, lhs, rhs in zip(
            left_embs, right_embs, left_attrs, right_attrs
        ):
            # combine all attrs using the same LSTM state
            h_l, left_lstm = self.__compose_attr(
                lhs_emb, lhs["attention_mask"], left_lstm
            )
            h_r, right_lstm = self.__compose_attr(
                rhs_emb, rhs["attention_mask"], right_lstm
            )

        diff = torch.abs(h_l - h_r)
        combined = self._similarity(diff)
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
