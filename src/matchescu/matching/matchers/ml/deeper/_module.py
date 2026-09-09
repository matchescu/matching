import logging

import torch
from torch import nn
from transformers import BertModel

from ._params import DeepERParams

_logger = logging.getLogger(__name__)


class DeepERModule(nn.Module):
    """DeepER matcher - simple, independent attribute comparisons."""

    __DEFAULT_BERT_MODEL = "google-bert/bert-base-uncased"

    def __init__(self, params: DeepERParams):
        super().__init__()
        self.num_attributes = params.num_attributes
        self.bert: BertModel = BertModel.from_pretrained(
            params.model_name or self.__DEFAULT_BERT_MODEL,
            attn_implementation="eager",
        )
        self._bert_trainable = self.__freeze_bert(params.frozen_layer_count)
        embedding_dim = self.bert.config.hidden_size
        self._lstm = nn.LSTM(
            embedding_dim,
            params.lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        lstm_out_dim = params.lstm_hidden_size * (2 if self._lstm.bidirectional else 1)
        self._similarity = nn.Sequential(
            # input is [|h_l - h_r| ; h_l * h_r], each lstm_out_dim wide
            nn.Linear(2 * lstm_out_dim, params.similarity_hidden_size),
            nn.ReLU(),
            nn.Linear(params.similarity_hidden_size, params.similarity_hidden_size),
        )
        self.classifier = nn.Linear(
            params.num_attributes * params.similarity_hidden_size, params.output_size
        )
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

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
        if len(left_attrs) != self.num_attributes:
            raise ValueError(
                f"expected {self.num_attributes} attributes, got {len(left_attrs)}"
            )

        left_embs = self.__encode_all_attrs(left_attrs)
        right_embs = self.__encode_all_attrs(right_attrs)

        similarities = []
        for lhs_emb, rhs_emb, lhs, rhs in zip(
            left_embs, right_embs, left_attrs, right_attrs
        ):
            h_l = self.__compose_attr(lhs_emb, lhs["attention_mask"])
            h_r = self.__compose_attr(rhs_emb, rhs["attention_mask"])
            diff = torch.abs(h_l - h_r)  # (B, H)
            prod = h_l * h_r
            similarities.append(
                self._similarity(torch.cat([diff, prod], dim=-1))
            )  # (B, sim_dim)

        combined = torch.cat(similarities, dim=-1)  # (B, sim_dim * K)
        return self.classifier(combined)  # (B, 2)

    def __empty_cache(self) -> None:
        device_type = self.device.type
        if device_type == "mps":
            torch.mps.empty_cache()
        elif device_type == "cuda":
            torch.cuda.empty_cache()

    def train(self, mode: bool = True) -> "DeepERModule":
        super().train(mode)
        if self._bert_trainable:
            self.bert.train(mode)
        else:
            self.bert.eval()
        if not mode:
            self.__empty_cache()
        return self

    def eval(self) -> "DeepERModule":
        return self.train(False)

    def to(self, *args, **kwargs) -> "DeepERModule":
        super().to(*args, **kwargs)
        return self

    def __freeze_bert(self, frozen_layer_count: int) -> bool:
        if frozen_layer_count < 1:
            return True

        num_layers = self.bert.config.num_hidden_layers
        if frozen_layer_count > num_layers:
            _logger.warning(
                "frozen_layer_count=%d exceeds num_hidden_layers=%d; clamping",
                frozen_layer_count,
                num_layers,
            )
            frozen_layer_count = num_layers

        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:frozen_layer_count]:
            for param in layer.parameters():
                param.requires_grad = False

        if frozen_layer_count >= num_layers:
            _logger.warning(
                "BERT is fully frozen (frozen_layer_count=%d of %d layers): it will "
                "act as a fixed feature extractor and receive no gradients.",
                frozen_layer_count,
                num_layers,
            )
            # the pooler is unused; freeze it too so optimizers see a clean graph
            if getattr(self.bert, "pooler", None) is not None:
                for param in self.bert.pooler.parameters():
                    param.requires_grad = False
            return False
        return True

    def __encode_all_attrs(self, attrs: list[dict]) -> tuple[torch.Tensor, ...]:
        # B = batch size, K = len(attrs), T = padded width (identical for all attrs)
        if not attrs:
            return ()
        shapes = {tuple(a["input_ids"].shape) for a in attrs}
        if len(shapes) != 1:
            raise ValueError(f"all attrs must share (B, T); got {sorted(shapes)}")
        batch_size = attrs[0]["input_ids"].size(0)
        device = self.device
        stacked_ids = torch.cat([a["input_ids"] for a in attrs], dim=0).to(
            device=device, dtype=torch.long
        )  # (B*K, T); dtype guards a float/int32 mask from the collator
        stacked_mask = torch.cat([a["attention_mask"] for a in attrs], dim=0).to(
            device=device, dtype=torch.long
        )  # (B*K, T)
        emb = self.bert(
            input_ids=stacked_ids, attention_mask=stacked_mask
        ).last_hidden_state

        return emb.split(batch_size, dim=0)

    def __compose_attr(
        self,
        emb: torch.Tensor,  # (B, T, H)
        attention_mask: torch.Tensor,  # (B, T)
    ) -> torch.Tensor:
        lengths = attention_mask.sum(1)
        nonempty = (lengths > 0).unsqueeze(-1).to(emb.device)
        lengths = lengths.to(dtype=torch.int64, device="cpu").clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self._lstm(packed)
        if self._lstm.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, 2 * hidden_size)
        else:
            h = h_n[-1]  # (B, hidden_size)
        return h * nonempty.to(h.dtype)
