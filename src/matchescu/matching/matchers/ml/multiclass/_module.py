from typing import cast

import torch
from torch import nn
from transformers import AutoModel, BertModel

from ._classifier import ClassificationHead
from ._cross_attention import PerAttributeCrossAttention, PooledCrossAttention
from ._params import MultiClassTrainingParams
from ._types import ArchitectureType, HeadType


class MultiClassModule(nn.Module):
    _CLASSIFIER_OUTPUT_SIZE = 3
    _DEFAULT_MODEL = "bert-base-uncased"

    def __init__(self, params: MultiClassTrainingParams):
        super().__init__()
        self._bert_name = params.model_name or self._DEFAULT_MODEL
        self._bert = cast(
            BertModel,
            AutoModel.from_pretrained(self._bert_name, attn_implementation="eager"),
        )
        self._head_type = params.head_type
        self._architecture = params.architecture
        hidden_size = self._bert.config.hidden_size

        if self._architecture == ArchitectureType.BERT_CROSS_ATTN:
            self._cross_attn = PooledCrossAttention(
                hidden_size, dropout=params.dropout_p
            )
        elif self._architecture == ArchitectureType.BERT_PER_ATTR_CROSS_ATTN:
            self._cross_attn = PerAttributeCrossAttention(
                hidden_size, dropout=params.dropout_p
            )
        else:
            self._cross_attn = None

        head_multiplier = 3 if self._head_type != HeadType.NONE else 2
        self._classifier = ClassificationHead(
            head_multiplier * hidden_size,
            hidden_size,
            self._CLASSIFIER_OUTPUT_SIZE,
            params.dropout_p,
            dtype=self._bert.dtype,
        )
        self._device = None

    @property
    def encoder_layers(self) -> nn.ModuleList:
        return self._bert.encoder.layer

    @property
    def embeddings_layer(self) -> nn.Module:
        return self._bert.embeddings

    @property
    def classifier(self) -> nn.Module:
        return self._classifier

    @property
    def cross_attention(self) -> nn.Module | None:
        return self._cross_attn

    @property
    def classifier_input_size(self) -> int:
        return self._classifier.input_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        col_positions: torch.Tensor | None = None,
    ):
        enc = self._bert_encode(
            input_ids, attention_mask, token_type_ids, col_positions
        )
        return self._classifier(enc)

    def _bert_encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        col_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self._bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        hidden = out[0]
        mask = attention_mask.unsqueeze(-1).float()

        mask_a = (token_type_ids == 0).unsqueeze(-1).float() * mask
        mask_b = (token_type_ids == 1).unsqueeze(-1).float() * mask

        enc_a, enc_b = self._compute_encodings(hidden, mask_a, mask_b, col_positions)
        return self._apply_head(enc_a, enc_b)

    def _compute_encodings(
        self,
        hidden: torch.Tensor,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
        col_positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_a_flat = mask_a.squeeze(-1)
        mask_b_flat = mask_b.squeeze(-1)

        if self._architecture == ArchitectureType.BERT_CROSS_ATTN:
            return self._cross_attn(hidden, mask_a_flat, mask_b_flat)
        elif self._architecture == ArchitectureType.BERT_PER_ATTR_CROSS_ATTN:
            if col_positions is None:
                raise ValueError(
                    "col_positions required for bert_per_attr_cross_attn architecture"
                )
            return self._cross_attn(hidden, mask_a_flat, mask_b_flat, col_positions)
        else:
            enc_a = (hidden * mask_a).sum(1) / mask_a.sum(1).clamp(min=1e-9)
            enc_b = (hidden * mask_b).sum(1) / mask_b.sum(1).clamp(min=1e-9)
            return enc_a, enc_b

    def _apply_head(self, enc_a: torch.Tensor, enc_b: torch.Tensor) -> torch.Tensor:
        match self._head_type:
            case HeadType.NONE:
                return torch.cat([enc_a, enc_b], dim=-1)
            case HeadType.ABS:
                return torch.cat([enc_a, enc_b, torch.abs(enc_a - enc_b)], dim=-1)

        return torch.cat([enc_a, enc_b, enc_a - enc_b], dim=-1)

    def with_frozen_bert_layers(
        self, frozen_layer_count: int = 6
    ) -> "MultiClassModule":
        if frozen_layer_count < 1:
            return self

        for param in self._bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self._bert.encoder.layer[:frozen_layer_count]:
            for param in layer.parameters():
                param.requires_grad = False
        return self

    def eval(self):
        super().eval()
        self.to(torch.device("cpu"))

    def to(self, device: str | torch.device) -> None:
        self._bert = self._bert.to(device)
        self._classifier = self._classifier.to(device)
        if self._cross_attn is not None:
            self._cross_attn = self._cross_attn.to(device)
        self._device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )

    def train(self, mode: bool = True) -> "MultiClassModule":
        super().train(mode)
        if not mode and self._device is not None:
            match self._device.type:
                case "mps":
                    torch.mps.empty_cache()
                case "cuda":
                    torch.cuda.empty_cache()
        return self
