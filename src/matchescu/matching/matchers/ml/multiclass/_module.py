from typing import cast

import torch
import torch.nn as nn
from transformers import AutoModel, BertModel

from ._params import MultiClassTrainingParams
from ._classifier import ClassificationHead


class MultiClassModule(nn.Module):
    _CLASSIFIER_OUTPUT_SIZE = 3
    _DEFAULT_MODEL = "bert-base-uncased"

    def __init__(self, params: MultiClassTrainingParams):
        super().__init__()
        self._bert_name = params.model_name or self._DEFAULT_MODEL
        self._bert = cast(BertModel, AutoModel.from_pretrained(self._bert_name))
        hidden_size = self._bert.config.hidden_size
        self._classifier = ClassificationHead(
            3 * hidden_size,
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ):
        enc = self._bert_encode(input_ids, attention_mask, token_type_ids)
        return self._classifier(enc)

    def _bert_encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ):
        out = self._bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        hidden = out[0]
        mask = attention_mask.unsqueeze(-1).float()

        # separate segment A (lhs text) from segment B (rhs text)
        mask_a = (token_type_ids == 0).unsqueeze(-1).float() * mask
        mask_b = (token_type_ids == 1).unsqueeze(-1).float() * mask

        enc_a = (hidden * mask_a).sum(1) / mask_a.sum(1).clamp(min=1e-9)
        enc_b = (hidden * mask_b).sum(1) / mask_b.sum(1).clamp(min=1e-9)

        # difference -> higher contrast based on input order
        return torch.cat([enc_a, enc_b, enc_a - enc_b], dim=-1)

    def with_frozen_bert_layers(
        self, frozen_layer_count: int = 6
    ) -> "MultiClassModule":
        if frozen_layer_count < 1:
            return self

        for param in self._bert.embeddings.parameters():
            param.requires_grad = False
        # Freeze encoder layers
        for layer in self._bert.encoder.layer[:frozen_layer_count]:
            for param in layer.parameters():
                param.requires_grad = False
        return self

    def eval(self):
        self.to(torch.device("cpu"))
        self._bert.eval()
        self._classifier.eval()

    def to(self, device: str | torch.device) -> None:
        self._bert = self._bert.to(device)
        self._classifier = self._classifier.to(device)
        self._device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )

    def train(self, mode: bool = True) -> "MultiClassModule":
        if mode:
            self._bert.train(True)
            self._classifier.train(True)
        else:
            self._classifier.train(False)
            self._bert.train(False)
            if self._device is not None:
                match self._device.type:
                    case "mps":
                        torch.mps.empty_cache()
                    case "cuda":
                        torch.cuda.empty_cache()
        return self
