from typing import cast

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, BertModel

from ._params import MultiClassTrainingParams
from ._resnet import ResidualHead


class MultiClassModule(nn.Module):
    def __init__(self, params: MultiClassTrainingParams):
        super().__init__()
        self._alpha_aug = params.alpha_aug
        self._bert_name = params.model_name or "roberta-base"
        self._bert = cast(BertModel, AutoModel.from_pretrained(self._bert_name))
        hidden_size = self._bert.config.hidden_size
        self._classifier = ResidualHead(
            hidden_size, params.output_size, params.dropout_p, dtype=self._bert.dtype
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

    def forward(self, x1, x2=None):
        enc = self._bert_encode(x1, x2)
        return self._classifier(enc)

    def _bert_encode(self, x1, x2=None):
        if x2 is not None:
            # MixDA
            x_concat = torch.cat((x1, x2))
            enc = self._bert(x_concat)[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size]  # (batch_size, emb_size)
            enc2 = enc[batch_size:]  # (batch_size, emb_size)

            aug_lam = np.random.beta(self._alpha_aug, self._alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            enc = self._bert(x1)[0][:, 0, :]
        return enc

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
