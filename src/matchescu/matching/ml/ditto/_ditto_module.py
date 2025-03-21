from typing import cast

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, BertModel


class DittoModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        alpha_aug: float = 0.8,
        device: torch.device | None = None,
    ):
        super().__init__()
        self._alpha_aug = alpha_aug
        self._bert_name = pretrained_model_name
        self._bert = cast(BertModel, AutoModel.from_pretrained(pretrained_model_name))
        hidden_size = self._bert.config.hidden_size

        self._classifier = torch.nn.Linear(hidden_size, 1)
        self._device = device

    @property
    def device(self) -> torch.device:
        return self._device

    def forward(self, x1, x2=None):
        enc = self._bert_encode(x1, x2)
        return self._classifier(enc).squeeze(1)

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

    def with_frozen_bert_layers(self, frozen_layer_count: int = 0) -> "DittoModel":
        if frozen_layer_count < 1:
            return self
        for param in self._bert.embeddings.parameters():
            param.requires_grad = False

        for layer in self._bert.encoder.layer[
            :frozen_layer_count
        ]:  # Freeze encoder layers
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
        self._device = device

    def train(self, mode: bool = True) -> "DittoModel":
        if mode:
            self._bert.train(True)
            self._bert.gradient_checkpointing_enable()
            self._classifier.train(True)
        else:
            self._classifier.train(False)
            self._bert.gradient_checkpointing_disable()
            self._bert.train(False)
            torch.mps.empty_cache()
            torch.cuda.empty_cache()
        return self
