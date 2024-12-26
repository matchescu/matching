import torch

from sklearn.metrics import precision_score, recall_score, f1_score
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class TorchEngine:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        device: str = "cpu",
    ) -> None:
        self._model = model
        self._loss = loss_fn
        self._optimizer = optimizer
        self._device = device
        self._stats = {}

    @property
    def stats(self):
        return self._stats

    def _train_epoch(self, data_loader: DataLoader) -> float:
        self._model.train(True)
        total_loss = 0.0
        for feats, labels in data_loader:
            feats = feats.to(self._device)
            labels = labels.to(self._device).squeeze()

            self._optimizer.zero_grad()
            out = self._model(feats)
            loss_val = self._loss(out, labels)
            loss_val.backward()
            self._optimizer.step()

            total_loss += loss_val.item()

        total_loss /= len(data_loader)
        return total_loss

    def _compute_stats(self, predictions: list, target: list) -> None:
        predictions = list(map(lambda x: torch.argmax(x).item(), predictions))
        target = list(map(lambda x: x.item(), target))
        self._stats = {
            "precision": precision_score(target, predictions),
            "recall": recall_score(target, predictions),
            "f1": f1_score(target, predictions),
        }

    def evaluate(self, data_loader: DataLoader, compute_stats: bool = False) -> float:
        self._model.eval()
        total_loss = 0.0
        predictions = []
        target = []

        with torch.no_grad():
            for feats, labels in data_loader:
                feats = feats.to(self._device)
                labels = labels.to(self._device).squeeze()

                out = self._model(feats)
                loss_val = self._loss(out, labels)
                if compute_stats:
                    predictions.extend(out)
                    target.extend(labels.float())
                total_loss += loss_val.item()
        if compute_stats:
            self._compute_stats(predictions, target)
        total_loss /= len(data_loader)
        return total_loss

    def train(
        self,
        train_loader: DataLoader,
        x_validation_loader: DataLoader,
        epochs: int = 10,
    ) -> nn.Module:
        self._model.to(self._device)

        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader)
            x_validation_loss = self.evaluate(x_validation_loader)

            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {x_validation_loss:.4f}"
            )

        return self._model
