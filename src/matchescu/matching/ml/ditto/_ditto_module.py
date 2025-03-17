import logging
import os
from functools import partial
from pathlib import Path
from typing import Callable, Any, cast

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch import autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, get_linear_schedule_with_warmup, BertModel

from matchescu.matching.ml.ditto._ditto_dataset import DittoDataset, ListDataset


class DittoModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        alpha_aug: float = 0.8,
        device: torch.device | None = None,
    ):
        super().__init__()
        self._bert_name = pretrained_model_name
        self._bert = cast(BertModel, AutoModel.from_pretrained(pretrained_model_name))

        self._device = device or self.__get_device()
        self._alpha_aug = alpha_aug

        # linear layer
        hidden_size = self._bert.config.hidden_size
        self._bert.to(self._device)
        self._final = torch.nn.Linear(hidden_size, 2).to(self._device)
        self.to(self._device)

    @staticmethod
    def __get_device():
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
        elif torch.backends.mps.is_available():
            device = torch.device("mps:0")
        else:
            if not torch.backends.mps.is_built():
                print("PyTorch does not support MPS.")
            else:
                print("This is not Mac OSX or OS X version too old to use MPS.")
        return device

    def forward(self, x1, x2=None):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        # when in training, we're running bert only once on the input
        return self._final(self._bert_encode(x1, x2))

    def _bert_encode(self, x1, x2=None):
        x1 = x1.to(self._device)
        if x2 is not None:
            # MixDA
            x2 = x2.to(self._device)  # (batch_size, seq_len)
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

    def _freeze_bert_layers(self, frozen_layer_count: int = 0):
        if frozen_layer_count < 1:
            return
        for param in self._bert.embeddings.parameters():
            param.requires_grad = False

        for layer in self._bert.encoder.layer[:frozen_layer_count]:  # Freeze encoder layers
            for param in layer.parameters():
                param.requires_grad = False

    def _train_one_epoch(
        self,
        train_iter: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        monitoring_callback: Callable[[str, ...], None] | None = None,
    ):
        try:
            self._bert.train(True)
            self._bert.gradient_checkpointing_enable()
            loss_fn = nn.CrossEntropyLoss().to(self._device)

            total_loss = 0.0
            batch_count = 0
            for i, batch in enumerate(train_iter):
                device_batch = tuple(item.to(self._device) for item in batch)

                optimizer.zero_grad()

                if len(device_batch) == 2:
                    x, y = device_batch
                    prediction = self(x)
                else:
                    x1, x2, y = device_batch
                    prediction = self(x1, x2)

                loss = loss_fn(prediction, y)

                loss.backward()
                optimizer.step()
                scheduler.step()

                step_loss = loss.item()
                total_loss += step_loss
                batch_count = i + 1
                if batch_count % 10 == 0 and monitoring_callback is not None:
                    avg_loss = total_loss / batch_count
                    monitoring_callback(
                        "processed batch no %d: loss=%.4f, avg loss=%.4f",
                        batch_count,
                        step_loss,
                        avg_loss
                    )
                del loss

            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            monitoring_callback("epoch avg loss=%.4f", avg_loss)
        finally:
            self._bert.train(False)

    def evaluate(self, data_loader: DataLoader, threshold: float | None = None):
        all_y = []
        all_probs = []
        with torch.no_grad():
            for batch in data_loader:
                x, y = tuple(vec.to(self._device) for vec in batch)
                logits = self(x)
                probs = logits.softmax(dim=1)[:, 1]
                all_probs += probs.cpu().numpy().tolist()
                all_y += y.cpu().numpy().tolist()

        if threshold is not None:
            pred = [1 if p > threshold else 0 for p in all_probs]
            f1 = metrics.f1_score(all_y, pred)
            return f1, threshold
        else:
            best_th = 0.5
            f1 = 0.0  # metrics.f1_score(all_y, all_p)

            for th in np.arange(0.0, 1.0, 0.05):
                pred = [1 if p > th else 0 for p in all_probs]
                new_f1 = metrics.f1_score(all_y, pred)
                if new_f1 > f1:
                    f1 = new_f1
                    best_th = th

            return f1, best_th

    @staticmethod
    def _write_epoch_summary(fmt: str, *args: Any, global_step: int, writer: SummaryWriter, log: logging.Logger, run_tag: str) -> None:
        writer.add_text(f"{run_tag}:train_one_step", fmt % args, global_step)
        log.info(fmt, *args)

    def run_training(self, dataset: DittoDataset, task_name: str, **kwargs: Any):
        batch_size = int(kwargs.get("batch_size", 64))
        epochs = int(kwargs.get("epochs", 20))
        learning_rate = float(kwargs.get("learning_rate", 3e-5))
        log_dir = Path(kwargs.get("log_dir", "./logs"))
        train_proportion = float(kwargs.get("train", 0.6))
        xv_proportion = float(kwargs.get("xv", (1 - train_proportion) / 2))
        test_proportion = float(kwargs.get("test", (1 - train_proportion) / 2))
        save_model = bool(kwargs.get("save_model", False))
        run_tag = str(kwargs.get("tag", "ditto-training"))
        log = logging.getLogger(self.__class__.__name__)
        frozen_layer_count = int(kwargs.get("frozen_layer_count", 0))

        self._freeze_bert_layers(frozen_layer_count)
        log.info("froze first %d BERT layers", frozen_layer_count)

        train_iter, xv_iter, test_iter = dataset.split(
            train_proportion, xv_proportion, test_proportion, batch_size
        )

        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        num_steps = (len(dataset) // batch_size) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_steps
        )

        writer = SummaryWriter(log_dir=str(log_dir.absolute()))
        best_dev_f1 = best_test_f1 = 0.0

        for epoch in range(1, epochs + 1):
            log.info("[%s] - epoch %d - train start", task_name, epoch)
            try:
                write_epoch_summary = partial(
                    self._write_epoch_summary,
                    global_step=epoch,
                    writer=writer,
                    log=log,
                    run_tag=run_tag
                )
                self._train_one_epoch(train_iter, optimizer, scheduler, write_epoch_summary)
            finally:
                log.info("[%s] - epoch: %d - train end", task_name, epoch)

            self.eval()
            dev_f1, best_xv_threshold = self.evaluate(xv_iter)
            test_f1, test_threshold = self.evaluate(test_iter, threshold=best_xv_threshold)

            log.info("[%s] - epoch %d - dev F1=%.4f, test F1=%.4f", task_name, epoch, dev_f1, test_f1)

            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_test_f1 = test_f1
                if save_model:
                    # create the directory if not exist
                    directory = os.path.join(log_dir, task_name)
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    # save the checkpoints for each component
                    ckpt_path = os.path.join(log_dir, task_name, "model.pt")
                    ckpt = {
                        "model": self.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                    }
                    torch.save(ckpt, ckpt_path)

            print(
                f"epoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}"
            )

            # logging
            scalars = {"f1": dev_f1, "t_f1": test_f1}
            writer.add_scalars(run_tag, scalars, epoch)

        writer.close()
