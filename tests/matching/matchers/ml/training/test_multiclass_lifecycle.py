from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from matchescu.matching.matchers.ml.core import AdditionalModelInfo
from matchescu.matching.matchers.ml.multiclass._params import MultiClassTrainingParams
from matchescu.matching.matchers.ml.multiclass.training._evaluator import (
    TrainingEvaluator,
)
from matchescu.matching.matchers.ml.multiclass.training._trainer import (
    MultiClassTrainer,
)
from matchescu.matching.matchers.ml.training import BaseTrainer


def _make_trainer(tmp_path: Path) -> MultiClassTrainer:
    params = MultiClassTrainingParams(epochs=2, batch_size=2, learning_rate=1e-3)
    return MultiClassTrainer("lifecycle", params, model_dir=tmp_path)


def test_train_one_epoch_adds_train_metrics(monkeypatch):
    """The unsampled training population is evaluated after each epoch."""
    trainer = _make_trainer(Path("/tmp"))
    train_dataset = MagicMock()
    train_dataset.get_data_loader.return_value = []
    train_iter = MagicMock()
    train_iter.dataset = train_dataset

    def fake_super_train_one_epoch(
        self, epoch, device, model, train_iter, optimizer, scheduler
    ):
        return {"average_loss": 0.5}

    monkeypatch.setattr(BaseTrainer, "_train_one_epoch", fake_super_train_one_epoch)
    monkeypatch.setattr(
        TrainingEvaluator,
        "_measure",
        staticmethod(lambda *args, **kwargs: {"mcc": 0.75, "order_acc": 0.25}),
    )

    result = trainer._train_one_epoch(
        1, torch.device("cpu"), MagicMock(), train_iter, MagicMock(), MagicMock()
    )

    assert result["average_loss"] == pytest.approx(0.5)
    assert result["train_mcc"] == pytest.approx(0.75)
    assert result["train_order_acc"] == pytest.approx(0.25)
    train_dataset.get_data_loader.assert_called_once_with(
        trainer._params.batch_size, shuffle=False, sampler=None
    )


def test_run_training_requires_save_model_with_evaluator():
    """An evaluator without checkpointing is a configuration error."""
    trainer = _make_trainer(Path("/tmp"))
    evaluator = MagicMock()
    evaluator.reset_selector = MagicMock()

    with pytest.raises(ValueError, match="save_model=True"):
        trainer.run_training(MagicMock(), MagicMock(), evaluator, save_model=False)


def test_run_training_without_evaluator_calls_super_only(monkeypatch):
    """Training-only mode delegates to the base trainer unchanged."""
    trainer = _make_trainer(Path("/tmp"))
    super_called = []

    def fake_super_run_training(self, model, training_data, evaluator, save_model):
        super_called.append((evaluator, save_model))

    monkeypatch.setattr(BaseTrainer, "run_training", fake_super_run_training)

    trainer.run_training(MagicMock(), MagicMock(), None, save_model=False)

    assert super_called == [(None, False)]


def test_run_training_restores_best_checkpoint_and_evaluates_test_once(
    tmp_path, monkeypatch
):
    """After training, the best dev checkpoint is restored and test is run once."""
    trainer = _make_trainer(tmp_path)
    model = MagicMock()
    model.state_dict.return_value = {"weight": 1.0}
    optimizer = MagicMock()
    optimizer.state_dict.return_value = {"state": 1}
    scheduler = MagicMock()
    scheduler.state_dict.return_value = {"state": 2}
    monkeypatch.setattr(trainer, "_create_optimizer", lambda m: optimizer)
    monkeypatch.setattr(trainer, "_create_scheduler", lambda ds, opt: scheduler)

    xv = MagicMock()
    test = MagicMock()
    evaluator = TrainingEvaluator("lifecycle", xv, test, tmp_path)

    final_config = {"test_mcc": 0.85}
    run_model_calls = []

    def fake_run_model(self, model, data, best_config):
        run_model_calls.append(dict(best_config))
        return True, final_config

    monkeypatch.setattr(TrainingEvaluator, "_run_model", fake_run_model)

    additional_info = AdditionalModelInfo(
        hyperparameters=trainer._params,
        best_config={"dev_mcc": 0.9},
    )

    def fake_super_run_training(self, model, training_data, evaluator, save_model):
        # Simulate two epochs: epoch 2 is the best.
        evaluator._selector._best = 0.9
        evaluator._selector._best_epoch = 2
        trainer._save_checkpoint(2, model, optimizer, scheduler, additional_info)

    monkeypatch.setattr(BaseTrainer, "run_training", fake_super_run_training)

    training_data = MagicMock()
    training_data.dataset = MagicMock()

    trainer.run_training(model, training_data, evaluator, save_model=True)

    model.load_state_dict.assert_called_once_with({"weight": 1.0})
    optimizer.load_state_dict.assert_called_once_with({"state": 1})
    scheduler.load_state_dict.assert_called_once_with({"state": 2})
    assert len(run_model_calls) == 1
    assert run_model_calls[0]["is_evaluating"] is True
    assert run_model_calls[0]["dev_mcc"] == 0.9

    ckpt_path = tmp_path / "lifecycle" / "model.pt"
    assert ckpt_path.exists()
    updated = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert updated["epoch"] == 2
    assert updated["additional_info"]["best_config"]["test_mcc"] == 0.85


def test_run_training_fails_when_no_checkpoint_saved(tmp_path, monkeypatch):
    """If no epoch improved, there is nothing to restore and test."""
    trainer = _make_trainer(tmp_path)
    monkeypatch.setattr(trainer, "_create_optimizer", lambda m: MagicMock())
    monkeypatch.setattr(trainer, "_create_scheduler", lambda ds, opt: MagicMock())

    evaluator = TrainingEvaluator("lifecycle", MagicMock(), MagicMock(), tmp_path)

    def fake_super_run_training(self, model, training_data, evaluator, save_model):
        # No improvement, no checkpoint written.
        pass

    monkeypatch.setattr(BaseTrainer, "run_training", fake_super_run_training)

    with pytest.raises(RuntimeError, match="no best checkpoint"):
        trainer.run_training(MagicMock(), MagicMock(), evaluator, save_model=True)


def test_run_training_fails_on_checkpoint_epoch_mismatch(tmp_path, monkeypatch):
    """A stale or mismatched checkpoint must not be evaluated."""
    trainer = _make_trainer(tmp_path)
    monkeypatch.setattr(trainer, "_create_optimizer", lambda m: MagicMock())
    monkeypatch.setattr(trainer, "_create_scheduler", lambda ds, opt: MagicMock())

    evaluator = TrainingEvaluator("lifecycle", MagicMock(), MagicMock(), tmp_path)
    evaluator._selector._best_epoch = 3

    ckpt_path = tmp_path / "lifecycle" / "model.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": 1,
            "model": {},
            "optimizer": {},
            "scheduler": {},
            "additional_info": {"best_config": {}},
        },
        ckpt_path,
    )

    def fake_super_run_training(self, model, training_data, evaluator, save_model):
        pass

    monkeypatch.setattr(BaseTrainer, "run_training", fake_super_run_training)

    with pytest.raises(RuntimeError, match="does not match best epoch"):
        trainer.run_training(MagicMock(), MagicMock(), evaluator, save_model=True)
