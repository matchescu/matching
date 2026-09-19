import random
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch

from matchescu.matching.matchers.ml.multiclass import MultiClassModule
from matchescu.matching.matchers.ml.multiclass.training import (
    AsymmetricMultiClassDataset,
)
from matchescu.matching.matchers.ml.torch import set_random_seed
from matchescu.matching.matchers.ml.training import __main__ as training


@pytest.fixture
def benchmark():
    split = Mock(
        to_comparison_labels=Mock(return_value=([], np.array([0, 1, 1, 2] * 8)))
    )
    return SimpleNamespace(
        name="seed-test",
        id_table=Mock(),
        splits={name: split for name in ("train_split", "valid_split", "test_split")},
    )


@pytest.mark.parametrize("seed", [42, 17])
@pytest.mark.parametrize("draw", [random.random, np.random.random, torch.rand])
def test_dataset_seed_reproduces_rng_when_constructed_directly(benchmark, seed, draw):
    values = []
    for _ in range(2):
        AsymmetricMultiClassDataset(
            benchmark.id_table,
            benchmark.splits["train_split"],
            Mock(return_value={"input_ids": [7]}),
            random_seed=seed,
        )
        values.append(draw(1).item() if draw is torch.rand else draw())

    assert values[0] == values[1]


@pytest.fixture
def seed_calls(monkeypatch):
    seed = Mock(wraps=torch.manual_seed)
    monkeypatch.setattr(torch, "manual_seed", seed)
    return seed


@pytest.fixture
def native_runs(monkeypatch, benchmark, make_params, tmp_path, seed_calls):
    runs = []
    trainer = Mock()
    trainer.return_value.run_training.side_effect = (
        lambda model, loader, *_: runs.append(
            (model.classifier.state_dict(), list(loader.sampler))
        )
    )
    monkeypatch.setitem(
        training._TRAINER_MAPPINGS,
        trainer,
        (MultiClassModule, AsymmetricMultiClassDataset),
    )
    for seed in (11, 29):
        set_random_seed(seed)
        training.train_on_benchmark_data(
            tmp_path,
            "seed-test",
            trainer,
            MagicMock(),
            benchmark,
            Mock(return_value={"input_ids": [7]}),
            make_params(),
        )
    return runs


def test_native_run_reproduces_model_initialization(native_runs):
    first, second = [run[0] for run in native_runs]

    assert all(torch.equal(first[name], second[name]) for name in first)


def test_native_run_reproduces_sampler_draws(native_runs):
    assert native_runs[0][1] == native_runs[1][1]


def test_native_run_seeds_once_when_building_three_datasets(native_runs, seed_calls):
    assert [call.args[0] for call in seed_calls.call_args_list] == [11, 42, 29, 42]
