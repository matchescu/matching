import logging
import os
import random

import numpy as np
import pytest
import torch

from matchescu.matching.matchers.ml.core import ModelTrainingParams
from matchescu.matching.matchers.ml.torch import set_random_seed

CUBLAS_VAR = "CUBLAS_WORKSPACE_CONFIG"
HASH_SEED_VAR = "PYTHONHASHSEED"
SEED = 42


@pytest.fixture(autouse=True)
def restore_global_state(monkeypatch):
    monkeypatch.setenv(HASH_SEED_VAR, "0")
    monkeypatch.delenv(CUBLAS_VAR, raising=False)
    cudnn = torch.backends.cudnn
    saved = (
        cudnn.deterministic,
        cudnn.benchmark,
        cudnn.conv.fp32_precision,
        torch.backends.cuda.matmul.fp32_precision,
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    yield
    cudnn.deterministic, cudnn.benchmark = saved[:2]
    cudnn.conv.fp32_precision = saved[2]
    torch.backends.cuda.matmul.fp32_precision = saved[3]
    torch.use_deterministic_algorithms(saved[4], warn_only=saved[5])


def _draw():
    return (
        random.random(),
        np.random.random(),
        torch.rand(3).tolist(),
    )


def test_set_random_seed_reproduces_draws_when_seed_repeats():
    set_random_seed(SEED)
    first = _draw()
    set_random_seed(SEED)

    assert _draw() == first


def test_set_random_seed_changes_draws_when_seed_differs():
    set_random_seed(SEED)
    first = _draw()
    set_random_seed(SEED + 1)

    assert _draw() != first


def test_set_random_seed_enables_deterministic_algorithms():
    set_random_seed(SEED)

    assert torch.are_deterministic_algorithms_enabled()


def test_set_random_seed_only_warns_when_operation_is_nondeterministic():
    set_random_seed(SEED)

    assert torch.is_deterministic_algorithms_warn_only_enabled()


def test_set_random_seed_makes_cudnn_deterministic():
    torch.backends.cudnn.deterministic = False
    set_random_seed(SEED)

    assert torch.backends.cudnn.deterministic


def test_set_random_seed_disables_cudnn_benchmarking():
    torch.backends.cudnn.benchmark = True
    set_random_seed(SEED)

    assert not torch.backends.cudnn.benchmark


def test_set_random_seed_disables_tf32_in_cudnn_convolutions():
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    set_random_seed(SEED)

    assert torch.backends.cudnn.conv.fp32_precision == "ieee"


def test_set_random_seed_disables_tf32_in_matmul():
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    set_random_seed(SEED)

    assert torch.backends.cuda.matmul.fp32_precision == "ieee"


def test_set_random_seed_sets_cublas_workspace_config_when_absent():
    set_random_seed(SEED)

    assert os.environ[CUBLAS_VAR] == ":4096:8"


def test_set_random_seed_keeps_cublas_workspace_config_when_present(monkeypatch):
    monkeypatch.setenv(CUBLAS_VAR, ":16:8")
    set_random_seed(SEED)

    assert os.environ[CUBLAS_VAR] == ":16:8"


def test_set_random_seed_warns_when_python_hash_seed_is_unset(monkeypatch, caplog):
    monkeypatch.delenv(HASH_SEED_VAR)
    with caplog.at_level(logging.WARNING):
        set_random_seed(SEED)

    assert HASH_SEED_VAR in caplog.text


def test_set_random_seed_is_silent_when_python_hash_seed_is_set(caplog):
    with caplog.at_level(logging.WARNING):
        set_random_seed(SEED)

    assert HASH_SEED_VAR not in caplog.text


def test_training_params_default_to_seed_42():
    assert ModelTrainingParams().seed == SEED
