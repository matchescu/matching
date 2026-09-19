from itertools import permutations
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from matchescu.matching.matchers.ml.multiclass.training import (
    AsymmetricMultiClassDataset,
)
from matchescu.matching.matchers.ml.training.__main__ import get_benchmark_data_loaders


@pytest.mark.parametrize("validation_name", ["valid_split", "dev_split"])
@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_multiclass_loaders_bind_roles_when_splits_are_unordered(
    validation_name, order, make_params
):
    names = ["train_split", validation_name, "test_split"]
    splits = [
        Mock(to_comparison_labels=Mock(return_value=([], np.array([label]))))
        for label in range(3)
    ]
    benchmark = SimpleNamespace(
        id_table=Mock(), splits={names[i]: splits[i] for i in order}
    )
    tokenizer = Mock(return_value={"input_ids": [7]})

    loaders = get_benchmark_data_loaders(
        AsymmetricMultiClassDataset, benchmark, tokenizer, make_params()
    )

    assert [loader.dataset._labels.tolist() for loader in loaders] == [[0], [1], [2]]
