from unittest.mock import Mock

import pytest
import torch
from matchescu.typing import EntityReference

from matchescu.matching.matchers.ml.multiclass._similarity import MultiClassSimilarity


@pytest.fixture
def similarity_model(local_tokenizer, monkeypatch):
    similarity = MultiClassSimilarity(local_tokenizer)
    model = Mock(return_value=torch.tensor([[0.0, 1.0, 0.0]]))
    monkeypatch.setattr(similarity, "_MultiClassSimilarity__model", model)
    return similarity, model


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ({}, {}, [[-1]]),
        ({"name": "Acme"}, {}, [[1]]),
        ({"name": "Acme", "city": "NYC"}, {"name": "Other"}, [[1, 5, 10]]),
    ],
)
def test_similarity_passes_token_indexes_when_col_count_varies(
    similarity_model, ref_pair, left, right, expected
):
    similarity, model = similarity_model

    similarity(
        EntityReference(ref_pair[0].id, left), EntityReference(ref_pair[1].id, right)
    )

    assert model.call_args.kwargs["col_positions"].tolist() == expected


def test_similarity_passes_boundary_mask_when_tokenizing(similarity_model, ref_pair):
    similarity, model = similarity_model

    similarity(*ref_pair)

    encoding = model.call_args.kwargs
    expected = torch.isin(encoding["input_ids"], torch.tensor([0, 2, 3])).long()
    assert torch.equal(encoding["special_tokens_mask"], expected)


def test_similarity_value_mask_matches_dataset_when_values_are_empty(
    similarity_model, make_dataset, ref_pair
):
    similarity, model = similarity_model
    left = EntityReference(ref_pair[0].id, {"name": "", "city": "NYC"})
    dataset = make_dataset(pairs=[(left, ref_pair[1])])

    similarity(left, ref_pair[1])

    actual = model.call_args.kwargs["value_mask"]
    assert torch.equal(actual, dataset[0][0]["value_mask"].unsqueeze(0))
