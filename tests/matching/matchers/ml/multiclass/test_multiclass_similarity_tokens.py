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
