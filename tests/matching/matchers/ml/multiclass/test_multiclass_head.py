import pytest

from matchescu.matching.matchers.ml.multiclass._module import MultiClassModule
from matchescu.matching.matchers.ml.multiclass._types import (
    ArchitectureType,
    HeadType,
)

from .._constants import HIDDEN


@pytest.mark.parametrize(
    "head_type,expected_multiplier",
    [
        (HeadType.NONE, 2),
        (HeadType.DIFF, 4),
    ],
)
def test_classifier_input_size_when_head_type_is(
    make_params, head_type, expected_multiplier
):
    module = MultiClassModule(
        make_params(head_type=head_type, architecture=ArchitectureType.BERT)
    )
    assert module.classifier_input_size == expected_multiplier * HIDDEN
