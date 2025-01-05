from typing import Iterator, Hashable, Callable
from unittest.mock import MagicMock, call

import numpy as np
import pytest
from matchescu.matching.ml.datasets._base import BaseDataSet
from matchescu.typing import EntityReference


class DataSetStub(BaseDataSet):
    def __init__(
        self,
        comparison_tuples=None,
        ground_truth=None,
        sample_factory=None,
        left_id=None,
        right_id=None,
    ):
        super().__init__(ground_truth or set(), sample_factory, left_id, right_id)
        self.__comparison_tuples = comparison_tuples or []

    def _comparison_tuples(self) -> Iterator[tuple[EntityReference, EntityReference]]:
        return iter(self.__comparison_tuples)


def _callable_mock(request, name, signature):
    if not hasattr(request, "param"):
        return None
    return (
        MagicMock(spec=signature, name=name, side_effect=request.param)
        if isinstance(request.param, Exception)
        else MagicMock(spec=signature, name=name, return_value=request.param)
    )


@pytest.fixture
def ground_truth(request):
    if not hasattr(request, "param"):
        return set()
    return set(x for x in request.param)


@pytest.fixture
def sample_factory(request):
    return _callable_mock(
        request, "sample_factory", Callable[[EntityReference, EntityReference], dict]
    )


@pytest.fixture
def left_id(request):
    return _callable_mock(request, "left_id", Callable[[EntityReference], Hashable])


@pytest.fixture
def right_id(request):
    return _callable_mock(request, "right_id", Callable[[EntityReference], Hashable])


@pytest.fixture
def dataset(request, ground_truth, sample_factory, left_id, right_id):
    return (
        DataSetStub(
            comparison_tuples=request.param,
            ground_truth=ground_truth,
            sample_factory=sample_factory,
            left_id=left_id,
            right_id=right_id,
        )
        if hasattr(request, "param")
        else DataSetStub(
            ground_truth=ground_truth,
            sample_factory=sample_factory,
            left_id=left_id,
            right_id=right_id,
        )
    )


def test_feature_matrix_raises_value_error_by_default(dataset):
    with pytest.raises(ValueError) as verr:
        _ = dataset.feature_matrix

    assert str(verr.value) == "comparison matrix was not computed"


def test_target_vector_raises_value_error_by_default(dataset):
    with pytest.raises(ValueError) as verr:
        _ = dataset.target_vector

    assert str(verr.value) == "comparison matrix was not computed"


def test_training_data_raises_value_error_by_default(dataset):
    with pytest.raises(ValueError) as verr:
        _ = dataset.training_data

    assert str(verr.value) == "comparison matrix was not computed"


@pytest.mark.parametrize(
    "dataset,ground_truth,left_id,right_id,expected",
    [
        ([(("a",), ("b",))], {(1, 2)}, 1, 2, [1]),
        ([(("a",), ("b",))], {(1, 2)}, 1, 3, [0]),
    ],
    indirect=["dataset", "ground_truth", "left_id", "right_id"],
)
def test_target_vector_has_expected_values(
    dataset, ground_truth, left_id, right_id, expected
):
    dataset.cross_sources()

    actual = dataset.target_vector

    assert actual == expected
    assert left_id.call_args == call(("a",))
    assert right_id.call_args == call(("b",))


@pytest.mark.parametrize(
    "dataset,ground_truth,sample_factory,expected",
    [
        ([(("a",), ("b",))], {(1, 2)}, {"first_col": "a==b"}, [("a==b",)]),
        ([(("a",), ("b",))], {(1, 2)}, {"first_col": "a!=b"}, [("a!=b",)]),
    ],
    indirect=["dataset", "ground_truth", "sample_factory"],
)
def test_feature_matrix_has_expected_values(
    dataset, ground_truth, sample_factory, expected
):
    dataset.cross_sources()

    actual = dataset.feature_matrix

    assert actual == expected


@pytest.mark.parametrize(
    "dataset,ground_truth",
    [([(("a",), ("b",))], {(1, 2)})],
    indirect=True,
)
def test_not_specifying_id_funcs_does_not_create_target_col(dataset, ground_truth):
    dataset.cross_sources()

    with pytest.raises(ValueError) as verr:
        _ = dataset.target_vector

    assert str(verr.value) == "target vector was not computed"


@pytest.mark.parametrize("dataset", [[(("a",), ("b",))]], indirect=True)
def test_single_transform(dataset):
    expected = ("another", "tuple", "than", "the", "input")
    transform = MagicMock(return_value=expected)
    dataset.transforms.append(transform)

    dataset.cross_sources()

    assert np.all(
        np.equal(
            dataset.feature_matrix,
            [
                (
                    "another",
                    "tuple",
                    "than",
                    "the",
                    "input",
                    "another",
                    "tuple",
                    "than",
                    "the",
                    "input",
                )
            ],
        )
    )
    assert transform.call_args_list == [call(("a",)), call(("b",))]


@pytest.mark.parametrize("dataset", [[(("a",), ("b",))]], indirect=True)
def test_multiple_transforms(dataset):
    expected_t1 = ("t1_value1", "t1_value2")
    expected = ("t2_value1", "t2_value2")
    transform1 = MagicMock(return_value=expected_t1)
    transform2 = MagicMock(return_value=expected)
    dataset.transforms.append(transform1).append(transform2)

    dataset.cross_sources()

    assert np.all(
        np.equal(
            dataset.feature_matrix,
            [("t2_value1", "t2_value2", "t2_value1", "t2_value2")],
        )
    )
    assert transform1.call_args_list == [call(("a",)), call(("b",))]
    assert transform2.call_args_list == [
        call(("t1_value1", "t1_value2")),
        call(("t1_value1", "t1_value2")),
    ]
