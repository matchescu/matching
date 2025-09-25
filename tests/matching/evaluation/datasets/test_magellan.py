from functools import partial

import pytest

from matchescu.matching.evaluation.datasets._magellan import MagellanDataset


@pytest.fixture
def sut(ag_dir):
    return MagellanDataset(ag_dir)


def test_load_left(sut, ag_traits, ag_id_factory):
    ds = sut.load_left(ag_traits, partial(ag_id_factory, source="tableA"))

    assert ds.left_source == "tableA"
    assert len(ds.id_table) == 1363


def test_load_right(sut, ag_traits, ag_id_factory):
    ds = sut.load_right(ag_traits, partial(ag_id_factory, source="tableB"))

    assert ds.right_source == "tableB"
    assert len(ds.id_table) == 3226


def test_load_splits(sut, ag_traits, ag_id_factory):
    sut.load_left(ag_traits, partial(ag_id_factory, source="tableA"))
    sut.load_right(ag_traits, partial(ag_id_factory, source="tableB"))

    sut.load_splits()

    assert sut.left_source == "tableA"
    assert sut.right_source == "tableB"
    assert len(sut.id_table) == 1363 + 3226
    assert len(sut.train_split.comparison_space) == 6874
    assert len(sut.train_split.ground_truth) == 699
    assert len(sut.valid_split.comparison_space) == 2293
    assert len(sut.valid_split.ground_truth) == 234
    assert len(sut.test_split.comparison_space) == 2293
    assert len(sut.test_split.ground_truth) == 234
