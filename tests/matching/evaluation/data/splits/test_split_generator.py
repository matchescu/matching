import pytest

from matchescu.matching.evaluation.data.splits import SplitGenerator

SCARCE_CLUSTERS = 2
SCARCE_BRIDGES = 1
RATIO_CLUSTERS = 10
EXPECTED_MATCHES = {"train": 6, "dev": 2, "test": 2}


@pytest.fixture
def generator_class():
    return SplitGenerator


def test_generate_raises_when_class_has_fewer_than_minimum_pairs(
    make_generator, make_pair_data
):
    generator = make_generator(make_pair_data(SCARCE_CLUSTERS, SCARCE_BRIDGES))

    with pytest.raises(ValueError, match="MIN_PER_CLASS"):
        generator.generate()


def test_get_splits_distributes_matches_by_split_ratio(make_generator, make_pair_data):
    generator = make_generator(make_pair_data(RATIO_CLUSTERS, 0)).generate()

    splits = generator.get_splits()

    assert {k: len(v.matcher_labels) for k, v in splits.items()} == EXPECTED_MATCHES
