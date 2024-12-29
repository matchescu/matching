import pytest

from matchescu.matching.blocking import BlockEngine


@pytest.fixture
def engine(ds1, ds2, ds_identifier):
    return BlockEngine().add_source(ds1, ds_identifier).add_source(ds2, ds_identifier)


@pytest.mark.parametrize("overlap,expected", [(0.6, 0), (0.5, 1)])
def test_blocking_engine_returns_expected_comparisons(engine, overlap, expected):
    engine.create_blocks_with_overlap(0, overlap)

    pairs = list(engine.candidate_pairs())

    assert len(pairs) == expected


def test_blocking_engine_pair_completeness(engine):
    engine.create_blocks_with_overlap(0, 0.5)

    metrics = engine.calculate_metrics(ground_truth={(33053, 10221960)})

    assert metrics.pair_completeness == 1


def test_blocking_engine_pair_completeness_empty_ground_truth(engine):
    engine.create_blocks_with_overlap(0, 0.5)

    metrics = engine.calculate_metrics(ground_truth=set())

    assert metrics.pair_completeness == 0


def test_blocking_engine_pair_quality(engine):
    engine.create_blocks_with_overlap(0, 0.5)

    metrics = engine.calculate_metrics(ground_truth={(33053, 10221960)})

    assert metrics.pair_quality == 1


def test_blocking_engine_pair_quality_no_comparisons(engine):
    metrics = engine.calculate_metrics(ground_truth={(33053, 10221960)})

    assert metrics.pair_quality == 0


def test_blocking_engine_reduction_ratio(engine):
    engine.create_blocks_with_overlap(0, 0.5)

    metrics = engine.calculate_metrics(ground_truth={(33053, 10221960)})

    assert metrics.reduction_ratio == 0


@pytest.mark.parametrize("gt", [{(33053, 10221960)}, set()])
def test_blocking_engine_reduction_ratio_no_data(gt):
    engine = BlockEngine()

    metrics = engine.calculate_metrics(ground_truth=gt)

    # complete data loss compared to any ground truth
    assert metrics.reduction_ratio == 1
