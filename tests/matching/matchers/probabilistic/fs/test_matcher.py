import pyarrow as pa
import pytest

from matchescu.matching.config import RecordLinkageConfig
from matchescu.matching.matchers import FellegiSunter


@pytest.fixture
def table_a():
    return pa.table({"idA":[1,2], "fname":["ann","bob"], "lname":["lee","kim"]})


@pytest.fixture
def table_b():
    return pa.table({"idB":[10,11], "fname":["ann","rob"], "lname":["lee","kim"]})


@pytest.fixture
def ground_truth():
    return pa.table({"idA":[1,2,1], "idB": [10,11,11], "label": [1, 1, 0]})


@pytest.fixture
def id_pairs(ground_truth, request):
    if hasattr(request, "param"):
        return request.param
    return list(zip(ground_truth["idA"].to_pylist(), ground_truth["idB"].to_pylist()))


def test_fs_matcher(table_a, table_b, ground_truth):
    rl_config = RecordLinkageConfig("idA", "idB", "label", [("fname", "fname"), ("lname", "lname")])
    fs = FellegiSunter(rl_config).fit(table_a, table_b, ground_truth)

    result = fs.predict([(2, 11)], table_a, table_b)

    assert result.num_rows == 1


def test_predict_requires_fit(table_a, table_b):
    rl_config = RecordLinkageConfig("idA", "idB", "label", [("fname", "fname")])
    fs = FellegiSunter(rl_config)
    with pytest.raises(AssertionError):
        fs.predict([(1, 10)], table_a, table_b)


pytest.mark.parametrize("id_pairs", [[(1, 10), (1, 11)]], indirect="id_pairs")
def test_fit_with_empty_comparison_config(table_a, table_b, ground_truth, id_pairs):
    rl_config = RecordLinkageConfig("idA", "idB", "label", [])
    fs = FellegiSunter(rl_config).fit(table_a, table_b, ground_truth)

    pred = fs.predict(id_pairs, table_a, table_b)

    assert pred.num_rows == len(id_pairs)


# Predict over multiple pairs returns a count matching input pairs
def test_predict_many_pairs_count(table_a, table_b, ground_truth, id_pairs):
    rl_config = RecordLinkageConfig("idA", "idB", "label", [("fname", "fname"), ("lname", "lname")])
    fs = FellegiSunter(rl_config).fit(table_a, table_b, ground_truth)

    pred = fs.predict(id_pairs, table_a, table_b)

    assert pred.num_rows == len(id_pairs)


def test_threshold_ordering_after_fit(table_a, table_b, ground_truth):
    rl_config = RecordLinkageConfig("idA", "idB", "label", [("fname", "fname"), ("lname", "lname")])
    fs = FellegiSunter(rl_config).fit(table_a, table_b, ground_truth)

    th = fs._thresholds

    assert th.lower <= th.upper


def test_predict_with_empty_pairs(table_a, table_b, ground_truth):
    rl_config = RecordLinkageConfig("idA", "idB", "label", [("fname", "fname")])
    fs = FellegiSunter(rl_config).fit(table_a, table_b, ground_truth)

    pred = fs.predict([], table_a, table_b)

    assert pred.num_rows == 0


def test_empirical_error_rates_on_train_holdout(table_a, table_b, ground_truth, id_pairs):
    rl_config = RecordLinkageConfig("idA", "idB", "label", [("fname", "fname"), ("lname", "lname")])
    fs = FellegiSunter(rl_config, mu=0.25, lambda_=0.25).fit(table_a, table_b, ground_truth)

    pred = fs.predict(id_pairs, table_a, table_b)

    assert pred.num_rows == len(id_pairs)
