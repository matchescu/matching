import pickle
from os import unlink, access, F_OK

import pyarrow as pa
import pytest

from matchescu.matching.config import RecordLinkageConfig
import matchescu.matching.matchers as m


@pytest.fixture
def rl_config(request) -> RecordLinkageConfig:
    col_config = [("fname", "fname"), ("lname", "lname")]
    if hasattr(request, "param") and isinstance(request.param, list):
        col_config = request.param
    return RecordLinkageConfig("idA", "idB", "label", col_config)


@pytest.fixture
def table_a():
    return pa.table({"idA": [1, 2], "fname": ["ann", "bob"], "lname": ["lee", "kim"]})


@pytest.fixture
def table_b():
    return pa.table({"idB": [10, 11], "fname": ["ann", "rob"], "lname": ["lee", "kim"]})


@pytest.fixture
def ground_truth():
    return pa.table({"idA": [1, 2, 1], "idB": [10, 11, 11], "label": [1, 1, 0]})


@pytest.fixture
def id_pairs(ground_truth, request):
    if hasattr(request, "param"):
        return request.param
    return list(zip(ground_truth["idA"].to_pylist(), ground_truth["idB"].to_pylist()))


@pytest.fixture
def model_file_name():
    model_file = "model.pkl"

    yield model_file

    if access(model_file, F_OK):
        unlink(model_file)


def test_fs_matcher(table_a, table_b, ground_truth, rl_config):
    fs = m.FellegiSunter(rl_config).fit(table_a, table_b, ground_truth)

    result = fs.predict([(2, 11)], table_a, table_b)

    assert result.num_rows == 1


@pytest.mark.parametrize("rl_config", [[("fname", "fname")]], indirect=True)
def test_predict_requires_fit(table_a, table_b, rl_config):
    fs = m.FellegiSunter(rl_config)
    with pytest.raises(AssertionError):
        fs.predict([(1, 10)], table_a, table_b)


@pytest.mark.parametrize(
    "id_pairs,rl_config", [([(1, 10), (1, 11)], [])], indirect=True
)
def test_fit_with_empty_comparison_config(
    table_a, table_b, ground_truth, id_pairs, rl_config
):
    fs = m.FellegiSunter(rl_config).fit(table_a, table_b, ground_truth)

    pred = fs.predict(id_pairs, table_a, table_b)

    assert pred.num_rows == len(id_pairs)


def test_predict_many_pairs_count(table_a, table_b, ground_truth, id_pairs, rl_config):
    fs = m.FellegiSunter(rl_config).fit(table_a, table_b, ground_truth)

    pred = fs.predict(id_pairs, table_a, table_b)

    assert pred.num_rows == len(id_pairs)


def test_threshold_ordering_after_fit(table_a, table_b, ground_truth, rl_config):
    fs = m.FellegiSunter(rl_config).fit(table_a, table_b, ground_truth)

    th = fs._thresholds

    assert th.lower <= th.upper


@pytest.mark.parametrize("rl_config", [[("fname", "fname")]], indirect=True)
def test_predict_with_empty_pairs(table_a, table_b, ground_truth, rl_config):
    fs = m.FellegiSunter(rl_config).fit(table_a, table_b, ground_truth)

    pred = fs.predict([], table_a, table_b)

    assert pred.num_rows == 0


def test_empirical_error_rates_on_train_holdout(
    table_a, table_b, ground_truth, id_pairs, rl_config
):
    fs = m.FellegiSunter(rl_config, mu=0.25, lambda_=0.25).fit(
        table_a, table_b, ground_truth
    )

    pred = fs.predict(id_pairs, table_a, table_b)

    assert pred.num_rows == len(id_pairs)


def test_save_raises_assertion_error_if_not_fit(rl_config):
    fs = m.FellegiSunter(rl_config)

    with pytest.raises(AssertionError) as err_proxy:
        fs.save("some file")

    assert str(err_proxy.value) == "model not trained. run fit() before saving."


def test_save_pickles_train_params(
    table_a, table_b, ground_truth, rl_config, monkeypatch, model_file_name
):
    fs = m.FellegiSunter(rl_config).fit(table_a, table_b, ground_truth)
    captured = {}

    def fake_dump(obj, file, *_, **__):
        captured[file.name] = obj

    monkeypatch.setattr(pickle, "dump", fake_dump)

    fs.save(model_file_name)

    assert model_file_name in captured
    assert isinstance(captured[model_file_name], tuple)
    assert len(captured[model_file_name]) == 5
    assert captured[model_file_name] == (
        fs.parameters,
        fs.thresholds,
        fs.mu,
        fs.lambda_,
        fs.config,
    )
