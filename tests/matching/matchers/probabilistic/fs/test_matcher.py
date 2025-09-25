import numpy as np
import pytest

from matchescu.extraction import Traits
from matchescu.matching.config import RecordLinkageConfig, AttrCmpConfig
from matchescu.matching.evaluation.datasets import MagellanDataset
import matchescu.matching.matchers as m
from matchescu.matching.similarity._string import BucketedLevenshteinSimilarity
from matchescu.typing import EntityReferenceIdentifier


@pytest.fixture
def rl_config(request) -> RecordLinkageConfig:
    col_config = [("fname", "fname"), ("lname", "lname")]
    if hasattr(request, "param") and isinstance(request.param, list):
        col_config = request.param
    return RecordLinkageConfig("idA", "idB", "label", col_config)


@pytest.fixture
def amazon_google_config():
    possible_values = np.linspace(0.0, 1.0, 11).tolist()
    sim = BucketedLevenshteinSimilarity(True)
    attr_comparisons = [
        AttrCmpConfig("title", "title", possible_values, sim),
        AttrCmpConfig("manufacturer", "manufacturer", possible_values, sim),
    ]
    return RecordLinkageConfig("id", "id", "label", attr_comparisons)


@pytest.fixture
def amazon_google(data_dir, amazon_google_config):
    ret = MagellanDataset(data_dir / "amazon_google_exp_data")

    ret.load_left(
        Traits().int(["id"]).string(["title", "manufacturer"]).currency(["price"]),
        lambda records: EntityReferenceIdentifier(
            records[0][amazon_google_config.left_id], ret.left_source
        ),
    )
    ret.load_right(
        Traits().int(["id"]).string(["title", "manufacturer"]).currency(["price"]),
        lambda records: EntityReferenceIdentifier(
            records[0][amazon_google_config.right_id], ret.right_source
        ),
    )
    ret.load_splits()

    return ret


def test_amazon_google(amazon_google, amazon_google_config):
    fs = m.FellegiSunter(amazon_google_config)
    fs = fs.fit(
        amazon_google.train_split.comparison_space,
        amazon_google.id_table,
        amazon_google.train_split.ground_truth,
    )

    result = fs.predict(
        amazon_google.test_split.comparison_space, amazon_google.id_table
    )

    assert result is not None

    matches = set(
        (
            EntityReferenceIdentifier(left_id, amazon_google.left_source),
            EntityReferenceIdentifier(right_id, amazon_google.right_source),
        )
        for left_id, right_id in result
    )

    from pyresolvemetrics import precision, recall, f1

    assert precision(amazon_google.test_split.ground_truth, matches) > 0
    assert recall(amazon_google.test_split.ground_truth, matches) > 0
    assert f1(amazon_google.test_split.ground_truth, matches) > 0
