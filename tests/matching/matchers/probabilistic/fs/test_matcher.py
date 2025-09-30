from functools import partial

import pytest

from matchescu.extraction import Traits
from matchescu.matching.config import RecordLinkageConfig, AttrCmpConfig
from matchescu.matching.evaluation.datasets import MagellanDataset
import matchescu.matching.matchers as m
from matchescu.matching.similarity._numeric import BucketedNorm
from matchescu.matching.similarity._string import BucketedJaccard
from matchescu.typing import EntityReferenceIdentifier


@pytest.fixture
def rl_config(request) -> RecordLinkageConfig:
    col_config = [("fname", "fname"), ("lname", "lname")]
    if hasattr(request, "param") and isinstance(request.param, list):
        col_config = request.param
    return RecordLinkageConfig("idA", "idB", "label", col_config)


@pytest.fixture
def amazon_google_config():
    strsim = BucketedJaccard(False, 5)
    price_sim = BucketedNorm(
        {
            0.0: 1.0,
            0.1: 0.9,
            0.11: 0.8,
            0.12: 0.7,
            0.13: 0.6,
            0.14: 0.5,
            0.15: 0.4,
            0.151: 0.3,
            0.152: 0.2,
            0.153: 0.1,
        }
    )
    attr_comparisons = [
        AttrCmpConfig("title", "title", strsim.agreement_levels, strsim),
        AttrCmpConfig("manufacturer", "manufacturer", strsim.agreement_levels, strsim),
        AttrCmpConfig("price", "price", price_sim.agreement_levels, price_sim),
    ]
    extraction_traits = (
        Traits().int(["id"]).string(["title", "manufacturer"]).currency(["price"])
    )
    return RecordLinkageConfig(extraction_traits, extraction_traits, attr_comparisons)


@pytest.fixture
def id_col():
    return "id"


@pytest.fixture
def amazon_google(data_dir, amazon_google_config, id_col):
    ret = MagellanDataset(data_dir / "amazon_google_exp_data")

    def ref_id(records, source):
        return EntityReferenceIdentifier(records[0][id_col], source)

    ret.load_left(
        amazon_google_config.left_traits, partial(ref_id, source=ret.left_source)
    )
    ret.load_right(
        amazon_google_config.right_traits, partial(ref_id, source=ret.right_source)
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
    assert result
    assert fs.clerical_review

    def _get_ref_ids(
        id_pair: tuple,
    ) -> tuple[EntityReferenceIdentifier, EntityReferenceIdentifier]:
        return tuple(
            EntityReferenceIdentifier(*arg)
            for arg in zip(
                id_pair, [amazon_google.left_source, amazon_google.right_source]
            )
        )

    matching_ref_ids = set(map(_get_ref_ids, result))
    clerical_ref_ids = set(map(_get_ref_ids, fs.clerical_review))

    from pyresolvemetrics import precision, recall, f1

    gt = amazon_google.test_split.ground_truth - clerical_ref_ids
    results = {
        "p": precision(gt, matching_ref_ids),
        "r": recall(gt, matching_ref_ids),
        "f1": f1(gt, matching_ref_ids),
    }
    assert results
