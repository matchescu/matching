from functools import partial

import pytest

from matchescu.extraction import Traits
from matchescu.matching.config import AttrCmpConfig, RecordLinkageConfig
from matchescu.matching.evaluation.datasets import MagellanDataset
from matchescu.matching.similarity import BucketedJaccard, BucketedNorm
from matchescu.typing import EntityReferenceIdentifier


@pytest.fixture
def amazon_google_traits():
    return Traits().int(["id"]).string(["title", "manufacturer"]).currency(["price"])


@pytest.fixture
def amazon_google_config(amazon_google_traits):
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
    return RecordLinkageConfig(
        amazon_google_traits, amazon_google_traits, attr_comparisons
    )


@pytest.fixture
def id_col():
    return "id"


@pytest.fixture
def amazon_google(data_dir, amazon_google_traits, id_col):
    ret = MagellanDataset(data_dir / "amazon_google_exp_data")

    def ref_id(records, source):
        return EntityReferenceIdentifier(records[0][id_col], source)

    ret.load_left(amazon_google_traits, partial(ref_id, source=ret.left_source))
    ret.load_right(amazon_google_traits, partial(ref_id, source=ret.right_source))
    ret.load_splits()

    return ret
