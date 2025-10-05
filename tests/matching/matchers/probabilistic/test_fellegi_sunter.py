import pytest

from matchescu.matching.config import RecordLinkageConfig
import matchescu.matching.matchers as m
from matchescu.typing import EntityReferenceIdentifier


@pytest.fixture
def rl_config(request) -> RecordLinkageConfig:
    col_config = [("fname", "fname"), ("lname", "lname")]
    if hasattr(request, "param") and isinstance(request.param, list):
        col_config = request.param
    return RecordLinkageConfig("idA", "idB", "label", col_config)


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
