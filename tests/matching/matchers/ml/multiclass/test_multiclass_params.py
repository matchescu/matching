import pytest

from matchescu.matching.matchers.ml.ditto._params import DittoModelTrainingParams
from matchescu.matching.matchers.ml.multiclass._params import MultiClassTrainingParams


@pytest.mark.parametrize("key", ["alphaAug", "alpha_aug"])
def test_multiclass_parsing_does_not_retain_alpha_aug(key):
    params = MultiClassTrainingParams.model_validate({key: 0.3})
    assert not hasattr(params, "alpha_aug")


@pytest.mark.parametrize("by_alias,key", [(True, "alphaAug"), (False, "alpha_aug")])
def test_multiclass_dump_omits_alpha_aug(by_alias, key):
    assert key not in MultiClassTrainingParams().model_dump(by_alias=by_alias)


def test_multiclass_schema_omits_alpha_aug():
    assert "alphaAug" not in MultiClassTrainingParams.model_json_schema()["properties"]


def test_ditto_retains_augmentation_parameter():
    params = DittoModelTrainingParams.model_validate({"alphaAug": 0.3})
    assert params.alpha_aug == 0.3
    assert params.model_dump()["alphaAug"] == 0.3
