import pytest

from matchescu.matching.matchers import PPJoin


@pytest.fixture
def threshold(request) -> float:
    return request.param if hasattr(request, "param") else 0.5


@pytest.fixture
def sut(threshold) -> PPJoin:
    return PPJoin(threshold)


def test_ppjoin_init(sut):
    assert sut is not None


@pytest.mark.parametrize("threshold", [-0.01, 1.01, None, "abc"], indirect=True)
def test_ppjoin_invalid_thresholds(threshold):
    with pytest.raises(ValueError) as err_proxy:
        PPJoin(threshold)
    assert str(err_proxy.value) == f"'{threshold}' is not a valid Jaccard threshold"


def test_ppjoin_on_amazon_google(sut, amazon_google):
    matches = sut.predict(
        amazon_google.test_split.comparison_space, amazon_google.id_table
    )

    assert matches is not None
