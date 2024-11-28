import pytest

from matchescu.matching.attribute import RawMatch


@pytest.fixture
def sut(similarity_stub):
    return RawMatch(similarity_stub)
