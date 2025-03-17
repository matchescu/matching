import pytest
from matchescu.matching.ml.ditto import Augmenter


@pytest.fixture
def text() -> str:
    return "COL content VAL vldb conference papers 2020-01-01 COL year VAL 2020 [SEP] COL content VAL sigmod conference 2010 papers 2019-12-31 COL year VAL 2019"


@pytest.mark.parametrize(
    "op, expected",
    [
        (
            "del",
            "COL content VAL vldb conference COL year VAL 2020 [SEP] COL content VAL sigmod conference 2010 papers 2019-12-31 COL year VAL 2019",
        ),
        # ("drop_col", 'COL content VAL sigmod conference 2019-12-31 COL year VAL 2019 [SEP] COL content VAL vldb conference papers 2020-01-01 COL year VAL 2020'),
        # ("append_col", 'COL content VAL sigmod conference 2019-12-31 COL year VAL 2019 [SEP] COL content VAL vldb conference papers 2020-01-01 COL year VAL 2020'),
        # ("drop_token", 'COL content VAL sigmod conference 2019-12-31 COL year VAL 2019 [SEP] COL content VAL vldb conference papers 2020-01-01 COL year VAL 2020'),
        # ("drop_len", 'COL content VAL sigmod conference 2019-12-31 COL year VAL 2019 [SEP] COL content VAL vldb conference papers 2020-01-01 COL year VAL 2020'),
        # ("drop_sym", 'COL content VAL sigmod conference 2019-12-31 COL year VAL 2019 [SEP] COL content VAL vldb conference papers 2020-01-01 COL year VAL 2020'),
        # ("drop_same", 'COL content VAL sigmod conference 2019-12-31 COL year VAL 2019 [SEP] COL content VAL vldb conference papers 2020-01-01 COL year VAL 2020'),
        # ("swap", 'COL content VAL sigmod conference 2019-12-31 COL year VAL 2019 [SEP] COL content VAL vldb conference papers 2020-01-01 COL year VAL 2020'),
        # ("ins", 'COL content VAL sigmod conference 2019-12-31 COL year VAL 2019 [SEP] COL content VAL vldb conference papers 2020-01-01 COL year VAL 2020'),
        # ("all", 'COL content VAL sigmod conference 2019-12-31 COL year VAL 2019 [SEP] COL content VAL vldb conference papers 2020-01-01 COL year VAL 2020'),
    ],
)
def test_all_ops(text, op, expected):
    ag = Augmenter()
    augmented = ag.augment_sent(text, op=op)
    assert augmented == expected
