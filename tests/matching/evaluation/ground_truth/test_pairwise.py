import pytest

from matchescu.matching.evaluation.ground_truth._pairwise import read_csv

@pytest.fixture
def mapping_file_path(data_dir, request):
    file_name = getattr(
        request, "param", "affiliationstrings/affiliationstrings_mapping.csv"
    )
    return data_dir / file_name


def test_read_csv_with_defaults(mapping_file_path):
    gt = read_csv(mapping_file_path, "affiliationstrings")

    assert len(gt) == 32815


@pytest.mark.parametrize("mapping_file_path", ["amazon_google_exp_data/test.csv"], indirect=True)
def test_read_csv_with_label(mapping_file_path):
    gt = read_csv(mapping_file_path, "amz-ggl", label_col="label")

    assert len(gt) != 2293
    assert len(gt) == 234
    assert all(
        x.source == "amz-ggl" and y.source == "amz-ggl"
        for x, y in gt
    )


@pytest.mark.parametrize("mapping_file_path,expected", [
    ("affiliationstrings/directed_with_label.csv", 3640),
    ("affiliationstrings/undirected_with_label.csv", 3566),
], indirect=["mapping_file_path"])
def test_read_csv_with_source_and_label(mapping_file_path, expected):
    gt = read_csv(
        mapping_file_path,
        source_cols=("left_source", "right_source"),
        label_col="label"
    )

    assert len(gt) == expected


def test_read_csv_must_specify_sources(mapping_file_path):
    with pytest.raises(ValueError) as err_proxy:
        read_csv(mapping_file_path)

    assert str(err_proxy.value) == "at least one of 'sources' or 'source_cols' must be specified"


def test_read_csv_must_specify_sources_unambiguously(mapping_file_path):
    with pytest.raises(ValueError) as err_proxy:
        read_csv(mapping_file_path, "abc", source_cols=("left_source", "right_source"))

    assert str(err_proxy.value) == "only one of 'sources' and 'source_cols' can be specified"


@pytest.mark.parametrize("mapping_file_path,expected", [
    ("affiliationstrings/directed_with_label.csv", {1,2,3}),
    ("affiliationstrings/undirected_with_label.csv", {1}),
], indirect=["mapping_file_path"])
def test_read_csv_reads_label_classes(mapping_file_path, expected):
    gt = read_csv(
        mapping_file_path,
        source_cols=("left_source", "right_source"),
        label_col="label"
    )

    assert set(gt.values()) == expected


@pytest.mark.parametrize("mapping_file_path", ["abt-buy/abt_buy_perfectMapping.csv"], indirect=True)
def test_read_csv_id_cols_with_sources(mapping_file_path):
    gt = read_csv(mapping_file_path, "abt", "buy", id_cols=("idAbt", "idBuy"))

    lhs, rhs = [], []
    for x, y in gt:
        lhs.append(x)
        rhs.append(y)

    assert all(x.source == "abt" for x in lhs)
    assert all(y.source == "buy" for y in rhs)