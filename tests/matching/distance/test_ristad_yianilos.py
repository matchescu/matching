import polars as pl
import pytest

from matchescu.matching.similarity import LevenshteinLearner


@pytest.fixture(scope="module")
def abt_buy_dir(data_dir):
    return data_dir / "abt-buy"


@pytest.fixture(scope="module")
def abt(abt_buy_dir) -> pl.DataFrame:
    return pl.read_csv(abt_buy_dir / "Abt.csv", ignore_errors=True)


@pytest.fixture(scope="module")
def buy(abt_buy_dir) -> pl.DataFrame:
    return pl.read_csv(abt_buy_dir / "Buy.csv", ignore_errors=True)


@pytest.fixture(scope="module")
def perfect_mapping(abt_buy_dir):
    return pl.read_csv(abt_buy_dir / "abt_buy_perfectMapping.csv", ignore_errors=True)


@pytest.fixture(scope="module")
def corpus(abt, buy, perfect_mapping) -> list[tuple[str, str]]:
    perfect_abt = abt.join(
        perfect_mapping, left_on="id", right_on="idAbt", how="inner"
    ).select("name")
    perfect_buy = buy.join(
        perfect_mapping, left_on="id", right_on="idBuy", how="inner"
    ).select("name")
    matching_names = pl.concat(
        [
            perfect_abt.rename({"name": "abt_name"}),
            perfect_buy.rename({"name": "buy_name"}),
        ],
        how="horizontal",
    )
    return [(str(val[0]), str(val[1])) for val in matching_names.iter_rows()]


@pytest.fixture(scope="module")
def model(corpus) -> LevenshteinLearner:
    result = LevenshteinLearner()
    result.fit(corpus, 10)
    return result


def test_similarity(model):
    same = round(model("linksys", "linksys"), 2)
    slight_difference = round(model("astana", "agata"), 2)
    bigger_difference = round(
        model(
            "Linksys EtherFast 8-Port 10/100 Switch - EZXS88W",
            "Linksys EtherFast EZXS88W Ethernet Switch - EZXS88W",
        )
    )

    assert same >= slight_difference >= bigger_difference


def test_distance(model):
    same = round(model.compute_distance("linksys", "linksys"), 2)
    slight_difference = round(model.compute_distance("astana", "agata"), 2)
    bigger_difference = round(
        model.compute_distance(
            "Linksys EtherFast 8-Port 10/100 Switch - EZXS88W",
            "Linksys EtherFast EZXS88W Ethernet Switch - EZXS88W",
        )
    )

    assert same < slight_difference < bigger_difference
