from matchescu.matching.blocking import BlockEngine


def test_abt_buy_blocking(abt, buy, abt_buy_identifier):
    engine = (
        BlockEngine()
        .add_source(abt, abt_buy_identifier)
        .add_source(buy, abt_buy_identifier)
        .tf_idf()
    )

    assert engine.blocks


def test_abt_buy_tf_idf_blocking_metrics(
    abt, buy, abt_buy_identifier, abt_buy_perfect_mapping
):
    engine = (
        BlockEngine()
        .add_source(abt, abt_buy_identifier)
        .add_source(buy, abt_buy_identifier)
        .tf_idf(0.3)
    )
    engine.update_candidate_pairs(False)

    metrics = engine.calculate_metrics(abt_buy_perfect_mapping)

    assert metrics.pair_completeness > 0
    assert metrics.pair_quality > 0
    assert metrics.reduction_ratio < 1
