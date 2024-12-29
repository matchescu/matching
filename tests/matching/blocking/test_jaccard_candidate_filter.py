from matchescu.matching.blocking import BlockEngine


def test_jaccard_filter_removes_items(
    abt, buy, abt_buy_identifier, abt_buy_perfect_mapping
):
    engine = (
        BlockEngine()
        .add_source(abt, abt_buy_identifier)
        .add_source(buy, abt_buy_identifier)
        .tf_idf(0.3)
    )
    engine.update_candidate_pairs(False)
    blocking_metrics = engine.calculate_metrics(abt_buy_perfect_mapping)

    engine.filter_candidates_jaccard(0.15)

    filter_metrics = engine.calculate_metrics(abt_buy_perfect_mapping)
    assert blocking_metrics.reduction_ratio < filter_metrics.reduction_ratio
    ratio1 = filter_metrics.pair_completeness / blocking_metrics.pair_completeness
    ratio2 = blocking_metrics.pair_quality / filter_metrics.pair_quality
    assert ratio1 > ratio2
