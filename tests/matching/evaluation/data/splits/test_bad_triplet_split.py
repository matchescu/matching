import pytest

FILE_NAMES = {"train": "a.csv", "dev": "b.csv", "test": "c.csv"}
SYNTHETIC_COMPONENTS = 5
SYNTHETIC_LABEL_TWO = {"train": 9, "dev": 3, "test": 3}
DATASETS = [
    "bad_triplet_synthetic",
    "amb_fodors_zagat_nophone_a050_dir",
    "amb_affiliationstrings_a050_dir",
]
SCATTERED_CLUSTERS = 20
SCATTERED_BRIDGES = 5


def test_generate_raises_when_matcher_ground_truth_has_class_three(
    make_generator, label_three_data
):
    generator = make_generator(label_three_data)

    with pytest.raises(ValueError, match="does not support classes"):
        generator.generate()


def test_get_splits_returns_one_split_per_split_name(generated, split_ratio):
    splits = generated.get_splits()

    assert list(splits) == list(split_ratio)


def test_save_writes_configured_file_names(generated, tmp_path):
    generated.save(tmp_path, file_names=FILE_NAMES)

    assert all((tmp_path / name).is_file() for name in FILE_NAMES.values())


def test_generate_logs_bridge_graph_component_count(make_generator, dataset, caplog):
    generator = make_generator(dataset)

    generator.generate()

    assert f"bridge graph: {SYNTHETIC_COMPONENTS} components" in caplog.messages


@pytest.mark.parametrize("dataset", DATASETS, indirect=True)
def test_each_cluster_lands_in_one_split(cluster_splits):
    assert all(len(splits) == 1 for splits in cluster_splits.values())


@pytest.mark.parametrize("dataset", DATASETS, indirect=True)
def test_bridge_footprint_lands_in_one_split(footprint_splits):
    assert all(len(splits) == 1 for splits in footprint_splits.values())


@pytest.mark.parametrize("dataset", DATASETS, indirect=True)
def test_splits_keep_every_label_two_pair(label_two_rows, label_two_truth):
    kept = set().union(*label_two_rows.values())

    assert kept == label_two_truth


def test_label_two_rows_follow_split_ratio_when_components_allow(label_two_rows):
    counts = {name: len(pairs) for name, pairs in label_two_rows.items()}

    assert counts == SYNTHETIC_LABEL_TWO


def test_generate_warns_when_a_split_has_few_label_two_pairs(
    make_generator, make_pair_data, caplog
):
    data = make_pair_data(SCATTERED_CLUSTERS, SCATTERED_BRIDGES)

    make_generator(data).generate()

    assert "fewer than MIN_PER_CLASS" in caplog.text


def test_generate_raises_when_bridge_has_no_triplet_source_pair(
    make_generator, sourceless_data
):
    generator = make_generator(sourceless_data)

    with pytest.raises(ValueError, match="triplet source pair"):
        generator.generate()


def test_init_warns_when_drop_orphans_is_false(make_generator, dataset, caplog):
    make_generator(dataset, drop_orphans=False)

    assert "drop_orphans=False is ignored" in caplog.text


def test_generate_keeps_label_two_pairs_when_bridges_are_selected_by_source(
    make_generator, dataset, select_by_source, label_two_truth
):
    generator = make_generator(dataset).with_bridge_selector(select_by_source)

    splits = generator.generate().get_splits()

    labels = [v for split in splits.values() for v in split.matcher_labels.values()]
    assert labels.count(2) == len(label_two_truth)


def test_generate_raises_when_label_two_pair_has_no_selected_bridge(
    make_generator, dataset, select_nothing
):
    generator = make_generator(dataset).with_bridge_selector(select_nothing)

    with pytest.raises(ValueError, match="not a bridge"):
        generator.generate()
