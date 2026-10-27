import pytest

FILE_NAMES = {"train": "a.csv", "dev": "b.csv", "test": "c.csv"}
SYNTHETIC_COMPONENTS = 5


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


def test_generate_logs_bridge_graph_component_count(generation_messages):
    expected = f"bridge graph: {SYNTHETIC_COMPONENTS} components"

    assert expected in generation_messages
