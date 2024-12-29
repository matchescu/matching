import pytest

from matchescu.matching.blocking import Block


DEFAULT_SOURCE_NAME = "DEFAULT"
DEFAULT_BLOCK_KEY = "test block"


@pytest.fixture(scope="module")
def default_entity_reference() -> tuple:
    return 1, "reference name", "a short reference description"


@pytest.fixture
def entity_references(request, default_entity_reference):
    if hasattr(request, "param") and request.param:
        return request.param
    return [default_entity_reference]


@pytest.fixture
def block(request):
    return Block(DEFAULT_BLOCK_KEY)


@pytest.mark.parametrize("block_key", [None, "", "\r", "\n", "\t", " "])
def test_block_init_raises_value_error(block_key):
    with pytest.raises(ValueError) as ve:
        block = Block(block_key)

    assert str(ve.value) == "invalid blocking key"


def test_block_add_ref(block):
    block.append(1)

    assert block.references == {DEFAULT_SOURCE_NAME: [1]}


@pytest.mark.parametrize(
    "source,expected",
    [
        (None, DEFAULT_SOURCE_NAME),
        ("", DEFAULT_SOURCE_NAME),
        (" ", DEFAULT_SOURCE_NAME),
        ("\t", DEFAULT_SOURCE_NAME),
        ("\r", DEFAULT_SOURCE_NAME),
        ("\n", DEFAULT_SOURCE_NAME),
        ("a", "a"),
    ],
)
def test_block_add_ref_from_source(block, source, expected):
    block.append(1, source_name=source)

    assert block.references == {expected: [1]}


def test_block_add_multiple_refs(block):
    block.extend([1, 2])

    assert block.references == {DEFAULT_SOURCE_NAME: [1, 2]}


@pytest.mark.parametrize(
    "source,expected",
    [
        (None, DEFAULT_SOURCE_NAME),
        ("", DEFAULT_SOURCE_NAME),
        (" ", DEFAULT_SOURCE_NAME),
        ("\t", DEFAULT_SOURCE_NAME),
        ("\r", DEFAULT_SOURCE_NAME),
        ("\n", DEFAULT_SOURCE_NAME),
        ("a", "a"),
    ],
)
def test_block_add_multiple_refs_from_source(block, source, expected):
    block.extend([1, 2], source_name=source)

    assert block.references == {expected: [1, 2]}


@pytest.mark.parametrize(
    "ref_ids,expected",
    [
        ([1], []),
        ([1, 2], [(1, 2)]),
        (
            [1, 2, 3],
            [
                (1, 2),
                (1, 3),
                (2, 3),
            ],
        ),
    ],
)
def test_candidate_pairs_single_source(block, ref_ids, expected):
    block.extend(ref_ids)

    actual = list(block.candidate_pairs())

    assert actual == expected


@pytest.mark.parametrize(
    "ref_ids,expected",
    [
        ({"a": [1], "b": [1]}, [(1, 1)]),
        ({"a": [1, 2], "b": [1]}, [(1, 1), (2, 1)]),
        ({"a": [1, 2], "b": [1, 2]}, [(1, 1), (1, 2), (2, 1), (2, 2)]),
    ],
)
def test_candidate_pairs_two_sources(block, ref_ids, expected):
    for src, src_ids in ref_ids.items():
        block.extend(src_ids, src)

    actual = list(block.candidate_pairs())

    assert actual == expected


@pytest.mark.parametrize(
    "ref_ids,expected",
    [
        ({"a": [1], "b": [1], "c": [1]}, [(1, 1), (1, 1), (1, 1)]),
        ({"a": [1, 2], "b": [1], "c": [1]}, [(1, 1), (2, 1), (1, 1), (2, 1), (1, 1)]),
        (
            {"a": [1, 2], "b": [1, 2], "c": [1]},
            [(1, 1), (1, 2), (2, 1), (2, 2), (1, 1), (2, 1), (1, 1), (2, 1)],
        ),
        (
            {"a": [1, 2], "b": [1, 2], "c": [1, 2]},
            [
                (1, 1),
                (1, 2),
                (2, 1),
                (2, 2),
                (1, 1),
                (1, 2),
                (2, 1),
                (2, 2),
                (1, 1),
                (1, 2),
                (2, 1),
                (2, 2),
            ],
        ),
        (
            {"a": [1, 2], "b": [1], "c": [1, 2]},
            [(1, 1), (2, 1), (1, 1), (1, 2), (2, 1), (2, 2), (1, 1), (1, 2)],
        ),
        (
            {"a": [1], "b": [1, 2], "c": [1, 2]},
            [(1, 1), (1, 2), (1, 1), (1, 2), (1, 1), (1, 2), (2, 1), (2, 2)],
        ),
    ],
)
def test_candidate_pairs_multiple_sources(block, ref_ids, expected):
    for src, src_ids in ref_ids.items():
        block.extend(src_ids, src)

    actual = list(block.candidate_pairs())

    assert actual == expected
