from collections.abc import Iterable

import pytest

from matchescu.matching.blocking import Block
from matchescu.references import EntityReference
from matchescu.typing import EntityReferenceIdentifier

DEFAULT_SOURCE_NAME = "test"
DEFAULT_BLOCK_KEY = "test block"


def _id(label, source=DEFAULT_SOURCE_NAME):
    return EntityReferenceIdentifier(label, source)


def _ref(label, source=DEFAULT_SOURCE_NAME):
    return EntityReference(_id(label, source), [label])


@pytest.fixture(scope="module")
def default_entity_reference() -> EntityReference:
    return EntityReference(
        EntityReferenceIdentifier(1, "test"),
        [1, "reference name", "a short reference description"]
    )


@pytest.fixture
def block():
    return Block(DEFAULT_BLOCK_KEY)


@pytest.mark.parametrize("block_key", [None, "", "\r", "\n", "\t", " "])
def test_block_init_raises_value_error(block_key):
    with pytest.raises(ValueError) as ve:
        Block(block_key)

    assert str(ve.value) == "invalid blocking key"


def test_block_add_ref(block, default_entity_reference):
    block.append(default_entity_reference)

    assert list(block) == [default_entity_reference.id]


def test_block_add_multiple_refs(block):
    block.extend(_ref(i) for i in range(10))

    assert list(block) == [
        EntityReferenceIdentifier(i, DEFAULT_SOURCE_NAME)
        for i in range(10)
    ]


@pytest.mark.parametrize(
    "refs,expected",
    [
        ([_ref(1)], []),
        (
                [_ref(1), _ref(2)],
                [(_id(1), _id(2))]
        ),
        (
            [_ref(1), _ref(2), _ref(3)],
            [
                (_id(1), _id(2)),
                (_id(1), _id(3)),
                (_id(2), _id(3)),
            ],
        ),
    ],
)
def test_candidate_pairs_single_source(block, refs, expected):
    block.extend(refs)

    actual = list(block.candidate_pairs())

    assert actual == expected


@pytest.mark.parametrize(
    "refs,expected",
    [
        ([_ref(1, "a"), _ref(1, "b")], [(_id(1, "a"), _id(1, "b"))]),
        ([_ref(1, "a"), _ref(2, "a"), _ref(1, "b")], [(_id(1, "a"), _id(1, "b")), (_id(2, "a"), _id(1, "b"))]),
        ([_ref(1, "a"), _ref(2, "a"), _ref(1, "b"), _ref(2, "b")], [(_id(1, "a"), _id(1, "b")), (_id(1, "a"), _id(2, "b")), (_id(2, "a"), _id(1, "b")), (_id(2, "a"), _id(2, "b"))]),
    ],
)
def test_candidate_pairs_two_sources(block, refs, expected):
    block.extend(refs)

    actual = list(block.candidate_pairs())

    assert actual == expected


@pytest.mark.parametrize(
    "refs,expected",
    [
        ([], 0),
        ([_ref(1, "a")], 1),
        ([_ref(1, "a"), _ref(2, "a")], 1),
        ([_ref(1, "a"), _ref(1, "b")], 2),
    ]
)
def test_block_len(block, refs, expected):
    block.extend(refs)

    assert block.count_sources() == expected