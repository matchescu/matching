import pytest
import torch
from transformers import BertConfig, BertModel

from matchescu.matching.matchers.ml.deeper import _module as deeper_module
from matchescu.matching.matchers.ml.deeper._module import DeepERModule
from matchescu.matching.matchers.ml.deeper._params import DeepERParams

NUM_ATTRS = 3
MAX_LEN = 12
HIDDEN = 32
VOCAB = 64
LSTM_HIDDEN = 16
SIM_HIDDEN = 8

# lengths chosen so the sort permutation is NOT an involution:
#   lengths      = [5, 9, 3, 7]
#   sorted_idx   = [1, 3, 0, 2]
#   unsorted_idx = [2, 0, 3, 1]      pi(pi(0)) = pi(2) = 3 != 0
# a double un-permutation therefore produces observably wrong rows.
LENGTHS = [5, 9, 3, 7]


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(1234)
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


@pytest.fixture
def tiny_bert(monkeypatch):
    """Replace ``BertModel.from_pretrained`` with a small randomly-init BERT.

    Keeps the tests offline and fast while exercising the real BertModel code path.
    """
    config = BertConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=MAX_LEN + 2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )

    def _fake_from_pretrained(*_args, **_kwargs):
        torch.manual_seed(7)
        return BertModel(config)

    monkeypatch.setattr(
        deeper_module.BertModel, "from_pretrained", _fake_from_pretrained
    )
    return config


def _params(**overrides) -> DeepERParams:
    kwargs = {
        "num_attributes": NUM_ATTRS,
        "lstm_hidden_size": LSTM_HIDDEN,
        "similarity_hidden_size": SIM_HIDDEN,
        "frozen_layer_count": 0,
        "output_size": 2,
    }
    kwargs.update(overrides)
    return DeepERParams(**kwargs)


@pytest.fixture
def model(tiny_bert):
    m = DeepERModule(_params())
    return m.eval()  # no dropout -> forward is deterministic


def _make_attr(lengths: list[int], seed: int) -> dict:
    """One attribute's batch: right-padded ids plus matching attention mask."""
    gen = torch.Generator().manual_seed(seed)
    batch_size = len(lengths)
    input_ids = torch.zeros(batch_size, MAX_LEN, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, MAX_LEN, dtype=torch.long)
    for row, length in enumerate(lengths):
        input_ids[row, :length] = torch.randint(1, VOCAB, (length,), generator=gen)
        attention_mask[row, :length] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _make_batch(lengths: list[int], seed: int = 0) -> list[dict]:
    return [_make_attr(lengths, seed + k) for k in range(NUM_ATTRS)]


def _index_batch(attrs: list[dict], idx) -> list[dict]:
    return [
        {"input_ids": a["input_ids"][idx], "attention_mask": a["attention_mask"][idx]}
        for a in attrs
    ]


def _capture_similarity_inputs(model, left, right) -> list[torch.Tensor]:
    """Run a forward pass, returning the per-attribute inputs to _similarity."""
    captured: list[torch.Tensor] = []

    def _hook(_module, args):
        captured.append(args[0].detach().clone())

    handle = model._similarity.register_forward_pre_hook(_hook)
    try:
        with torch.no_grad():
            model(left, right)
    finally:
        handle.remove()
    return captured


# --------------------------------------------------------------------------- #
# batch-order correctness
# --------------------------------------------------------------------------- #


def test_identical_sides_yield_zero_difference(model):
    """The comparison features must show a perfect match on a self-comparison.

    Left and right are composed by separate __compose_attr calls, so a
    batch-order bug permutes them differently. Asserting on the logits is not
    enough: the features are [|h_l - h_r| ; h_l * h_r], and the Hadamard half
    is h_l**2 here, which legitimately differs per row. So inspect the halves
    directly -- |diff| must be exactly 0, and the product must be non-negative
    (a mispaired row squares two different vectors and yields negatives).
    """
    attrs = _make_batch(LENGTHS)

    captured = _capture_similarity_inputs(
        model, attrs, _index_batch(attrs, slice(None))
    )

    assert len(captured) == NUM_ATTRS, "one _similarity call per attribute"
    for k, feats in enumerate(captured):
        assert feats.shape[0] == len(LENGTHS)
        half = feats.size(-1) // 2
        diff, prod = feats[:, :half], feats[:, half:]
        assert torch.count_nonzero(diff) == 0, (
            f"attribute {k}: |h_l - h_r| is non-zero on a self-comparison; "
            "left and right hidden states are not aligned to the same records"
        )
        assert (prod >= 0).all(), (
            f"attribute {k}: h_l * h_r has negative entries on a "
            "self-comparison; rows are mispaired across the batch"
        )


def test_similarity_width_matches_composition(model):
    """The head's declared input width must match what __compose_attr emits.

    Guards the regression where the LSTM went bidirectional but the composer
    still returned a single direction (32 features into a 64-wide Linear).
    """
    directions = 2 if model._lstm.bidirectional else 1
    expected = 2 * directions * model._lstm.hidden_size  # [diff ; prod]

    assert model._similarity[0].in_features == expected
    assert model.classifier.in_features == NUM_ATTRS * SIM_HIDDEN

    attrs = _make_batch(LENGTHS)
    captured = _capture_similarity_inputs(model, attrs, attrs)
    for feats in captured:
        assert feats.size(-1) == expected


def test_batched_forward_matches_per_row_forward(model):
    """Batching must not change results.

    A batch of size 1 cannot be permuted, so the row-by-row pass is a
    trustworthy reference for the batched pass.
    """
    left = _make_batch(LENGTHS, seed=0)
    right = _make_batch([4, 6, 11, 2], seed=100)

    with torch.no_grad():
        batched = model(left, right)
        rows = torch.cat(
            [
                model(
                    _index_batch(left, slice(i, i + 1)),
                    _index_batch(right, slice(i, i + 1)),
                )
                for i in range(len(LENGTHS))
            ],
            dim=0,
        )

    torch.testing.assert_close(batched, rows, atol=1e-5, rtol=1e-5)


def test_forward_is_equivariant_under_row_permutation(model):
    """forward(x[perm]) == forward(x)[perm].

    Fails on the old code because each batch's sort permutation depends on the
    lengths present in that batch.
    """
    left = _make_batch(LENGTHS, seed=0)
    right = _make_batch([4, 6, 11, 2], seed=100)
    perm = torch.tensor([2, 0, 3, 1])

    with torch.no_grad():
        base = model(left, right)
        permuted = model(_index_batch(left, perm), _index_batch(right, perm))

    torch.testing.assert_close(permuted, base[perm], atol=1e-5, rtol=1e-5)


def test_equal_lengths_hide_the_bug(model):
    """Documents why the regression was latent.

    With uniform lengths the sort permutation is the identity, so the buggy
    double un-permutation was a no-op. Any padding scheme that made lengths
    uniform (e.g. a null policy that produced constant-width text, or lengths
    read off the padded width instead of attention_mask.sum(1)) masked it.
    """
    uniform = _make_batch([6, 6, 6, 6], seed=0)
    perm = torch.tensor([2, 0, 3, 1])

    with torch.no_grad():
        base = model(uniform, uniform)
        permuted = model(_index_batch(uniform, perm), _index_batch(uniform, perm))

    torch.testing.assert_close(permuted, base[perm], atol=1e-6, rtol=1e-6)


def test_empty_attribute_is_handled(model):
    """An all-zero attention mask (empty attribute text) must not crash.

    _attr_text maps None/"none" to "", and a fully padded row has length 0,
    which pack_padded_sequence rejects.
    """
    attrs = _make_batch(LENGTHS)
    attrs[1]["attention_mask"][2] = 0  # attribute 1, row 2 -> empty

    with torch.no_grad():
        logits = model(attrs, attrs)

    assert logits.shape == (len(LENGTHS), 2)
    assert torch.isfinite(logits).all()


def test_empty_attribute_composes_to_zero(model):
    """A missing value must contribute 0, not whatever the forced position encodes.

    lengths.clamp(min=1) keeps pack_padded_sequence happy by pretending the row
    has one token; the nonempty mask is what stops that fiction from reaching
    the head.
    """
    attrs = _make_batch(LENGTHS)
    attrs[1]["attention_mask"][2] = 0

    captured = _capture_similarity_inputs(model, attrs, attrs)
    feats = captured[1]  # the attribute with the emptied row
    half = feats.size(-1) // 2

    # the empty row is zeroed in BOTH halves: h_l = h_r = 0 -> diff = prod = 0
    assert (
        torch.count_nonzero(feats[2]) == 0
    ), "an empty attribute leaked a non-zero comparison vector"

    # a populated row is a self-comparison: the diff half is exactly 0, but the
    # Hadamard half is h_l**2 and must NOT be zero -- otherwise this test would
    # still pass with the whole attribute silently zeroed out
    diff, prod = feats[0, :half], feats[0, half:]
    assert torch.count_nonzero(diff) == 0, "self-comparison diff must be zero"
    assert (prod >= 0).all(), "h_l * h_r went negative on a self-comparison"
    assert (
        prod > 0
    ).any(), "the nonempty mask zeroed a populated row, not just the empty one"
    assert torch.isfinite(feats).all()


# --------------------------------------------------------------------------- #
# nn.Module contract
# --------------------------------------------------------------------------- #


def test_to_returns_self_and_does_not_relocate_on_eval(model):
    assert model.to(torch.device("cpu")) is model, "to() must return self"
    assert model.eval() is model, "eval() must return self"
    assert model.train(True) is model, "train() must return self"


def test_device_property_tracks_parameters(model):
    assert model.device == next(model.parameters()).device


def test_train_mode_propagates(model):
    model.train(True)
    assert model.training is True, "super().train(mode) was not called"
    assert model._lstm.training is True
    model.eval()
    assert model.training is False
    assert model._lstm.training is False


def test_fully_frozen_bert_stays_in_eval_mode(tiny_bert):
    m = DeepERModule(_params(frozen_layer_count=tiny_bert.num_hidden_layers))
    m.train(True)

    assert not any(p.requires_grad for p in m.bert.parameters())
    assert m.bert.training is False, "frozen BERT must not run dropout"
    assert m._lstm.training is True


def test_partial_freeze_leaves_top_layers_trainable(tiny_bert):
    m = DeepERModule(_params(frozen_layer_count=1))

    assert not any(p.requires_grad for p in m.bert.embeddings.parameters())
    assert not any(p.requires_grad for p in m.bert.encoder.layer[0].parameters())
    assert all(p.requires_grad for p in m.bert.encoder.layer[1].parameters())


def test_mismatched_attribute_counts_raise(model):
    left = _make_batch(LENGTHS)
    with pytest.raises(ValueError):
        model(left, left[:-1])
    with pytest.raises(ValueError):
        model(left[:-1], left[:-1])  # matches each other but not num_attributes


def test_mismatched_padded_widths_raise(model):
    """split(B) would misalign rows against labels rather than raise."""
    left = _make_batch(LENGTHS)
    left[1] = {
        "input_ids": left[1]["input_ids"][:, :-2],
        "attention_mask": left[1]["attention_mask"][:, :-2],
    }
    with pytest.raises(ValueError):
        model(left, _make_batch(LENGTHS))


# --------------------------------------------------------------------------- #
# end-to-end learning signal
# --------------------------------------------------------------------------- #


def test_can_overfit_a_tiny_batch(tiny_bert):
    """The plumbing must be able to memorize 8 pairs.

    If this plateaus near ln(2) ~ 0.693, the gradient path is broken, not the
    hyperparameters. This test also fails on the scrambled-batch version:
    features are decorrelated from labels, so no amount of steps helps.
    """
    model = DeepERModule(_params(frozen_layer_count=tiny_bert.num_hidden_layers)).train(
        True
    )

    lengths = [5, 9, 3, 7, 2, 11, 4, 8]
    left = _make_batch(lengths, seed=0)
    # matches (label 1) are exact copies; non-matches use unrelated tokens
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])
    other = _make_batch(lengths[::-1], seed=500)
    right = [
        {
            "input_ids": torch.where(
                labels.bool().unsqueeze(1), l["input_ids"], o["input_ids"]
            ),
            "attention_mask": torch.where(
                labels.bool().unsqueeze(1), l["attention_mask"], o["attention_mask"]
            ),
        }
        for l, o in zip(left, other)
    ]

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert not any(
        p is q for p in trainable for q in model.bert.parameters()
    ), "BERT is fully frozen: the head is the only trainable path"

    optimizer = torch.optim.Adam(trainable, lr=3e-3)
    criterion = torch.nn.CrossEntropyLoss()

    first_loss, last_loss = None, None
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(left, right), labels)
        loss.backward()
        optimizer.step()
        first_loss = first_loss if first_loss is not None else loss.item()
        last_loss = loss.item()

    assert last_loss < 0.1, (
        f"failed to overfit 8 pairs: {first_loss:.4f} -> {last_loss:.4f}; "
        "the model is not receiving a usable learning signal"
    )

    model.eval()
    with torch.no_grad():
        preds = model(left, right).argmax(dim=-1)
    assert torch.equal(preds, labels)
