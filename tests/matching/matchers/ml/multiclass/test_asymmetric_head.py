from io import BytesIO
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from matchescu.matching.matchers.ml.multiclass import _classifier
from matchescu.matching.matchers.ml.multiclass._module import MultiClassModule

from .._constants import BATCH, HIDDEN


@pytest.fixture
def head():
    head = _classifier.AsymmetricHead(2, rank=2, dropout_p=0.0)
    with torch.no_grad():
        head.U.weight.copy_(torch.eye(2))
        head.V.weight.copy_(torch.tensor([[0.0, 1.0], [2.0, 0.0]]))
    return head


@pytest.fixture
def pair():
    return torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 5.0]])


@pytest.mark.parametrize("use_bilinear", [False, True])
@pytest.mark.parametrize("use_order", [False, True])
@pytest.mark.parametrize("rank", [2, 5])
def test_features_have_expected_blocks_when_flags_vary(
    pair, use_bilinear, use_order, rank
):
    head = _classifier.AsymmetricHead(
        2, rank=rank, use_bilinear=use_bilinear, use_order=use_order
    )
    a, b = pair
    expected = [a, b, (a - b).abs()]
    if use_bilinear:
        expected.append(head.U(a) * head.V(b))
    if use_order:
        expected.append(torch.tensor([[-13.0]]))

    features = head.features(a, b)

    assert features.shape[-1] == head.input_size == 6 + rank * use_bilinear + use_order
    torch.testing.assert_close(features, torch.cat(expected, dim=-1))
    assert head(features).shape == (1, 3)


def test_bilinear_projections_have_independent_bias_free_weights(head):
    assert head.U is not head.V
    assert head.U.weight.data_ptr() != head.V.weight.data_ptr()
    assert head.U.bias is head.V.bias is None


def test_disabling_bilinear_omits_projection_parameters():
    head = _classifier.AsymmetricHead(2, use_bilinear=False)
    assert not any(name.startswith(("U.", "V.")) for name, _ in head.named_parameters())


def test_bilinear_product_is_asymmetric_when_transforms_differ(head, pair):
    a, b = pair
    forward = head.features(a, b)[:, 6:8]
    reverse = head.features(b, a)[:, 6:8]
    torch.testing.assert_close(forward, torch.tensor([[5.0, 12.0]]))
    torch.testing.assert_close(reverse, torch.tensor([[6.0, 10.0]]))


def test_product_block_is_symmetric_when_transform_is_intentionally_tied(head, pair):
    head.V = head.U
    a, b = pair
    torch.testing.assert_close(head.features(a, b)[:, 6:8], head.features(b, a)[:, 6:8])


def test_order_delta_is_antisymmetric_when_pair_is_reversed(head, pair):
    a, b = pair
    forward = head.features(a, b)[:, -1]
    reverse = head.features(b, a)[:, -1]
    torch.testing.assert_close(forward, torch.tensor([-13.0]))
    torch.testing.assert_close(reverse, -forward)


def test_asymmetric_head_keeps_existing_mlp_widths_and_dropout():
    head = _classifier.AsymmetricHead(HIDDEN, dropout_p=0.3)
    mlp = next(
        m for m in head.modules() if isinstance(m, _classifier.ClassificationHead)
    )
    layers = list(mlp._model)
    assert [
        (m.in_features, m.out_features) for m in layers if isinstance(m, nn.Linear)
    ] == [
        (3 * HIDDEN + 256 + 1, HIDDEN),
        (HIDDEN, 3),
    ]
    assert isinstance(layers[1], nn.ReLU)
    assert isinstance(layers[2], nn.Dropout) and layers[2].p == 0.3


def test_forward_consumes_features_without_recomputing(head, monkeypatch):
    features = torch.randn(BATCH, head.input_size)
    spy = Mock(side_effect=AssertionError("features must not be recomputed"))
    monkeypatch.setattr(head, "features", spy)
    mlp = next(
        m for m in head.modules() if isinstance(m, _classifier.ClassificationHead)
    )

    torch.testing.assert_close(head(features), mlp(features))

    spy.assert_not_called()


def test_module_builds_features_once_when_classifying(
    make_params, fake_bert, monkeypatch
):
    module = MultiClassModule(make_params(head_type="asymmetric", dropout_p=0.0))
    fake_bert.return_value = (torch.randn(BATCH, 2, HIDDEN),)
    spy = Mock(wraps=module.classifier.features)
    monkeypatch.setattr(module.classifier, "features", spy)

    logits = module(
        torch.ones(BATCH, 2, dtype=torch.long),
        torch.ones(BATCH, 2),
        torch.tensor([[0, 1]] * BATCH),
    )

    assert logits.shape == (BATCH, 3)
    spy.assert_called_once()


@pytest.mark.parametrize("projection", ["U", "V"])
def test_optimizer_includes_projection_in_classifier_group(
    make_params, make_trainer, fake_bert, projection
):
    fake_bert.encoder = Mock(layer=nn.ModuleList([nn.Linear(HIDDEN, HIDDEN)]))
    fake_bert.embeddings = nn.Embedding(4, HIDDEN)
    module = MultiClassModule(make_params(head_type="asymmetric"))
    trainer = make_trainer()

    optimizer = trainer._create_optimizer(module)

    weight = getattr(module.classifier, projection).weight
    groups = [
        g for g in optimizer.param_groups if any(p is weight for p in g["params"])
    ]
    assert len(groups) == 1
    assert (
        groups[0]["lr"]
        == trainer._params.learning_rate / trainer._params.lr_decay_factor
    )
    assert groups[0]["weight_decay"] == trainer._params.weight_decay


def test_classifier_backpropagates_to_projections_and_embeddings(
    make_params, fake_bert
):
    module = MultiClassModule(make_params(head_type="asymmetric", dropout_p=0.0))
    with torch.no_grad():
        for parameter in module.classifier.parameters():
            parameter.fill_(0.1)
    hidden = torch.ones(BATCH, 2, HIDDEN, requires_grad=True)
    fake_bert.return_value = (hidden,)

    logits = module(
        torch.ones(BATCH, 2, dtype=torch.long),
        torch.ones(BATCH, 2),
        torch.tensor([[0, 1]] * BATCH),
    )
    logits.square().sum().backward()

    for tensor in (hidden, module.classifier.U.weight, module.classifier.V.weight):
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0


def test_asymmetric_checkpoint_roundtrip_preserves_logits(make_params):
    params = make_params(head_type="asymmetric", dropout_p=0.0)
    module = MultiClassModule(params)
    module._bert = nn.Linear(HIDDEN, HIDDEN)
    a, b = torch.randn(BATCH, HIDDEN), torch.randn(BATCH, HIDDEN)
    expected = module.classifier(module._apply_head(a, b))
    checkpoint = BytesIO()
    torch.save(module.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = MultiClassModule(params)
    restored._bert = nn.Linear(HIDDEN, HIDDEN)

    restored.load_state_dict(torch.load(checkpoint, weights_only=True))

    assert {"_classifier.U.weight", "_classifier.V.weight"} <= set(
        restored.state_dict()
    )
    torch.testing.assert_close(
        restored.classifier(restored._apply_head(a, b)), expected
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_head_uses_encoder_dtype_when_initialized(make_params, fake_bert, dtype):
    fake_bert.dtype = dtype
    module = MultiClassModule(make_params(head_type="asymmetric"))
    a, b = torch.randn(BATCH, HIDDEN, dtype=dtype), torch.randn(
        BATCH, HIDDEN, dtype=dtype
    )

    features = module._apply_head(a, b)

    assert {p.dtype for p in module.classifier.parameters()} == {dtype}
    assert features.dtype == module.classifier(features).dtype == dtype


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_custom_module_to_moves_classifier_projections(make_params, device):
    module = MultiClassModule(make_params(head_type="asymmetric"))

    module.to(torch.device(device))

    assert {p.device for p in module.classifier.parameters()} == {torch.device(device)}


def test_head_to_converts_projection_dtype(head, pair):
    head.to(dtype=torch.float64)
    a, b = (encoding.double() for encoding in pair)
    assert {p.dtype for p in head.parameters()} == {torch.float64}
    assert head(head.features(a, b)).dtype == torch.float64
