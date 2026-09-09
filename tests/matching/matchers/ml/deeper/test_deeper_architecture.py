import pytest
import torch

from matchescu.matching.matchers.ml.deeper._module import DeepERModule

from .._constants import BATCH


def test_forward_returns_expected_shape_when_inputs_match(deeper_module, deeper_batch):
    left, right = deeper_batch
    out = deeper_module(left, right)
    assert out.shape == (BATCH, deeper_module.classifier.out_features)


def test_forward_raises_when_attr_counts_differ(deeper_module, make_attr):
    left = [make_attr(), make_attr()]
    right = [make_attr()]
    with pytest.raises(ValueError, match="same number of attrs"):
        deeper_module(left, right)


def test_compose_attr_returns_expected_shape_when_given_one_attr(
    deeper_module, make_attr
):
    attr = make_attr()
    emb = deeper_module._DeepERModule__encode_all_attrs([attr])[0]
    h = deeper_module._DeepERModule__compose_attr(emb, attr["attention_mask"])
    assert h.shape == (BATCH, deeper_module._lstm.hidden_size)


@pytest.mark.parametrize("num_attributes", [1, 2, 3])
def test_forward_returns_batched_rows_when_attr_count_varies(
    make_deeper_params, make_attr, num_attributes
):
    module = DeepERModule(make_deeper_params(num_attributes=num_attributes))
    left = [make_attr() for _ in range(num_attributes)]
    right = [make_attr() for _ in range(num_attributes)]
    out = module(left, right)
    assert out.shape[0] == BATCH


def test_forward_returns_same_output_when_called_twice(deeper_module, deeper_batch):
    left, right = deeper_batch
    out1 = deeper_module(left, right)
    out2 = deeper_module(left, right)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_encode_all_attrs_returns_empty_list_when_no_attrs(deeper_module):
    result = deeper_module._DeepERModule__encode_all_attrs([])
    assert result == []


def test_forward_mean_pools_across_attributes_when_multiple_attrs(
    make_deeper_params, make_attr
):
    n_attrs = 3
    module = DeepERModule(make_deeper_params(num_attributes=n_attrs))
    left = [make_attr() for _ in range(n_attrs)]
    right = [make_attr() for _ in range(n_attrs)]
    out = module(left, right)
    assert out.shape == (BATCH, module.classifier.out_features)
