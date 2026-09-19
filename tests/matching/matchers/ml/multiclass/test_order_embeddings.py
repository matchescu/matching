from unittest.mock import Mock

import pytest
import torch

from matchescu.matching.matchers.ml.multiclass._module import MultiClassModule
from matchescu.matching.matchers.ml.multiclass._types import ArchitectureType, HeadType

from .._constants import HIDDEN


@pytest.fixture(params=list(ArchitectureType))
def encoded_model(request, make_params, fake_bert):
    module = MultiClassModule(
        make_params(architecture=request.param, dropout_p=0.0, order_margin=3.0)
    )
    hidden = (
        torch.arange(8 * HIDDEN).reshape(2, 4, HIDDEN).float() / 100
    ).requires_grad_()
    fake_bert.return_value = (hidden,)
    batch = {
        "input_ids": torch.ones(2, 4, dtype=torch.long),
        "attention_mask": torch.ones(2, 4),
        "token_type_ids": torch.tensor([[0, 0, 1, 1]] * 2),
        "col_positions": torch.tensor([[0, 2]] * 2),
        "value_mask": torch.ones(2, 4, dtype=torch.bool),
    }
    return module, batch, hidden


@pytest.mark.parametrize("head", list(HeadType))
def test_optional_embeddings_are_attached_pre_head_pair_from_same_pass(
    encoded_model, fake_bert, monkeypatch, head
):
    module, batch, hidden = encoded_model
    module._head_type = head
    if head == HeadType.NONE:
        module._classifier = torch.nn.Linear(2 * HIDDEN, 3)
    spy = Mock(wraps=module._apply_head)
    monkeypatch.setattr(module, "_apply_head", spy)
    logits, a, b = module(**batch, return_embeddings=True)
    assert spy.call_args.args[0] is a and spy.call_args.args[1] is b
    assert fake_bert.call_count == 1
    assert a.shape == b.shape == (2, HIDDEN)
    (a.square().sum() + b.square().sum()).backward()
    assert hidden.grad.abs().sum() > 0
    torch.testing.assert_close(module(**batch), logits)


def test_model_exposes_configured_order_margin(encoded_model):
    module, _, _ = encoded_model
    assert module.order_margin == 3.0


def test_trainer_returns_forward_embeddings_in_flat_tensor_tuple(make_trainer):
    logits, reverse_logits = torch.zeros(3, 3), torch.ones(3, 3)
    a, b = torch.zeros(3, 2, requires_grad=True), torch.ones(3, 2, requires_grad=True)
    model = Mock(side_effect=[(logits, a, b), reverse_logits])
    batch = (
        {"input_ids": torch.tensor([1])},
        {"input_ids": torch.tensor([2])},
        torch.tensor([0, 1, 2]),
    )
    result = make_trainer()._forward_pass(model, batch, torch.device("cpu"))
    assert len(result) == 6 and all(isinstance(x, torch.Tensor) for x in result)
    assert result[4] is a and result[5] is b
    assert model.call_args_list[0].kwargs["return_embeddings"] is True
    assert "return_embeddings" not in model.call_args_list[1].kwargs
    torch.testing.assert_close(result[3], torch.tensor([0, 1, 0]))
