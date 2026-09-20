from unittest.mock import MagicMock

import pytest
import torch
from torch.nn.modules.loss import _Loss

from matchescu.matching.matchers.ml.multiclass._types import LossType

DEFAULT_LOSS = 1.0


@pytest.fixture(scope="package")
def default_loss():
    return DEFAULT_LOSS


@pytest.fixture
def trainer(make_trainer, request):
    rev_penalty = request.param if hasattr(request, "param") else 0.0
    return make_trainer(
        loss_type=LossType.WEIGHTED_CE, reverse_penalty_weight=rev_penalty
    )


@pytest.fixture
def targets():
    result = torch.randint(0, 2, (8,), dtype=torch.long)
    if not any(result == 2):
        result[0] = 2
    return result


@pytest.fixture
def targets_rev(targets):
    ret = targets.clone()
    ret[targets == 2] = 0
    return ret


@pytest.fixture
def loss_fn(request):
    loss_value = request.param if hasattr(request, "param") else DEFAULT_LOSS
    ret = MagicMock(spec=_Loss)
    ret.return_value = torch.scalar_tensor(loss_value, dtype=torch.float)
    return ret


@pytest.fixture
def logits():
    return torch.randn(8, 3, requires_grad=True)


@pytest.fixture
def logits_rev():
    return torch.randn(8, 3, requires_grad=True)
