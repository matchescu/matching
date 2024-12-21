import pytest
import torch


@pytest.fixture
def input_size(request):
    if hasattr(request, "param") and isinstance(request.param, int):
        return request.param
    return 300


@pytest.fixture
def hidden_size(request):
    if hasattr(request, "param") and isinstance(request.param, int):
        return request.param
    return 128


@pytest.fixture
def output_size(request):
    if hasattr(request, "param") and isinstance(request.param, int):
        return request.param
    return 2


@pytest.fixture
def test_input(input_size):
    return torch.rand(input_size)
