import pytest
import torch

from matchescu.matching.ml.modules import HighwayMatchClassifier


@pytest.fixture
def match_classifier(request, input_size, hidden_size):
    kwargs = {}
    if hasattr(request, "param") and isinstance(request.param, dict):
        kwargs = request.param
    return HighwayMatchClassifier(input_size, hidden_size, **kwargs)


@pytest.mark.parametrize("input_size,hidden_size", [(32, 20), (128, 2)], indirect=True)
def test_default_init_args(match_classifier, input_size, test_input, hidden_size):
    out = match_classifier.forward(test_input)

    assert out.shape == (1, 2)


@pytest.mark.parametrize("input_size,hidden_size", [(32, 20), (128, 2)], indirect=True)
def test_argmax(match_classifier, input_size, test_input, hidden_size):
    out = match_classifier.forward(test_input)

    assert torch.argmax(out, dim=1).item() in {0, 1}
