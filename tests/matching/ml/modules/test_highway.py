import pytest

from matchescu.matching.ml.modules import HighwayNetwork
from matching.ml.modules.conftest import input_size, hidden_size, output_size, test_input


@pytest.fixture
def highway_net(request, input_size, hidden_size, output_size):
    kwargs = {}
    if hasattr(request, "param") and isinstance(request.param, dict):
        kwargs = request.param
    return HighwayNetwork(input_size, hidden_size, output_size, **kwargs)


@pytest.mark.parametrize("input_size,output_size", [(32, 20), (128, 2)], indirect=True)
def test_init_args(input_size, output_size, test_input, highway_net):
    out = highway_net.forward(test_input)

    assert out.shape[0] == output_size
