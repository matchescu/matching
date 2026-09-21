from torch import Tensor, cat, nn

from matchescu.matching.matchers.ml.multiclass._types import HeadType


class AsymmetricHead(nn.Module):
    def __init__(
        self, input_size: int, rank: int = 256, head_type: HeadType = HeadType.NONE
    ) -> None:
        """Init the layer that introduces asymmetry in the classifier.

        When ``head_type`` is set to ``None`` this will simply return the input.

        :param input_size: size of the input tensors
        :param rank: size of the linear layers that provide the bi-linear differentiation signal
        :param head_type: the type of asymmetric head to initialize
        """
        super().__init__()

        self._head_type = head_type
        self._u = self._v = None

        match head_type:
            case HeadType.BILINEAR:
                self._output_size = 3 * input_size + rank
                self._u = nn.Linear(input_size, rank, bias=False)
                self._v = nn.Linear(input_size, rank, bias=False)
            case HeadType.DIFF:
                self._output_size = 4 * input_size
            case _:
                self._output_size = 2 * input_size

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, enc_a: Tensor, enc_b: Tensor) -> Tensor:
        out_features = [enc_a, enc_b]
        match self._head_type:
            case HeadType.BILINEAR:
                out_features.append((enc_a - enc_b).abs())
                ua = self._u(enc_a)
                ub = self._u(enc_b)
                va = self._v(enc_a)
                vb = self._v(enc_b)
                skew_diff = (ua * vb) - (ub * va)
                out_features.append(skew_diff)
            case HeadType.DIFF:
                out_features.append(enc_a - enc_b)
                out_features.append(enc_a * enc_b)
        return cat(out_features, dim=-1)
