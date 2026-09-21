import torch
from torch import Tensor


def decode_logits(logits: Tensor) -> Tensor:
    """``(N, 4)`` head logits -> published labels ``(N, )`` in ``{0, 1, 2}``.

    Per-bit ``argmax``, then the hierarchical rule: p=0 -> 0, else q=1 -> 1,
    else 2.
    """
    bits = logits.view(-1, 2, 2).argmax(dim=-1)
    p, q = bits[:, 0], bits[:, 1]
    labels = torch.full_like(p, 2)  # init with 2
    labels[p == 0] = 0  # overwrite with 0 where p is 0
    labels[(p == 1) & (q == 1)] = 1  # overwrite with 1 where p and q are both 1
    return labels


def per_class_probabilities(logits: Tensor) -> Tensor:
    """``(N, 4)`` head logits -> ``(N, 3)`` scores over published classes."""
    first_bit = logits.view(-1, 2, 2).softmax(dim=-1)[..., 1]
    p, q = first_bit.unbind(dim=-1)
    return torch.stack((1.0 - p, p * q, p * (1.0 - q)), dim=-1)
