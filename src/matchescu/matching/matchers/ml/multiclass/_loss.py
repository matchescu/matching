import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss


def directional_margin_loss(
    logits_fwd: Tensor, logits_rev: Tensor, y: Tensor, margin: float = 2.0
) -> Tensor:
    m = y == 2
    if not m.any():
        return logits_fwd.new_zeros(())
    gap = logits_fwd[m, 2] - logits_rev[m, 2]
    return F.relu(margin - gap).mean()


def order_energy(u: Tensor, v: Tensor) -> Tensor:
    return F.relu(v - u).pow(2).sum(-1)


def order_loss(ea: Tensor, eb: Tensor, y: Tensor, margin: float = 1.0) -> Tensor:
    fwd, rev = order_energy(ea, eb), order_energy(eb, ea)
    terms = []
    m2, m0 = y == 2, y == 0
    if m2.any():
        terms.append(fwd[m2].mean() + F.relu(margin - rev[m2]).mean())
    if m0.any():
        terms.append(F.relu(margin - fwd[m0]).mean() + F.relu(margin - rev[m0]).mean())
    return sum(terms, ea.new_zeros(())) / max(len(terms), 1)


class FocalLoss(_Loss):
    """
    Multiclass focal loss implementation.

    This type of loss function is described in Lin et al., "Focal Loss for Dense
    Object Detection", ICCV 2017 https://arxiv.org/abs/1708.02002. The formula
    is:

    .. math:: L = -alpha * (1 - x)^gamma * log(x)
        :label: focal loss

    :param alpha (Tensor, optional): Per-class weights (shape: [num_classes]).
        Analogous to `weight` in ``nn.CrossEntropyLoss``.
    :param gamma (float): Focusing parameter. gamma=0 reduces to weighted CE.
        gamma=2 is recommended in the paper.
    :param reduction (str): 'mean' | 'sum' | 'none'
    :param ignore_index (int): Class index to ignore (mirrors nn.CrossEntropyLoss).
    """

    def __init__(
        self,
        alpha: Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction="none", ignore_index=ignore_index
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # --- Step 1: log-softmax
        log_p = F.log_softmax(x, dim=-1)  # (N, C)

        # --- Step 2: weighted NLL loss = -alpha_t * log(pt)
        ce = self.nll_loss(log_p, y)  # (N,)

        # --- Step 3: extract log(pt) for the TRUE class of each sample
        all_rows = torch.arange(len(x), device=x.device)
        log_pt = log_p[all_rows, y]  # (N,)

        # --- Step 4: focusing term = (1 - x)^gamma
        pt = log_pt.exp()  # (N,)
        focal_term = (1 - pt) ** self.gamma  # (N,)

        # --- Step 5: full focal loss = focal_term * CE
        loss = focal_term * ce  # (N,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # 'none'
