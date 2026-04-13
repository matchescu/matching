from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss


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
        alpha: Optional[Tensor] = None,
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
