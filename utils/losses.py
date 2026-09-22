from typing import Optional, Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def get_loss_fn(loss_name: str,
                delta: Optional[float] = 1.0,
                beta: Optional[float] = 1.0) -> Callable:
    return {'mse': F.mse_loss,
            'mae': F.l1_loss,
            'huber': partial(F.huber_loss, delta=delta),
            'smooth_l1': partial(F.smooth_l1_loss, beta=beta)}[loss_name]


class MultiTaskLoss(nn.Module):
    """
    Dynamically weights multiple loss functions using homoscedastic uncertainty.
    Theory: Maximizing Gaussian likelihood with learnable noise scalars.
    Reference: Kendall, A. et al.
               "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics".
    """
    def __init__(self, num_tasks):
        super(MultiTaskLoss, self).__init__()
        self.num_tasks = num_tasks
        if num_tasks > 1:
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        if self.num_tasks == 1:
            return losses[0]

        final_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            final_loss += 0.5 * precision * loss + 0.5 * self.log_vars[i]
        return final_loss
