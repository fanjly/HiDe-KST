from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from einops import reduce


def default_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_memory_stats(device: torch.device = default_device()):
    alloc = torch.cuda.memory_allocated(device) / (1024**3)
    reserv = torch.cuda.memory_reserved(device) / (1024**3)
    max_alloc = torch.cuda.max_memory_allocated(device) / (1024**3)
    return alloc, reserv, max_alloc


def to_tensor(array: np.ndarray, non_blocking: Optional[bool] = True, to_device: Optional[torch.device] = default_device()) -> Tensor:
    return torch.as_tensor(array, dtype=torch.float32).to(to_device, non_blocking=non_blocking)


def divide_no_nan(a, b):
    mask = b == .0
    b[mask] = 1.
    result = a / b
    result[mask] = .0
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


def scale(x: Tensor,
          scaling_factor: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    if scaling_factor is not None:
        x = x / scaling_factor
        return x, scaling_factor

    scaling_factor = reduce(torch.abs(x).data, 'b t d -> b 1 d', 'mean')
    scaling_factor[scaling_factor == 0.0] = 1.0
    x = x / scaling_factor
    return x, scaling_factor


def descale(forecast: Tensor, scaling_factor: Tensor) -> Tensor:
    return forecast * scaling_factor
