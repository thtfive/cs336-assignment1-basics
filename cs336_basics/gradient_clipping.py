import torch
from torch import nn
from collections.abc import Iterable
import math

def gradient_clipping(
        parameters: Iterable[torch.nn.Parameter],
        max_l2_norm: float,
        eps: float = 1e-6
    ):
    # Flatten gradients into one vector to compute the global L2 norm
    grads = []
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
    if not grads:
        return  # no gradients to clip

    grad_vec = torch.cat(grads)
    total_norm = torch.norm(grad_vec, p=2)  # L2 norm

    if total_norm >= max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale)