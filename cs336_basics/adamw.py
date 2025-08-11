import torch
from torch import nn
from collections.abc import Callable, Iterable
from typing import Optional
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults=defaults)


    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data
                m = state.get("m", torch.zeros_like(grad))
                m = betas[0] * m + (1 - betas[0]) * grad
                v = state.get("v", torch.zeros_like(grad))
                v = betas[1] * v + (1 - betas[1]) * (grad**2)

                t = state.get("t", 1)
                alpha_t = lr * math.sqrt(1 - betas[1]**t) / (1 - betas[0]**t)                
                p.data -= alpha_t * m / (torch.sqrt(v) + eps) 
                p.data = p.data * (1 - lr * weight_decay)
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1

