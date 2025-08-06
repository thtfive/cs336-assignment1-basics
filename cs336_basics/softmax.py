import torch
from torch import nn
from jaxtyping import Float, Int


def softmax(x: Float[torch.Tensor, " ..."], dim):
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_norm = x - x_max
    x_exp = torch.exp(x_norm)
    x_softmax = x_exp/torch.sum(x_exp, dim=dim, keepdim=True)
    return x_softmax