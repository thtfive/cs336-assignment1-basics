import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int


def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    x_max = torch.max(inputs, dim = -1, keepdim=True).values
    x = inputs - x_max
    loss = -torch.gather(x, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) + torch.log(torch.sum(torch.exp(x), dim = -1))
    return loss.mean()