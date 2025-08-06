import torch
from torch import nn, Tensor
from jaxtyping import Float
from cs336_basics.softmax import softmax
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Float[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, " ... queries d_v"]:
        d_k = K.shape[-1]
        dot_product = Q @ K.mT/math.sqrt(d_k)

        if mask is not None:
            dot_product = dot_product.masked_fill(~mask, float('-inf'))

        return softmax(dot_product, -1) @ V
