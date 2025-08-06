import torch
from torch import nn
import numpy as np


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        i = torch.arange(0, max_seq_len)
        k = torch.arange(0, d_k, 2)
        inv_freqs = 1/(theta ** (k / d_k))
        
        freqs = torch.outer(i, inv_freqs)
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_complex", freqs_complex, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process
        an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note
        that you should tolerate x with an arbitrary number of batch dimensions. You should assume
        that the token positions are a tensor of shape (..., seq_len) specifying the token positions of
        x along the sequence dimension.
        """
        freqs = self.freqs_complex[token_positions]
        x_complex = torch.view_as_complex(
            x.float().reshape(*x.shape[:-1], -1, 2)
        )
        result_complex = x_complex * freqs
        result = torch.view_as_real(result_complex)

        return result.flatten(start_dim=-2, end_dim=-1).type_as(x)