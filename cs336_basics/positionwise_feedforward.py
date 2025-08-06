import torch
from torch import nn
from cs336_basics.linear import Linear

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model:int, d_ff:int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.w1 = Linear(in_features=d_model, out_features=d_ff, **factory_kwargs)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, **factory_kwargs)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, **factory_kwargs)

    
    def forward(self, x: torch.Tensor):
        """
        Process an input tensor of shape
        (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """
        w1x = self.w1(x)
        silu = w1x * torch.sigmoid(w1x)
        w3x = self.w3(x)
        return self.w2(silu * w3x)
        