import torch
from torch import nn
from einops import einsum, repeat

class RMSNorm2(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module. This function should accept the following parameters:
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        def forward(self, x: torch.Tensor) -> torch.Tensor Process an input tensor of shape
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.d_model = d_model
        self.eps = eps


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape
        (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        ms = einsum(x*x/self.d_model, "batch_size sequence_length d_model -> batch_size sequence_length")
        rms = torch.sqrt(ms + self.eps)
        rms = repeat(rms, "batch_size sequence_length-> batch_size sequence_length d_model", d_model=self.d_model)
        x = einsum(x/rms, self.weight, "batch_size sequence_length d_model, d_model -> batch_size sequence_length d_model")
        return x.to(in_dtype)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Root Mean Square Layer Normalization.
        Args:
            d_model (int): Dimension of the input feature.
            eps (float): Small constant to avoid division by zero.
            device (optional): Device to place the parameters.
            dtype (optional): Data type of the parameters.
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))  # Learnable scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the RMS norm (no centering)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # L2 norm along the last dimension
        x_normed = x / rms        # Normalize
        return (x_normed * self.weight).to(in_dtype)            # Apply learnable scale

