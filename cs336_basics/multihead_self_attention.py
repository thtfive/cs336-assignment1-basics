import torch
from torch import nn, Tensor
from jaxtyping import Float, Int
from cs336_basics.rope import RotaryPositionEmbedding
from cs336_basics.scaled_dot_product_attention import ScaledDotProductAttention
from cs336_basics.linear import Linear
from einops import rearrange
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, theta: float | None = None, max_seq_len: int | None = None):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        assert self.d_k * n_heads == d_model, f"d_model ({d_model}) is not divisible by n_heads ({n_heads})."
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention()

        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)
        else: 
            self.rope = None
 

    def forward(self, x: Float[Tensor, "... seq_len d_model"], 
                token_positions: Int[Tensor, "... seq_len"] | None = None) -> Float[Tensor, "... seq_len d_model"]:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q_heads = rearrange(q, "batch seq (heads d) -> batch heads seq d", heads=self.n_heads)
        k_heads = rearrange(k, "batch seq (heads d) -> batch heads seq d", heads=self.n_heads)
        v_heads = rearrange(v, "batch seq (heads d) -> batch heads seq d", heads=self.n_heads)

        
        if self.rope is not None:
            q_heads = self.rope(q_heads, token_positions=token_positions)
            k_heads = self.rope(k_heads, token_positions=token_positions)

        mask = torch.tril(torch.ones(1, 1, q.shape[-2], q.shape[-2])).bool()
        o_heads = self.attn(Q=q_heads, K=k_heads, V=v_heads, mask=mask)
        o = rearrange(o_heads, "batch heads seq d -> batch seq (heads d)")
        return self.output_proj(o)
