import torch
from torch import nn, Tensor
from jaxtyping import Float, Int
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.positionwise_feedforward import SwiGLUFFN
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.multihead_self_attention import MultiHeadSelfAttention
from einops import repeat
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff:int, 
                 dropout: float | None = None, theta: float | None = None, max_seq_len: int | None = None):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, theta=theta, max_seq_len=max_seq_len)
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
    

    def forward(self, x: Float[Tensor, "batch sequence_length d_model"]):
        token_positions = torch.arange(x.shape[-2])
        token_positions = repeat(token_positions, "sequence_length -> batch sequence_length", batch=x.shape[0])
        x = x + self.attn(self.ln1(x), token_positions)
        return x + self.ffn(self.ln2(x))



if __name__=="__main__":
    model = TransformerBlock(d_model=512, n_heads=8, d_ff=2048, theta=10000, max_seq_len=2048)
    # test the parameter names
    for name, param in model.named_parameters():
        print(name)