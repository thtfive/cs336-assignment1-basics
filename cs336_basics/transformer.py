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


class TransformerLM(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float
    ):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(
                d_model=d_model,
                n_heads=num_heads,
                d_ff=d_ff,
                theta=rope_theta,
                max_seq_len=context_length
                ) for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)


    def forward(self, in_indices: Int[Tensor, " batch_size sequence_length"]):
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x


if __name__=="__main__":
    # model = TransformerBlock(d_model=512, n_heads=8, d_ff=2048, theta=10000, max_seq_len=2048)
    # # test the parameter names
    # for name, param in model.named_parameters():
    #     print(name)
    
    # test the parameter names for TransformLM
    model = TransformerLM(
        vocab_size=1000,
        context_length=1024,
        d_model=512,
        num_layers=12,
        num_heads=8,
        d_ff=2048, 
        rope_theta=10000
    )
    for name, param in model.named_parameters():
        print(name)