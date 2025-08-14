import torch
from torch import nn
from cs336_basics.transformer import TransformerLM
import argparse


def parse_args():
    # parse parameters
    parser = argparse.ArgumentParser(description='Train TransformerLM')

    # TransformLM model parameters
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # parse parameters
    params = parser.parse_args()
    return params


def train(params):
    model = TransformerLM(
        vocab_size = params.vocab_size,
        context_length = params.context_length,
        d_model = params.d_model,
        num_layers = params.num_layers,
        num_heads = params.num_heads,
        d_ff = params.d_ff,
        rope_theta = params.rope_theta
    )
    



if __name__ == "__main__":
    params = parse_args()
    train(params)


