import torch
from torch import nn
import argparse
import json
import numpy as np
from typing import Sequence

from cs336_basics.transformer import TransformerLM
from cs336_basics.log_setup import init_logger, get_logger
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.data_loading import get_batch
from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.data_loader import TrainingDataSet, build_or_load
from torch.utils.data import Dataset, DataLoader

def parse_args():
    # parse parameters
    parser = argparse.ArgumentParser(description='Train TransformerLM')

    # TransformLM model parameters
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Tokenizer parameters
    parser.add_argument("--vocab_filepath", type=str, default="data/owt-vocab.txt")
    parser.add_argument("--merges_filepath", type=str, default="data/owt-merges.txt")
    parser.add_argument("--eot_text", type=str, default="<|endoftext|>")

    # Data 
    parser.add_argument("--train_filepath", type=str, default="data/TinyStoriesV2-GPT4-valid-tiny.txt")
    parser.add_argument("--preprocessing", type=bool, default=True)
    parser.add_argument("--dataset_path", type=str, default="cache/train_dataset.pth")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--betas", type=tuple, default=(0.99, 0.999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # checkpoints
    parser.add_argument("--checkpoint_dir", type=str, default="logs/checkpoints/")

    # parse parameters
    params = parser.parse_args()
    return params


def save_training_checkpoint(model, optimizer, iteration, checkpoint_path):
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=iteration,
        out=checkpoint_path
    )
    logger.info("Save checkpoint to {path}", path=checkpoint_path)


def train(params):
    tokenizer = Tokenizer.from_files(
        vocab_filepath=params.vocab_filepath,
        merges_filepath=params.merges_filepath,
    )

    logger.info("Vocab size: {}", len(tokenizer.vocab))
    params.vocab_size = len(tokenizer.vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = TransformerLM(
        vocab_size = params.vocab_size,
        context_length = params.context_length,
        d_model = params.d_model,
        num_layers = params.num_layers,
        num_heads = params.num_heads,
        d_ff = params.d_ff,
        rope_theta = params.rope_theta
    )
    model.to(device)

    # load training data
    train_dataset = build_or_load(
        pth_path=params.dataset_path,
        vocab_filepath=params.vocab_filepath,
        merges_filepath=params.merges_filepath,
        train_filepath=params.train_filepath,
        eot_text=params.eot_text,
        context_length=params.context_length,
        device=device
    )
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

    # training
    optim = AdamW(
        model.parameters(),
        lr=params.lr,
        betas=params.betas,
        eps=params.eps,
        weight_decay=params.weight_decay
    )

    iteration = 0
    for i in range(10000):
        iteration = i
        x, targets = next(iter(loader))
        x = x.to(device)
        targets = targets.to(device)
        
        optim.zero_grad()
        y = model(x)
        loss = cross_entropy(y, targets)

        loss.backward()
        optim.step()
        if i % 10 == 0:
            logger.info("Step {i}, Loss:{loss:.4f}", i=i, loss=loss.item())
        if i % 1000 == 0:
            checkpoint_path = params.checkpoint_dir + "model_step{step}.pth".format(step=iteration)
            save_training_checkpoint(
                model=model,
                optimizer=optim,
                iteration=iteration,
                checkpoint_path=checkpoint_path
            )


    # save checkpoint
    checkpoint_path = params.checkpoint_dir + "model_step{step}.pth".format(step=100)
    save_training_checkpoint(
        model=model,
        optimizer=optim,
        iteration=iteration,
        checkpoint_path=checkpoint_path
    )


def log_params(params):
    # Convert Namespace -> dict
    params_dict = vars(params)
    logger.info("Parameters:\n{}", json.dumps(params_dict, indent=4))


if __name__ == "__main__":
    params = parse_args()
    init_logger(log_dir="logs", level="DEBUG")
    logger = get_logger(__name__)
    logger.info("Start training")
    log_params(params)
    train(params)


