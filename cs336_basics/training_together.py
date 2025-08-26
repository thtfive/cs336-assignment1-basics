import torch
from torch import nn
import argparse
import json
import numpy as np
from typing import Sequence
import wandb
import os
from datetime import datetime

from cs336_basics.transformer import TransformerLM
from cs336_basics.log_setup import init_logger, get_logger
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.data_loading import get_batch
from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.data_loader import TrainingDataSet, build_or_load
from torch.utils.data import Dataset, DataLoader


wandb_api_key = os.environ["WANDB_API_KEY"]
wandb.login(key=wandb_api_key)

def parse_args():
    # parse parameters
    parser = argparse.ArgumentParser(description='Train TransformerLM')

    # Experiment name
    parser.add_argument("--exp_name", type=str, default="train gpt")

    # TransformLM model parameters
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Tokenizer parameters
    parser.add_argument("--vocab_filepath", type=str, default="data/TinyStoriesV2-GPT4-vocab.txt")
    parser.add_argument("--merges_filepath", type=str, default="data/TinyStoriesV2-GPT4-merges.txt")
    parser.add_argument("--eot_text", type=str, default="<|endoftext|>")

    # Data 
    parser.add_argument("--train_filepath", type=str, default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--preprocessing", type=bool, default=True)
    parser.add_argument("--train_dataset_path", type=str, default="cache/train_dataset.pth")
    parser.add_argument("--valid_filepath", type=str, default="data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--valid_dataset_path", type=str, default="cache/valid_dataset.pth")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--betas", type=tuple, default=(0.99, 0.999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=1)

    # checkpoints
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")

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


def format_param_count(num_params):
    """convert params number to human-friendly format"""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
      return f"{num_params}"


def eval_model(model, data_loader, iteration):
    model.eval()
    total_loss = 0.0
    logger.info("Start Step {i} Validation...", i=iteration)
    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    logger.info("Step {i}, Validation Loss:{loss:.4f}", i=iteration, loss=avg_loss)
    wandb.log({"step_validation_loss": loss.item()}, step=iteration)


def loss_fn(inputs, targets):
    return cross_entropy(inputs=inputs, targets=targets)


def train(params):
    tokenizer = Tokenizer.from_files(
        vocab_filepath=params.vocab_filepath,
        merges_filepath=params.merges_filepath,
    )

    logger.info("Vocab size: {}", len(tokenizer.vocab))
    params.vocab_size = len(tokenizer.vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = f"{params.exp_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wandb.init(
        project="cs336-project",  # Project name (required, will create or reuse this project in W&B)
        name=run_name,     # Experiment name (optional, random name will be generated if not set)
        config=vars(params),                # Record hyperparameters (dictionary format)
    )

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
    wandb.watch(model, log="all", log_freq=10)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params} ({format_param_count(total_params)})")
    logger.info(f"Total trainable params: {trainable_params}")

    # load training data
    train_dataset = build_or_load(
        pth_path=params.train_dataset_path,
        vocab_filepath=params.vocab_filepath,
        merges_filepath=params.merges_filepath,
        train_filepath=params.train_filepath,
        eot_text=params.eot_text,
        context_length=params.context_length,
    )
    train_data_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
    # load valid data
    valid_dataset = build_or_load(
        pth_path=params.valid_dataset_path,
        vocab_filepath=params.vocab_filepath,
        merges_filepath=params.merges_filepath,
        train_filepath=params.valid_filepath,
        eot_text=params.eot_text,
        context_length=params.context_length,
    )
    valid_data_loader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)

    # training
    optim = AdamW(
        model.parameters(),
        lr=params.lr,
        betas=params.betas,
        eps=params.eps,
        weight_decay=params.weight_decay
    )

    iteration = 0
    for iteration in range(10000):
        x, targets = next(iter(train_data_loader))
        x = x.to(device)
        targets = targets.to(device)
        
        optim.zero_grad()
        y = model(x)
        loss = loss_fn(y, targets)

        loss.backward()
        optim.step()
        if iteration % 10 == 0:
            logger.info("Step {i}, Loss:{loss:.4f}", i=iteration, loss=loss.item())
            wandb.log({"step_loss": loss.item()}, step=iteration)
        if iteration % 1000 == 0:
            eval_model(model, valid_data_loader, iteration)
            checkpoint_path = params.checkpoint_dir + "model_step{step}.pth".format(step=iteration)
            save_training_checkpoint(
                model=model,
                optimizer=optim,
                iteration=iteration,
                checkpoint_path=checkpoint_path
            )

    # save checkpoint
    checkpoint_path = params.checkpoint_dir + "model_step{step}.pth".format(step=iteration)
    save_training_checkpoint(
        model=model,
        optimizer=optim,
        iteration=iteration,
        checkpoint_path=checkpoint_path
    )
    wandb.save("model_final.pth")
    wandb.finish()


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


