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
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint, load_checkpoint_model
from cs336_basics.data_loader import TrainingDataSet, build_or_load
from torch.utils.data import Dataset, DataLoader
from cs336_basics.softmax import softmax

def parse_args():
    # parse parameters
    parser = argparse.ArgumentParser(description='Train TransformerLM')

    parser.add_argument("--input", type=str, default="Tom likes swimming in the river")

    # TransformLM model parameters
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=64)
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
    parser.add_argument("--train_filepath", type=str, default="data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--preprocessing", type=bool, default=True)
    parser.add_argument("--dataset_path", type=str, default="cache/train_dataset.pth")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--betas", type=tuple, default=(0.99, 0.999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # checkpoints
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    parser.add_argument("--model_path", type=str, default="checkpoints/model_step6000.pth")

    # parse parameters
    params = parser.parse_args()
    return params


def inference(params):
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
    load_checkpoint_model(
        src=params.model_path,
        model=model
    )
    model.to(device)

    eot_token_id = tokenizer.encode(params.eot_text)[0]

    text = params.input
    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids, dtype=torch.long) 
    input_ids = input_ids.unsqueeze(0)
    print("input_ids shape: ", input_ids.shape)
    max_length = params.context_length - input_ids.shape[1]
    generated_sequence = input_ids
    for step in range(max_length):
        # 1. model inference, get logits of next token
        output = model(input_ids)
        next_token_logits = output[:, -1, :]
        next_token_probs = softmax(next_token_logits, dim=-1) # get y_id

        # 2. get next token
        next_token_id = torch.argmax(next_token_probs, dim=-1, keepdim=True) # keepdim
        next_token_id = next_token_id.to(input_ids.device)

        # 3. format next input
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        generated_sequence = torch.cat([generated_sequence, next_token_id], dim=-1)

        # 4. check if it's end
        if next_token_id == eot_token_id:
            break
    
    generated_sequence = generated_sequence.squeeze(0)
    print("generated_sequence shape: ", generated_sequence.shape)
    print(tokenizer.decode(generated_sequence))




def log_params(params):
    # Convert Namespace -> dict
    params_dict = vars(params)
    logger.info("Parameters:\n{}", json.dumps(params_dict, indent=4))


if __name__ == "__main__":
    params = parse_args()
    init_logger(log_dir="logs", level="DEBUG")
    logger = get_logger(__name__)
    logger.info("Start inferencing")
    log_params(params)
    inference(params)


