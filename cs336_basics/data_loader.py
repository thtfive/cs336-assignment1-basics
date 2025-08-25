import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Sequence
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.log_setup import init_logger, get_logger

logger = get_logger(__name__)

class TrainingDataSet(Dataset):
    def __init__(
        self,
        vocab_filepath,
        merges_filepath,
        train_filepath,
        eot_text,
        context_length
    ):
        # if params.preprocessing:
        self.token_ids, self.eot_id = self.tokenize_data(
            vocab_filepath=vocab_filepath,
            merges_filepath=merges_filepath,
            train_filepath=train_filepath,
            eot_text=eot_text
        )
        self.context_length = context_length
        self.positions = self.get_positions(token_ids=self.token_ids)


    def __len__(self):
        return len(self.positions)


    def __getitem__(self, index):
        start = self.positions[index]
        x = self.token_ids[start : start + self.context_length]
        y = self.token_ids[start + 1 : start + 1 + self.context_length]
        return x, y


    def tokenize_data(self, vocab_filepath, merges_filepath, train_filepath, eot_text):
        tokenizer = Tokenizer.from_files(
            vocab_filepath=vocab_filepath,
            merges_filepath=merges_filepath,
            special_tokens=[eot_text]
        )
        # load training data
        text = ""
        with open(train_filepath, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            text = "".join(lines)

        # sanity check on eot_text
        logger.info("EOT_TEXT: {}", eot_text)
        eot_id = tokenizer.encode(eot_text)[0]
        logger.info("EOT_ID: {}", eot_id)
        assert eot_text == tokenizer.decode([eot_id]), "EOT_TEXT {EOT_TEXT} is not valid"

        # do tokenizer to training data
        token_ids = np.array(tokenizer.encode(text))
        return token_ids, eot_id


    def get_positions(
        self,
        token_ids: Sequence[int],
        device: str = "cpu",
    ):
        data = torch.as_tensor(token_ids, dtype=torch.long)

        positions = []
        start = 0
        for pos, tid in enumerate(data):
            if tid == self.eot_id:
                if pos - start > 0:
                    positions.extend([i for i in range(start, pos - self.context_length + 1)]) # eot is included
                start = pos + 1
        if start < len(data):
            positions.extend([i for i in range(start, len(data) - self.context_length)])
        return positions


def test_dataset():
    vocab_filepath="data/owt-vocab.txt"
    merges_filepath="data/owt-merges.txt"
    train_filepath="data/TinyStoriesV2-GPT4-valid-tiny.txt"
    eot_text="<|endoftext|>"
    context_length=6
    dataset = TrainingDataSet(
        vocab_filepath=vocab_filepath,
        merges_filepath=merges_filepath,
        train_filepath=train_filepath,
        eot_text=eot_text,
        context_length=context_length
    )

    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_filepath,
        merges_filepath=merges_filepath,
        special_tokens=[eot_text]
    )
    for i in range(len(dataset)):
        x, y = dataset[i]
        x_text = tokenizer.decode(x)
        y_text = tokenizer.decode(y)
        print("x_text: ", x_text)
        print("y_text: ", y_text)
        print('-'*80)

if __name__ == "__main__":
    test_dataset()