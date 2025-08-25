import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Sequence, Optional, Union, Dict, Any
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.log_setup import init_logger, get_logger

logger = get_logger(__name__)

class TrainingDataSet(Dataset):
    def __init__(
        self,
        vocab_filepath: Optional[str] = None,
        merges_filepath: Optional[str] = None,
        train_filepath: Optional[str] = None,
        eot_text: Optional[str] = None,
        context_length: Optional[int] = None,
        *,
        save_path: Optional[str] = None,
        load_path: Optional[str] = None,
        force_rebuild: bool = False,
    ):
        if load_path and not force_rebuild and os.path.isfile(load_path):
            ds = self.load(load_path)
            self._adopt(ds)
            return

        assert vocab_filepath and merges_filepath and train_filepath and eot_text is not None and context_length is not None, \
            "When not loading from .pth, all construction arguments must be provided."
        self.token_ids, self.eot_id = self.tokenize_data(
            vocab_filepath=vocab_filepath,
            merges_filepath=merges_filepath,
            train_filepath=train_filepath,
            eot_text=eot_text
        )
        self.context_length = context_length
        self.positions = self.get_positions(token_ids=self.token_ids)

        self.meta: Dict[str, Any] = dict(
            vocab_filepath=os.path.abspath(vocab_filepath),
            merges_filepath=os.path.abspath(merges_filepath),
            train_filepath=os.path.abspath(train_filepath),
            eot_text=eot_text,
            context_length=self.context_length,
            num_tokens=int(self.token_ids.numel()),
            num_positions=int(len(self.positions)),
        )

        # optional: save the data after build
        if save_path:
            self.save(save_path)


    def __len__(self):
        return len(self.positions)


    def __getitem__(self, index):
        start = self.positions[index]
        x = self.token_ids[start : start + self.context_length]
        y = self.token_ids[start + 1 : start + 1 + self.context_length]
        return x, y


    def tokenize_data(
        self, 
        vocab_filepath:str,
        merges_filepath: str,
        train_filepath: str,
        eot_text: str,
        device:str = "cpu",
    ):
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
        token_ids = torch.as_tensor(token_ids, dtype=torch.long, device=device)
        return token_ids, eot_id


    def get_positions(
        self,
        token_ids: Sequence[int],
    ):
        data = torch.as_tensor(token_ids, dtype=torch.long) # CPU

        positions = []
        start = 0
        for pos, tid in enumerate(data):
            if tid == self.eot_id:
                if pos - start > 0:
                    positions.extend([i for i in range(start, pos - self.context_length + 1)]) # eot is included
                start = pos + 1
        if start < len(data):
            positions.extend([i for i in range(start, len(data) - self.context_length)])
        if len(positions) == 0:
            raise ValueError("No valid positions. Check that documents are long enough for the context_length.")
    
        return torch.as_tensor(positions, dtype=torch.long)


    def to_state(self) -> Dict[str, Any]:
        state = dict(
            token_ids=self.token_ids.detach().to("cpu"),
            positions=self.positions.detach().to("cpu"),
            eot_id=int(self.eot_id),
            context_length=int(self.context_length),
            meta=getattr(self, "meta", {}),
        )
        return state


    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "TrainingDataSet":
        self = cls.__new__(cls)  # skip __init__
        self.token_ids = state["token_ids"]
        self.positions = state["positions"]
        self.eot_id = int(state["eot_id"])
        self.context_length = int(state["context_length"])
        self.meta = state.get("meta", {})
        return self


    def save(self, path: str):
        state = self.to_state()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(state, path)
        logger.info("Dataset saved to {}", os.path.abspath(path))


    @classmethod
    def load(cls, path: str) -> "TrainingDataSet":
        state = torch.load(path, map_location="cpu")
        for k in ("token_ids", "positions"):
            if isinstance(state.get(k), torch.Tensor):
                state[k] = state[k].to("cpu")
        ds = cls.from_state(state)
        logger.info("Dataset loaded from {}", os.path.abspath(path))
        return ds


    def _adopt(self, other: "TrainingDataSet"):
        self.token_ids = other.token_ids
        self.positions = other.positions
        self.eot_id = other.eot_id
        self.context_length = other.context_length
        self.meta = getattr(other, "meta", {})


def build_or_load(
    pth_path: str,
    *,
    vocab_filepath: str,
    merges_filepath: str,
    train_filepath: str,
    eot_text: str,
    context_length: int,
    device: str = "cpu",
    force_rebuild: bool = False,
) -> TrainingDataSet:
    if (not force_rebuild) and os.path.isfile(pth_path):
        return TrainingDataSet(load_path=pth_path)
    return TrainingDataSet(
        vocab_filepath=vocab_filepath,
        merges_filepath=merges_filepath,
        train_filepath=train_filepath,
        eot_text=eot_text,
        context_length=context_length,
        save_path=pth_path,
        force_rebuild=force_rebuild,
    )


def test_dataset():
    vocab_filepath="data/owt-vocab.txt"
    merges_filepath="data/owt-merges.txt"
    train_filepath="data/TinyStoriesV2-GPT4-valid-tiny.txt"
    eot_text="<|endoftext|>"
    context_length=6

    dataset_path = "cache/train_dataset.pth"
    dataset = build_or_load(
        dataset_path,
        vocab_filepath=vocab_filepath,
        merges_filepath=merges_filepath,
        train_filepath=train_filepath,
        eot_text=eot_text,
        context_length=context_length,
        device="cpu",
    )

    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_filepath,
        merges_filepath=merges_filepath,
        special_tokens=[eot_text]
    )
    for i in range(min(len(dataset), 10)):
        x, y = dataset[i]
        print(x)
        x_text = tokenizer.decode(x)
        y_text = tokenizer.decode(y)
        print("x_text: ", x_text)
        print("y_text: ", y_text)
        print('-' * 80)
    
    # second time: directly load from .pth
    dataset2 = TrainingDataSet(load_path=dataset_path, device="cpu")
    assert len(dataset2) == len(dataset)
    x0_1, y0_1 = dataset[0]
    x0_2, y0_2 = dataset2[0]
    assert torch.equal(x0_1, x0_2) and torch.equal(y0_1, y0_2)
    print("Reload check passed.")

if __name__ == "__main__":
    test_dataset()