import torch
from torch import nn
import numpy.typing as npt
import numpy as np

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    start_indices = np.random.randint(0, len(dataset) - context_length, size=batch_size)

    x = np.array([dataset[start_idx : start_idx + context_length] for start_idx in start_indices])
    y = np.array([dataset[start_idx + 1 : start_idx + context_length + 1] for start_idx in start_indices])
    return torch.tensor(x, device=device, dtype=torch.long), torch.tensor(y, device=device, dtype=torch.long)