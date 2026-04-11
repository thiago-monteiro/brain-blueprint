from __future__ import annotations

import torch
from torch.utils.data import Dataset


def scramble_tensor_time(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 2:
        raise ValueError("Expected a time-major tensor with at least 2 dimensions.")
    permutation = torch.randperm(tensor.shape[0])
    return tensor[permutation]


class TemporallyScrambledDataset(Dataset):
    def __init__(self, base_dataset: Dataset) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        item = dict(self.base_dataset[idx])
        item["frames"] = scramble_tensor_time(item["frames"])
        return item
