import polars as pl
import torch
from torch.utils.data import Dataset


class PlTorchDataset(Dataset):
    def __init__(self, df: pl.DataFrame, target_col: int | str = "y"):
        self._features = df.drop(target_col).to_numpy()
        self._targets = df[[target_col]].to_numpy()

    def __len__(self):
        return len(self._features)

    def __getitem__(self, idx):
        features = torch.tensor(self._features[idx], dtype=torch.float32)
        targets = torch.tensor(self._targets[idx], dtype=torch.float32)
        return features, targets
