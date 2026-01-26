import numpy as np
import torch
from torch.utils.data import Dataset


class TrafficDataset(Dataset):
    """
    Sliding window dataset for traffic forecasting.

    Input  : (T, N, F)
    Output :
        x -> (input_len, N, F)
        y -> (pred_len,  N, F)
    """

    def __init__(self, data, input_len=12, pred_len=3):
        self.data = data.astype(np.float32)  # ✅ force float32
        self.input_len = input_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.input_len - self.pred_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_len]
        y = self.data[idx + self.input_len : idx + self.input_len + self.pred_len]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )


def load_data(path):
    """
    Load data.npz file.
    Expected key: 'data' with shape (T, N, F)
    """
    return np.load(path)["data"].astype(np.float32)


def split_data(data, train_ratio=0.7, val_ratio=0.1):
    """
    Time-based split (ASTCL-style).
    """
    T = data.shape[0]
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data   = data[train_end:val_end]
    test_data  = data[val_end:]

    return train_data, val_data, test_data
