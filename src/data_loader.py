import numpy as np
import torch
from torch.utils.data import Dataset

class TrafficDataset(Dataset):
    def __init__(self, traffic, exo, input_len=12, pred_len=3):
        self.traffic = traffic
        self.exo = exo
        self.input_len = input_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.traffic) - self.input_len - self.pred_len

    def __getitem__(self, idx):
        x = self.traffic[idx:idx+self.input_len]
        y = self.traffic[idx+self.input_len:idx+self.input_len+self.pred_len]
        e = self.exo[idx:idx+self.input_len]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(e, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
