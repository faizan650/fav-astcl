import torch
import torch.nn as nn
import torch.nn.functional as F

class ExogenousEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))
