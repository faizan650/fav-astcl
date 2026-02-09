import torch

def mae(pred, true):
    return torch.mean(torch.abs(pred - true))

def rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2))

def masked_mape(pred, true, eps=1e-5):
    mask = true.abs() > eps
    return (torch.abs((pred - true) / (true + eps)) * mask).sum() / mask.sum() * 100
