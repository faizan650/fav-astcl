import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import pickle

from torch.utils.data import DataLoader

import metrics
from model import FAV_ASTCL
from data_loader import TrafficDataset

# -----------------------------
# Device
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load METR-LA traffic data
# -----------------------------
traffic_path = "../datasets/METR-LA/data.npz"
traffic_data = np.load(traffic_path)
traffic = traffic_data["data"]   # (T, N, 1)

# -----------------------------
# Load Hyderabad exogenous context
# -----------------------------
exo_path = "../datasets/exo_hyderabad.npy"
exo = np.load(exo_path)          # (T, E)

# -----------------------------
# Load adjacency matrix
# -----------------------------
adj_path = "../datasets/METR-LA/adj_mx.pkl"

with open(adj_path, "rb") as f:
    adj_data = pickle.load(f)

# Handle (adj, ids) or (adj, ids, mapping)
if isinstance(adj_data, tuple):
    adj = adj_data[0]
else:
    adj = adj_data

adj = torch.tensor(adj, dtype=torch.float32)
print("Adjacency shape:", adj.shape)

# -----------------------------
# Normalize traffic
# -----------------------------
mean = traffic.mean()
std = traffic.std() + 1e-5
traffic = (traffic - mean) / std

# -----------------------------
# Train / Test split
# -----------------------------
split = int(0.8 * len(traffic))

train_dataset = TrafficDataset(
    traffic=traffic[:split],
    exo=exo[:split],
    input_len=12,
    pred_len=3
)

test_dataset = TrafficDataset(
    traffic=traffic[split:],
    exo=exo[split:],
    input_len=12,
    pred_len=3
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    drop_last=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    drop_last=True
)

# -----------------------------
# Model
# -----------------------------
model = FAV_ASTCL(adj=adj).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# Training loop
# -----------------------------
EPOCHS = 60

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for x, exo_batch, y in train_loader:
        x = x.to(DEVICE)               # (B, T, N, 1)
        exo_batch = exo_batch.to(DEVICE)  # (B, T, E)
        y = y.to(DEVICE)               # (B, P, N, 1)

        pred = model(x, exo_batch)     # (B, N, P)

        # Align target shape
        y = y.squeeze(-1).permute(0, 2, 1)  # (B, N, P)

        loss = metrics.mae(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1:03d} | Train MAE: {epoch_loss / len(train_loader):.4f}")

# -----------------------------
# Final Evaluation (TEST MAE)
# -----------------------------
model.eval()
test_mae = 0.0

with torch.no_grad():
    for x, exo_batch, y in test_loader:
        x = x.to(DEVICE)
        exo_batch = exo_batch.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x, exo_batch)
        y = y.squeeze(-1).permute(0, 2, 1)

        test_mae += metrics.mae(pred, y).item()

test_mae /= len(test_loader)

print("=" * 60)
print(f"FINAL TEST MAE: {test_mae:.4f}")
print("=" * 60)

# -----------------------------
# Save model
# -----------------------------
os.makedirs("../results", exist_ok=True)
torch.save(model.state_dict(), "../results/fav_astcl.pth")

print("Training complete.")
print("Model saved at: results/fav_astcl.pth")
