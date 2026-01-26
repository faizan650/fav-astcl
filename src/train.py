import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import load_data, split_data, TrafficDataset
from model import FAV_ASTCL
from metrics import mae, rmse, masked_mape
from utils import load_adj


# -------------------------------------------------
# Configuration
# -------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET = "METR-LA"   # change to "PEMS-BAY" if needed
DATA_PATH = f"../datasets/{DATASET}/data.npz"
ADJ_PATH  = f"../datasets/{DATASET}/adj_mx.pkl"

INPUT_LEN = 12      # 1 hour history
PRED_LEN  = 3       # 15 min forecast
BATCH_SIZE = 16
EPOCHS =30  
LR = 0.001


# -------------------------------------------------
# Load data
# -------------------------------------------------
data = load_data(DATA_PATH)
train_data, _, test_data = split_data(data)

# -------- Normalization --------
mean = train_data.mean()
std = train_data.std() + 1e-5

train_data = (train_data - mean) / std
test_data  = (test_data  - mean) / std


train_loader = DataLoader(
    TrafficDataset(train_data, INPUT_LEN, PRED_LEN),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    TrafficDataset(test_data, INPUT_LEN, PRED_LEN),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# -------------------------------------------------
# Load adjacency
# -------------------------------------------------
adj = load_adj(ADJ_PATH)


# -------------------------------------------------
# Model
# -------------------------------------------------
model = FAV_ASTCL(
    in_dim=1,
    hidden_dim=64,
    pred_len=PRED_LEN,
    adj=adj,
    top_k=10
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)


# -------------------------------------------------
# Training
# -------------------------------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for x, y in train_loader:
        x = x.to(DEVICE).float()
        y = y.to(DEVICE).float()

        optimizer.zero_grad()

        pred = model(x)
        y = y.squeeze(-1).permute(0, 2, 1)

        # -------- Horizon-aware weighted MAE --------
        weights = torch.tensor([1.0, 0.7, 0.4], device=pred.device)
        weights = weights / weights.sum()

        loss_mae = torch.mean(
            weights * torch.mean(torch.abs(pred - y), dim=(0, 1))
        )

        # -------- Temporal smoothness --------
        smooth_loss = torch.mean(torch.abs(pred[:, :, 1:] - pred[:, :, :-1]))

        loss = loss_mae + 0.1 * smooth_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch+1:02d} | Train Loss: {total_loss / len(train_loader):.4f}")


# -------------------------------------------------
# Testing (denormalized metrics)
# -------------------------------------------------
model.eval()
MAE = RMSE = MAPE = 0.0

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE).float()
        y = y.to(DEVICE).float()

        pred = model(x)
        y = y.squeeze(-1).permute(0, 2, 1)

        # -------- Denormalize --------
        pred = pred * std + mean
        y    = y * std + mean

        MAE  += mae(pred, y).item()
        RMSE += rmse(pred, y).item()
        MAPE += masked_mape(pred, y).item()

n = len(test_loader)
print("\nTest Results")
print("MAE :", MAE / n)
print("RMSE:", RMSE / n)
print("MAPE:", MAPE / n)
