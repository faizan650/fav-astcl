import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# Learnable Context Selector
# -------------------------------------------------
class LearnableContextSelector(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: (B, T, N, F)
        x = x.mean(dim=1)                 # (B, N, F)
        h = F.relu(self.fc1(x))
        h = self.fc2(h)                   # (B, N, H)
        sim = torch.matmul(h, h.transpose(1, 2))
        return F.softmax(sim, dim=-1)


# -------------------------------------------------
# FAV-ASTCL Model (Improved)
# -------------------------------------------------
class FAV_ASTCL(nn.Module):
    def __init__(self, in_dim, hidden_dim, pred_len, adj, top_k=10):
        super().__init__()

        self.top_k = top_k

        # Base adjacency
        self.register_buffer(
            "adj",
            torch.tensor(adj, dtype=torch.float32)
        )

        # Learnable fusion gate
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.context = LearnableContextSelector(in_dim, hidden_dim)

        # Temporal encoder
        self.encoder = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.decoder = nn.Linear(hidden_dim, pred_len)

    def forward(self, x):
        """
        x: (B, T, N, F)
        return: (B, N, T_out)
        """

        # -----------------------------
        # Adaptive context graph
        # -----------------------------
        A_learned = self.context(x)  # (B, N, N)

        # -------- Sparse Top-K neighbors --------
        k = min(self.top_k, A_learned.size(-1))
        topk = torch.topk(A_learned, k, dim=-1)
        mask = torch.zeros_like(A_learned)
        mask.scatter_(-1, topk.indices, 1.0)
        A_learned = A_learned * mask

        # -------- Gated fusion --------
        alpha = torch.sigmoid(self.alpha)
        A = alpha * A_learned + (1 - alpha) * self.adj

        # -----------------------------
        # Spatial aggregation + residual
        # -----------------------------
        x_res = x
        x = torch.einsum("bnm,btmf->btnf", A, x)
        x = x + x_res

        # -----------------------------
        # Temporal modeling
        # -----------------------------
        B, T, N, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, F)

        _, h = self.encoder(x)
        out = self.decoder(h.squeeze(0))

        return out.view(B, N, -1)
