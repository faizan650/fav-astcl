import torch
import torch.nn as nn
import torch.nn.functional as F
from exogenous import ExogenousEncoder


class LearnableContextSelector(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: (B, T, N, 1)
        x = x.mean(dim=1)  # (B, N, 1)
        h = self.fc2(F.relu(self.fc1(x)))
        sim = torch.matmul(h, h.transpose(1, 2))
        return F.softmax(sim, dim=-1)


class FAV_ASTCL(nn.Module):
    def __init__(self, adj):
        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.register_buffer("adj", adj)

        self.context = LearnableContextSelector(1, 64)

        # Exogenous
        self.exo_encoder = ExogenousEncoder(6, 64)
        self.exo_gate = nn.Linear(64, 64)

        # GRU expects 1 (traffic) + 64 (exo)
        self.encoder = nn.GRU(1 + 64, 64, batch_first=True)
        self.decoder = nn.Linear(64, 3)

        # Adapter for online correction
        self.adapter = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, x, exo):
        """
        x   : (B, T, N, 1)
        exo : (B, T, E)
        """
        B, T, N, _ = x.shape

        # -------- Adaptive Graph --------
        A_dyn = self.context(x)
        A = torch.sigmoid(self.alpha) * A_dyn + (1 - torch.sigmoid(self.alpha)) * self.adj

        x = torch.einsum("bnm,btmf->btnf", A, x)  # (B, T, N, 1)

        # -------- Exogenous Encoding --------
        exo_emb = self.exo_encoder(exo)           # (B, T, 64)
        exo_emb = exo_emb.unsqueeze(2).repeat(1, 1, N, 1)  # (B, T, N, 64)

        gate = torch.sigmoid(self.exo_gate(exo_emb))
        exo_mod = gate * exo_emb                  # (B, T, N, 64)

        # -------- CONCAT (IMPORTANT FIX) --------
        x = torch.cat([x, exo_mod], dim=-1)       # (B, T, N, 65)

        # -------- Temporal Modeling --------
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, 65)

        _, h = self.encoder(x)                    # (1, B*N, 64)
        h = h.squeeze(0)

        h = h + self.adapter(h)

        out = self.decoder(h)                     # (B*N, 3)
        return out.view(B, N, -1)
