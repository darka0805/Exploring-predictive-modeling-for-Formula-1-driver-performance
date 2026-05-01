from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import DL_BATCH_SIZE, DL_EPOCHS, DL_LR, SEED, TEMPORAL_FEATURES


DISTANCE_LIKE_FEATURES = (
    "DistanceToDriverAhead",
    "SpatialGap_m",
    "TTC",
    "DistanceToNextCorner",
    "StraightLength",
)
ATTACKER_SPEED_FEATURE = "att_Speed"


def _resolve_indices(feature_order: list[str]) -> tuple[list[int], int, list[int]]:
    name_to_idx = {name: i for i, name in enumerate(feature_order)}
    dist_idx = [name_to_idx[n] for n in DISTANCE_LIKE_FEATURES if n in name_to_idx]
    speed_idx = name_to_idx.get(ATTACKER_SPEED_FEATURE, 0)
    other_idx = [i for i in range(len(feature_order))
                 if i not in dist_idx and i != speed_idx]
    return dist_idx, speed_idx, other_idx


class _End2RaceGRU(nn.Module):
    """End2Race-style GRU with sigmoid pressure tokens, speed embedding + mask."""

    def __init__(self, n_dist: int, n_other: int,
                 speed_embed_dim: int = 16,
                 mask_prob: float = 0.1,
                 hidden_mult: int = 4,
                 pressure_k: float = 0.25):
        super().__init__()
        self.n_dist = n_dist
        self.n_other = n_other
        self.speed_embed_dim = speed_embed_dim
        self.mask_prob = mask_prob
        self.pressure_k = pressure_k

        input_dim = n_dist + speed_embed_dim + n_other
        hidden = input_dim * hidden_mult

        self.speed_proj = nn.Linear(1, speed_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(speed_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    @staticmethod
    def pressure_token(x: torch.Tensor, k: float = 0.25) -> torch.Tensor:
        """End2Race sigmoid normalization: close -> 1, far -> 0, bounded in [0, 1]."""
        return (-1.0 / (1.0 + torch.exp(-k * x)) + 1.0) * 2.0

    def forward(self, dist_feats: torch.Tensor,
                speed_feat: torch.Tensor,
                other_feats: torch.Tensor) -> torch.Tensor:
        """Return per-timestep logits, shape (B, T)."""
        B, T, _ = dist_feats.shape

        pressure = self.pressure_token(dist_feats, k=self.pressure_k)     # (B, T, n_dist)

        speed_embed = self.speed_proj(speed_feat.unsqueeze(-1))          # (B, T, E)

        if self.training and self.mask_prob > 0:
            mask = (torch.rand(B, T, 1, device=speed_embed.device)
                    < self.mask_prob).float()
            speed_embed = (1.0 - mask) * speed_embed + mask * self.mask_token

        obs = torch.cat([pressure, speed_embed, other_feats], dim=-1)    # (B, T, D)

        h_seq, _ = self.gru(obs)                                         # (B, T, H)
        logits = self.head(h_seq).squeeze(-1)                            # (B, T)
        return logits


class End2RaceModel:
    """End2Race-adapted GRU base model for binary overtake prediction."""

    name = "End2Race"

    def __init__(self, extra_params: dict | None = None):
        self.extra = extra_params or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model: _End2RaceGRU | None = None

        self.feature_order = list(self.extra.get("feature_order", TEMPORAL_FEATURES))
        self.speed_embed_dim = int(self.extra.get("speed_embed_dim", 16))
        self.mask_prob = float(self.extra.get("mask_prob", 0.1))
        self.hidden_mult = int(self.extra.get("hidden_mult", 4))
        self.pressure_k = float(self.extra.get("pressure_k", 0.25))
        self.loss_speed_weight = float(self.extra.get("loss_speed_weight", 1.0))

        self.lr = float(self.extra.get("lr", DL_LR))
        self.batch_size = int(self.extra.get("batch_size", max(DL_BATCH_SIZE, 64)))
        self.epochs = int(self.extra.get("epochs", max(DL_EPOCHS, 60)))
        self.patience = int(self.extra.get("patience", 10))
        self.lr_patience = int(self.extra.get("lr_patience", 4))
        self.eval_batch_size = int(self.extra.get(
            "eval_batch_size", max(self.batch_size, 1024)))

    def _split_features(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dist_idx, speed_idx, other_idx = _resolve_indices(self.feature_order)
        return X[:, :, dist_idx], X[:, :, speed_idx], X[:, :, other_idx]

    def _scale_other(self, X_other: np.ndarray, fit: bool) -> np.ndarray:
        n, t, f = X_other.shape
        if f == 0:
            return X_other
        flat = X_other.reshape(-1, f)
        if fit:
            self.scaler.fit(flat)
        return self.scaler.transform(flat).reshape(n, t, f)

    def _prepare(self, X: np.ndarray, fit: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dist, speed, other = self._split_features(X)
        speed_scaled = speed / 300.0
        other_scaled = self._scale_other(other, fit=fit)
        return (dist.astype(np.float32),
                speed_scaled.astype(np.float32),
                other_scaled.astype(np.float32))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        dist_idx, _, other_idx = _resolve_indices(self.feature_order)
        n_dist = len(dist_idx)
        n_other = len(other_idx)

        D_tr, S_tr, O_tr = self._prepare(X_train, fit=True)

        ds = TensorDataset(
            torch.tensor(D_tr), torch.tensor(S_tr),
            torch.tensor(O_tr), torch.tensor(y_train, dtype=torch.float32),
        )
        loader = DataLoader(ds, batch_size=self.batch_size,
                            shuffle=True, drop_last=False)

        self.model = _End2RaceGRU(
            n_dist=n_dist, n_other=n_other,
            speed_embed_dim=self.speed_embed_dim,
            mask_prob=self.mask_prob,
            hidden_mult=self.hidden_mult,
            pressure_k=self.pressure_k,
        ).to(self.device)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=self.lr_patience)

        pos = max(int((y_train == 1).sum()), 1)
        neg = max(int((y_train == 0).sum()), 1)
        pos_weight = torch.tensor([neg / pos], device=self.device)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val = float("inf")
        wait = 0
        self._best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            n_seen = 0
            for dxb, sxb, oxb, yb in loader:
                dxb = dxb.to(self.device)
                sxb = sxb.to(self.device)
                oxb = oxb.to(self.device)
                yb = yb.to(self.device)

                opt.zero_grad()
                logits_seq = self.model(dxb, sxb, oxb)                  # (B, T)
                y_expand = yb.unsqueeze(1).expand_as(logits_seq)        # (B, T)
                loss = crit(logits_seq, y_expand)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

                epoch_loss += loss.item() * dxb.size(0)
                n_seen += dxb.size(0)

            train_loss = epoch_loss / max(n_seen, 1)

            if X_val is not None:
                val_loss = self._val_loss(X_val, y_val, crit)
                sched.step(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    wait = 0
                    self._best_state = {k: v.detach().cpu().clone()
                                        for k, v in self.model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
            else:
                sched.step(train_loss)

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        return self

    def _val_loss(self, X_val, y_val, crit) -> float:
        self.model.eval()
        D_v, S_v, O_v = self._prepare(X_val, fit=False)
        total = 0.0
        n_total = 0
        with torch.no_grad():
            for i in range(0, len(D_v), self.eval_batch_size):
                sl = slice(i, i + self.eval_batch_size)
                dxb = torch.tensor(D_v[sl]).to(self.device)
                sxb = torch.tensor(S_v[sl]).to(self.device)
                oxb = torch.tensor(O_v[sl]).to(self.device)
                yb = torch.tensor(y_val[sl], dtype=torch.float32).to(self.device)
                logits_seq = self.model(dxb, sxb, oxb)
                loss = crit(logits_seq, yb.unsqueeze(1).expand_as(logits_seq))
                total += loss.item() * dxb.size(0)
                n_total += dxb.size(0)
        return total / max(n_total, 1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        D, S, O = self._prepare(X, fit=False)
        out = []
        with torch.no_grad():
            for i in range(0, len(D), self.eval_batch_size):
                sl = slice(i, i + self.eval_batch_size)
                dxb = torch.tensor(D[sl]).to(self.device)
                sxb = torch.tensor(S[sl]).to(self.device)
                oxb = torch.tensor(O[sl]).to(self.device)
                logits_seq = self.model(dxb, sxb, oxb)                  # (B, T)
                logits = logits_seq[:, -1]                              # last timestep
                out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out, axis=0)
