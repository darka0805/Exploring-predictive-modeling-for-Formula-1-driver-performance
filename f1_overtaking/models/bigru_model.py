import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler

from config import DL_BATCH_SIZE, DL_EPOCHS, DL_LR, DL_WEIGHT_DECAY, DL_DROPOUT, SEED


class _BiGRU(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 64,
                 n_layers: int = 2, dropout: float = DL_DROPOUT):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, num_layers=n_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)         
        last = out[:, -1, :]          
        return self.head(last).squeeze(-1)


def _valid_num_heads(embed_dim: int, requested_heads: int) -> int:
    heads = max(1, min(int(requested_heads), embed_dim))
    while embed_dim % heads != 0 and heads > 1:
        heads -= 1
    return heads


class _BiGRUAttention(nn.Module):
    """BiGRU + self-attention + attentive pooling for richer sequence modeling."""

    def __init__(self, n_feat: int, hidden: int = 96,
                 n_layers: int = 3, dropout: float = 0.35,
                 attn_heads: int = 4, fc_hidden: int | None = None):
        super().__init__()
        self.gru = nn.GRU(
            n_feat,
            hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        embed_dim = hidden * 2
        heads = _valid_num_heads(embed_dim, attn_heads)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.pool_score = nn.Linear(embed_dim, 1)
        head_hidden = int(fc_hidden or embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)                 
        attn_out, _ = self.attn(out, out, out, need_weights=False)
        h = self.norm(out + attn_out)

        weights = torch.softmax(self.pool_score(h).squeeze(-1), dim=1)
        pooled = torch.sum(h * weights.unsqueeze(-1), dim=1)
        last = h[:, -1, :]
        return self.head(torch.cat([pooled, last], dim=1)).squeeze(-1)


class BiGRUModel:
    name = "BiGRU"

    def __init__(self, extra_params: dict | None = None):
        self.extra = extra_params or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model = None

        self.hidden = int(self.extra.get("hidden", 64))
        self.n_layers = int(self.extra.get("n_layers", 2))
        self.dropout = float(self.extra.get("dropout", DL_DROPOUT))
        self.architecture = str(self.extra.get("architecture", "bigru_last"))
        self.attn_heads = int(self.extra.get("attn_heads", 4))
        self.fc_hidden = self.extra.get("fc_hidden")
        self.name = "BiGRU-Attn" if self.architecture == "bigru_attn" else "BiGRU"
        self.lr = float(self.extra.get("lr", DL_LR))
        self.weight_decay = float(self.extra.get("weight_decay", DL_WEIGHT_DECAY))
        self.batch_size = int(self.extra.get("batch_size", DL_BATCH_SIZE))
        self.epochs = int(self.extra.get("epochs", DL_EPOCHS))
        self.patience = int(self.extra.get("patience", 8))
        self.use_weighted_sampler = bool(self.extra.get("use_weighted_sampler", True))
        self.eval_batch_size = int(self.extra.get("eval_batch_size", max(self.batch_size, 1024)))

    def _normalise(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        n, t, f = X.shape
        flat = X.reshape(-1, f)
        if fit:
            self.scaler.fit(flat)
        return self.scaler.transform(flat).reshape(n, t, f)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
        torch.manual_seed(SEED)
        X_n = self._normalise(X_train, fit=True)

        sampler = None
        if self.use_weighted_sampler:
            spw = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0)
            weights = np.where(y_train == 1, spw, 1.0)
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        ds = TensorDataset(torch.tensor(X_n, dtype=torch.float32),
                           torch.tensor(y_train, dtype=torch.float32))
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            drop_last=False,
        )

        if self.architecture == "bigru_attn":
            self.model = _BiGRUAttention(
                X_train.shape[2],
                hidden=self.hidden,
                n_layers=self.n_layers,
                dropout=self.dropout,
                attn_heads=self.attn_heads,
                fc_hidden=self.fc_hidden,
            ).to(self.device)
        else:
            self.model = _BiGRU(
                X_train.shape[2],
                hidden=self.hidden,
                n_layers=self.n_layers,
                dropout=self.dropout,
            ).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(self.epochs, 1))
        crit = nn.BCEWithLogitsLoss()

        best_loss = float("inf")
        wait = 0

        for epoch in range(self.epochs):
            self.model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = crit(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            sched.step()

            if X_val is not None:
                vl = self._val_loss(X_val, y_val, crit)
                if vl < best_loss:
                    best_loss = vl
                    wait = 0
                    self._best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        break

        if X_val is not None and hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)
        return self

    def _val_loss(self, X_val, y_val, crit):
        self.model.eval()
        X_n = self._normalise(X_val)
        total = 0.0
        n_total = 0
        with torch.no_grad():
            for i in range(0, len(X_n), self.eval_batch_size):
                xb = torch.tensor(X_n[i: i + self.eval_batch_size], dtype=torch.float32).to(self.device)
                yb = torch.tensor(y_val[i: i + self.eval_batch_size], dtype=torch.float32).to(self.device)
                batch_loss = crit(self.model(xb), yb).item()
                total += batch_loss * len(xb)
                n_total += len(xb)
        return total / max(n_total, 1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_n = self._normalise(X)
        out = []
        with torch.no_grad():
            for i in range(0, len(X_n), self.eval_batch_size):
                xb = torch.tensor(X_n[i: i + self.eval_batch_size], dtype=torch.float32).to(self.device)
                logits = self.model(xb)
                out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out, axis=0)
