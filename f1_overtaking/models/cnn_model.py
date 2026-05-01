import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler

from config import DL_BATCH_SIZE, DL_EPOCHS, DL_LR, DL_WEIGHT_DECAY, DL_DROPOUT, SEED


class _CNN(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 64, dropout: float = DL_DROPOUT):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        # x: (B, T, F) → (B, F, T)
        h = self.conv(x.permute(0, 2, 1)).squeeze(-1)
        return self.head(h).squeeze(-1)


class CNNModel:
    name = "CNN"

    def __init__(self, extra_params: dict | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model = None
        self.extra = extra_params or {}

        self.hidden = int(self.extra.get("hidden", 64))
        self.dropout = float(self.extra.get("dropout", DL_DROPOUT))
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

        self.model = _CNN(X_train.shape[2], hidden=self.hidden, dropout=self.dropout).to(self.device)
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
