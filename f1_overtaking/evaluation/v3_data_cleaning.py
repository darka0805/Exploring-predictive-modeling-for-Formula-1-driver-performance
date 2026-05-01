from __future__ import annotations

import numpy as np
import pandas as pd

from config import RESULTS_DIR, SEED
from models import MODEL_REGISTRY
from train import train_and_evaluate


def _balance(X: np.ndarray, y: np.ndarray,
             ratio: int = 20) -> tuple[np.ndarray, np.ndarray]:
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0:
        return X, y
    n_neg = min(len(neg), len(pos) * ratio)
    rng = np.random.default_rng(SEED)
    keep = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
    rng.shuffle(keep)
    return X[keep], y[keep]


def run_preprocessing_ablation(
    X_train_raw: np.ndarray, y_train_raw: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> pd.DataFrame:
    """Test each preprocessing step independently with XGBoost."""
    print("\n" + "=" * 70)
    print("V3: DATA PREPROCESSING ABLATION")
    print("=" * 70)

    model_cls = MODEL_REGISTRY["xgboost"]
    rows: list[dict] = []

    def _run(tag: str, Xtr, ytr, Xte, yte):
        out = train_and_evaluate(
            model_cls, Xtr, ytr, Xte, yte,
            experiment_name=f"V3_{tag}",
            save_plots=False, persist_metrics=False,
        )
        rows.append({
            "preprocessing": tag,
            "train_samples": len(ytr),
            "train_pos": int(ytr.sum()),
            **{k: round(float(v), 6) if isinstance(v, float) else v
               for k, v in out["metrics"].items()},
        })

    X_bal, y_bal = _balance(X_train_raw, y_train_raw, ratio=20)
    _run("P0_baseline_1to20", X_bal, y_bal, X_test, y_test)

    X_ff = X_train_raw.copy()
    for i in range(X_ff.shape[0]):
        for f in range(X_ff.shape[2]):
            col = X_ff[i, :, f]
            mask = np.isnan(col)
            if mask.any():
                idx = np.where(~mask, np.arange(len(col)), 0)
                np.maximum.accumulate(idx, out=idx)
                col[mask] = col[idx[mask]]
    X_ff = np.nan_to_num(X_ff, nan=0.0, posinf=30.0, neginf=-30.0)
    X_ff_b, y_ff_b = _balance(X_ff, y_train_raw, ratio=20)
    _run("P1a_nan_forwardfill", X_ff_b, y_ff_b, X_test.copy(), y_test)

    X_med = X_train_raw.copy()
    for f in range(X_med.shape[2]):
        vals = X_med[:, :, f].ravel()
        median = np.nanmedian(vals)
        X_med[:, :, f] = np.where(
            np.isnan(X_med[:, :, f]), median, X_med[:, :, f])
    X_med = np.nan_to_num(X_med, posinf=30.0, neginf=-30.0)
    X_med_b, y_med_b = _balance(X_med, y_train_raw, ratio=20)
    X_test_med = X_test.copy()
    for f in range(X_test_med.shape[2]):
        vals = X_test_med[:, :, f].ravel()
        median = np.nanmedian(vals)
        X_test_med[:, :, f] = np.where(
            np.isnan(X_test_med[:, :, f]), median, X_test_med[:, :, f])
    X_test_med = np.nan_to_num(X_test_med, posinf=30.0, neginf=-30.0)
    _run("P1b_nan_median", X_med_b, y_med_b, X_test_med, y_test)

    for ratio in [10, 50]:
        X_r, y_r = _balance(X_train_raw, y_train_raw, ratio=ratio)
        _run(f"P2_balance_1to{ratio}", X_r, y_r, X_test, y_test)

    _run("P2_balance_none", X_train_raw, y_train_raw, X_test, y_test)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    n, t, feat = X_bal.shape
    X_scaled = scaler.fit_transform(X_bal.reshape(-1, feat)).reshape(n, t, feat)
    nt, tt, ft = X_test.shape
    X_test_sc = scaler.transform(X_test.reshape(-1, ft)).reshape(nt, tt, ft)
    _run("P8_standardscaler_on", X_scaled, y_bal, X_test_sc, y_test)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "v3_preprocessing_ablation.csv", index=False)
    print("\n" + df[["preprocessing", "train_samples", "train_pos",
                     "pr_auc", "f1", "roc_auc"]].to_string(index=False))
    return df


def run_v3() -> pd.DataFrame:
    """Standalone runner: builds AD-only train + 2024 test."""
    from config import PRIMARY_TRACK
    from data_preprocessing_and_labeling.dataset_builder import (
        build_single_race_dataset, load_dataset)

    train_parts = []
    for y in (2022, 2023):
        p = build_single_race_dataset(y, PRIMARY_TRACK)
        if p is not None:
            train_parts.append(load_dataset(p))
    test_path = build_single_race_dataset(2024, PRIMARY_TRACK)
    if not train_parts or test_path is None:
        raise SystemExit("V3: required datasets missing.")

    test_ds = load_dataset(test_path)
    Xtr = np.concatenate([d["X"] for d in train_parts])
    ytr = np.concatenate([d["y"] for d in train_parts])
    return run_preprocessing_ablation(Xtr, ytr, test_ds["X"], test_ds["y"])


if __name__ == "__main__":
    run_v3()
