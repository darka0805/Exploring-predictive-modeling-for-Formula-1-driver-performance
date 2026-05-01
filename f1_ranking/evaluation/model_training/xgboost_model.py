import numpy as np
import pandas as pd

from config import DEFAULT_XGB_PARAMS
from .splits_and_rank import impute_train_test


def _ensure_xgb():
    try:
        import xgboost as xgb
    except Exception:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost"])
        import xgboost as xgb
    return xgb


def train_xgboost(folds: list[dict],
                  params: dict | None = None,
                  random_state: int = 42):
    """Train XGBoost per fold and rank within each race."""
    xgb = _ensure_xgb()
    if params is None:
        params = DEFAULT_XGB_PARAMS

    all_predictions = []
    last_model = None

    for fold in folds:
        X_train, X_test = impute_train_test(fold["X_train"], fold["X_test"])
        y_train = fold["y_train"].to_numpy(dtype=np.float32, copy=True)

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            min_child_weight=float(params["min_child_weight"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            gamma=float(params["gamma"]),
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        last_model = model

        preds = model.predict(X_test).astype(float)
        result = fold["meta_test"].copy()
        result["pred_score"] = preds
        result["pred_position"] = (
            result.groupby("race_id")["pred_score"].rank(method="min", ascending=True).astype(float)
        )
        all_predictions.append(result)

    return pd.concat(all_predictions, ignore_index=True), last_model


def _sample_params(rng: np.random.Generator) -> dict:
    return {
        "n_estimators": int(rng.integers(200, 1201)),
        "learning_rate": float(rng.uniform(0.01, 0.2)),
        "max_depth": int(rng.integers(2, 9)),
        "min_child_weight": float(rng.uniform(1.0, 12.0)),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "reg_alpha": float(rng.uniform(0.0, 3.0)),
        "reg_lambda": float(rng.uniform(0.1, 10.0)),
        "gamma": float(rng.uniform(0.0, 3.0)),
    }


def tune_xgboost(folds: list[dict], evaluate_predictions, n_trials: int = 30, seed: int = 42):
    """Random search; pick best by Spearman, tie-break by MAE then NDCG."""
    rng = np.random.default_rng(seed)
    best = None

    print("=" * 100)
    print(f"XGBoost tuning: {n_trials} trials")
    print("=" * 100)
    for t in range(1, n_trials + 1):
        params = _sample_params(rng)
        preds, model = train_xgboost(folds, params, random_state=seed)
        ev = evaluate_predictions(preds)
        summary = {
            "spearman": float(ev["spearman_model"].mean()),
            "mae": float(ev["mae_model"].mean()),
            "ndcg": float(ev["ndcg_model"].mean()),
            "top3": float(ev["top3_model"].mean()),
            "top5": float(ev["top5_model"].mean()),
            "within2": float(ev["within2_model"].mean()),
        }
        is_better = (
            best is None
            or summary["spearman"] > best["spearman"]
            or (summary["spearman"] == best["spearman"] and summary["mae"] < best["mae"])
        )
        if is_better:
            best = {"trial": t, "params": params, "predictions": preds,
                    "eval_df": ev, "model": model, **summary}
            print(f"XGB trial {t:02d} NEW BEST | Spearman={summary['spearman']:.4f} "
                  f"MAE={summary['mae']:.4f} NDCG={summary['ndcg']:.4f}")
        else:
            print(f"XGB trial {t:02d}          | Spearman={summary['spearman']:.4f} "
                  f"MAE={summary['mae']:.4f} NDCG={summary['ndcg']:.4f}")

    return best
