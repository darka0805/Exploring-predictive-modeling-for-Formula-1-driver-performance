from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

from config import (
    ARTIFACT_ROOT,
    DEFAULT_LGBM_RANKER_PARAMS,
    DEFAULT_XGB_RANKER_PARAMS,
    ENSEMBLE_MAX_POS,
    YEARS,
)
from .splits_and_rank import impute_train_test


def _ensure_lgb():
    try:
        import lightgbm as lgb
    except Exception:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm"])
        import lightgbm as lgb
    return lgb


def _ensure_xgb():
    try:
        import xgboost as xgb
    except Exception:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost"])
        import xgboost as xgb
    return xgb


def _normalize_scores(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else np.zeros_like(s)


def train_lgbm_xgb_ensemble(folds: list[dict],
                            lgb_params: dict | None = None,
                            xgb_params: dict | None = None,
                            max_pos: int = ENSEMBLE_MAX_POS,
                            verbose: bool = True):
    """Train LightGBM + XGBoost rankers per fold; blend by min-max normalized average.

    Returns (predictions_df, last_lgb_model, last_xgb_model).
    """
    lgb = _ensure_lgb()
    xgb = _ensure_xgb()

    lgb_params = dict(lgb_params or DEFAULT_LGBM_RANKER_PARAMS)
    xgb_params = dict(xgb_params or DEFAULT_XGB_RANKER_PARAMS)

    all_predictions: list[pd.DataFrame] = []
    last_lgb = None
    last_xgb = None

    for i, fold in enumerate(folds):
        X_train, X_test = impute_train_test(fold["X_train"], fold["X_test"])
        y_train = fold["y_train"].to_numpy(dtype=np.float32, copy=True)
        y_test = fold["y_test"].to_numpy(dtype=np.float32, copy=True)
        meta = fold["meta_test"].copy()

        y_train_rel = np.clip(max_pos + 1 - y_train, 0, max_pos)

        lgb_model = lgb.LGBMRanker(**lgb_params)
        lgb_model.fit(X_train, y_train_rel, group=fold["train_groups"])
        lgb_scores = lgb_model.predict(X_test)
        last_lgb = lgb_model

        xgb_model = xgb.XGBRanker(**xgb_params)
        xgb_model.fit(X_train, y_train_rel, group=fold["train_groups"])
        xgb_scores = xgb_model.predict(X_test)
        last_xgb = xgb_model

        ensemble_scores = 0.5 * _normalize_scores(lgb_scores) + 0.5 * _normalize_scores(xgb_scores)

        pred_positions = pd.Series(ensemble_scores).rank(ascending=False, method="min").values

        result = meta.copy()
        result["pred_score"] = ensemble_scores
        result["pred_position"] = pred_positions
        result["lgb_score"] = lgb_scores
        result["xgb_score"] = xgb_scores
        all_predictions.append(result)

        if verbose:
            sp, _ = spearmanr(y_test, pred_positions)
            mae = mean_absolute_error(y_test, pred_positions)
            print(f"Fold {i+1:2d} | {fold['test_year']} R{fold['test_round']:2d} "
                  f"{fold['test_event']:30s} | Spearman={sp:.3f} | MAE={mae:.2f}")

    return pd.concat(all_predictions, ignore_index=True), last_lgb, last_xgb


def save_full_lgbm_xgb_ensemble(dataset: pd.DataFrame,
                                feature_cols: list[str],
                                lgb_params: dict | None = None,
                                xgb_params: dict | None = None,
                                max_pos: int = ENSEMBLE_MAX_POS,
                                models_dir: Path | None = None) -> list[str]:
    """Refit LightGBM and XGBoost rankers on the full dataset and save weights."""
    lgb = _ensure_lgb()
    xgb = _ensure_xgb()

    lgb_params = dict(lgb_params or DEFAULT_LGBM_RANKER_PARAMS)
    xgb_params = dict(xgb_params or DEFAULT_XGB_RANKER_PARAMS)
    models_dir = (models_dir or ARTIFACT_ROOT / "models") / "lgbm_xgb_ensemble"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = dataset.sort_values("race_id").reset_index(drop=True)
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean(numeric_only=True)).fillna(0.0).to_numpy(dtype=np.float32)
    y = df["finish_position"].to_numpy(dtype=np.float32)
    y_rel = np.clip(max_pos + 1 - y, 0, max_pos)
    groups = df.groupby("race_id", sort=False).size().values

    lgb_model = lgb.LGBMRanker(**lgb_params)
    lgb_model.fit(X, y_rel, group=groups)

    xgb_model = xgb.XGBRanker(**xgb_params)
    xgb_model.fit(X, y_rel, group=groups)

    lgb_path = models_dir / "lgbm_ranker_full.joblib"
    xgb_path = models_dir / "xgb_ranker_full.json"
    meta_path = models_dir / "ensemble_metadata.joblib"
    cfg_path = models_dir / "ensemble_configs.json"

    joblib.dump(lgb_model, lgb_path)
    xgb_model.save_model(str(xgb_path))
    joblib.dump(
        {"feature_cols": feature_cols, "years": YEARS, "target": "finish_position",
         "n_rows": int(len(df)), "max_pos": int(max_pos),
         "lgb_params": lgb_params, "xgb_params": xgb_params,
         "blend": "0.5*minmax(lgb) + 0.5*minmax(xgb)"},
        meta_path,
    )
    cfg_path.write_text(
        json.dumps({"lgb_params": lgb_params, "xgb_params": xgb_params,
                    "max_pos": int(max_pos)}, indent=2),
        encoding="utf-8",
    )

    print(f"Saved LightGBM+XGBoost ranker ensemble -> {models_dir}")
    return [str(lgb_path), str(xgb_path), str(meta_path), str(cfg_path)]
