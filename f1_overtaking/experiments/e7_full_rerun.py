from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from models import MODEL_REGISTRY
from train import train_and_evaluate
from experiments.e7_config import EXPERIMENT_TAG, FULL_DATA_RERUN_TOP_K


def rerun_top_models_on_full_data(final_df: pd.DataFrame,
                                  best_cfg_by_model: dict[str, dict],
                                  X_train_full: np.ndarray,
                                  y_train_full: np.ndarray,
                                  X_test: np.ndarray,
                                  y_test: np.ndarray,
                                  top_k: int = FULL_DATA_RERUN_TOP_K
                                  ) -> pd.DataFrame:
    """Retrain top base models (by F1) on full raw train data and re-evaluate."""
    top_models = (final_df.sort_values(["f1", "pr_auc"],
                                       ascending=[False, False])["model"])
    top_models = list(top_models.head(top_k).values)

    rows: list[dict] = []
    for model_name in top_models:
        model_cls = MODEL_REGISTRY[model_name]
        cfg = best_cfg_by_model.get(model_name, {})
        exp_name = f"{EXPERIMENT_TAG}_fullraw_{model_name}"

        print("\n" + "-" * 72)
        print(f"Full-data rerun: {model_name} on raw 2022-2024 train")
        print("-" * 72)

        t0 = time.perf_counter()
        out = train_and_evaluate(
            model_cls,
            X_train_full,
            y_train_full,
            X_test,
            y_test,
            experiment_name=exp_name,
            model_extra=cfg,
            save_plots=True,
            persist_metrics=False,
        )
        elapsed = time.perf_counter() - t0
        m = out["metrics"]

        rows.append({
            "model": model_name,
            "train_mode": "full_raw_2022_2024",
            "pr_auc": float(m.get("pr_auc", np.nan)),
            "roc_auc": float(m.get("roc_auc", np.nan)),
            "f1": float(m.get("f1", np.nan)),
            "precision": float(m.get("precision", np.nan)),
            "recall": float(m.get("recall", np.nan)),
            "threshold": float(m.get("threshold", np.nan)),
            "n_total": int(m.get("n_total", 0)),
            "n_pos": int(m.get("n_pos", 0)),
            "fit_seconds": round(elapsed, 3),
            "best_params_json": json.dumps(cfg, sort_keys=True),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "model", "train_mode", "pr_auc", "roc_auc", "f1", "precision",
            "recall", "threshold", "n_total", "n_pos", "fit_seconds",
            "best_params_json",
        ])
    return pd.DataFrame(rows).sort_values("f1", ascending=False)
