from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import SEED
from evaluation.evaluate import compute_metrics, plot_pr_roc, plot_confusion
from models import MODEL_REGISTRY
from experiments.e7_config import EXPERIMENT_TAG, STACKING_CONFIGS


def run_stacking_ensembles(best_cfg_by_model: dict[str, dict],
                           fitted_full_models: dict[str, object],
                           X_tr: np.ndarray, y_tr: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray
                           ) -> pd.DataFrame:
    rows: list[dict] = []

    for stack_cfg in STACKING_CONFIGS:
        ensemble_name = stack_cfg["name"]
        base_models = [m for m in stack_cfg["base_models"]
                       if m in best_cfg_by_model]
        if len(base_models) < 2:
            continue

        print("\n" + "-" * 72)
        print(f"Stacking: {ensemble_name} | bases={base_models}")
        print("-" * 72)

        val_features = []
        test_features = []

        for model_name in base_models:
            model_cls = MODEL_REGISTRY[model_name]
            cfg = best_cfg_by_model[model_name]

            base_for_val = model_cls(extra_params=cfg)
            base_for_val.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
            val_features.append(base_for_val.predict_proba(X_val))

            base_full = fitted_full_models[model_name]
            test_features.append(base_full.predict_proba(X_test))

        X_meta_train = np.column_stack(val_features)
        X_meta_test = np.column_stack(test_features)

        t0 = time.perf_counter()
        meta = LogisticRegression(
            C=float(stack_cfg.get("meta_c", 1.0)),
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=SEED,
        )
        meta.fit(X_meta_train, y_val)
        y_prob = meta.predict_proba(X_meta_test)[:, 1]
        elapsed = time.perf_counter() - t0

        metrics = compute_metrics(y_test, y_prob)
        label = f"{EXPERIMENT_TAG}_{ensemble_name}"
        print(f"Stacked metrics for {ensemble_name}: "
              f"PR-AUC={metrics.get('pr_auc', np.nan):.4f}, "
              f"F1={metrics.get('f1', np.nan):.4f}")

        plot_pr_roc(y_test, y_prob, label)
        y_pred = (y_prob >= metrics["threshold"]).astype(int)
        plot_confusion(y_test, y_pred, label)

        rows.append({
            "ensemble": ensemble_name,
            "base_models_json": json.dumps(base_models),
            "meta_model": "LogisticRegression",
            "meta_params_json": json.dumps({
                "C": float(stack_cfg.get("meta_c", 1.0)),
                "max_iter": 2000,
                "class_weight": "balanced",
            }, sort_keys=True),
            "pr_auc": float(metrics.get("pr_auc", np.nan)),
            "roc_auc": float(metrics.get("roc_auc", np.nan)),
            "f1": float(metrics.get("f1", np.nan)),
            "precision": float(metrics.get("precision", np.nan)),
            "recall": float(metrics.get("recall", np.nan)),
            "threshold": float(metrics.get("threshold", np.nan)),
            "n_total": int(metrics.get("n_total", 0)),
            "n_pos": int(metrics.get("n_pos", 0)),
            "fit_seconds": round(elapsed, 3),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "ensemble", "base_models_json", "meta_model", "meta_params_json",
            "pr_auc", "roc_auc", "f1", "precision", "recall", "threshold",
            "n_total", "n_pos", "fit_seconds",
        ])

    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
