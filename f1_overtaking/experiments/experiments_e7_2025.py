"""E7 orchestrator: train on 2022-2024, evaluate on 2025.

Pipeline:
  1. Load + balance training data, split into tune/val.
  2. Sweep hyperparameters per model, pick best by val F1.
  3. Refit best config on full balanced train, score on 2025 test.
  4. Build stacking ensembles over the trained base models.
  5. Rerun top base models on full raw train for comparison.
  6. Save CSVs, plots, and a markdown report.

Sub-modules: e7_config, e7_data, e7_plots, e7_stacking, e7_full_rerun, e7_report.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import PRIMARY_TRACK, RESULTS_DIR, SEED
from models import MODEL_REGISTRY
from train import train_and_evaluate

from experiments.e7_config import (
    TRAIN_YEARS, TEST_YEAR, NEG_POS_RATIO, VAL_SIZE,
    FULL_DATA_RERUN_TOP_K, EXPERIMENT_TAG, MODEL_SWEEPS,
)
from experiments.e7_data import (
    to_1_to_k_balance, load_single_track_year, combine, safe_score,
)
from experiments.e7_plots import (
    plot_class_balance, plot_final_model_comparison,
    plot_combined_model_comparison, plot_full_rerun_comparison,
    plot_tuning_quality,
)
from experiments.e7_stacking import run_stacking_ensembles
from experiments.e7_full_rerun import rerun_top_models_on_full_data
from experiments.e7_report import write_report

warnings.filterwarnings("ignore")


def run_2025_holdout_experiments() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 72)
    print("E7: Train on 2022-2024, evaluate on 2025")
    print("=" * 72)

    print("\nPreparing datasets...")
    train_datasets = [load_single_track_year(y, PRIMARY_TRACK) for y in TRAIN_YEARS]
    test_dataset = load_single_track_year(TEST_YEAR, PRIMARY_TRACK)

    X_train_raw, y_train_raw = combine(train_datasets)
    X_test, y_test = test_dataset["X"], test_dataset["y"]

    print(f"Train raw: {len(y_train_raw)} samples (pos={int(y_train_raw.sum())})")
    print(f"Test 2025: {len(y_test)} samples (pos={int(y_test.sum())})")

    X_train_bal, y_train_bal, _ = to_1_to_k_balance(
        X_train_raw, y_train_raw, neg_per_pos=NEG_POS_RATIO, seed=SEED)
    print(f"Train balanced: {len(y_train_bal)} samples "
          f"(pos={int(y_train_bal.sum())}, neg={int((y_train_bal == 0).sum())})")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_bal, y_train_bal,
        test_size=VAL_SIZE, random_state=SEED, stratify=y_train_bal,
    )
    print(f"Tune split: train={len(y_tr)} val={len(y_val)}")

    tuning_rows: list[dict] = []
    final_rows: list[dict] = []
    best_cfg_by_model: dict[str, dict] = {}
    fitted_full_models: dict[str, object] = {}

    for model_name in MODEL_REGISTRY.keys():
        model_cls = MODEL_REGISTRY[model_name]
        sweep = MODEL_SWEEPS.get(model_name, [{}])

        print("\n" + "-" * 72)
        print(f"Model: {model_name} | trials: {len(sweep)}")
        print("-" * 72)

        best_score = (-1.0, -1.0)
        best_cfg: dict = {}

        for trial_id, cfg in enumerate(sweep, start=1):
            trial_tag = f"{EXPERIMENT_TAG}_tune_{model_name}_t{trial_id:02d}"
            t0 = time.perf_counter()
            out = train_and_evaluate(
                model_cls, X_tr, y_tr, X_val, y_val,
                experiment_name=trial_tag, model_extra=cfg,
                save_plots=False, persist_metrics=False,
            )
            elapsed = time.perf_counter() - t0
            m = out["metrics"]

            tuning_rows.append({
                "model": model_name,
                "trial": trial_id,
                "pr_auc": float(m.get("pr_auc", np.nan)),
                "roc_auc": float(m.get("roc_auc", np.nan)),
                "f1": float(m.get("f1", np.nan)),
                "precision": float(m.get("precision", np.nan)),
                "recall": float(m.get("recall", np.nan)),
                "threshold": float(m.get("threshold", np.nan)),
                "fit_seconds": round(elapsed, 3),
                "params_json": json.dumps(cfg, sort_keys=True),
            })

            score = (safe_score(m.get("f1", np.nan)),
                     safe_score(m.get("pr_auc", np.nan)))
            if score > best_score:
                best_score = score
                best_cfg = cfg

        print(f"Best config by val F1: {json.dumps(best_cfg, sort_keys=True)}")
        best_cfg_by_model[model_name] = dict(best_cfg)

        final_tag = f"{EXPERIMENT_TAG}_test_{model_name}"
        t0 = time.perf_counter()
        final_out = train_and_evaluate(
            model_cls, X_train_bal, y_train_bal, X_test, y_test,
            experiment_name=final_tag, model_extra=best_cfg,
            save_plots=True, persist_metrics=False,
        )
        elapsed = time.perf_counter() - t0
        m = final_out["metrics"]
        fitted_full_models[model_name] = final_out["model"]

        final_rows.append({
            "model": model_name,
            "pr_auc": float(m.get("pr_auc", np.nan)),
            "roc_auc": float(m.get("roc_auc", np.nan)),
            "f1": float(m.get("f1", np.nan)),
            "precision": float(m.get("precision", np.nan)),
            "recall": float(m.get("recall", np.nan)),
            "threshold": float(m.get("threshold", np.nan)),
            "n_total": int(m.get("n_total", 0)),
            "n_pos": int(m.get("n_pos", 0)),
            "fit_seconds": round(elapsed, 3),
            "best_params_json": json.dumps(best_cfg, sort_keys=True),
            "trials_run": len(sweep),
        })

    tuning_df = pd.DataFrame(tuning_rows)
    final_df = pd.DataFrame(final_rows).sort_values("pr_auc", ascending=False)

    stacking_df = run_stacking_ensembles(
        best_cfg_by_model=best_cfg_by_model,
        fitted_full_models=fitted_full_models,
        X_tr=X_tr, y_tr=y_tr, X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )

    full_rerun_df = rerun_top_models_on_full_data(
        final_df=final_df,
        best_cfg_by_model=best_cfg_by_model,
        X_train_full=X_train_raw, y_train_full=y_train_raw,
        X_test=X_test, y_test=y_test,
        top_k=FULL_DATA_RERUN_TOP_K,
    )

    tuning_path = RESULTS_DIR / "e7_2025_tuning_results.csv"
    final_path = RESULTS_DIR / "e7_2025_test_results.csv"
    stacking_path = RESULTS_DIR / "e7_2025_stacking_results.csv"
    full_rerun_path = RESULTS_DIR / "e7_2025_full_train_rerun_results.csv"
    top_trials_path = RESULTS_DIR / "e7_2025_top_validation_trials.csv"

    tuning_df.to_csv(tuning_path, index=False)
    final_df.to_csv(final_path, index=False)
    stacking_df.to_csv(stacking_path, index=False)
    full_rerun_df.to_csv(full_rerun_path, index=False)

    top_trials = (
        tuning_df.sort_values(["model", "pr_auc", "f1"],
                              ascending=[True, False, False])
        .groupby("model", as_index=False)
        .head(3)
        .copy()
    )
    top_trials.to_csv(top_trials_path, index=False)

    class_stats = pd.DataFrame([
        {"split": "train_raw",
         "episodes": int(len(y_train_raw)),
         "positive": int((y_train_raw == 1).sum()),
         "negative": int((y_train_raw == 0).sum()),
         "pos_rate": round(float((y_train_raw == 1).mean()), 6)},
        {"split": f"train_balanced_1_to_{NEG_POS_RATIO}",
         "episodes": int(len(y_train_bal)),
         "positive": int((y_train_bal == 1).sum()),
         "negative": int((y_train_bal == 0).sum()),
         "pos_rate": round(float((y_train_bal == 1).mean()), 6)},
        {"split": "test_2025",
         "episodes": int(len(y_test)),
         "positive": int((y_test == 1).sum()),
         "negative": int((y_test == 0).sum()),
         "pos_rate": round(float((y_test == 1).mean()), 6)},
    ])

    sweep_summary = pd.DataFrame([
        {"model": model_name,
         "trials": len(MODEL_SWEEPS.get(model_name, [{}])),
         "selection_metric": "max val F1, then PR-AUC"}
        for model_name in MODEL_REGISTRY.keys()
    ])

    class_plot = RESULTS_DIR / "e7_2025_class_balance.png"
    compare_plot = RESULTS_DIR / "e7_2025_final_model_comparison.png"
    combined_plot = RESULTS_DIR / "e7_2025_base_vs_stacking.png"
    full_rerun_plot = RESULTS_DIR / "e7_2025_full_train_rerun_comparison.png"
    tuning_plot = RESULTS_DIR / "e7_2025_tuning_quality.png"

    plot_class_balance(y_train_raw, y_train_bal, y_test, class_plot)
    plot_final_model_comparison(final_df, compare_plot)
    plot_combined_model_comparison(final_df, stacking_df, combined_plot)
    plot_full_rerun_comparison(full_rerun_df, full_rerun_plot)
    plot_tuning_quality(tuning_df, tuning_plot)

    report_df = final_df.copy()
    for c in ["pr_auc", "roc_auc", "f1", "precision", "recall", "threshold"]:
        report_df[c] = report_df[c].map(lambda x: round(float(x), 6))

    report_top = top_trials[[
        "model", "trial", "pr_auc", "roc_auc", "f1", "precision",
        "recall", "fit_seconds", "params_json"
    ]].copy()
    for c in ["pr_auc", "roc_auc", "f1", "precision", "recall"]:
        report_top[c] = report_top[c].map(lambda x: round(float(x), 6))

    report_stacking = stacking_df.copy()
    for c in ["pr_auc", "roc_auc", "f1", "precision", "recall", "threshold"]:
        if c in report_stacking.columns:
            report_stacking[c] = report_stacking[c].map(
                lambda x: round(float(x), 6))

    report_full_rerun = full_rerun_df.copy()
    for c in ["pr_auc", "roc_auc", "f1", "precision", "recall", "threshold"]:
        if c in report_full_rerun.columns:
            report_full_rerun[c] = report_full_rerun[c].map(
                lambda x: round(float(x), 6))

    report_path = Path("RESULTS_2025.md")
    plot_refs = {
        "Class distribution": str(Path("results") / class_plot.name),
        "Final model comparison": str(Path("results") / compare_plot.name),
        "Base vs stacking comparison": str(Path("results") / combined_plot.name),
        "Full-data rerun comparison": str(Path("results") / full_rerun_plot.name),
        "Tuning quality": str(Path("results") / tuning_plot.name),
    }
    write_report(
        report_path=report_path,
        class_stats=class_stats,
        sweep_summary=sweep_summary,
        top_trials=report_top,
        stacking_df=report_stacking,
        final_df=report_df,
        full_rerun_df=report_full_rerun,
        plots=plot_refs,
    )

    print("\nSaved artifacts:")
    for p in [tuning_path, final_path, stacking_path, full_rerun_path,
              top_trials_path, class_plot, compare_plot, combined_plot,
              full_rerun_plot, tuning_plot, report_path]:
        print(f"  - {p}")

    print("\nFinal ranking (by PR-AUC):")
    print(final_df[["model", "pr_auc", "f1", "precision", "recall",
                    "n_pos", "n_total"]].to_string(index=False))

    if len(stacking_df) > 0:
        print("\nStacking ranking (by PR-AUC):")
        print(stacking_df[["ensemble", "pr_auc", "f1", "precision",
                           "recall", "n_pos", "n_total"]].to_string(index=False))

    if len(full_rerun_df) > 0:
        print("\nFull-data rerun ranking (by F1):")
        print(full_rerun_df[["model", "f1", "pr_auc", "precision",
                             "recall", "n_pos", "n_total"]].to_string(index=False))

    return tuning_df, final_df, stacking_df


if __name__ == "__main__":
    run_2025_holdout_experiments()
