from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.calibration import calibration_curve

from config import (
    PRIMARY_TRACK, RESULTS_DIR, MODEL_DIR, SEED,
)
from data_preprocessing_and_labeling.dataset_builder import (
    build_single_race_dataset, load_dataset,
)
from evaluation.evaluate import plot_confusion
from evaluation.v1_remove_groups import run_feature_group_ablation
from evaluation.v2_remove_individual import run_individual_feature_ablation
from evaluation.v3_data_cleaning import run_preprocessing_ablation
from evaluation.v4_italian_gp import run_italian_exclusion
from evaluation.v5_cross_track import run_cross_track_cv
from models import MODEL_REGISTRY
from train import train_and_evaluate

warnings.filterwarnings("ignore")

TRAIN_YEARS = [2022, 2023, 2024]
TEST_YEAR = 2025
NEG_POS_RATIO = 20


def _load_single(year: int, track: str) -> dict:
    path = build_single_race_dataset(year, track)
    if path is None:
        raise RuntimeError(f"Cannot build dataset {year} {track}")
    return load_dataset(path)


def _combine(datasets: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return (np.concatenate([d["X"] for d in datasets]),
            np.concatenate([d["y"] for d in datasets]))


def _balance(X: np.ndarray, y: np.ndarray,
             ratio: int = NEG_POS_RATIO) -> tuple[np.ndarray, np.ndarray]:
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0:
        return X, y
    n_neg = min(len(neg), len(pos) * ratio)
    rng = np.random.default_rng(SEED)
    keep = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
    rng.shuffle(keep)
    return X[keep], y[keep]


def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def run_prediction_label_correlation(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> pd.DataFrame:
    """Point-biserial correlation between predicted probability and label."""
    print("\n" + "=" * 70)
    print("V6: PREDICTION-LABEL CORRELATION")
    print("=" * 70)

    rows: list[dict] = []

    for mname, mcls in MODEL_REGISTRY.items():
        print(f"\n--- {mname} ---")
        out = train_and_evaluate(
            mcls, X_train, y_train, X_test, y_test,
            experiment_name=f"V6_corr_{mname}",
            save_plots=False, persist_metrics=False,
        )
        y_prob = out["y_prob"]

        corr, pval = pointbiserialr(y_test, y_prob)
        rows.append({
            "model": mname,
            "point_biserial_r": round(float(corr), 6),
            "p_value": float(pval),
            "pr_auc": round(float(out["metrics"].get("pr_auc", np.nan)), 6),
            "f1": round(float(out["metrics"].get("f1", 0)), 6),
        })
        print(f"  Correlation r={corr:.4f}  p={pval:.2e}")

        try:
            fraction_pos, mean_pred = calibration_curve(
                y_test, y_prob, n_bins=10, strategy="uniform")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(mean_pred, fraction_pos, "o-", label=mname)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfectly calibrated")
            ax.set(xlabel="Mean predicted probability",
                   ylabel="Fraction of positives",
                   title=f"Calibration - {mname}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(RESULTS_DIR / f"v6_calibration_{mname}.png", dpi=120)
            plt.close(fig)
        except Exception:
            pass

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "v6_prediction_label_correlation.csv", index=False)
    print("\n" + df.to_string(index=False))
    return df


def run_best_model_cm_and_weights(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> pd.DataFrame:
    """Train all models, save confusion matrices and weights for the best."""
    print("\n" + "=" * 70)
    print("V7: BEST MODEL CONFUSION MATRIX + WEIGHT SAVING")
    print("=" * 70)

    rows: list[dict] = []
    all_outputs: dict[str, dict] = {}

    for mname, mcls in MODEL_REGISTRY.items():
        out = train_and_evaluate(
            mcls, X_train, y_train, X_test, y_test,
            experiment_name=f"V7_final_{mname}",
            save_plots=True, persist_metrics=True,
            save_weights=True,
        )
        all_outputs[mname] = out
        rows.append({
            "model": mname,
            **{k: round(float(v), 6) if isinstance(v, float) else v
               for k, v in out["metrics"].items()},
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "v7_final_results.csv", index=False)

    if len(df) > 0:
        best_model = df.loc[df["f1"].astype(float).idxmax(), "model"]
        print(f"\nBest model by F1: {best_model}")

        best_out = all_outputs[best_model]
        y_prob = best_out["y_prob"]
        threshold = best_out["metrics"]["threshold"]
        y_pred = (y_prob >= threshold).astype(int)

        plot_confusion(y_test, y_pred, f"V7_BEST_{best_model}")
        print(f"Confusion matrix saved for {best_model}")

    print("\nAll model weights saved to saved_models/")
    print(df[["model", "pr_auc", "f1", "roc_auc",
              "precision", "recall"]].to_string(index=False))
    return df


def run_all_validation_experiments():
    """Execute the full validation suite."""
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("FULL VALIDATION AND ABLATION SUITE")
    print("=" * 70)

    print("\nPreparing base datasets (Abu Dhabi 2022-2025)...")
    train_datasets = [_load_single(y, PRIMARY_TRACK) for y in TRAIN_YEARS]
    test_dataset = _load_single(TEST_YEAR, PRIMARY_TRACK)

    X_train_raw, y_train_raw = _combine(train_datasets)
    X_test, y_test = test_dataset["X"], test_dataset["y"]

    X_train_bal, y_train_bal = _balance(X_train_raw, y_train_raw)

    print(f"Train raw  : {len(y_train_raw)} (pos={int(y_train_raw.sum())})")
    print(f"Train 1:{NEG_POS_RATIO}: {len(y_train_bal)} (pos={int(y_train_bal.sum())})")
    print(f"Test {TEST_YEAR}  : {len(y_test)} (pos={int(y_test.sum())})")

    v1_df = run_feature_group_ablation(X_train_bal, y_train_bal, X_test, y_test)

    v2_df = run_individual_feature_ablation(
        X_train_bal, y_train_bal, X_test, y_test, v1_df, top_n=3)

    v3_df = run_preprocessing_ablation(X_train_raw, y_train_raw, X_test, y_test)

    v4_df = run_italian_exclusion(all_models=True)

    v5_df = run_cross_track_cv()

    v6_df = run_prediction_label_correlation(
        X_train_bal, y_train_bal, X_test, y_test)

    v7_df = run_best_model_cm_and_weights(
        X_train_bal, y_train_bal, X_test, y_test)

    report_lines = [
        "# Validation and Ablation Suite - Results",
        "",
        "## V1: Feature Group Ablation",
        "",
        _markdown_table(v1_df[["group_removed", "features_remaining",
                               "pr_auc", "f1", "roc_auc"]]),
        "",
        "## V2: Individual Feature Ablation",
        "",
        _markdown_table(v2_df[["group", "feature_removed",
                               "pr_auc", "f1"]]) if len(v2_df) > 0
            else "No individual ablation results.",
        "",
        "## V3: Preprocessing Ablation",
        "",
        _markdown_table(v3_df[["preprocessing", "train_samples",
                               "pr_auc", "f1", "roc_auc"]]),
        "",
        "## V4: Italian Race Exclusion",
        "",
        _markdown_table(v4_df[["variant", "model",
                               "pr_auc", "f1"]]) if len(v4_df) > 0
            else "No Italian exclusion results.",
        "",
        "## V5: Cross-Track CV",
        "",
        _markdown_table(v5_df[["fold", "model",
                               "pr_auc", "f1"]]) if len(v5_df) > 0
            else "No cross-track results.",
        "",
        "## V6: Prediction-Label Correlation",
        "",
        _markdown_table(v6_df),
        "",
        "## V7: Final Models (with weights saved)",
        "",
        _markdown_table(v7_df[["model", "pr_auc", "f1",
                               "roc_auc", "precision", "recall"]]),
    ]

    report_path = Path("RESULTS_VALIDATION.md")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nFull report saved to {report_path}")

    print("\n" + "=" * 70)
    print("VALIDATION SUITE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_all_validation_experiments()
