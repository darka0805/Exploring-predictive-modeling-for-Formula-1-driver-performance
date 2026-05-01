import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    f1_score, precision_score, recall_score,
    confusion_matrix
)

from config import RESULTS_DIR, E7_NEG_POS_RATIO


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float | None = None) -> dict:
    """Compute a full set of binary classification metrics.

    If threshold is None, the optimal F1 threshold is found automatically.
    """
    metrics: dict = {}

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
        metrics["f1"] = 0.0
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["threshold"] = 0.5
        metrics["n_total"] = int(len(y_true))
        metrics["n_pos"] = int(y_true.sum())
        return metrics

    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))

    if threshold is None:
        prec_arr, rec_arr, thr_arr = precision_recall_curve(y_true, y_prob)
        f1_arr = 2 * prec_arr * rec_arr / np.clip(prec_arr + rec_arr, 1e-8, None)
        best_idx = np.argmax(f1_arr)
        threshold = float(thr_arr[min(best_idx, len(thr_arr) - 1)])

    y_pred = (y_prob >= threshold).astype(int)
    metrics["threshold"] = threshold
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["n_total"] = int(len(y_true))
    metrics["n_pos"] = int(y_true.sum())
    metrics["n_pred_pos"] = int(y_pred.sum())

    return metrics


def plot_pr_roc(y_true: np.ndarray, y_prob: np.ndarray,
                label: str, save_dir: Path | None = None):
    """Save PR and ROC curves side by side."""
    if save_dir is None:
        save_dir = RESULTS_DIR
    save_dir.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ax1.plot(rec, prec)
    ax1.set(xlabel="Recall", ylabel="Precision",
            title=f"PR Curve (AP={ap:.3f})")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax2.plot(fpr, tpr)
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax2.set(xlabel="FPR", ylabel="TPR",
            title=f"ROC Curve (AUC={auc:.3f})")

    fig.suptitle(label, fontsize=13)
    plt.tight_layout()
    safe_name = label.replace(" ", "_").replace("/", "_")
    fig.savefig(save_dir / f"curves_{safe_name}.png", dpi=120)
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray,
                   label: str, save_dir: Path | None = None):
    """Save a confusion matrix heatmap."""
    if save_dir is None:
        save_dir = RESULTS_DIR
    save_dir.mkdir(exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xlabel="Predicted", ylabel="Actual",
           title=f"Confusion Matrix — {label}",
           xticks=[0, 1], yticks=[0, 1])
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    safe_name = label.replace(" ", "_").replace("/", "_")
    fig.savefig(save_dir / f"cm_{safe_name}.png", dpi=120)
    plt.close(fig)


def save_metrics(metrics: dict, experiment_name: str,
                 model_name: str, save_dir: Path | None = None):
    """Append metrics to a JSON-lines file for later analysis."""
    if save_dir is None:
        save_dir = RESULTS_DIR
    save_dir.mkdir(exist_ok=True)

    row = {"experiment": experiment_name, "model": model_name, **metrics}
    path = save_dir / "all_metrics.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def print_metrics(metrics: dict, label: str = ""):
    """Pretty-print a metrics dict."""
    print(f"\n{'─' * 50}")
    if label:
        print(f"  {label}")
        print(f"{'─' * 50}")
    print(f"  ROC-AUC   : {metrics.get('roc_auc', float('nan')):.4f}")
    print(f"  PR-AUC    : {metrics.get('pr_auc', float('nan')):.4f}")
    print(f"  F1        : {metrics.get('f1', 0):.4f}")
    print(f"  Precision : {metrics.get('precision', 0):.4f}")
    print(f"  Recall    : {metrics.get('recall', 0):.4f}")
    print(f"  Threshold : {metrics.get('threshold', 0.5):.4f}")
    print(f"  Samples   : {metrics.get('n_total', '?')} "
          f"(pos={metrics.get('n_pos', '?')})")
    print(f"{'─' * 50}")


# ----- E7 plot helpers -----

def plot_class_balance(y_train_raw: np.ndarray,
                       y_train_bal: np.ndarray,
                       y_test: np.ndarray,
                       out_path: Path):
    labels = ["train_raw", "train_balanced", "test_2025"]
    pos = [int((y_train_raw == 1).sum()),
           int((y_train_bal == 1).sum()),
           int((y_test == 1).sum())]
    neg = [int((y_train_raw == 0).sum()),
           int((y_train_bal == 0).sum()),
           int((y_test == 0).sum())]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, pos, width, label="positive")
    ax.bar(x + width / 2, neg, width, label="negative")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Episode count")
    ax.set_title(f"Class distribution before/after 1:{E7_NEG_POS_RATIO} balancing")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_final_model_comparison(final_df: pd.DataFrame, out_path: Path):
    view = final_df.sort_values("pr_auc", ascending=False).copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax1.bar(view["model"], view["pr_auc"])
    ax1.set_title("Final test PR-AUC by model")
    ax1.set_ylim(0, max(0.05, float(view["pr_auc"].max()) * 1.15))
    ax1.tick_params(axis="x", rotation=20)
    ax2.bar(view["model"], view["f1"])
    ax2.set_title("Final test F1 by model")
    ax2.set_ylim(0, max(0.05, float(view["f1"].max()) * 1.15))
    ax2.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_tuning_quality(tuning_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, group in tuning_df.groupby("model"):
        vals = group.sort_values("pr_auc", ascending=False)["pr_auc"].values
        ax.plot(np.arange(1, len(vals) + 1), vals, marker="o", label=model_name)
    ax.set_xlabel("Trial rank (best to worst)")
    ax.set_ylabel("Validation PR-AUC")
    ax.set_title("Hyperparameter sweep quality per model")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_combined_model_comparison(base_df: pd.DataFrame,
                                   stacking_df: pd.DataFrame,
                                   out_path: Path):
    combined = base_df[["model", "pr_auc", "f1"]].copy()
    combined["kind"] = "base"
    if len(stacking_df) > 0:
        stack_view = stacking_df[["ensemble", "pr_auc", "f1"]].copy()
        stack_view = stack_view.rename(columns={"ensemble": "model"})
        stack_view["kind"] = "stack"
        combined = pd.concat([combined, stack_view], ignore_index=True)
    combined = combined.sort_values("pr_auc", ascending=False)
    labels = [f"{m} ({k})" for m, k in zip(combined["model"], combined["kind"])]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    ax1.bar(labels, combined["pr_auc"])
    ax1.set_title("Base vs stacking PR-AUC")
    ax1.set_ylim(0, max(0.05, float(combined["pr_auc"].max()) * 1.15))
    ax1.tick_params(axis="x", rotation=30)
    ax2.bar(labels, combined["f1"])
    ax2.set_title("Base vs stacking F1")
    ax2.set_ylim(0, max(0.05, float(combined["f1"].max()) * 1.15))
    ax2.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_full_rerun_comparison(full_rerun_df: pd.DataFrame, out_path: Path):
    if len(full_rerun_df) == 0:
        return
    view = full_rerun_df.sort_values("f1", ascending=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    ax1.bar(view["model"], view["f1"])
    ax1.set_title("Full-data rerun F1")
    ax1.set_ylim(0, max(0.05, float(view["f1"].max()) * 1.15))
    ax1.tick_params(axis="x", rotation=20)
    ax2.bar(view["model"], view["pr_auc"])
    ax2.set_title("Full-data rerun PR-AUC")
    ax2.set_ylim(0, max(0.05, float(view["pr_auc"].max()) * 1.15))
    ax2.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
