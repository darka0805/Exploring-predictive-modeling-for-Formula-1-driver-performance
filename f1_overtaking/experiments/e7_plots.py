from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.e7_config import NEG_POS_RATIO


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
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Episode count")
    ax.set_title(f"Class distribution before/after 1:{NEG_POS_RATIO} balancing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


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

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_tuning_quality(tuning_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, group in tuning_df.groupby("model"):
        vals = group.sort_values("pr_auc", ascending=False)["pr_auc"].values
        ax.plot(np.arange(1, len(vals) + 1), vals, marker="o", label=model_name)
    ax.set_xlabel("Trial rank (best to worst)")
    ax.set_ylabel("Validation PR-AUC")
    ax.set_title("Hyperparameter sweep quality per model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


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

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


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

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
