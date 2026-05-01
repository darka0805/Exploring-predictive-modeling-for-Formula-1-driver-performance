import numpy as np
import pandas as pd

from .metrics import summarize_eval


_METRIC_LABELS = {
    "spearman": "Spearman rho (higher)",
    "mae": "MAE positions (lower)",
    "ndcg": "NDCG (higher)",
    "top3": "Top-3 accuracy (higher)",
    "top5": "Top-5 accuracy (higher)",
    "within2": "Within +/-2 pos (higher)",
}


def print_overall_summary(metrics_df: pd.DataFrame, model_name: str = "Model"):
    """Print Model vs Grid Baseline mean metrics with deltas."""
    print("=" * 80)
    print(f"OVERALL EVALUATION SUMMARY — {model_name} (Dry races only)")
    print("=" * 80)
    print(f'{"Metric":<30s} {model_name:>10s} {"Grid Baseline":>15s} {"Delta":>10s}')
    print("-" * 80)
    for key, label in _METRIC_LABELS.items():
        m = metrics_df[f"{key}_model"].mean()
        b = metrics_df[f"{key}_baseline"].mean()
        print(f"{label:<30s} {m:>10.3f} {b:>15.3f} {m - b:>+10.3f}")
    print("=" * 80)


def print_mae_by_position_group(predictions: pd.DataFrame):
    """MAE bucketed by finishing-position group (P1-3, P4-5, …, P16-20)."""
    predictions = predictions.copy()
    predictions["pos_group"] = pd.cut(
        predictions["finish_position"],
        bins=[0, 3, 5, 10, 15, 20],
        labels=["P1-3", "P4-5", "P6-10", "P11-15", "P16-20"],
    )
    predictions["abs_error"] = np.abs(predictions["pred_position"] - predictions["finish_position"])
    predictions["baseline_error"] = np.abs(predictions["grid_position"] - predictions["finish_position"])

    print("MAE by finishing position group:")
    print(f'{"Group":<10s} {"Model MAE":>10s} {"Baseline MAE":>13s}')
    print("-" * 35)
    for group, g in predictions.groupby("pos_group", observed=True):
        m = g["abs_error"].mean()
        b = g["baseline_error"].mean()
        print(f"{str(group):<10s} {m:>10.2f} {b:>13.2f}")
    print(f'{"OVERALL":<10s} {predictions["abs_error"].mean():>10.2f} '
          f'{predictions["baseline_error"].mean():>13.2f}')


def print_models_comparison(eval_dfs: dict[str, pd.DataFrame]):
    """Side-by-side comparison of multiple models on the same folds.

    `eval_dfs` maps model name -> output of `evaluate_predictions`.
    The grid baseline is read from the first eval_df (identical across all).
    """
    if not eval_dfs:
        print("No models to compare.")
        return

    first_eval = next(iter(eval_dfs.values()))
    grid_summary = summarize_eval(first_eval, suffix="baseline")

    print("=" * 100)
    print("MODELS COMPARISON")
    print("=" * 100)
    header = f'{"Metric":<20s} ' + " ".join(f"{name:>15s}" for name in eval_dfs) + f' {"Grid Baseline":>15s}'
    print(header)
    print("-" * len(header))

    for key, label in _METRIC_LABELS.items():
        row = f"{label:<20s} "
        for ev in eval_dfs.values():
            row += f"{ev[f'{key}_model'].mean():>15.4f} "
        row += f"{grid_summary[key]:>15.4f}"
        print(row)
    print("=" * 100)
