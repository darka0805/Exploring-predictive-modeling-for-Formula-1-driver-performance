import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, ndcg_score


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per race with `*_model` and `*_baseline` columns.

    Baseline = grid_position used directly as the prediction.
    """
    rows = []
    for rid, grp in predictions.groupby("race_id"):
        actual = grp["finish_position"].values
        predicted = grp["pred_position"].values
        grid = grp["grid_position"].values
        event = grp["event_name"].iloc[0]
        year = int(grp["year"].iloc[0])
        rnd = int(grp["round_num"].iloc[0])

        sp_model, _ = spearmanr(actual, predicted)
        sp_baseline, _ = spearmanr(actual, grid)

        mae_model = mean_absolute_error(actual, predicted)
        mae_baseline = mean_absolute_error(actual, grid)

        n = len(actual)
        actual_rel = np.clip(n + 1 - actual, 0, None)
        pred_rel = np.clip(n + 1 - predicted, 0, None)
        grid_rel = np.clip(n + 1 - grid, 0, None)
        try:
            ndcg_model = ndcg_score([actual_rel], [pred_rel])
            ndcg_baseline = ndcg_score([actual_rel], [grid_rel])
        except Exception:
            ndcg_model = ndcg_baseline = np.nan

        actual_top3 = set(grp.nsmallest(3, "finish_position")["Driver"])
        pred_top3 = set(grp.nsmallest(3, "pred_position")["Driver"])
        grid_top3 = set(grp.nsmallest(3, "grid_position")["Driver"])
        actual_top5 = set(grp.nsmallest(5, "finish_position")["Driver"])
        pred_top5 = set(grp.nsmallest(5, "pred_position")["Driver"])
        grid_top5 = set(grp.nsmallest(5, "grid_position")["Driver"])

        rows.append({
            "race_id": int(rid), "year": year, "round_num": rnd,
            "event": event, "n_drivers": n,
            "spearman_model": sp_model, "spearman_baseline": sp_baseline,
            "mae_model": mae_model, "mae_baseline": mae_baseline,
            "ndcg_model": ndcg_model, "ndcg_baseline": ndcg_baseline,
            "top3_model": len(actual_top3 & pred_top3) / 3,
            "top3_baseline": len(actual_top3 & grid_top3) / 3,
            "top5_model": len(actual_top5 & pred_top5) / 5,
            "top5_baseline": len(actual_top5 & grid_top5) / 5,
            "within2_model": np.mean(np.abs(actual - predicted) <= 2),
            "within2_baseline": np.mean(np.abs(actual - grid) <= 2),
        })

    return pd.DataFrame(rows)


def summarize_eval(metrics_df: pd.DataFrame, suffix: str = "model") -> dict:
    """Mean each metric across races. `suffix` is 'model' or 'baseline'."""
    return {
        "spearman": float(metrics_df[f"spearman_{suffix}"].mean()),
        "mae": float(metrics_df[f"mae_{suffix}"].mean()),
        "ndcg": float(metrics_df[f"ndcg_{suffix}"].mean()),
        "top3": float(metrics_df[f"top3_{suffix}"].mean()),
        "top5": float(metrics_df[f"top5_{suffix}"].mean()),
        "within2": float(metrics_df[f"within2_{suffix}"].mean()),
    }
