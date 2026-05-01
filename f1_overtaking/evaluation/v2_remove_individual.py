from __future__ import annotations

import numpy as np
import pandas as pd

from config import (RESULTS_DIR, TEMPORAL_FEATURES, FEATURE_GROUPS, N_FEAT)
from models import MODEL_REGISTRY
from train import train_and_evaluate


def _feature_indices(feature_names: list[str]) -> list[int]:
    name2idx = {n: i for i, n in enumerate(TEMPORAL_FEATURES)}
    return [name2idx[n] for n in feature_names if n in name2idx]


def _select_features(X: np.ndarray, keep_indices: list[int]) -> np.ndarray:
    return X[:, :, keep_indices]


def run_individual_feature_ablation(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    group_ablation_df: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """Ablate individual features within the top-N most impactful groups."""
    print("\n" + "=" * 70)
    print("V2: INDIVIDUAL FEATURE ABLATION (top groups)")
    print("=" * 70)

    baseline_row = group_ablation_df[
        group_ablation_df["group_removed"] == "NONE (baseline)"
    ]
    baseline_prauc = float(baseline_row["pr_auc"].iloc[0])

    group_rows = group_ablation_df[
        group_ablation_df["group_removed"] != "NONE (baseline)"
    ].copy()
    group_rows["prauc_drop"] = baseline_prauc - group_rows["pr_auc"].astype(float)
    top_groups = group_rows.nlargest(top_n, "prauc_drop")["group_removed"].tolist()

    print(f"Top {top_n} impactful groups: {top_groups}")

    model_cls = MODEL_REGISTRY["xgboost"]
    rows: list[dict] = []

    for gname in top_groups:
        feats_in_group = FEATURE_GROUPS[gname]
        for feat in feats_in_group:
            fidx = _feature_indices([feat])
            if not fidx:
                continue
            keep_idx = [i for i in range(N_FEAT) if i != fidx[0]]
            print(f"\n--- Remove individual: {feat} from {gname} ---")

            X_tr_sub = _select_features(X_train, keep_idx)
            X_te_sub = _select_features(X_test, keep_idx)

            out = train_and_evaluate(
                model_cls, X_tr_sub, y_train, X_te_sub, y_test,
                experiment_name=f"V2_indiv_no_{feat}",
                save_plots=False, persist_metrics=False,
            )
            rows.append({
                "group": gname,
                "feature_removed": feat,
                **{k: round(float(v), 6) if isinstance(v, float) else v
                   for k, v in out["metrics"].items()},
            })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "v2_individual_feature_ablation.csv", index=False)
    if len(df) > 0:
        print("\n" + df[["group", "feature_removed",
                         "pr_auc", "f1", "roc_auc"]].to_string(index=False))
    return df


def run_v2() -> pd.DataFrame:
    """Standalone runner: builds datasets, runs V1, then V2."""
    from config import PRIMARY_TRACK, ALL_TRACKS, SEED
    from data_preprocessing_and_labeling.dataset_builder import (
        build_multi_race_dataset, build_single_race_dataset, load_dataset)
    from evaluation.v1_remove_groups import run_feature_group_ablation

    train_path = build_multi_race_dataset(
        [2022, 2023], ALL_TRACKS, tag="AllTracks_2022_2023")
    test_path = build_single_race_dataset(2024, PRIMARY_TRACK)
    if train_path is None or test_path is None:
        raise SystemExit("V2: required datasets missing.")

    train_ds = load_dataset(train_path)
    test_ds = load_dataset(test_path)

    pos = np.where(train_ds["y"] == 1)[0]
    neg = np.where(train_ds["y"] == 0)[0]
    rng = np.random.default_rng(SEED)
    n_neg = min(len(neg), len(pos) * 20)
    keep = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
    rng.shuffle(keep)
    Xtr, ytr = train_ds["X"][keep], train_ds["y"][keep]

    v1 = run_feature_group_ablation(Xtr, ytr, test_ds["X"], test_ds["y"])
    return run_individual_feature_ablation(
        Xtr, ytr, test_ds["X"], test_ds["y"], v1, top_n=3)


if __name__ == "__main__":
    run_v2()
