from __future__ import annotations

import numpy as np
import pandas as pd

from config import (RESULTS_DIR, TEMPORAL_FEATURES, FEATURE_GROUPS, N_FEAT)
from models import MODEL_REGISTRY
from train import train_and_evaluate


def _feature_indices(feature_names: list[str]) -> list[int]:
    name2idx = {n: i for i, n in enumerate(TEMPORAL_FEATURES)}
    return [name2idx[n] for n in feature_names if n in name2idx]


def _remove_group_indices(group_name: str) -> list[int]:
    """Indices of features to KEEP when one group is removed."""
    remove = set(_feature_indices(FEATURE_GROUPS[group_name]))
    return [i for i in range(N_FEAT) if i not in remove]


def _select_features(X: np.ndarray, keep_indices: list[int]) -> np.ndarray:
    return X[:, :, keep_indices]


def run_feature_group_ablation(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> pd.DataFrame:
    """Remove one feature group at a time, measure impact on XGBoost."""
    print("\n" + "=" * 70)
    print("V1: FEATURE GROUP ABLATION")
    print("=" * 70)

    model_cls = MODEL_REGISTRY["xgboost"]
    rows: list[dict] = []

    print("\n--- Baseline (all 45 features) ---")
    out = train_and_evaluate(
        model_cls, X_train, y_train, X_test, y_test,
        experiment_name="V1_ablation_baseline",
        save_plots=False, persist_metrics=False,
    )
    rows.append({
        "group_removed": "NONE (baseline)",
        "features_remaining": N_FEAT,
        **{k: round(float(v), 6) if isinstance(v, float) else v
           for k, v in out["metrics"].items()},
    })

    for gname, gfeats in FEATURE_GROUPS.items():
        keep_idx = _remove_group_indices(gname)
        print(f"\n--- Remove {gname} ({len(gfeats)} feats) -> "
              f"{len(keep_idx)} remain ---")

        X_tr_sub = _select_features(X_train, keep_idx)
        X_te_sub = _select_features(X_test, keep_idx)

        out = train_and_evaluate(
            model_cls, X_tr_sub, y_train, X_te_sub, y_test,
            experiment_name=f"V1_ablation_no_{gname}",
            save_plots=False, persist_metrics=False,
        )
        rows.append({
            "group_removed": gname,
            "features_remaining": len(keep_idx),
            **{k: round(float(v), 6) if isinstance(v, float) else v
               for k, v in out["metrics"].items()},
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "v1_feature_group_ablation.csv", index=False)
    print("\n" + df[["group_removed", "features_remaining",
                     "pr_auc", "f1", "roc_auc"]].to_string(index=False))
    return df


def run_v1() -> pd.DataFrame:
    """Standalone runner: builds datasets and runs V1."""
    from config import PRIMARY_TRACK, ALL_TRACKS, SEED
    from data_preprocessing_and_labeling.dataset_builder import (
        build_multi_race_dataset, build_single_race_dataset, load_dataset)

    train_path = build_multi_race_dataset(
        [2022, 2023], ALL_TRACKS, tag="AllTracks_2022_2023")
    test_path = build_single_race_dataset(2024, PRIMARY_TRACK)
    if train_path is None or test_path is None:
        raise SystemExit("V1: required datasets missing.")

    train_ds = load_dataset(train_path)
    test_ds = load_dataset(test_path)

    pos = np.where(train_ds["y"] == 1)[0]
    neg = np.where(train_ds["y"] == 0)[0]
    rng = np.random.default_rng(SEED)
    n_neg = min(len(neg), len(pos) * 20)
    keep = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
    rng.shuffle(keep)
    return run_feature_group_ablation(
        train_ds["X"][keep], train_ds["y"][keep],
        test_ds["X"], test_ds["y"])


if __name__ == "__main__":
    run_v1()
