"""V5: 3-fold cross-track cross-validation (Monza / Bahrain / Abu Dhabi)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import RESULTS_DIR, ALL_TRACKS, SEED
from data_preprocessing_and_labeling.dataset_builder import (
    build_multi_race_dataset, load_dataset)
from models import MODEL_REGISTRY
from train import train_and_evaluate


TRAIN_YEARS = [2022, 2023, 2024]
NEG_POS_RATIO = 20


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


def run_cross_track_cv() -> pd.DataFrame:
    """3-fold cross-track CV across Monza, Bahrain, Abu Dhabi."""
    print("\n" + "=" * 70)
    print("V5: 3-FOLD CROSS-TRACK CV")
    print("=" * 70)

    track_labels = {
        "Abu Dhabi Grand Prix": "AbuDhabi",
        "Bahrain Grand Prix": "Bahrain",
        "Italian Grand Prix": "Monza",
    }

    track_ds: dict[str, dict] = {}
    for track in ALL_TRACKS:
        tag = f"CV_{track_labels[track]}_2022_2024"
        path = build_multi_race_dataset(TRAIN_YEARS, [track], tag=tag)
        if path:
            track_ds[track] = load_dataset(path)

    folds = [
        {"name": "Fold1_test_Monza",
         "train": ["Bahrain Grand Prix", "Abu Dhabi Grand Prix"],
         "test": "Italian Grand Prix"},
        {"name": "Fold2_test_AbuDhabi",
         "train": ["Italian Grand Prix", "Bahrain Grand Prix"],
         "test": "Abu Dhabi Grand Prix"},
        {"name": "Fold3_test_Bahrain",
         "train": ["Italian Grand Prix", "Abu Dhabi Grand Prix"],
         "test": "Bahrain Grand Prix"},
    ]

    rows: list[dict] = []
    model_names = list(MODEL_REGISTRY.keys())

    for fold in folds:
        print(f"\n{'-' * 60}")
        print(f"  {fold['name']}: "
              f"train={[track_labels[t] for t in fold['train']]} "
              f"-> test={track_labels[fold['test']]}")
        print(f"{'-' * 60}")

        train_parts = [track_ds[t] for t in fold["train"] if t in track_ds]
        if not train_parts or fold["test"] not in track_ds:
            print("  SKIP - missing data")
            continue

        X_tr, y_tr = _combine(train_parts)
        X_tr, y_tr = _balance(X_tr, y_tr)
        test_d = track_ds[fold["test"]]
        X_te, y_te = test_d["X"], test_d["y"]

        for mname in model_names:
            out = train_and_evaluate(
                MODEL_REGISTRY[mname], X_tr, y_tr, X_te, y_te,
                experiment_name=f"V5_{fold['name']}_{mname}",
                save_plots=True, persist_metrics=False,
                save_weights=True,
            )
            rows.append({
                "fold": fold["name"],
                "model": mname,
                **{k: round(float(v), 6) if isinstance(v, float) else v
                   for k, v in out["metrics"].items()},
            })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "v5_cross_track_cv.csv", index=False)

    if len(df) > 0:
        print("\n\nPER-FOLD RESULTS:")
        print(df[["fold", "model", "pr_auc", "f1", "roc_auc",
                  "precision", "recall"]].to_string(index=False))

        avg = (df.groupby("model")[["pr_auc", "f1", "roc_auc"]]
               .mean().reset_index())
        avg.columns = ["model", "avg_pr_auc", "avg_f1", "avg_roc_auc"]
        print("\nAVERAGE ACROSS FOLDS:")
        print(avg.to_string(index=False))
        avg.to_csv(RESULTS_DIR / "v5_cross_track_cv_avg.csv", index=False)

    return df


def run_v5() -> pd.DataFrame:
    return run_cross_track_cv()


if __name__ == "__main__":
    run_v5()
