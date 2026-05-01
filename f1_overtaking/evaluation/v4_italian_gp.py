from __future__ import annotations

import numpy as np
import pandas as pd

from config import RESULTS_DIR, PRIMARY_TRACK, ALL_TRACKS, SEED
from data_preprocessing_and_labeling.dataset_builder import (
    build_multi_race_dataset, build_single_race_dataset, load_dataset)
from models import MODEL_REGISTRY
from train import train_and_evaluate


TRAIN_YEARS = [2022, 2023, 2024]
TEST_YEAR = 2025
NEG_POS_RATIO = 20


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


def run_italian_exclusion(all_models: bool = True) -> pd.DataFrame:
    """Compare training with and without Italian GP (Monza)."""
    print("\n" + "=" * 70)
    print("V4: ITALIAN RACE EXCLUSION TEST")
    print("=" * 70)

    rows: list[dict] = []
    model_names = (list(MODEL_REGISTRY.keys()) if all_models
                   else ["xgboost", "lgbm"])

    print("\n--- Baseline: All tracks (Abu Dhabi + Bahrain + Monza) ---")
    all_path = build_multi_race_dataset(
        TRAIN_YEARS, ALL_TRACKS, tag="AllTracks_2022_2023_2024")
    test_path = build_single_race_dataset(TEST_YEAR, PRIMARY_TRACK)

    if all_path and test_path:
        all_ds = load_dataset(all_path)
        test_ds = load_dataset(test_path)
        X_all, y_all = _balance(all_ds["X"], all_ds["y"])
        X_test, y_test = test_ds["X"], test_ds["y"]

        for mname in model_names:
            out = train_and_evaluate(
                MODEL_REGISTRY[mname], X_all, y_all, X_test, y_test,
                experiment_name=f"V4_with_italy_{mname}",
                save_plots=True, persist_metrics=False,
            )
            rows.append({
                "variant": "with_italy",
                "model": mname,
                **{k: round(float(v), 6) if isinstance(v, float) else v
                   for k, v in out["metrics"].items()},
            })

    print("\n--- No Italy: Abu Dhabi + Bahrain only ---")
    no_it_tracks = [t for t in ALL_TRACKS if "Italian" not in t]
    no_it_path = build_multi_race_dataset(
        TRAIN_YEARS, no_it_tracks, tag="NoItaly_2022_2023_2024")

    if no_it_path and test_path:
        no_it_ds = load_dataset(no_it_path)
        test_ds = load_dataset(test_path)
        X_noi, y_noi = _balance(no_it_ds["X"], no_it_ds["y"])
        X_test, y_test = test_ds["X"], test_ds["y"]

        for mname in model_names:
            out = train_and_evaluate(
                MODEL_REGISTRY[mname], X_noi, y_noi, X_test, y_test,
                experiment_name=f"V4_no_italy_{mname}",
                save_plots=True, persist_metrics=False,
            )
            rows.append({
                "variant": "no_italy",
                "model": mname,
                **{k: round(float(v), 6) if isinstance(v, float) else v
                   for k, v in out["metrics"].items()},
            })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "v4_italian_exclusion.csv", index=False)
    if len(df) > 0:
        print("\n" + df[["variant", "model", "pr_auc", "f1",
                         "roc_auc", "precision", "recall"]].to_string(index=False))
    return df


def run_v4() -> pd.DataFrame:
    return run_italian_exclusion(all_models=True)


if __name__ == "__main__":
    run_v4()
