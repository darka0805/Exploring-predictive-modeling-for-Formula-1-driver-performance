import numpy as np
import pandas as pd
from pathlib import Path

from config import ( RESULTS_DIR, PRIMARY_TRACK, ALL_TRACKS
                    )
from data_preprocessing_and_labeling.dataset_builder import build_single_race_dataset, build_multi_race_dataset, load_dataset
from train import train_and_evaluate
from models import MODEL_REGISTRY


def _load(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return load_dataset(path)


def _merge_datasets(*datasets) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate multiple dataset dicts into (X, y)."""
    Xs, ys = [], []
    for d in datasets:
        if d is not None:
            Xs.append(d["X"])
            ys.append(d["y"])
    return np.concatenate(Xs), np.concatenate(ys)


def _add_race_id(X: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Append a race_id feature (encoded as integer) to each timestep."""
    events = meta["event"].astype("category").cat.codes.values
    race_id = events[:, None, None] * np.ones((1, X.shape[1], 1))
    return np.concatenate([X, race_id.astype(np.float32)], axis=2)


def run_experiment(name: str, X_train, y_train, X_test, y_test,
                   model_names: list[str] | None = None) -> list[dict]:
    """Train all models on one experiment split and return results."""
    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())

    results = []
    for mname in model_names:
        cls = MODEL_REGISTRY[mname]
        r = train_and_evaluate(cls, X_train, y_train, X_test, y_test,
                               experiment_name=name)
        results.append({"experiment": name, "model": mname,
                        **r["metrics"]})
    return results


def run_all_experiments(model_names: list[str] | None = None) -> pd.DataFrame:
    """Execute the full experiment matrix and return a results DataFrame."""
    all_results: list[dict] = []

    # Load / build datasets
    print("\n\n" + "=" * 70)
    print("PHASE 1: Preparing datasets")
    print("=" * 70)

    ad22_p = build_single_race_dataset(2022, PRIMARY_TRACK)
    ad23_p = build_single_race_dataset(2023, PRIMARY_TRACK)
    ad24_p = build_single_race_dataset(2024, PRIMARY_TRACK)
    ad22_23_p = build_multi_race_dataset([2022, 2023], [PRIMARY_TRACK],
                                         tag="AbuDhabi_2022_2023")

    ad22 = _load(ad22_p)
    ad23 = _load(ad23_p)
    ad24 = _load(ad24_p)
    ad22_23 = _load(ad22_23_p)

    # E1: Single-year baseline
    print("\n\n" + "=" * 70)
    print("E1: Single-year — Abu Dhabi 2022 → 2023")
    print("=" * 70)
    if ad22 and ad23:
        all_results.extend(run_experiment(
            "E1_single_year", ad22["X"], ad22["y"],
            ad23["X"], ad23["y"], model_names))

    # E2: Multi-year
    print("\n\n" + "=" * 70)
    print("E2: Multi-year — Abu Dhabi 2022+2023 → 2024")
    print("=" * 70)
    if ad22_23 and ad24:
        all_results.extend(run_experiment(
            "E2_multi_year", ad22_23["X"], ad22_23["y"],
            ad24["X"], ad24["y"], model_names))

    # E3: Leave-one-year-out
    print("\n\n" + "=" * 70)
    print("E3: Leave-one-year-out")
    print("=" * 70)
    for hold_year, hold_ds in [(2022, ad22), (2023, ad23), (2024, ad24)]:
        if hold_ds is None:
            continue
        train_parts = [d for y, d in [(2022, ad22), (2023, ad23), (2024, ad24)]
                       if y != hold_year and d is not None]
        if not train_parts:
            continue
        Xt, yt = _merge_datasets(*train_parts)
        all_results.extend(run_experiment(
            f"E3_loyo_test{hold_year}", Xt, yt,
            hold_ds["X"], hold_ds["y"], model_names))

    # E4: Race ID as feature 
    print("\n\n" + "=" * 70)
    print("E4: Race ID feature — Abu Dhabi 2022+2023 → 2024")
    print("=" * 70)
    if ad22_23 and ad24:
        X_tr_id = _add_race_id(ad22_23["X"], ad22_23["meta"])
        X_te_id = _add_race_id(ad24["X"], ad24["meta"])
        tree_models = [m for m in (model_names or list(MODEL_REGISTRY.keys()))
                       if m in ("xgboost", "rf")]
        if tree_models:
            all_results.extend(run_experiment(
                "E4_race_id", X_tr_id, ad22_23["y"],
                X_te_id, ad24["y"], tree_models))

    # E5: Cross-track 
    print("\n\n" + "=" * 70)
    print("E5: Cross-track")
    print("=" * 70)
    for track in ALL_TRACKS:
        if track == PRIMARY_TRACK:
            continue
        tp = build_single_race_dataset(2023, track)
        td = _load(tp)
        if td is None or ad22_23 is None:
            continue
        tree_models = [m for m in (model_names or list(MODEL_REGISTRY.keys()))
                       if m in ("xgboost", "rf")]
        all_results.extend(run_experiment(
            f"E5_cross_{track.split()[0]}_2023",
            ad22_23["X"], ad22_23["y"],
            td["X"], td["y"], tree_models))

    # E6: All tracks combined 
    print("\n\n" + "=" * 70)
    print("E6: All tracks 2022-2023 → Abu Dhabi 2024")
    print("=" * 70)
    all_p = build_multi_race_dataset([2022, 2023], ALL_TRACKS,
                                     tag="AllTracks_2022_2023")
    all_ds = _load(all_p)
    if all_ds and ad24:
        tree_models = [m for m in (model_names or list(MODEL_REGISTRY.keys()))
                       if m in ("xgboost",)]
        all_results.extend(run_experiment(
            "E6_all_tracks", all_ds["X"], all_ds["y"],
            ad24["X"], ad24["y"], tree_models))

    #    Compile results 
    df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / "experiment_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    print("\n\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    if len(df) > 0:
        display_cols = ["experiment", "model", "roc_auc", "pr_auc",
                        "f1", "precision", "recall", "n_pos", "n_total"]
        existing = [c for c in display_cols if c in df.columns]
        print(df[existing].to_string(index=False))
    return df


if __name__ == "__main__":
    run_all_experiments()
