import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, YEARS, PRIMARY_TRACK, ALL_TRACKS, TEMPORAL_FEATURES, N_FEAT
from data_loading.data_loader import load_multiple_races, load_race_pairs
from data_preprocessing_and_labeling.labeling import extract_all_episodes


def _save_dataset(path: Path, X: np.ndarray, y: np.ndarray, meta: pd.DataFrame):
    with open(path, "wb") as f:
        pickle.dump({"X": X, "y": y, "meta": meta}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    pos = int(y.sum())
    print(f"  Saved {path.name}: {len(y)} episodes, {pos} overtakes "
          f"({pos / max(len(y), 1) * 100:.2f}%)")


def load_dataset(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_single_race_dataset(year: int, track: str, tag: str | None = None):
    """Build and save a dataset for one (year, track)."""
    tag = tag or f"{year}_{track.replace(' ', '_')}"
    out_path = DATA_DIR / f"dataset_{tag}.pkl"
    if out_path.exists():
        print(f"  [exists] {out_path.name}")
        return out_path

    pairs = load_race_pairs(year, track)
    if not pairs:
        print(f"  No pairs for {year} {track}")
        return None

    ep = extract_all_episodes(pairs)
    if len(ep["y"]) == 0:
        print(f"  No episodes for {year} {track}")
        return None

    _save_dataset(out_path, ep["X"], ep["y"], ep["meta"])
    return out_path


def build_multi_race_dataset(years: list[int], tracks: list[str],
                             tag: str):
    """Build and save a merged dataset for multiple (year, track) combos."""
    out_path = DATA_DIR / f"dataset_{tag}.pkl"
    if out_path.exists():
        print(f"  [exists] {out_path.name}")
        return out_path

    pairs = load_multiple_races(years, tracks)
    if not pairs:
        print(f"  No pairs for {tag}")
        return None

    ep = extract_all_episodes(pairs)
    if len(ep["y"]) == 0:
        print(f"  No episodes for {tag}")
        return None

    _save_dataset(out_path, ep["X"], ep["y"], ep["meta"])
    return out_path


def build_all_datasets():
    """Build every dataset needed for the experiment matrix."""
    print("=" * 60)
    print("Building datasets")
    print("=" * 60)

    for year in YEARS:
        print(f"\n--- Abu Dhabi {year} ---")
        build_single_race_dataset(year, PRIMARY_TRACK)

    print("\n--- Abu Dhabi 2022+2023 (merged) ---")
    build_multi_race_dataset([2022, 2023], [PRIMARY_TRACK],
                             tag="AbuDhabi_2022_2023")

    print("\n--- Abu Dhabi 2022+2023+2024 (merged) ---")
    build_multi_race_dataset([2022, 2023, 2024], [PRIMARY_TRACK],
                             tag="AbuDhabi_2022_2023_2024")

    for track in ALL_TRACKS:
        if track == PRIMARY_TRACK:
            continue
        for year in [2022, 2023]:
            print(f"\n--- {track} {year} ---")
            build_single_race_dataset(year, track)

    print("\n--- All tracks 2022-2023 ---")
    build_multi_race_dataset([2022, 2023], ALL_TRACKS,
                             tag="AllTracks_2022_2023")

    print("\n" + "=" * 60)
    print("Dataset build complete.")


if __name__ == "__main__":
    build_all_datasets()
