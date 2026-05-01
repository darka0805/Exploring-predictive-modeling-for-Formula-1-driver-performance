from __future__ import annotations

import numpy as np

from config import SEED
from data_preprocessing_and_labeling.dataset_builder import (
    build_single_race_dataset, load_dataset)
from experiments.e7_config import NEG_POS_RATIO


def to_1_to_k_balance(X: np.ndarray, y: np.ndarray,
                      neg_per_pos: int = NEG_POS_RATIO,
                      seed: int = SEED
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep all positives and sample negatives to achieve ~1:neg_per_pos."""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if len(pos_idx) == 0:
        raise ValueError(
            f"No positive samples found; cannot create 1:{neg_per_pos} train ratio."
        )

    n_neg_keep = min(len(neg_idx), len(pos_idx) * neg_per_pos)
    rng = np.random.default_rng(seed)

    if n_neg_keep < len(neg_idx):
        keep_neg_idx = rng.choice(neg_idx, size=n_neg_keep, replace=False)
    else:
        keep_neg_idx = neg_idx

    keep_idx = np.concatenate([pos_idx, keep_neg_idx])
    rng.shuffle(keep_idx)

    return X[keep_idx], y[keep_idx], keep_idx


def load_single_track_year(year: int, track: str) -> dict:
    path = build_single_race_dataset(year, track)
    if path is None:
        raise RuntimeError(
            f"Failed to build dataset for year={year}, track='{track}'.")
    return load_dataset(path)


def combine(datasets: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.concatenate([d["X"] for d in datasets], axis=0)
    y = np.concatenate([d["y"] for d in datasets], axis=0)
    return X, y


def safe_score(value: float) -> float:
    if value is None or np.isnan(value):
        return -1.0
    return float(value)
