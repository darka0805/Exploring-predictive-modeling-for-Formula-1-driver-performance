import numpy as np
import pandas as pd

from config import (
    TEMPORAL_FEATURES, N_FEAT,
    OBS_WINDOW, PRED_WINDOW, SAMPLE_STRIDE,
    MAX_EPISODE_GAP_S, OVERTAKE_GAP_THRESH, OVERTAKE_MIN_CONSEC,
)


def _detect_overtake_in_window(gap: np.ndarray) -> int:
    """Return 1 if the gap array contains an overtake event.

    An overtake is defined as the time-gap crossing <= OVERTAKE_GAP_THRESH
    for at least OVERTAKE_MIN_CONSEC consecutive samples.
    """
    below = gap <= OVERTAKE_GAP_THRESH
    if not below.any():
        return 0
    consec = 0
    for b in below:
        if b:
            consec += 1
            if consec >= OVERTAKE_MIN_CONSEC:
                return 1
        else:
            consec = 0
    return 0


def extract_episodes(pair: dict,
                     obs_window: int = OBS_WINDOW,
                     pred_window: int = PRED_WINDOW,
                     stride: int = SAMPLE_STRIDE,
                     max_gap: float = MAX_EPISODE_GAP_S) -> dict:
    """Extract (feature_matrix, label) episodes from one pair-lap.

    Returns dict with:
        X  : np.ndarray  (n_episodes, obs_window, N_FEAT)
        y  : np.ndarray  (n_episodes,)  binary
        meta : list[dict] per-episode metadata
    """
    df = pair["df"]
    n = len(df)
    min_len = obs_window + pred_window

    if n < min_len:
        return {"X": np.empty((0, obs_window, N_FEAT), dtype=np.float32),
                "y": np.empty(0, dtype=np.int8),
                "meta": []}

    vals = df[TEMPORAL_FEATURES].values.astype(np.float32)
    vals = np.nan_to_num(vals, nan=0.0, posinf=30.0, neginf=-30.0)
    gap = df["DistanceToDriverAhead"].values

    X_list, y_list, meta_list = [], [], []

    for i in range(obs_window, n - pred_window, stride):
        g = gap[i]
        if not np.isfinite(g) or g > max_gap or g <= 0:
            continue

        obs = vals[i - obs_window: i]           
        future_gap = gap[i: i + pred_window]   
        label = _detect_overtake_in_window(future_gap)

        X_list.append(obs)
        y_list.append(label)
        meta_list.append({
            "year": pair["year"],
            "event": pair["event"],
            "lap": pair["lap"],
            "attacker": pair["attacker"],
            "defender": pair["defender"],
            "gap_at_anchor": float(g),
        })

    if not X_list:
        return {"X": np.empty((0, obs_window, N_FEAT), dtype=np.float32),
                "y": np.empty(0, dtype=np.int8),
                "meta": []}

    return {
        "X": np.stack(X_list),
        "y": np.array(y_list, dtype=np.int8),
        "meta": meta_list,
    }


def extract_all_episodes(pairs: list[dict], **kwargs) -> dict:
    """Run episode extraction over a list of pair-laps and concatenate."""
    all_X, all_y, all_meta = [], [], []
    for p in pairs:
        ep = extract_episodes(p, **kwargs)
        if len(ep["y"]) > 0:
            all_X.append(ep["X"])
            all_y.append(ep["y"])
            all_meta.extend(ep["meta"])

    if not all_X:
        return {"X": np.empty((0, OBS_WINDOW, N_FEAT), dtype=np.float32),
                "y": np.empty(0, dtype=np.int8),
                "meta": pd.DataFrame()}

    return {
        "X": np.concatenate(all_X),
        "y": np.concatenate(all_y),
        "meta": pd.DataFrame(all_meta),
    }
