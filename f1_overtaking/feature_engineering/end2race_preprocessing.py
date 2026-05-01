from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


DISTANCE_LIKE_FEATURES = (
    "DistanceToDriverAhead",
    "SpatialGap_m",
    "TTC",
    "DistanceToNextCorner",
    "StraightLength",
)
ATTACKER_SPEED_FEATURE = "att_Speed"

DEFAULT_PRESSURE_K = 0.25        
DEFAULT_SPEED_NORMALIZER = 300.0  
DEFAULT_SPEED_EMBED_DIM = 16       
DEFAULT_MASK_PROB = 0.1              


def resolve_feature_indices(feature_order: list[str]) -> tuple[list[int], int, list[int]]:
    """Split feature_order into (distance, speed, other) index sets.
    """
    name_to_idx = {name: i for i, name in enumerate(feature_order)}
    dist_idx = [name_to_idx[n] for n in DISTANCE_LIKE_FEATURES if n in name_to_idx]
    speed_idx = name_to_idx.get(ATTACKER_SPEED_FEATURE, 0)
    other_idx = [i for i in range(len(feature_order))
                 if i not in dist_idx and i != speed_idx]
    return dist_idx, speed_idx, other_idx


def pressure_token_numpy(x: np.ndarray, k: float = DEFAULT_PRESSURE_K) -> np.ndarray:
    """End2Race sigmoid pressure-token transform.

    Maps each scalar x in [0, infinity) to a bounded value in [0, 1] where
    close threats saturate near 1 and distant ones decay toward 0.
    """
    return (-1.0 / (1.0 + np.exp(-k * x)) + 1.0) * 2.0


class End2RacePreprocessor:

    def __init__(self,
                 feature_order: list[str],
                 speed_normalizer: float = DEFAULT_SPEED_NORMALIZER):
        self.feature_order = list(feature_order)
        self.speed_normalizer = float(speed_normalizer)
        self.dist_idx, self.speed_idx, self.other_idx = resolve_feature_indices(
            self.feature_order)
        self.scaler = StandardScaler()
        self._fitted = False

    def _split(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """X has shape (N, T, F). Returns (dist, speed, other)."""
        dist = X[:, :, self.dist_idx]                                  
        speed = X[:, :, self.speed_idx]                                 
        other = X[:, :, self.other_idx]                               
        return dist, speed, other

    def fit(self, X: np.ndarray) -> "End2RacePreprocessor":
        """Fit the StandardScaler on the third stream only."""
        _, _, other = self._split(X)
        n, t, f = other.shape
        if f > 0:
            self.scaler.fit(other.reshape(-1, f))
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (dist, speed_scaled, other_scaled) all as float32."""
        if not self._fitted:
            raise RuntimeError("Call .fit(X_train) before .transform(X).")

        dist, speed, other = self._split(X)
        speed_scaled = (speed / self.speed_normalizer).astype(np.float32)

        n, t, f = other.shape
        if f > 0:
            other_scaled = self.scaler.transform(other.reshape(-1, f)).reshape(n, t, f)
        else:
            other_scaled = other
        return (dist.astype(np.float32),
                speed_scaled,
                other_scaled.astype(np.float32))

    def fit_transform(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.fit(X).transform(X)
