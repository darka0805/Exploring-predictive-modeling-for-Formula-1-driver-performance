import numpy as np
from sklearn.ensemble import RandomForestClassifier

from config import SEED


class RFModel:
    name = "RandomForest"

    def __init__(self, extra_params: dict | None = None):
        params = dict(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        )
        if extra_params:
            params.update(extra_params)
        self.model = RandomForestClassifier(**params)

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], -1)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
        self.model.fit(self._flatten(X_train), y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._flatten(X))[:, 1]
