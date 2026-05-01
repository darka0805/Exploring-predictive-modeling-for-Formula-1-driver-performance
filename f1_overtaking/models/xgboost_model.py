import numpy as np
from xgboost import XGBClassifier

from config import XGB_PARAMS, N_FEAT, OBS_WINDOW, SEED


class XGBoostModel:
    name = "XGBoost"

    def __init__(self, extra_params: dict | None = None):
        params = {**XGB_PARAMS}
        if extra_params:
            params.update(extra_params)
        self.model = XGBClassifier(**params)

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], -1)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
        X_flat = self._flatten(X_train)
        spw = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0)
        self.model.set_params(scale_pos_weight=spw)

        fit_kw: dict = {}
        if X_val is not None and y_val is not None:
            fit_kw["eval_set"] = [(self._flatten(X_val), y_val)]
            fit_kw["verbose"] = False
        self.model.fit(X_flat, y_train, **fit_kw)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._flatten(X))[:, 1]

    def get_feature_names(self) -> list[str]:
        from config import TEMPORAL_FEATURES
        names = []
        for t in range(OBS_WINDOW):
            for f in TEMPORAL_FEATURES:
                names.append(f"{f}_t{t}")
        return names
