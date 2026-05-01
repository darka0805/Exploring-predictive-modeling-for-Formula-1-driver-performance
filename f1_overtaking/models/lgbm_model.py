import numpy as np
from lightgbm import LGBMClassifier

from config import LGBM_PARAMS, SEED


class LGBMModel:
    name = "LightGBM"

    def __init__(self, extra_params: dict | None = None):
        params = {**LGBM_PARAMS}
        user = dict(extra_params or {})
        use_is_unbalance = bool(user.pop("use_is_unbalance", True))
        scale_pos_weight = user.pop("scale_pos_weight", None)

        if use_is_unbalance:
            params["is_unbalance"] = True
            params.pop("scale_pos_weight", None)
        else:
            params.pop("is_unbalance", None)
            if scale_pos_weight is not None:
                params["scale_pos_weight"] = float(scale_pos_weight)

        if user:
            params.update(user)
        self.model = LGBMClassifier(**params)

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], -1)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
        X_flat = self._flatten(X_train)

        fit_kw: dict = {}
        if X_val is not None and y_val is not None:
            fit_kw["eval_set"] = [(self._flatten(X_val), y_val)]
        self.model.fit(X_flat, y_train, **fit_kw)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._flatten(X))[:, 1]
