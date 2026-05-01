import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

from config import EXTRATREES_PARAMS, EXTRATREES_BLEND_ALPHA
from .splits_and_rank import impute_train_test


def train_extratrees(folds: list[dict],
                     params: dict | None = None,
                     blend_alpha: float = EXTRATREES_BLEND_ALPHA,
                     random_state: int = 42,
                     verbose: bool = True):
    """Train ExtraTrees per fold; blend prediction with grid_position prior.
    """
    if params is None:
        params = EXTRATREES_PARAMS

    all_predictions = []
    last_model = None

    for i, fold in enumerate(folds):
        X_train, X_test = impute_train_test(fold["X_train"], fold["X_test"])
        y_train = fold["y_train"].to_numpy(dtype=np.float32, copy=True)
        y_test = fold["y_test"].to_numpy(dtype=np.float32, copy=True)
        meta = fold["meta_test"].copy()

        model = ExtraTreesRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        last_model = model

        raw_pred = model.predict(X_test).astype(float)
        grid = meta["grid_position"].to_numpy(dtype=float)
        blended = blend_alpha * raw_pred + (1.0 - blend_alpha) * grid
        pred_positions = pd.Series(blended).rank(method="min", ascending=True).values

        result = meta.copy()
        result["pred_score"] = blended
        result["pred_position"] = pred_positions
        all_predictions.append(result)

        if verbose:
            sp, _ = spearmanr(y_test, pred_positions)
            mae = mean_absolute_error(y_test, pred_positions)
            print(f"Fold {i+1:2d} | {fold['test_year']} R{fold['test_round']:2d} "
                  f"{fold['test_event']:30s} | Spearman={sp:.3f} | MAE={mae:.2f}")

    return pd.concat(all_predictions, ignore_index=True), last_model
