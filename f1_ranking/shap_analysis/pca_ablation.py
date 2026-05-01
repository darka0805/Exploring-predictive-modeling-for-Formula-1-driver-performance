import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

from config import EXTRATREES_PARAMS, EXTRATREES_BLEND_ALPHA
from model_training.splits_and_rank import impute_train_test


def evaluate_extratrees_pca(folds: list[dict],
                            evaluate_predictions,
                            variance: float = 0.95,
                            random_state: int = 42):
    """Per-fold PCA -> ExtraTrees -> grid blend, then evaluate.

    Returns:
        pca_preds:      DataFrame of predictions across folds.
        pca_eval:       output of `evaluate_predictions`.
        n_components:   list of selected components per fold.
    """
    rows = []
    n_components = []

    for fold in folds:
        X_train, X_test = impute_train_test(fold["X_train"], fold["X_test"])
        y_train = fold["y_train"].to_numpy(dtype=np.float32, copy=True)
        meta = fold["meta_test"].copy()

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        pca = PCA(n_components=variance, svd_solver="full")
        X_train_p = pca.fit_transform(X_train_s)
        X_test_p = pca.transform(X_test_s)
        n_components.append(int(pca.n_components_))

        model = ExtraTreesRegressor(
            n_estimators=EXTRATREES_PARAMS["n_estimators"],
            max_depth=EXTRATREES_PARAMS["max_depth"],
            min_samples_leaf=EXTRATREES_PARAMS["min_samples_leaf"],
            max_features=EXTRATREES_PARAMS["max_features"],
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train_p, y_train)

        raw_pred = model.predict(X_test_p).astype(float)
        grid = meta["grid_position"].to_numpy(dtype=float)
        blended = EXTRATREES_BLEND_ALPHA * raw_pred + (1.0 - EXTRATREES_BLEND_ALPHA) * grid
        pred_pos = pd.Series(blended).rank(method="min", ascending=True).values

        out = meta.copy()
        out["pred_score"] = blended
        out["pred_position"] = pred_pos
        rows.append(out)

    pca_preds = pd.concat(rows, ignore_index=True)
    pca_eval = evaluate_predictions(pca_preds)
    return pca_preds, pca_eval, n_components
