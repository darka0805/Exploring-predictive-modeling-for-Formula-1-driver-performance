from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from config import (
    ARTIFACT_ROOT,
    DEFAULT_XGB_PARAMS,
    EXTRATREES_PARAMS,
    YEARS,
)


def _prepare_full_X(dataset: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = dataset[feature_cols].apply(pd.to_numeric, errors="coerce")
    return X.fillna(X.mean(numeric_only=True)).fillna(0.0)


def save_full_extratrees(dataset: pd.DataFrame,
                         feature_cols: list[str],
                         walkforward_model=None,
                         params: dict | None = None,
                         models_dir: Path | None = None) -> list[str]:
    params = params or EXTRATREES_PARAMS
    models_dir = (models_dir or ARTIFACT_ROOT / "models") / "extratrees"
    models_dir.mkdir(parents=True, exist_ok=True)

    X = _prepare_full_X(dataset, feature_cols).to_numpy(dtype=np.float32)
    y = dataset["finish_position"].to_numpy(dtype=np.float32)

    saved: list[str] = []
    if walkforward_model is not None:
        wf_path = models_dir / "extratrees_walkforward_last_fold.joblib"
        joblib.dump(walkforward_model, wf_path)
        saved.append(str(wf_path))

    full = ExtraTreesRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        random_state=42,
        n_jobs=-1,
    )
    full.fit(X, y)

    model_path = models_dir / "extratrees_full_model.joblib"
    meta_path = models_dir / "extratrees_metadata.joblib"
    joblib.dump(full, model_path)
    joblib.dump(
        {"feature_cols": feature_cols, "years": YEARS,
         "target": "finish_position", "n_rows": int(len(dataset))},
        meta_path,
    )
    saved.extend([str(model_path), str(meta_path)])
    print(f"Saved ExtraTrees full-data model -> {model_path}")
    return saved


def save_full_xgboost(dataset: pd.DataFrame,
                      feature_cols: list[str],
                      params: dict | None = None,
                      models_dir: Path | None = None) -> list[str]:
    try:
        import xgboost as xgb
    except Exception:
        print("XGBoost not available — skipping full-data export.")
        return []

    params = params or DEFAULT_XGB_PARAMS
    models_dir = (models_dir or ARTIFACT_ROOT / "models") / "xgboost"
    models_dir.mkdir(parents=True, exist_ok=True)

    X = _prepare_full_X(dataset, feature_cols).to_numpy(dtype=np.float32)
    y = dataset["finish_position"].to_numpy(dtype=np.float32)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        **params,
    )
    model.fit(X, y)

    model_path = models_dir / "xgboost_full_model.json"
    meta_path = models_dir / "xgboost_metadata.joblib"
    model.save_model(str(model_path))
    joblib.dump(
        {"feature_cols": feature_cols, "years": YEARS, "target": "finish_position",
         "n_rows": int(len(dataset)), "params": params},
        meta_path,
    )
    print(f"Saved XGBoost full-data model -> {model_path}")
    return [str(model_path), str(meta_path)]


def save_full_tabnet(dataset: pd.DataFrame,
                     feature_cols: list[str],
                     tabnet_params: dict,
                     last_fold_model=None,
                     models_dir: Path | None = None) -> list[str]:
    if not tabnet_params:
        print("No TabNet best params provided — skipping TabNet export.")
        return []
    try:
        import torch
        from pytorch_tabnet.tab_model import TabNetRegressor
    except Exception as e:
        print(f"TabNet/torch unavailable: {e} — skipping.")
        return []

    models_dir = (models_dir or ARTIFACT_ROOT / "models") / "tabnet"
    models_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    cfg_path = models_dir / "tabnet_configs.json"
    cfg_path.write_text(json.dumps({"best_tabnet_params": tabnet_params}, indent=2),
                        encoding="utf-8")
    saved.append(str(cfg_path))

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X = _prepare_full_X(dataset, feature_cols).to_numpy(dtype=np.float32)
    y = dataset["finish_position"].to_numpy(dtype=np.float32).reshape(-1, 1)

    n_total = len(X)
    val_size = max(1, int(np.ceil(n_total * 0.2)))
    if val_size >= n_total:
        val_size = max(1, n_total - 1)
    X_tr, y_tr = X[:-val_size], y[:-val_size]
    X_val, y_val = X[-val_size:], y[-val_size:]
    if len(X_tr) == 0:
        X_tr, y_tr, X_val, y_val = X, y, X, y

    model = TabNetRegressor(
        n_d=int(tabnet_params["n_d"]),
        n_a=int(tabnet_params["n_a"]),
        n_steps=int(tabnet_params["n_steps"]),
        gamma=float(tabnet_params["gamma"]),
        lambda_sparse=float(tabnet_params["lambda_sparse"]),
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=float(tabnet_params["lr"]),
                              weight_decay=float(tabnet_params["weight_decay"])),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": int(tabnet_params["step_size"]),
                          "gamma": float(tabnet_params["sched_gamma"])},
        mask_type=tabnet_params["mask_type"],
        seed=seed,
        verbose=0,
    )

    bs = int(tabnet_params["batch_size"])
    vbs = int(min(tabnet_params["virtual_batch_size"], bs))
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        eval_name=["train", "val"],
        eval_metric=["mae"],
        max_epochs=int(tabnet_params["max_epochs"]),
        patience=int(tabnet_params["patience"]),
        batch_size=bs,
        virtual_batch_size=vbs,
    )

    full_stub = models_dir / "tabnet_full_model"
    model.save_model(str(full_stub))
    saved.append(str(full_stub) + ".zip")
    print(f"Saved TabNet full-data model -> {full_stub}.zip")

    meta_path = models_dir / "tabnet_metadata.joblib"
    joblib.dump(
        {"feature_cols": feature_cols, "years": YEARS, "target": "finish_position",
         "n_rows": int(len(dataset)), "params": tabnet_params, "val_size": int(val_size)},
        meta_path,
    )
    saved.append(str(meta_path))

    if last_fold_model is not None:
        try:
            last_stub = models_dir / "tabnet_last_fold_model"
            last_fold_model.save_model(str(last_stub))
            saved.append(str(last_stub) + ".zip")
        except Exception as e:
            print(f"Could not save TabNet last-fold weights: {e}")

    return saved
