import random

import numpy as np
import pandas as pd

from config import DEFAULT_TABNET_PARAMS
from .splits_and_rank import impute_train_test


def _ensure_tabnet():
    try:
        from pytorch_tabnet.tab_model import TabNetRegressor
    except Exception:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytorch-tabnet"])
        from pytorch_tabnet.tab_model import TabNetRegressor
    return TabNetRegressor


def _set_seed(seed: int):
    import torch
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_one_fold(X_train, X_test, y_train, params, seed=42):
    """Fit TabNet on one fold, returns predictions for X_test and the model."""
    import torch
    TabNetRegressor = _ensure_tabnet()

    n_train = len(X_train)
    val_size = max(1, int(np.ceil(n_train * 0.2)))
    if val_size >= n_train:
        val_size = max(1, n_train - 1)

    if n_train <= 2 or val_size == 0:
        X_tr, y_tr = X_train, y_train
        X_val, y_val = X_train, y_train
    else:
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        X_tr, y_tr = X_train[:-val_size], y_train[:-val_size]
        if len(X_tr) == 0:
            X_tr, y_tr = X_train, y_train
            X_val, y_val = X_train, y_train

    model = TabNetRegressor(
        n_d=int(params["n_d"]),
        n_a=int(params["n_a"]),
        n_steps=int(params["n_steps"]),
        gamma=float(params["gamma"]),
        lambda_sparse=float(params["lambda_sparse"]),
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=float(params["lr"]),
                              weight_decay=float(params["weight_decay"])),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={
            "step_size": int(params["step_size"]),
            "gamma": float(params["sched_gamma"]),
        },
        mask_type=str(params["mask_type"]),
        seed=seed,
        verbose=0,
    )

    batch_size = int(params["batch_size"])
    vbs = int(min(params["virtual_batch_size"], batch_size))
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        eval_name=["train", "val"],
        eval_metric=["mae"],
        max_epochs=int(params["max_epochs"]),
        patience=int(params["patience"]),
        batch_size=batch_size,
        virtual_batch_size=vbs,
    )
    return model


def train_tabnet(folds: list[dict],
                 params: dict | None = None,
                 seed: int = 42):
    """Train TabNet across folds, rank within each race."""
    if params is None:
        params = DEFAULT_TABNET_PARAMS
    _set_seed(seed)

    rows = []
    last_model = None
    for fold in folds:
        X_train, X_test = impute_train_test(fold["X_train"], fold["X_test"])
        y_train = fold["y_train"].to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)

        model = _train_one_fold(X_train, X_test, y_train, params, seed=seed)
        last_model = model
        preds = model.predict(X_test).flatten().astype(float)

        result = fold["meta_test"].copy()
        result["pred_score"] = preds
        result["pred_position"] = (
            result.groupby("race_id")["pred_score"].rank(method="min", ascending=True).astype(float)
        )
        rows.append(result)

    return pd.concat(rows, ignore_index=True), last_model


def _sample_tabnet_params(rng: np.random.Generator) -> dict:
    nd = int(rng.choice([8, 16, 24, 32]))
    return {
        "n_d": nd,
        "n_a": nd,
        "n_steps": int(rng.choice([3, 4, 5, 6])),
        "gamma": float(rng.uniform(1.1, 2.0)),
        "lambda_sparse": float(10 ** rng.uniform(-6, -3)),
        "lr": float(10 ** rng.uniform(-2.4, -1.2)),
        "weight_decay": float(10 ** rng.uniform(-7, -3)),
        "step_size": int(rng.choice([10, 20, 30, 40])),
        "sched_gamma": float(rng.uniform(0.75, 0.98)),
        "mask_type": str(rng.choice(["entmax", "sparsemax"])),
        "batch_size": int(rng.choice([32, 64, 128, 256])),
        "virtual_batch_size": int(rng.choice([16, 32, 64])),
        "max_epochs": int(rng.choice([120, 160, 220])),
        "patience": int(rng.choice([15, 25, 35])),
    }


def tune_tabnet(folds: list[dict], evaluate_predictions, n_trials: int = 12, seed: int = 42):
    """Random search, pick best by Spearman tie-broken by MAE."""
    rng = np.random.default_rng(seed)
    best = None
    print("=" * 100)
    print(f"TabNet tuning: {n_trials} trials")
    print("=" * 100)
    for t in range(1, n_trials + 1):
        params = _sample_tabnet_params(rng)
        preds, model = train_tabnet(folds, params, seed=seed)
        ev = evaluate_predictions(preds)
        summary = {
            "spearman": float(ev["spearman_model"].mean()),
            "mae": float(ev["mae_model"].mean()),
            "ndcg": float(ev["ndcg_model"].mean()),
            "top3": float(ev["top3_model"].mean()),
            "top5": float(ev["top5_model"].mean()),
            "within2": float(ev["within2_model"].mean()),
        }
        is_better = (
            best is None
            or summary["spearman"] > best["spearman"]
            or (summary["spearman"] == best["spearman"] and summary["mae"] < best["mae"])
        )
        if is_better:
            best = {"trial": t, "params": params, "predictions": preds,
                    "eval_df": ev, "model": model, **summary}
            print(f"TabNet trial {t:02d} NEW BEST | Spearman={summary['spearman']:.4f} "
                  f"MAE={summary['mae']:.4f} NDCG={summary['ndcg']:.4f}")
        else:
            print(f"TabNet trial {t:02d}          | Spearman={summary['spearman']:.4f} "
                  f"MAE={summary['mae']:.4f} NDCG={summary['ndcg']:.4f}")

    return best


def calibrate_tabnet(tabnet_preds: pd.DataFrame, evaluate_predictions):
    """Linearly blend TabNet score with grid_position, pick the alpha minimising MAE."""
    best = None
    best_preds = None
    rows = []

    for alpha in np.linspace(0.0, 1.0, 41):
        tmp = tabnet_preds.copy()
        tmp["blend_score"] = alpha * tmp["pred_score"] + (1.0 - alpha) * tmp["grid_position"]
        tmp["pred_position"] = (
            tmp.groupby("race_id")["blend_score"].rank(method="min", ascending=True).astype(float)
        )
        ev = evaluate_predictions(tmp)
        summary = {
            "alpha": float(alpha),
            "spearman": float(ev["spearman_model"].mean()),
            "mae": float(ev["mae_model"].mean()),
            "ndcg": float(ev["ndcg_model"].mean()),
            "top3": float(ev["top3_model"].mean()),
            "top5": float(ev["top5_model"].mean()),
            "within2": float(ev["within2_model"].mean()),
        }
        rows.append(summary)
        if (best is None
                or summary["mae"] < best["mae"]
                or (summary["mae"] == best["mae"] and summary["spearman"] > best["spearman"])):
            best = summary
            best_preds = tmp.copy()

    return best, best_preds, pd.DataFrame(rows).sort_values(
        ["mae", "spearman"], ascending=[True, False]
    ).reset_index(drop=True)


def train_parity_booster(folds: list[dict],
                         dataset: pd.DataFrame,
                         tabnet_cfg: dict,
                         xgb_params: dict,
                         evaluate_predictions,
                         seeds=(11, 42, 77)):
    """Teacher-assisted residual TabNet — XGB teacher + delta-from-grid target.

    Searches a 2-D blend (mix_teacher × mix_grid) and returns the best config.
    """
    import torch
    import xgboost as xgb_lib
    TabNetRegressor = _ensure_tabnet()

    rows = []
    for fold in folds:
        x_train, x_test = impute_train_test(fold["X_train"], fold["X_test"])
        y_train = fold["y_train"].to_numpy(dtype=np.float32, copy=True)
        meta_test = fold["meta_test"].copy()

        grid_train = dataset.loc[fold["X_train"].index, "grid_position"].to_numpy(dtype=np.float32)
        grid_test = meta_test["grid_position"].to_numpy(dtype=np.float32)

        teacher = xgb_lib.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=int(xgb_params["n_estimators"]),
            learning_rate=float(xgb_params["learning_rate"]),
            max_depth=int(xgb_params["max_depth"]),
            min_child_weight=float(xgb_params["min_child_weight"]),
            subsample=float(xgb_params["subsample"]),
            colsample_bytree=float(xgb_params["colsample_bytree"]),
            reg_alpha=float(xgb_params["reg_alpha"]),
            reg_lambda=float(xgb_params["reg_lambda"]),
            gamma=float(xgb_params["gamma"]),
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
        teacher.fit(x_train, y_train)
        teacher_train = teacher.predict(x_train).astype(np.float32)
        teacher_test = teacher.predict(x_test).astype(np.float32)

        x_train_aug = np.column_stack([x_train, teacher_train, grid_train]).astype(np.float32)
        x_test_aug = np.column_stack([x_test, teacher_test, grid_test]).astype(np.float32)
        y_train_delta = (y_train - grid_train).astype(np.float32).reshape(-1, 1)

        seed_preds = []
        for sd in seeds:
            _set_seed(int(sd))
            model = _train_one_fold(x_train_aug, x_test_aug, y_train_delta, tabnet_cfg, seed=int(sd))
            seed_preds.append(model.predict(x_test_aug).flatten().astype(np.float32))

        mean_delta = np.mean(np.vstack(seed_preds), axis=0)
        tabnet_residual_score = (grid_test + mean_delta).astype(float)

        out = meta_test.copy()
        out["tabnet_residual_score"] = tabnet_residual_score
        out["xgb_teacher_score"] = teacher_test.astype(float)
        out["grid_score"] = grid_test.astype(float)
        rows.append(out)

    pred_df = pd.concat(rows, ignore_index=True)
    return _search_hybrid_blend(pred_df, evaluate_predictions)


def _search_hybrid_blend(pred_df: pd.DataFrame, evaluate_predictions):
    """2-D grid search over (mix_teacher, mix_grid). Pick min MAE then max Spearman."""
    best = None
    best_eval = None
    best_preds = None
    rows = []

    for mix_teacher in np.linspace(0.0, 1.0, 41):
        for mix_grid in np.linspace(0.0, 0.35, 15):
            tmp = pred_df.copy()
            model_score = (
                (1.0 - mix_teacher) * tmp["tabnet_residual_score"]
                + mix_teacher * tmp["xgb_teacher_score"]
            )
            tmp["final_score"] = (1.0 - mix_grid) * model_score + mix_grid * tmp["grid_score"]
            tmp["pred_score"] = tmp["final_score"]
            tmp["pred_position"] = (
                tmp.groupby("race_id")["final_score"].rank(method="min", ascending=True).astype(float)
            )

            ev = evaluate_predictions(tmp)
            summary = {
                "mix_teacher": float(mix_teacher),
                "mix_grid": float(mix_grid),
                "spearman": float(ev["spearman_model"].mean()),
                "mae": float(ev["mae_model"].mean()),
                "ndcg": float(ev["ndcg_model"].mean()),
                "top3": float(ev["top3_model"].mean()),
                "top5": float(ev["top5_model"].mean()),
                "within2": float(ev["within2_model"].mean()),
            }
            rows.append(summary)
            if (best is None
                    or summary["mae"] < best["mae"]
                    or (summary["mae"] == best["mae"] and summary["spearman"] > best["spearman"])):
                best = summary
                best_eval = ev.copy()
                best_preds = tmp.copy()

    return best, best_eval, best_preds, pd.DataFrame(rows).sort_values(
        ["mae", "spearman"], ascending=[True, False]
    ).reset_index(drop=True)
