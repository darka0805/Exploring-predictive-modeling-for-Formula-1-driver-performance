from __future__ import annotations

from itertools import product


TRAIN_YEARS = [2022, 2023, 2024]
TEST_YEAR = 2025
NEG_POS_RATIO = 20
VAL_SIZE = 0.20
FULL_DATA_RERUN_TOP_K = 2

EXPERIMENT_TAG = "E7_train2022_2024_test2025"


def _expand_grid(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


MODEL_SWEEPS: dict[str, list[dict]] = {
    "xgboost": _expand_grid({
        "n_estimators": [300, 500, 800],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.03, 0.05],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8],
        "min_child_weight": [1, 3],
        "tree_method": ["hist"],
    }),
    "lgbm": _expand_grid({
        "n_estimators": [300, 500, 800],
        "max_depth": [6, -1],
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.02, 0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }),
    "rf": _expand_grid({
        "n_estimators": [300, 600, 900],
        "max_depth": [10, None],
        "min_samples_leaf": [1, 3, 5],
        "max_features": ["sqrt", 0.5],
        "class_weight": ["balanced_subsample"],
    }),
    "cnn": _expand_grid({
        "hidden": [32, 64, 96],
        "dropout": [0.2, 0.35],
        "lr": [3e-4, 1e-3],
        "batch_size": [128, 256],
        "epochs": [28],
        "patience": [7],
        "use_weighted_sampler": [True],
    }),
    "bigru": _expand_grid({
        "hidden": [64, 96, 128],
        "n_layers": [1, 2, 3, 4],
        "dropout": [0.2, 0.35],
        "lr": [3e-4, 1e-3],
        "batch_size": [128],
        "epochs": [28],
        "patience": [7],
        "use_weighted_sampler": [True],
    }),
    "end2race": _expand_grid({
        "hidden_mult": [2, 4],
        "speed_embed_dim": [8, 16],
        "mask_prob": [0.1],
        "pressure_k": [0.1, 0.25, 0.5],
        "lr": [1e-3],
        "batch_size": [64, 128],
        "epochs": [40],
        "patience": [8],
    }),
}


STACKING_CONFIGS = [
    {
        "name": "stack_lr_all5",
        "base_models": ["xgboost", "lgbm", "rf", "cnn", "bigru"],
        "meta_c": 1.0,
    },
    {
        "name": "stack_lr_tree_plus_seq",
        "base_models": ["bigru", "lgbm", "xgboost"],
        "meta_c": 0.7,
    },
]
