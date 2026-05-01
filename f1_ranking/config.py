from pathlib import Path

YEARS = [2022, 2023, 2024, 2025]

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR
ARTIFACT_ROOT = PROJECT_DIR / "model_artifacts_2022_2025_new"

SESSION_FOLDER_MAP = {
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
    "Q": "Qualifying",
    "R": "Race",
}

CACHE_FILE = DATA_DIR / "data" / "all_laps_2022_2025_v2.pkl"

QUALIFYING_107_PCT = 1.07

EXTRATREES_PARAMS = {
    "n_estimators": 1491,
    "max_depth": 12,
    "min_samples_leaf": 1,
    "max_features": 0.5667368615030315,
}
EXTRATREES_BLEND_ALPHA = 0.975

DEFAULT_XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 2.0,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "gamma": 0.0,
}

DEFAULT_TABNET_PARAMS = {
    "n_d": 32,
    "n_a": 32,
    "n_steps": 4,
    "gamma": 1.5,
    "lambda_sparse": 1e-4,
    "lr": 2e-2,
    "weight_decay": 1e-6,
    "step_size": 20,
    "sched_gamma": 0.9,
    "mask_type": "entmax",
    "batch_size": 64,
    "virtual_batch_size": 32,
    "max_epochs": 180,
    "patience": 30,
}


def ensure_dirs():
    """Create cache and artifact directories if missing."""
    for y in YEARS:
        (PROJECT_DIR / str(y) / "cache").mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "models").mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "shap").mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "race_views").mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "pca_ablation").mkdir(parents=True, exist_ok=True)
