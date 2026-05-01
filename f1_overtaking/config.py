from pathlib import Path

SEED = 42

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "saved_models"

for d in (CACHE_DIR, DATA_DIR, RESULTS_DIR, MODEL_DIR):
    d.mkdir(exist_ok=True)

SAMPLE_HZ = 5

OBS_WINDOW_S = 5.0
OBS_WINDOW = int(OBS_WINDOW_S * SAMPLE_HZ)

PRED_WINDOW_S = 2.0
PRED_WINDOW = int(PRED_WINDOW_S * SAMPLE_HZ)

SAMPLE_STRIDE = SAMPLE_HZ

CONFRONTATION_GAP_S = 1.5
MAX_EPISODE_GAP_S = 2.0

OVERTAKE_GAP_THRESH = 0.0
OVERTAKE_MIN_CONSEC = 3

PRIMARY_TRACK = "Abu Dhabi Grand Prix"
ALL_TRACKS = [
    "Abu Dhabi Grand Prix",
    "Bahrain Grand Prix",
    "Italian Grand Prix",
]
YEARS = [2022, 2023, 2024]

ALL_2022_RACES = [
    "Bahrain Grand Prix",
    "Saudi Arabian Grand Prix",
    "Australian Grand Prix",
    "Emilia Romagna Grand Prix",
    "Miami Grand Prix",
    "Spanish Grand Prix",
    "Monaco Grand Prix",
    "Azerbaijan Grand Prix",
    "Canadian Grand Prix",
    "British Grand Prix",
    "Austrian Grand Prix",
    "French Grand Prix",
    "Hungarian Grand Prix",
    "Belgian Grand Prix",
    "Dutch Grand Prix",
    "Italian Grand Prix",
    "Singapore Grand Prix",
    "Japanese Grand Prix",
    "United States Grand Prix",
    "Mexico City Grand Prix",
    "São Paulo Grand Prix",
    "Abu Dhabi Grand Prix",
]

TEMPORAL_FEATURES = [
    "att_Speed", "def_Speed",
    "att_Throttle", "def_Throttle",
    "att_Brake", "def_Brake",
    "att_Gear", "def_Gear",
    "att_DRS", "def_DRS",
    "att_X", "att_Y",
    "def_X", "def_Y",
    "DistanceToDriverAhead", "SpatialGap_m",
    "RelativeVelocity", "TTC", "ClosingRate", "SpeedRatio",
    "SlipstreamCoeff", "EnergyDelta",
    "DefenderLineDeviation",
    "att_Speed_mean_1s", "def_Speed_mean_1s",
    "ClosingRate_mean_1s",
    "att_Throttle_std_1s", "def_Throttle_std_1s",
    "BattlePersistence",
    "att_FreshTyre", "def_FreshTyre",
    "CompoundAdvantage",
    "att_MandatoryPitDone", "def_MandatoryPitDone",
    "att_TyreLife", "def_TyreLife",
    "TrackCurvature", "DistanceToNextCorner",
    "StraightLength", "Sector_ID",
]
N_FEAT = len(TEMPORAL_FEATURES)

FEATURE_GROUPS = {
    "G1_RawTelemetry": [
        "att_Speed", "def_Speed", "att_Throttle", "def_Throttle",
        "att_Brake", "def_Brake", "att_Gear", "def_Gear",
        "att_DRS", "def_DRS",
    ],
    "G2_Position": ["att_X", "att_Y", "def_X", "def_Y"],
    "G3_Interaction": [
        "DistanceToDriverAhead", "SpatialGap_m",
        "RelativeVelocity", "TTC", "ClosingRate", "SpeedRatio",
    ],
    "G4_DRS_AI": ["SlipstreamCoeff", "EnergyDelta"],
    "G5_Kalman": ["DefenderLineDeviation"],
    "G6_RollingStats": [
        "att_Speed_mean_1s", "def_Speed_mean_1s",
        "ClosingRate_mean_1s",
        "att_Throttle_std_1s", "def_Throttle_std_1s",
    ],
    "G7_BattlePersistence": ["BattlePersistence"],
    "G8_DriverContext": [
        "att_FreshTyre", "def_FreshTyre", "CompoundAdvantage",
        "att_MandatoryPitDone", "def_MandatoryPitDone",
        "att_TyreLife", "def_TyreLife",
    ],
    "G9_TrackGeometry": [
        "TrackCurvature", "DistanceToNextCorner",
        "StraightLength", "Sector_ID",
    ],
}

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="aucpr",
    early_stopping_rounds=30,
    n_jobs=-1,
    random_state=SEED,
)

DL_BATCH_SIZE = 256
DL_EPOCHS = 40
DL_LR = 1e-3
DL_WEIGHT_DECAY = 1e-4
DL_DROPOUT = 0.3

# ----- E7 (2022-2024 train, 2025 holdout) -----

from itertools import product as _product


def _expand_grid(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in _product(*values)]


E7_TRAIN_YEARS = [2022, 2023, 2024]
E7_TEST_YEAR = 2025
E7_NEG_POS_RATIO = 20
E7_VAL_SIZE = 0.20
E7_FULL_DATA_RERUN_TOP_K = 2
E7_EXPERIMENT_TAG = "E7_train2022_2024_test2025"

E7_MODEL_SWEEPS: dict[str, list[dict]] = {
    "xgboost": _expand_grid({
        "n_estimators": [300, 500, 800],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.03, 0.05],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8],
        "min_child_weight": [1, 3],
        "tree_method": ["hist"],
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

E7_STACKING_CONFIGS = [
    {
        "name": "stack_lr_all5",
        "base_models": ["xgboost", "rf", "cnn", "bigru"],
        "meta_c": 1.0,
    },
    {
        "name": "stack_lr_tree_plus_seq",
        "base_models": ["bigru", "xgboost", "rf"],
        "meta_c": 0.7,
    },
]