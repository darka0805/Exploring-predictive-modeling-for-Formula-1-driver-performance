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

LGBM_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    metric="average_precision",
    n_jobs=-1,
    random_state=SEED,
    verbose=-1,
)

DL_BATCH_SIZE = 256
DL_EPOCHS = 40
DL_LR = 1e-3
DL_WEIGHT_DECAY = 1e-4
DL_DROPOUT = 0.3